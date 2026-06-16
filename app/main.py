from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.book_scanner import BookFile, describe_book_file, scan_books
from app.config import BOOKS_ROOT, DEFAULT_REPOSITORY, LLM_MODEL, SUPPORTED_LANGUAGES, SUPPORTED_REPOSITORIES
from app.rag import RagService

app = FastAPI(title="Private RAG Philosophy Bot")

rag = RagService()


class IngestRequest(BaseModel):
    file_name: str
    repository: str = DEFAULT_REPOSITORY


class ChatRequest(BaseModel):
    question: str
    repository: str = DEFAULT_REPOSITORY
    repositories: list[str] | None = None
    languages: list[str] | None = None
    model: str = LLM_MODEL


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/books/scan")
def books_scan():
    try:
        books = scan_books()
    except (FileNotFoundError, NotADirectoryError) as exc:
        return {
            "status": "error",
            "message": str(exc)
        }

    return {
        "status": "ok",
        "books_root": str(BOOKS_ROOT),
        "count": len(books),
        "books": [book.to_dict() for book in books]
    }


@app.post("/ingest")
def ingest(request: IngestRequest):
    _validate_values([request.repository], SUPPORTED_REPOSITORIES, "repository")

    books_root_error = _books_root_error()
    if books_root_error:
        return {
            "status": "error",
            "message": books_root_error
        }

    book = _resolve_book(request)
    if book is None:
        return {
            "status": "error",
            "message": "File not found in BOOKS_ROOT or not under the expected <repository>/<language>/... structure"
        }

    return rag.ingest_book(book)


@app.post("/ingest/all")
def ingest_all():
    try:
        books = scan_books()
    except (FileNotFoundError, NotADirectoryError) as exc:
        return {
            "status": "error",
            "message": str(exc)
        }

    details = []
    for book in books:
        try:
            details.append(rag.ingest_book(book))
        except Exception as exc:
            details.append({
                "file": book.relative_path,
                "repository": book.repository,
                "language": book.language,
                "document_hash": book.sha256,
                "status": "failed",
                "chunks": 0,
                "error": str(exc)
            })

    return {
        "status": "ok",
        "books_root": str(BOOKS_ROOT),
        "indexed": _count_status(details, "indexed"),
        "skipped": _count_status(details, "skipped"),
        "no_text": _count_status(details, "no_text"),
        "failed": _count_status(details, "failed"),
        "details": details
    }


@app.post("/chat")
def chat(request: ChatRequest):
    _validate_values([request.repository], SUPPORTED_REPOSITORIES, "repository")

    repositories = request.repositories or [request.repository]
    _validate_values(repositories, SUPPORTED_REPOSITORIES, "repositories")

    if request.languages is not None:
        _validate_values(request.languages, SUPPORTED_LANGUAGES, "languages")

    return rag.chat(
        question=request.question,
        repository=request.repository,
        repositories=repositories,
        languages=request.languages,
        model=request.model
    )


def _resolve_book(request: IngestRequest) -> BookFile | None:
    requested_path = Path(request.file_name)
    candidates = []

    if requested_path.is_absolute():
        candidates.append(requested_path)
    else:
        candidates.append(BOOKS_ROOT / request.file_name)
        candidates.append(BOOKS_ROOT / request.repository / request.file_name)

    for candidate in candidates:
        book = describe_book_file(candidate)
        if book is not None and book.repository == request.repository:
            return book

    return None


def _validate_values(values: list[str], supported_values: set[str], field_name: str) -> None:
    if not values:
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must not be empty"
        )

    invalid_values = sorted(set(values) - supported_values)
    if invalid_values:
        raise HTTPException(
            status_code=422,
            detail={
                "message": f"Unsupported {field_name}",
                "invalid": invalid_values,
                "supported": sorted(supported_values)
            }
        )


def _count_status(details: list[dict], status: str) -> int:
    return sum(1 for detail in details if detail.get("status") == status)


def _books_root_error() -> str | None:
    root = BOOKS_ROOT.resolve()
    if not root.exists():
        return f"BOOKS_ROOT does not exist: {root}"
    if not root.is_dir():
        return f"BOOKS_ROOT is not a directory: {root}"
    return None
