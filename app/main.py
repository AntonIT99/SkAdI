import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from app.book_scanner import BookFile, describe_book_file, scan_books
from app.config import (
    BOOKS_ROOT,
    COLLECTION_NAME,
    DEFAULT_REPOSITORY,
    EMBEDDING_MODEL,
    LLM_MODEL,
    SUPPORTED_LANGUAGES,
    SUPPORTED_REPOSITORIES,
)
from app.rag import RagService

app = FastAPI(title="Private RAG Philosophy Bot")

rag = RagService()
logger = logging.getLogger(__name__)


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
    response = {
        "status": "ok",
        "qdrant": "ok",
        "collection": COLLECTION_NAME,
        "points_count": 0,
        "embedding_model": EMBEDDING_MODEL,
        "default_llm": LLM_MODEL
    }

    try:
        response["points_count"] = rag.vector_store.points_count()
    except Exception as exc:
        response["status"] = "degraded"
        response["qdrant"] = "error"
        response["points_count"] = None
        response["qdrant_error"] = str(exc)

    return response


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
def ingest_all(
    limit: int | None = Query(default=None, ge=0),
    repository: str | None = None,
    language: str | None = None,
    dry_run: bool = False
):
    if repository is not None:
        _validate_values([repository], SUPPORTED_REPOSITORIES, "repository")
    if language is not None:
        _validate_values([language], SUPPORTED_LANGUAGES, "language")

    logger.info("[INGEST] Starting full ingestion")
    logger.info("[INGEST] Books root: %s", BOOKS_ROOT.resolve())
    logger.info("[INGEST] Scanning for PDF files")

    try:
        books = scan_books()
    except (FileNotFoundError, NotADirectoryError) as exc:
        return {
            "status": "error",
            "message": str(exc)
        }

    discovered_count = len(books)
    logger.info("[INGEST] Discovered %s PDF files", discovered_count)

    books = _filter_books(books, repository=repository, language=language)
    selected_before_limit = len(books)

    if limit is not None:
        books = books[:limit]

    if repository is not None or language is not None or limit is not None:
        logger.info(
            "[INGEST] Selected %s PDF files after filters repository=%s language=%s limit=%s",
            len(books),
            repository,
            language,
            limit,
        )

    details = []
    total = len(books)

    if dry_run:
        logger.info("[INGEST] Dry run enabled; no PDFs will be parsed, embedded, or upserted")
        for index, book in enumerate(books, start=1):
            logger.info(
                "[INGEST] [%s/%s] Dry run: would process %s repository=%s language=%s sha256=%s",
                index,
                total,
                book.relative_path,
                book.repository,
                book.language,
                book.sha256,
            )
            details.append(_book_result(book, status="dry_run"))

        response = _ingest_all_response(
            details=details,
            discovered=discovered_count,
            selected=total,
            selected_before_limit=selected_before_limit,
            dry_run=dry_run,
        )
        logger.info("[INGEST] Dry run complete: would_process=%s", response["would_process"])
        return response

    for index, book in enumerate(books, start=1):
        try:
            details.append(rag.ingest_book(book, index=index, total=total))
        except Exception as exc:
            logger.exception("[INGEST] [%s/%s] Unexpected ingestion failure", index, total)
            details.append(_book_result(book, status="failed", error=str(exc)))

    response = _ingest_all_response(
        details=details,
        discovered=discovered_count,
        selected=total,
        selected_before_limit=selected_before_limit,
        dry_run=dry_run,
    )
    logger.info(
        "[INGEST] Full ingestion complete: indexed=%s skipped=%s no_text=%s failed=%s",
        response["indexed"],
        response["skipped"],
        response["no_text"],
        response["failed"],
    )
    return response


def _ingest_all_response(
    details: list[dict],
    discovered: int,
    selected: int,
    selected_before_limit: int,
    dry_run: bool
) -> dict:
    response = {
        "status": "ok",
        "books_root": str(BOOKS_ROOT.resolve()),
        "discovered": discovered,
        "selected": selected,
        "selected_before_limit": selected_before_limit,
        "dry_run": dry_run,
        "indexed": _count_status(details, "indexed"),
        "skipped": _count_status(details, "skipped"),
        "no_text": _count_status(details, "no_text"),
        "failed": _count_status(details, "failed"),
        "details": details
    }

    if dry_run:
        response["would_process"] = _count_status(details, "dry_run")

    return response


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


def _filter_books(
    books: list[BookFile],
    repository: str | None,
    language: str | None
) -> list[BookFile]:
    return [
        book
        for book in books
        if (repository is None or book.repository == repository)
        and (language is None or book.language == language)
    ]


def _book_result(
    book: BookFile,
    status: str,
    chunks: int = 0,
    error: str | None = None
) -> dict:
    return {
        "file": book.relative_path,
        "repository": book.repository,
        "language": book.language,
        "document_hash": book.sha256,
        "status": status,
        "chunks": chunks,
        "error": error
    }


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
