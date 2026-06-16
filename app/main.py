import logging
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

from app.book_scanner import BookFile, calculate_sha256, describe_book_file, scan_books
from app.config import (
    BOOKS_ROOT,
    COLLECTION_NAME,
    DEFAULT_REPOSITORY,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL,
    LLM_MODEL,
    QDRANT_UPSERT_BATCH_SIZE,
    SUPPORTED_LANGUAGES,
    SUPPORTED_REPOSITORIES,
)
from app.embeddings import embedding_diagnostics
from app.llm import LlmServiceUnavailable
from app.rag import RagService

app = FastAPI(title="Private RAG Philosophy Bot")

_rag: RagService | None = None
logger = logging.getLogger(__name__)


class IngestRequest(BaseModel):
    file_name: str
    repository: str = DEFAULT_REPOSITORY


class DeindexRequest(BaseModel):
    document_hash: str | None = None
    relative_path: str | None = None
    repository: str | None = None
    language: str | None = None


class ChatRequest(BaseModel):
    question: str
    repository: str = DEFAULT_REPOSITORY
    repositories: list[str] | None = None
    languages: list[str] | None = None
    model: str = LLM_MODEL


def get_rag() -> RagService:
    global _rag

    if _rag is None:
        logger.info("[INGEST] Initializing RAG service")
        _rag = RagService()

    return _rag


@app.get("/health")
def health():
    embedding_info = embedding_diagnostics()
    gpu_name = _selected_embedding_gpu_name(embedding_info)

    response = {
        "status": "ok",
        "qdrant": "ok",
        "collection": COLLECTION_NAME,
        "points_count": 0,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_device_config": EMBEDDING_DEVICE,
        "embedding_device_selected": embedding_info["selected_device"],
        "cuda_available": embedding_info["cuda_available"],
        "cuda_version": embedding_info["torch_cuda_version"],
        "gpu_name": gpu_name,
        "default_llm": LLM_MODEL
    }

    try:
        rag = get_rag()
        response["points_count"] = rag.vector_store.points_count()
        response["ollama"] = rag.llm.diagnostics(LLM_MODEL)
        if response["ollama"]["status"] != "ok":
            response["status"] = "degraded"
    except Exception as exc:
        response["status"] = "degraded"
        response["qdrant"] = "error"
        response["points_count"] = None
        response["qdrant_error"] = str(exc)

    if response["status"] != "ok":
        return JSONResponse(status_code=503, content=response)

    return response


@app.get("/debug/routes")
def debug_routes():
    routes = []

    for route in app.routes:
        methods = sorted(route.methods or [])
        routes.append({
            "path": route.path,
            "methods": methods,
            "name": route.name
        })

    return {
        "status": "ok",
        "routes": sorted(routes, key=lambda item: (item["path"], item["methods"]))
    }


@app.get("/debug/config")
def debug_config():
    return {
        "books_root": str(BOOKS_ROOT.resolve()),
        "collection": COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_device_config": EMBEDDING_DEVICE,
        "default_llm": LLM_MODEL,
        "embedding_batch_size": EMBEDDING_BATCH_SIZE,
        "qdrant_upsert_batch_size": QDRANT_UPSERT_BATCH_SIZE
    }


@app.get("/debug/embedding")
def debug_embedding():
    return embedding_diagnostics()


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


@app.get("/books/hashes")
def books_hashes(
    repository: str | None = None,
    language: str | None = None,
    sort_by: str = Query(default="path", pattern="^(path|size)$"),
    max_mb: float | None = Query(default=None, ge=0),
    limit: int | None = Query(default=None, ge=0),
):
    if repository is not None:
        _validate_values([repository], SUPPORTED_REPOSITORIES, "repository")
    if language is not None:
        _validate_values([language], SUPPORTED_LANGUAGES, "language")

    try:
        candidates = _discover_book_candidates()
    except (FileNotFoundError, NotADirectoryError) as exc:
        return {
            "status": "error",
            "message": str(exc)
        }

    discovered_count = len(candidates)
    candidates = _select_candidates(
        candidates,
        repository=repository,
        language=language,
        sort_by=sort_by,
        max_mb=max_mb,
    )
    selected_before_limit = len(candidates)

    if limit is not None:
        candidates = candidates[:limit]

    books = _hash_selected_candidates(candidates)

    return {
        "status": "ok",
        "books_root": str(BOOKS_ROOT.resolve()),
        "discovered": discovered_count,
        "selected": len(books),
        "selected_before_limit": selected_before_limit,
        "books": [_book_hash_result(book) for book in books],
    }


@app.get("/books/no-text")
def books_no_text(
    repository: str | None = None,
    language: str | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
):
    if repository is not None:
        _validate_values([repository], SUPPORTED_REPOSITORIES, "repository")
    if language is not None:
        _validate_values([language], SUPPORTED_LANGUAGES, "language")

    books = get_rag().vector_store.list_no_text_documents(
        repository=repository,
        language=language,
        limit=limit,
    )

    return {
        "status": "ok",
        "count": len(books),
        "books": books,
    }


@app.post("/ingest")
def ingest(request: IngestRequest):
    logger.info("[INGEST] /ingest endpoint entered")
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

    return get_rag().ingest_book(book)


@app.post("/deindex")
def deindex(request: DeindexRequest):
    logger.info("[DEINDEX] /deindex endpoint entered")

    if request.repository is not None:
        _validate_values([request.repository], SUPPORTED_REPOSITORIES, "repository")
    if request.language is not None:
        _validate_values([request.language], SUPPORTED_LANGUAGES, "language")

    relative_path = _normalize_relative_path(request.relative_path)
    if not request.document_hash and not relative_path:
        raise HTTPException(
            status_code=422,
            detail="document_hash or relative_path is required"
        )

    selector = {
        "document_hash": request.document_hash,
        "relative_path": relative_path,
        "repository": request.repository,
        "language": request.language,
    }
    logger.info("[DEINDEX] selector=%s", selector)

    vector_store = get_rag().vector_store
    matched_points = vector_store.count_document_points(**selector)
    if matched_points > 0:
        vector_store.delete_document_points(**selector)

    logger.info("[DEINDEX] deleted %s point(s)", matched_points)
    return {
        "status": "ok",
        "deleted_points": matched_points,
        "matched_points": matched_points,
        "selector": selector,
    }


@app.post("/books/sync-index")
def books_sync_index(dry_run: bool = False):
    logger.info("[SYNC] /books/sync-index endpoint entered dry_run=%s", dry_run)

    try:
        candidates = _select_candidates(
            _discover_book_candidates(),
            repository=None,
            language=None,
            sort_by="path",
            max_mb=None,
        )
    except (FileNotFoundError, NotADirectoryError) as exc:
        return {
            "status": "error",
            "message": str(exc),
        }

    books = _hash_selected_candidates(candidates)
    filesystem_by_hash = _group_books_by_hash(books)
    vector_store = get_rag().vector_store
    indexed_documents = vector_store.list_indexed_documents()
    indexed_hashes = {document["document_hash"] for document in indexed_documents}

    removed = []
    moved = []
    unchanged = 0
    ambiguous = []

    for document in indexed_documents:
        document_hash = document["document_hash"]
        current_books = filesystem_by_hash.get(document_hash, [])

        if not current_books:
            detail = {
                "document_hash": document_hash,
                "old": _indexed_document_metadata(document),
                "point_count": document["point_count"],
                "statuses": document["statuses"],
            }
            if not dry_run:
                vector_store.delete_document_points(document_hash=document_hash)
            removed.append(detail)
            continue

        current_metadata_options = [_book_sync_metadata(book) for book in current_books]
        old_metadata = _indexed_document_metadata(document)
        matching_metadata = [
            metadata
            for metadata in current_metadata_options
            if metadata == old_metadata
        ]

        if matching_metadata:
            unchanged += 1
            continue

        if len(current_metadata_options) > 1:
            ambiguous.append({
                "document_hash": document_hash,
                "old": old_metadata,
                "current_options": current_metadata_options,
                "point_count": document["point_count"],
                "statuses": document["statuses"],
            })
            continue

        new_metadata = current_metadata_options[0]
        detail = {
            "document_hash": document_hash,
            "old": old_metadata,
            "new": new_metadata,
            "point_count": document["point_count"],
            "statuses": document["statuses"],
        }
        if not dry_run:
            vector_store.update_document_metadata(document_hash, new_metadata)
        moved.append(detail)

    new_unindexed = [
        _book_hash_result(book)
        for document_hash, current_books in filesystem_by_hash.items()
        if document_hash not in indexed_hashes
        for book in current_books
    ]
    duplicate_files = [
        {
            "document_hash": document_hash,
            "files": [_book_hash_result(book) for book in current_books],
        }
        for document_hash, current_books in filesystem_by_hash.items()
        if len(current_books) > 1
    ]

    response = {
        "status": "ok",
        "dry_run": dry_run,
        "books_root": str(BOOKS_ROOT.resolve()),
        "filesystem_documents": len(filesystem_by_hash),
        "filesystem_files": len(books),
        "indexed_documents": len(indexed_documents),
        "removed": len(removed),
        "moved": len(moved),
        "unchanged": unchanged,
        "ambiguous": len(ambiguous),
        "new_unindexed": len(new_unindexed),
        "duplicate_hashes": len(duplicate_files),
        "details": {
            "removed": removed,
            "moved": moved,
            "ambiguous": ambiguous,
            "new_unindexed": new_unindexed,
            "duplicate_hashes": duplicate_files,
        },
    }
    logger.info(
        "[SYNC] complete: removed=%s moved=%s ambiguous=%s new_unindexed=%s dry_run=%s",
        response["removed"],
        response["moved"],
        response["ambiguous"],
        response["new_unindexed"],
        dry_run,
    )
    return response


@app.post("/ingest/all")
def ingest_all(
    limit: int | None = Query(default=None, ge=0),
    repository: str | None = None,
    language: str | None = None,
    dry_run: bool = False,
    sort_by: str = Query(default="path", pattern="^(path|size)$"),
    max_mb: float | None = Query(default=None, ge=0),
    force_reindex: bool = False
):
    logger.info("[INGEST] /ingest/all endpoint entered")
    logger.info(
        "[INGEST] query params: dry_run=%s, limit=%s, repository=%s, language=%s, sort_by=%s, max_mb=%s, force_reindex=%s",
        dry_run,
        limit,
        repository,
        language,
        sort_by,
        max_mb,
        force_reindex,
    )
    if repository is not None:
        _validate_values([repository], SUPPORTED_REPOSITORIES, "repository")
    if language is not None:
        _validate_values([language], SUPPORTED_LANGUAGES, "language")

    logger.info("[INGEST] Starting full ingestion")
    logger.info("[INGEST] Books root: %s", BOOKS_ROOT.resolve())
    logger.info("[INGEST] Scanning for PDF files")

    try:
        candidates = _discover_book_candidates()
    except (FileNotFoundError, NotADirectoryError) as exc:
        return {
            "status": "error",
            "message": str(exc)
        }

    discovered_count = len(candidates)
    logger.info("[INGEST] Discovered %s PDF files", discovered_count)

    candidates = _select_candidates(
        candidates,
        repository=repository,
        language=language,
        sort_by=sort_by,
        max_mb=max_mb,
    )
    selected_before_limit = len(candidates)

    if limit is not None:
        candidates = candidates[:limit]

    logger.info(
        "[INGEST] Selected %s PDF files after filters repository=%s language=%s sort_by=%s max_mb=%s limit=%s",
        len(candidates),
        repository,
        language,
        sort_by,
        max_mb,
        limit,
    )

    books = _hash_selected_candidates(candidates)

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
            details.append(
                get_rag().ingest_book(
                    book,
                    index=index,
                    total=total,
                    force_reindex=force_reindex,
                )
            )
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


@app.post("/ingest/test-one")
def ingest_test_one(
    relative_path: str | None = None,
    repository: str | None = None,
    language: str | None = None,
    sort_by: str = Query(default="path", pattern="^(path|size)$"),
    max_mb: float | None = Query(default=None, ge=0),
    force_reindex: bool = False
):
    logger.info("[INGEST] /ingest/test-one endpoint entered")
    logger.info(
        "[INGEST] query params: relative_path=%s, repository=%s, language=%s, sort_by=%s, max_mb=%s, force_reindex=%s",
        relative_path,
        repository,
        language,
        sort_by,
        max_mb,
        force_reindex,
    )

    if repository is not None:
        _validate_values([repository], SUPPORTED_REPOSITORIES, "repository")
    if language is not None:
        _validate_values([language], SUPPORTED_LANGUAGES, "language")

    logger.info("[INGEST] Books root: %s", BOOKS_ROOT.resolve())

    try:
        if relative_path:
            candidate = _candidate_from_relative_path(relative_path)
            candidates = [candidate]
        else:
            logger.info("[INGEST] Scanning for first debug PDF")
            candidates = _select_candidates(
                _discover_book_candidates(),
                repository=repository,
                language=language,
                sort_by=sort_by,
                max_mb=max_mb,
            )
            candidates = candidates[:1]
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        return {
            "status": "error",
            "message": str(exc)
        }

    if not candidates:
        return {
            "status": "error",
            "message": "No matching PDF found"
        }

    candidate = candidates[0]

    if repository is not None and candidate["repository"] != repository:
        return {
            "status": "error",
            "message": f"Selected file repository is {candidate['repository']}, not {repository}"
        }
    if language is not None and candidate["language"] != language:
        return {
            "status": "error",
            "message": f"Selected file language is {candidate['language']}, not {language}"
        }

    book = _hash_selected_candidates([candidate])[0]

    try:
        detail = get_rag().ingest_book(
            book,
            index=1,
            total=1,
            force_reindex=force_reindex,
        )
    except Exception as exc:
        logger.exception("[INGEST] [1/1] Unexpected ingestion failure")
        detail = _book_result(book, status="failed", error=str(exc))

    return {
        "status": "ok",
        "books_root": str(BOOKS_ROOT.resolve()),
        "detail": detail
    }


@app.post("/chat")
def chat(request: ChatRequest):
    _validate_values([request.repository], SUPPORTED_REPOSITORIES, "repository")

    repositories = request.repositories or [request.repository]
    _validate_values(repositories, SUPPORTED_REPOSITORIES, "repositories")

    if request.languages is not None:
        _validate_values(request.languages, SUPPORTED_LANGUAGES, "languages")

    try:
        return get_rag().chat(
            question=request.question,
            repository=request.repository,
            repositories=repositories,
            languages=request.languages,
            model=request.model
        )
    except LlmServiceUnavailable as exc:
        logger.warning("[CHAT] LLM service unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail={
                "message": "LLM service unavailable",
                "error": str(exc),
            }
        ) from exc


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


def _normalize_relative_path(relative_path: str | None) -> str | None:
    if not relative_path:
        return None

    requested_path = Path(relative_path)
    if requested_path.is_absolute():
        try:
            return requested_path.resolve().relative_to(BOOKS_ROOT.resolve()).as_posix()
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail="relative_path must point to a file under BOOKS_ROOT"
            ) from exc

    return relative_path.replace("\\", "/").lstrip("/")


def _discover_book_candidates() -> list[dict]:
    root = BOOKS_ROOT.resolve()

    if not root.exists():
        raise FileNotFoundError(f"BOOKS_ROOT does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"BOOKS_ROOT is not a directory: {root}")

    candidates = []

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() != ".pdf":
            continue

        candidate = _candidate_from_path(path, root)
        if candidate is not None:
            candidates.append(candidate)

    return candidates


def _candidate_from_relative_path(relative_path: str) -> dict:
    root = BOOKS_ROOT.resolve()
    requested_path = Path(relative_path)

    if requested_path.is_absolute():
        absolute_path = requested_path.resolve()
    else:
        absolute_path = (root / requested_path).resolve()

    try:
        absolute_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("relative_path must point to a PDF under BOOKS_ROOT") from exc

    candidate = _candidate_from_path(absolute_path, root)
    if candidate is None:
        raise ValueError(
            "relative_path must point to a PDF under BOOKS_ROOT/<repository>/<language>/..."
        )

    return candidate


def _candidate_from_path(path: Path, root: Path) -> dict | None:
    absolute_path = path.resolve()

    if not absolute_path.is_file() or absolute_path.suffix.lower() != ".pdf":
        return None

    try:
        relative = absolute_path.relative_to(root)
    except ValueError:
        return None

    parts = relative.parts
    if len(parts) < 3:
        return None

    repository = parts[0]
    language = parts[1]

    if repository not in SUPPORTED_REPOSITORIES or language not in SUPPORTED_LANGUAGES:
        return None

    return {
        "absolute_path": absolute_path,
        "relative_path": relative.as_posix(),
        "repository": repository,
        "language": language,
        "topic_path": "/".join(parts[2:-1]),
        "file_name": absolute_path.name,
        "size_bytes": absolute_path.stat().st_size,
    }


def _select_candidates(
    candidates: list[dict],
    repository: str | None,
    language: str | None,
    sort_by: str,
    max_mb: float | None
) -> list[dict]:
    selected = [
        candidate
        for candidate in candidates
        if (repository is None or candidate["repository"] == repository)
        and (language is None or candidate["language"] == language)
    ]

    if max_mb is not None:
        max_bytes = max_mb * 1024 * 1024
        before_max_size = len(selected)
        selected = [
            candidate
            for candidate in selected
            if candidate["size_bytes"] <= max_bytes
        ]
        logger.info(
            "[INGEST] Size filter max_mb=%s kept %s/%s PDFs",
            max_mb,
            len(selected),
            before_max_size,
        )

    if sort_by == "size":
        selected.sort(key=lambda candidate: (candidate["size_bytes"], candidate["relative_path"]))
    else:
        selected.sort(key=lambda candidate: candidate["relative_path"])

    return selected


def _hash_selected_candidates(candidates: list[dict]) -> list[BookFile]:
    books = []
    total = len(candidates)

    logger.info("[INGEST] Hashing %s selected PDF files", total)
    for index, candidate in enumerate(candidates, start=1):
        logger.info(
            "[INGEST] [%s/%s] Hashing %s size=%.2f MB",
            index,
            total,
            candidate["relative_path"],
            candidate["size_bytes"] / 1024 / 1024,
        )
        step_start = time.perf_counter()
        books.append(_candidate_to_book(candidate))
        logger.info(
            "[INGEST] [%s/%s] SHA-256 took %.1fs",
            index,
            total,
            time.perf_counter() - step_start,
        )

    return books


def _candidate_to_book(candidate: dict) -> BookFile:
    return BookFile(
        absolute_path=candidate["absolute_path"],
        relative_path=candidate["relative_path"],
        repository=candidate["repository"],
        language=candidate["language"],
        topic_path=candidate["topic_path"],
        file_name=candidate["file_name"],
        sha256=calculate_sha256(candidate["absolute_path"]),
    )


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


def _book_hash_result(book: BookFile) -> dict:
    return {
        "file": book.relative_path,
        "repository": book.repository,
        "language": book.language,
        "document_hash": book.sha256,
        "topic_path": book.topic_path,
        "file_name": book.file_name,
    }


def _group_books_by_hash(books: list[BookFile]) -> dict[str, list[BookFile]]:
    grouped = {}
    for book in books:
        grouped.setdefault(book.sha256, []).append(book)
    return grouped


def _book_sync_metadata(book: BookFile) -> dict:
    return {
        "repository": book.repository,
        "language": book.language,
        "relative_path": book.relative_path,
        "topic_path": book.topic_path,
        "file_name": book.file_name,
        "book": Path(book.file_name).stem,
    }


def _indexed_document_metadata(document: dict) -> dict:
    return {
        "repository": document.get("repository"),
        "language": document.get("language"),
        "relative_path": document.get("relative_path"),
        "topic_path": document.get("topic_path"),
        "file_name": document.get("file_name"),
        "book": document.get("book"),
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


def _selected_embedding_gpu_name(embedding_info: dict) -> str | None:
    selected_device = embedding_info.get("selected_device")
    if not selected_device or not selected_device.startswith("cuda"):
        return None

    device_index = 0
    if ":" in selected_device:
        try:
            device_index = int(selected_device.split(":", 1)[1])
        except ValueError:
            return None

    for device in embedding_info.get("devices", []):
        if device.get("index") == device_index:
            return device.get("name")

    return None
