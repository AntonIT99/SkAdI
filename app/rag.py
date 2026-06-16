import logging
import time
from math import ceil
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client.models import PointStruct

from app.book_scanner import BookFile, describe_book_file
from app.pdf_loader import load_pdf_pages
from app.chunking import chunk_text
from app.config import EMBEDDING_BATCH_SIZE, LLM_MODEL, QDRANT_UPSERT_BATCH_SIZE
from app.embeddings import EmbeddingService
from app.vector_store import VectorStore
from app.llm import LlmService


logger = logging.getLogger(__name__)


class RagService:
    def __init__(self):
        self.embeddings = EmbeddingService()
        self.vector_store = VectorStore(vector_size=1024)
        self.llm = LlmService()

    def ingest_pdf(self, pdf_path: Path, repository: str = "default", force_reindex: bool = False) -> dict:
        book = describe_book_file(pdf_path)
        if book is None:
            return {
                "file": str(pdf_path),
                "repository": repository,
                "language": None,
                "document_hash": None,
                "status": "failed",
                "chunks": 0,
                "error": "PDF must be under BOOKS_ROOT/<repository>/<language>/..."
            }

        return self.ingest_book(book, force_reindex=force_reindex)

    def ingest_book(
        self,
        book: BookFile,
        index: int | None = None,
        total: int | None = None,
        force_reindex: bool = False
    ) -> dict:
        log_prefix = self._ingest_log_prefix(index, total)
        total_start = time.perf_counter()
        result = {
            "file": book.relative_path,
            "repository": book.repository,
            "language": book.language,
            "document_hash": book.sha256,
            "status": "indexed",
            "chunks": 0,
            "error": None
        }

        logger.info(
            "%s Processing %s repository=%s language=%s sha256=%s",
            log_prefix,
            book.relative_path,
            book.repository,
            book.language,
            book.sha256,
        )

        try:
            step_start = time.perf_counter()
            if force_reindex:
                logger.warning(
                    "%s force_reindex=true: complete-document skip is disabled; old chunks for changed files are not deleted",
                    log_prefix,
                )
            elif self.vector_store.document_exists(book.sha256):
                logger.info("%s Hash check took %.1fs", log_prefix, time.perf_counter() - step_start)
                result["status"] = "skipped"
                logger.info("%s Skipped: complete document already exists", log_prefix)
                logger.info("%s Done: skipped", log_prefix)
                logger.info("%s Total took %.1fs", log_prefix, time.perf_counter() - total_start)
                return result
            logger.info("%s Hash check took %.1fs", log_prefix, time.perf_counter() - step_start)
        except Exception as exc:
            result["status"] = "failed"
            result["error"] = str(exc)
            logger.exception("%s Failed while checking document existence", log_prefix)
            logger.info("%s Done: failed", log_prefix)
            logger.info("%s Total took %.1fs", log_prefix, time.perf_counter() - total_start)
            return result

        try:
            step_start = time.perf_counter()
            pages = load_pdf_pages(book.absolute_path)
            logger.info("%s PDF extraction took %.1fs", log_prefix, time.perf_counter() - step_start)
            extracted_characters = sum(len(page["text"]) for page in pages)
            logger.info(
                "%s Extracted %s pages, %s characters",
                log_prefix,
                len(pages),
                extracted_characters,
            )

            step_start = time.perf_counter()
            chunks = self._build_chunks(book, pages)
            logger.info("%s Chunking took %.1fs", log_prefix, time.perf_counter() - step_start)
            logger.info("%s Created %s chunks", log_prefix, len(chunks))
        except Exception as exc:
            result["status"] = "failed"
            result["error"] = str(exc)
            logger.exception("%s Failed while parsing or chunking PDF", log_prefix)
            logger.info("%s Done: failed", log_prefix)
            logger.info("%s Total took %.1fs", log_prefix, time.perf_counter() - total_start)
            return result

        if not chunks:
            result["status"] = "no_text"
            logger.info("%s Done: no_text", log_prefix)
            logger.info("%s Total took %.1fs", log_prefix, time.perf_counter() - total_start)
            return result

        try:
            inserted_count, embedding_seconds, qdrant_seconds = self._embed_and_upsert_chunks(
                book,
                chunks,
                log_prefix,
            )
            logger.info("%s Embedding took %.1fs", log_prefix, embedding_seconds)
            logger.info("%s Qdrant upsert took %.1fs", log_prefix, qdrant_seconds)
        except Exception as exc:
            result["status"] = "failed"
            result["error"] = str(exc)
            logger.exception("%s Failed while embedding or upserting chunks", log_prefix)
            logger.info("%s Done: failed", log_prefix)
            logger.info("%s Total took %.1fs", log_prefix, time.perf_counter() - total_start)
            return result

        result["chunks"] = inserted_count
        logger.info("%s Upserted %s chunks", log_prefix, inserted_count)
        logger.info("%s Done: indexed", log_prefix)
        logger.info("%s Total took %.1fs", log_prefix, time.perf_counter() - total_start)
        return result

    def _build_chunks(self, book: BookFile, pages: list[dict]) -> list[dict]:
        chunks = []

        for page in pages:
            page_chunks = chunk_text(page["text"])

            for chunk in page_chunks:
                chunks.append({
                    "text": chunk,
                    "payload": {
                        "document_hash": book.sha256,
                        "document_status": "indexing",
                        "repository": book.repository,
                        "language": book.language,
                        "relative_path": book.relative_path,
                        "topic_path": book.topic_path,
                        "file_name": book.file_name,
                        "book": page["book"],
                        "page": page["page"],
                        "chunk_index": len(chunks),
                        "text": chunk,
                    }
                })

        return chunks

    def _embed_and_upsert_chunks(self, book: BookFile, chunks: list[dict], log_prefix: str) -> tuple[int, float, float]:
        point_ids = []
        point_buffer = []
        inserted_count = 0
        upsert_batch_number = 0
        embedding_seconds = 0.0
        qdrant_seconds = 0.0
        embedding_batches = ceil(len(chunks) / EMBEDDING_BATCH_SIZE)
        upsert_batches = ceil(len(chunks) / QDRANT_UPSERT_BATCH_SIZE)

        logger.info("%s Embedding starts: %s chunks", log_prefix, len(chunks))

        for batch_index, start in enumerate(range(0, len(chunks), EMBEDDING_BATCH_SIZE), start=1):
            batch = chunks[start:start + EMBEDDING_BATCH_SIZE]
            logger.info("%s Embedding batch %s/%s", log_prefix, batch_index, embedding_batches)
            step_start = time.perf_counter()
            vectors = self.embeddings.embed([chunk["text"] for chunk in batch])
            embedding_seconds += time.perf_counter() - step_start
            if len(vectors) != len(batch):
                raise RuntimeError(
                    f"Embedding service returned {len(vectors)} vectors for {len(batch)} chunks"
                )

            for vector, chunk in zip(vectors, batch):
                point_id = self._point_id(book.sha256, chunk["payload"]["chunk_index"])
                point_ids.append(point_id)
                point_buffer.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=chunk["payload"],
                    )
                )

            while len(point_buffer) >= QDRANT_UPSERT_BATCH_SIZE:
                upsert_batch_number += 1
                batch_count, batch_seconds = self._flush_point_batch(
                    point_buffer[:QDRANT_UPSERT_BATCH_SIZE],
                    log_prefix,
                    upsert_batch_number,
                    upsert_batches,
                )
                inserted_count += batch_count
                qdrant_seconds += batch_seconds
                del point_buffer[:QDRANT_UPSERT_BATCH_SIZE]

        if point_buffer:
            upsert_batch_number += 1
            batch_count, batch_seconds = self._flush_point_batch(
                point_buffer,
                log_prefix,
                upsert_batch_number,
                upsert_batches,
            )
            inserted_count += batch_count
            qdrant_seconds += batch_seconds

        qdrant_seconds += self._mark_points_complete(point_ids, log_prefix)
        return inserted_count, embedding_seconds, qdrant_seconds

    def _flush_point_batch(
        self,
        points: list[PointStruct],
        log_prefix: str,
        batch_number: int,
        total_batches: int
    ) -> tuple[int, float]:
        if batch_number == 1:
            logger.info(
                "%s Qdrant upsert starts: batch_size=%s batches=%s",
                log_prefix,
                QDRANT_UPSERT_BATCH_SIZE,
                total_batches,
            )

        logger.info(
            "%s Qdrant upsert batch %s/%s chunks=%s",
            log_prefix,
            batch_number,
            total_batches,
            len(points),
        )
        step_start = time.perf_counter()
        self.vector_store.upsert_chunks(points)
        return len(points), time.perf_counter() - step_start

    def _mark_points_complete(self, point_ids: list[str], log_prefix: str) -> float:
        logger.info("%s Marking %s chunks complete", log_prefix, len(point_ids))
        step_start = time.perf_counter()
        for start in range(0, len(point_ids), QDRANT_UPSERT_BATCH_SIZE):
            self.vector_store.mark_points_complete(
                point_ids[start:start + QDRANT_UPSERT_BATCH_SIZE]
            )
        return time.perf_counter() - step_start

    def _point_id(self, document_hash: str, chunk_index: int) -> str:
        return str(uuid5(NAMESPACE_URL, f"{document_hash}:{chunk_index}"))

    def _ingest_log_prefix(self, index: int | None, total: int | None) -> str:
        if index is not None and total is not None:
            return f"[INGEST] [{index}/{total}]"
        return "[INGEST]"

    def chat(
        self,
        question: str,
        repository: str = "default",
        repositories: list[str] | None = None,
        languages: list[str] | None = None,
        model: str | None = None
    ):
        search_repositories = repositories or [repository]
        query_vector = self.embeddings.embed([question])[0]
        results = self.vector_store.search(
            query_vector=query_vector,
            repositories=search_repositories,
            languages=languages,
            limit=5
        )

        sources = []
        for hit in results:
            payload = hit.payload or {}
            sources.append({
                "book": payload.get("book"),
                "page": payload.get("page"),
                "text": payload.get("text"),
                "score": hit.score,
                "document_hash": payload.get("document_hash"),
                "repository": payload.get("repository"),
                "language": payload.get("language"),
                "relative_path": payload.get("relative_path"),
                "file_name": payload.get("file_name")
            })

        answer = self.llm.generate(
            question=question,
            sources=sources,
            model=model or LLM_MODEL
        )

        return {
            "answer": answer,
            "sources": sources
        }
