from pathlib import Path
from uuid import uuid4

from qdrant_client.models import PointStruct

from app.book_scanner import BookFile, describe_book_file
from app.pdf_loader import load_pdf_pages
from app.chunking import chunk_text
from app.config import LLM_MODEL
from app.embeddings import EmbeddingService
from app.vector_store import VectorStore
from app.llm import LlmService


class RagService:
    def __init__(self):
        self.embeddings = EmbeddingService()
        self.vector_store = VectorStore(vector_size=1024)
        self.llm = LlmService()

    def ingest_pdf(self, pdf_path: Path, repository: str = "default") -> dict:
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

        return self.ingest_book(book)

    def ingest_book(self, book: BookFile) -> dict:
        result = {
            "file": book.relative_path,
            "repository": book.repository,
            "language": book.language,
            "document_hash": book.sha256,
            "status": "indexed",
            "chunks": 0
        }

        try:
            if self.vector_store.document_exists(book.sha256):
                result["status"] = "skipped"
                return result
        except Exception as exc:
            result["status"] = "failed"
            result["error"] = str(exc)
            return result

        try:
            pages = load_pdf_pages(book.absolute_path)
            points = self._build_points(book, pages)
        except Exception as exc:
            result["status"] = "failed"
            result["error"] = str(exc)
            return result

        if not points:
            result["status"] = "no_text"
            return result

        try:
            self.vector_store.upsert_chunks(points)
        except Exception as exc:
            result["status"] = "failed"
            result["error"] = str(exc)
            return result

        result["chunks"] = len(points)
        return result

    def _build_points(self, book: BookFile, pages: list[dict]) -> list[PointStruct]:
        texts = []
        payloads = []

        for page in pages:
            chunks = chunk_text(page["text"])

            for chunk in chunks:
                texts.append(chunk)
                payloads.append({
                    "document_hash": book.sha256,
                    "repository": book.repository,
                    "language": book.language,
                    "relative_path": book.relative_path,
                    "topic_path": book.topic_path,
                    "file_name": book.file_name,
                    "book": page["book"],
                    "page": page["page"],
                    "text": chunk
                })

        if not texts:
            return []

        vectors = self.embeddings.embed(texts)

        return [
            PointStruct(
                id=str(uuid4()),
                vector=vector,
                payload=payload
            )
            for vector, payload in zip(vectors, payloads)
        ]

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
