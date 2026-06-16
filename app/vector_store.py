from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    IsEmptyCondition,
    MatchAny,
    MatchValue,
    PayloadField,
    PointStruct,
    VectorParams,
)

from app.config import QDRANT_URL, COLLECTION_NAME


class VectorStore:
    def __init__(self, vector_size: int = 1024):
        self.client = QdrantClient(url=QDRANT_URL)
        self.vector_size = vector_size
        self.ensure_collection()

    def ensure_collection(self):
        collections = self.client.get_collections().collections
        names = [c.name for c in collections]

        if COLLECTION_NAME not in names:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )

    def upsert_chunks(self, points: list[PointStruct]):
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

    def upsert_no_text_marker(self, point: PointStruct):
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[point]
        )

    def document_exists(self, document_hash: str) -> bool:
        result = self.client.count(
            collection_name=COLLECTION_NAME,
            count_filter=self._complete_document_filter(document_hash),
            exact=True
        )
        return result.count > 0

    def mark_points_complete(self, point_ids: list[str]) -> None:
        if not point_ids:
            return

        self.client.set_payload(
            collection_name=COLLECTION_NAME,
            payload={"document_status": "complete"},
            points=point_ids,
        )

    def points_count(self) -> int:
        collection = self.client.get_collection(collection_name=COLLECTION_NAME)
        return collection.points_count or 0

    def count_document_points(
        self,
        document_hash: str | None = None,
        relative_path: str | None = None,
        repository: str | None = None,
        language: str | None = None,
    ) -> int:
        result = self.client.count(
            collection_name=COLLECTION_NAME,
            count_filter=self._document_filter(
                document_hash=document_hash,
                relative_path=relative_path,
                repository=repository,
                language=language,
            ),
            exact=True,
        )
        return result.count

    def delete_document_points(
        self,
        document_hash: str | None = None,
        relative_path: str | None = None,
        repository: str | None = None,
        language: str | None = None,
    ) -> None:
        self.client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=self._document_filter(
                document_hash=document_hash,
                relative_path=relative_path,
                repository=repository,
                language=language,
            ),
        )

    def delete_points_by_id(self, point_ids: list[str]) -> None:
        if not point_ids:
            return

        self.client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=point_ids,
        )

    def update_document_metadata(self, document_hash: str, metadata: dict) -> None:
        self.client.set_payload(
            collection_name=COLLECTION_NAME,
            payload=metadata,
            points=self._document_filter(document_hash=document_hash),
        )

    def list_indexed_documents(self) -> list[dict]:
        documents = {}
        offset = None
        payload_fields = [
            "document_hash",
            "document_status",
            "repository",
            "language",
            "relative_path",
            "topic_path",
            "file_name",
            "book",
        ]

        while True:
            batch, offset = self.client.scroll(
                collection_name=COLLECTION_NAME,
                limit=256,
                with_payload=payload_fields,
                with_vectors=False,
                offset=offset,
            )

            if not batch:
                break

            for record in batch:
                payload = record.payload or {}
                document_hash = payload.get("document_hash")
                if not document_hash:
                    continue

                document = documents.setdefault(
                    document_hash,
                    {
                        "document_hash": document_hash,
                        "repository": None,
                        "language": None,
                        "relative_path": None,
                        "topic_path": None,
                        "file_name": None,
                        "book": None,
                        "point_count": 0,
                        "statuses": set(),
                    },
                )

                document["point_count"] += 1
                status = payload.get("document_status")
                if status:
                    document["statuses"].add(status)

                for key in ("repository", "language", "relative_path", "topic_path", "file_name", "book"):
                    value = payload.get(key)
                    if document.get(key) is None and value is not None:
                        document[key] = value

            if offset is None:
                break

        return [
            {
                **document,
                "statuses": sorted(document["statuses"]),
            }
            for document in documents.values()
        ]

    def list_no_text_documents(
        self,
        repository: str | None = None,
        language: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        records = []
        offset = None
        scroll_filter = self._no_text_filter(repository=repository, language=language)

        while len(records) < limit:
            batch, offset = self.client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=scroll_filter,
                limit=min(100, limit - len(records)),
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )

            if not batch:
                break

            for record in batch:
                payload = record.payload or {}
                records.append({
                    "file": payload.get("relative_path"),
                    "repository": payload.get("repository"),
                    "language": payload.get("language"),
                    "document_hash": payload.get("document_hash"),
                    "status": payload.get("document_status"),
                    "file_name": payload.get("file_name"),
                    "topic_path": payload.get("topic_path"),
                    "extracted_pages": payload.get("extracted_pages"),
                    "extracted_characters": payload.get("extracted_characters"),
                })

            if offset is None:
                break

        return records

    def search(
        self,
        query_vector: list[float],
        repositories: list[str] | str,
        languages: list[str] | None = None,
        limit: int = 5
    ):
        query_filter = self._build_search_filter(repositories, languages)

        if hasattr(self.client, "search"):
            return self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=limit,
                query_filter=query_filter
            )

        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        return response.points

    def _build_search_filter(
        self,
        repositories: list[str] | str,
        languages: list[str] | None
    ) -> Filter:
        if isinstance(repositories, str):
            repositories = [repositories]

        conditions = [
            FieldCondition(
                key="repository",
                match=self._match_values(repositories)
            ),
            Filter(
                should=[
                    FieldCondition(
                        key="document_status",
                        match=MatchValue(value="complete")
                    ),
                    IsEmptyCondition(
                        is_empty=PayloadField(key="document_status")
                    )
                ]
            )
        ]

        if languages:
            conditions.append(
                FieldCondition(
                    key="language",
                    match=self._match_values(languages)
                )
            )

        return Filter(must=conditions)

    def _match_values(self, values: list[str]):
        if len(values) == 1:
            return MatchValue(value=values[0])
        return MatchAny(any=values)

    def _complete_document_filter(self, document_hash: str) -> Filter:
        return Filter(
            must=[
                FieldCondition(
                    key="document_hash",
                    match=MatchValue(value=document_hash)
                ),
                FieldCondition(
                    key="document_status",
                    match=MatchValue(value="complete")
                )
            ]
        )

    def _document_filter(
        self,
        document_hash: str | None = None,
        relative_path: str | None = None,
        repository: str | None = None,
        language: str | None = None,
    ) -> Filter:
        conditions = []

        if document_hash is not None:
            conditions.append(
                FieldCondition(
                    key="document_hash",
                    match=MatchValue(value=document_hash)
                )
            )
        if relative_path is not None:
            conditions.append(
                FieldCondition(
                    key="relative_path",
                    match=MatchValue(value=relative_path)
                )
            )
        if repository is not None:
            conditions.append(
                FieldCondition(
                    key="repository",
                    match=MatchValue(value=repository)
                )
            )
        if language is not None:
            conditions.append(
                FieldCondition(
                    key="language",
                    match=MatchValue(value=language)
                )
            )

        if not conditions:
            raise ValueError("At least one document selector is required")

        return Filter(must=conditions)

    def _no_text_filter(
        self,
        repository: str | None = None,
        language: str | None = None,
    ) -> Filter:
        conditions = [
            FieldCondition(
                key="document_status",
                match=MatchValue(value="no_text")
            )
        ]

        if repository is not None:
            conditions.append(
                FieldCondition(
                    key="repository",
                    match=MatchValue(value=repository)
                )
            )
        if language is not None:
            conditions.append(
                FieldCondition(
                    key="language",
                    match=MatchValue(value=language)
                )
            )

        return Filter(must=conditions)
