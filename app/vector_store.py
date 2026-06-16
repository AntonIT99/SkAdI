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
