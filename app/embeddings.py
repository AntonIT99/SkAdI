import logging

from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL


logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self):
        self.model = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.model is None:
            logger.info("[INGEST] Loading embedding model: %s", EMBEDDING_MODEL)
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("[INGEST] Embedding model loaded: %s", EMBEDDING_MODEL)

        return self.model.encode(texts, normalize_embeddings=True).tolist()
