import os
from pathlib import Path

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "books"

BOOKS_ROOT = Path(os.getenv("BOOKS_ROOT", r"C:\Users\alpha\OneDrive\Dokumente\Books"))
DEFAULT_REPOSITORY = "default"
SUPPORTED_REPOSITORIES = {"default", "sensitive"}
SUPPORTED_LANGUAGES = {"de", "en", "fr"}

EMBEDDING_MODEL = "BAAI/bge-m3"
LLM_MODEL = "qwen3:14b"
