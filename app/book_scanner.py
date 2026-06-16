from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

from app.config import BOOKS_ROOT, SUPPORTED_LANGUAGES, SUPPORTED_REPOSITORIES


@dataclass
class BookFile:
    absolute_path: Path
    relative_path: str
    repository: str
    language: str
    topic_path: str
    file_name: str
    sha256: str

    def to_dict(self) -> dict:
        data = asdict(self)
        data["absolute_path"] = str(self.absolute_path)
        return data


def calculate_sha256(path: Path) -> str:
    digest = sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def scan_books(books_root: Path = BOOKS_ROOT) -> list[BookFile]:
    root = books_root.resolve()

    if not root.exists():
        raise FileNotFoundError(f"BOOKS_ROOT does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"BOOKS_ROOT is not a directory: {root}")

    books = []

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() != ".pdf":
            continue

        book = describe_book_file(path, root)
        if book is None:
            continue

        books.append(book)

    return books


def describe_book_file(path: Path, books_root: Path = BOOKS_ROOT) -> BookFile | None:
    root = books_root.resolve()
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

    topic_parts = parts[2:-1]
    return BookFile(
        absolute_path=absolute_path,
        relative_path=relative.as_posix(),
        repository=repository,
        language=language,
        topic_path="/".join(topic_parts),
        file_name=absolute_path.name,
        sha256=calculate_sha256(absolute_path),
    )
