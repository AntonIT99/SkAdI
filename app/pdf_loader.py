import fitz
from pathlib import Path


def load_pdf_pages(pdf_path: Path) -> list[dict]:
    pages = []

    with fitz.open(pdf_path) as doc:
        for index, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                pages.append({
                    "book": pdf_path.stem,
                    "page": index + 1,
                    "text": text
                })

    return pages
