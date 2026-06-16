import logging

import fitz
from pathlib import Path


logger = logging.getLogger(__name__)


def load_pdf_pages(pdf_path: Path) -> list[dict]:
    pages = []
    extracted_characters = 0

    logger.info("[PDF] Opening %s", pdf_path)
    with fitz.open(pdf_path) as doc:
        page_count = len(doc)
        logger.info("[PDF] Pages: %s", page_count)

        for index, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                extracted_characters += len(text)
                pages.append({
                    "book": pdf_path.stem,
                    "page": index + 1,
                    "text": text
                })

            processed = index + 1
            if processed % 25 == 0 or processed == page_count:
                logger.info("[PDF] Processed %s/%s pages", processed, page_count)

    logger.info("[PDF] Total extracted characters: %s", extracted_characters)
    return pages
