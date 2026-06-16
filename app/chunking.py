def chunk_text(text: str, max_chars: int = 1800, overlap: int = 250) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + max_chars
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += max_chars - overlap

    return chunks