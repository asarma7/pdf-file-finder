def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[tuple[int, int, str]]:
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + chunk_size)
        chunk = text[start:end]
        chunks.append((start, end, chunk))
        if end == length:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks
