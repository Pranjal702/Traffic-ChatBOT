def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    """Split text into chunks with overlap.

    Args:
        text: input string
        chunk_size: maximum characters per chunk
        overlap: number of overlapping characters between chunks

    Returns:
        List of chunks (strings)
    """
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        if end == length:
            break
        start = max(end - overlap, end)
    return chunks
