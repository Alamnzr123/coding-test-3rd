from typing import List, Dict
import uuid


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """
    Simple whitespace chunker with overlap.
    Returns list of {id: str, text: str}
    """
    if not text:
        return []
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words).strip()
        if chunk_text:
            chunks.append({"id": str(uuid.uuid4()), "text": chunk_text})
        if end >= len(words):
            break
        start = max(0, end - overlap)
    return chunks