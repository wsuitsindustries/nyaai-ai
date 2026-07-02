"""Document chunking — splits text into manageable pieces for embedding & retrieval."""

import re

DEFAULT_CHUNK_SIZE = 512
DEFAULT_OVERLAP = 64


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks of roughly `chunk_size` characters.

    Chunks are split on paragraph or sentence boundaries when possible.
    """
    if not text.strip():
        return []

    paragraphs = re.split(r"\n\s*\n", text.strip())
    chunks: list[str] = []
    buffer = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(buffer) + len(para) + 1 <= chunk_size:
            buffer = (buffer + "\n\n" + para).strip()
        else:
            if buffer:
                chunks.append(buffer)
            # If the paragraph itself is longer than chunk_size, split on sentences
            if len(para) > chunk_size:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                for sent in sentences:
                    if len(buffer) + len(sent) + 1 <= chunk_size:
                        buffer = (buffer + " " + sent).strip()
                    else:
                        if buffer:
                            chunks.append(buffer)
                        buffer = sent
            else:
                buffer = para

    if buffer:
        chunks.append(buffer)

    # Apply overlap
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            overlap_text = prev[-overlap:] if len(prev) > overlap else prev
            overlapped.append(overlap_text + chunks[i])
        chunks = overlapped

    return chunks


def chunk_file(path: str, **kwargs) -> list[str]:
    """Read a file and return its chunks."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    return chunk_text(text, **kwargs)
