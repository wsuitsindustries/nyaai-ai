"""Document chunking — splits text into manageable pieces for embedding & retrieval.

Uses semantic boundary detection (headers, lists, code blocks, paragraph breaks)
rather than pure character count, producing more coherent chunks.
"""

import re

DEFAULT_CHUNK_SIZE = 512
DEFAULT_OVERLAP = 64


def _detect_boundaries(text: str) -> list[tuple[int, int, str]]:
    """Detect semantic boundaries in text. Returns list of (start, end, type)."""
    boundaries = []
    patterns = [
        (r"^#{1,6}\s.+", "header"),
        (r"^[-*]\s.+", "list_item"),
        (r"^\d+\.\s.+", "numbered_item"),
        (r"^```", "code_fence"),
        (r"^---", "thematic_break"),
        (r"^\w.+[:：]$", "label"),
    ]

    lines = text.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        for pattern, btype in patterns:
            if re.match(pattern, stripped):
                pos = sum(len(l) + 1 for l in lines[:i])
                boundaries.append((pos, pos + len(line), btype))
                break

    return boundaries


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    min_chunk: int = 100,
) -> list[str]:
    """Split text into semantically coherent chunks.

    Parameters
    ----------
    text : str
        Document text to chunk.
    chunk_size : int
        Target chunk size in characters.
    overlap : int
        Number of overlapping characters between adjacent chunks.
    min_chunk : int
        Minimum chunk size; smaller chunks are merged.

    Returns
    -------
    list[str]
        List of text chunks.
    """
    if not text.strip():
        return []

    paragraphs = re.split(r"\n\s*\n", text.strip())

    chunks: list[str] = []
    buffer = ""
    buffer_start_is_header = False

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        is_header = bool(re.match(r"^#{1,6}\s", para))

        if len(buffer) + len(para) + 1 <= chunk_size:
            if buffer:
                buffer += "\n\n"
            buffer += para
            if is_header:
                buffer_start_is_header = True
        else:
            if buffer:
                chunks.append(buffer)
            buffer = para
            buffer_start_is_header = is_header

            if len(para) > chunk_size:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                buffer = sentences[0] if sentences else para
                for sent in sentences[1:]:
                    if len(buffer) + len(sent) + 1 <= chunk_size:
                        buffer += " " + sent
                    else:
                        if buffer:
                            chunks.append(buffer)
                        buffer = sent

    if buffer:
        chunks.append(buffer)

    # Merge tiny chunks with neighbors
    merged = [chunks[0]] if chunks else []
    for chunk in chunks[1:]:
        if len(merged[-1]) < min_chunk:
            merged[-1] = merged[-1] + "\n\n" + chunk
        else:
            merged.append(chunk)
    chunks = merged

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
