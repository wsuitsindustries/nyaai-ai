from ai.embeddings import embed, cosine_similarity


def retrieve(
    query: str,
    chunks: list[str],
    top_k: int = 5,
) -> list[tuple[float, str]]:
    if not chunks:
        return []

    qvec = embed(query)
    scored = [
        (cosine_similarity(qvec, embed(chunk)), chunk) for chunk in chunks
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(score, chunk) for score, chunk in scored[:top_k]]


def retrieve_texts(
    query: str,
    chunks: list[str],
    top_k: int = 5,
) -> list[str]:
    return [chunk for _, chunk in retrieve(query, chunks, top_k)]