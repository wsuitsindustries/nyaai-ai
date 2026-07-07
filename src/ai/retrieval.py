from ai.embeddings import embed, cosine_similarity


def retrieve(
    query: str,
    chunks: list[str],
    chunk_embeddings: list[list[float]] | None = None,
    top_k: int = 5,
) -> list[tuple[float, str]]:
    if not chunks:
        return []

    qvec = embed(query)

    if chunk_embeddings and len(chunk_embeddings) == len(chunks):
        scored = [
            (cosine_similarity(qvec, ce), chunk)
            for ce, chunk in zip(chunk_embeddings, chunks)
        ]
    else:
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


def retrieve_with_embeddings(
    query: str,
    chunks: list[str],
    chunk_embeddings: list[list[float]] | None = None,
    top_k: int = 5,
) -> list[str]:
    return [chunk for _, chunk in retrieve(query, chunks, chunk_embeddings, top_k)]