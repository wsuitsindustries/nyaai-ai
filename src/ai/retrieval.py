"""Hybrid retrieval — BM25 keyword + semantic embedding search with reranking.

Combines dense (embedding-based) and sparse (keyword-based) retrieval signals,
then reranks candidates for final selection.
"""

import math
import re

from ai.embeddings import embed, embed_batch, cosine_similarity

# ── BM25 (keyword search) ──────────────────────────────────────

_stopwords = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "by", "with", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "can", "could", "shall", "should", "may", "might", "it", "its", "that",
    "this", "these", "those", "what", "which", "who", "whom", "whose",
    "when", "where", "why", "how", "all", "each", "every", "both", "few",
    "more", "most", "some", "any", "no", "not", "only", "own", "same",
    "so", "than", "too", "very", "just", "because", "as", "until", "while",
    "about", "between", "through", "during", "before", "after", "above",
    "below", "up", "down", "out", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "i", "me", "my", "myself", "we",
    "our", "ours", "ourselves", "you", "your", "yours", "yourself",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "they", "them", "their", "theirs", "themselves",
}


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in re.findall(r"\w+", text) if t.lower() not in _stopwords and len(t) > 2]


def _idf(token: str, tokenized_docs: list[list[str]]) -> float:
    n = sum(1 for dt in tokenized_docs if token in dt)
    return math.log((len(tokenized_docs) - n + 0.5) / (n + 0.5) + 1.0)


def _bm25_score(query_tokens: list[str], doc: str, doc_tokens: list[str] | None, avg_dl: float, idf_cache: dict[str, float]) -> float:
    k1, b = 1.5, 0.75
    dl = len(doc_tokens) if doc_tokens else len(doc.split())
    score = 0.0
    doc_tokens_lower = [t.lower() for t in (doc_tokens or [])]
    for t in query_tokens:
        if t not in doc_tokens_lower:
            continue
        tf = doc_tokens_lower.count(t)
        idf_val = idf_cache.get(t) or 1.0
        score += idf_val * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
    return score


def _keyword_score(query: str, chunks: list[str]) -> list[float]:
    """Compute BM25-like keyword relevance scores."""
    qtokens = _tokenize(query)
    if not qtokens:
        return [0.0] * len(chunks)

    doc_tokens_list = [_tokenize(c) for c in chunks]
    total_tokens = sum(len(t) for t in doc_tokens_list)
    avg_dl = total_tokens / max(len(chunks), 1)

    all_tokens = set()
    for dt in doc_tokens_list:
        all_tokens.update(dt)
    idf_cache = {t: _idf(t, doc_tokens_list) for t in qtokens if t in all_tokens}

    scores = []
    for i, chunk in enumerate(chunks):
        scores.append(_bm25_score(qtokens, chunk, doc_tokens_list[i], avg_dl, idf_cache))
    return scores


def _normalize(scores: list[float]) -> list[float]:
    if not scores:
        return scores
    mn, mx = min(scores), max(scores)
    if mx - mn < 1e-10:
        return [0.5] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


# ── Query expansion ─────────────────────────────────────────────

async def expand_query(question: str) -> list[str]:
    """Generate query variations using LLM, with heuristic fallback."""
    from ai.llm import complete
    from ai.prompts import build_query_expansion_prompt

    queries = [question]

    try:
        prompt = build_query_expansion_prompt(question)
        response = await complete([{"role": "user", "content": prompt}])
        variations = [q.strip() for q in response.strip().split("\n") if q.strip()]
        queries.extend(v for v in variations if v.lower() != question.lower())
    except Exception:
        pass

    # Deduplicate
    seen = set()
    unique = []
    for q in queries:
        key = q.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(q)

    return unique[:5]  # max 5 variations


# ── Hybrid search ──────────────────────────────────────────────

async def hybrid_search(
    query: str,
    chunks: list[str],
    chunk_embeddings: list[list[float]] | None = None,
    top_k: int = 15,
    alpha: float = 0.5,
) -> list[tuple[float, str, int]]:
    """Hybrid search: fuses BM25 keyword scores with semantic embedding scores.

    Returns list of (combined_score, chunk_text, index) sorted by score descending.
    """
    if not chunks:
        return []

    # Semantic scores
    qvec = await embed(query)
    if chunk_embeddings and len(chunk_embeddings) == len(chunks):
        sem_scores = [cosine_similarity(qvec, ce) for ce in chunk_embeddings]
    else:
        chunk_vecs = await embed_batch(chunks)
        sem_scores = [cosine_similarity(qvec, cv) for cv in chunk_vecs]

    # Keyword scores
    kw_scores = _keyword_score(query, chunks)

    # Normalize both to [0, 1]
    sem_norm = _normalize(sem_scores)
    kw_norm = _normalize(kw_scores)

    # Fuse
    combined = [
        (alpha * sem_norm[i] + (1 - alpha) * kw_norm[i], chunks[i], i)
        for i in range(len(chunks))
    ]

    combined.sort(key=lambda x: x[0], reverse=True)
    return combined[:top_k]


# ── Reranking ──────────────────────────────────────────────────

async def rerank(
    query: str,
    candidates: list[tuple[float, str, int]],
    top_k: int = 5,
) -> list[tuple[float, str]]:
    """Rerank candidates using content-aware scoring.

    For each candidate, computes a detailed relevance score that considers:
    - Query term overlap density
    - Presence of query words near each other in the text
    - Chunk position bonus (earlier chunks tend to be more important)
    """
    qtokens = _tokenize(query)
    if not qtokens or not candidates:
        return [(s, c) for s, c, i in candidates[:top_k]]

    reranked = []
    for score, chunk, idx in candidates:
        chunk_lower = chunk.lower()
        qtokens_in_chunk = [t for t in qtokens if t in chunk_lower]

        # Term density: fraction of query tokens found in chunk
        token_coverage = len(qtokens_in_chunk) / max(len(qtokens), 1)

        # Proximity bonus: if multiple query tokens appear close together
        proximity = 0.0
        if len(qtokens_in_chunk) >= 2:
            positions = []
            for t in qtokens_in_chunk:
                pos = chunk_lower.find(t)
                if pos >= 0:
                    positions.append(pos)
            if len(positions) >= 2:
                max_gap = max(positions) - min(positions)
                proximity = 1.0 / (1.0 + max_gap / max(len(chunk), 1))

        # Position bonus: earlier chunks tend to be intro/overview
        position_bonus = 1.0 - (idx / max(len(candidates), 1)) * 0.1

        # Combine: original hybrid score (70%) + content signals (30%)
        rerank_score = 0.7 * score + 0.2 * token_coverage + 0.1 * proximity
        rerank_score *= position_bonus

        reranked.append((rerank_score, chunk))

    reranked.sort(key=lambda x: x[0], reverse=True)
    return reranked[:top_k]


# ── Public API ─────────────────────────────────────────────────

async def retrieve(
    query: str,
    chunks: list[str],
    chunk_embeddings: list[list[float]] | None = None,
    top_k: int = 10,
    use_hybrid: bool = True,
    use_rerank: bool = True,
) -> list[tuple[float, str]]:
    """Full retrieval pipeline: expansion → hybrid search → rerank."""
    if not chunks:
        return []

    # Step 1: Query expansion
    queries = await expand_query(query)

    # Step 2: Search for each query variation
    all_candidates: dict[int, list[tuple[float, str, int]]] = {}
    for q in queries:
        if use_hybrid:
            candidates = await hybrid_search(q, chunks, chunk_embeddings, top_k=top_k + 5)
        else:
            qvec = await embed(q)
            if chunk_embeddings and len(chunk_embeddings) == len(chunks):
                scores = [(cosine_similarity(qvec, ce), chunks[i], i) for i, ce in enumerate(chunk_embeddings)]
            else:
                chunk_vecs = await embed_batch(chunks)
                scores = [(cosine_similarity(qvec, cv), chunks[i], i) for i, cv in enumerate(chunk_vecs)]
            scores.sort(key=lambda x: x[0], reverse=True)
            candidates = scores[:top_k + 5]
        all_candidates[q] = candidates

    # Step 3: Fuse results from all query variations (unique chunks)
    seen_chunks: set[str] = set()
    fused = []
    # Interleave by max score
    chunk_scores: dict[str, float] = {}
    for q_candidates in all_candidates.values():
        for score, chunk, idx in q_candidates:
            if chunk not in chunk_scores or score > chunk_scores[chunk]:
                chunk_scores[chunk] = score

    fused = [(score, chunk) for chunk, score in chunk_scores.items()]
    fused.sort(key=lambda x: x[0], reverse=True)

    # Step 4: Rerank top candidates
    if use_rerank:
        indexed = [(score, chunk, i) for i, (score, chunk) in enumerate(fused[:top_k + 10])]
        return await rerank(query, indexed, top_k=top_k)

    return fused[:top_k]


async def retrieve_texts(
    query: str,
    chunks: list[str],
    top_k: int = 10,
) -> list[str]:
    """Convenience: return just the chunk texts."""
    results = await retrieve(query, chunks, top_k=top_k)
    return [chunk for _, chunk in results]


async def retrieve_with_embeddings(
    query: str,
    chunks: list[str],
    chunk_embeddings: list[list[float]] | None = None,
    top_k: int = 10,
) -> list[str]:
    """Convenience: retrieve with pre-computed embeddings."""
    results = await retrieve(query, chunks, chunk_embeddings, top_k=top_k)
    return [chunk for _, chunk in results]
