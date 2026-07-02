"""Embedding interface — converts text to vector representations.

Currently uses a simple character-n-gram hashing approach as a
zero-dependency placeholder. Swap in sentence-transformers, OpenAI,
or any embedding model by implementing `embed(text) -> list[float]`.
"""

import hashlib

EMBEDDING_DIM = 256


def _hash_features(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Min-hash style embedding: hash each n-gram to a dimension."""
    vec = [0.0] * dim
    text = text.lower()
    for i in range(len(text) - 2):
        ngram = text[i : i + 3]
        h = int(hashlib.md5(ngram.encode()).hexdigest(), 16)
        idx = h % dim
        vec[idx] += 1.0
    # Normalize
    magnitude = sum(v * v for v in vec) ** 0.5
    if magnitude > 0:
        vec = [v / magnitude for v in vec]
    return vec


def embed(text: str) -> list[float]:
    """Return a vector embedding for *text*."""
    return _hash_features(text)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(ai * bi for ai, bi in zip(a, b))
    ma = sum(ai * ai for ai in a) ** 0.5
    mb = sum(bi * bi for bi in b) ** 0.5
    if ma == 0 or mb == 0:
        return 0.0
    return dot / (ma * mb)
