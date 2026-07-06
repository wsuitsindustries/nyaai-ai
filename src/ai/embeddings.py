"""Embedding interface — converts text to vector representations.

Uses FastEmbed (ONNX-based, no PyTorch) for real semantic embeddings
when available. Falls back to hash-based character n-gram embeddings.
"""

import importlib.util
import hashlib
import logging

logger = logging.getLogger(__name__)

_FASTEMBED_AVAILABLE = importlib.util.find_spec("fastembed") is not None

_model = None
_model_dim = 384 if _FASTEMBED_AVAILABLE else 256


def _load_model():
    global _model
    if _model is not None:
        return
    try:
        from fastembed import TextEmbedding
        _model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", max_length=512)
        logger.info("Loaded FastEmbed model (BAAI/bge-small-en-v1.5)")
    except Exception as e:
        logger.warning("Failed to load FastEmbed: %s — falling back to hash embeddings", e)
        global _FASTEMBED_AVAILABLE
        _FASTEMBED_AVAILABLE = False


def _hash_features(text: str, dim: int = 256) -> list[float]:
    vec = [0.0] * dim
    text = text.lower()
    for i in range(len(text) - 2):
        ngram = text[i : i + 3]
        h = int(hashlib.md5(ngram.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    magnitude = sum(v * v for v in vec) ** 0.5
    if magnitude > 0:
        vec = [v / magnitude for v in vec]
    return vec


_cache: dict[int, list[float]] = {}
_MAX_CACHE = 1000


def embed(text: str) -> list[float]:
    key = hash(text)
    if key in _cache:
        return _cache[key]
    if _FASTEMBED_AVAILABLE:
        _load_model()
        if _model is not None:
            emb = next(_model.embed([text]))
            vec = [float(v) for v in emb]
        else:
            vec = _hash_features(text)
    else:
        vec = _hash_features(text)
    if len(_cache) >= _MAX_CACHE:
        _cache.pop(next(iter(_cache)))
    _cache[key] = vec
    return vec


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(ai * bi for ai, bi in zip(a, b))
    ma = sum(ai * ai for ai in a) ** 0.5
    mb = sum(bi * bi for bi in b) ** 0.5
    if ma == 0 or mb == 0:
        return 0.0
    return dot / (ma * mb)


__all__ = ["embed", "cosine_similarity"]
