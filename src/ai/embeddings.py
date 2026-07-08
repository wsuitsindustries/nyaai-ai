"""Embedding interface — converts text to vector representations.

Uses FastEmbed (ONNX-based, no PyTorch) for real semantic embeddings
with a lightweight model (gte-tiny, ~22MB). Falls back to hash-based
character n-gram embeddings. Supports Redis cluster cache when available.
"""

import importlib.util
import hashlib
import logging
import os

logger = logging.getLogger(__name__)

_FASTEMBED_AVAILABLE = importlib.util.find_spec("fastembed") is not None
_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

_USE_REDIS = os.getenv("REDIS_URL") is not None
_redis = None

_model = None
_model_dim = _EMBEDDING_DIM if _FASTEMBED_AVAILABLE else 256


def _get_redis():
    global _redis
    if _redis is None and _USE_REDIS:
        try:
            import redis.asyncio as aioredis
            _redis = aioredis.from_url(
                os.environ["REDIS_URL"],
                decode_responses=False,
                socket_connect_timeout=2,
            )
        except Exception:
            pass
    return _redis


def _load_model():
    global _model
    if _model is not None:
        return
    try:
        from fastembed import TextEmbedding
        _model = TextEmbedding(model_name=_EMBEDDING_MODEL, max_length=512)
        logger.info("Loaded FastEmbed model (%s)", _EMBEDDING_MODEL)
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


async def embed(text: str) -> list[float]:
    key = hash(text)

    # Try Redis cluster cache first
    r = _get_redis()
    if r is not None:
        try:
            cached = await r.get(f"emb:{key}")
            if cached is not None:
                import struct
                n = len(cached) // 4
                return [struct.unpack("f", cached[i:i+4])[0] for i in range(0, len(cached), 4)]
        except Exception:
            pass

    # Local in-memory cache
    if key in _cache:
        return _cache[key]

    # Compute embedding
    if _FASTEMBED_AVAILABLE:
        _load_model()
        if _model is not None:
            emb = next(_model.embed([text]))
            vec = [float(v) for v in emb]
        else:
            vec = _hash_features(text)
    else:
        vec = _hash_features(text)

    # Store in local cache
    if len(_cache) >= _MAX_CACHE:
        _cache.pop(next(iter(_cache)))
    _cache[key] = vec

    # Store in Redis cluster cache
    if r is not None:
        try:
            import struct
            packed = struct.pack(f"{len(vec)}f", *vec)
            await r.setex(f"emb:{key}", 86400, packed)
        except Exception:
            pass

    return vec


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts in a single batch (more efficient with FastEmbed)."""
    if _FASTEMBED_AVAILABLE:
        _load_model()
        if _model is not None:
            try:
                results = []
                for emb in _model.embed(texts):
                    results.append([float(v) for v in emb])
                return results
            except Exception:
                pass
    return [await embed(t) for t in texts]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(ai * bi for ai, bi in zip(a, b))
    ma = sum(ai * ai for ai in a) ** 0.5
    mb = sum(bi * bi for bi in b) ** 0.5
    if ma == 0 or mb == 0:
        return 0.0
    return dot / (ma * mb)


__all__ = ["embed", "embed_batch", "cosine_similarity"]
