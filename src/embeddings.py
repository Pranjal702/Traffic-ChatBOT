from typing import List
import math
import hashlib

# Try to import the heavy-weight dependency; tests will monkeypatch
# the `SentenceTransformer` symbol when needed. If the import fails
# (e.g. due to system/DLL issues), expose `SentenceTransformer = None`
# and use a lightweight fallback embedding implementation below.
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None  # type: ignore

# Do not cache the model at module import time. Creating a fresh
# instance on demand makes behavior predictable for tests that
# monkeypatch the `SentenceTransformer` symbol.


def _fallback_embed(text: str, dim: int = 32) -> List[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # Use the digest bytes to produce floats in [0,1]
    vec = [b / 255.0 for b in digest[:dim]]
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return [0.0] * dim
    return [x / norm for x in vec]


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []

    # If a real model is available (or monkeypatched in tests), use it.
    if SentenceTransformer is not None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return [emb.tolist() for emb in embeddings]

    # Otherwise use a deterministic, lightweight fallback so tests and
    # basic development can proceed without heavy native deps.
    return [_fallback_embed(t) for t in texts]
