import re
from typing import List, Dict
import numpy as np

import src.embeddings as embeddings


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _lexical_overlap(query: str, text: str) -> float:
    query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    if not query_tokens:
        return 0.0
    text_tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    if not text_tokens:
        return 0.0
    overlap = len(query_tokens.intersection(text_tokens)) / max(len(query_tokens), 1)
    return float(overlap)


def is_relevant_context(query: str, passages: List[Dict], min_score: float = 0.1, min_overlap: float = 0.05) -> bool:
    """Return True when the top passage looks meaningfully related to the query."""
    if not passages:
        return False

    top_passage = passages[0]
    top_score = float(top_passage.get("score", 0.0) or 0.0)
    overlap = _lexical_overlap(query, top_passage.get("text", ""))
    return top_score >= min_score or overlap >= min_overlap


def rerank_by_embedding(passages: List[Dict], query: str, top_k: int = 5) -> List[Dict]:
    """Rerank passages using a hybrid score: retrieval score + lexical overlap + embedding similarity.

    passages: list of dicts with at least `id` and `text` keys.
    Returns the top_k passages (descending score) with an added `score` key.
    """
    if not passages:
        return []

    texts = [p["text"] for p in passages]
    vectors = embeddings.embed_texts(texts + [query])
    passage_vecs = np.array(vectors[:-1])
    query_vec = np.array(vectors[-1])

    embedding_scores = [_cosine_similarity(query_vec, pv) for pv in passage_vecs]

    scored = []
    for passage, emb_score in zip(passages, embedding_scores):
        retrieval_score = float(passage.get("score", 0.0) or 0.0)
        lexical_score = _lexical_overlap(query, passage.get("text", ""))

        # Use the original retrieval score as a useful signal but let strong keyword overlap
        # and embedding similarity lift the most relevant passages.
        combined_score = (0.35 * retrieval_score) + (0.25 * emb_score) + (0.40 * lexical_score)
        combined_score = max(combined_score, lexical_score)

        entry = dict(passage)
        entry["score"] = float(combined_score)
        scored.append(entry)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
