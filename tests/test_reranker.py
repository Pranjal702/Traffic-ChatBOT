import numpy as np
import src.embedding_reranker as reranker
import src.embeddings as embeddings


def test_reranker_returns_top_k_sorted(monkeypatch):
    # 3 passages, predictable embeddings
    def fake_embed(texts):
        # passages: v0=[1,0], v1=[0,1], v2=[0.5,0.5], query=[1,0]
        mapping = {
            "p0": [1.0, 0.0],
            "p1": [0.0, 1.0],
            "p2": [0.5, 0.5],
            "q": [1.0, 0.0],
        }
        return [mapping[t] for t in texts]

    monkeypatch.setattr(embeddings, "embed_texts", fake_embed)

    passages = [
        {"id": "p0", "text": "p0"},
        {"id": "p1", "text": "p1"},
        {"id": "p2", "text": "p2"},
    ]

    res = reranker.rerank_by_embedding(passages, "q", top_k=2)
    assert len(res) == 2
    assert res[0]["id"] == "p0"
    assert res[1]["id"] == "p2"
    assert res[0]["score"] > res[1]["score"]


def test_reranker_prefers_keyword_overlap_without_embeddings(monkeypatch):
    monkeypatch.setattr(embeddings, "embed_texts", lambda texts: [[0.0, 0.0] for _ in texts])

    passages = [
        {"id": "p1", "text": "The refund policy is described in section 2.", "score": 0.1},
        {"id": "p2", "text": "The weather today is sunny and warm.", "score": 0.9},
    ]

    res = reranker.rerank_by_embedding(passages, "refund policy", top_k=2)
    assert res[0]["id"] == "p1"


def test_context_guard_rejects_irrelevant_results():
    assert not reranker.is_relevant_context("refund policy", [{"score": 0.01, "text": "The weather is sunny today."}])


def test_context_guard_accepts_relevant_results():
    assert reranker.is_relevant_context("refund policy", [{"score": 0.12, "text": "Refund policy details"}])


def test_reranker_handles_empty_passages():
    assert reranker.rerank_by_embedding([], "q") == []
