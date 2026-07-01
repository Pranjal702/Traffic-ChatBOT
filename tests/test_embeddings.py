import numpy as np
import src.embeddings as embeddings


class DummyModel:
    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        return np.array([[float(len(text)), 1.0] for text in texts])


def test_embed_texts_returns_same_number_of_vectors(monkeypatch):
    monkeypatch.setattr(embeddings, "SentenceTransformer", lambda model_name: DummyModel())

    texts = ["Hello world", "Test sentence"]
    vectors = embeddings.embed_texts(texts)

    assert len(vectors) == len(texts)
    assert all(len(vec) == 2 for vec in vectors)


def test_embed_texts_vectors_are_normalized(monkeypatch):
    class NormalizedModel:
        def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
            arr = np.array([[3.0, 4.0]])
            return arr / np.linalg.norm(arr, axis=1, keepdims=True)

    monkeypatch.setattr(embeddings, "SentenceTransformer", lambda model_name: NormalizedModel())

    embeddings_result = embeddings.embed_texts(["Hello world"])
    assert len(embeddings_result) == 1
    norm = np.linalg.norm(embeddings_result[0])
    assert abs(norm - 1.0) < 1e-6
