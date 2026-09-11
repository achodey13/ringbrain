from typing import Protocol


class Embedder(Protocol):
    """Anything that turns text into a fixed-size vector.

    Kept as a Protocol (not a base class) so tests can inject a fast,
    deterministic fake instead of loading a real transformer model.
    """

    def embed(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Default embedder: sentence-transformers/all-MiniLM-L6-v2, 384 dims, CPU-friendly."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()
