"""
Embedding service for SmartStudy AI using Sentence Transformers.
Encodes text chunks and student queries into dense vector representations.
"""

from typing import List, Optional
import numpy as np

from config.settings import DEFAULT_EMBEDDING_MODEL


class EmbeddingError(Exception):
    """Raised when embeddings fail to generate."""
    pass


class EmbeddingService:
    """Manages SentenceTransformer model loading and vector encoding."""

    _instance: Optional["EmbeddingService"] = None
    _model = None

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self._ensure_model_loaded()

    @classmethod
    def get_instance(cls, model_name: str = DEFAULT_EMBEDDING_MODEL) -> "EmbeddingService":
        """Get or initialize singleton instance."""
        if cls._instance is None or cls._instance.model_name != model_name:
            cls._instance = cls(model_name=model_name)
        return cls._instance

    def _ensure_model_loaded(self):
        """Lazy load SentenceTransformer model."""
        if EmbeddingService._model is None or getattr(self, "_loaded_name", None) != self.model_name:
            try:
                from sentence_transformers import SentenceTransformer
                EmbeddingService._model = SentenceTransformer(self.model_name)
                self._loaded_name = self.model_name
            except Exception as e:
                raise EmbeddingError(
                    f"Failed to load embedding model '{self.model_name}'. "
                    f"Ensure 'sentence-transformers' is installed and internet is accessible for first-time download. "
                    f"Error: {str(e)}"
                )

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Compute dense vector embeddings for a list of text strings.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim) as float32.
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        self._ensure_model_loaded()
        try:
            # normalize_embeddings=True allows cosine similarity via inner product (dot product)
            embeddings = EmbeddingService._model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return embeddings.astype(np.float32)
        except Exception as e:
            raise EmbeddingError(f"Error computing embeddings: {str(e)}")

    def embed_query(self, query: str) -> np.ndarray:
        """
        Compute dense vector embedding for a single query string.

        Returns:
            np.ndarray of shape (1, embedding_dim) as float32.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        self._ensure_model_loaded()
        try:
            embedding = EmbeddingService._model.encode(
                [query.strip()],
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return embedding.astype(np.float32)
        except Exception as e:
            raise EmbeddingError(f"Error computing query embedding: {str(e)}")
