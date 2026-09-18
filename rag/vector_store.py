"""
FAISS Vector Store and Metadata Persistence for SmartStudy AI.
Persists FAISS index, embeddings, and chunk metadata across Streamlit sessions.
Includes resilient pure-numpy fallback if FAISS binary is not available.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

# Safe FAISS import with pure-numpy cosine similarity fallback
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


from config.settings import (
    VECTOR_STORE_DIR,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    DOCUMENTS_DIR,
    TOP_K_RESULTS,
)


class VectorStoreError(Exception):
    """Base exception for Vector Store operations."""
    pass


class NumpyFlatIPIndex:
    """Pure-Numpy drop-in fallback for FAISS IndexFlatIP (Cosine Similarity)."""

    def __init__(self, dimension: int):
        self.d = dimension
        self.vectors = np.empty((0, dimension), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return len(self.vectors)

    def add(self, x: np.ndarray):
        if len(self.vectors) == 0:
            self.vectors = x.astype(np.float32)
        else:
            self.vectors = np.vstack([self.vectors, x.astype(np.float32)])

    def search(self, q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.vectors) == 0:
            return np.array([[]]), np.array([[]])
        scores = np.dot(self.vectors, q.T).flatten()
        top_k = min(k, len(scores))
        indices = np.argsort(scores)[::-1][:top_k]
        sorted_scores = scores[indices]
        return np.array([sorted_scores]), np.array([indices])


class VectorStore:
    """Manages vector index and chunk metadata with disk persistence."""

    def __init__(
        self,
        vector_dir: Path = VECTOR_STORE_DIR,
        dimension: int = 384,
    ):
        self.vector_dir = Path(vector_dir)
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.vector_dir / "index.faiss"
        self.metadata_path = self.vector_dir / "metadata.json"
        self.embeddings_path = self.vector_dir / "embeddings.npy"
        self.dimension = dimension

        self.index: Any = None
        self.chunks: List[Dict[str, Any]] = []
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Optional[np.ndarray] = None

        self._load_or_initialize()

    def _load_or_initialize(self):
        """Load persisted index and metadata from disk, or initialize fresh instances."""
        try:
            if self.metadata_path.exists():
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    self.chunks = meta.get("chunks", [])
                    self.documents = meta.get("documents", {})

                if self.embeddings_path.exists():
                    self.embeddings = np.load(str(self.embeddings_path))

                # Initialize index
                if HAS_FAISS and self.index_path.exists():
                    self.index = faiss.read_index(str(self.index_path))
                else:
                    self._create_empty_index()
                    if self.embeddings is not None and len(self.embeddings) > 0:
                        self.index.add(self.embeddings)

                # Verify integrity
                if self.index.ntotal != len(self.chunks):
                    self._create_empty_index()
            else:
                self._create_empty_index()
        except Exception:
            self._create_empty_index()

    def _create_empty_index(self):
        """Create a new empty Inner-Product FAISS or Numpy index."""
        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self.index = NumpyFlatIPIndex(self.dimension)
        self.chunks = []
        self.documents = {}
        self.embeddings = np.empty((0, self.dimension), dtype=np.float32)

    def _save_to_disk(self):
        """Persist vector index, metadata, and vectors to disk."""
        try:
            self.vector_dir.mkdir(parents=True, exist_ok=True)

            # 1. Save FAISS index if available
            if HAS_FAISS and hasattr(self.index, "ntotal"):
                try:
                    faiss.write_index(self.index, str(self.index_path))
                except Exception:
                    pass

            # 2. Save metadata JSON
            metadata_payload = {
                "chunks": self.chunks,
                "documents": self.documents,
                "total_chunks": len(self.chunks),
            }
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata_payload, f, indent=2, ensure_ascii=False)

            # 3. Save raw embeddings numpy array
            if self.embeddings is not None and len(self.embeddings) > 0:
                np.save(str(self.embeddings_path), self.embeddings)
            elif self.embeddings_path.exists():
                self.embeddings_path.unlink()

        except Exception as e:
            raise VectorStoreError(f"Failed to persist vector store: {str(e)}")

    def add_document(
        self,
        filename: str,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray,
        doc_stats: Dict[str, Any],
    ):
        """
        Add a document's chunks and vector embeddings to the index and persist.
        """
        if filename in self.documents:
            self.delete_document(filename, delete_pdf_file=False)

        if len(chunks) == 0 or embeddings.shape[0] == 0:
            return

        if embeddings.shape[1] != self.dimension:
            raise VectorStoreError(
                f"Embedding dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}"
            )

        vectors = embeddings.astype(np.float32)

        # Add to index
        self.index.add(vectors)

        # Update metadata
        self.chunks.extend(chunks)
        self.documents[filename] = doc_stats

        # Update cached embeddings array
        if self.embeddings is None or len(self.embeddings) == 0:
            self.embeddings = vectors
        else:
            self.embeddings = np.vstack([self.embeddings, vectors])

        # Persist changes
        self._save_to_disk()

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = TOP_K_RESULTS,
        min_score: float = 0.25,
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search using normalized inner product.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vector.astype(np.float32), k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if score < min_score:
                continue
            if 0 <= idx < len(self.chunks):
                chunk_copy = dict(self.chunks[idx])
                chunk_copy["score"] = float(score)
                results.append(chunk_copy)

        return results

    def delete_document(self, filename: str, delete_pdf_file: bool = True) -> bool:
        """
        Remove a document and its chunks from index and metadata.
        """
        if filename not in self.documents:
            return False

        keep_indices = []
        new_chunks = []
        for i, chunk in enumerate(self.chunks):
            if chunk.get("filename") != filename:
                keep_indices.append(i)
                new_chunks.append(chunk)

        del self.documents[filename]
        self.chunks = new_chunks

        # Rebuild index
        if keep_indices and self.embeddings is not None and len(self.embeddings) > 0:
            new_embeddings = self.embeddings[keep_indices]
            if HAS_FAISS:
                new_index = faiss.IndexFlatIP(self.dimension)
            else:
                new_index = NumpyFlatIPIndex(self.dimension)
            new_index.add(new_embeddings)
            self.index = new_index
            self.embeddings = new_embeddings
        else:
            self._create_empty_index()

        if delete_pdf_file:
            pdf_path = DOCUMENTS_DIR / filename
            if pdf_path.exists():
                try:
                    pdf_path.unlink()
                except Exception:
                    pass

        self._save_to_disk()
        return True

    def get_documents_summary(self) -> List[Dict[str, Any]]:
        """Return list of indexed document information."""
        return list(self.documents.values())

    def get_stats(self) -> Dict[str, Any]:
        """Return overall vector store statistics."""
        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "faiss_total_vectors": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "engine": "FAISS Flat-IP" if HAS_FAISS else "Numpy Flat-IP",
        }

    def clear_all(self):
        """Reset index and delete all documents."""
        self._create_empty_index()
        self._save_to_disk()

        if DOCUMENTS_DIR.exists():
            for f in DOCUMENTS_DIR.glob("*.pdf"):
                try:
                    f.unlink()
                except Exception:
                    pass
