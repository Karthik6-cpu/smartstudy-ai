"""
LangChain-Compatible Retriever and RAG Orchestrator for SmartStudy AI.
Wraps the persistent FAISS vector store with langchain_core.retrievers.BaseRetriever
and provides source citation utilities.
"""

from typing import List, Dict, Any, Optional
from pydantic import Field
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from rag.embeddings import EmbeddingService
from rag.vector_store import VectorStore
from config.settings import TOP_K_RESULTS


class LangChainFAISSRetriever(BaseRetriever):
    """
    Modern LangChain-compatible retriever wrapping local FAISS vector store.
    Inherits from langchain_core.retrievers.BaseRetriever to integrate seamlessly
    with LCEL (LangChain Expression Language) pipelines.
    """

    vector_store: Any = Field(description="SmartStudy FAISS VectorStore instance")
    embedding_service: Any = Field(description="SentenceTransformers EmbeddingService instance")
    top_k: int = Field(default=TOP_K_RESULTS, description="Top K relevant chunks to fetch")
    min_score: float = Field(default=0.20, description="Cosine similarity score threshold")

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Convert query into embedding, search FAISS index, and return LangChain Documents.
        """
        if self.vector_store.index is None or self.vector_store.index.ntotal == 0:
            return []

        # 1. Embed query
        query_vector = self.embedding_service.embed_query(query)

        # 2. Search FAISS index
        raw_results = self.vector_store.search(
            query_vector,
            top_k=self.top_k,
            min_score=self.min_score,
        )

        # 3. Convert to LangChain Document instances with full metadata
        documents = []
        for chunk in raw_results:
            doc = Document(
                page_content=chunk.get("text", ""),
                metadata={
                    "source": chunk.get("filename", "Unknown"),
                    "filename": chunk.get("filename", "Unknown"),
                    "page": chunk.get("page", 1),
                    "chunk_index": chunk.get("chunk_index", 1),
                    "score": round(chunk.get("score", 0.0), 3),
                }
            )
            documents.append(doc)

        return documents


class RAGRetriever:
    """
    High-level retriever coordinator providing LangChain retriever access,
    context formatting, and UI citation extraction.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStore] = None,
        top_k: int = TOP_K_RESULTS,
    ):
        self.embedding_service = embedding_service or EmbeddingService.get_instance()
        self.vector_store = vector_store or VectorStore()
        self.top_k = top_k

        # Initialize underlying LangChain BaseRetriever
        self.langchain_retriever = LangChainFAISSRetriever(
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
            top_k=self.top_k,
        )

    def retrieve_documents(self, query: str) -> List[Document]:
        """Retrieve LangChain Documents for a given query."""
        return self.langchain_retriever.invoke(query)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Backwards-compatible raw dictionary retrieval for UI components.
        """
        k = top_k or self.top_k
        docs = self.langchain_retriever.invoke(query)
        results = []
        for doc in docs[:k]:
            results.append({
                "filename": doc.metadata.get("filename", doc.metadata.get("source", "Unknown")),
                "page": doc.metadata.get("page", 1),
                "chunk_index": doc.metadata.get("chunk_index", 1),
                "text": doc.page_content,
                "score": doc.metadata.get("score", 0.0),
            })
        return results

    def format_docs_for_context(self, docs: List[Document]) -> str:
        """Format a list of LangChain Documents into a clean RAG prompt context block."""
        if not docs:
            return "No relevant context found in uploaded documents."

        context_parts = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            score = doc.metadata.get("score", "")
            header = f"[Excerpt {i} | Document: {source} | Page: {page}]"
            context_parts.append(f"{header}\n{doc.page_content.strip()}")

        return "\n\n".join(context_parts)

    def format_citations(self, docs_or_chunks: List[Any]) -> List[Dict[str, Any]]:
        """Extract citations for Streamlit UI display."""
        citations = []
        for item in docs_or_chunks:
            if isinstance(item, Document):
                fname = item.metadata.get("source", item.metadata.get("filename", "Unknown"))
                page = item.metadata.get("page", 1)
                chunk_idx = item.metadata.get("chunk_index", 1)
                score = item.metadata.get("score", 0.0)
                text = item.page_content
            else:
                fname = item.get("filename", "Unknown")
                page = item.get("page", 1)
                chunk_idx = item.get("chunk_index", 1)
                score = item.get("score", 0.0)
                text = item.get("text", "")

            snippet = text[:250] + ("..." if len(text) > 250 else "")
            citations.append({
                "filename": fname,
                "page": page,
                "chunk_index": chunk_idx,
                "score": score,
                "snippet": snippet,
            })
        return citations
