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

    filter_filename: Optional[str] = Field(default=None, description="Optional filename to restrict search scope")

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
            filter_filename=self.filter_filename,
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

    def retrieve_documents(self, query: str, filter_filename: Optional[str] = None) -> List[Document]:
        """Retrieve LangChain Documents for a given query, optionally filtered to a file."""
        if filter_filename:
            scoped_retriever = LangChainFAISSRetriever(
                vector_store=self.vector_store,
                embedding_service=self.embedding_service,
                top_k=self.top_k,
                filter_filename=filter_filename,
            )
            return scoped_retriever.invoke(query)
        return self.langchain_retriever.invoke(query)

    def retrieve(self, query: str, top_k: Optional[int] = None, filter_filename: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Backwards-compatible raw dictionary retrieval for UI components.
        """
        k = top_k or self.top_k
        if self.vector_store.index is None or self.vector_store.index.ntotal == 0:
            return []
        query_vector = self.embedding_service.embed_query(query)
        return self.vector_store.search(
            query_vector,
            top_k=k,
            filter_filename=filter_filename,
        )

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

    def build_rag_prompt(self, query: str, docs_or_chunks: List[Any]) -> str:
        """Construct prompt with context from retrieved documents."""
        if not docs_or_chunks:
            return f"Student Question: {query}\nAnswer as an expert study assistant."

        context_lines = []
        for d in docs_or_chunks:
            if isinstance(d, Document):
                src = d.metadata.get("source", "Unknown")
                pg = d.metadata.get("page", "?")
                txt = d.page_content.strip()
            else:
                src = d.get("filename", "Unknown")
                pg = d.get("page", "?")
                txt = d.get("text", "").strip()
            context_lines.append(f"[Document: {src} | Page: {pg}]\n{txt}")

        joined_context = "\n\n".join(context_lines)
        return (
            f"Context from Uploaded Notes:\n{joined_context}\n\n"
            f"Student Question: {query}\n"
            f"Provide a clear, accurate explanation grounded in the notes above."
        )

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
