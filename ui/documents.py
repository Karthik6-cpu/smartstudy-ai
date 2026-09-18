"""
Documents Page for SmartStudy AI.
Enables PDF uploading, processing, status tracking, document deletion,
and vector database management.
"""

import time
from pathlib import Path
import streamlit as st

from rag.pdf_processor import (
    PDFProcessor,
    CorruptedPDFError,
    EmptyPDFError,
    NoExtractableTextError,
)
from rag.embeddings import EmbeddingService, EmbeddingError
from rag.vector_store import VectorStore, VectorStoreError
from config.settings import DOCUMENTS_DIR


def render_documents_page():
    """Render the Document Ingestion and Management UI."""
    st.markdown("## 📚 Document Library & MCA Notes")
    st.caption("Upload course PDFs, lecture slides, and question banks to power local RAG search.")

    # Initialize components
    pdf_processor = PDFProcessor()
    vector_store = VectorStore()

    # Metrics Bar
    stats = vector_store.get_stats()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Indexed Documents", stats["total_documents"])
    with col2:
        st.metric("Total Chunks", stats["total_chunks"])
    with col3:
        st.metric("Vector Dimensions", stats["dimension"])
    with col4:
        st.metric("Index Type", "FAISS Flat-IP")

    st.markdown("---")

    # Upload Section
    st.subheader("📤 Upload New Documents")
    uploaded_files = st.file_uploader(
        "Select one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload PDF notes, syllabus copies, or textbooks.",
    )

    if uploaded_files:
        if st.button("🚀 Process & Index Uploaded Documents", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.info("Loading embedding model (`all-MiniLM-L6-v2`)...")
                embedding_service = EmbeddingService.get_instance()
            except EmbeddingError as ee:
                st.error(f"❌ Failed to load embedding service: {str(ee)}")
                return

            total_files = len(uploaded_files)
            success_count = 0

            for idx, uploaded_file in enumerate(uploaded_files):
                fname = uploaded_file.name
                status_text.text(f"[{idx+1}/{total_files}] Processing '{fname}'...")

                file_bytes = uploaded_file.read()

                try:
                    # 1. Save locally to storage/documents
                    saved_path = pdf_processor.save_pdf(file_bytes, fname)

                    # 2. Extract and chunk text
                    status_text.text(f"[{idx+1}/{total_files}] Extracting text from '{fname}'...")
                    chunks, doc_stats = pdf_processor.process_pdf(file_bytes, fname)
                    doc_stats["file_size_kb"] = round(len(file_bytes) / 1024, 1)

                    # 3. Generate embeddings
                    status_text.text(f"[{idx+1}/{total_files}] Generating embeddings for {len(chunks)} chunks...")
                    chunk_texts = [c["text"] for c in chunks]
                    embeddings = embedding_service.embed_texts(chunk_texts)

                    # 4. Add to FAISS and persist
                    status_text.text(f"[{idx+1}/{total_files}] Storing in FAISS vector store...")
                    vector_store.add_document(fname, chunks, embeddings, doc_stats)

                    success_count += 1

                except EmptyPDFError as e:
                    st.error(f"⚠️ **{fname}**: {str(e)}")
                except CorruptedPDFError as e:
                    st.error(f"❌ **{fname}**: {str(e)}")
                except NoExtractableTextError as e:
                    st.warning(f"⚠️ **{fname}**: {str(e)}")
                except VectorStoreError as e:
                    st.error(f"❌ **{fname}**: Vector store error - {str(e)}")
                except Exception as e:
                    st.error(f"❌ **{fname}**: Unexpected error: {str(e)}")

                progress_bar.progress((idx + 1) / total_files)

            status_text.empty()
            progress_bar.empty()

            if success_count > 0:
                st.success(f"✅ Successfully processed and indexed {success_count} document(s)!")
                time.sleep(0.5)
                st.rerun()

    st.markdown("---")

    # Document Inventory Section
    st.subheader("📑 Indexed Study Documents")
    indexed_docs = vector_store.get_documents_summary()

    if not indexed_docs:
        st.info("No documents indexed yet. Upload a PDF above to start searching your notes!")
    else:
        for doc in indexed_docs:
            fname = doc.get("filename", "Unknown")
            pages = doc.get("total_pages", 0)
            chunks = doc.get("total_chunks", 0)
            chars = doc.get("total_characters", 0)
            size_kb = doc.get("file_size_kb", "N/A")

            card_col1, card_col2, card_col3, card_col4 = st.columns([4, 2, 2, 2])

            with card_col1:
                st.markdown(f"📄 **{fname}**")
                st.caption(f"Size: {size_kb} KB · Characters: {chars:,}")

            with card_col2:
                st.markdown(f"**{pages}** page(s)")

            with card_col3:
                st.markdown(f"**{chunks}** chunks indexed")

            with card_col4:
                delete_key = f"del_{fname}"
                if st.button("🗑️ Delete", key=delete_key, help=f"Remove {fname} and its vectors"):
                    vector_store.delete_document(fname)
                    st.success(f"Deleted '{fname}'")
                    time.sleep(0.3)
                    st.rerun()

            st.markdown("<hr style='margin: 4px 0;'/>", unsafe_allow_html=True)

        col_clear1, col_clear2 = st.columns([4, 1])
        with col_clear2:
            if st.button("⚠️ Clear All Documents", use_container_width=True, type="secondary"):
                vector_store.clear_all()
                st.warning("Cleared all documents and vector index.")
                time.sleep(0.5)
                st.rerun()
