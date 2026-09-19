"""
Documents Page for SmartStudy AI.
Enables reliable PDF uploading, SHA-256 duplicate detection, document library with actions
(Ask, Summarize, Quiz, Flashcards, Search, Delete), hierarchical summarization,
and semantic search across notes.
"""

import time
import io
from pathlib import Path
import streamlit as st

from rag.pdf_processor import (
    PDFProcessor,
    compute_sha256,
    CorruptedPDFError,
    PasswordProtectedPDFError,
    EmptyPDFError,
    NoExtractableTextError,
)
from rag.embeddings import EmbeddingService, EmbeddingError
from rag.vector_store import VectorStore, VectorStoreError
from chains.summarizer import generate_document_summary
from memory.storage_manager import (
    register_document,
    get_registered_documents,
    delete_registered_document,
    save_flashcard,
)
from config.settings import DEFAULT_MODEL, DEFAULT_OLLAMA_HOST


def render_documents_page():
    """Render the Document Library, Upload Pipeline, Summarizer, and Semantic Search."""
    st.markdown("## 📚 Document Management & Study Library")
    st.caption("Manage course PDFs, verify SHA-256 duplicate integrity, summarize notes, and search across course materials.")

    pdf_processor = PDFProcessor()
    vector_store = VectorStore()
    stats = vector_store.get_stats()

    # Metric Bar
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Indexed Documents", stats["total_documents"])
    with col_m2:
        st.metric("Total Chunks", stats["total_chunks"])
    with col_m3:
        st.metric("Vector Dimensions", stats["dimension"])
    with col_m4:
        st.metric("Storage Engine", stats.get("engine", "FAISS Flat-IP"))

    st.markdown("---")

    tab_library, tab_upload, tab_summarize, tab_search = st.tabs([
        "📖 Document Library",
        "📤 Upload Notes",
        "📄 Summarize Document",
        "🔎 Search Notes",
    ])

    # =========================================================================
    # TAB 1: DOCUMENT LIBRARY & ACTIONS
    # =========================================================================
    with tab_library:
        indexed_docs = vector_store.get_documents_summary()

        if not indexed_docs:
            st.info("No documents indexed yet. Switch to the **'📤 Upload Notes'** tab to add your first lecture PDF!")
        else:
            st.markdown(f"### 📋 Indexed Documents ({len(indexed_docs)})")

            for doc in indexed_docs:
                fname = doc.get("filename", "Unknown")
                pages = doc.get("total_pages", 0)
                chunks = doc.get("total_chunks", 0)
                chars = doc.get("total_characters", 0)
                size_kb = doc.get("file_size_kb", "N/A")
                upload_date = doc.get("upload_date", "Recently")
                file_hash = doc.get("file_hash", "N/A")
                status = doc.get("status", "Ready")

                with st.container():
                    st.markdown(f"#### 📄 `{fname}`")
                    c_info1, c_info2, c_info3, c_info4 = st.columns([2, 2, 2, 2])
                    with c_info1:
                        st.caption(f"**Pages:** {pages} | **Chunks:** {chunks}")
                    with c_info2:
                        st.caption(f"**Size:** {size_kb} KB | **Status:** ✅ {status}")
                    with c_info3:
                        st.caption(f"**Uploaded:** {upload_date}")
                    with c_info4:
                        short_hash = file_hash[:10] + "..." if len(file_hash) > 10 else file_hash
                        st.caption(f"**SHA-256:** `{short_hash}`")

                    # Document Actions: Ask, Summarize, Quiz, Flashcards, Search, Delete
                    btn_col1, btn_col2, btn_col3, btn_col4, btn_col5, btn_col6 = st.columns(6)

                    with btn_col1:
                        if st.button("💬 Ask", key=f"ask_{fname}", use_container_width=True, help=f"Ask questions about {fname}"):
                            st.session_state.active_document = fname
                            st.session_state.chat_query_preset = f"According to {fname}, what are the core concepts?"
                            st.info(f"Context set to '{fname}'. Switch to 💬 AI Chat to converse.")

                    with btn_col2:
                        if st.button("📄 Summarize", key=f"sum_{fname}", use_container_width=True, help=f"Generate structured summary of {fname}"):
                            st.session_state.summarize_target_doc = fname
                            st.session_state.active_document = fname
                            st.info(f"Selected '{fname}' for summarization. Switch to the 📄 Summarize tab above.")

                    with btn_col3:
                        if st.button("📝 Quiz", key=f"quiz_{fname}", use_container_width=True, help=f"Generate practice quiz from {fname}"):
                            st.session_state.quiz_target_doc = fname
                            st.session_state.active_document = fname
                            st.info(f"Selected '{fname}' for quiz generation. Switch to 📝 Quiz page.")

                    with btn_col4:
                        if st.button("🃏 Flashcards", key=f"flash_{fname}", use_container_width=True, help=f"Create flashcards from {fname}"):
                            # Automatically generate 3-5 high-yield flashcards from document chunks
                            doc_chunks = vector_store.get_document_chunks(fname)
                            created = 0
                            deck_name = fname.replace(".pdf", "")
                            for c in doc_chunks[:5]:
                                txt = c.get("text", "")
                                lines = [ln.strip() for ln in txt.split("\n") if len(ln.strip()) > 20]
                                if len(lines) >= 2:
                                    front = f"What is explained in {fname} regarding: {lines[0][:80]}?"
                                    back = lines[1][:250]
                                    save_flashcard(deck_name=deck_name, front=front, back=back, source_doc=fname)
                                    created += 1
                            if created > 0:
                                st.success(f"Created {created} flashcards in deck '{deck_name}'! View them in Flashcards.")
                            else:
                                st.warning("Not enough text lines in document to auto-generate cards.")

                    with btn_col5:
                        if st.button("🔎 Search", key=f"search_{fname}", use_container_width=True, help=f"Search inside {fname}"):
                            st.session_state.search_filter_doc = fname
                            st.info(f"Filter set to '{fname}'. Switch to 🔎 Search Notes tab.")

                    with btn_col6:
                        if st.button("🗑️ Delete", key=f"del_{fname}", type="secondary", use_container_width=True, help=f"Completely remove {fname}"):
                            vector_store.delete_document(fname)
                            delete_registered_document(fname)
                            st.success(f"Removed '{fname}' and deleted all associated vectors.")
                            time.sleep(0.4)
                            st.rerun()

                    st.markdown("<hr style='margin: 12px 0;'/>", unsafe_allow_html=True)

            col_c1, col_c2 = st.columns([4, 1])
            with col_c2:
                if st.button("⚠️ Clear All Documents", use_container_width=True, type="secondary"):
                    vector_store.clear_all()
                    st.warning("Cleared all documents, FAISS vectors, and metadata.")
                    time.sleep(0.5)
                    st.rerun()

    # =========================================================================
    # TAB 2: UPLOAD NOTES WITH SHA-256 DUPLICATE DETECTION
    # =========================================================================
    with tab_upload:
        st.subheader("📤 Upload Course Notes (PDF)")
        st.caption("Upload lecture slides, textbooks, or syllabus documents. All processing is 100% local.")

        uploaded_files = st.file_uploader(
            "Choose PDF file(s) to process",
            type=["pdf"],
            accept_multiple_files=True,
            help="Select one or multiple PDF documents.",
        )

        if uploaded_files:
            st.write(f"Selected **{len(uploaded_files)}** file(s). Click below to validate and index.")

            if st.button("🚀 Process & Index Documents", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_box = st.empty()

                try:
                    status_box.info("Initializing SentenceTransformer embedding service (`all-MiniLM-L6-v2`)...")
                    embedding_service = EmbeddingService.get_instance()
                except EmbeddingError as ee:
                    st.error(f"❌ Failed to load embedding service: {str(ee)}")
                    return

                total_files = len(uploaded_files)
                processed_count = 0
                duplicate_count = 0

                for idx, uploaded_file in enumerate(uploaded_files):
                    fname = uploaded_file.name
                    status_box.text(f"[{idx+1}/{total_files}] Validating '{fname}'...")

                    file_bytes = uploaded_file.read()
                    file_hash = compute_sha256(file_bytes)

                    # 1. Duplicate Detection Check via SHA-256
                    existing_match = vector_store.has_file_hash(file_hash)
                    if existing_match:
                        st.warning(f"⚠️ **{fname}**: This document is already available (matches `{existing_match}`). Skipping duplicate indexing.")
                        duplicate_count += 1
                        progress_bar.progress((idx + 1) / total_files)
                        continue

                    try:
                        # 2. Save locally
                        saved_path = pdf_processor.save_pdf(file_bytes, fname)

                        # 3. Extract and chunk text
                        status_box.text(f"[{idx+1}/{total_files}] Extracting text from '{fname}'...")
                        chunks, doc_stats = pdf_processor.process_pdf(file_bytes, fname)
                        doc_stats["file_hash"] = file_hash

                        # 4. Compute embeddings
                        status_box.text(f"[{idx+1}/{total_files}] Computing embeddings for {len(chunks)} chunks...")
                        chunk_texts = [c["text"] for c in chunks]
                        embeddings = embedding_service.embed_texts(chunk_texts)

                        # 5. Store in FAISS & update metadata
                        status_box.text(f"[{idx+1}/{total_files}] Storing in FAISS index...")
                        vector_store.add_document(fname, chunks, embeddings, doc_stats)

                        # 6. Register in SQLite registry
                        register_document(
                            filename=fname,
                            file_hash=file_hash,
                            file_size_kb=doc_stats.get("file_size_kb", 0.0),
                            page_count=doc_stats.get("total_pages", 0),
                            chunk_count=len(chunks),
                            status="Ready",
                        )

                        processed_count += 1
                        st.success(f"✅ **{fname}**: Successfully indexed {len(chunks)} chunks across {doc_stats.get('total_pages', 0)} pages!")

                    except PasswordProtectedPDFError as pe:
                        st.error(f"❌ **{fname}**: Password protected. Please upload an unlocked PDF.")
                    except EmptyPDFError as ee:
                        st.error(f"⚠️ **{fname}**: File is empty or has 0 pages.")
                    except CorruptedPDFError as ce:
                        st.error(f"❌ **{fname}**: PDF file is corrupted or unreadable ({str(ce)}).")
                    except NoExtractableTextError as ne:
                        st.warning(f"⚠️ **{fname}**: {str(ne)}")
                    except VectorStoreError as ve:
                        st.error(f"❌ **{fname}**: Vector storage error: {str(ve)}")
                    except Exception as e:
                        st.error(f"❌ **{fname}**: Unexpected processing failure: {str(e)}")

                    progress_bar.progress((idx + 1) / total_files)

                status_box.empty()
                progress_bar.empty()

                if processed_count > 0:
                    st.success(f"🎉 Processing complete! Added {processed_count} new document(s) to your study library.")
                    time.sleep(0.5)
                    st.rerun()
                elif duplicate_count > 0 and processed_count == 0:
                    st.info("No new documents to process. All uploaded files were already indexed.")

    # =========================================================================
    # TAB 3: DOCUMENT SUMMARIZER (MAP-REDUCE)
    # =========================================================================
    with tab_summarize:
        st.subheader("📄 Grounded Document Summarizer")
        st.caption("Generate hierarchical, grounded summaries of uploaded notes without context limits.")

        doc_keys = list(vector_store.documents.keys())
        if not doc_keys:
            st.info("Please upload a PDF in the Upload tab first.")
        else:
            default_target = st.session_state.get("summarize_target_doc", doc_keys[0])
            sel_idx = doc_keys.index(default_target) if default_target in doc_keys else 0

            sc1, sc2 = st.columns([3, 2])
            with sc1:
                target_doc = st.selectbox("Select Document to Summarize:", options=doc_keys, index=sel_idx)
                summary_mode = st.selectbox(
                    "Summary Mode:",
                    ["Exam Notes", "Quick Summary", "Detailed Summary", "Revision Summary", "Chapter/Section Summary"],
                    help="Choose the structure and detail level of your summary.",
                )

            with sc2:
                scope = st.radio("Summarize Scope:", ["Entire Document", "Specific Pages", "Specific Topic"], horizontal=True)

                page_range = None
                topic_filter = None

                if scope == "Specific Pages":
                    doc_meta = vector_store.documents.get(target_doc, {})
                    max_p = doc_meta.get("total_pages", 10)
                    p_col1, p_col2 = st.columns(2)
                    with p_col1:
                        start_page = st.number_input("Start Page:", min_value=1, max_value=max_p, value=1)
                    with p_col2:
                        end_page = st.number_input("End Page:", min_value=1, max_value=max_p, value=min(max_p, 5))
                    page_range = (int(start_page), int(end_page))
                elif scope == "Specific Topic":
                    topic_filter = st.text_input("Enter Topic (e.g. 'CMMI', 'Waterfall Model'):", "CMMI")

            active_model = st.session_state.get("selected_model", DEFAULT_MODEL)
            active_host = st.session_state.get("ollama_host", DEFAULT_OLLAMA_HOST)

            if st.button("✨ Generate Grounded Summary", type="primary", use_container_width=True):
                with st.spinner(f"Analyzing '{target_doc}' with {active_model} (hierarchical map-reduce)..."):
                    result = generate_document_summary(
                        filename=target_doc,
                        mode=summary_mode,
                        page_range=page_range,
                        topic=topic_filter,
                        model=active_model,
                        base_url=active_host,
                    )
                    st.session_state.current_summary = result
                    st.session_state.current_summary_doc = target_doc

            # Display generated summary with actions
            if "current_summary" in st.session_state:
                res = st.session_state.current_summary
                summary_text = res.get("summary", "")
                word_count = res.get("word_count", 0)

                st.markdown("---")
                st.markdown(f"### 📑 Summary Results ({word_count} words)")
                st.markdown(summary_text)

                # Export & Next-Action Buttons
                act1, act2, act3, act4, act5 = st.columns(5)
                with act1:
                    st.download_button(
                        "💾 Download TXT",
                        data=summary_text,
                        file_name=f"{target_doc}_summary.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
                with act2:
                    st.download_button(
                        "💾 Download Markdown",
                        data=summary_text,
                        file_name=f"{target_doc}_summary.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )
                with act3:
                    if st.button("📝 Generate Quiz", use_container_width=True, help="Create a quiz from this summary"):
                        st.session_state.quiz_target_doc = target_doc
                        st.info(f"Pre-selected '{target_doc}' for quiz. Switch to 📝 Quiz page.")
                with act4:
                    if st.button("🃏 Flashcards", use_container_width=True, help="Create flashcards from this summary"):
                        # Extract 3 flashcard definitions from summary
                        deck_name = target_doc.replace(".pdf", "")
                        lines = [line.strip() for line in summary_text.split("\n") if line.strip().startswith("- **") or line.strip().startswith("## ")]
                        added = 0
                        for ln in lines[:4]:
                            save_flashcard(deck_name=deck_name, front=f"Key Concept: {ln[:50]}", back=ln[:200], source_doc=target_doc)
                            added += 1
                        st.success(f"Added {added} flashcards to deck '{deck_name}'!")
                with act5:
                    st.caption("📋 Select and copy text directly from the markdown above.")

    # =========================================================================
    # TAB 4: SEMANTIC SEARCH IN NOTES
    # =========================================================================
    with tab_search:
        st.subheader("🔎 Semantic Document Search")
        st.caption("Search across all indexed course notes with similarity scoring and exact page quotes.")

        doc_keys = ["All Documents"] + list(vector_store.documents.keys())
        default_filter = st.session_state.get("search_filter_doc", "All Documents")
        filter_idx = doc_keys.index(default_filter) if default_filter in doc_keys else 0

        q_col1, q_col2 = st.columns([3, 1])
        with q_col1:
            search_query = st.text_input("Enter search query or concept:", "CMMI maturity levels")
        with q_col2:
            filter_doc = st.selectbox("Search in Document:", options=doc_keys, index=filter_idx)

        param_col1, param_col2 = st.columns(2)
        with param_col1:
            top_k = st.slider("Max Chunks to Return:", min_value=2, max_value=10, value=4)
        with param_col2:
            min_score = st.slider("Minimum Similarity Threshold:", min_value=0.10, max_value=0.80, value=0.20, step=0.05)

        if st.button("🔍 Search Notes", type="primary", use_container_width=True):
            if not search_query.strip():
                st.warning("Please enter a query.")
            else:
                with st.spinner("Searching FAISS vector index..."):
                    embedding_service = EmbeddingService.get_instance()
                    q_vec = embedding_service.embed_query(search_query)

                    target_file = None if filter_doc == "All Documents" else filter_doc
                    results = vector_store.search(
                        query_vector=q_vec,
                        top_k=top_k,
                        min_score=min_score,
                        filter_filename=target_file,
                    )

                    if not results:
                        st.info("No matching notes found above the similarity threshold.")
                    else:
                        st.markdown(f"### Found {len(results)} relevant excerpt(s):")
                        for idx, match in enumerate(results, 1):
                            score_pct = round(match.get("score", 0.0) * 100, 1)
                            doc_name = match.get("filename", "Unknown")
                            page_num = match.get("page", 1)
                            text_snippet = match.get("text", "")

                            with st.expander(f"📌 Result {idx}: {doc_name} (Page {page_num}) — Relevance: {score_pct}%", expanded=(idx == 1)):
                                st.markdown(f"**Document:** `{doc_name}` | **Page:** {page_num} | **Similarity:** `{score_pct}%`")
                                st.markdown(f"> {text_snippet}")
