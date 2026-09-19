"""
AI Chat page for SmartStudy AI powered by LangGraph Agent Workflow.
Supports both Normal Mode and Developer/Debug Mode showing complete LangGraph traces:
User question ➔ Selected tool ➔ Tool input ➔ Tool result ➔ Final answer.
"""

import streamlit as st
from llm.ollama_client import OllamaService
from llm.langchain_client import LangChainOllamaFactory
from rag.vector_store import VectorStore
from graph.study_graph import run_study_agent
from config.settings import DEFAULT_MODEL, DEFAULT_OLLAMA_HOST


def render_chat_page(ollama_service: OllamaService, active_model: str):
    """Render the LangGraph-powered chat interface with Developer/Debug mode."""
    st.markdown("## 💬 AI Study Chat (LangGraph Agent)")
    st.caption("Structured state machine routing requests across tools, persistent memory, and local LLM.")

    # Status & Configuration Bar
    vector_store = VectorStore()
    stats = vector_store.get_stats()
    has_docs = stats["total_documents"] > 0

    col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 1, 1])
    with col1:
        st.markdown(f"**Model:** `{active_model}` *(LangGraph)*")
    with col2:
        if has_docs:
            st.success(f"📚 {stats['total_documents']} doc(s) indexed", icon="🟢")
        else:
            st.caption("ℹ️ No documents uploaded")
    with col3:
        # Developer / Debug Mode Toggle
        debug_mode = st.toggle(
            "🛠️ Debug Mode",
            value=st.session_state.get("debug_mode", False),
            help="Show complete LangGraph trace: User question ➔ Selected tool ➔ Tool input ➔ Result ➔ Final answer",
        )
        st.session_state.debug_mode = debug_mode
    with col4:
        is_ok, _ = ollama_service.check_connection()
        if is_ok:
            st.success("Online", icon="🟢")
        else:
            st.error("Offline", icon="🔴")
    with col5:
        if st.button("🗑️ Clear", use_container_width=True, help="Reset conversation history"):
            st.session_state.messages = []
            st.rerun()

    # Pre-check Ollama status to give friendly early warnings
    is_connected, conn_msg = ollama_service.check_connection()
    if not is_connected:
        st.warning(
            "⚠️ **Ollama is not running locally.**\n\n"
            "Please start Ollama on your machine:\n"
            "- Start the Ollama desktop app, or run `ollama serve` in a terminal."
        )
    elif not ollama_service.is_model_installed(active_model):
        st.warning(
            f"⚠️ **Model `{active_model}` is not installed in Ollama.**\n\n"
            f"Run `ollama pull {active_model}` in your terminal to download it."
        )

    # Document Context Selector & In-Chat PDF Ingestion Popover
    doc_keys = list(vector_store.documents.keys())
    d_col1, d_col2 = st.columns([3, 1])
    with d_col1:
        if doc_keys:
            current_active = st.session_state.get("active_document", doc_keys[-1])
            if current_active not in doc_keys and current_active != "All Documents":
                current_active = doc_keys[-1]
            options = ["All Documents"] + doc_keys
            active_idx = options.index(current_active) if current_active in options else 0
            sel_doc = st.selectbox(
                "📄 **Active Document Context:**",
                options,
                index=active_idx,
                help="Choose which uploaded notes the AI prioritizes for QA, summaries, and quizzes"
            )
            st.session_state.active_document = sel_doc
        else:
            st.info("💡 **Tip:** Upload your course PDF below so you can ask questions or summarize it!")

    with d_col2:
        with st.popover("📎 Upload PDF Notes", use_container_width=True):
            st.markdown("#### 📤 Upload Notes to Chat")
            chat_files = st.file_uploader(
                "Upload course PDFs",
                type=["pdf"],
                accept_multiple_files=True,
                key="chat_inline_pdf",
                help="Uploaded files are chunked and added to local vector store."
            )
            if chat_files and st.button("🚀 Index & Add to Chat", type="primary", use_container_width=True, key="btn_idx_chat"):
                with st.spinner("Processing & indexing PDF notes..."):
                    from rag.pdf_processor import PDFProcessor
                    from rag.embeddings import EmbeddingService
                    proc = PDFProcessor()
                    emb_service = EmbeddingService.get_instance()
                    uploaded_names = []
                    for cf in chat_files:
                        cf_bytes = cf.read()
                        chunks, doc_stats = proc.process_pdf(cf_bytes, cf.name)
                        c_texts = [c["text"] for c in chunks]
                        embs = emb_service.embed_texts(c_texts)
                        vector_store.add_document(cf.name, chunks, embs, doc_stats)
                        uploaded_names.append(cf.name)
                    st.session_state.active_document = uploaded_names[-1]
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"📄 **I have processed and indexed your notes:** `{', '.join(uploaded_names)}`.\n\nYou can now ask me to:\n- 📖 *'Summarize this PDF'*\n- 🔍 *'What does the PDF say about [topic]?'*\n- 📝 *'Give me 10 questions from this PDF'*",
                    })
                    st.success("Indexed successfully!")
                    st.rerun()

    # Initialize session state for chat messages if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Clean up any orphaned user message from a previously interrupted execution
    if st.session_state.messages and st.session_state.messages[-1].get("role") == "user":
        st.session_state.messages.pop()

    # Display welcome and quick tool-calling suggestions
    show_samples = len(st.session_state.messages) == 0
    with st.expander("👋 Quick Actions & Natural Chat Examples (Click to Test)", expanded=show_samples):
        sample_col1, sample_col2 = st.columns(2)
        with sample_col1:
            if st.button("💬 'Hello' (Normal Conversation)", use_container_width=True, key="btn_hello"):
                st.session_state.prompt_to_send = "Hello"
                st.rerun()

            if st.button("🧮 'What is 20 added to 30?' (Math Tool)", use_container_width=True, key="btn_math_20_30"):
                st.session_state.prompt_to_send = "What is 20 added to 30?"
                st.rerun()

            if st.button("📅 'Create a 30-day study plan for TCS NQT.' (Planner)", use_container_width=True, key="btn_plan_30"):
                st.session_state.prompt_to_send = "Create a 30-day study plan for TCS NQT."
                st.rerun()

            if st.button("💡 'Explain binary search.' (Educational AI)", use_container_width=True, key="btn_bin_search"):
                st.session_state.prompt_to_send = "Explain binary search."
                st.rerun()

        with sample_col2:
            if st.button("📖 'Summarize this PDF.' (PDF Summarizer)", use_container_width=True, key="btn_pdf_sum"):
                st.session_state.prompt_to_send = "Summarize this PDF."
                st.rerun()

            if st.button("🔍 'What does the PDF say about CMMI?' (Document QA)", use_container_width=True, key="btn_cmmi_qa"):
                st.session_state.prompt_to_send = "What does the PDF say about CMMI?"
                st.rerun()

            if st.button("📝 'Give me 5 questions from this PDF.' (PDF Quiz Generator)", use_container_width=True, key="btn_pdf_quiz"):
                st.session_state.prompt_to_send = "Give me 5 questions from this PDF."
                st.rerun()

            if st.button("❓ 'Explain question 5.' (Conversational Follow-up)", use_container_width=True, key="btn_follow_up"):
                st.session_state.prompt_to_send = "Explain question 5 in detail."
                st.rerun()

    # Display conversation history
    for msg in st.session_state.messages:
        role = msg.get("role", "user")
        avatar = "🎓" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            # If in Debug Mode, display complete LangGraph audit trail
            debug_trace = msg.get("debug_trace")
            tools_used = msg.get("tools_used")

            if debug_mode and debug_trace:
                with st.expander("🛠️ LangGraph Developer Trace", expanded=False):
                    for idx, step in enumerate(debug_trace, start=1):
                        st.markdown(f"**Step {idx} - Node:** `{step.get('node')}`")
                        if "selected_tools" in step:
                            st.write(f"- **Selected Tools:** `{step.get('selected_tools')}`")
                        if "tool_executions" in step:
                            for tex in step.get("tool_executions", []):
                                st.write(f"- **Tool:** `{tex.get('tool')}`")
                                st.write(f"- **Input:** `{tex.get('args')}`")
                                st.write(f"- **Result:** `{tex.get('result')[:200]}...`")
                        if "final_answer" in step:
                            st.write(f"- **Final Answer Length:** {len(step.get('final_answer', ''))} chars")
                        st.markdown("---")

            # Normal Mode: Show clean collapsed tools card if tools were invoked
            elif not debug_mode and tools_used:
                with st.expander(f"🛠️ Tools Used ({len(tools_used)})", expanded=False):
                    for t in tools_used:
                        st.markdown(f"**Tool:** `{t['tool']}`")
                        st.markdown(f"**Input:** `{t['args']}`")
                        st.caption(f"**Result:** {t['result'][:200]}...")

            st.markdown(msg.get("content", ""))

    # Handle queued prompt
    queued_prompt = st.session_state.pop("prompt_to_send", None)

    # User input
    user_input = st.chat_input("Ask a question, request a calculation, or ask to search your notes...")
    prompt = user_input or queued_prompt

    if prompt:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Render user message
        with st.chat_message("user", avatar="🎓"):
            st.markdown(prompt)

        # Execute LangGraph workflow
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner(f"🧠 Local AI (`{active_model}`) is thinking..."):
                try:
                    result = run_study_agent(
                        user_query=prompt,
                        conversation_history=st.session_state.messages[:-1],
                        thread_id=None,
                        model=active_model,
                        base_url=ollama_service.host,
                    )
                    final_text = result.get("final_response", "I could not generate an answer.")
                    tools_used = result.get("tool_results", [])
                    debug_trace = result.get("debug_trace", [])
                except Exception as e:
                    final_text = f"⚠️ Error processing request: {str(e)}"
                    tools_used = []
                    debug_trace = [{"error": str(e)}]

            # Developer / Debug Mode Visual Card
            if debug_mode:
                st.markdown("### 🛠️ LangGraph Execution Trace")
                t_col1, t_col2 = st.columns([1, 1])
                with t_col1:
                    st.info(f"**1. User Question:**\n{prompt}")
                    sel_tools = result.get("selected_tools", [])
                    st.warning(f"**2. Selected Tool(s):**\n`{sel_tools or 'None (Direct Answer)'}`")
                with t_col2:
                    if tools_used:
                        st.success(f"**3. Tool Input:**\n`{tools_used[0].get('args')}`")
                        st.code(f"4. Tool Result:\n{tools_used[0].get('result')}", language="text")
                    else:
                        st.info("No tools required for this query.")

                st.markdown("#### **5. Final Answer:**")

            # Normal Mode: Show clean collapsed tools card
            elif tools_used:
                with st.expander(f"🛠️ Tools Used ({len(tools_used)})", expanded=False):
                    for t in tools_used:
                        st.markdown(f"**Tool:** `{t['tool']}`")
                        st.markdown(f"**Input:** `{t['args']}`")
                        st.caption(f"**Result:** {t['result'][:250]}...")

            st.markdown(final_text)

        # Store in session state
        msg_obj = {
            "role": "assistant",
            "content": final_text,
            "tools_used": tools_used,
            "debug_trace": debug_trace,
        }
        st.session_state.messages.append(msg_obj)
