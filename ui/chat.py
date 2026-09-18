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

    # Initialize session state for chat messages if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display welcome and quick tool-calling suggestions if chat is empty
    if len(st.session_state.messages) == 0:
        st.markdown("---")
        st.markdown("### 👋 Try LangGraph Agent Workflows:")

        sample_col1, sample_col2 = st.columns(2)
        with sample_col1:
            if st.button("🧮 Calculate 25% of 480 (Calculator Node)", use_container_width=True):
                st.session_state.prompt_to_send = "Calculate 25% of 480."
                st.rerun()

            if st.button("📑 Explain CMMI from my uploaded notes (Doc Search Node)", use_container_width=True):
                st.session_state.prompt_to_send = "Explain CMMI from my uploaded Software Engineering notes."
                st.rerun()

            if st.button("🧠 Remember that my target exam score is 90% (Memory Node)", use_container_width=True):
                st.session_state.prompt_to_send = "Remember that my target exam score is 90%."
                st.rerun()

        with sample_col2:
            if st.button("📝 Give me 5 MCQs about DBMS joins (Quiz Node)", use_container_width=True):
                st.session_state.prompt_to_send = "Give me 5 MCQs about DBMS joins."
                st.rerun()

            if st.button("📅 Create a study plan for DSA and OS (Planner Node)", use_container_width=True):
                st.session_state.prompt_to_send = "Create a study plan for DSA and OS for the next 4 weeks."
                st.rerun()

            if st.button("🔍 What is my target exam score? (Memory Recall Node)", use_container_width=True):
                st.session_state.prompt_to_send = "What is my target exam score?"
                st.rerun()

        st.markdown("---")

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
            with st.spinner("LangGraph Agent reasoning across graph nodes..."):
                result = run_study_agent(
                    user_query=prompt,
                    conversation_history=st.session_state.messages[:-1],
                    thread_id="mca_student_session",
                    model=active_model,
                    base_url=ollama_service.host,
                )

            final_text = result.get("final_response", "I could not generate an answer.")
            tools_used = result.get("tool_results", [])
            debug_trace = result.get("debug_trace", [])

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
