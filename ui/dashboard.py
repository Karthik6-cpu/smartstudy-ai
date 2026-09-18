"""
Dashboard page for SmartStudy AI.
Overview of study assistant features, system status, and MCA learning tracks.
"""

import streamlit as st
from llm.ollama_client import OllamaService
from rag.vector_store import VectorStore
from config.settings import APP_NAME, APP_SUBTITLE, APP_VERSION


def render_dashboard_page(ollama_service: OllamaService, active_model: str):
    """Render the dashboard view."""
    st.markdown(f"# 🎓 {APP_NAME}")
    st.markdown(f"**{APP_SUBTITLE}** · *v{APP_VERSION} (LangChain Powered)*")
    st.markdown("---")

    # Overview Banner
    st.markdown(
        """
        Welcome to **SmartStudy AI** — your privacy-first, 100% local AI study companion built to help
        Master of Computer Applications (MCA) students master core computer science subjects, prepare for semester exams,
        and query your course lecture notes using local Retrieval-Augmented Generation (RAG) orchestrated with **LangChain**.
        """
    )

    # System Status Section
    st.subheader("🖥️ Local AI Engine & Vector Store Status")
    is_connected, msg = ollama_service.check_connection()
    installed_models = ollama_service.get_installed_models() if is_connected else []

    vector_store = VectorStore()
    rag_stats = vector_store.get_stats()

    status_col1, status_col2, status_col3, status_col4 = st.columns(4)

    with status_col1:
        if is_connected:
            st.metric("Ollama Service", "Connected 🟢", f"Host: {ollama_service.host}")
        else:
            st.metric("Ollama Service", "Disconnected 🔴", "Offline")

    with status_col2:
        st.metric("Active Model", f"{active_model} (LangChain)")

    with status_col3:
        st.metric("Indexed Documents", rag_stats["total_documents"])

    with status_col4:
        st.metric("Indexed Chunks", f"{rag_stats['total_chunks']} chunks")

    if not is_connected:
        st.error(
            f"**Ollama is currently unreachable.**\n\n"
            f"Please make sure Ollama is installed and running on your system:\n\n"
            f"```bash\n# Run in your terminal:\nollama serve\n```\n"
            f"*(Or launch the Ollama app on Windows / macOS)*"
        )
    elif not ollama_service.is_model_installed(active_model):
        st.warning(
            f"⚠️ Model `{active_model}` is not yet downloaded.\n\n"
            f"Run `ollama pull {active_model}` in your terminal to install it."
        )

    st.markdown("---")

    # Architecture Overview
    st.subheader("🏛️ Architecture (Modern LangChain Integration)")
    st.info(
        "**Streamlit UI** ➔ **LangChain LCEL Pipeline** ➔ **Ollama (`langchain-ollama`)** ➔ **Local LLM (`llama3.2`)**\n\n"
        "**RAG Flow:**\n"
        "**User Question** ➔ **LangChainFAISSRetriever** ➔ **FAISS Vector Store** ➔ **Prompt Template (`prompts/templates.py`)** ➔ **ChatOllama** ➔ **Streamlit Stream + Citations**\n\n"
        "- 🔒 **100% Local & Private:** Zero cloud data exposure, no OpenAI keys.\n"
        "- 🔗 **LangChain LCEL:** Declarative, composable chains with streaming and fallback resilience.\n"
        "- 📑 **Persistent Vector Index:** FAISS index and chunk metadata survive application restarts.\n"
        "- 🛠️ **Modular Tools:** Reusable calculation, note search, and curriculum lookup tools."
    )

    # MCA Core Subjects Quick Reference
    st.subheader("📚 Core MCA Subjects Supported")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            #### 🌲 Data Structures & Algorithms
            - Arrays, Linked Lists, Trees, Graphs
            - Sorting & Searching Algorithms
            - Dynamic Programming & Greedy Approaches
            """
        )
        st.markdown(
            """
            #### 🗄️ Database Management Systems
            - Relational Model & SQL Queries
            - Normalization (1NF to BCNF)
            - ACID Properties & Concurrency Control
            """
        )

    with c2:
        st.markdown(
            """
            #### 💻 Operating Systems
            - Process Scheduling & Threads
            - Deadlock Prevention & Avoidance
            - Memory Management & Virtual Memory
            """
        )
        st.markdown(
            """
            #### 🌐 Computer Networks
            - OSI vs TCP/IP Reference Models
            - Routing Protocols (OSPF, BGP, RIP)
            - Transport Layer (TCP, UDP, Handshakes)
            """
        )

    with c3:
        st.markdown(
            """
            #### ⚙️ Software Engineering & OOP
            - SDLC, Agile & CMMI Maturity Levels
            - Design Patterns & UML Diagrams
            - Java / C++ / Python Concepts
            """
        )
        st.markdown(
            """
            #### 🔐 Cybersecurity & Cloud Basics
            - Cryptography & Hashing
            - Network Security & Firewalls
            - Cloud Service Models (IaaS, PaaS, SaaS)
            """
        )

    st.markdown("---")
    st.caption("Tip: Use the sidebar to explore **💬 AI Chat**, **📚 Documents**, **📝 Quiz**, and **📅 Study Planner**!")
