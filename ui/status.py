"""
System Status and Real-Time Live Health Diagnostics Page for SmartStudy AI.
Uses native Streamlit @st.fragment(run_every="1s") for seamless 1-second live auto-refreshing telemetry.
"""

import sys
import time
import platform
import streamlit as st
from datetime import datetime
from llm.ollama_client import OllamaService
from rag.vector_store import VectorStore
from rag.pdf_processor import HAS_OCR
from memory.mongo_manager import (
    is_mongodb_available,
    get_mongo_db,
    DEFAULT_MONGO_URI,
    DEFAULT_MONGO_DB_NAME,
)
from memory.auth_manager import list_users, get_active_db_type
from graph.study_graph import create_study_agent_graph
from tools.study_tools import ALL_STUDY_TOOLS
from config.settings import (
    APP_NAME,
    APP_VERSION,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_HOST,
    MEMORY_DB_PATH,
)


@st.fragment(run_every="1s")
def render_live_telemetry_fragment(ollama_service: OllamaService, active_model: str):
    """
    Live telemetry fragment that automatically updates every 1 second.
    Renders without full-page flickering using Streamlit Fragments.
    """
    now_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    # Header with Live Pulse Indicator
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 18px; padding: 12px 20px; background-color: #FFFFFF; border: 1.5px solid #E2E8F0; border-radius: 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.02);">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="display: inline-block; width: 12px; height: 12px; background-color: #10B981; border-radius: 50%; box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.25);"></span>
                <span style="font-weight: 800; color: #0F172A; font-size: 1.05rem;">LIVE TELEMETRY STREAM</span>
                <span style="background-color: #EFF6FF; color: #2563EB; font-size: 0.82rem; font-weight: 700; padding: 3px 10px; border-radius: 12px;">Auto-Refreshing every 1s</span>
            </div>
            <div style="color: #64748B; font-size: 0.9rem; font-family: monospace; font-weight: 600;">
                Timestamp: <strong style="color: #1E3A8A;">{now_str}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Real-time MongoDB Latency Check
    mongo_t0 = time.time()
    mongo_ok, mongo_msg = is_mongodb_available(DEFAULT_MONGO_URI)
    mongo_latency = (time.time() - mongo_t0) * 1000

    # 2. Real-time Ollama Check
    ollama_ok, _ = ollama_service.check_connection()

    # 3. Vector RAG Stats
    vector_store = VectorStore()
    rag_stats = vector_store.get_stats()

    # 4. Agent Graph Check
    try:
        g = create_study_agent_graph()
        graph_ok = g is not None
    except Exception:
        graph_ok = False

    # --- 4 REAL-TIME STATUS CARDS ---
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        with st.container(border=True):
            st.markdown("#### 🗄️ MongoDB Atlas")
            if mongo_ok:
                st.markdown("### 🟢 **Online**")
                st.caption(f"Cluster: `byv2wdo` · Latency: **{mongo_latency:.1f}ms**")
            else:
                st.markdown("### 🔴 **Offline**")
                st.caption("Fallback to local SQLite")

    with m2:
        with st.container(border=True):
            st.markdown("#### 🦙 Ollama LLM")
            if ollama_ok:
                model_installed = ollama_service.is_model_installed(active_model)
                st.markdown("### 🟢 **Online**")
                st.caption(f"Model: `{active_model}` ({'Ready' if model_installed else 'Missing'})")
            else:
                st.markdown("### 🔴 **Offline**")
                st.caption("Run: `ollama serve`")

    with m3:
        with st.container(border=True):
            st.markdown("#### 📚 FAISS Vector RAG")
            st.markdown("### 🟢 **Active**")
            st.caption(f"{rag_stats['total_documents']} Docs · {rag_stats['total_chunks']} Chunks")

    with m4:
        with st.container(border=True):
            st.markdown("#### ⚡ LangGraph Agent")
            if graph_ok:
                st.markdown("### 🟢 **Compiled**")
                st.caption(f"{len(ALL_STUDY_TOOLS)} Active Tools")
            else:
                st.markdown("### ⚠️ **Degraded**")

    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

    # --- DETAILED DIAGNOSTIC TELEMETRY ---
    col_left, col_right = st.columns(2, gap="medium")

    with col_left:
        # MongoDB Database Details
        with st.container(border=True):
            st.markdown("### 🗄️ Live Database Telemetry")
            st.markdown(f"**Database Engine:** MongoDB Atlas Cloud Cluster")
            st.markdown(f"**Database Name:** `{DEFAULT_MONGO_DB_NAME}`")
            st.markdown(f"**Host Node:** `cluster0.byv2wdo.mongodb.net`")
            st.markdown(f"**Live Ping Roundtrip:** `{mongo_latency:.1f} ms`")

            if mongo_ok:
                db = get_mongo_db(DEFAULT_MONGO_URI, DEFAULT_MONGO_DB_NAME)
                if db is not None:
                    user_count = db.users.count_documents({})
                    st.success(f"✅ MongoDB Atlas Connected · **{user_count}** Registered User(s)")
            else:
                st.info(f"SQLite Backup Path: `{MEMORY_DB_PATH}`")

        # Local RAG & Document Pipeline
        with st.container(border=True):
            st.markdown("### 📚 Local PDF Vector Store (FAISS)")
            st.markdown(f"**Embedding Model:** `all-MiniLM-L6-v2` (Local PyTorch)")
            st.markdown(f"**Dimension:** `384` float32 vectors")
            st.markdown(f"**Chunk Size:** `600` chars (100 char overlap)")
            st.markdown(f"**Indexed Documents:** **{rag_stats['total_documents']}** files")
            st.markdown(f"**Vector Chunks Indexed:** **{rag_stats['total_chunks']}** embeddings")

    with col_right:
        # Ollama LLM & Agent Details
        with st.container(border=True):
            st.markdown("### 🦙 Local AI Inference & Tools")
            st.markdown(f"**Ollama Host URL:** `{ollama_service.host}`")
            st.markdown(f"**Active Model:** `{active_model}`")

            if ollama_ok:
                installed_models = ollama_service.get_installed_models()
                st.markdown(f"**Installed Models:** {', '.join([f'`{m}`' for m in installed_models]) if installed_models else 'None'}")
            
            st.markdown("---")
            st.markdown(f"**LangGraph StateGraph:** `StudyAgentState` (Compiled ✅)")
            st.markdown(f"**Registered Agent Tools ({len(ALL_STUDY_TOOLS)}):**")
            tool_names = [t.name for t in ALL_STUDY_TOOLS]
            st.code(", ".join(tool_names), language="text")

        # Host System & Platform Info
        with st.container(border=True):
            st.markdown("### 💻 Laptop Environment & Platform")
            st.markdown(f"**Python Version:** `{sys.version.split()[0]}` ({platform.architecture()[0]})")
            st.markdown(f"**Operating System:** `{platform.system()} {platform.release()}`")
            st.markdown(f"**OCR Engine:** {'Available ✅' if HAS_OCR else 'Tesseract not detected (Text-based PDF)'}")
            st.markdown(f"**App Version:** `{APP_VERSION}`")


def render_status_page(ollama_service: OllamaService = None, show_back_button: bool = True):
    """Render the full system health & diagnostic status dashboard with 1-second live telemetry."""
    if ollama_service is None:
        ollama_service = OllamaService(host=st.session_state.get("ollama_host", DEFAULT_OLLAMA_HOST))

    active_model = st.session_state.get("selected_model", DEFAULT_MODEL)

    if show_back_button:
        col_b, _ = st.columns([1.5, 4])
        with col_b:
            if st.button("← Back to Workspace", help="Return to previous view"):
                if not st.session_state.get("authenticated", False):
                    st.session_state["unauth_view"] = "landing"
                st.rerun()

    st.markdown(
        f"""
        <div style="margin-bottom: 12px;">
            <h1 style="margin: 0; font-size: 2.2rem; color: #1E3A8A !important;">🖥️ Real-Time System Status Matrix</h1>
            <p style="color: #64748B; font-size: 1rem; margin-top: 4px;">Live health diagnostics auto-refreshing continuously every 1 second</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Render the 1-second auto-refreshing live telemetry fragment
    render_live_telemetry_fragment(ollama_service, active_model)
