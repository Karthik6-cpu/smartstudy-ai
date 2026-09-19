"""
SmartStudy AI - Main Application Entry Point.
A privacy-first, local AI study assistant for MCA students.
"""

import streamlit as st

# Page Configuration - Must be the first Streamlit command
st.set_page_config(
    page_title="SmartStudy AI - MCA Study Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config.settings import (
    APP_NAME,
    APP_VERSION,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_HOST,
    PAGE_DASHBOARD,
    PAGE_CHAT,
    PAGE_DOCUMENTS,
    PAGE_QUIZ,
    PAGE_FLASHCARDS,
    PAGE_PLANNER,
    PAGE_PROGRESS,
    PAGE_MEMORY,
    PAGE_SETTINGS,
    PAGES,
)
from llm.ollama_client import OllamaService
from ui.dashboard import render_dashboard_page
from ui.chat import render_chat_page
from ui.documents import render_documents_page
from ui.quiz import render_quiz_page
from ui.flashcards import render_flashcards_page
from ui.planner import render_planner_page
from ui.progress import render_progress_page
from ui.memory import render_memory_page
from ui.settings import render_settings_page


def init_session_state():
    """Initialize necessary session state variables."""
    if "selected_model" not in st.session_state or st.session_state.selected_model == "llama3.2":
        st.session_state.selected_model = "llama3.2:1b"

    if "ollama_host" not in st.session_state:
        st.session_state.ollama_host = DEFAULT_OLLAMA_HOST

    if "messages" not in st.session_state:
        st.session_state.messages = []


def main():
    """Main routing and layout loop."""
    init_session_state()

    # Create service instance based on session configuration
    service = OllamaService(host=st.session_state.ollama_host)
    active_model = st.session_state.selected_model

    # --- SIDEBAR NAVIGATION ---
    with st.sidebar:
        st.markdown(f"## 🎓 {APP_NAME}")
        st.caption(f"Local AI Study Assistant · v{APP_VERSION}")
        st.markdown("---")

        selected_page = st.radio(
            "Navigation",
            options=PAGES,
            index=1,  # Default to AI Chat as it is the functional core of Part 1
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Sidebar Live Status Card
        is_connected, _ = service.check_connection()
        if is_connected:
            model_installed = service.is_model_installed(active_model)
            if model_installed:
                st.success(f"🟢 **Ollama Ready**\nModel: `{active_model}`")
            else:
                st.warning(f"🟡 **Model Missing**\nPull: `ollama pull {active_model}`")
        else:
            st.error("🔴 **Ollama Offline**\nRun: `ollama serve`")

        st.markdown("---")
        st.caption("🔒 100% Local · No API Keys · Zero Cloud Data")

    # --- PAGE ROUTING ---
    if selected_page == PAGE_DASHBOARD:
        render_dashboard_page(service, active_model)

    elif selected_page == PAGE_CHAT:
        render_chat_page(service, active_model)

    elif selected_page == PAGE_DOCUMENTS:
        render_documents_page()

    elif selected_page == PAGE_QUIZ:
        render_quiz_page()

    elif selected_page == PAGE_FLASHCARDS:
        render_flashcards_page()

    elif selected_page == PAGE_PLANNER:
        render_planner_page()

    elif selected_page == PAGE_PROGRESS:
        render_progress_page()

    elif selected_page == PAGE_MEMORY:
        render_memory_page()

    elif selected_page == PAGE_SETTINGS:
        render_settings_page(service)


if __name__ == "__main__":
    main()
