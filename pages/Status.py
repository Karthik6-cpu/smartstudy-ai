"""
Standalone /Status Endpoint for SmartStudy AI.
Directly accessible via URL: http://localhost:8501/Status
"""

import streamlit as st

st.set_page_config(
    page_title="System Status - SmartStudy AI",
    page_icon="🖥️",
    layout="wide",
)

from ui.status import render_status_page
from llm.ollama_client import OllamaService
from config.settings import DEFAULT_OLLAMA_HOST

service = OllamaService(host=st.session_state.get("ollama_host", DEFAULT_OLLAMA_HOST))
render_status_page(ollama_service=service, show_back_button=True)
