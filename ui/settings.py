"""
Settings page for SmartStudy AI.
Allows configuration of Ollama connection, LLM model selection, and diagnostics.
"""

import streamlit as st
from llm.ollama_client import OllamaService
from config.settings import DEFAULT_MODEL, DEFAULT_OLLAMA_HOST, RECOMMENDED_MODELS


def render_settings_page(ollama_service: OllamaService):
    """Render the application configuration and diagnostics view."""
    st.markdown("## ⚙️ Configuration & Diagnostics")
    st.caption("Customize your local Ollama connection and study assistant model.")

    # 1. Ollama Host Configuration
    st.subheader("1. Local Ollama Server")
    current_host = st.session_state.get("ollama_host", DEFAULT_OLLAMA_HOST)
    new_host = st.text_input(
        "Ollama Host URL",
        value=current_host,
        help="Default is http://localhost:11434 for local Ollama instances.",
    )

    if new_host != current_host:
        st.session_state.ollama_host = new_host
        st.rerun()

    # Connection Test
    col1, col2 = st.columns([1, 2])
    with col1:
        test_clicked = st.button("🔄 Test Connection", use_container_width=True)

    if test_clicked:
        service = OllamaService(host=new_host)
        is_ok, msg = service.check_connection()
        if is_ok:
            st.success(f"Successfully connected to Ollama at `{new_host}`!")
        else:
            st.error(f"Connection Failed:\n\n{msg}")

    st.markdown("---")

    # 2. Model Selection
    st.subheader("2. Active LLM Model")

    # Fetch installed models from Ollama
    installed = ollama_service.get_installed_models()
    current_model = st.session_state.get("selected_model", DEFAULT_MODEL)

    if installed:
        st.write(f"Found **{len(installed)}** model(s) installed on this machine.")

        # Build options list: installed models + recommended + custom
        options = list(installed)
        if current_model not in options:
            options.append(current_model)

        selected = st.selectbox(
            "Select from installed models:",
            options=options,
            index=options.index(current_model) if current_model in options else 0,
            help="Choose the model to power SmartStudy AI chat.",
        )
        if selected != current_model:
            st.session_state.selected_model = selected
            st.success(f"Active model set to `{selected}`!")
    else:
        st.info("No installed models detected or Ollama is offline.")

    # Option to manually type a model name
    custom_model = st.text_input(
        "Or specify a custom model name / tag:",
        value=current_model,
        help="For example: llama3.2, mistral, deepseek-r1:1.5b, etc.",
    )
    if custom_model and custom_model != current_model:
        if st.button("Set Custom Model"):
            st.session_state.selected_model = custom_model.strip()
            st.success(f"Active model updated to `{custom_model.strip()}`!")
            st.rerun()

    st.markdown("---")

    # 3. Model Recommendations & Setup Guide
    st.subheader("3. Recommended Local Models for Students")
    st.markdown(
        """
        Below are lightweight, capable models that run comfortably on standard laptops (8GB–16GB RAM):
        """
    )

    for m in RECOMMENDED_MODELS:
        cols = st.columns([2, 3])
        with cols[0]:
            is_present = ollama_service.is_model_installed(m)
            badge = "✅ Installed" if is_present else "⬇️ Not Installed"
            st.markdown(f"**`{m}`** ({badge})")
        with cols[1]:
            st.code(f"ollama pull {m}", language="bash")

    st.markdown("---")
    st.subheader("4. Setup & Troubleshooting Reference")
    st.markdown(
        """
        - **Install Ollama:** Download the installer for your OS from [ollama.com](https://ollama.com/download).
        - **Verify Installation:** Open a terminal and run `ollama --version`.
        - **Download Default Model:** Run `ollama pull llama3.2`.
        - **Start Ollama Daemon (if not running):** Run `ollama serve`.
        """
    )
