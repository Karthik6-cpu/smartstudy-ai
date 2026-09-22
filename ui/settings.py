"""
Settings page for SmartStudy AI.
Allows configuration of Ollama connection, LLM model selection, Database backend (SQLite / MongoDB), and diagnostics.
"""

import streamlit as st
from llm.ollama_client import OllamaService
from config.settings import DEFAULT_MODEL, DEFAULT_OLLAMA_HOST, RECOMMENDED_MODELS
from memory.mongo_manager import is_mongodb_available, DEFAULT_MONGO_URI, DEFAULT_MONGO_DB_NAME
from memory.auth_manager import get_active_db_type


def render_settings_page(ollama_service: OllamaService):
    """Render the application configuration and diagnostics view."""
    st.markdown("## ⚙️ Configuration & Diagnostics")
    st.caption("Customize your local Ollama connection, database backend, and study assistant model.")

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
        test_clicked = st.button("🔄 Test Ollama Connection", use_container_width=True)

    if test_clicked:
        service = OllamaService(host=new_host)
        is_ok, msg = service.check_connection()
        if is_ok:
            st.success(f"Successfully connected to Ollama at `{new_host}`!")
        else:
            st.error(f"Connection Failed:\n\n{msg}")

    st.markdown("---")

    # 2. Database Backend Configuration (SQLite & MongoDB)
    st.subheader("2. Database Engine & Persistence (SQLite / MongoDB)")
    active_db_name, _ = get_active_db_type()
    st.info(f"Currently Active Database: **{active_db_name}**")

    db_choice = st.radio(
        "Select Primary Database Backend:",
        options=["SQLite (Local File - Default)", "MongoDB (Local or Cloud Atlas)"],
        index=0 if "SQLite" in active_db_name else 1,
        horizontal=True,
    )

    if "MongoDB" in db_choice:
        mongo_uri = st.text_input(
            "MongoDB Connection URI:",
            value=st.session_state.get("mongo_uri", DEFAULT_MONGO_URI),
            placeholder="mongodb://localhost:27017 or mongodb+srv://...",
        )
        mongo_db = st.text_input(
            "MongoDB Database Name:",
            value=st.session_state.get("mongo_db_name", DEFAULT_MONGO_DB_NAME),
        )

        c_test, c_save = st.columns([1, 1])
        with c_test:
            if st.button("🔌 Test MongoDB Connection", use_container_width=True):
                ok, status_msg = is_mongodb_available(mongo_uri)
                if ok:
                    st.success(f"✅ {status_msg}")
                else:
                    st.warning(f"⚠️ {status_msg}")
        with c_save:
            if st.button("💾 Apply MongoDB Settings", use_container_width=True, type="primary"):
                st.session_state.mongo_uri = mongo_uri
                st.session_state.mongo_db_name = mongo_db
                st.success("MongoDB settings updated!")
                st.rerun()
    else:
        st.caption("Using embedded SQLite database at `data/memory/student_memory.db`. Zero setup needed.")

    st.markdown("---")

    # 3. Model Selection
    st.subheader("3. Active LLM Model")

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

    # 4. Model Recommendations & Setup Guide
    st.subheader("4. Recommended Local Models for Students")
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
    st.subheader("5. Setup & Troubleshooting Reference")
    st.markdown(
        """
        - **Install Ollama:** Download the installer for your OS from [ollama.com](https://ollama.com/download).
        - **Verify Installation:** Open a terminal and run `ollama --version`.
        - **Download Default Model:** Run `ollama pull llama3.2`.
        - **Start Ollama Daemon (if not running):** Run `ollama serve`.
        """
    )
