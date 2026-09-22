"""
Landing Page for SmartStudy AI.
Clean, modern Blue & White presentation page with guaranteed high-contrast typography,
centered container, smooth animations, and clear product benefits.
"""

import streamlit as st
from memory.auth_manager import authenticate_user, init_auth_db
from config.settings import APP_NAME, APP_SUBTITLE, APP_VERSION


def render_landing_page():
    """Render the standard product landing page with sleek Blue & White styling."""
    init_auth_db()

    # --- AGGRESSIVE HIGH-CONTRAST CSS ---
    st.markdown(
        """
        <style>
        /* 1. Global Page Background */
        .stApp {
            background-color: #F8FAFC !important;
        }

        /* 2. Container Max Width & Centering */
        .main .block-container {
            max-width: 1080px !important;
            padding-top: 1.5rem !important;
            padding-bottom: 3.5rem !important;
            margin: 0 auto !important;
        }

        /* 3. Primary Button: ALWAYS Vibrant Blue with Crisp Bright White Text */
        div.stButton > button[kind="primary"] {
            color: #FFFFFF !important;
            background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.25) !important;
            transition: all 0.2s ease !important;
        }

        div.stButton > button[kind="primary"]:hover {
            box-shadow: 0 8px 20px rgba(37, 99, 235, 0.4) !important;
            transform: translateY(-2px) !important;
        }

        div.stButton > button[kind="primary"] * {
            color: #FFFFFF !important;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            font-weight: 700 !important;
        }

        /* 4. Secondary Button: Crisp White with Blue Border and Blue Text */
        div.stButton > button[kind="secondary"],
        div.stButton > button:not([kind="primary"]) {
            color: #1E3A8A !important;
            background-color: #FFFFFF !important;
            border: 1.5px solid #93C5FD !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04) !important;
            transition: all 0.2s ease !important;
        }

        div.stButton > button[kind="secondary"]:hover,
        div.stButton > button:not([kind="primary"]):hover {
            border-color: #2563EB !important;
            background-color: #EFF6FF !important;
            color: #2563EB !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15) !important;
        }

        div.stButton > button[kind="secondary"] *,
        div.stButton > button:not([kind="primary"]) * {
            color: inherit !important;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            font-weight: 600 !important;
        }

        /* 5. Clean White Feature Containers */
        div[data-testid="stVerticalBlock"] > div[data-testid="stContainer"] {
            background-color: #FFFFFF !important;
            border: 1.5px solid #E2E8F0 !important;
            border-radius: 16px !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04) !important;
            padding: 1.4rem !important;
            transition: all 0.25s ease !important;
        }

        div[data-testid="stVerticalBlock"] > div[data-testid="stContainer"]:hover {
            transform: translateY(-4px) !important;
            border-color: #93C5FD !important;
            box-shadow: 0 12px 24px rgba(37, 99, 235, 0.1) !important;
        }

        /* 6. Typography */
        div[data-testid="stMarkdownContainer"] h2,
        div[data-testid="stMarkdownContainer"] h3 {
            color: #1E3A8A !important;
            font-weight: 800 !important;
        }

        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stMarkdownContainer"] li {
            color: #334155 !important;
            font-size: 0.98rem !important;
            line-height: 1.65 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- TOP NAVBAR ---
    col_brand, col_actions = st.columns([1.8, 2.2], vertical_alignment="center")

    with col_brand:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 2.2rem;">🎓</span>
                <div>
                    <h2 style="margin: 0; padding: 0; font-size: 1.6rem; font-weight: 900; color: #1E3A8A !important; line-height: 1.1;">{APP_NAME}</h2>
                    <span style="font-size: 0.82rem; color: #64748B; font-weight: 600;">Private Local AI Study Workspace</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_actions:
        c_demo, c_login, c_reg = st.columns([1.3, 1, 1])
        with c_demo:
            if st.button("⚡ Live Demo", use_container_width=True, type="primary", help="Instant 1-click access"):
                success, msg, user_data = authenticate_user("mca_student", "password123")
                if success and user_data:
                    st.session_state["authenticated"] = True
                    st.session_state["user"] = user_data
                    st.rerun()
        with c_login:
            if st.button("🔑 Sign In", use_container_width=True):
                st.session_state["unauth_view"] = "login"
                st.rerun()
        with c_reg:
            if st.button("📝 Register", use_container_width=True):
                st.session_state["unauth_view"] = "register"
                st.rerun()

    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

    # --- 1. HERO BANNER WITH INLINE FORCED WHITE TEXT ---
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 50%, #1D4ED8 100%) !important;
            padding: 50px 30px !important;
            border-radius: 20px !important;
            text-align: center !important;
            margin-bottom: 24px !important;
            box-shadow: 0 15px 35px -5px rgba(37, 99, 235, 0.4) !important;
        ">
            <span style="
                display: inline-block !important;
                background-color: rgba(255, 255, 255, 0.22) !important;
                color: #FFFFFF !important;
                padding: 6px 18px !important;
                border-radius: 20px !important;
                font-size: 0.85rem !important;
                font-weight: 700 !important;
                text-transform: uppercase !important;
                letter-spacing: 0.05em !important;
                border: 1px solid rgba(255, 255, 255, 0.35) !important;
            ">
                🔒 100% OFFLINE · PRIVATE · ON-DEVICE AI
            </span>
            <h1 style="
                color: #FFFFFF !important;
                font-size: 2.6rem !important;
                font-weight: 900 !important;
                margin-top: 18px !important;
                margin-bottom: 14px !important;
                line-height: 1.25 !important;
                text-shadow: 0 2px 8px rgba(0, 0, 0, 0.25) !important;
            ">
                Your Personal AI Study Companion for MCA & Computer Science
            </h1>
            <p style="
                color: #F8FAFC !important;
                font-size: 1.18rem !important;
                font-weight: 400 !important;
                max-width: 740px !important;
                margin: 0 auto !important;
                line-height: 1.6 !important;
                opacity: 0.96 !important;
                text-shadow: 0 1px 4px rgba(0, 0, 0, 0.2) !important;
            ">
                SmartStudy AI turns your lecture notes, textbooks, and syllabus into an intelligent study assistant. Everything runs directly on your laptop with zero cloud tracking and zero subscription fees.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- HERO CTA BUTTONS ---
    btn_col1, btn_col2, btn_col3 = st.columns([1.3, 1, 1])
    with btn_col1:
        if st.button("⚡ Try Live Demo (One-Click)", use_container_width=True, type="primary"):
            success, msg, user_data = authenticate_user("mca_student", "password123")
            if success and user_data:
                st.session_state["authenticated"] = True
                st.session_state["user"] = user_data
                st.rerun()
    with btn_col2:
        if st.button("🚀 Sign In to Your Workspace", use_container_width=True):
            st.session_state["unauth_view"] = "login"
            st.rerun()
    with btn_col3:
        if st.button("✨ Create Free Student Account", use_container_width=True):
            st.session_state["unauth_view"] = "register"
            st.rerun()

    st.markdown("<div style='margin-top: 35px;'></div>", unsafe_allow_html=True)

    # --- 2. ABOUT THE APP ---
    st.markdown("## 📖 About SmartStudy AI")
    st.markdown(
        """
        **SmartStudy AI** is an intelligent, offline academic copilot built specifically for Master of Computer Applications (MCA) 
        and Computer Science engineering students. 

        Unlike cloud chatbots that send your private lecture notes to remote servers, SmartStudy AI runs locally on your computer 
        using **Ollama**, **LangGraph**, and **local vector search (FAISS)**. It provides context-grounded answers strictly based on 
        your uploaded course materials.
        """
    )

    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)

    # --- 3. WHAT YOU GET (BENEFITS & FEATURES) ---
    st.markdown("## 🎁 What You Get as a Student")
    st.caption("Everything you need to study smarter, retain concepts, and ace your semester exams:")

    f_col1, f_col2 = st.columns(2, gap="medium")

    with f_col1:
        with st.container(border=True):
            st.markdown("### 💬 1. Autonomous AI Study Agent")
            st.markdown(
                "Ask questions in natural language. The LangGraph agent automatically chooses whether to explain concepts, "
                "solve math queries (*like 'two plus two' or percentages*), search your uploaded lecture notes, or recall your study goals."
            )

        with st.container(border=True):
            st.markdown("### 📚 2. Offline PDF Document RAG")
            st.markdown(
                "Upload your professor's lecture slides, PDFs, and syllabi. The local vector database indexes your documents "
                "instantly and gives you accurate answers grounded in your notes with page citations."
            )

        with st.container(border=True):
            st.markdown("### 📝 3. Automated Practice Quizzes & MCQs")
            st.markdown(
                "Generate custom multiple-choice tests from any computer science topic or uploaded document. "
                "Get instant score breakdowns and track weak topics automatically."
            )

    with f_col2:
        with st.container(border=True):
            st.markdown("### 📇 4. Spaced Repetition Flashcards (SM-2)")
            st.markdown(
                "Master tricky definitions and algorithms using the proven SuperMemo-2 spaced repetition algorithm. "
                "Flashcards automatically schedule reviews right before you forget."
            )

        with st.container(border=True):
            st.markdown("### 📅 5. Adaptive Semester Study Planner")
            st.markdown(
                "Set your exam date and daily available study hours. SmartStudy AI builds a balanced subject-by-subject "
                "timetable and dynamically adjusts your daily targets if you fall behind."
            )

        with st.container(border=True):
            st.markdown("### 🔒 6. 100% Privacy & Zero Cloud Costs")
            st.markdown(
                "All student accounts, lecture notes, passwords (PBKDF2-HMAC-SHA256 salted), and quiz history remain strictly on your laptop. "
                "Supports both embedded SQLite and MongoDB."
            )

    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

    # --- 4. HOW IT WORKS (3 SIMPLE STEPS) ---
    st.markdown("## 🚀 How It Works")
    st.caption("Get started in 3 simple steps:")

    s1, s2, s3 = st.columns(3)
    with s1:
        with st.container(border=True):
            st.markdown("#### **Step 1: Sign In**")
            st.markdown("Create a local student account or click **Live Demo** to enter your workspace instantly.")
    with s2:
        with st.container(border=True):
            st.markdown("#### **Step 2: Add Notes**")
            st.markdown("Upload your PDF notes or ask questions directly on core CS subjects like DSA, DBMS, and OS.")
    with s3:
        with st.container(border=True):
            st.markdown("#### **Step 3: Test & Retain**")
            st.markdown("Take AI-generated practice quizzes, review active recall flashcards, and track your semester progress.")

    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

    # --- 5. READY TO START CTA BANNER ---
    with st.container(border=True):
        st.markdown(
            f"""
            <div style="text-align: center; padding: 15px 10px;">
                <h2 style="margin-bottom: 8px; color: #1E3A8A !important;">Ready to Start Studying Smarter?</h2>
                <p style="color: #64748B !important; font-size: 1.05rem; margin-bottom: 20px;">
                    Join now and experience your private, local AI academic workspace.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c_b1, c_b2, c_b3 = st.columns([1.3, 1, 1])
        with c_b1:
            if st.button("⚡ Launch One-Click Demo", key="cta_bot_demo", use_container_width=True, type="primary"):
                success, msg, user_data = authenticate_user("mca_student", "password123")
                if success and user_data:
                    st.session_state["authenticated"] = True
                    st.session_state["user"] = user_data
                    st.rerun()
        with c_b2:
            if st.button("🚀 Sign In", key="cta_bot_login", use_container_width=True):
                st.session_state["unauth_view"] = "login"
                st.rerun()
        with c_b3:
            if st.button("✨ Create Account", key="cta_bot_reg", use_container_width=True):
                st.session_state["unauth_view"] = "register"
                st.rerun()

    # --- FOOTER ---
    st.markdown("---")
    st.markdown(
        f"""
        <div style="text-align: center; color: #64748B; font-size: 0.85rem; padding: 10px 0;">
            SmartStudy AI · v{APP_VERSION} · 100% Local & Privacy-Preserving Study Assistant for MCA Students
        </div>
        """,
        unsafe_allow_html=True,
    )
