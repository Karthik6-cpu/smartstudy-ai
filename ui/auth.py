"""
Authentication UI for SmartStudy AI.
Provides clean Blue & White login, registration, and local account management with smooth CSS animations.
"""

import streamlit as st
from memory.auth_manager import authenticate_user, register_user, init_auth_db
from config.settings import APP_NAME, APP_SUBTITLE, APP_VERSION


def render_auth_page(initial_tab: str = "login"):
    """
    Render the dedicated student login & registration gateway.

    Args:
        initial_tab: 'login' or 'register'
    """
    init_auth_db()

    # --- BLUE & WHITE THEME STYLING WITH SMOOTH ANIMATIONS ---
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #F8FAFC;
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(16px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes pulseGlow {
            0%, 100% {
                box-shadow: 0 0 0 0 rgba(37, 99, 235, 0.4);
            }
            50% {
                box-shadow: 0 0 0 6px rgba(37, 99, 235, 0);
            }
        }

        div[data-testid="stVerticalBlock"] > div[data-testid="stContainer"] {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.04);
            padding: 1.75rem;
            animation: fadeInUp 0.5s ease-out;
            transition: all 0.3s ease;
        }

        div[data-testid="stVerticalBlock"] > div[data-testid="stContainer"]:hover {
            box-shadow: 0 12px 28px rgba(37, 99, 235, 0.08);
        }

        button[kind="primary"] {
            background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25) !important;
            transition: all 0.2s ease !important;
        }

        button[kind="primary"]:hover {
            transform: translateY(-2px) scale(1.02) !important;
            box-shadow: 0 8px 20px rgba(37, 99, 235, 0.4) !important;
        }

        button[kind="primary"] * {
            color: #FFFFFF !important;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }

        button[kind="secondary"] {
            background-color: #FFFFFF !important;
            border: 1px solid #CBD5E1 !important;
            color: #1E293B !important;
            border-radius: 10px !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
        }

        button[kind="secondary"]:hover {
            transform: translateY(-2px) !important;
            border-color: #2563EB !important;
            color: #2563EB !important;
        }

        button[kind="secondary"] * {
            color: inherit !important;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }

        h1, h2, h3, h4 {
            color: #0F172A !important;
            font-weight: 700;
        }

        p, span, label {
            color: #334155;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Top navigation back to Landing Page
    col_back, _ = st.columns([1.5, 4])
    with col_back:
        if st.button("← Back to Landing Page", use_container_width=True, help="Return to the main landing page"):
            st.session_state["unauth_view"] = "landing"
            st.rerun()

    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

    # Center-aligned container layout
    _, center_col, _ = st.columns([1, 2.2, 1])

    with center_col:
        st.markdown(
            f"""
            <div style="text-align: center; margin-bottom: 20px;">
                <h1 style="margin-bottom: 4px; font-size: 2.2rem; color: #1E3A8A !important;">🎓 {APP_NAME}</h1>
                <p style="color: #64748B; font-size: 1.05rem; margin-top: 0px;">{APP_SUBTITLE}</p>
                <span style="background-color: #EFF6FF; color: #1D4ED8; padding: 5px 14px; border-radius: 14px; font-size: 0.85rem; font-weight: 600; border: 1px solid #DBEAFE; display: inline-block;">
                    🔒 100% Offline & Local Privacy · v{APP_VERSION}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        auth_mode = st.radio(
            "Authentication Mode",
            options=["🔐 Student Sign In", "📝 Create New Account"],
            index=0 if initial_tab == "login" else 1,
            horizontal=True,
            label_visibility="collapsed",
        )

        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

        # ==========================================
        # SECTION 1: LOGIN
        # ==========================================
        if auth_mode == "🔐 Student Sign In":
            with st.container(border=True):
                st.markdown("#### Access Your Study Workspace")
                st.caption("Sign in with your local credentials or use the instant demo profile.")

                with st.form("dedicated_login_form", clear_on_submit=False):
                    login_user = st.text_input("Username or Email", placeholder="e.g. mca_student or student@smartstudy.local")
                    login_pass = st.text_input("Password", type="password", placeholder="Enter password")
                    
                    col_btn, col_demo = st.columns([1, 1])
                    with col_btn:
                        submitted_login = st.form_submit_button("🚀 Sign In", use_container_width=True, type="primary")
                    with col_demo:
                        submitted_demo = st.form_submit_button("⚡ Quick Demo Login", use_container_width=True)

                if submitted_login:
                    if not login_user or not login_pass:
                        st.error("⚠️ Please provide both username/email and password.")
                    else:
                        success, msg, user_data = authenticate_user(login_user, login_pass)
                        if success and user_data:
                            st.session_state["authenticated"] = True
                            st.session_state["user"] = user_data
                            st.success(f"✅ {msg}")
                            st.rerun()
                        else:
                            st.error(f"❌ {msg}")

                if submitted_demo:
                    success, msg, user_data = authenticate_user("mca_student", "password123")
                    if success and user_data:
                        st.session_state["authenticated"] = True
                        st.session_state["user"] = user_data
                        st.success("✅ Logged in as Demo Student!")
                        st.rerun()
                    else:
                        st.error("Demo account could not be loaded. Please register a new account.")

                st.info("💡 **Instant Demo**: Username: `mca_student` | Password: `password123`")

        # ==========================================
        # SECTION 2: REGISTER
        # ==========================================
        else:
            with st.container(border=True):
                st.markdown("#### Register New Student Profile")
                st.caption("Create your personalized local study profile. All data remains strictly on your device.")

                with st.form("dedicated_register_form", clear_on_submit=False):
                    reg_name = st.text_input("Full Name", placeholder="e.g. Rahul Sharma")
                    reg_username = st.text_input("Desired Username", placeholder="e.g. rahul_mca (3+ chars, alphanumeric)")
                    reg_email = st.text_input("Email Address", placeholder="e.g. rahul@example.com")
                    
                    col_deg, col_sem = st.columns(2)
                    with col_deg:
                        reg_degree = st.selectbox(
                            "Degree / Program",
                            options=[
                                "MCA (Master of Computer Applications)",
                                "B.Tech Computer Science & Engineering",
                                "M.Sc Computer Science",
                                "BCA (Bachelor of Computer Applications)",
                                "Other / Self Study",
                            ],
                            index=0,
                        )
                    with col_sem:
                        reg_sem = st.selectbox(
                            "Current Semester / Year",
                            options=[
                                "Semester 1",
                                "Semester 2",
                                "Semester 3",
                                "Semester 4",
                                "Semester 5",
                                "Semester 6",
                                "Final Year / Passed Out",
                            ],
                            index=0,
                        )

                    reg_pwd1 = st.text_input("Password", type="password", placeholder="At least 6 characters")
                    reg_pwd2 = st.text_input("Confirm Password", type="password", placeholder="Repeat password")

                    submitted_reg = st.form_submit_button("✨ Create Account & Log In", use_container_width=True, type="primary")

                if submitted_reg:
                    if not reg_name or not reg_username or not reg_email or not reg_pwd1:
                        st.error("⚠️ Please fill in all required fields.")
                    elif reg_pwd1 != reg_pwd2:
                        st.error("❌ Passwords do not match. Please re-enter carefully.")
                    else:
                        success, msg, user_data = register_user(
                            username=reg_username,
                            email=reg_email,
                            full_name=reg_name,
                            password=reg_pwd1,
                            degree_program=reg_degree,
                            semester=reg_sem,
                        )
                        if success and user_data:
                            st.session_state["authenticated"] = True
                            st.session_state["user"] = user_data
                            st.success(f"🎉 {msg} Welcome, {user_data['full_name']}!")
                            st.rerun()
                        else:
                            st.error(f"❌ {msg}")

        # Security & Privacy badge
        st.markdown("---")
        with st.container():
            st.caption(
                "🛡️ **Local Security Guarantee**: User accounts & passwords are cryptographically salted "
                "(PBKDF2-HMAC-SHA256, 100k iterations) and stored in local SQLite (`data/memory/student_memory.db`) or MongoDB. "
                "No credentials or study notes are ever transmitted to any external server or cloud."
            )
