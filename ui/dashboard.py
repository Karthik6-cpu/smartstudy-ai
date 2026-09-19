"""
Dashboard Page for SmartStudy AI.
Overview of study assistant features, system health matrix, explainable recommendation engine,
and live study metrics calculated strictly from SQLite.
"""

from datetime import date
import streamlit as st
from llm.ollama_client import OllamaService
from rag.vector_store import VectorStore
from rag.pdf_processor import HAS_OCR
from memory.storage_manager import (
    get_smart_recommendation,
    get_study_analytics,
    get_study_tasks,
    get_weak_topics,
    get_quiz_stats,
    get_revision_schedule,
)
from config.settings import APP_NAME, APP_SUBTITLE, APP_VERSION


def render_dashboard_page(ollama_service: OllamaService, active_model: str):
    """Render the main dashboard overview."""
    st.markdown(f"# 🎓 {APP_NAME}")
    st.markdown(f"**{APP_SUBTITLE}** · *v{APP_VERSION} (100% Local & Private)*")
    st.markdown("---")

    # 1. EXPLAINABLE RECOMMENDATION ENGINE CARD
    rec = get_smart_recommendation()
    with st.container():
        st.markdown(f"### {rec['title']}")
        st.markdown(f"💡 **{rec['recommendation']}**")
        st.caption(f"**Reason:** {rec['reason']}")

    st.markdown("---")

    # 2. LIVE METRICS ROW (FROM SQLITE)
    study_analytics = get_study_analytics()
    quiz_stats = get_quiz_stats()
    weak_topics = get_weak_topics(60.0)
    rev_schedule = get_revision_schedule()
    today_rev_count = len(rev_schedule.get("today", []))

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            "Semester Progress",
            f"{study_analytics['completion_percentage']}%",
            f"{study_analytics['completed_tasks']}/{study_analytics['total_tasks']} Tasks",
        )
    with m2:
        st.metric(
            "Quiz Accuracy",
            f"{quiz_stats['avg_score']}%",
            f"{quiz_stats['total_quizzes']} Quizzes Taken",
        )
    with m3:
        st.metric(
            "Weak Areas Identified",
            len(weak_topics),
            "Topics below 60%" if weak_topics else "All Topics > 60% ✅",
        )
    with m4:
        st.metric(
            "Today's Spaced Revisions",
            today_rev_count,
            "Due today" if today_rev_count > 0 else "Caught up! 🎉",
        )

    st.markdown("---")

    # 3. SYSTEM HEALTH & DIAGNOSTICS MATRIX
    st.subheader("🖥️ Local AI Engine & System Health")
    is_connected, msg = ollama_service.check_connection()
    vector_store = VectorStore()
    rag_stats = vector_store.get_stats()

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        if is_connected:
            st.metric("Ollama LLM", "Online 🟢", f"Host: {ollama_service.host}")
        else:
            st.metric("Ollama LLM", "Offline 🔴", "Run: ollama serve")

    with s2:
        model_ready = ollama_service.is_model_installed(active_model) if is_connected else False
        st.metric("Active Model", f"`{active_model}`", "Installed ✅" if model_ready else "Missing ⚠️")

    with s3:
        st.metric("Vector Index", f"{rag_stats['total_documents']} Docs", f"{rag_stats['total_chunks']} chunks")

    with s4:
        ocr_status = "Available ✅" if HAS_OCR else "Text-only (No Tesseract)"
        st.metric("Scanned OCR Engine", ocr_status)

    if not is_connected:
        st.warning(
            "⚠️ **Ollama is not running locally.**\n\n"
            "To use LLM capabilities, open a terminal and run `ollama serve`, or start the Ollama desktop app."
        )

    st.markdown("---")

    # 4. TODAY'S TASKS & STUDY OVERVIEW
    st.subheader("📋 Today's Study Agenda")
    all_tasks = get_study_tasks()
    today_str = date.today().isoformat()
    todays_tasks = [t for t in all_tasks if t.get("task_date") == today_str and t.get("status") != "Completed"]

    if not todays_tasks:
        if not all_tasks:
            st.info("No tasks planned. Go to **📅 Study Planner** to create your semester study schedule.")
        else:
            st.success("🎉 You have completed all scheduled tasks for today!")
    else:
        for t in todays_tasks:
            p_badge = "🔴 High" if t.get("priority") == "High" else "🟡 Medium"
            st.markdown(f"- **{t['topic']}** (`{t['subject']}`) — Priority: {p_badge} | Allocated: {t.get('allocated_hours', 2.0)}h")

    st.markdown("---")

    # 5. CORE MCA SUBJECTS QUICK REFERENCE
    st.subheader("📚 Core MCA Subjects Supported")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            #### 🌲 Data Structures & Algorithms
            - Binary Search Trees, AVL, Heaps, Graphs
            - Dynamic Programming, Greedy, Recursion
            - Big-O Time & Space Complexity
            """
        )
        st.markdown(
            """
            #### 🗄️ Database Management Systems
            - SQL Joins, Aggregations, Subqueries
            - 1NF, 2NF, 3NF, BCNF Normalization
            - ACID Transactions & Concurrency Control
            """
        )
    with c2:
        st.markdown(
            """
            #### 💻 Operating Systems
            - CPU Scheduling (FCFS, SJF, Round Robin)
            - Deadlocks: Banker's Algorithm, Coffman conditions
            - Virtual Memory, Paging, Belady's Anomaly
            """
        )
        st.markdown(
            """
            #### 🌐 Computer Networks
            - OSI vs TCP/IP Protocol Layers
            - TCP 3-Way Handshake & Congestion Control
            - Subnetting, Routing Protocols (OSPF, BGP)
            """
        )
    with c3:
        st.markdown(
            """
            #### ⚙️ Software Engineering & OOP
            - SDLC Models: Waterfall, Spiral, Agile Scrum
            - CMMI Staged & Continuous Maturity Levels
            - Cyclomatic Complexity & Testing Metrics
            """
        )
        st.markdown(
            """
            #### 🔒 Cloud Computing & Security
            - IaaS, PaaS, SaaS Service Architectures
            - Symmetric vs Asymmetric Cryptography, Hashing
            """
        )
