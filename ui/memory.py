"""
Fully Functional Memory Page for SmartStudy AI.
Persistent SQLite Student Memory and Learning Profile Manager.
Allows students to view stored study preferences, search, add, and clear memory.
"""

import time
import streamlit as st
from memory.sqlite_memory import (
    list_all_memories,
    save_memory,
    delete_memory,
    clear_all_memories,
    retrieve_memory,
)
from memory.storage_manager import log_activity


def seed_default_memories_if_empty():
    """Seed initial student study preferences if memory table is fresh."""
    memories = list_all_memories()
    if not memories:
        defaults = [
            ("degree_program", "Master of Computer Applications (MCA)"),
            ("target_exam_score", "90% in Semester Final Examinations"),
            ("preferred_languages", "Python, C++, and SQL"),
            ("primary_focus_area", "Data Structures, Operating Systems & DBMS"),
            ("revision_strategy", "Spaced Repetition & Daily Practice Quizzes"),
        ]
        for k, v in defaults:
            save_memory(k, v)


def render_memory_page():
    """Render the SQLite Student Memory management interface."""
    st.markdown("## 🧠 Student Memory & Learning Profile")
    st.caption("Persistent SQLite knowledge store holding your personal learning profile, target goals, and study habits.")

    # Seed initial profile if empty
    seed_default_memories_if_empty()

    memories = list_all_memories()

    # Metrics Row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Stored Facts", len(memories))
    with col2:
        st.metric("Storage Engine", "SQLite (`student_memory.db`)")
    with col3:
        st.metric("Persistence", "Survives Restarts ✅")

    st.markdown("---")

    # Tabs
    tab_view, tab_add, tab_search = st.tabs(["📋 Stored Study Information", "➕ Add New Study Fact", "🔍 Search Memory"])

    with tab_view:
        st.markdown("### 🎓 Current Student Profile & Preferences")
        st.caption("The LangGraph agent consults these facts to customize answers to your curriculum and learning style.")

        if not memories:
            st.info("No memories stored yet. Add your study goals below or tell the AI in chat!")
        else:
            for item in memories:
                m_key = item["key"]
                m_val = item["value"]
                m_time = item["updated_at"]

                c1, c2, c3 = st.columns([3, 5, 1])
                with c1:
                    st.markdown(f"**`{m_key}`**")
                with c2:
                    st.markdown(f"{m_val}")
                    st.caption(f"Saved: {m_time}")
                with c3:
                    if st.button("🗑️", key=f"del_mem_{m_key}", help=f"Delete {m_key}"):
                        delete_memory(m_key)
                        log_activity("Memory Deleted", f"Removed '{m_key}' from student memory.")
                        st.success(f"Deleted '{m_key}'")
                        time.sleep(0.3)
                        st.rerun()

                st.markdown("<hr style='margin: 4px 0;'/>", unsafe_allow_html=True)

            col_btn1, col_btn2 = st.columns([4, 1])
            with col_btn2:
                if st.button("⚠️ Clear All Memory", type="secondary", use_container_width=True):
                    clear_all_memories()
                    log_activity("Memory Cleared", "Cleared all stored student memories.")
                    st.warning("All memories cleared.")
                    time.sleep(0.4)
                    st.rerun()

    with tab_add:
        st.markdown("### ➕ Save New Study Information")
        new_key = st.text_input(
            "Memory Identifier / Key:",
            placeholder="e.g. weak_subject, target_company, semester_gpa_goal",
            help="A concise key describing the fact.",
        )
        new_val = st.text_area(
            "Memory Details / Value:",
            placeholder="e.g. Need more practice on Dynamic Programming and Banker's Algorithm",
            help="The detailed fact or preference to remember.",
        )

        if st.button("💾 Save to SQLite Memory", type="primary"):
            if new_key and new_val:
                res = save_memory(new_key, new_val)
                log_activity("Memory Saved", f"Saved '{new_key}': '{new_val[:40]}...'")
                st.success(res)
                time.sleep(0.4)
                st.rerun()
            else:
                st.error("Please provide both a key and a value.")

    with tab_search:
        st.markdown("### 🔍 Test Memory Retrieval Query")
        st.caption("See how the `retrieve_memory` tool looks up stored facts based on search terms.")
        search_query = st.text_input("Enter search query:", placeholder="e.g. language, score, focus")
        if search_query:
            result = retrieve_memory(search_query)
            st.markdown(result)
