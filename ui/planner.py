"""
Fully Functional Study Planner Page for SmartStudy AI.
Interactive semester study timetable generator, task tracker with status management
(Not Started, In Progress, Completed), document topic extraction, adaptive replan
(prioritizing weak topics without deleting completed tasks), and spaced-repetition revision.
"""

from datetime import date, timedelta
import streamlit as st
from rag.vector_store import VectorStore
from memory.storage_manager import (
    save_study_tasks,
    update_task_status,
    get_study_tasks,
    get_study_analytics,
    delete_task,
    clear_all_tasks,
    adaptive_replan,
    get_weak_topics,
    get_revision_schedule,
    mark_revision_completed,
    schedule_revision,
    get_smart_recommendation,
)
from tools.study_tools import create_study_plan, calculate_study_schedule


def render_planner_page():
    """Render the interactive Study Planner and Task Tracker view."""
    st.markdown("## 📅 Adaptive Semester Study Planner & Task Tracker")
    st.caption("Plan your semester timetable, adapt when behind schedule, import topics from lecture notes, and track spaced revision.")

    tab_tracker, tab_generate, tab_adaptive, tab_revision = st.tabs([
        "📋 Task Tracker & Schedule",
        "✨ Generate New Plan",
        "🔄 Adaptive Re-plan",
        "🔄 Spaced Revision Center",
    ])

    # =========================================================================
    # TAB 1: INTERACTIVE TASK TRACKER & SCHEDULE
    # =========================================================================
    with tab_tracker:
        analytics = get_study_analytics()
        total_tasks = analytics["total_tasks"]
        completed = analytics["completed_tasks"]
        in_progress = analytics["in_progress_tasks"]
        not_started = analytics["not_started_tasks"]
        completion_pct = analytics["completion_percentage"]
        total_hours = analytics["total_hours"]

        # Metric Cards
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Total Topics Planned", total_tasks)
        with col_m2:
            st.metric("Completed ✅", f"{completed} ({completion_pct}%)")
        with col_m3:
            st.metric("In Progress ⏳", in_progress)
        with col_m4:
            st.metric("Estimated Study Time", f"{total_hours} hrs")

        # Progress Bar
        st.markdown(f"**Overall Completion Progress: {completion_pct}%**")
        st.progress(completion_pct / 100.0 if total_tasks > 0 else 0.0)

        # Smart Recommendation Card
        rec = get_smart_recommendation()
        with st.container():
            st.markdown(f"### {rec['title']}")
            st.markdown(f"💡 **{rec['recommendation']}**")
            st.caption(f"*Rationale:* {rec['reason']}")

        st.markdown("---")

        # Tasks List
        tasks = get_study_tasks()

        if not tasks:
            st.info(
                "No study tasks planned yet! Switch to the **'✨ Generate New Plan'** tab "
                "to create your first personalized study schedule."
            )
        else:
            # Filter control
            filter_col1, filter_col2 = st.columns([3, 1])
            with filter_col1:
                status_filter = st.selectbox(
                    "Filter by Status:",
                    ["All Statuses", "Not Started", "In Progress", "Completed"],
                )
            with filter_col2:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                if st.button("🗑️ Clear Plan", use_container_width=True, type="secondary"):
                    clear_all_tasks()
                    st.warning("All study tasks cleared.")
                    st.rerun()

            filtered_tasks = get_study_tasks(status_filter=status_filter)

            for t in filtered_tasks:
                tid = t["id"]
                subj = t["subject"]
                top = t["topic"]
                prio = t.get("priority", "Medium")
                cur_status = t.get("status", "Not Started")
                task_d = t.get("task_date") or t.get("exam_date", "TBD")
                task_type = t.get("task_type", "Learning")
                hrs = t.get("allocated_hours", 2.0)

                # Visual priority badge
                prio_badge = "🔴 High" if prio == "High" else ("🟡 Medium" if prio == "Medium" else "🟢 Low")

                t_col1, t_col2, t_col3, t_col4 = st.columns([4, 2, 2, 1])

                with t_col1:
                    status_prefix = "✅" if cur_status == "Completed" else ("⏳" if cur_status == "In Progress" else "⭕")
                    st.markdown(f"**{status_prefix} {top}**")
                    st.caption(f"Subject: `{subj}` | Type: {task_type} | Scheduled: `{task_d}` ({hrs}h)")

                with t_col2:
                    st.markdown(f"Priority: **{prio_badge}**")

                with t_col3:
                    status_options = ["Not Started", "In Progress", "Completed"]
                    cur_idx = status_options.index(cur_status) if cur_status in status_options else 0

                    new_status = st.selectbox(
                        "Status",
                        options=status_options,
                        index=cur_idx,
                        key=f"status_sel_{tid}",
                        label_visibility="collapsed",
                    )
                    if new_status != cur_status:
                        update_task_status(tid, new_status)
                        st.rerun()

                with t_col4:
                    if st.button("❌", key=f"del_task_{tid}", help="Delete this task"):
                        delete_task(tid)
                        st.rerun()

                st.markdown("<hr style='margin: 4px 0;'/>", unsafe_allow_html=True)

    # =========================================================================
    # TAB 2: GENERATE NEW PLAN (WITH DOCUMENT IMPORT)
    # =========================================================================
    with tab_generate:
        st.subheader("✨ Generate Personalized Study Plan")
        st.caption("Plan your revision schedule across subjects, weak areas, and imported lecture notes.")

        vector_store = VectorStore()
        indexed_docs = vector_store.get_documents_summary()
        uploaded_doc_names = [d["filename"] for d in indexed_docs]

        # Import topics from uploaded document option
        import_from_doc = False
        selected_import_doc = None
        if uploaded_doc_names:
            import_from_doc = st.checkbox("📥 Import topics from uploaded study document")
            if import_from_doc:
                selected_import_doc = st.selectbox("Select Document:", uploaded_doc_names)

        g_col1, g_col2 = st.columns(2)
        with g_col1:
            if import_from_doc and selected_import_doc:
                doc_meta = vector_store.documents.get(selected_import_doc, {})
                default_subj = selected_import_doc.replace(".pdf", "")
                subject_input = st.text_input("Subject:", value=default_subj)

                # Extract sample topics from document chunks
                doc_chunks = vector_store.get_document_chunks(selected_import_doc)
                sample_topics = []
                for c in doc_chunks[:6]:
                    lines = [ln.strip() for ln in c.get("text", "").split("\n") if len(ln.strip()) > 15]
                    if lines:
                        sample_topics.append(lines[0][:50])
                default_topics_str = ", ".join(sample_topics[:5]) if sample_topics else "CMMI Levels, Waterfall, Agile"
                topics_input = st.text_area("Topics to Master (comma-separated):", value=default_topics_str)
            else:
                subject_input = st.text_input("Subject:", value="Software Engineering")
                topics_input = st.text_area(
                    "Topics to Master (comma-separated):",
                    value="CMMI Maturity Levels, SDLC Waterfall Model, Spiral Risk Model, Agile Scrum, Software Testing Metrics",
                )

        with g_col2:
            today = date.today()
            exam_date_val = st.date_input("Target Exam Date:", value=today + timedelta(days=14), min_value=today)
            daily_hours = st.slider("Daily Study Hours:", min_value=1.0, max_value=8.0, value=3.0, step=0.5)

            # Auto-detect weak topics to prioritize
            weak_topics_data = get_weak_topics(60.0)
            if weak_topics_data:
                weak_names = [w["topic"] for w in weak_topics_data]
                st.warning(f"⚠️ **Detected Weak Topics:** {', '.join(weak_names[:3])}. These will be given High priority!")

        if st.button("🚀 Generate & Save Study Plan", type="primary", use_container_width=True):
            topic_list = [t.strip() for t in topics_input.split(",") if t.strip()]
            if not topic_list:
                st.error("Please provide at least one topic.")
            else:
                with st.spinner("Generating structured study timetable..."):
                    weak_set = set(w["topic"].lower() for w in weak_topics_data)
                    new_tasks = []

                    for idx, top in enumerate(topic_list):
                        is_weak = any(w_name in top.lower() or top.lower() in w_name for w_name in weak_set)
                        priority = "High" if (is_weak or idx == 0) else ("Medium" if idx < len(topic_list) - 1 else "Low")
                        task_date = (today + timedelta(days=idx // 2)).isoformat()

                        new_tasks.append({
                            "plan_id": f"plan_{subject_input.lower()[:8]}",
                            "subject": subject_input,
                            "topic": top,
                            "priority": priority,
                            "status": "Not Started",
                            "exam_date": str(exam_date_val),
                            "task_date": task_date,
                            "task_type": "Revision" if is_weak else "Learning",
                            "allocated_hours": round(daily_hours / 2.0, 1),
                        })

                    saved_count = save_study_tasks(new_tasks)
                    st.success(f"✅ Generated and saved {saved_count} study tasks! View them in the Task Tracker tab.")
                    st.rerun()

    # =========================================================================
    # TAB 3: ADAPTIVE RE-PLAN (WHEN BEHIND SCHEDULE)
    # =========================================================================
    with tab_adaptive:
        st.subheader("🔄 Adaptive Study Plan Recalculator")
        st.caption(
            "Fell behind on your schedule? Recalculate remaining tasks starting from today. "
            "CRITICAL: Completed tasks are never deleted or modified."
        )

        tasks = get_study_tasks()
        completed_tasks = [t for t in tasks if t.get("status") == "Completed"]
        uncompleted_tasks = [t for t in tasks if t.get("status") != "Completed"]

        ad_col1, ad_col2 = st.columns(2)
        with ad_col1:
            st.metric("Completed Tasks (Safe)", len(completed_tasks))
        with ad_col2:
            st.metric("Uncompleted Tasks to Reschedule", len(uncompleted_tasks))

        replan_hours = st.slider("Adjusted Daily Hours:", min_value=1.0, max_value=8.0, value=2.5, step=0.5)

        if st.button("⚡ Recalculate & Re-prioritize Remaining Tasks", type="primary", use_container_width=True):
            with st.spinner("Reorganizing remaining tasks..."):
                res = adaptive_replan(daily_hours=replan_hours)
                st.success(f"✅ {res['message']} (Boosted {res.get('weak_boosted_count', 0)} weak topics to High Priority).")
                st.rerun()

    # =========================================================================
    # TAB 4: SPACED REPETITION REVISION CENTER
    # =========================================================================
    with tab_revision:
        st.subheader("🔄 Spaced Repetition Revision Center")
        st.caption("Review schedule automatically generated from completed study tasks and quizzes (Day 1, Day 3, Day 7 intervals).")

        schedule = get_revision_schedule()
        today_rev = schedule.get("today", [])
        upcoming_rev = schedule.get("upcoming", [])
        completed_rev = schedule.get("completed", [])

        st.markdown(f"#### 📅 Today's Due Revisions ({len(today_rev)})")
        if not today_rev:
            st.success("🎉 No revisions due today! You are all caught up.")
        else:
            for rev in today_rev:
                rid = rev["id"]
                top = rev["topic"]
                subj = rev["subject"]
                src = rev.get("source_type", "Quiz")
                due = rev["due_date"]

                r_col1, r_col2 = st.columns([5, 2])
                with r_col1:
                    st.markdown(f"**{top}** (`{subj}`)")
                    st.caption(f"Source: {src} | Due Date: `{due}`")
                with r_col2:
                    if st.button("✅ Mark Revised", key=f"rev_done_{rid}", use_container_width=True):
                        mark_revision_completed(rid, next_interval_days=3)
                        st.rerun()
                st.markdown("<hr style='margin: 4px 0;'/>", unsafe_allow_html=True)

        st.markdown(f"#### ⏳ Upcoming Revisions ({len(upcoming_rev)})")
        if not upcoming_rev:
            st.caption("No upcoming revisions currently scheduled.")
        else:
            for rev in upcoming_rev:
                st.markdown(f"- **{rev['topic']}** (`{rev['subject']}`) — Scheduled for `{rev['due_date']}` (Interval: {rev['interval_days']} days)")
