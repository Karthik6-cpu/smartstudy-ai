"""
Fully Functional Study Planner Page for SmartStudy AI.
Interactive semester study timetable generator and task tracker with status management
(Not Started, In Progress, Completed) and SQLite persistence.
"""

from datetime import date, timedelta
import streamlit as st
from tools.study_tools import create_study_plan, calculate_study_schedule
from memory.storage_manager import (
    save_study_tasks,
    update_task_status,
    get_study_tasks,
    get_study_analytics,
    delete_task,
    clear_all_tasks,
)
from config.settings import DEFAULT_MODEL, DEFAULT_OLLAMA_HOST


def render_planner_page():
    """Render the interactive Study Planner and Task Tracker view."""
    st.markdown("## 📅 Semester Study Planner & Task Tracker")
    st.caption("Generate structured revision timetables using LangGraph tools and track topic progress across semesters.")

    tab_tracker, tab_generate = st.tabs(["📋 Interactive Task Tracker", "✨ Generate New Study Plan"])

    # --- TAB 1: INTERACTIVE TASK TRACKER ---
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
        st.markdown(f"**Overall Semester Progress: {completion_pct}%**")
        st.progress(completion_pct / 100.0 if total_tasks > 0 else 0.0)
        st.markdown("---")

        # Tasks List
        tasks = get_study_tasks()

        if not tasks:
            st.info(
                "No study tasks planned yet! Switch to the **'✨ Generate New Study Plan'** tab "
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

            filtered_tasks = tasks
            if status_filter != "All Statuses":
                filtered_tasks = [t for t in tasks if t["status"] == status_filter]

            st.write(f"Showing **{len(filtered_tasks)}** of **{len(tasks)}** task(s):")

            for t in filtered_tasks:
                tid = t["id"]
                subj = t["subject"]
                topic = t["topic"]
                prio = t["priority"]
                status = t["status"]
                edate = t.get("exam_date", "TBD")
                hours = t.get("allocated_hours", 2.0)

                # Priority Badge styling
                prio_badge = "🔴 High" if prio == "High" else ("🟡 Medium" if prio == "Medium" else "🟢 Low")

                t_col1, t_col2, t_col3, t_col4 = st.columns([4, 2, 3, 1])

                with t_col1:
                    st.markdown(f"**{topic}**")
                    st.caption(f"Subject: {subj} · Exam: {edate} · Target: {hours} hrs")

                with t_col2:
                    st.markdown(f"**Priority:** {prio_badge}")

                with t_col3:
                    # Status Action Buttons
                    sc1, sc2, sc3 = st.columns(3)
                    with sc1:
                        if st.button("⭕", key=f"ns_{tid}", help="Mark Not Started", disabled=(status == "Not Started")):
                            update_task_status(tid, "Not Started")
                            st.rerun()
                    with sc2:
                        if st.button("⏳", key=f"ip_{tid}", help="Mark In Progress", disabled=(status == "In Progress")):
                            update_task_status(tid, "In Progress")
                            st.rerun()
                    with sc3:
                        if st.button("✅", key=f"cp_{tid}", help="Mark Completed", disabled=(status == "Completed")):
                            update_task_status(tid, "Completed")
                            st.rerun()

                    # Current status label
                    st.caption(f"Current: **{status}**")

                with t_col4:
                    if st.button("🗑️", key=f"del_t_{tid}", help="Delete this task"):
                        delete_task(tid)
                        st.rerun()

                st.markdown("<hr style='margin: 4px 0;'/>", unsafe_allow_html=True)

            # Clear All Tasks button
            if st.button("⚠️ Clear All Study Tasks", type="secondary"):
                clear_all_tasks()
                st.warning("Cleared all study tasks from SQLite.")
                st.rerun()

    # --- TAB 2: GENERATE NEW STUDY PLAN ---
    with tab_generate:
        st.markdown("### 🛠️ Configure Study Plan Generator")

        col_in1, col_in2 = st.columns(2)

        with col_in1:
            subjects_input = st.text_area(
                "Subjects to Cover (Comma-separated):",
                value="Data Structures & Algorithms, Operating Systems, Database Management Systems",
                help="Enter course titles for this semester.",
            )
            topics_input = st.text_area(
                "Specific Topics to Master (One per line or comma-separated):",
                value="Binary Search Trees & Traversals\nCPU Scheduling Algorithms\nDeadlock Avoidance & Banker's Algorithm\nDatabase Normalization (1NF to BCNF)\nSQL Joins & Nested Queries",
                help="List high-yield topics to convert into trackable milestones.",
            )

        with col_in2:
            default_exam = date.today() + timedelta(days=30)
            exam_date_val = st.date_input("Target Exam Date:", value=default_exam)
            days_left = max((exam_date_val - date.today()).days, 1)
            st.info(f"⏳ **{days_left} days** remaining until exam day.")

            daily_hours_val = st.slider("Daily Available Study Hours:", min_value=1.0, max_value=8.0, value=3.0, step=0.5)
            priority_val = st.selectbox("Topic Priority Tier:", ["High", "Medium", "Low"])

        # Quick calculation preview
        calc_summary = calculate_study_schedule.invoke({
            "subjects": subjects_input,
            "total_weeks": max(days_left // 7, 1),
            "daily_hours": daily_hours_val,
        })
        with st.expander("⏱️ Workload Estimate Preview", expanded=False):
            st.markdown(calc_summary)

        st.markdown("---")
        if st.button("🚀 Generate & Save Structured Study Plan", type="primary", use_container_width=True):
            with st.spinner("LangGraph study planner generating personalized revision plan..."):
                # 1. Call study_planner tool
                plan_text = create_study_plan.invoke({
                    "subjects": subjects_input,
                    "available_hours": daily_hours_val,
                    "exam_date": f"{exam_date_val} ({days_left} days left)",
                })

                # 2. Parse individual topics into structured tasks
                raw_topics = [
                    t.strip() for t in topics_input.replace(",", "\n").split("\n") if t.strip()
                ]
                if not raw_topics:
                    raw_topics = [
                        "Data Structures Fundamentals",
                        "OS Process Scheduling",
                        "DBMS Normalization",
                    ]

                sub_list = [s.strip() for s in subjects_input.split(",") if s.strip()]
                allocated_hrs_per_topic = round((daily_hours_val * days_left) / max(len(raw_topics), 1), 1)

                new_tasks = []
                for i, top in enumerate(raw_topics):
                    assigned_sub = sub_list[i % len(sub_list)] if sub_list else "Computer Science"
                    new_tasks.append({
                        "plan_id": f"plan_{date.today().isoformat()}",
                        "subject": assigned_sub,
                        "topic": top,
                        "priority": priority_val,
                        "status": "Not Started",
                        "exam_date": str(exam_date_val),
                        "allocated_hours": allocated_hrs_per_topic,
                    })

                # 3. Save to SQLite
                count = save_study_tasks(new_tasks)

                st.success(f"✅ Generated and saved **{count}** trackable study tasks to SQLite!")
                st.markdown("### 🗓️ Generated Study Plan Overview")
                st.markdown(plan_text)
                st.info("Head over to the **'📋 Interactive Task Tracker'** tab above to start marking your progress!")
