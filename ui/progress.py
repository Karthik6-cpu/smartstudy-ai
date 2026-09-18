"""
Fully Functional Progress & Learning Analytics Page for SmartStudy AI.
Displays topic completion rates, quiz score trends, study hour distributions,
and recent chronological student activity feeds with SQLite persistence.
"""

import streamlit as st
import pandas as pd
from memory.storage_manager import (
    get_study_analytics,
    get_quiz_stats,
    get_quiz_history,
    get_recent_activities,
    get_study_tasks,
)


def render_progress_page():
    """Render the learning analytics, performance metrics, and activity feed view."""
    st.markdown("## 📊 Learning Analytics & Study Progress")
    st.caption("Track your syllabus completion, quiz accuracy over time, and semester study velocity.")

    # 1. Fetch persistent SQLite metrics
    study_stats = get_study_analytics()
    quiz_stats = get_quiz_stats()
    quiz_history = get_quiz_history(limit=20)
    activities = get_recent_activities(limit=15)
    all_tasks = get_study_tasks()

    # Top KPI Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            "Topics Completed",
            f"{study_stats['completed_tasks']} / {study_stats['total_tasks']}",
            f"{study_stats['completion_percentage']}% Done",
        )
    with m2:
        st.metric(
            "Average Quiz Score",
            f"{quiz_stats['avg_score']}%",
            f"Best: {quiz_stats['best_score']}%",
        )
    with m3:
        st.metric(
            "Quizzes Attempted",
            quiz_stats["total_quizzes"],
            f"{quiz_stats['total_questions_answered']} Qs answered",
        )
    with m4:
        st.metric(
            "Planned Study Time",
            f"{study_stats['total_hours']} hrs",
            f"{study_stats['in_progress_tasks']} In Progress",
        )

    # Main Progress Bar
    st.markdown("### 🎯 Overall Semester Syllabus Completion")
    comp_pct = study_stats["completion_percentage"]
    st.progress(comp_pct / 100.0 if study_stats["total_tasks"] > 0 else 0.0)
    st.caption(
        f"**{study_stats['completed_tasks']}** completed, "
        f"**{study_stats['in_progress_tasks']}** in progress, "
        f"**{study_stats['not_started_tasks']}** not started."
    )

    st.markdown("---")

    # 2. Visual Analytics Section
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("📈 Quiz Score Progression (%)")
        if not quiz_history:
            st.info("No quizzes taken yet! Take a practice quiz in the **📝 Quiz** page to see your score trend.")
        else:
            # Format data for chart (chronological order)
            quiz_df_data = []
            for item in reversed(quiz_history):
                short_topic = item["topic"].split(":")[0][:20]
                quiz_df_data.append({
                    "Quiz": f"{short_topic} (#{item['id']})",
                    "Score (%)": item["percentage"],
                })

            df_quiz = pd.DataFrame(quiz_df_data)
            st.line_chart(df_quiz.set_index("Quiz"))

    with col_chart2:
        st.subheader("📚 Subject Topic Coverage")
        subjects_data = study_stats.get("subjects", [])
        if not subjects_data:
            st.info("No study tasks logged yet! Create a revision plan in the **📅 Study Planner**.")
        else:
            subj_df_data = []
            for s in subjects_data:
                subj_df_data.append({
                    "Subject": s["subject"][:22],
                    "Completed": s["completed"] or 0,
                    "Total Planned": s["count"],
                })
            df_subj = pd.DataFrame(subj_df_data)
            st.bar_chart(df_subj.set_index("Subject"))

    st.markdown("---")

    # 3. Topic Breakdown & Recent Activity Feed
    col_topics, col_activity = st.columns([3, 2])

    with col_topics:
        st.subheader("📑 Topic Mastery Status")
        if not all_tasks:
            st.info("No active topics in planner.")
        else:
            for task in all_tasks[:8]:
                t_title = task["topic"]
                t_subj = task["subject"]
                t_stat = task["status"]
                t_prio = task["priority"]

                status_icon = "✅" if t_stat == "Completed" else ("⏳" if t_stat == "In Progress" else "⭕")
                st.markdown(f"{status_icon} **{t_title}** *({t_subj})*")
                st.caption(f"Status: **{t_stat}** · Priority: **{t_prio}** · Estimated: **{task.get('allocated_hours', 2.0)} hrs**")
                st.markdown("<hr style='margin: 2px 0;'/>", unsafe_allow_html=True)

    with col_activity:
        st.subheader("🕒 Recent Study Activity Timeline")
        if not activities:
            st.info("No activity recorded yet.")
        else:
            for act in activities:
                a_type = act["activity_type"]
                a_desc = act["description"]
                a_time = act["created_at"]

                # Icon selector
                if "Quiz" in a_type:
                    badge = "📝"
                elif "Completed" in a_type:
                    badge = "🏆"
                elif "Plan" in a_type:
                    badge = "📅"
                elif "Memory" in a_type:
                    badge = "🧠"
                else:
                    badge = "📌"

                st.markdown(f"**{badge} {a_type}**")
                st.write(f"{a_desc}")
                st.caption(f"Time: {a_time}")
                st.markdown("<hr style='margin: 4px 0;'/>", unsafe_allow_html=True)
