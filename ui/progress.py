"""
Progress & Learning Analytics Page for SmartStudy AI.
Displays real topic completion rates, quiz score trends, subject coverage,
weak-topic accuracy trees, and chronological student audit logs with SQLite persistence.
"""

from collections import defaultdict
import streamlit as st
import pandas as pd
from memory.storage_manager import (
    get_study_analytics,
    get_quiz_stats,
    get_quiz_history,
    get_recent_activities,
    get_study_tasks,
    get_topic_performance,
    get_weak_topics,
    get_smart_recommendation,
)


def render_progress_page():
    """Render the learning analytics, performance metrics, and activity feed view."""
    st.markdown("## 📊 Learning Analytics & Study Progress")
    st.caption("Track your syllabus completion, quiz accuracy trends, weak-topic detection, and revision velocity.")

    # 1. Fetch persistent SQLite metrics
    study_stats = get_study_analytics()
    quiz_stats = get_quiz_stats()
    quiz_history = get_quiz_history(limit=20)
    activities = get_recent_activities(limit=15)
    all_tasks = get_study_tasks()
    topic_perf = get_topic_performance()
    weak_topics = get_weak_topics(60.0)

    # Smart Recommendation Banner
    rec = get_smart_recommendation()
    st.info(f"**{rec['title']}**: {rec['recommendation']}\n\n*Why:* {rec['reason']}")

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
            "Weak Areas Identified",
            len(weak_topics),
            "Topics < 60% accuracy" if weak_topics else "All topics >= 60% ✅",
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

    # 2. TOPIC PERFORMANCE TREE (REQUIREMENT 19)
    st.subheader("🎯 Topic Mastery & Weak-Topic Analysis")
    st.caption("Grounded directly in actual quiz answers. Repeated errors identify topics needing revision.")

    if not topic_perf:
        st.info("No topic performance data yet. Take a quiz in **📝 Quiz** to start generating your mastery tree!")
    else:
        # Group topics by subject
        subject_tree = defaultdict(list)
        for tp in topic_perf:
            subj = tp["subject"]
            subject_tree[subj].append(tp)

        for subj, topics in subject_tree.items():
            st.markdown(f"#### 📁 {subj}")
            for t in topics:
                top_name = t["topic"]
                acc = t["accuracy_pct"]
                tot = t["total_questions"]
                status = t["status"]
                is_weak = acc < 60.0

                badge = "⚠️ Needs Revision" if is_weak else "✅ Mastered"
                warn_prefix = "⚠️ " if is_weak else ""

                t_c1, t_c2, t_c3 = st.columns([5, 2, 2])
                with t_c1:
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;└── **{warn_prefix}{top_name}**: `{acc}%` ({tot} questions)")
                with t_c2:
                    if is_weak:
                        st.warning(badge)
                    else:
                        st.success(badge)
                with t_c3:
                    st.caption(f"Status: {status}")

    st.markdown("---")

    # 3. Visual Analytics Section
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("📈 Quiz Score Progression (%)")
        if not quiz_history:
            st.info("No quizzes taken yet! Take a practice quiz in **📝 Quiz** to see your score trend.")
        else:
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
            st.info("No study tasks logged yet! Create a revision plan in **📅 Study Planner**.")
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

    # 4. Chronological Activity Log
    st.subheader("🕒 Recent Study Activity Timeline")
    if not activities:
        st.caption("No activity recorded yet.")
    else:
        for act in activities:
            a_type = act.get("activity_type", "Activity")
            desc = act.get("description", "")
            ts = act.get("created_at", "")
            st.markdown(f"- **{a_type}** · *{ts}* — {desc}")
