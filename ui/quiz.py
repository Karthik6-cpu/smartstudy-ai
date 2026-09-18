"""
Fully Functional Quiz Page for SmartStudy AI.
Interactive MCQ assessments with document grounding, instant scoring,
detailed explanations, and SQLite persistence.
"""

import streamlit as st
from chains.quiz_generator import generate_structured_quiz
from rag.vector_store import VectorStore
from memory.storage_manager import (
    save_quiz_result,
    get_quiz_history,
    get_quiz_stats,
    clear_quiz_history,
)
from config.settings import DEFAULT_MODEL, DEFAULT_OLLAMA_HOST


def render_quiz_page():
    """Render the interactive Quiz & Assessment view."""
    st.markdown("## 📝 Mock Quiz & Self-Assessment")
    st.caption("Test your conceptual mastery with interactive MCQs grounded in MCA curriculum or your uploaded lecture notes.")

    # Fetch available uploaded documents
    vector_store = VectorStore()
    doc_summaries = vector_store.get_documents_summary()
    uploaded_doc_names = [d["filename"] for d in doc_summaries]

    tab_take, tab_history = st.tabs(["🎯 Take Practice Quiz", "📜 Past Quiz History"])

    with tab_take:
        col_setup1, col_setup2 = st.columns([3, 2])

        with col_setup1:
            topic_options = [
                "Software Engineering: CMMI Maturity Levels & Process Models",
                "Data Structures: Binary Search Trees, AVL & Heaps",
                "Operating Systems: Deadlocks & CPU Scheduling",
                "Database Management Systems: Normalization & SQL Joins",
                "Computer Networks: TCP 3-Way Handshake & OSI Layers",
                "Custom Topic (Enter below)",
            ]
            selected_topic_opt = st.selectbox("Select Core Subject / Topic:", options=topic_options)

            if selected_topic_opt == "Custom Topic (Enter below)":
                quiz_topic = st.text_input("Enter Topic:", "Software Engineering CMMI")
            else:
                quiz_topic = selected_topic_opt

            # Optional Document Grounding
            doc_options = ["None (General Knowledge)"] + uploaded_doc_names
            selected_doc = st.selectbox(
                "Ground in Uploaded Document (Optional RAG Quiz):",
                options=doc_options,
                help="Select an uploaded PDF to generate questions specifically from your lecture notes!",
            )

        with col_setup2:
            difficulty = st.selectbox(
                "Difficulty Level:",
                ["Intermediate (Exam Ready)", "Beginner (Concept Building)", "Advanced (Technical Interview)"],
            )
            num_questions = st.slider("Number of Questions:", min_value=3, max_value=10, value=5)

            active_model = st.session_state.get("selected_model", DEFAULT_MODEL)
            active_host = st.session_state.get("ollama_host", DEFAULT_OLLAMA_HOST)

            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            generate_clicked = st.button("🚀 Generate Quiz", type="primary", use_container_width=True)

        if generate_clicked:
            with st.spinner(f"Generating {num_questions} questions for '{quiz_topic}'..."):
                questions = generate_structured_quiz(
                    topic=quiz_topic,
                    num_questions=num_questions,
                    difficulty=difficulty,
                    document_name=selected_doc,
                    model=active_model,
                    base_url=active_host,
                )
                st.session_state.active_quiz = questions
                st.session_state.active_quiz_meta = {
                    "topic": quiz_topic,
                    "difficulty": difficulty,
                    "document": selected_doc,
                    "total": len(questions),
                }
                st.session_state.quiz_submitted = False
                st.session_state.quiz_score = 0
                st.session_state.quiz_results = []
                st.rerun()

        st.markdown("---")

        # Display Active Quiz
        if "active_quiz" in st.session_state and st.session_state.active_quiz:
            meta = st.session_state.active_quiz_meta
            st.markdown(f"### 📋 Assessment: **{meta['topic']}**")
            st.caption(f"Difficulty: **{meta['difficulty']}** | Source: **{meta['document']}** | Questions: **{meta['total']}**")

            questions = st.session_state.active_quiz
            is_submitted = st.session_state.get("quiz_submitted", False)

            with st.form("quiz_submission_form"):
                user_answers = {}

                for q in questions:
                    qid = q["id"]
                    q_text = q["question"]
                    options = q["options"]

                    st.markdown(f"**Q{qid}. {q_text}**")
                    selected = st.radio(
                        f"Select answer for Q{qid}:",
                        options=options,
                        key=f"q_radio_{qid}",
                        label_visibility="collapsed",
                        disabled=is_submitted,
                    )
                    user_answers[qid] = selected
                    st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

                submit_button = st.form_submit_button("✅ Submit Answers", disabled=is_submitted, type="primary")

            if submit_button and not is_submitted:
                # Grade Quiz
                score = 0
                details = []

                for q in questions:
                    qid = q["id"]
                    chosen = user_answers.get(qid, "")
                    correct_letter = q.get("correct_answer", "A").strip().upper()
                    # Check if chosen option starts with correct letter (e.g. 'B) ...')
                    is_correct = chosen.strip().upper().startswith(correct_letter)
                    if is_correct:
                        score += 1

                    details.append({
                        "id": qid,
                        "question": q["question"],
                        "options": q["options"],
                        "chosen": chosen,
                        "correct_answer": correct_letter,
                        "is_correct": is_correct,
                        "explanation": q.get("explanation", ""),
                    })

                total = len(questions)
                percentage = round((score / total) * 100.0, 1)

                # Persist to SQLite
                save_quiz_result(
                    topic=meta["topic"],
                    difficulty=meta["difficulty"],
                    total_questions=total,
                    score=score,
                    percentage=percentage,
                    details=details,
                    document_name=meta["document"],
                )

                st.session_state.quiz_submitted = True
                st.session_state.quiz_score = score
                st.session_state.quiz_percentage = percentage
                st.session_state.quiz_results = details
                st.rerun()

            # Render Results & Explanations if submitted
            if is_submitted and "quiz_results" in st.session_state:
                score = st.session_state.quiz_score
                total = len(questions)
                pct = st.session_state.quiz_percentage

                st.markdown("---")
                if pct >= 80:
                    st.success(f"🎉 **Outstanding Job!** Score: **{score}/{total}** ({pct}%)")
                elif pct >= 50:
                    st.info(f"👍 **Good Effort!** Score: **{score}/{total}** ({pct}%) — Review the concepts below.")
                else:
                    st.warning(f"📚 **Keep Practicing!** Score: **{score}/{total}** ({pct}%) — Read through the explanations below.")

                st.markdown("### 🔍 Detailed Explanations & Review")

                for res in st.session_state.quiz_results:
                    qid = res["id"]
                    q_text = res["question"]
                    is_corr = res["is_correct"]
                    chosen = res["chosen"]
                    correct_letter = res["correct_answer"]
                    explanation = res["explanation"]

                    if is_corr:
                        st.markdown(f"**Q{qid}. {q_text}**")
                        st.success(f"✅ **Correct!** Your Answer: `{chosen}`")
                    else:
                        st.markdown(f"**Q{qid}. {q_text}**")
                        st.error(f"❌ **Incorrect.** Your Answer: `{chosen}` | Correct Option: **{correct_letter}**")

                    with st.expander(f"💡 Explanation for Q{qid}", expanded=not is_corr):
                        st.markdown(explanation)

                    st.markdown("<hr style='margin: 8px 0;'/>", unsafe_allow_html=True)

    with tab_history:
        st.markdown("### 📊 Your Quiz Performance History")
        stats = get_quiz_stats()

        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        with col_stat1:
            st.metric("Total Quizzes", stats["total_quizzes"])
        with col_stat2:
            st.metric("Average Score", f"{stats['avg_score']}%")
        with col_stat3:
            st.metric("Best Score", f"{stats['best_score']}%")
        with col_stat4:
            st.metric("Questions Answered", stats["total_questions_answered"])

        st.markdown("---")

        history_records = get_quiz_history()
        if not history_records:
            st.info("No quiz records yet! Take your first quiz in the tab above to track your scores.")
        else:
            for item in history_records:
                q_id = item["id"]
                topic = item["topic"]
                diff = item["difficulty"]
                score = item["score"]
                total = item["total_questions"]
                pct = item["percentage"]
                doc = item.get("document_name", "General")
                date_str = item["created_at"]

                hc1, hc2, hc3 = st.columns([4, 2, 2])
                with hc1:
                    st.markdown(f"**{topic}**")
                    st.caption(f"Source: {doc} · Difficulty: {diff} · Date: {date_str}")
                with hc2:
                    if pct >= 80:
                        st.success(f"{score}/{total} ({pct}%)")
                    elif pct >= 50:
                        st.info(f"{score}/{total} ({pct}%)")
                    else:
                        st.warning(f"{score}/{total} ({pct}%)")
                with hc3:
                    with st.expander(f"Review Qs (#{q_id})"):
                        for d in item.get("details", []):
                            status_emoji = "✅" if d.get("is_correct") else "❌"
                            st.markdown(f"**{status_emoji} Q: {d.get('question')}**")
                            st.write(f"- Your Answer: {d.get('chosen')}")
                            st.write(f"- Explanation: {d.get('explanation')}")
                            st.markdown("---")

                st.markdown("<hr style='margin: 4px 0;'/>", unsafe_allow_html=True)

            if st.button("🗑️ Clear Quiz History", type="secondary"):
                clear_quiz_history()
                st.warning("Cleared all quiz history records.")
                st.rerun()
