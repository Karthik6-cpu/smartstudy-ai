"""
Fully Functional Quiz & Exam Page for SmartStudy AI.
Interactive MCQ assessments with document grounding, exam-mode navigation (one question at a time,
question palette, mark for review, answer state preservation), mathematical grading,
SQLite persistence, and weak-topic tracking.
"""

import streamlit as st
from chains.quiz_generator import generate_structured_quiz
from rag.vector_store import VectorStore
from memory.storage_manager import (
    save_quiz_result,
    get_quiz_history,
    get_quiz_stats,
    clear_quiz_history,
    get_weak_topics,
)
from config.settings import DEFAULT_MODEL, DEFAULT_OLLAMA_HOST


def render_quiz_page():
    """Render the interactive Quiz & Exam view."""
    st.markdown("## 📝 Mock Quiz & Exam Assessment")
    st.caption("Test your conceptual mastery with interactive MCQs grounded in MCA curriculum or your uploaded lecture notes.")

    vector_store = VectorStore()
    doc_summaries = vector_store.get_documents_summary()
    uploaded_doc_names = [d["filename"] for d in doc_summaries]

    tab_take, tab_history, tab_weak = st.tabs([
        "🎯 Take Practice Quiz",
        "📜 Past Quiz History",
        "🎯 Weak Topics Identified",
    ])

    # =========================================================================
    # TAB 1: TAKE PRACTICE QUIZ
    # =========================================================================
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
            selected_topic_opt = st.selectbox("Select Subject / Topic:", options=topic_options)

            if selected_topic_opt == "Custom Topic (Enter below)":
                quiz_topic = st.text_input("Enter Topic:", "Software Engineering CMMI")
            else:
                quiz_topic = selected_topic_opt

            # Grounding: Clearly distinguish Document Grounding vs General Knowledge
            preselected_doc = st.session_state.get("quiz_target_doc", "None (General Knowledge)")
            doc_options = ["None (General Knowledge)"] + uploaded_doc_names
            default_doc_idx = doc_options.index(preselected_doc) if preselected_doc in doc_options else 0

            selected_doc = st.selectbox(
                "Source Grounding:",
                options=doc_options,
                index=default_doc_idx,
                help="Choose whether questions are extracted from your uploaded lecture notes or general CS knowledge.",
            )

            if selected_doc != "None (General Knowledge)":
                st.info(f"📚 **Grounded in uploaded document:** `{selected_doc}`")
            else:
                st.caption("🌐 **General MCA Curriculum knowledge** will be used.")

        with col_setup2:
            difficulty = st.selectbox(
                "Difficulty Level:",
                ["Medium", "Easy", "Hard", "Mixed"],
            )
            num_questions = st.select_slider(
                "Number of Questions:",
                options=[5, 10, 15, 20, 25],
                value=5,
            )

            exam_mode = st.radio(
                "Exam Interface Style:",
                ["Interactive Exam (One-by-One with Palette)", "Classic (All Questions)"],
                horizontal=True,
            )

            active_model = st.session_state.get("selected_model", DEFAULT_MODEL)
            active_host = st.session_state.get("ollama_host", DEFAULT_OLLAMA_HOST)

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
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
                    "mode": exam_mode,
                }
                st.session_state.quiz_answers = {}
                st.session_state.quiz_flags = set()
                st.session_state.quiz_current_q = 0
                st.session_state.quiz_submitted = False
                st.session_state.quiz_score = 0
                st.session_state.quiz_results = []
                st.rerun()

        st.markdown("---")

        # Display Active Quiz
        if "active_quiz" in st.session_state and st.session_state.active_quiz:
            questions = st.session_state.active_quiz
            meta = st.session_state.active_quiz_meta
            total_q = len(questions)
            is_submitted = st.session_state.get("quiz_submitted", False)

            if "quiz_answers" not in st.session_state:
                st.session_state.quiz_answers = {}
            if "quiz_flags" not in st.session_state:
                st.session_state.quiz_flags = set()
            if "quiz_current_q" not in st.session_state:
                st.session_state.quiz_current_q = 0

            st.markdown(f"### 📋 Assessment: **{meta['topic']}**")
            ground_tag = f"📚 `{meta['document']}`" if meta['document'] != "None (General Knowledge)" else "🌐 General MCA"
            st.caption(f"Difficulty: **{meta['difficulty']}** | Source: **{ground_tag}** | Questions: **{total_q}**")

            # -----------------------------------------------------------------
            # INTERACTIVE EXAM MODE (ONE QUESTION AT A TIME)
            # -----------------------------------------------------------------
            if meta.get("mode") == "Interactive Exam (One-by-One with Palette)" and not is_submitted:
                curr_idx = st.session_state.quiz_current_q
                curr_q = questions[curr_idx]
                qid = curr_q["id"]

                # Question Palette Navigation Bar
                st.markdown("**Question Palette:**")
                palette_cols = st.columns(min(total_q, 10))
                for idx, q in enumerate(questions):
                    col_idx = idx % min(total_q, 10)
                    with palette_cols[col_idx]:
                        q_num = idx + 1
                        has_ans = q["id"] in st.session_state.quiz_answers
                        is_flagged = q["id"] in st.session_state.quiz_flags
                        is_curr = idx == curr_idx

                        btn_label = f"🚩 Q{q_num}" if is_flagged else (f"✓ Q{q_num}" if has_ans else f"Q{q_num}")
                        btn_type = "primary" if is_curr else "secondary"

                        if st.button(btn_label, key=f"pal_{idx}", type=btn_type, use_container_width=True):
                            st.session_state.quiz_current_q = idx
                            st.rerun()

                st.progress((curr_idx + 1) / total_q)
                st.markdown(f"**Question {curr_idx + 1} of {total_q}**")

                # Question Text
                st.markdown(f"#### Q{curr_idx + 1}. {curr_q['question']}")

                # Selected Answer Handling
                options = curr_q["options"]
                saved_ans = st.session_state.quiz_answers.get(qid)
                saved_idx = options.index(saved_ans) if saved_ans in options else None

                chosen_opt = st.radio(
                    "Choose option:",
                    options=options,
                    index=saved_idx,
                    key=f"q_radio_one_{qid}_{curr_idx}",
                )
                if chosen_opt:
                    st.session_state.quiz_answers[qid] = chosen_opt

                # Flag for Review Checkbox
                is_currently_flagged = qid in st.session_state.quiz_flags
                flag_toggle = st.checkbox("🚩 Mark for Review", value=is_currently_flagged, key=f"flag_{qid}")
                if flag_toggle:
                    st.session_state.quiz_flags.add(qid)
                else:
                    st.session_state.quiz_flags.discard(qid)

                st.markdown("---")

                # Navigation Buttons: Previous, Next, Submit
                nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 2])
                with nav_col1:
                    if st.button("⬅️ Previous", disabled=(curr_idx == 0), use_container_width=True):
                        st.session_state.quiz_current_q = curr_idx - 1
                        st.rerun()
                with nav_col2:
                    if st.button("Next ➡️", disabled=(curr_idx == total_q - 1), use_container_width=True):
                        st.session_state.quiz_current_q = curr_idx + 1
                        st.rerun()
                with nav_col3:
                    ans_count = len(st.session_state.quiz_answers)
                    if st.button(f"✅ Submit Exam ({ans_count}/{total_q} Answered)", type="primary", use_container_width=True):
                        # Calculate Mathematical Score
                        score = 0
                        details = []
                        for q in questions:
                            q_item_id = q["id"]
                            user_chosen = st.session_state.quiz_answers.get(q_item_id, "")
                            corr_letter = q.get("correct_answer", "A").strip().upper()
                            is_correct = bool(user_chosen and user_chosen.strip().upper().startswith(corr_letter))
                            if is_correct:
                                score += 1

                            details.append({
                                "id": q_item_id,
                                "question": q["question"],
                                "options": q["options"],
                                "chosen": user_chosen or "Unanswered",
                                "correct_answer": corr_letter,
                                "is_correct": is_correct,
                                "explanation": q.get("explanation", ""),
                                "source": q.get("source", meta["document"]),
                            })

                        pct = round((score / total_q) * 100.0, 1)

                        # Save to SQLite & update topic performance
                        save_quiz_result(
                            topic=meta["topic"],
                            difficulty=meta["difficulty"],
                            total_questions=total_q,
                            score=score,
                            percentage=pct,
                            details=details,
                            document_name=meta["document"],
                            subject=meta["topic"].split(":")[0] if ":" in meta["topic"] else "General",
                        )

                        st.session_state.quiz_submitted = True
                        st.session_state.quiz_score = score
                        st.session_state.quiz_percentage = pct
                        st.session_state.quiz_results = details
                        st.rerun()

            # -----------------------------------------------------------------
            # CLASSIC MODE (ALL QUESTIONS LIST)
            # -----------------------------------------------------------------
            elif not is_submitted:
                with st.form("quiz_all_submission_form"):
                    user_answers = {}
                    for q in questions:
                        qid = q["id"]
                        st.markdown(f"**Q{qid}. {q['question']}**")
                        selected = st.radio(
                            f"Answer for Q{qid}:",
                            options=q["options"],
                            key=f"q_radio_all_{qid}",
                            label_visibility="collapsed",
                        )
                        user_answers[qid] = selected
                        st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)

                    submit_btn = st.form_submit_button("✅ Submit Answers", type="primary", use_container_width=True)

                if submit_btn:
                    score = 0
                    details = []
                    for q in questions:
                        qid = q["id"]
                        chosen = user_answers.get(qid, "")
                        corr_letter = q.get("correct_answer", "A").strip().upper()
                        is_correct = chosen.strip().upper().startswith(corr_letter)
                        if is_correct:
                            score += 1
                        details.append({
                            "id": qid,
                            "question": q["question"],
                            "options": q["options"],
                            "chosen": chosen,
                            "correct_answer": corr_letter,
                            "is_correct": is_correct,
                            "explanation": q.get("explanation", ""),
                            "source": q.get("source", meta["document"]),
                        })

                    pct = round((score / total_q) * 100.0, 1)

                    save_quiz_result(
                        topic=meta["topic"],
                        difficulty=meta["difficulty"],
                        total_questions=total_q,
                        score=score,
                        percentage=pct,
                        details=details,
                        document_name=meta["document"],
                        subject=meta["topic"].split(":")[0] if ":" in meta["topic"] else "General",
                    )

                    st.session_state.quiz_submitted = True
                    st.session_state.quiz_score = score
                    st.session_state.quiz_percentage = pct
                    st.session_state.quiz_results = details
                    st.rerun()

            # -----------------------------------------------------------------
            # POST-SUBMISSION SCORECARD & DETAILED EXPLANATION REVIEW
            # -----------------------------------------------------------------
            if is_submitted and "quiz_results" in st.session_state:
                score = st.session_state.quiz_score
                pct = st.session_state.quiz_percentage
                unanswered = sum(1 for r in st.session_state.quiz_results if r.get("chosen") == "Unanswered")
                incorrect = total_q - score - unanswered

                st.markdown("---")
                st.markdown("### 📊 Quiz Scorecard")

                sc_col1, sc_col2, sc_col3, sc_col4 = st.columns(4)
                with sc_col1:
                    st.metric("Total Questions", total_q)
                with sc_col2:
                    st.metric("Correct Answers ✅", score)
                with sc_col3:
                    st.metric("Incorrect Answers ❌", incorrect)
                with sc_col4:
                    st.metric("Final Score %", f"{pct}%")

                if pct >= 80:
                    st.success(f"🎉 **Outstanding Mastery!** You scored **{score}/{total_q}** ({pct}%)")
                elif pct >= 60:
                    st.info(f"👍 **Solid Progress!** You scored **{score}/{total_q}** ({pct}%) — Review weak questions below.")
                else:
                    st.warning(f"⚠️ **Needs Revision!** You scored **{score}/{total_q}** ({pct}%) — This topic has been queued for revision.")

                st.markdown("---")
                st.markdown("### 🔍 Question-by-Question Review & Explanations")

                for res in st.session_state.quiz_results:
                    qid = res["id"]
                    q_text = res["question"]
                    is_corr = res["is_correct"]
                    chosen = res["chosen"]
                    corr_letter = res["correct_answer"]
                    explanation = res["explanation"]
                    src = res.get("source", "General")

                    if is_corr:
                        st.markdown(f"**Q{qid}. {q_text}**")
                        st.success(f"✅ **Correct!** Your Choice: `{chosen}`")
                    else:
                        st.markdown(f"**Q{qid}. {q_text}**")
                        st.error(f"❌ **Incorrect.** Your Choice: `{chosen}` | Correct Answer: **{corr_letter}**")

                    st.caption(f"**Source Reference:** {src}")
                    with st.expander(f"💡 Explanation & Rationale for Q{qid}", expanded=not is_corr):
                        st.markdown(explanation)

                    st.markdown("<hr style='margin: 8px 0;'/>", unsafe_allow_html=True)

                if st.button("🔄 Retake Quiz / Start New Assessment", use_container_width=True):
                    st.session_state.active_quiz = None
                    st.session_state.quiz_submitted = False
                    st.rerun()

    # =========================================================================
    # TAB 2: PAST QUIZ HISTORY
    # =========================================================================
    with tab_history:
        st.markdown("### 📊 Historical Quiz Performance")
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
            st.info("No quiz records yet! Complete your first quiz to track your mastery.")
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
                    elif pct >= 60:
                        st.info(f"{score}/{total} ({pct}%)")
                    else:
                        st.warning(f"{score}/{total} ({pct}%) ⚠️")
                with hc3:
                    with st.expander(f"Review Details (#{q_id})"):
                        for d in item.get("details", []):
                            status_emoji = "✅" if d.get("is_correct") else "❌"
                            st.markdown(f"**{status_emoji} Q: {d.get('question')}**")
                            st.write(f"- Your Answer: `{d.get('chosen')}`")
                            st.write(f"- Explanation: {d.get('explanation')}")
                            st.markdown("---")

                st.markdown("<hr style='margin: 4px 0;'/>", unsafe_allow_html=True)

            if st.button("🗑️ Clear Past Quiz History", type="secondary"):
                clear_quiz_history()
                st.warning("Cleared all quiz history records.")
                st.rerun()

    # =========================================================================
    # TAB 3: WEAK TOPICS IDENTIFIED
    # =========================================================================
    with tab_weak:
        st.markdown("### 🎯 Weak Topics & Knowledge Gaps")
        st.caption("Topics where your quiz accuracy is below 60%. Automatically updated from your actual quiz answers.")

        weak_list = get_weak_topics(threshold=60.0)
        if not weak_list:
            st.success("🎉 **No weak topics detected!** All tested topics have an accuracy of 60% or higher.")
        else:
            for wt in weak_list:
                subj = wt["subject"]
                top = wt["topic"]
                tot = wt["total_questions"]
                cor = wt["correct_count"]
                inc = wt["incorrect_count"]
                acc = wt["accuracy_pct"]
                status = wt["status"]

                w_col1, w_col2, w_col3 = st.columns([4, 2, 2])
                with w_col1:
                    st.markdown(f"**{top}** (`{subj}`)")
                    st.caption(f"Total: {tot} Qs | Correct: {cor} | Incorrect: {inc}")
                with w_col2:
                    st.warning(f"Accuracy: **{acc}%** ({status})")
                with w_col3:
                    if st.button("📝 Retest Topic", key=f"retest_{top}"):
                        st.session_state.active_quiz = None
                        st.info(f"Retest queued for '{top}'. Return to the Take Quiz tab!")

                st.markdown("<hr style='margin: 4px 0;'/>", unsafe_allow_html=True)
