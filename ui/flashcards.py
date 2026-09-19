"""
Interactive Flashcard & Spaced Repetition Page for SmartStudy AI.
Enables deck creation from uploaded PDF notes, interactive 3D card flip review,
and SuperMemo SM-2 interval scheduling with automatic revision queueing.
"""

from datetime import date
import streamlit as st
from rag.vector_store import VectorStore
from memory.storage_manager import (
    save_flashcard,
    get_flashcards,
    get_flashcard_decks,
    review_flashcard,
    delete_flashcard,
    clear_deck,
)


def render_flashcards_page():
    """Render the Interactive Flashcards Study & Spaced Repetition view."""
    st.markdown("## 📇 Interactive Flashcards & Spaced Repetition")
    st.caption("Active recall and SuperMemo SM-2 intervals grounded in your lecture notes.")

    decks = get_flashcard_decks()

    tab_study, tab_create, tab_manage = st.tabs([
        "🧠 Study Decks",
        "➕ Create Flashcards",
        "⚙️ Manage Decks",
    ])

    # =========================================================================
    # TAB 1: STUDY FLASHCARDS
    # =========================================================================
    with tab_study:
        if not decks:
            st.info("No flashcard decks found! Create your first deck in the **'➕ Create Flashcards'** tab or from your 📚 Documents.")
        else:
            deck_col1, deck_col2 = st.columns([3, 1])
            with deck_col1:
                selected_deck = st.selectbox("Select Flashcard Deck:", options=decks)

            cards = get_flashcards(deck_name=selected_deck)

            if not cards:
                st.warning(f"Deck '{selected_deck}' has no cards.")
            else:
                total_cards = len(cards)

                if "card_index" not in st.session_state:
                    st.session_state.card_index = 0
                if "show_back" not in st.session_state:
                    st.session_state.show_back = False

                idx = st.session_state.card_index
                if idx >= total_cards:
                    idx = 0
                    st.session_state.card_index = 0

                card = cards[idx]

                st.progress((idx + 1) / total_cards)
                st.caption(f"Card {idx + 1} of {total_cards} | Source: `{card.get('source_doc', 'General')}` | Next Due: `{card.get('next_review_due', 'Today')}`")

                # Flashcard Box
                with st.container():
                    st.markdown(
                        f"""
                        <div style="background-color: #1E293B; border: 2px solid #3B82F6; border-radius: 12px; padding: 24px; min-height: 180px; margin: 12px 0;">
                            <h4 style="color: #94A3B8; margin-top: 0;">FRONT (QUESTION):</h4>
                            <p style="font-size: 18px; font-weight: 600; color: #F8FAFC;">{card['front']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if st.session_state.show_back:
                        st.markdown(
                            f"""
                            <div style="background-color: #0F172A; border: 2px solid #10B981; border-radius: 12px; padding: 24px; min-height: 180px; margin: 12px 0;">
                                <h4 style="color: #10B981; margin-top: 0;">BACK (EXPLANATION & ANSWER):</h4>
                                <p style="font-size: 16px; color: #E2E8F0; line-height: 1.6;">{card['back']}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                # Flip Button
                f_col1, f_col2, f_col3 = st.columns([1, 2, 1])
                with f_col2:
                    flip_label = "🙈 Hide Answer" if st.session_state.show_back else "🔄 Show Answer / Flip Card"
                    if st.button(flip_label, use_container_width=True, type="primary"):
                        st.session_state.show_back = not st.session_state.show_back
                        st.rerun()

                # Rating Buttons (SM-2 Spaced Repetition)
                if st.session_state.show_back:
                    st.markdown("---")
                    st.markdown("**How well did you recall this concept? (SM-2 Interval Scheduling):**")
                    r0, r1, r2, r3 = st.columns(4)

                    with r0:
                        if st.button("🔴 Again (0)", help="Failed recall - resets to 1 day and queues for revision", use_container_width=True):
                            review_flashcard(card["id"], rating=0)
                            st.session_state.show_back = False
                            st.session_state.card_index = (idx + 1) % total_cards
                            st.rerun()

                    with r1:
                        if st.button("🟠 Hard (1)", help="Difficult recall - 1 day interval, scheduled for review", use_container_width=True):
                            review_flashcard(card["id"], rating=1)
                            st.session_state.show_back = False
                            st.session_state.card_index = (idx + 1) % total_cards
                            st.rerun()

                    with r2:
                        if st.button("🟢 Good (2)", help="Standard successful recall - advances interval", use_container_width=True):
                            review_flashcard(card["id"], rating=2)
                            st.session_state.show_back = False
                            st.session_state.card_index = (idx + 1) % total_cards
                            st.rerun()

                    with r3:
                        if st.button("🔵 Easy (3)", help="Instant recall - multiplies interval by ease factor", use_container_width=True):
                            review_flashcard(card["id"], rating=3)
                            st.session_state.show_back = False
                            st.session_state.card_index = (idx + 1) % total_cards
                            st.rerun()

                # Navigation (Prev / Next)
                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                p_col1, p_col2 = st.columns(2)
                with p_col1:
                    if st.button("⬅️ Previous Card", disabled=(idx == 0), use_container_width=True):
                        st.session_state.card_index = max(0, idx - 1)
                        st.session_state.show_back = False
                        st.rerun()
                with p_col2:
                    if st.button("Next Card ➡️", disabled=(idx == total_cards - 1), use_container_width=True):
                        st.session_state.card_index = min(total_cards - 1, idx + 1)
                        st.session_state.show_back = False
                        st.rerun()

    # =========================================================================
    # TAB 2: CREATE FLASHCARDS
    # =========================================================================
    with tab_create:
        st.subheader("➕ Create New Flashcards")
        c_mode = st.radio("Creation Mode:", ["From Uploaded Document", "Manual Entry"], horizontal=True)

        if c_mode == "From Uploaded Document":
            vector_store = VectorStore()
            doc_summaries = vector_store.get_documents_summary()
            doc_names = [d["filename"] for d in doc_summaries]

            if not doc_names:
                st.info("No documents uploaded yet. Upload a PDF in 📚 Documents first.")
            else:
                sel_doc = st.selectbox("Select Source PDF:", doc_names)
                deck_input = st.text_input("Deck Name:", value=sel_doc.replace(".pdf", ""))
                card_count = st.slider("Number of cards to generate:", 3, 10, 5)

                if st.button("✨ Auto-Generate Flashcards from PDF", type="primary", use_container_width=True):
                    chunks = vector_store.get_document_chunks(sel_doc)
                    created = 0
                    for c in chunks[:card_count]:
                        txt = c.get("text", "")
                        lines = [ln.strip() for ln in txt.split("\n") if len(ln.strip()) > 20]
                        if len(lines) >= 2:
                            front = f"What is explained regarding: {lines[0][:80]}?"
                            back = "\n".join(lines[1:3])[:300]
                            save_flashcard(deck_name=deck_input, front=front, back=back, source_doc=sel_doc)
                            created += 1

                    if created > 0:
                        st.success(f"🎉 Generated {created} flashcards in deck '{deck_input}'! Ready to study.")
                        st.rerun()
                    else:
                        st.warning("Could not extract enough text lines from document chunks.")

        else:
            deck_name = st.text_input("Deck Name:", value="Computer Science Core")
            front_text = st.text_area("Front (Question or Concept):", placeholder="e.g. What are Coffman's four conditions for Deadlock?")
            back_text = st.text_area("Back (Explanation or Answer):", placeholder="e.g. 1. Mutual Exclusion, 2. Hold and Wait, 3. No Preemption, 4. Circular Wait.")

            if st.button("💾 Save Flashcard", type="primary", use_container_width=True):
                if not front_text.strip() or not back_text.strip():
                    st.error("Both Front and Back must be provided.")
                else:
                    cid = save_flashcard(deck_name=deck_name, front=front_text, back=back_text)
                    st.success(f"Saved flashcard #{cid} to deck '{deck_name}'!")
                    st.rerun()

    # =========================================================================
    # TAB 3: MANAGE DECKS
    # =========================================================================
    with tab_manage:
        st.subheader("⚙️ Manage Flashcard Decks")
        if not decks:
            st.caption("No decks to manage.")
        else:
            for d in decks:
                cards = get_flashcards(deck_name=d)
                m_col1, m_col2, m_col3 = st.columns([5, 2, 2])
                with m_col1:
                    st.markdown(f"**Deck: `{d}`** ({len(cards)} cards)")
                with m_col2:
                    if st.button("🗑️ Clear Deck", key=f"clr_{d}", type="secondary"):
                        clear_deck(d)
                        st.success(f"Cleared deck '{d}'.")
                        st.rerun()
                with m_col3:
                    st.caption("Active")
                st.markdown("<hr style='margin: 4px 0;'/>", unsafe_allow_html=True)
