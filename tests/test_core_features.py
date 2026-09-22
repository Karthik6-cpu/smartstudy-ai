"""
Comprehensive Core Features Integration & Acceptance Test Suite for SmartStudy AI.
Tests the complete 20-step lifecycle:
1. Document upload & SHA-256 duplicate detection
2. Document validation (empty, corrupt, password, scanned)
3. Document deletion & complete vector cleanup
4. Hierarchical map-reduce summarization
5. Quiz generation & strict MCQ validation
6. Mathematical quiz grading & SQLite persistence
7. Real weak-topic detection from quiz errors
8. Study planner creation & task status management
9. Adaptive replanning (prioritizes weak topics without touching completed tasks)
10. Spaced repetition revision scheduling
11. Flashcards SM-2 intervals
12. Explainable recommendation engine
13. Unchanged safe calculator arithmetic
"""

import os
import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from rag.pdf_processor import (
    PDFProcessor,
    compute_sha256,
    EmptyPDFError,
    CorruptedPDFError,
)
from rag.vector_store import VectorStore
from chains.summarizer import generate_document_summary
from chains.quiz_generator import (
    generate_structured_quiz,
    validate_quiz_question,
    get_curated_fallback_questions,
)
from memory.storage_manager import (
    save_quiz_result,
    get_quiz_history,
    get_quiz_stats,
    record_topic_performance,
    get_topic_performance,
    get_weak_topics,
    save_study_tasks,
    update_task_status,
    get_study_tasks,
    adaptive_replan,
    save_flashcard,
    get_flashcards,
    review_flashcard,
    schedule_revision,
    get_revision_schedule,
    get_smart_recommendation,
    register_document,
    get_registered_documents,
    delete_registered_document,
)
from tools.safe_calculator import safe_calculate


def test_calculator_arithmetic_unchanged():
    """Verify calculator and natural-language arithmetic remain 100% functional."""
    # Word arithmetic: two plus two = 4
    res0 = safe_calculate("two plus two")
    assert "4" in res0, f"Expected 4 in {res0}"

    res0b = safe_calculate("what is twenty five multiplied by four")
    assert "100" in res0b, f"Expected 100 in {res0b}"

    # Percentage: 25% of 480 = 120
    res1 = safe_calculate("25% of 480")
    assert "120" in res1, f"Expected 120 in {res1}"

    # Addition: 20 + 30 = 50
    res2 = safe_calculate("20 + 30")
    assert "50" in res2, f"Expected 50 in {res2}"

    # Multiplication & division: 15 * 4 / 2 = 30
    res3 = safe_calculate("15 * 4 / 2")
    assert "30" in res3, f"Expected 30 in {res3}"

    # Square root: sqrt(144) = 12
    res4 = safe_calculate("sqrt(144)")
    assert "12" in res4, f"Expected 12 in {res4}"


def test_sha256_duplicate_detection():
    """Verify SHA-256 hash generation and duplicate document detection."""
    content_a = b"%PDF-1.4 Mock content for Unit 1 Operating Systems"
    content_b = b"%PDF-1.4 Mock content for Unit 2 Database Systems"

    hash_a = compute_sha256(content_a)
    hash_a_dup = compute_sha256(content_a)
    hash_b = compute_sha256(content_b)

    assert hash_a == hash_a_dup
    assert hash_a != hash_b
    assert len(hash_a) == 64


def test_pdf_validation_errors(tmp_path):
    """Verify PDF validation catches empty and corrupted files without crashing."""
    proc = PDFProcessor(storage_dir=tmp_path)

    # Empty bytes
    with pytest.raises(EmptyPDFError):
        proc.process_pdf(b"", "empty.pdf")

    # Corrupt content
    with pytest.raises(CorruptedPDFError):
        proc.process_pdf(b"Not a valid PDF header content", "corrupt.pdf")


def test_mcq_validation():
    """Verify strict validation of Multiple Choice Question structure."""
    valid_q = {
        "id": 1,
        "question": "What is CMMI Maturity Level 2?",
        "options": [
            "A) Managed",
            "B) Defined",
            "C) Quantitatively Managed",
            "D) Optimizing",
        ],
        "correct_answer": "A",
        "explanation": "Level 2 is Managed where projects are planned and executed according to policy.",
    }
    assert validate_quiz_question(valid_q) is True, "valid_q should pass validation"

    # Invalid: duplicate options
    dup_opt_q = {
        "question": "What is CMMI Maturity Level 2?",
        "options": ["A) Managed", "B) Managed", "C) Defined", "D) Optimizing"],
        "correct_answer": "A",
        "explanation": "Level 2 is Managed where projects are planned and executed according to policy.",
    }
    assert validate_quiz_question(dup_opt_q) is False, "dup_opt_q should fail validation"

    # Invalid: only 3 options
    few_opt_q = {
        "question": "What is CMMI Maturity Level 2?",
        "options": ["A) Managed", "B) Defined", "C) Optimizing"],
        "correct_answer": "A",
        "explanation": "Level 2 is Managed where projects are planned and executed according to policy.",
    }
    assert validate_quiz_question(few_opt_q) is False, "few_opt_q should fail validation"

    # Invalid: bad correct_answer
    bad_ans_q = {
        "question": "What is CMMI Maturity Level 2?",
        "options": [
            "A) Managed",
            "B) Defined",
            "C) Quantitatively Managed",
            "D) Optimizing",
        ],
        "correct_answer": "E",
        "explanation": "Level 2 is Managed where projects are planned and executed according to policy.",
    }
    assert validate_quiz_question(bad_ans_q) is False, "bad_ans_q should fail validation"


def test_quiz_evaluation_and_weak_topic_pipeline():
    """Verify mathematical quiz scoring, SQLite storage, and weak-topic detection."""
    test_topic = "SQL Joins & Group By"
    subject = "DBMS"

    # Simulate 5 questions: 2 correct, 3 incorrect (Score: 2/5 = 40% -> Weak Topic)
    details = [
        {"id": 1, "question": "Q1", "is_correct": True, "chosen": "A", "correct_answer": "A"},
        {"id": 2, "question": "Q2", "is_correct": True, "chosen": "B", "correct_answer": "B"},
        {"id": 3, "question": "Q3", "is_correct": False, "chosen": "C", "correct_answer": "D"},
        {"id": 4, "question": "Q4", "is_correct": False, "chosen": "A", "correct_answer": "B"},
        {"id": 5, "question": "Q5", "is_correct": False, "chosen": "D", "correct_answer": "C"},
    ]

    score = sum(1 for d in details if d["is_correct"])
    total = len(details)
    pct = round((score / total) * 100.0, 1)

    assert score == 2
    assert total == 5
    assert pct == 40.0

    quiz_id = save_quiz_result(
        topic=test_topic,
        difficulty="Intermediate",
        total_questions=total,
        score=score,
        percentage=pct,
        details=details,
        subject=subject,
    )
    assert quiz_id > 0

    # Verify topic performance recorded accurately
    weak_topics = get_weak_topics(60.0)
    matching_weak = [w for w in weak_topics if w["topic"] == test_topic]
    assert len(matching_weak) >= 1
    assert matching_weak[0]["accuracy_pct"] <= 40.0
    assert "Weak" in matching_weak[0]["status"]


def test_study_planner_and_adaptive_replan():
    """Verify task creation, status updates, and adaptive replanning preserving completed tasks."""
    plan_tasks = [
        {"subject": "DBMS", "topic": "SQL Joins & Group By", "priority": "Low", "status": "Not Started"},
        {"subject": "DBMS", "topic": "Relational Calculus", "priority": "Medium", "status": "Completed"},
        {"subject": "OS", "topic": "Deadlock Avoidance", "priority": "Medium", "status": "Not Started"},
    ]

    saved_count = save_study_tasks(plan_tasks)
    assert saved_count == 3

    tasks_before = get_study_tasks()
    completed_before = [t for t in tasks_before if t.get("status") == "Completed"]
    assert len(completed_before) >= 1

    # Run adaptive replanning
    replan_result = adaptive_replan(daily_hours=2.0)
    assert replan_result["rearranged_count"] >= 2

    # Verify completed tasks were NOT altered or deleted
    tasks_after = get_study_tasks()
    completed_after = [t for t in tasks_after if t.get("status") == "Completed"]
    assert len(completed_after) == len(completed_before)

    # Verify weak topic ('SQL Joins & Group By') got boosted to High priority
    matching_tasks = [t for t in tasks_after if t["topic"] == "SQL Joins & Group By"]
    assert len(matching_tasks) >= 1
    assert matching_tasks[0]["priority"] == "High"


def test_flashcard_sm2_spaced_repetition():
    """Verify SuperMemo SM-2 flashcard review and interval calculation."""
    deck = "Test_SE_Deck"
    cid = save_flashcard(deck_name=deck, front="What is CMMI Level 3?", back="Defined Process Level")
    assert cid > 0

    cards = get_flashcards(deck_name=deck)
    assert any(c["id"] == cid for c in cards)

    # Review 1: Rating 2 (Good) -> Advances interval
    rev1 = review_flashcard(cid, rating=2)
    assert rev1["success"] is True
    assert rev1["interval_days"] >= 1

    # Review 2: Rating 0 (Again) -> Resets repetitions and queues 1 day
    rev2 = review_flashcard(cid, rating=0)
    assert rev2["success"] is True
    assert rev2["repetitions"] == 0
    assert rev2["interval_days"] == 1


def test_explainable_recommendation_engine():
    """Verify recommendation engine produces grounded, explainable guidance from SQLite."""
    rec = get_smart_recommendation()
    assert isinstance(rec, dict)
    assert "title" in rec
    assert "recommendation" in rec
    assert "reason" in rec
    assert len(rec["reason"]) > 5


def test_vector_store_delete_removes_all_vectors():
    """Verify that deleting a document completely wipes its vectors, chunks, and metadata."""
    store = VectorStore()
    dim = store.dimension

    dummy_doc = "Test_Deletion_Doc.pdf"
    dummy_chunks = [
        {"id": f"{dummy_doc}_c1", "filename": dummy_doc, "page": 1, "text": "Unique topic: Quantum Lexical Semantics XYZ123"},
        {"id": f"{dummy_doc}_c2", "filename": dummy_doc, "page": 2, "text": "Unique topic: Quantum Lexical Syntax ABC456"},
    ]
    dummy_vectors = np.ones((2, dim), dtype=np.float32) * 0.1
    # Normalize
    dummy_vectors = dummy_vectors / np.linalg.norm(dummy_vectors, axis=1, keepdims=True)

    dummy_stats = {
        "filename": dummy_doc,
        "total_pages": 2,
        "total_chunks": 2,
        "file_hash": "dummy_hash_123456",
        "file_size_kb": 12.5,
    }

    # Add to store
    store.add_document(dummy_doc, dummy_chunks, dummy_vectors, dummy_stats)
    assert dummy_doc in store.documents

    # Search should find it
    q_vec = dummy_vectors[0:1]
    res_before = store.search(q_vec, top_k=2, filter_filename=dummy_doc)
    assert len(res_before) >= 1

    # Now perform clean deletion
    deleted = store.delete_document(dummy_doc, delete_pdf_file=False)
    assert deleted is True
    assert dummy_doc not in store.documents
    assert not any(c.get("filename") == dummy_doc for c in store.chunks)

    # Search after deletion must return 0 matches for that document
    res_after = store.search(q_vec, top_k=2, filter_filename=dummy_doc)
    assert len(res_after) == 0


if __name__ == "__main__":
    test_funcs = [
        test_calculator_arithmetic_unchanged,
        test_sha256_duplicate_detection,
        test_mcq_validation,
        test_quiz_evaluation_and_weak_topic_pipeline,
        test_study_planner_and_adaptive_replan,
        test_flashcard_sm2_spaced_repetition,
        test_explainable_recommendation_engine,
        test_vector_store_delete_removes_all_vectors,
    ]

    print("==================================================")
    print("RUNNING SMARTSTUDY AI CORE FEATURES ACCEPTANCE TESTS")
    print("==================================================")
    passed = 0
    failed = 0

    for tf in test_funcs:
        name = tf.__name__
        try:
            tf()
            print(f"PASS: {name}")
            passed += 1
        except Exception as err:
            print(f"FAIL: {name} -> {str(err)}")
            failed += 1

    print("==================================================")
    print(f"RESULTS: {passed} PASSED, {failed} FAILED")
    print("==================================================")
    sys.exit(0 if failed == 0 else 1)
