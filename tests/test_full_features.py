"""
Automated Test Suite for SmartStudy AI Persistent Features.
Tests Quiz Result Storage, Study Planner Task State Transitions,
Learning Analytics Calculations, Activity Timeline Logging,
and Structured Quiz Generator with Fallbacks.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memory.storage_manager import (
    save_quiz_result,
    get_quiz_history,
    get_quiz_stats,
    clear_quiz_history,
    save_study_tasks,
    update_task_status,
    get_study_tasks,
    get_study_analytics,
    delete_task,
    clear_all_tasks,
    log_activity,
    get_recent_activities,
)
from chains.quiz_generator import (
    generate_structured_quiz,
    parse_quiz_json,
    get_curated_fallback_questions,
)


def test_quiz_storage_and_stats():
    """Verify quiz score saving, percentage calculation, and history retrieval."""
    print("Testing Quiz Storage & Statistics...")
    clear_quiz_history()

    sample_details = [
        {"id": 1, "question": "What is CMMI Level 2?", "chosen": "B) Managed", "correct_answer": "B", "is_correct": True, "explanation": "Managed processes."},
        {"id": 2, "question": "What is CMMI Level 5?", "chosen": "A) Initial", "correct_answer": "D", "is_correct": False, "explanation": "Optimizing."},
    ]

    # Save a quiz
    qid = save_quiz_result(
        topic="Software Engineering: CMMI",
        difficulty="Intermediate",
        total_questions=2,
        score=1,
        percentage=50.0,
        details=sample_details,
        document_name="MCA_SE_Notes.pdf",
    )
    assert qid > 0

    # Retrieve history
    history = get_quiz_history()
    assert len(history) >= 1
    latest = history[0]
    assert latest["topic"] == "Software Engineering: CMMI"
    assert latest["score"] == 1
    assert latest["percentage"] == 50.0
    assert latest["document_name"] == "MCA_SE_Notes.pdf"
    assert len(latest["details"]) == 2

    # Stats calculation
    stats = get_quiz_stats()
    assert stats["total_quizzes"] >= 1
    assert stats["avg_score"] == 50.0

    print(" Quiz Storage & Statistics passed.")


def test_study_planner_tasks_and_analytics():
    """Verify task saving, status transitions (Not Started -> In Progress -> Completed), and analytics."""
    print("Testing Study Tasks & Progress Analytics...")
    clear_all_tasks()

    new_tasks = [
        {"plan_id": "p1", "subject": "DSA", "topic": "Binary Search Trees", "priority": "High", "status": "Not Started", "allocated_hours": 3.0},
        {"plan_id": "p1", "subject": "OS", "topic": "Process Scheduling", "priority": "Medium", "status": "Not Started", "allocated_hours": 2.5},
        {"plan_id": "p1", "subject": "DBMS", "topic": "Normalization 1NF-BCNF", "priority": "High", "status": "Not Started", "allocated_hours": 4.0},
    ]

    count = save_study_tasks(new_tasks)
    assert count == 3

    tasks = get_study_tasks()
    assert len(tasks) == 3

    # Check initial analytics
    a1 = get_study_analytics()
    assert a1["total_tasks"] == 3
    assert a1["completed_tasks"] == 0
    assert a1["completion_percentage"] == 0.0

    # Update task 1 to In Progress
    task1_id = tasks[0]["id"]
    ok_ip = update_task_status(task1_id, "In Progress")
    assert ok_ip is True

    # Update task 2 to Completed
    task2_id = tasks[1]["id"]
    ok_cp = update_task_status(task2_id, "Completed")
    assert ok_cp is True

    # Verify updated analytics
    a2 = get_study_analytics()
    assert a2["total_tasks"] == 3
    assert a2["completed_tasks"] == 1
    assert a2["in_progress_tasks"] == 1
    assert a2["not_started_tasks"] == 1
    assert round(a2["completion_percentage"], 1) == 33.3

    print(" Study Tasks & Progress Analytics passed.")


def test_activity_logging():
    """Verify chronological activity logging and retrieval."""
    print("Testing Activity Logging...")
    log_activity("Test Event", "Student completed revision session.")
    acts = get_recent_activities(limit=5)
    assert len(acts) >= 1
    assert acts[0]["activity_type"] == "Test Event"
    assert "Student completed" in acts[0]["description"]
    print(" Activity Logging passed.")


def test_quiz_generator_parsing_and_fallbacks():
    """Verify structured quiz JSON parsing and curated fallbacks."""
    print("Testing Quiz Generator parsing and fallback question banks...")

    # 1. Test JSON parsing
    sample_json = """
    [
      {
        "id": 1,
        "question": "What is the complexity of Quicksort?",
        "options": ["A) O(N)", "B) O(N log N)", "C) O(N^2)", "D) O(1)"],
        "correct_answer": "B",
        "explanation": "Average case is O(N log N)."
      }
    ]
    """
    parsed = parse_quiz_json(sample_json)
    assert parsed is not None
    assert len(parsed) == 1
    assert parsed[0]["correct_answer"] == "B"

    # 2. Test fallback question banks for MCA topics
    dsa_qs = get_curated_fallback_questions("Data Structures", count=3)
    assert len(dsa_qs) == 3
    assert "Binary Search" in dsa_qs[0]["question"]

    os_qs = get_curated_fallback_questions("Operating Systems", count=2)
    assert len(os_qs) == 2
    assert "Deadlock" in os_qs[0]["question"]

    se_qs = get_curated_fallback_questions("CMMI Software Engineering", count=2)
    assert len(se_qs) == 2
    assert "CMMI" in se_qs[0]["question"]

    # 3. Test generate_structured_quiz fallback behavior
    quiz = generate_structured_quiz("Operating Systems", num_questions=3, difficulty="Beginner", base_url="http://localhost:59999")
    assert len(quiz) == 3
    assert "options" in quiz[0]
    assert len(quiz[0]["options"]) == 4

    print(" Quiz Generator parsing & fallbacks passed.")


if __name__ == "__main__":
    test_quiz_storage_and_stats()
    test_study_planner_tasks_and_analytics()
    test_activity_logging()
    test_quiz_generator_parsing_and_fallbacks()
    print("\nAll Quiz, Study Planner, Memory & Progress tests passed successfully! [SUCCESS]")
