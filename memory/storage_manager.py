"""
Unified SQLite Storage Manager for SmartStudy AI.
Persists quiz history, study planner tasks, task progress statuses,
and student activity logs across application restarts.
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from config.settings import MEMORY_DIR


DB_PATH = MEMORY_DIR / "study_data.db"


def _get_connection() -> sqlite3.Connection:
    """Initialize database directory and create required tables."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    with conn:
        # 1. Quiz History Table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS quiz_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                document_name TEXT,
                total_questions INTEGER NOT NULL,
                score INTEGER NOT NULL,
                percentage REAL NOT NULL,
                details_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # 2. Study Tasks Table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS study_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_id TEXT,
                subject TEXT NOT NULL,
                topic TEXT NOT NULL,
                priority TEXT DEFAULT 'Medium',
                status TEXT DEFAULT 'Not Started',
                exam_date TEXT,
                allocated_hours REAL DEFAULT 2.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # 3. Activity Log Table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                activity_type TEXT NOT NULL,
                description TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    return conn


# --- QUIZ STORAGE METHODS ---

def save_quiz_result(
    topic: str,
    difficulty: str,
    total_questions: int,
    score: int,
    percentage: float,
    details: List[Dict[str, Any]],
    document_name: Optional[str] = None,
) -> int:
    """Save completed quiz assessment to SQLite history."""
    conn = _get_connection()
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO quiz_history (
                    topic, difficulty, document_name, total_questions,
                    score, percentage, details_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    topic,
                    difficulty,
                    document_name or "General",
                    total_questions,
                    score,
                    percentage,
                    json.dumps(details, ensure_ascii=False),
                ),
            )
            quiz_id = cursor.lastrowid

        log_activity(
            activity_type="Quiz Completed",
            description=f"Scored {score}/{total_questions} ({percentage:.1f}%) in '{topic}' ({difficulty})",
        )
        return quiz_id
    finally:
        conn.close()


def get_quiz_history(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve past quiz records."""
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, topic, difficulty, document_name, total_questions,
                   score, percentage, details_json, created_at
            FROM quiz_history
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        results = []
        for r in rows:
            d = dict(r)
            try:
                d["details"] = json.loads(d.get("details_json") or "[]")
            except Exception:
                d["details"] = []
            results.append(d)
        return results
    finally:
        conn.close()


def get_quiz_stats() -> Dict[str, Any]:
    """Calculate aggregate quiz performance statistics."""
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                COUNT(*) as total_quizzes,
                AVG(percentage) as avg_score,
                MAX(percentage) as best_score,
                SUM(total_questions) as total_questions_answered
            FROM quiz_history
            """
        )
        row = cursor.fetchone()
        return {
            "total_quizzes": row["total_quizzes"] or 0,
            "avg_score": round(row["avg_score"] or 0.0, 1),
            "best_score": round(row["best_score"] or 0.0, 1),
            "total_questions_answered": row["total_questions_answered"] or 0,
        }
    finally:
        conn.close()


def clear_quiz_history() -> bool:
    """Clear all quiz history records."""
    conn = _get_connection()
    try:
        with conn:
            conn.execute("DELETE FROM quiz_history")
            log_activity("Quiz History Cleared", "User cleared all past quiz records.")
            return True
    finally:
        conn.close()


# --- STUDY TASKS & PLANNER METHODS ---

def save_study_tasks(tasks: List[Dict[str, Any]]) -> int:
    """Save a list of parsed study tasks from a newly generated study plan."""
    conn = _get_connection()
    saved_count = 0
    try:
        with conn:
            for t in tasks:
                conn.execute(
                    """
                    INSERT INTO study_tasks (
                        plan_id, subject, topic, priority, status,
                        exam_date, allocated_hours, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """,
                    (
                        t.get("plan_id", "default_plan"),
                        t.get("subject", "General MCA"),
                        t.get("topic", "Topic"),
                        t.get("priority", "Medium"),
                        t.get("status", "Not Started"),
                        t.get("exam_date", "TBD"),
                        t.get("allocated_hours", 2.0),
                    ),
                )
                saved_count += 1

        log_activity(
            activity_type="Study Plan Created",
            description=f"Generated and saved {saved_count} structured study tasks.",
        )
        return saved_count
    finally:
        conn.close()


def update_task_status(task_id: int, new_status: str) -> bool:
    """Update progress status ('Not Started', 'In Progress', 'Completed') of a task."""
    conn = _get_connection()
    try:
        with conn:
            cursor = conn.execute(
                """
                UPDATE study_tasks
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (new_status, task_id),
            )
            updated = cursor.rowcount > 0

        if updated and new_status == "Completed":
            # Fetch topic name for logging
            c = conn.cursor()
            c.execute("SELECT topic FROM study_tasks WHERE id = ?", (task_id,))
            row = c.fetchone()
            topic_name = row["topic"] if row else f"Task #{task_id}"
            log_activity("Topic Completed", f"Marked '{topic_name}' as Completed ✅")

        return updated
    finally:
        conn.close()


def get_study_tasks(status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve all study tasks, optionally filtered by status."""
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        if status_filter:
            cursor.execute(
                "SELECT * FROM study_tasks WHERE status = ? ORDER BY id ASC",
                (status_filter,),
            )
        else:
            cursor.execute("SELECT * FROM study_tasks ORDER BY id ASC")
        return [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()


def get_study_analytics() -> Dict[str, Any]:
    """Calculate topic progress, completion percentages, and subject distribution."""
    conn = _get_connection()
    try:
        cursor = conn.cursor()

        # Total and status counts
        cursor.execute(
            """
            SELECT
                COUNT(*) as total_tasks,
                SUM(CASE WHEN status = 'Completed' THEN 1 ELSE 0 END) as completed_tasks,
                SUM(CASE WHEN status = 'In Progress' THEN 1 ELSE 0 END) as in_progress_tasks,
                SUM(CASE WHEN status = 'Not Started' THEN 1 ELSE 0 END) as not_started_tasks,
                SUM(allocated_hours) as total_hours
            FROM study_tasks
            """
        )
        row = cursor.fetchone()
        total = row["total_tasks"] or 0
        completed = row["completed_tasks"] or 0
        in_progress = row["in_progress_tasks"] or 0
        not_started = row["not_started_tasks"] or 0
        hours = row["total_hours"] or 0.0

        percent = round((completed / total) * 100.0, 1) if total > 0 else 0.0

        # Subject breakdown
        cursor.execute(
            """
            SELECT subject, COUNT(*) as count,
                   SUM(CASE WHEN status = 'Completed' THEN 1 ELSE 0 END) as completed
            FROM study_tasks
            GROUP BY subject
            """
        )
        subject_data = [dict(r) for r in cursor.fetchall()]

        return {
            "total_tasks": total,
            "completed_tasks": completed,
            "in_progress_tasks": in_progress,
            "not_started_tasks": not_started,
            "completion_percentage": percent,
            "total_hours": round(hours, 1),
            "subjects": subject_data,
        }
    finally:
        conn.close()


def delete_task(task_id: int) -> bool:
    """Delete an individual study task."""
    conn = _get_connection()
    try:
        with conn:
            cursor = conn.execute("DELETE FROM study_tasks WHERE id = ?", (task_id,))
            return cursor.rowcount > 0
    finally:
        conn.close()


def clear_all_tasks() -> bool:
    """Wipe all study tasks."""
    conn = _get_connection()
    try:
        with conn:
            conn.execute("DELETE FROM study_tasks")
            log_activity("Study Plan Cleared", "User cleared all planned study tasks.")
            return True
    finally:
        conn.close()


# --- ACTIVITY LOG METHODS ---

def log_activity(activity_type: str, description: str):
    """Record an action into the student audit timeline."""
    conn = _get_connection()
    try:
        with conn:
            conn.execute(
                "INSERT INTO activity_log (activity_type, description) VALUES (?, ?)",
                (activity_type, description),
            )
    finally:
        conn.close()


def get_recent_activities(limit: int = 15) -> List[Dict[str, Any]]:
    """Retrieve recent student activities in descending order."""
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, activity_type, description, created_at FROM activity_log ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()
