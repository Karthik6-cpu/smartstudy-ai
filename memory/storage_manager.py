"""
Unified SQLite Storage Manager for SmartStudy AI.
Persists quiz history, study planner tasks, task progress statuses,
student activity logs, weak-topic performance, revision schedules,
flashcard decks with SM-2 spaced repetition, and document registry.
Uses WAL journal mode and busy_timeout for multi-process safety.
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, date, timedelta
from config.settings import MEMORY_DIR


DB_PATH = MEMORY_DIR / "study_data.db"


def _get_connection() -> sqlite3.Connection:
    """Initialize database directory and create/migrate required tables with WAL mode."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
    except Exception:
        pass

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
                time_taken_seconds INTEGER DEFAULT 0,
                details_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Migration for quiz_history columns
        try:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(quiz_history)")
            q_cols = [r[1] for r in cursor.fetchall()]
            if "time_taken_seconds" not in q_cols:
                conn.execute("ALTER TABLE quiz_history ADD COLUMN time_taken_seconds INTEGER DEFAULT 0")
        except Exception:
            pass

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
                task_date TEXT,
                task_type TEXT DEFAULT 'Learning',
                allocated_hours REAL DEFAULT 2.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Migration for study_tasks columns
        try:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(study_tasks)")
            t_cols = [r[1] for r in cursor.fetchall()]
            if "task_date" not in t_cols:
                conn.execute("ALTER TABLE study_tasks ADD COLUMN task_date TEXT")
            if "task_type" not in t_cols:
                conn.execute("ALTER TABLE study_tasks ADD COLUMN task_type TEXT DEFAULT 'Learning'")
        except Exception:
            pass

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

        # 4. Topic Performance Table (Weak-Topic Tracking)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS topic_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                topic TEXT NOT NULL,
                total_questions INTEGER DEFAULT 0,
                correct_count INTEGER DEFAULT 0,
                incorrect_count INTEGER DEFAULT 0,
                accuracy_pct REAL DEFAULT 0.0,
                status TEXT DEFAULT 'Needs Revision',
                last_attempt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(subject, topic)
            )
            """
        )

        # 5. Revision Schedule Table (Spaced Repetition)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS revision_schedule (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                topic TEXT NOT NULL,
                due_date TEXT NOT NULL,
                interval_days INTEGER DEFAULT 1,
                status TEXT DEFAULT 'Due Today',
                source_type TEXT DEFAULT 'Quiz',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # 6. Flashcards Table (SuperMemo SM-2 Spaced Repetition)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS flashcards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                deck_name TEXT NOT NULL,
                source_doc TEXT DEFAULT 'General',
                front TEXT NOT NULL,
                back TEXT NOT NULL,
                difficulty_rating INTEGER DEFAULT 0,
                repetitions INTEGER DEFAULT 0,
                interval_days INTEGER DEFAULT 1,
                ease_factor REAL DEFAULT 2.5,
                next_review_due TEXT,
                last_reviewed TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # 7. Document Registry Table (Duplicate Detection & File Status)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS document_registry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL UNIQUE,
                file_hash TEXT NOT NULL,
                file_size_kb REAL DEFAULT 0.0,
                page_count INTEGER DEFAULT 0,
                chunk_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'Ready',
                error_msg TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    return conn


# --- QUIZ STORAGE & WEAK TOPICS METHODS ---

def save_quiz_result(
    topic: str,
    difficulty: str,
    total_questions: int,
    score: int,
    percentage: float,
    details: List[Dict[str, Any]],
    document_name: Optional[str] = None,
    time_taken_seconds: int = 0,
    subject: str = "General",
) -> int:
    """Save completed quiz assessment to SQLite history and update topic performance."""
    conn = _get_connection()
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO quiz_history (
                    topic, difficulty, document_name, total_questions,
                    score, percentage, time_taken_seconds, details_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    topic,
                    difficulty,
                    document_name or "General",
                    total_questions,
                    score,
                    percentage,
                    time_taken_seconds,
                    json.dumps(details, ensure_ascii=False),
                ),
            )
            quiz_id = cursor.lastrowid
    finally:
        conn.close()

    # Update topic performance and weak topic tracker after conn is closed
    incorrect = total_questions - score
    record_topic_performance(
        subject=subject,
        topic=topic,
        total_q=total_questions,
        correct_q=score,
        incorrect_q=incorrect,
    )

    # If score is low (< 60%), automatically queue for spaced revision
    if percentage < 60.0:
        schedule_revision(
            subject=subject,
            topic=topic,
            interval_days=1,
            status="Due Today",
            source_type=f"Quiz ({percentage:.0f}%)",
        )

    log_activity(
        activity_type="Quiz Completed",
        description=f"Scored {score}/{total_questions} ({percentage:.1f}%) in '{topic}' ({difficulty})",
    )
    return quiz_id


def get_quiz_history(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve past quiz records."""
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, topic, difficulty, document_name, total_questions,
                   score, percentage, time_taken_seconds, details_json, created_at
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
            conn.execute("DELETE FROM topic_performance")
            log_activity("Quiz History Cleared", "User cleared all past quiz and topic records.", conn=conn)
            return True
    finally:
        conn.close()


# --- TOPIC PERFORMANCE & WEAK AREAS ---

def record_topic_performance(
    subject: str,
    topic: str,
    total_q: int,
    correct_q: int,
    incorrect_q: int,
):
    """Update topic accuracy metrics from quiz results."""
    conn = _get_connection()
    clean_sub = (subject or "General").strip()
    clean_top = (topic or "General").strip()

    try:
        with conn:
            cursor = conn.execute(
                "SELECT * FROM topic_performance WHERE subject = ? AND topic = ?",
                (clean_sub, clean_top),
            )
            existing = cursor.fetchone()

            if existing:
                new_tot = existing["total_questions"] + total_q
                new_cor = existing["correct_count"] + correct_q
                new_inc = existing["incorrect_count"] + incorrect_q
                new_acc = round((new_cor / new_tot * 100.0), 1) if new_tot > 0 else 0.0
                status = "Weak ⚠️" if new_acc < 50.0 else ("Needs Revision ⏳" if new_acc < 75.0 else "Mastered ✅")

                conn.execute(
                    """
                    UPDATE topic_performance
                    SET total_questions = ?, correct_count = ?, incorrect_count = ?,
                        accuracy_pct = ?, status = ?, last_attempt = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (new_tot, new_cor, new_inc, new_acc, status, existing["id"]),
                )
            else:
                acc = round((correct_q / total_q * 100.0), 1) if total_q > 0 else 0.0
                status = "Weak ⚠️" if acc < 50.0 else ("Needs Revision ⏳" if acc < 75.0 else "Mastered ✅")
                conn.execute(
                    """
                    INSERT INTO topic_performance (
                        subject, topic, total_questions, correct_count,
                        incorrect_count, accuracy_pct, status, last_attempt
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (clean_sub, clean_top, total_q, correct_q, incorrect_q, acc, status),
                )
    finally:
        conn.close()


def get_topic_performance() -> List[Dict[str, Any]]:
    """Retrieve all tracked topics sorted by accuracy (weakest first)."""
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, subject, topic, total_questions, correct_count,
                   incorrect_count, accuracy_pct, status, last_attempt
            FROM topic_performance
            ORDER BY accuracy_pct ASC, total_questions DESC
            """
        )
        return [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()


def get_weak_topics(threshold: float = 60.0) -> List[Dict[str, Any]]:
    """Retrieve topics where accuracy percentage is below threshold."""
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, subject, topic, total_questions, correct_count,
                   incorrect_count, accuracy_pct, status, last_attempt
            FROM topic_performance
            WHERE accuracy_pct < ?
            ORDER BY accuracy_pct ASC
            """,
            (threshold,),
        )
        return [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()


# --- SMART EXPLAINABLE RECOMMENDATION ENGINE ---

def get_smart_recommendation() -> Dict[str, str]:
    """
    Generate an explainable next-action recommendation strictly from SQLite data.
    Does NOT fabricate facts or scores.
    """
    conn = _get_connection()
    try:
        # Check weak topics
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT topic, subject, accuracy_pct, total_questions
            FROM topic_performance
            WHERE accuracy_pct < 60.0
            ORDER BY accuracy_pct ASC
            LIMIT 1
            """
        )
        weak = cursor.fetchone()
        if weak:
            return {
                "title": f"🎯 Revise Weak Topic: {weak['topic']}",
                "recommendation": f"Spend 30 minutes revising '{weak['topic']}' in {weak['subject']}.",
                "reason": f"Your accuracy on '{weak['topic']}' is currently {weak['accuracy_pct']}% across {weak['total_questions']} questions.",
                "action": "Revise Topic",
                "topic": weak["topic"],
            }

        # Check overdue revision items
        today_str = date.today().isoformat()
        cursor.execute(
            """
            SELECT topic, subject, due_date
            FROM revision_schedule
            WHERE due_date <= ? AND status != 'Completed'
            ORDER BY due_date ASC
            LIMIT 1
            """,
            (today_str,),
        )
        rev = cursor.fetchone()
        if rev:
            return {
                "title": f"🔄 Spaced Repetition Due: {rev['topic']}",
                "recommendation": f"Review '{rev['topic']}' for 15 minutes today.",
                "reason": f"Scheduled spaced repetition is due today ({rev['due_date']}) to strengthen retention.",
                "action": "Start Revision",
                "topic": rev["topic"],
            }

        # Check pending study tasks
        cursor.execute(
            """
            SELECT topic, subject, priority
            FROM study_tasks
            WHERE status != 'Completed'
            ORDER BY CASE priority WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END ASC, id ASC
            LIMIT 1
            """
        )
        task = cursor.fetchone()
        if task:
            return {
                "title": f"📅 Next Planned Topic: {task['topic']}",
                "recommendation": f"Focus on '{task['topic']}' ({task['subject']}). Priority: {task['priority']}.",
                "reason": "This is your highest priority uncompleted task in your study planner.",
                "action": "Study Topic",
                "topic": task["topic"],
            }

        # Check total quizzes taken
        cursor.execute("SELECT COUNT(*) as cnt FROM quiz_history")
        q_cnt = cursor.fetchone()["cnt"]
        if q_cnt == 0:
            return {
                "title": "📝 Calibrate Your Baseline",
                "recommendation": "Take a 5-question diagnostic quiz or upload your course notes.",
                "reason": "You haven't completed any quizzes yet. Taking one helps identify your strengths and weak areas.",
                "action": "Take Quiz",
                "topic": "General MCA",
            }

        return {
            "title": "🎉 All Caught Up!",
            "recommendation": "Great work! All planned topics and revisions are currently on track.",
            "reason": "You have maintained over 60% accuracy across all tested topics.",
            "action": "Explore New Topic",
            "topic": "Advanced Computer Science",
        }
    finally:
        conn.close()


# --- REVISION SCHEDULER METHODS ---

def schedule_revision(
    subject: str,
    topic: str,
    interval_days: int = 1,
    status: str = "Due Today",
    source_type: str = "Manual",
) -> int:
    """Add a spaced repetition schedule item for a topic."""
    conn = _get_connection()
    due_date = (date.today() + timedelta(days=interval_days)).isoformat()
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO revision_schedule (
                    subject, topic, due_date, interval_days, status, source_type, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (subject, topic, due_date, interval_days, status, source_type),
            )
            return cursor.lastrowid
    finally:
        conn.close()


def get_revision_schedule() -> Dict[str, List[Dict[str, Any]]]:
    """Retrieve revision schedule categorized into Today's, Upcoming, and Completed."""
    conn = _get_connection()
    today_str = date.today().isoformat()
    try:
        cursor = conn.cursor()

        # Today's Due
        cursor.execute(
            """
            SELECT * FROM revision_schedule
            WHERE due_date <= ? AND status != 'Completed'
            ORDER BY due_date ASC
            """,
            (today_str,),
        )
        today_tasks = [dict(r) for r in cursor.fetchall()]

        # Upcoming
        cursor.execute(
            """
            SELECT * FROM revision_schedule
            WHERE due_date > ? AND status != 'Completed'
            ORDER BY due_date ASC
            """,
            (today_str,),
        )
        upcoming_tasks = [dict(r) for r in cursor.fetchall()]

        # Completed
        cursor.execute(
            """
            SELECT * FROM revision_schedule
            WHERE status = 'Completed'
            ORDER BY id DESC LIMIT 20
            """
        )
        completed_tasks = [dict(r) for r in cursor.fetchall()]

        return {
            "today": today_tasks,
            "upcoming": upcoming_tasks,
            "completed": completed_tasks,
        }
    finally:
        conn.close()


def mark_revision_completed(revision_id: int, next_interval_days: int = 3) -> bool:
    """Mark a revision item as completed and optionally schedule the next spaced interval."""
    conn = _get_connection()
    item_subject = None
    item_topic = None
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM revision_schedule WHERE id = ?", (revision_id,))
            item = cursor.fetchone()
            if not item:
                return False

            item_subject = item["subject"]
            item_topic = item["topic"]

            conn.execute(
                "UPDATE revision_schedule SET status = 'Completed' WHERE id = ?",
                (revision_id,),
            )
    finally:
        conn.close()

    # Schedule next spaced repetition step after connection is closed
    if next_interval_days > 0 and item_subject and item_topic:
        schedule_revision(
            subject=item_subject,
            topic=item_topic,
            interval_days=next_interval_days,
            status="Upcoming",
            source_type="Spaced Repetition Step",
        )

    log_activity(
        "Revision Completed",
        f"Completed revision for '{item_topic}'. Next scheduled in {next_interval_days} days.",
    )
    return True


# --- STUDY TASKS & ADAPTIVE PLANNER METHODS ---

def save_study_tasks(tasks: List[Dict[str, Any]]) -> int:
    """Save a list of parsed study tasks from a newly generated study plan."""
    conn = _get_connection()
    saved_count = 0
    today = date.today()

    try:
        with conn:
            for idx, t in enumerate(tasks):
                task_date = t.get("task_date") or (today + timedelta(days=idx // 2)).isoformat()
                conn.execute(
                    """
                    INSERT INTO study_tasks (
                        plan_id, subject, topic, priority, status,
                        exam_date, task_date, task_type, allocated_hours,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """,
                    (
                        t.get("plan_id", "default_plan"),
                        t.get("subject", "General MCA"),
                        t.get("topic", "Topic"),
                        t.get("priority", "Medium"),
                        t.get("status", "Not Started"),
                        t.get("exam_date", "TBD"),
                        task_date,
                        t.get("task_type", "Learning"),
                        float(t.get("allocated_hours", 2.0)),
                    ),
                )
                saved_count += 1

        log_activity(
            activity_type="Study Plan Created",
            description=f"Generated and saved {saved_count} structured study tasks.",
            conn=conn,
        )
        return saved_count
    finally:
        conn.close()


def update_task_status(task_id: int, new_status: str) -> bool:
    """Update progress status ('Not Started', 'In Progress', 'Completed') of a task."""
    conn = _get_connection()
    topic_name = None
    subj_name = None
    updated = False
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
                c = conn.cursor()
                c.execute("SELECT subject, topic FROM study_tasks WHERE id = ?", (task_id,))
                row = c.fetchone()
                if row:
                    topic_name = row["topic"]
                    subj_name = row["subject"]
    finally:
        conn.close()

    if updated and new_status == "Completed" and topic_name and subj_name:
        log_activity("Topic Completed", f"Marked '{topic_name}' as Completed ✅")
        schedule_revision(
            subject=subj_name,
            topic=topic_name,
            interval_days=1,
            status="Due Tomorrow",
            source_type="Task Completed",
        )

    return updated


def get_study_tasks(status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve all study tasks, optionally filtered by status."""
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        if status_filter and status_filter != "All Statuses":
            cursor.execute(
                "SELECT * FROM study_tasks WHERE status = ? ORDER BY id ASC",
                (status_filter,),
            )
        else:
            cursor.execute("SELECT * FROM study_tasks ORDER BY id ASC")
        return [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()


def adaptive_replan(
    exam_date: Optional[str] = None,
    daily_hours: float = 2.0,
) -> Dict[str, Any]:
    """
    Adaptively recalculate the study schedule when a student falls behind.
    CRITICAL: Does NOT delete completed tasks.
    Prioritizes overdue, weak topics, and high-priority uncompleted tasks.
    """
    # Fetch weak topics before opening conn
    weak_topics = {w["topic"].lower(): w for w in get_weak_topics(65.0)}

    conn = _get_connection()
    try:
        cursor = conn.cursor()

        # 1. Fetch completed tasks (preserved as is)
        cursor.execute("SELECT * FROM study_tasks WHERE status = 'Completed'")
        completed_tasks = [dict(r) for r in cursor.fetchall()]

        # 2. Fetch uncompleted tasks ('Not Started' or 'In Progress')
        cursor.execute("SELECT * FROM study_tasks WHERE status != 'Completed' ORDER BY id ASC")
        uncompleted_tasks = [dict(r) for r in cursor.fetchall()]

        if not uncompleted_tasks:
            return {
                "message": "All tasks are already completed! Nothing to replan.",
                "rearranged_count": 0,
                "completed_count": len(completed_tasks),
            }

        # Sort uncompleted tasks: Weak topics first, then High priority
        def task_sort_key(t):
            t_name = t.get("topic", "").lower()
            is_weak = 0 if t_name in weak_topics else 1
            priority_map = {"high": 0, "medium": 1, "low": 2}
            p_val = priority_map.get(t.get("priority", "medium").lower(), 1)
            return (is_weak, p_val, t.get("id"))

        reordered = sorted(uncompleted_tasks, key=task_sort_key)

        # 5. Redistribute dates starting from today
        today = date.today()
        # Allocate roughly 2 tasks per study day based on daily hours
        tasks_per_day = max(1, int(daily_hours // 1.5))

        with conn:
            for idx, t in enumerate(reordered):
                day_offset = idx // tasks_per_day
                new_date = (today + timedelta(days=day_offset)).isoformat()
                new_priority = "High" if t.get("topic", "").lower() in weak_topics else t.get("priority", "Medium")

                conn.execute(
                    """
                    UPDATE study_tasks
                    SET task_date = ?, priority = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (new_date, new_priority, t["id"]),
                )

        log_activity(
            "Adaptive Replan",
            f"Recalculated schedule for {len(reordered)} remaining tasks (boosted {len(weak_topics)} weak topics).",
            conn=conn,
        )

        return {
            "message": f"Successfully rearranged {len(reordered)} uncompleted tasks.",
            "rearranged_count": len(reordered),
            "completed_count": len(completed_tasks),
            "weak_boosted_count": sum(1 for t in reordered if t.get("topic", "").lower() in weak_topics),
        }
    finally:
        conn.close()


def get_study_analytics() -> Dict[str, Any]:
    """Calculate topic progress, completion percentages, and subject distribution."""
    conn = _get_connection()
    try:
        cursor = conn.cursor()
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
            log_activity("Study Plan Cleared", "User cleared all planned study tasks.", conn=conn)
            return True
    finally:
        conn.close()


# --- FLASHCARD STORAGE & SM-2 METHODS ---

def save_flashcard(
    deck_name: str,
    front: str,
    back: str,
    source_doc: str = "General",
) -> int:
    """Save a flashcard to SQLite."""
    conn = _get_connection()
    today_str = date.today().isoformat()
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO flashcards (
                    deck_name, source_doc, front, back, next_review_due, created_at
                ) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (deck_name.strip(), source_doc, front.strip(), back.strip(), today_str),
            )
            card_id = cursor.lastrowid
            log_activity("Flashcard Created", f"Added card to deck '{deck_name}'", conn=conn)
            return card_id
    finally:
        conn.close()


def get_flashcards(deck_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve flashcards, optionally filtered by deck."""
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        if deck_name and deck_name != "All Decks":
            cursor.execute("SELECT * FROM flashcards WHERE deck_name = ? ORDER BY id ASC", (deck_name,))
        else:
            cursor.execute("SELECT * FROM flashcards ORDER BY id ASC")
        return [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()


def get_flashcard_decks() -> List[str]:
    """Return distinct list of flashcard deck names."""
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT deck_name FROM flashcards ORDER BY deck_name ASC")
        return [r["deck_name"] for r in cursor.fetchall()]
    finally:
        conn.close()


def review_flashcard(card_id: int, rating: int) -> Dict[str, Any]:
    """
    Update flashcard spaced repetition interval using SuperMemo SM-2 algorithm.
    Ratings:
      0 = Again (Failed)
      1 = Hard (Difficult recall)
      2 = Good (Successful recall)
      3 = Easy (Immediate recall)
    """
    conn = _get_connection()
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM flashcards WHERE id = ?", (card_id,))
            card = cursor.fetchone()
            if not card:
                return {"success": False, "error": "Card not found"}

            rep = card["repetitions"]
            interval = card["interval_days"]
            ef = card["ease_factor"]

            if rating < 2:
                # Failed recall: reset repetitions to 0, review again tomorrow
                rep = 0
                interval = 1
            else:
                # Successful recall: advance spaced interval
                if rep == 0:
                    interval = 1
                elif rep == 1:
                    interval = 3
                else:
                    interval = max(1, int(interval * ef))
                rep += 1

            # Update ease factor (SM-2 standard formula)
            ef = max(1.3, ef + (0.1 - (3 - rating) * (0.08 + (3 - rating) * 0.02)))
            next_due = (date.today() + timedelta(days=interval)).isoformat()

            card_front = card["front"][:40]
            conn.execute(
                """
                UPDATE flashcards
                SET difficulty_rating = ?, repetitions = ?, interval_days = ?,
                    ease_factor = ?, next_review_due = ?, last_reviewed = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (rating, rep, interval, round(ef, 2), next_due, card_id),
            )
    finally:
        conn.close()

    # If difficult (rating <= 1), schedule in revision schedule after conn is closed
    if rating <= 1:
        schedule_revision(
            subject="Flashcard",
            topic=card_front,
            interval_days=1,
            status="Due Today",
            source_type="Flashcard Review",
        )

    return {
        "success": True,
        "interval_days": interval,
        "next_review_due": next_due,
        "repetitions": rep,
    }


def delete_flashcard(card_id: int) -> bool:
    """Delete a flashcard."""
    conn = _get_connection()
    try:
        with conn:
            cursor = conn.execute("DELETE FROM flashcards WHERE id = ?", (card_id,))
            return cursor.rowcount > 0
    finally:
        conn.close()


def clear_deck(deck_name: str) -> bool:
    """Delete all cards in a deck."""
    conn = _get_connection()
    try:
        with conn:
            conn.execute("DELETE FROM flashcards WHERE deck_name = ?", (deck_name,))
            return True
    finally:
        conn.close()


# --- DOCUMENT REGISTRY METHODS ---

def register_document(
    filename: str,
    file_hash: str,
    file_size_kb: float,
    page_count: int,
    chunk_count: int,
    status: str = "Ready",
    error_msg: Optional[str] = None,
):
    """Store or update document indexing metadata in SQLite."""
    conn = _get_connection()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO document_registry (
                    filename, file_hash, file_size_kb, page_count,
                    chunk_count, status, error_msg, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(filename) DO UPDATE SET
                    file_hash = excluded.file_hash,
                    file_size_kb = excluded.file_size_kb,
                    page_count = excluded.page_count,
                    chunk_count = excluded.chunk_count,
                    status = excluded.status,
                    error_msg = excluded.error_msg,
                    created_at = CURRENT_TIMESTAMP
                """,
                (filename, file_hash, file_size_kb, page_count, chunk_count, status, error_msg),
            )
            log_activity("Document Indexed", f"Indexed '{filename}' ({page_count} pages, {chunk_count} chunks)", conn=conn)
    finally:
        conn.close()


def get_document_by_hash(file_hash: str) -> Optional[Dict[str, Any]]:
    """Check if a file with the same SHA-256 hash already exists."""
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM document_registry WHERE file_hash = ?", (file_hash,))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_registered_documents() -> List[Dict[str, Any]]:
    """Retrieve list of all registered documents."""
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM document_registry ORDER BY created_at DESC")
        return [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()


def delete_registered_document(filename: str) -> bool:
    """Remove a document from registry."""
    conn = _get_connection()
    try:
        with conn:
            cursor = conn.execute("DELETE FROM document_registry WHERE filename = ?", (filename,))
            log_activity("Document Deleted", f"Removed '{filename}' from system", conn=conn)
            return cursor.rowcount > 0
    finally:
        conn.close()


# --- ACTIVITY LOG METHODS ---

def log_activity(
    activity_type: str,
    description: str,
    conn: Optional[sqlite3.Connection] = None,
):
    """Record an action into the student audit timeline."""
    if conn is not None:
        conn.execute(
            "INSERT INTO activity_log (activity_type, description) VALUES (?, ?)",
            (activity_type, description),
        )
        return

    c = _get_connection()
    try:
        with c:
            c.execute(
                "INSERT INTO activity_log (activity_type, description) VALUES (?, ?)",
                (activity_type, description),
            )
    finally:
        c.close()


def get_recent_activities(limit: int = 15) -> List[Dict[str, Any]]:
    """Retrieve recent student activities in descending order."""
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, activity_type, description, created_at FROM activity_log ORDER BY created_at DESC, id DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()
