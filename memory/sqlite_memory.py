"""
SQLite Persistent Memory Store for SmartStudy AI.
Stores and searches key-value student facts, preferences, exam goals, and learning notes.
"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Any
from config.settings import MEMORY_DIR, MEMORY_DB_PATH


def _get_connection() -> sqlite3.Connection:
    """Initialize directory and return SQLite connection with row factory."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(MEMORY_DB_PATH))
    conn.row_factory = sqlite3.Row
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS student_memory (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
    return conn


def save_memory(key: str, value: str) -> str:
    """
    Persist or update a student fact, preference, or goal in SQLite.

    Args:
        key: Unique identifier or topic (e.g. 'preferred_language', 'target_exam_score').
        value: The information or preference to remember.

    Returns:
        Confirmation message.
    """
    clean_key = key.strip().lower().replace(" ", "_")
    clean_val = value.strip()
    if not clean_key or not clean_val:
        return "Error: Memory key and value cannot be empty."

    conn = _get_connection()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO student_memory (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (clean_key, clean_val),
            )
        return f"Successfully saved memory: [{clean_key}] = '{clean_val}'"
    except Exception as e:
        return f"Error saving memory: {str(e)}"
    finally:
        conn.close()


def retrieve_memory(query: str) -> str:
    """
    Search stored student memories by keyword matching across keys and values.

    Args:
        query: Search term or concept (e.g. 'language', 'score', 'goal').

    Returns:
        Formatted string of matching memories or notice if none found.
    """
    clean_q = query.strip()
    if not clean_q:
        return "Please provide a search term to retrieve memory."

    conn = _get_connection()
    try:
        cursor = conn.cursor()
        search_pattern = f"%{clean_q}%"
        wildcard_pattern = f"%{clean_q.replace(' ', '%')}%"
        cursor.execute(
            """
            SELECT key, value, updated_at FROM student_memory
            WHERE key LIKE ? OR value LIKE ? OR key LIKE ?
            ORDER BY updated_at DESC
            """,
            (search_pattern, search_pattern, wildcard_pattern),
        )
        rows = cursor.fetchall()
        if not rows:
            return f"No memories found matching query: '{query}'."

        results = []
        for r in rows:
            results.append(f"- **{r['key']}**: {r['value']} *(saved {r['updated_at']})*")
        return "Stored Student Memories:\n" + "\n".join(results)
    except Exception as e:
        return f"Error retrieving memory: {str(e)}"
    finally:
        conn.close()


def get_memory(key: str) -> Optional[str]:
    """Retrieve the exact value for a specific key."""
    clean_key = key.strip().lower().replace(" ", "_")
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM student_memory WHERE key = ?", (clean_key,))
        row = cursor.fetchone()
        return row["value"] if row else None
    finally:
        conn.close()


def list_all_memories() -> List[Dict[str, Any]]:
    """Retrieve all memories as a list of dicts for UI display."""
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT key, value, updated_at FROM student_memory ORDER BY updated_at DESC")
        return [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()


def delete_memory(key: str) -> bool:
    """Delete a memory item by key."""
    clean_key = key.strip().lower().replace(" ", "_")
    conn = _get_connection()
    try:
        with conn:
            cursor = conn.execute("DELETE FROM student_memory WHERE key = ?", (clean_key,))
            return cursor.rowcount > 0
    finally:
        conn.close()


def clear_all_memories() -> bool:
    """Clear all memories from the table."""
    conn = _get_connection()
    try:
        with conn:
            conn.execute("DELETE FROM student_memory")
            return True
    finally:
        conn.close()
