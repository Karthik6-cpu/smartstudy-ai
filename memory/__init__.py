"""
Memory package for SmartStudy AI.
Persistent SQLite storage for student preferences, learning profiles, and revision goals.
"""

from memory.sqlite_memory import (
    save_memory,
    retrieve_memory,
    get_memory,
    list_all_memories,
    delete_memory,
    clear_all_memories,
)

__all__ = [
    "save_memory",
    "retrieve_memory",
    "get_memory",
    "list_all_memories",
    "delete_memory",
    "clear_all_memories",
]
