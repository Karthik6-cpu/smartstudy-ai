"""
Memory and Authentication package for SmartStudy AI.
Persistent SQLite storage for student preferences, learning profiles, and user authentication.
"""

from memory.sqlite_memory import (
    save_memory,
    retrieve_memory,
    get_memory,
    list_all_memories,
    delete_memory,
    clear_all_memories,
)
from memory.auth_manager import (
    init_auth_db,
    register_user,
    authenticate_user,
    get_user_by_username,
    list_users,
    seed_demo_user,
)

__all__ = [
    "save_memory",
    "retrieve_memory",
    "get_memory",
    "list_all_memories",
    "delete_memory",
    "clear_all_memories",
    "init_auth_db",
    "register_user",
    "authenticate_user",
    "get_user_by_username",
    "list_users",
    "seed_demo_user",
]
