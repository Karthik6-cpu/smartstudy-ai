"""
Authentication and User Management for SmartStudy AI.
Supports both Local SQLite and MongoDB database backends.
Uses PBKDF2-HMAC-SHA256 password hashing with unique per-user salts.
"""

import os
import sqlite3
import hashlib
import secrets
import re
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List
from config.settings import MEMORY_DIR, MEMORY_DB_PATH
from memory.mongo_manager import (
    is_mongodb_available,
    init_mongo_collections,
    mongo_register_user,
    mongo_authenticate_user,
    mongo_get_user_by_username,
    mongo_list_users,
    DEFAULT_MONGO_URI,
    DEFAULT_MONGO_DB_NAME,
)

# Active Database Backend preference ("mongodb" or "sqlite")
DB_BACKEND = os.getenv("DB_BACKEND", "mongodb").lower()


def get_active_db_type() -> Tuple[str, bool]:
    """
    Check the active database type and connection health.
    Returns: (db_name: str, is_connected: bool)
    """
    if DB_BACKEND == "mongodb":
        ok, _ = is_mongodb_available()
        if ok:
            return "MongoDB", True
    return "SQLite (Local File)", True


def _get_connection() -> sqlite3.Connection:
    """Initialize memory directory and return SQLite connection with row factory."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(MEMORY_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_auth_db():
    """Create the users table and seed initial demo account if empty in SQLite and MongoDB."""
    conn = _get_connection()
    try:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL COLLATE NOCASE,
                    email TEXT UNIQUE NOT NULL COLLATE NOCASE,
                    full_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    degree_program TEXT DEFAULT 'MCA',
                    semester TEXT DEFAULT 'Semester 1',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
        seed_demo_user()
    finally:
        conn.close()

    # Also initialize MongoDB if available
    try:
        init_mongo_collections()
    except Exception:
        pass


def _hash_password(password: str, salt_hex: str) -> str:
    """Hash password using PBKDF2-HMAC-SHA256 with salt."""
    salt_bytes = bytes.fromhex(salt_hex)
    pwd_bytes = password.encode("utf-8")
    key = hashlib.pbkdf2_hmac("sha256", pwd_bytes, salt_bytes, 100_000)
    return key.hex()


def seed_demo_user() -> bool:
    """Seed a default demo student account if not present."""
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = ?", ("mca_student",))
        if cursor.fetchone():
            return False

        salt = secrets.token_hex(16)
        pwd_hash = _hash_password("password123", salt)
        with conn:
            conn.execute(
                """
                INSERT INTO users (username, email, full_name, password_hash, salt, degree_program, semester)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "mca_student",
                    "student@smartstudy.local",
                    "MCA Scholar",
                    pwd_hash,
                    salt,
                    "MCA (Master of Computer Applications)",
                    "Semester 1",
                ),
            )
        return True
    finally:
        conn.close()


def validate_email(email: str) -> bool:
    """Simple regex check for email format."""
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    return bool(re.match(pattern, email.strip()))


def validate_username(username: str) -> Tuple[bool, str]:
    """Validate username rules: length 3-30, alphanumeric or underscore/hyphen."""
    u = username.strip()
    if len(u) < 3:
        return False, "Username must be at least 3 characters long."
    if len(u) > 30:
        return False, "Username cannot exceed 30 characters."
    if not re.match(r"^[a-zA-Z0-9_.-]+$", u):
        return False, "Username can only contain letters, numbers, underscores, dots, and hyphens."
    return True, "Valid username."


def register_user(
    username: str,
    email: str,
    full_name: str,
    password: str,
    degree_program: str = "MCA",
    semester: str = "Semester 1",
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Register a new student user in SQLite database (and MongoDB if active).

    Returns:
        (success: bool, message: str, user_dict: Optional[dict])
    """
    u = username.strip()
    e = email.strip().lower()
    fn = full_name.strip()

    if not fn:
        return False, "Please provide your Full Name.", None

    u_valid, u_msg = validate_username(u)
    if not u_valid:
        return False, u_msg, None

    if not validate_email(e):
        return False, "Please enter a valid email address.", None

    if len(password) < 6:
        return False, "Password must be at least 6 characters long.", None

    # If MongoDB backend preferred and online, use MongoDB
    if DB_BACKEND == "mongodb":
        mongo_ok, _ = is_mongodb_available()
        if mongo_ok:
            m_ok, m_msg, m_user = mongo_register_user(u, e, fn, password, degree_program, semester)
            if m_ok:
                # Also mirror into local SQLite for offline resilience
                try:
                    conn = _get_connection()
                    salt = secrets.token_hex(16)
                    pwd_hash = _hash_password(password, salt)
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with conn:
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO users (username, email, full_name, password_hash, salt, degree_program, semester, created_at, last_login)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (u, e, fn, pwd_hash, salt, degree_program, semester, now, now),
                        )
                    conn.close()
                except Exception:
                    pass
            return m_ok, m_msg, m_user

    # SQLite Registration
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = ?", (u,))
        if cursor.fetchone():
            return False, f"Username '{u}' is already taken. Please choose another.", None

        cursor.execute("SELECT id FROM users WHERE email = ?", (e,))
        if cursor.fetchone():
            return False, f"Email '{e}' is already registered. Please log in.", None

        salt = secrets.token_hex(16)
        pwd_hash = _hash_password(password, salt)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with conn:
            cursor.execute(
                """
                INSERT INTO users (username, email, full_name, password_hash, salt, degree_program, semester, created_at, last_login)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (u, e, fn, pwd_hash, salt, degree_program, semester, now, now),
            )
            new_id = cursor.lastrowid

        user_dict = {
            "id": new_id,
            "username": u,
            "email": e,
            "full_name": fn,
            "degree_program": degree_program,
            "semester": semester,
            "created_at": now,
            "last_login": now,
        }
        return True, "Account registered successfully!", user_dict

    except sqlite3.IntegrityError as err:
        return False, f"Registration failed: Database constraint error ({str(err)})", None
    except Exception as err:
        return False, f"Registration error: {str(err)}", None
    finally:
        conn.close()


def authenticate_user(username_or_email: str, password: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Authenticate a user by username or email and password.

    Returns:
        (success: bool, message: str, user_dict: Optional[dict])
    """
    identifier = username_or_email.strip()
    if not identifier:
        return False, "Please enter your username or email.", None
    if not password:
        return False, "Please enter your password.", None

    # Check MongoDB if preferred
    if DB_BACKEND == "mongodb":
        mongo_ok, _ = is_mongodb_available()
        if mongo_ok:
            return mongo_authenticate_user(identifier, password)

    # SQLite Authentication
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, username, email, full_name, password_hash, salt, degree_program, semester, created_at
            FROM users
            WHERE username = ? OR email = ?
            """,
            (identifier, identifier.lower()),
        )
        row = cursor.fetchone()
        if not row:
            return False, "Invalid username/email or password.", None

        stored_hash = row["password_hash"]
        salt = row["salt"]
        computed_hash = _hash_password(password, salt)

        if not secrets.compare_digest(stored_hash, computed_hash):
            return False, "Invalid username/email or password.", None

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with conn:
            conn.execute("UPDATE users SET last_login = ? WHERE id = ?", (now, row["id"]))

        user_dict = {
            "id": row["id"],
            "username": row["username"],
            "email": row["email"],
            "full_name": row["full_name"],
            "degree_program": row["degree_program"] or "MCA",
            "semester": row["semester"] or "Semester 1",
            "created_at": row["created_at"],
            "last_login": now,
        }
        return True, f"Welcome back, {row['full_name']}!", user_dict

    except Exception as err:
        return False, f"Authentication error: {str(err)}", None
    finally:
        conn.close()


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """Retrieve user details by username from active database."""
    if DB_BACKEND == "mongodb":
        mongo_ok, _ = is_mongodb_available()
        if mongo_ok:
            u_doc = mongo_get_user_by_username(username)
            if u_doc:
                return u_doc

    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, username, email, full_name, degree_program, semester, created_at, last_login FROM users WHERE username = ?",
            (username.strip(),),
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_users() -> List[Dict[str, Any]]:
    """List all registered users from active database (excluding password hashes and salts)."""
    if DB_BACKEND == "mongodb":
        mongo_ok, _ = is_mongodb_available()
        if mongo_ok:
            m_users = mongo_list_users()
            if m_users:
                return m_users

    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, username, email, full_name, degree_program, semester, created_at, last_login FROM users ORDER BY id ASC"
        )
        return [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()
