"""
MongoDB Database Manager for SmartStudy AI.
Provides MongoDB storage and collection management for Users, Memories, Quizzes, and Study Tasks.
Enables seamless switching between local SQLite and MongoDB (Local or Atlas).
"""

import os
import secrets
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List
import pymongo
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# MongoDB Default Settings
DEFAULT_MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://ramanichallapalli6_db_user:1vS483QaDBtMDR6W@cluster0.byv2wdo.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
)
DEFAULT_MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "smartstudy_ai")

_mongo_client: Optional[MongoClient] = None


def get_mongo_client(uri: str = DEFAULT_MONGO_URI, timeout_ms: int = 2000) -> Optional[MongoClient]:
    """Get or initialize MongoDB client with quick connection timeout."""
    global _mongo_client
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)
        # Test connection with ping
        client.admin.command("ping")
        _mongo_client = client
        return client
    except Exception:
        return None


def is_mongodb_available(uri: str = DEFAULT_MONGO_URI) -> Tuple[bool, str]:
    """Check if MongoDB server is reachable at the given URI."""
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=1500)
        client.admin.command("ping")
        return True, "MongoDB connected successfully 🟢"
    except (ConnectionFailure, ServerSelectionTimeoutError) as err:
        return False, f"MongoDB offline: {str(err)}"
    except Exception as err:
        return False, f"Connection error: {str(err)}"


def get_mongo_db(uri: str = DEFAULT_MONGO_URI, db_name: str = DEFAULT_MONGO_DB_NAME):
    """Retrieve MongoDB database object."""
    client = get_mongo_client(uri)
    if client:
        return client[db_name]
    return None


def _hash_password(password: str, salt_hex: str) -> str:
    """Hash password using PBKDF2-HMAC-SHA256."""
    salt_bytes = bytes.fromhex(salt_hex)
    pwd_bytes = password.encode("utf-8")
    return hashlib.pbkdf2_hmac("sha256", pwd_bytes, salt_bytes, 100_000).hex()


def init_mongo_collections(uri: str = DEFAULT_MONGO_URI, db_name: str = DEFAULT_MONGO_DB_NAME) -> bool:
    """Initialize MongoDB indexes and seed demo user."""
    db = get_mongo_db(uri, db_name)
    if db is None:
        return False

    try:
        # Create unique indexes on username and email
        db.users.create_index("username", unique=True)
        db.users.create_index("email", unique=True)
        db.student_memory.create_index("key", unique=True)

        # Seed demo student if not present
        if not db.users.find_one({"username": "mca_student"}):
            salt = secrets.token_hex(16)
            pwd_hash = _hash_password("password123", salt)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            db.users.insert_one(
                {
                    "username": "mca_student",
                    "email": "student@smartstudy.local",
                    "full_name": "MCA Scholar",
                    "password_hash": pwd_hash,
                    "salt": salt,
                    "degree_program": "MCA (Master of Computer Applications)",
                    "semester": "Semester 1",
                    "created_at": now,
                    "last_login": now,
                }
            )
        return True
    except Exception:
        return False


def mongo_register_user(
    username: str,
    email: str,
    full_name: str,
    password: str,
    degree_program: str = "MCA",
    semester: str = "Semester 1",
    uri: str = DEFAULT_MONGO_URI,
    db_name: str = DEFAULT_MONGO_DB_NAME,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """Register a new student user in MongoDB."""
    db = get_mongo_db(uri, db_name)
    if db is None:
        return False, "MongoDB connection unavailable.", None

    u = username.strip().lower()
    e = email.strip().lower()
    fn = full_name.strip()

    if db.users.find_one({"username": u}):
        return False, f"Username '{u}' is already taken.", None
    if db.users.find_one({"email": e}):
        return False, f"Email '{e}' is already registered.", None

    salt = secrets.token_hex(16)
    pwd_hash = _hash_password(password, salt)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    doc = {
        "username": u,
        "email": e,
        "full_name": fn,
        "password_hash": pwd_hash,
        "salt": salt,
        "degree_program": degree_program,
        "semester": semester,
        "created_at": now,
        "last_login": now,
    }

    try:
        res = db.users.insert_one(doc)
        user_dict = {
            "id": str(res.inserted_id),
            "username": u,
            "email": e,
            "full_name": fn,
            "degree_program": degree_program,
            "semester": semester,
            "created_at": now,
            "last_login": now,
        }
        return True, "Account registered in MongoDB successfully!", user_dict
    except Exception as err:
        return False, f"MongoDB registration error: {str(err)}", None


def mongo_authenticate_user(
    username_or_email: str,
    password: str,
    uri: str = DEFAULT_MONGO_URI,
    db_name: str = DEFAULT_MONGO_DB_NAME,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """Authenticate a student user in MongoDB."""
    db = get_mongo_db(uri, db_name)
    if db is None:
        return False, "MongoDB connection unavailable.", None

    identifier = username_or_email.strip().lower()
    user_doc = db.users.find_one({"$or": [{"username": identifier}, {"email": identifier}]})

    if not user_doc:
        return False, "Invalid username or password.", None

    stored_hash = user_doc.get("password_hash", "")
    salt = user_doc.get("salt", "")
    computed_hash = _hash_password(password, salt)

    if not secrets.compare_digest(stored_hash, computed_hash):
        return False, "Invalid username or password.", None

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    db.users.update_one({"_id": user_doc["_id"]}, {"$set": {"last_login": now}})

    user_dict = {
        "id": str(user_doc["_id"]),
        "username": user_doc.get("username"),
        "email": user_doc.get("email"),
        "full_name": user_doc.get("full_name"),
        "degree_program": user_doc.get("degree_program", "MCA"),
        "semester": user_doc.get("semester", "Semester 1"),
        "created_at": user_doc.get("created_at"),
        "last_login": now,
    }
    return True, f"Welcome back, {user_doc.get('full_name')}!", user_dict


def mongo_get_user_by_username(
    username: str,
    uri: str = DEFAULT_MONGO_URI,
    db_name: str = DEFAULT_MONGO_DB_NAME,
) -> Optional[Dict[str, Any]]:
    """Retrieve user details by username from MongoDB."""
    db = get_mongo_db(uri, db_name)
    if db is None:
        return None
    try:
        user_doc = db.users.find_one({"username": username.strip().lower()})
        if not user_doc:
            return None
        return {
            "id": str(user_doc["_id"]),
            "username": user_doc.get("username"),
            "email": user_doc.get("email"),
            "full_name": user_doc.get("full_name"),
            "degree_program": user_doc.get("degree_program", "MCA"),
            "semester": user_doc.get("semester", "Semester 1"),
            "created_at": user_doc.get("created_at"),
            "last_login": user_doc.get("last_login"),
        }
    except Exception:
        return None


def mongo_list_users(
    uri: str = DEFAULT_MONGO_URI,
    db_name: str = DEFAULT_MONGO_DB_NAME,
) -> List[Dict[str, Any]]:
    """List all registered users from MongoDB without sensitive hashes."""
    db = get_mongo_db(uri, db_name)
    if db is None:
        return []
    try:
        cursor = db.users.find({}, {"password_hash": 0, "salt": 0})
        users = []
        for doc in cursor:
            doc_dict = {
                "id": str(doc["_id"]),
                "username": doc.get("username"),
                "email": doc.get("email"),
                "full_name": doc.get("full_name"),
                "degree_program": doc.get("degree_program", "MCA"),
                "semester": doc.get("semester", "Semester 1"),
                "created_at": doc.get("created_at"),
                "last_login": doc.get("last_login"),
            }
            users.append(doc_dict)
        return users
    except Exception:
        return []
