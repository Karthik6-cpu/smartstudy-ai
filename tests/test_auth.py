"""
Unit Tests for SmartStudy AI Authentication & User Management.
Tests user registration, validation, PBKDF2 password hashing, authentication, and session security.
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.auth_manager import (
    init_auth_db,
    register_user,
    authenticate_user,
    get_user_by_username,
    list_users,
    validate_email,
    validate_username,
)


class TestAuthManager(unittest.TestCase):
    """Test suite for authentication database and operations."""

    @classmethod
    def setUpClass(cls):
        """Ensure database table is initialized."""
        init_auth_db()

    def test_01_demo_user_exists_and_can_login(self):
        """Verify demo student user can authenticate with default credentials."""
        success, msg, user = authenticate_user("mca_student", "password123")
        self.assertTrue(success, f"Demo login failed: {msg}")
        self.assertIsNotNone(user)
        self.assertEqual(user["username"], "mca_student")
        self.assertEqual(user["full_name"], "MCA Scholar")

    def test_02_demo_login_with_email(self):
        """Verify demo user can authenticate using email address."""
        success, msg, user = authenticate_user("student@smartstudy.local", "password123")
        self.assertTrue(success, f"Login with email failed: {msg}")
        self.assertEqual(user["username"], "mca_student")

    def test_03_invalid_password(self):
        """Verify authentication fails for wrong password."""
        success, msg, user = authenticate_user("mca_student", "wrongpass999")
        self.assertFalse(success)
        self.assertIsNone(user)

    def test_04_nonexistent_user(self):
        """Verify authentication fails for non-existent username."""
        success, msg, user = authenticate_user("nonexistent_user_xyz", "pass123")
        self.assertFalse(success)
        self.assertIsNone(user)

    def test_05_username_validation(self):
        """Verify username format rules."""
        valid, _ = validate_username("rahul_123")
        self.assertTrue(valid)

        short, _ = validate_username("ab")
        self.assertFalse(short)

        invalid_char, _ = validate_username("rahul@123")
        self.assertFalse(invalid_char)

    def test_06_email_validation(self):
        """Verify email format validation."""
        self.assertTrue(validate_email("test@example.com"))
        self.assertTrue(validate_email("student.mca@uni.edu"))
        self.assertFalse(validate_email("notanemail"))
        self.assertFalse(validate_email("bad@email"))

    def test_07_register_new_user_success(self):
        """Verify new user registration with unique username and email."""
        import time
        ts = int(time.time() * 1000)
        test_user = f"user_{ts}"
        test_email = f"user_{ts}@mca.edu"
        
        success, msg, user = register_user(
            username=test_user,
            email=test_email,
            full_name="Test Coder",
            password="mypassword123",
            degree_program="MCA",
            semester="Semester 2",
        )
        self.assertTrue(success, f"Registration failed: {msg}")
        self.assertEqual(user["username"], test_user)
        self.assertEqual(user["degree_program"], "MCA")

        # Verify lookup
        fetched = get_user_by_username(test_user)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["username"], test_user)

        # Now verify login with newly registered user
        auth_ok, auth_msg, auth_user = authenticate_user(test_user, "mypassword123")
        self.assertTrue(auth_ok, f"Login failed for newly registered user: {auth_msg}")
        self.assertEqual(auth_user["full_name"], "Test Coder")

    def test_08_prevent_duplicate_username(self):
        """Verify registration prevents duplicate usernames."""
        success, msg, user = register_user(
            username="mca_student",
            email="unique_email_123@domain.com",
            full_name="Duplicate Name",
            password="password123",
        )
        self.assertFalse(success)
        self.assertIn("already taken", msg.lower())

    def test_09_prevent_duplicate_email(self):
        """Verify registration prevents duplicate email addresses."""
        success, msg, user = register_user(
            username="unique_user_999",
            email="student@smartstudy.local",
            full_name="Duplicate Email",
            password="password123",
        )
        self.assertFalse(success)
        self.assertIn("already registered", msg.lower())

    def test_10_list_users_omits_sensitive_hash(self):
        """Verify user listing returns safe metadata without password hashes or salts."""
        users = list_users()
        self.assertGreater(len(users), 0)
        for u in users:
            self.assertIn("username", u)
            self.assertIn("email", u)
            self.assertNotIn("password_hash", u)
            self.assertNotIn("salt", u)


if __name__ == "__main__":
    unittest.main()
