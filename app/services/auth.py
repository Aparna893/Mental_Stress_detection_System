
"""
StressSense - User Authentication Service

Handles user registration, password hashing,
and login authentication.
"""

import re
import bcrypt

from datetime import datetime, timezone
from pymongo.errors import DuplicateKeyError, PyMongoError

from app.services.database import get_users_collection


def validate_email(email):
    """Check basic email format."""
    pattern = r"^[\w.+-]+@[\w-]+\.[\w.-]+$"
    return bool(re.match(pattern, email))


def register_user(username, email, password):
    """
    Register a new user.

    Returns:
        (True, message, user_id) on success
        (False, message, None) on failure
    """

    username = username.strip()
    email = email.strip().lower()

    if not username:
        return False, "Username is required.", None

    if not validate_email(email):
        return False, "Please enter a valid email address.", None

    if len(password) < 8:
        return False, "Password must contain at least 8 characters.", None

    if len(password.encode("utf-8")) > 72:
        return False, "Password must not exceed 72 UTF-8 bytes.", None

    try:
        users = get_users_collection()

        # Hash the password before storing it.
        password_hash = bcrypt.hashpw(
            password.encode("utf-8"),
            bcrypt.gensalt()
        )

        user = {
            "username": username,
            "email": email,
            "password_hash": password_hash,
            "created_at": datetime.now(timezone.utc)
        }

        result = users.insert_one(user)

        return True, "Registration successful!", str(result.inserted_id)

    except DuplicateKeyError:
        return False, "An account with this email already exists.", None

    except PyMongoError:
        return False, "Database error. Please try again.", None


def authenticate_user(email, password):
    """
    Authenticate an existing user.

    Returns:
        (True, message, user_data) on success
        (False, message, None) on failure
    """

    email = email.strip().lower()

    if not email or not password:
        return False, "Email and password are required.", None

    try:
        users = get_users_collection()

        user = users.find_one({"email": email})

        if not user:
            return False, "Invalid email or password.", None

        stored_hash = user["password_hash"]

        password_matches = bcrypt.checkpw(
            password.encode("utf-8"),
            stored_hash
        )

        if not password_matches:
            return False, "Invalid email or password.", None

        user_data = {
            "user_id": str(user["_id"]),
            "username": user["username"],
            "email": user["email"]
        }

        return True, "Login successful!", user_data

    except (PyMongoError, ValueError, KeyError, TypeError):
        return False, "Unable to authenticate. Please try again.", None
