import sqlite3
import hashlib
import os
import re
from datetime import datetime


DB_PATH = "users.db"


def get_connection():
    """Get SQLite database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database and create tables if they don't exist"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_login TEXT
        )
    """)
    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    """Hash password using SHA-256 with salt"""
    salt = "car_damage_salt_2024"
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()


def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_password(password: str) -> tuple[bool, str]:
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number"
    return True, "Password is strong"


def register_user(full_name: str, email: str, username: str, password: str) -> tuple[bool, str]:
    """Register a new user"""
    # Validate inputs
    if not full_name.strip():
        return False, "Full name is required"
    if not validate_email(email):
        return False, "Invalid email format"
    if len(username) < 3:
        return False, "Username must be at least 3 characters"
    if not username.isalnum():
        return False, "Username must contain only letters and numbers"

    is_valid, msg = validate_password(password)
    if not is_valid:
        return False, msg

    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Check for existing email or username
        cursor.execute("SELECT id FROM users WHERE email = ? OR username = ?", (email, username))
        existing = cursor.fetchone()
        if existing:
            cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
            if cursor.fetchone():
                return False, "An account with this email already exists"
            return False, "Username is already taken"

        # Insert new user
        password_hash = hash_password(password)
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO users (full_name, email, username, password_hash, created_at) VALUES (?, ?, ?, ?, ?)",
            (full_name.strip(), email.lower(), username.lower(), password_hash, created_at)
        )
        conn.commit()
        conn.close()
        return True, "Account created successfully!"

    except sqlite3.Error as e:
        return False, f"Database error: {str(e)}"


def login_user(username_or_email: str, password: str) -> tuple[bool, str, dict]:
    """Authenticate a user"""
    if not username_or_email or not password:
        return False, "Please fill in all fields", {}

    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Find user by username or email
        cursor.execute(
            "SELECT * FROM users WHERE username = ? OR email = ?",
            (username_or_email.lower(), username_or_email.lower())
        )
        user = cursor.fetchone()

        if not user:
            conn.close()
            return False, "Invalid username/email or password", {}

        # Verify password
        if user["password_hash"] != hash_password(password):
            conn.close()
            return False, "Invalid username/email or password", {}

        # Update last login
        last_login = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("UPDATE users SET last_login = ? WHERE id = ?", (last_login, user["id"]))
        conn.commit()
        conn.close()

        user_data = {
            "id": user["id"],
            "full_name": user["full_name"],
            "email": user["email"],
            "username": user["username"],
            "created_at": user["created_at"],
            "last_login": last_login
        }
        return True, "Login successful!", user_data

    except sqlite3.Error as e:
        return False, f"Database error: {str(e)}", {}


def get_all_users() -> list:
    """Get all registered users (for admin purposes)"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, full_name, email, username, created_at, last_login FROM users ORDER BY created_at DESC")
        users = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return users
    except sqlite3.Error:
        return []