from Backend.database import (
    add_user, validate_user,
    get_user_settings, save_user_settings,
    get_user_role, get_all_users, update_user_role, delete_user as db_delete_user,
    is_database_available,
)


def handle_signup(username, email, password):
    """Create a new user account with default user role.
    
    Returns (success: bool, message: str).
    """
    if not username or not email or not password:
        return False, "All fields are required!"
    if not is_database_available():
        return False, "Database connection failed. Make sure MongoDB is running locally."
    return add_user(username, email, password)


def handle_login(email, password):
    """Validate credentials and return a 4-tuple.
    
    Returns (success: bool, message: str, email: str|None, role: str|None).
    """
    if not email or not password:
        return False, "All fields are required!", None, None
    if not is_database_available():
        return False, "Database connection failed. Make sure MongoDB is running locally.", None, None

    user_doc = validate_user(email, password)
    if user_doc:
        role = user_doc.get("role", "user")
        return True, "Login successful!", email, role
    else:
        return False, "Invalid credentials", None, None


def get_settings(email):
    """Get user settings from database."""
    return get_user_settings(email)


def save_settings(email, name, location, rain, snow, smog=True):
    """Save user settings to database."""
    return save_user_settings(email, name, location, rain, snow, smog)


# ── Role management wrappers ──────────────────────────────────────────────────

def fetch_user_role(email):
    """Return the role string for a given email."""
    return get_user_role(email)


def fetch_all_users():
    """Return list of all user dicts (email, username, role)."""
    return get_all_users()


def change_user_role(email, new_role):
    """Update the role of a user. Returns True on success."""
    return update_user_role(email, new_role)


def remove_user(email):
    """Delete a user from the database. Returns True on success."""
    return db_delete_user(email)


# ── Admin panel wrappers ──────────────────────────────────────────────────────

from Backend import admin_service as _admin  # noqa: E402


def get_admin_stats():
    return _admin.get_admin_stats()


def get_admin_devices():
    return _admin.get_all_devices_admin()


def update_device_flags(device_id, actor_email, actor_role="admin", **kwargs):
    return _admin.update_device_flags(device_id, actor_email, actor_role, **kwargs)


def change_user_role_safe(actor_email, actor_role, target_email, new_role):
    return _admin.change_user_role_safe(actor_email, actor_role, target_email, new_role)


def remove_user_safe(actor_email, actor_role, actor_password, target_email):
    return _admin.remove_user_safe(actor_email, actor_role, actor_password, target_email)


def create_broadcast(actor_email, actor_role, city, title, message):
    return _admin.create_broadcast(actor_email, actor_role, city, title, message)


def get_active_broadcasts(city=""):
    return _admin.get_active_broadcasts(city)


def get_recent_broadcasts(limit=15):
    return _admin.get_recent_broadcasts(limit)


def get_audit_log(limit=50):
    return _admin.get_audit_log(limit)


def search_users(query=""):
    return _admin.search_users(query)
