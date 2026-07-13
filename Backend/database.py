from pymongo import MongoClient
from pymongo.errors import PyMongoError

from Backend.config import DATABASE_NAME, DATABASE_URI

if not DATABASE_URI:
    DATABASE_URI = "mongodb://localhost:27017"

# Connect to MongoDB
client = MongoClient(DATABASE_URI, serverSelectionTimeoutMS=3000)

# Create or use database
db = client[DATABASE_NAME]
users_collection = db["users"]
air_quality_collection = db["air_quality"]


def is_database_available() -> bool:
    """Return True when the local MongoDB server is reachable."""
    try:
        client.admin.command("ping")
        return True
    except PyMongoError:
        return False


def add_user(username, email, password, role="user"):
    """Insert new user into MongoDB.
    
    The very first user ever registered is automatically promoted to 'admin'
    regardless of the requested role.
    """
    try:
        existing = users_collection.find_one({"email": email})
        if existing:
            return False, "Email already exists!"

        # Auto-promote first user to admin
        if users_collection.count_documents({}) == 0:
            role = "admin"

        users_collection.insert_one({
            "username": username,
            "email": email,
            "password": password,
            "role": role,          # "user" | "authority" | "admin"
        })
        return True, "Signup successful!"
    except PyMongoError:
        return False, "Database connection failed. Make sure MongoDB is running locally."


def validate_user(email, password):
    """Validate login credentials.
    
    Returns the full user document if valid, or None on failure.
    """
    try:
        user = users_collection.find_one({"email": email, "password": password})
        return user  # None when invalid
    except PyMongoError:
        return None


def get_user_role(email):
    """Return the role string for the given email, or 'user' as a safe default."""
    try:
        user = users_collection.find_one({"email": email}, {"role": 1})
        if user:
            return user.get("role", "user")
    except PyMongoError:
        pass
    return "user"


def get_all_users():
    """Return a list of all users as dicts with keys: email, username, role."""
    try:
        return list(
            users_collection.find(
                {},
                {"_id": 0, "email": 1, "username": 1, "role": 1}
            )
        )
    except PyMongoError:
        return []


def update_user_role(email, new_role):
    """Change the role of the user identified by email."""
    try:
        result = users_collection.update_one(
            {"email": email},
            {"$set": {"role": new_role}}
        )
        return result.modified_count > 0
    except PyMongoError:
        return False


def delete_user(email):
    """Permanently remove a user from the database."""
    try:
        result = users_collection.delete_one({"email": email})
        return result.deleted_count > 0
    except PyMongoError:
        return False


def get_user_settings(email):
    """Get user settings from database."""
    try:
        user = users_collection.find_one({"email": email})
    except PyMongoError:
        return {
            "name": "User",
            "location": "Unknown Location",
            "rain": False,
            "snow": False
        }
    if user and "settings" in user:
        settings = user["settings"]
        # If location exists and is not dummy, return it
        if settings.get("location") and settings.get("location") != "Lahore":
            return settings
        # If location is dummy or missing, try to get real location
        from Backend.location_service import get_location_from_ip
        real_location = get_location_from_ip()
        if real_location:
            settings["location"] = real_location
            users_collection.update_one(
                {"email": email},
                {"$set": {"settings.location": real_location}},
                upsert=False
            )
            return settings

    # Return default settings if none exist, but try to get real location
    from Backend.location_service import get_location_from_ip
    real_location = get_location_from_ip()
    default_location = real_location if real_location else "Unknown Location"

    return {
        "name": user.get("username", "User") if user else "User",
        "location": default_location,
        "rain": False,
        "snow": False,
        "smog": True,
    }


def save_user_settings(email, name, location, rain, snow, smog=True):
    """Save user settings to database."""
    settings = {
        "name": name,
        "location": location,
        "rain": rain,
        "snow": snow,
        "smog": smog,
    }
    try:
        users_collection.update_one(
            {"email": email},
            {"$set": {"settings": settings}},
            upsert=False
        )
        return True
    except PyMongoError:
        return False


def save_air_quality_data(location, air_quality_data):
    """Save air quality data to database."""
    try:
        air_quality_collection.update_one(
            {"location": location},
            {"$set": {
                **air_quality_data,
                "updated_at": air_quality_data.get("timestamp")
            }},
            upsert=True
        )
        return True
    except PyMongoError:
        return False


def get_air_quality_data(location):
    """Get latest air quality data from database."""
    try:
        data = air_quality_collection.find_one({"location": location}, sort=[("updated_at", -1)])
        return data
    except PyMongoError:
        return None
