"""Admin panel business logic: users, devices, broadcasts, audit log."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from pymongo import DESCENDING
from pymongo.errors import PyMongoError

from Backend.database import client, db, users_collection, validate_user

devices_collection = db["devices"]
iot_data_collection = db["iot_data"]
audit_logs_collection = db["audit_logs"]
broadcasts_collection = db["broadcasts"]

VALID_ROLES = ("user", "authority", "admin")
BROADCAST_TTL_HOURS = 24


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _safe_city(location: str) -> str:
    if not location:
        return ""
    return str(location).split(",")[0].strip()


def _count_admins() -> int:
    try:
        return users_collection.count_documents({"role": "admin"})
    except PyMongoError:
        return 0


def log_audit(actor_email: str, action: str, target: str = "", details: str = "") -> None:
    try:
        audit_logs_collection.insert_one({
            "actor": actor_email,
            "action": action,
            "target": target,
            "details": details,
            "timestamp": _utcnow(),
        })
    except PyMongoError:
        pass


def get_audit_log(limit: int = 50) -> List[Dict[str, Any]]:
    try:
        rows = list(
            audit_logs_collection.find({}, {"_id": 0})
            .sort("timestamp", DESCENDING)
            .limit(limit)
        )
        for row in rows:
            ts = row.get("timestamp")
            if isinstance(ts, datetime):
                row["timestamp"] = ts.astimezone(timezone.utc).strftime("%d %b %Y, %I:%M %p")
        return rows
    except PyMongoError:
        return []


def get_admin_stats() -> Dict[str, Any]:
    try:
        users = list(users_collection.find({}, {"_id": 0, "email": 1, "role": 1, "settings": 1}))
    except PyMongoError:
        return {
            "total_users": 0,
            "role_counts": {},
            "city_counts": {},
            "total_devices": 0,
            "active_devices": 0,
            "disabled_devices": 0,
        }

    role_counts = Counter(u.get("role", "user") for u in users)
    city_counts: Counter = Counter()
    for u in users:
        loc = (u.get("settings") or {}).get("location", "")
        city = _safe_city(loc)
        if city and city != "Unknown Location":
            city_counts[city] += 1

    try:
        device_docs = list(devices_collection.find({}, {"admin_disabled": 1, "buffer": 1}))
        total_devices = len(device_docs)
        disabled = sum(1 for d in device_docs if d.get("admin_disabled"))
        active = sum(1 for d in device_docs if (d.get("buffer") or []) and not d.get("admin_disabled"))
    except PyMongoError:
        total_devices = active = disabled = 0

    return {
        "total_users": len(users),
        "role_counts": dict(role_counts),
        "city_counts": dict(city_counts.most_common(10)),
        "total_devices": total_devices,
        "active_devices": active,
        "disabled_devices": disabled,
        "admin_count": role_counts.get("admin", 0),
    }


def get_all_devices_admin() -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    try:
        docs = list(devices_collection.find({}).sort("updated_at", DESCENDING))
    except PyMongoError:
        return results

    for doc in docs:
        device_id = doc.get("device_id", "")
        buffer = doc.get("buffer") or []
        last_event = buffer[-1] if buffer else {}

        last_iot = None
        try:
            last_iot = iot_data_collection.find_one(
                {"device_id": device_id},
                sort=[("ingested_at", DESCENDING)],
            )
        except PyMongoError:
            pass

        ts = last_event.get("timestamp") or (last_iot or {}).get("ingested_at")
        if isinstance(ts, datetime):
            ts_str = ts.astimezone(timezone.utc).strftime("%d %b, %I:%M %p")
        else:
            ts_str = str(ts) if ts else "—"

        results.append({
            "device_id": device_id,
            "location": doc.get("location") or "—",
            "status": doc.get("status", "active"),
            "admin_disabled": bool(doc.get("admin_disabled")),
            "force_api": bool(doc.get("force_api")),
            "marked_test": bool(doc.get("marked_test")),
            "last_seen": ts_str,
            "pm2_5": last_event.get("pm2_5") or last_event.get("pm25") or (last_iot or {}).get("pm2_5"),
            "pm10": last_event.get("pm10") or (last_iot or {}).get("pm10"),
            "buffer_size": len(buffer),
        })
    return results


def update_device_flags(
    device_id: str,
    actor_email: str,
    actor_role: str = "admin",
    *,
    admin_disabled: Optional[bool] = None,
    force_api: Optional[bool] = None,
    marked_test: Optional[bool] = None,
) -> Tuple[bool, str]:
    if actor_role != "admin":
        return False, "Only admins can manage devices."
    updates: Dict[str, Any] = {"updated_at": _utcnow()}
    if admin_disabled is not None:
        updates["admin_disabled"] = admin_disabled
    if force_api is not None:
        updates["force_api"] = force_api
    if marked_test is not None:
        updates["marked_test"] = marked_test

    if len(updates) <= 1:
        return False, "No changes requested."

    try:
        devices_collection.update_one(
            {"device_id": device_id},
            {"$set": updates, "$setOnInsert": {"device_id": device_id, "buffer": []}},
            upsert=True,
        )
        parts = []
        if admin_disabled is not None:
            parts.append(f"disabled={admin_disabled}")
        if force_api is not None:
            parts.append(f"force_api={force_api}")
        if marked_test is not None:
            parts.append(f"marked_test={marked_test}")
        detail = ", ".join(parts)
        log_audit(actor_email, "device_update", device_id, detail)
        return True, "Device updated."
    except PyMongoError:
        return False, "Database error."


def change_user_role_safe(
    actor_email: str,
    actor_role: str,
    target_email: str,
    new_role: str,
) -> Tuple[bool, str]:
    if actor_role != "admin":
        return False, "Only admins can change roles."
    if new_role not in VALID_ROLES:
        return False, "Invalid role."
    if actor_email == target_email:
        return False, "You cannot change your own role."

    target = users_collection.find_one({"email": target_email}, {"role": 1})
    if not target:
        return False, "User not found."

    old_role = target.get("role", "user")
    if old_role == "admin" and new_role != "admin" and _count_admins() <= 1:
        return False, "Cannot demote the last admin."

    try:
        result = users_collection.update_one(
            {"email": target_email},
            {"$set": {"role": new_role}},
        )
        if result.modified_count == 0 and old_role == new_role:
            return True, "Role unchanged."
        if result.modified_count == 0:
            return False, "Update failed."
        log_audit(
            actor_email, "role_change", target_email,
            f"{old_role} → {new_role}",
        )
        return True, f"Role updated to '{new_role}'."
    except PyMongoError:
        return False, "Database error."


def remove_user_safe(
    actor_email: str,
    actor_role: str,
    actor_password: str,
    target_email: str,
) -> Tuple[bool, str]:
    if actor_role != "admin":
        return False, "Only admins can delete users."
    if actor_email == target_email:
        return False, "You cannot delete your own account."
    if not validate_user(actor_email, actor_password):
        return False, "Incorrect password."

    target = users_collection.find_one({"email": target_email}, {"role": 1})
    if not target:
        return False, "User not found."
    if target.get("role") == "admin" and _count_admins() <= 1:
        return False, "Cannot delete the last admin."

    try:
        result = users_collection.delete_one({"email": target_email})
        if result.deleted_count == 0:
            return False, "Delete failed."
        log_audit(actor_email, "user_delete", target_email, "permanent")
        return True, "User deleted."
    except PyMongoError:
        return False, "Database error."


def create_broadcast(
    actor_email: str,
    actor_role: str,
    city: str,
    title: str,
    message: str,
) -> Tuple[bool, str]:
    if actor_role not in ("admin", "authority"):
        return False, "Insufficient permissions."
    if not title.strip() or not message.strip():
        return False, "Title and message are required."

    city_norm = city.strip() or "*"
    now = _utcnow()
    try:
        broadcasts_collection.insert_one({
            "city": city_norm,
            "title": title.strip(),
            "message": message.strip(),
            "created_by": actor_email,
            "created_at": now,
            "expires_at": now + timedelta(hours=BROADCAST_TTL_HOURS),
            "active": True,
        })
        log_audit(actor_email, "broadcast", city_norm, title.strip())
        return True, f"Advisory sent for {city_norm}."
    except PyMongoError:
        return False, "Database error."


def get_active_broadcasts(city: str = "") -> List[Dict[str, Any]]:
    now = _utcnow()
    city_key = _safe_city(city)
    try:
        query: Dict[str, Any] = {
            "active": True,
            "expires_at": {"$gte": now},
        }
        rows = list(broadcasts_collection.find(query).sort("created_at", DESCENDING).limit(20))
    except PyMongoError:
        return []

    out: List[Dict[str, Any]] = []
    for row in rows:
        bc_city = row.get("city", "*")
        if bc_city not in ("*", "") and city_key and bc_city.lower() != city_key.lower():
            continue
        ts = row.get("created_at")
        if isinstance(ts, datetime):
            ts_str = ts.astimezone(timezone.utc).strftime("%d %b, %I:%M %p")
        else:
            ts_str = "—"
        out.append({
            "city": bc_city,
            "title": row.get("title", "Advisory"),
            "message": row.get("message", ""),
            "created_by": row.get("created_by", ""),
            "created_at": ts_str,
            "_id": str(row.get("_id", "")),
        })
    return out


def get_recent_broadcasts(limit: int = 15) -> List[Dict[str, Any]]:
    try:
        rows = list(
            broadcasts_collection.find({}, {"_id": 0})
            .sort("created_at", DESCENDING)
            .limit(limit)
        )
        for row in rows:
            ts = row.get("created_at")
            if isinstance(ts, datetime):
                row["created_at"] = ts.astimezone(timezone.utc).strftime("%d %b, %I:%M %p")
            exp = row.get("expires_at")
            if isinstance(exp, datetime):
                row["expires_at"] = exp.astimezone(timezone.utc).strftime("%d %b, %I:%M %p")
                row["is_active"] = exp >= _utcnow() and row.get("active", True)
        return rows
    except PyMongoError:
        return []


def search_users(query: str = "") -> List[Dict[str, Any]]:
    try:
        users = list(
            users_collection.find(
                {},
                {"_id": 0, "email": 1, "username": 1, "role": 1, "settings.location": 1},
            )
        )
    except PyMongoError:
        return []

    q = query.strip().lower()
    if q:
        users = [
            u for u in users
            if q in (u.get("email") or "").lower()
            or q in (u.get("username") or "").lower()
            or q in ((u.get("settings") or {}).get("location") or "").lower()
        ]
    for u in users:
        u["location"] = (u.get("settings") or {}).get("location", "—")
    return users
