"""HTTP auth / settings / admin client — portable for desktop and APK."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import requests

from app_config import get_backend_url


class RemoteAuthError(Exception):
    pass


def _url(path: str) -> str:
    return f"{get_backend_url()}{path}"


def signup(username: str, email: str, password: str) -> Tuple[bool, str]:
    try:
        resp = requests.post(
            _url("/auth/signup"),
            json={"username": username, "email": email, "password": password},
            timeout=20,
        )
        data = resp.json()
        return bool(data.get("success")), data.get("message", resp.text)
    except Exception as exc:
        return False, f"Cannot reach backend ({get_backend_url()}): {exc}"


def login(email: str, password: str) -> Tuple[bool, str, Optional[str], Optional[str]]:
    try:
        resp = requests.post(
            _url("/auth/login"),
            json={"email": email, "password": password},
            timeout=20,
        )
        data = resp.json()
        return (
            bool(data.get("success")),
            data.get("message", resp.text),
            data.get("email"),
            data.get("role"),
        )
    except Exception as exc:
        return False, f"Cannot reach backend ({get_backend_url()}): {exc}", None, None


def get_settings(email: str) -> Dict[str, Any]:
    try:
        resp = requests.get(_url("/auth/settings"), params={"email": email}, timeout=15)
        data = resp.json()
        return data.get("settings") or {}
    except Exception:
        return {
            "name": "User",
            "location": "Unknown Location",
            "rain": False,
            "snow": False,
            "smog": True,
        }


def save_settings(
    email: str,
    name: str,
    location: str,
    rain: bool,
    snow: bool,
    smog: bool = True,
) -> bool:
    try:
        resp = requests.put(
            _url("/auth/settings"),
            json={
                "email": email,
                "name": name,
                "location": location,
                "rain": rain,
                "snow": snow,
                "smog": smog,
            },
            timeout=15,
        )
        return bool(resp.json().get("success"))
    except Exception:
        return False


def get_weather_alerts(location: str) -> Dict[str, Any]:
    try:
        resp = requests.get(
            _url("/weather/alerts"),
            params={"location": location or "Lahore"},
            timeout=15,
        )
        return resp.json().get("weather") or {}
    except Exception:
        return {}


def get_active_broadcasts(city: str = "") -> List[Dict[str, Any]]:
    try:
        resp = requests.get(
            _url("/broadcasts/active"),
            params={"city": city or ""},
            timeout=10,
        )
        return resp.json().get("broadcasts") or []
    except Exception:
        return []


def get_admin_stats() -> Dict[str, Any]:
    try:
        return requests.get(_url("/admin/stats"), timeout=15).json()
    except Exception:
        return {}


def get_admin_devices() -> List[Dict[str, Any]]:
    try:
        return requests.get(_url("/admin/devices"), timeout=15).json().get("devices") or []
    except Exception:
        return []


def search_users(query: str = "") -> List[Dict[str, Any]]:
    try:
        return (
            requests.get(_url("/admin/users"), params={"query": query}, timeout=15)
            .json()
            .get("users")
            or []
        )
    except Exception:
        return []


def get_recent_broadcasts(limit: int = 15) -> List[Dict[str, Any]]:
    try:
        return (
            requests.get(_url("/admin/broadcasts"), params={"limit": limit}, timeout=15)
            .json()
            .get("broadcasts")
            or []
        )
    except Exception:
        return []


def get_audit_log(limit: int = 50) -> List[Dict[str, Any]]:
    try:
        return (
            requests.get(_url("/admin/audit"), params={"limit": limit}, timeout=15)
            .json()
            .get("logs")
            or []
        )
    except Exception:
        return []


def create_broadcast(
    actor_email: str,
    actor_role: str,
    city: str,
    title: str,
    message: str,
) -> Tuple[bool, str]:
    try:
        data = requests.post(
            _url("/admin/broadcast"),
            json={
                "actor_email": actor_email,
                "actor_role": actor_role,
                "city": city,
                "title": title,
                "message": message,
            },
            timeout=15,
        ).json()
        return bool(data.get("success")), data.get("message", "")
    except Exception as exc:
        return False, str(exc)


def update_device_flags(
    device_id: str,
    actor_email: str,
    actor_role: str = "admin",
    **flags: Any,
) -> Tuple[bool, str]:
    try:
        body = {
            "device_id": device_id,
            "actor_email": actor_email,
            "actor_role": actor_role,
            **flags,
        }
        data = requests.post(_url("/admin/device-flags"), json=body, timeout=15).json()
        return bool(data.get("success")), data.get("message", "")
    except Exception as exc:
        return False, str(exc)


def change_user_role_safe(
    actor_email: str,
    actor_role: str,
    target_email: str,
    new_role: str,
) -> Tuple[bool, str]:
    try:
        data = requests.post(
            _url("/admin/change-role"),
            json={
                "actor_email": actor_email,
                "actor_role": actor_role,
                "target_email": target_email,
                "new_role": new_role,
            },
            timeout=15,
        ).json()
        return bool(data.get("success")), data.get("message", "")
    except Exception as exc:
        return False, str(exc)


def remove_user_safe(
    actor_email: str,
    actor_role: str,
    actor_password: str,
    target_email: str,
) -> Tuple[bool, str]:
    try:
        data = requests.post(
            _url("/admin/delete-user"),
            json={
                "actor_email": actor_email,
                "actor_role": actor_role,
                "actor_password": actor_password,
                "target_email": target_email,
            },
            timeout=15,
        ).json()
        return bool(data.get("success")), data.get("message", "")
    except Exception as exc:
        return False, str(exc)
