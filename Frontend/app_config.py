"""Portable runtime config for desktop and Android APK."""

from __future__ import annotations

import json
import os
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _frontend_root() -> Path:
    return Path(__file__).resolve().parent


def load_dotenv_files() -> None:
    """Load .env from Frontend/, project root, and Backend/ without overwriting OS env."""
    candidates = [
        _frontend_root() / ".env",
        _project_root() / ".env",
        _project_root() / "portable.env",
        _project_root() / "Backend" / ".env",
        _frontend_root() / "config.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix.lower() == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    for key, value in data.items():
                        if key and value is not None and key not in os.environ:
                            os.environ[str(key)] = str(value)
            except Exception:
                pass
            continue
        try:
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        except Exception:
            pass


def get_backend_url() -> str:
    load_dotenv_files()
    return (
        os.getenv("BACKEND_URL")
        or os.getenv("ATMOSCARE_BACKEND_URL")
        or "http://127.0.0.1:8000"
    ).rstrip("/")


def get_aqi_api_url() -> str:
    load_dotenv_files()
    return (
        os.getenv("AQI_API_URL")
        or os.getenv("PAKISTAN_AQI_URL")
        or "http://127.0.0.1:3000"
    ).rstrip("/")


def kv_path(name: str) -> str:
    """Absolute path to a KV file next to main.py (works regardless of CWD)."""
    return str(_frontend_root() / name)


def asset_path(*parts: str) -> str:
    return str(_frontend_root().joinpath("assets", *parts))
