"""Portable runtime config for desktop and Android APK — auto-detects backend IP."""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from typing import List, Optional
from urllib.request import Request, urlopen

_CACHED_BACKEND: Optional[str] = None
DISCOVERY_PORT = int(os.getenv("ATMOSCARE_DISCOVERY_PORT", "3847"))
DISCOVERY_QUERY = b"ATMOSCARE_DISCOVER"
DEFAULT_PORT = 8000


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _frontend_root() -> Path:
    return Path(__file__).resolve().parent


def load_dotenv_files() -> None:
    """Load .env / config.json without overwriting OS env."""
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


def _configured_urls() -> List[str]:
    load_dotenv_files()
    urls: List[str] = []
    for key in ("BACKEND_URL", "ATMOSCARE_BACKEND_URL"):
        val = (os.getenv(key) or "").strip().rstrip("/")
        if val and val.lower() not in ("auto", "*", "discover", "0.0.0.0"):
            urls.append(val)

    # Optional list in config.json
    cfg = _frontend_root() / "config.json"
    if cfg.exists():
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
            extra = data.get("BACKEND_URLS") or data.get("BACKEND_CANDIDATES") or []
            if isinstance(extra, list):
                for item in extra:
                    u = str(item).strip().rstrip("/")
                    if u and u.lower() not in ("auto", "*", "discover"):
                        urls.append(u)
        except Exception:
            pass
    return urls


def _probe_health(base_url: str, timeout: float = 1.2) -> bool:
    try:
        req = Request(f"{base_url.rstrip('/')}/health", headers={"Accept": "application/json"})
        with urlopen(req, timeout=timeout) as resp:
            return 200 <= getattr(resp, "status", 200) < 300
    except Exception:
        return False


def _udp_discover(timeout: float = 1.5) -> List[str]:
    """Broadcast LAN discovery query; collect backend URLs."""
    found: List[str] = []
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.settimeout(timeout)
    try:
        sock.sendto(DISCOVERY_QUERY, ("255.255.255.255", DISCOVERY_PORT))
        # Also try limited broadcast to common subnets via interface IP
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.connect(("8.8.8.8", 80))
                local_ip = probe.getsockname()[0]
            parts = local_ip.split(".")
            if len(parts) == 4:
                sock.sendto(DISCOVERY_QUERY, (f"{parts[0]}.{parts[1]}.{parts[2]}.255", DISCOVERY_PORT))
        except Exception:
            pass

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                data, _addr = sock.recvfrom(4096)
                payload = json.loads(data.decode("utf-8"))
                for u in payload.get("urls") or []:
                    found.append(str(u).rstrip("/"))
                if payload.get("primary"):
                    found.insert(0, str(payload["primary"]).rstrip("/"))
            except socket.timeout:
                break
            except Exception:
                continue
    finally:
        sock.close()
    # de-dupe
    seen = set()
    ordered = []
    for u in found:
        if u not in seen:
            seen.add(u)
            ordered.append(u)
    return ordered


def _local_device_ip() -> Optional[str]:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except Exception:
        return None


def _subnet_candidates(port: int = DEFAULT_PORT) -> List[str]:
    """Try gateway and a few nearby hosts on the phone/PC LAN."""
    ip = _local_device_ip()
    urls: List[str] = []
    if not ip:
        return urls
    parts = ip.split(".")
    if len(parts) != 4:
        return urls
    a, b, c, d = parts
    # Common gateway / router hosts first, then this machine
    for last in ("1", "2", d, "100", "101", "10"):
        urls.append(f"http://{a}.{b}.{c}.{last}:{port}")
    return urls


def _builtin_candidates(port: int = DEFAULT_PORT) -> List[str]:
    return [
        f"http://127.0.0.1:{port}",
        f"http://localhost:{port}",
        f"http://10.0.2.2:{port}",  # Android emulator → host machine
    ]


def discover_backend_url(force: bool = False) -> str:
    """
    Auto-detect a reachable AtmosCare backend.

    Order:
      1. Cached URL
      2. Configured BACKEND_URL (if not 'auto')
      3. UDP LAN discovery (works for APK on same Wi‑Fi)
      4. localhost / emulator
      5. Same-subnet guesses
    Falls back to http://127.0.0.1:8000 if nothing responds (desktop default).
    """
    global _CACHED_BACKEND
    if _CACHED_BACKEND and not force:
        return _CACHED_BACKEND

    load_dotenv_files()
    port = int(os.getenv("PORT", str(DEFAULT_PORT)))

    candidates: List[str] = []
    candidates.extend(_configured_urls())
    candidates.extend(_udp_discover())
    candidates.extend(_builtin_candidates(port))
    candidates.extend(_subnet_candidates(port))

    seen = set()
    for url in candidates:
        url = url.rstrip("/")
        if not url or url in seen:
            continue
        seen.add(url)
        if _probe_health(url):
            _CACHED_BACKEND = url
            print(f"[Config] Backend auto-detected: {url}")
            return url

    fallback = f"http://127.0.0.1:{port}"
    _CACHED_BACKEND = fallback
    print(f"[Config] No backend found — using {fallback} (will retry on next call)")
    return fallback


def get_backend_url() -> str:
    """Return discovered backend URL (auto-detect unless a fixed URL is set and healthy)."""
    load_dotenv_files()
    raw = (os.getenv("BACKEND_URL") or "auto").strip()
    mode = raw.lower()
    if mode in ("", "auto", "*", "discover", "0.0.0.0"):
        return discover_backend_url()
    # Fixed URL: still fall back to discovery if unreachable (phones / LAN)
    fixed = raw.rstrip("/")
    if fixed and _probe_health(fixed, timeout=1.0):
        global _CACHED_BACKEND
        _CACHED_BACKEND = fixed
        return fixed
    return discover_backend_url(force=True)


def refresh_backend_url() -> str:
    return discover_backend_url(force=True)


def get_aqi_api_url() -> str:
    load_dotenv_files()
    return (
        os.getenv("AQI_API_URL")
        or os.getenv("PAKISTAN_AQI_URL")
        or "http://127.0.0.1:3000"
    ).rstrip("/")


def kv_path(name: str) -> str:
    return str(_frontend_root() / name)


def asset_path(*parts: str) -> str:
    return str(_frontend_root().joinpath("assets", *parts))
