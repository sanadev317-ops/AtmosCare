"""LAN discovery: answer UDP probes so phones/PCs find the backend without hardcoding IPs."""

from __future__ import annotations

import json
import os
import socket
import threading
from typing import List

DISCOVERY_PORT = int(os.getenv("ATMOSCARE_DISCOVERY_PORT", "3847"))
DISCOVERY_QUERY = b"ATMOSCARE_DISCOVER"
API_PORT = int(os.getenv("PORT", "8000"))


def local_ipv4_addresses() -> List[str]:
    """Return all non-loopback IPv4 addresses for this machine."""
    found: set[str] = set()
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith("127."):
                found.add(ip)
    except Exception:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            found.add(sock.getsockname()[0])
    except Exception:
        pass
    return sorted(found)


def public_base_urls(api_port: int | None = None) -> List[str]:
    port = api_port or API_PORT
    urls = [f"http://127.0.0.1:{port}", f"http://localhost:{port}"]
    for ip in local_ipv4_addresses():
        urls.append(f"http://{ip}:{port}")
    # de-dupe preserve order
    seen = set()
    ordered = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            ordered.append(u)
    return ordered


def _listen() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    except Exception:
        pass
    try:
        sock.bind(("", DISCOVERY_PORT))
    except OSError as exc:
        print(f"[Discovery] UDP bind failed on {DISCOVERY_PORT}: {exc}")
        return

    print(f"[Discovery] Listening on UDP {DISCOVERY_PORT} — LAN phones can auto-find this backend")
    while True:
        try:
            data, addr = sock.recvfrom(1024)
            if not data.startswith(DISCOVERY_QUERY):
                continue
            payload = {
                "service": "AtmosCare",
                "port": API_PORT,
                "urls": public_base_urls(API_PORT),
                "primary": public_base_urls(API_PORT)[-1] if local_ipv4_addresses() else f"http://127.0.0.1:{API_PORT}",
            }
            sock.sendto(json.dumps(payload).encode("utf-8"), addr)
            print(f"[Discovery] Replied to {addr[0]}")
        except Exception as exc:
            print(f"[Discovery] error: {exc}")


def start_discovery_server() -> None:
    thread = threading.Thread(target=_listen, name="atmoscare-discovery", daemon=True)
    thread.start()
