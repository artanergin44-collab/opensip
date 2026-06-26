"""Misc helpers: branch/tag generation, IPv4 discovery, time helpers."""

from __future__ import annotations

import os
import socket
import secrets
import time

# SIP magic cookie that MUST prefix every Via branch (RFC 3261 §8.1.1.7).
BRANCH_MAGIC = "z9hG4bK"


def new_branch() -> str:
    """Generate a new, RFC-3261-compliant Via branch parameter."""
    return BRANCH_MAGIC + secrets.token_hex(8)


def new_tag() -> str:
    """Generate a From/To tag parameter."""
    return secrets.token_hex(5)


def new_call_id(host: str | None = None) -> str:
    """Generate a globally-unique Call-ID."""
    rnd = secrets.token_hex(8)
    if host:
        return f"{rnd}@{host}"
    return rnd


def now_ms() -> int:
    return int(time.time() * 1000)


def guess_local_ip(remote: tuple[str, int] | None = None) -> str:
    """Best-effort local IPv4 detection.

    If *remote* is provided, opens a UDP socket "towards" it (no packet is
    actually sent) and reads the chosen source address. Falls back to
    ``socket.gethostbyname`` and finally ``127.0.0.1``.
    """
    target = remote or ("8.8.8.8", 80)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(target)
        return s.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"
    finally:
        s.close()


def env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}
