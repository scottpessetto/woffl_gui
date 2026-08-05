"""Request identity - who is behind this session.

Mirrors the precedence of prop_hist_client.resolve_entry_user for DISPLAY:
env WOFFL_ENTRY_USER > X-Forwarded-Email > X-Forwarded-User > None.
Databricks Apps injects the X-Forwarded-* headers; locally they are absent.
v1 performs no writes, so this identity is informational only.
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import Request


def request_user(request: Request) -> Optional[str]:
    env_user = os.environ.get("WOFFL_ENTRY_USER", "").strip()
    if env_user:
        return env_user
    for header in ("x-forwarded-email", "x-forwarded-user"):
        value = request.headers.get(header, "").strip()
        if value:
            return value
    return None
