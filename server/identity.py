"""Request identity - who is behind this session.

Mirrors the precedence of prop_hist_client.resolve_entry_user:
env WOFFL_ENTRY_USER > X-Forwarded-Email > X-Forwarded-User > None.
Databricks Apps injects the X-Forwarded-* headers; locally they are absent.

Write attribution: prop_hist writes stamp entry_user via
``prop_hist_client.resolve_entry_user()``, which consults the provider
registered below. The provider reads a ContextVar bound per request
(``bind_entry_user``), so concurrent requests on the shared host each stamp
their OWN engineer - an env var cannot do that (process-global). This is the
FastAPI equivalent of the Streamlit app's ``set_entry_user_provider``
registration in app.py.
"""

from __future__ import annotations

import os
from contextvars import ContextVar
from typing import Optional

from fastapi import Request

from woffl.assembly.prop_hist_client import set_entry_user_provider

_entry_user: ContextVar[Optional[str]] = ContextVar("woffl_entry_user", default=None)


def request_user(request: Request) -> Optional[str]:
    env_user = os.environ.get("WOFFL_ENTRY_USER", "").strip()
    if env_user:
        return env_user
    for header in ("x-forwarded-email", "x-forwarded-user"):
        value = request.headers.get(header, "").strip()
        if value:
            return value
    return None


def bind_entry_user(request: Request) -> Optional[str]:
    """Bind the acting user to the current request context.

    MUST be called at the top of every write endpoint, before any
    ``push_prop``. Falsy binds are fine: resolve_entry_user falls through to
    the SQL session's current_user() (the service principal) as last resort.
    """
    user = request_user(request)
    _entry_user.set(user)
    return user


def _provider() -> Optional[str]:
    return _entry_user.get()


def register_entry_user_provider() -> None:
    """(Re-)register the request-context provider.

    Runs at import (below). The provider slot in prop_hist_client is
    process-global and single-occupancy; tests that exercise it reset it to
    None, so anything sharing a process with such tests re-registers via
    this hook. FastAPI runs sync endpoints in a threadpool that inherits the
    request's contextvars, so the provider sees the value bound by
    bind_entry_user for THIS request only. Falsy/broken provider returns
    fall through to the next precedence tier inside resolve_entry_user.
    """
    set_entry_user_provider(_provider)


register_entry_user_provider()
