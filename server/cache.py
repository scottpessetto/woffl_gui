"""Thread-safe TTL cache - the server-side replacement for @st.cache_data.

Semantics preserved from the Streamlit app (AGENTS.md section 6):
- Failures are NEVER cached: exceptions propagate without storing.
- Keys are the positional/keyword args as passed, so callers must pass
  cache-key args explicitly (no defaulted-arg aliasing).
- TTLs mirror the Streamlit sites: fleet well tests 24h, chars/PF 1h,
  saved IPR 5min, profiles/surveys 1h.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from functools import wraps
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

_REGISTRY: list["_TtlCache"] = []
_REGISTRY_LOCK = threading.Lock()


class _TtlCache:
    def __init__(self, ttl: float, maxsize: int) -> None:
        self.ttl = ttl
        self.maxsize = maxsize
        self._data: OrderedDict[tuple, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: tuple) -> tuple[bool, Any]:
        now = time.monotonic()
        with self._lock:
            hit = self._data.get(key)
            if hit is None:
                return False, None
            expires, value = hit
            if expires < now:
                del self._data[key]
                return False, None
            self._data.move_to_end(key)
            return True, value

    def put(self, key: tuple, value: Any) -> None:
        with self._lock:
            self._data[key] = (time.monotonic() + self.ttl, value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


def ttl_cache(ttl: float, maxsize: int = 32) -> Callable[[F], F]:
    """Decorator. Per-key single-flight is deliberately NOT implemented:
    concurrent misses recompute in parallel, matching Streamlit behavior;
    reads are idempotent SELECTs and the solver wrappers are pure."""

    def decorate(fn: F) -> F:
        cache = _TtlCache(ttl, maxsize)
        with _REGISTRY_LOCK:
            _REGISTRY.append(cache)

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = (args, tuple(sorted(kwargs.items())))
            ok, value = cache.get(key)
            if ok:
                return value
            value = fn(*args, **kwargs)
            cache.put(key, value)
            return value

        wrapper.cache_clear = cache.clear  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorate


def clear_all_caches() -> int:
    """Testing/ops hook. Returns the number of caches cleared."""
    with _REGISTRY_LOCK:
        for cache in _REGISTRY:
            cache.clear()
        return len(_REGISTRY)
