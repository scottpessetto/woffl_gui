"""Thread-safe TTL cache - the server-side replacement for @st.cache_data.

Semantics preserved from the Streamlit app (AGENTS.md section 6):
- Failures are NEVER cached: exceptions propagate without storing.
- Keys are the positional/keyword args as passed, so callers must pass
  cache-key args explicitly (no defaulted-arg aliasing).
- TTLs mirror the Streamlit sites: fleet well tests 24h, chars/PF 1h,
  saved IPR 5min, profiles/surveys 1h.

Beyond Streamlit - stale-while-revalidate (SWR): an expired entry is served
IMMEDIATELY while a background refresh replaces it, so TTL expiry never
lands on a user request. Rules:
- Stale is served for up to one extra TTL (the grace window); beyond that a
  read blocks and fetches like a cold miss.
- Refreshes are single-flight per key and run on a small persistent thread
  pool - persistent threads matter on Databricks, where the client keeps
  one connection per thread (spawn-per-refresh would handshake every time).
- A failed refresh keeps serving stale within the grace window and clears
  the in-flight latch so the next stale read retries.
- clear() bumps a version; any fetch that started before the clear cannot
  store its (possibly pre-write) result afterwards. This preserves the
  read-your-writes contract of the save/clear-pin/lock endpoints.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

log = logging.getLogger("woffl.web.cache")

_REGISTRY: list["_TtlCache"] = []
_REGISTRY_LOCK = threading.Lock()

# Shared refresher: persistent daemon threads (Databricks connections are
# thread-local; reusing threads reuses their warehouse connections).
_REFRESH_POOL: ThreadPoolExecutor | None = None
_REFRESH_POOL_LOCK = threading.Lock()


def _refresh_pool() -> ThreadPoolExecutor:
    global _REFRESH_POOL
    with _REFRESH_POOL_LOCK:
        if _REFRESH_POOL is None:
            _REFRESH_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cache-swr")
        return _REFRESH_POOL


class _TtlCache:
    def __init__(self, ttl: float, maxsize: int) -> None:
        self.ttl = ttl
        self.maxsize = maxsize
        # key -> (fresh_until, stale_until, value)
        self._data: OrderedDict[tuple, tuple[float, float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._refreshing: set[tuple] = set()
        self.version = 0

    def get(self, key: tuple) -> tuple[str, Any]:
        """Returns (state, value); state is "fresh" | "stale" | "miss"."""
        now = time.monotonic()
        with self._lock:
            hit = self._data.get(key)
            if hit is None:
                return "miss", None
            fresh_until, stale_until, value = hit
            if now < fresh_until:
                self._data.move_to_end(key)
                return "fresh", value
            if now < stale_until:
                return "stale", value
            del self._data[key]
            return "miss", None

    def put(self, key: tuple, value: Any, version: int) -> bool:
        """Store unless clear() ran after the fetch began (version bump)."""
        with self._lock:
            if version != self.version:
                return False
            now = time.monotonic()
            self._data[key] = (now + self.ttl, now + 2 * self.ttl, value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)
            return True

    def try_begin_refresh(self, key: tuple) -> bool:
        with self._lock:
            if key in self._refreshing:
                return False
            self._refreshing.add(key)
            return True

    def end_refresh(self, key: tuple) -> None:
        with self._lock:
            self._refreshing.discard(key)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.version += 1

    def evict(self, key: tuple) -> None:
        """Drop ONE key. Bumps the version like clear(), so an in-flight fetch
        that started before this call cannot store a pre-write result — the
        same read-your-writes guarantee, without costing every other key its
        entry (a one-well save used to cold-start the whole fleet)."""
        with self._lock:
            self._data.pop(key, None)
            self.version += 1


def ttl_cache(ttl: float, maxsize: int = 32) -> Callable[[F], F]:
    """Decorator. Per-key single-flight on COLD misses is deliberately NOT
    implemented: concurrent cold misses recompute in parallel, matching
    Streamlit behavior; reads are idempotent SELECTs and the solver wrappers
    are pure. Background refreshes (the SWR path) ARE single-flight."""

    def decorate(fn: F) -> F:
        cache = _TtlCache(ttl, maxsize)
        with _REGISTRY_LOCK:
            _REGISTRY.append(cache)

        def _refresh(key: tuple, args: tuple, kwargs: dict, version: int) -> None:
            try:
                value = fn(*args, **kwargs)
                cache.put(key, value, version)
            except Exception as exc:  # noqa: BLE001 - stale keeps serving; next read retries
                log.warning("swr refresh failed (%s): %s", getattr(fn, "__name__", fn), exc)
            finally:
                cache.end_refresh(key)

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = (args, tuple(sorted(kwargs.items())))
            state, value = cache.get(key)
            if state == "fresh":
                return value
            if state == "stale":
                if cache.try_begin_refresh(key):
                    version = cache.version
                    try:
                        _refresh_pool().submit(_refresh, key, args, dict(kwargs), version)
                    except RuntimeError:  # interpreter shutdown
                        cache.end_refresh(key)
                return value
            version = cache.version
            value = fn(*args, **kwargs)
            cache.put(key, value, version)
            return value

        def cache_evict(*args: Any, **kwargs: Any) -> None:
            """Drop the entry for ONE call signature. Args must match the
            decorated call exactly (same positional/keyword split), because the
            key is built the same way as in `wrapper`."""
            cache.evict((args, tuple(sorted(kwargs.items()))))

        wrapper.cache_evict = cache_evict  # type: ignore[attr-defined]
        wrapper.cache_clear = cache.clear  # type: ignore[attr-defined]
        wrapper._cache = cache  # type: ignore[attr-defined] - test/ops introspection
        return wrapper  # type: ignore[return-value]

    return decorate


def clear_all_caches() -> int:
    """Testing/ops hook. Returns the number of caches cleared."""
    with _REGISTRY_LOCK:
        for cache in _REGISTRY:
            cache.clear()
        return len(_REGISTRY)
