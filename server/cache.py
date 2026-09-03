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
- EXCEPT for entries a warm loop owns: ``cache_refresh`` stores with a
  retention floor (``set_warm_retention``), so an entry the warmup rewrites
  every N hours stays servable for that whole span instead of being deleted
  one TTL later. Freshness is unchanged - a read past the TTL still gets the
  stale value plus a background refresh - only the "delete, then block the
  next reader" cliff goes away. That cliff is what made a warm interval have
  to be shorter than the shortest TTL it protected.
- Refreshes are single-flight per key and run on a small persistent thread
  pool - persistent threads matter on Databricks, where the client keeps
  one connection per thread (spawn-per-refresh would handshake every time).
- A failed refresh keeps serving stale within the grace window and clears
  the in-flight latch so the next stale read retries.
- clear() bumps a version; any fetch that started before the clear cannot
  store its (possibly pre-write) result afterwards. This preserves the
  read-your-writes contract of the save/clear-pin/lock endpoints.
- ``cache_refresh(*args)`` is the warm loop's WRITE path: it re-fetches
  unconditionally and overwrites the entry, where a plain call short-circuits
  on a fresh entry and refreshes nothing. It is single-flight against the SWR
  latch, so a warm pass and a stale read never duplicate one query.
- ``cache_prime(value, *args)`` is the same write with the fetch already done
  elsewhere: it stores a value the caller obtained by OTHER means under the
  key the wrapper would have built, with the warm retention floor. It exists
  because one fleet-wide query can answer ~90 per-well cache keys, and paying
  90 per-well queries to fill them is exactly the warehouse bill this cache is
  supposed to avoid. Pass ``version=`` captured BEFORE the shared fetch so the
  clear() guard covers it just as it covers an in-flight fetch.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import partial, wraps
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

log = logging.getLogger("woffl.web.cache")

_REGISTRY: list["_TtlCache"] = []
_REGISTRY_LOCK = threading.Lock()

# Shared refresher: persistent daemon threads (Databricks connections are
# thread-local; reusing threads reuses their warehouse connections).
_REFRESH_POOL: ThreadPoolExecutor | None = None
_REFRESH_POOL_LOCK = threading.Lock()

# Retention floor for entries written by `cache_refresh` - the warm loop's
# promise that it will rewrite them before this expires. Seconds; 0 keeps the
# plain SWR grace (2 x TTL). Owned by server.warmup (set once at startup),
# which lives here so cache.py imports nothing from it.
_WARM_RETENTION = 0.0
_WARM_RETENTION_LOCK = threading.Lock()


def set_warm_retention(seconds: float) -> None:
    """Guarantee `cache_refresh`-written entries stay servable this long.

    Called by the warm loop with a span covering its own interval, so a warmed
    entry can never be deleted between two passes and land a cold, blocking
    query on a user. Applies to entries written AFTER the call.
    """
    global _WARM_RETENTION
    with _WARM_RETENTION_LOCK:
        _WARM_RETENTION = max(0.0, float(seconds))


def warm_retention() -> float:
    with _WARM_RETENTION_LOCK:
        return _WARM_RETENTION


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

    def put(self, key: tuple, value: Any, version: int, retention: float = 0.0) -> bool:
        """Store unless clear() ran after the fetch began (version bump).

        `retention` is a floor on how long the entry stays SERVABLE (stale
        included), not on how long it stays fresh: a warm-owned entry still
        goes stale on its TTL and still triggers an SWR refresh on the next
        read, it just is not deleted out from under the next reader.
        """
        with self._lock:
            if version != self.version:
                return False
            now = time.monotonic()
            servable = max(2 * self.ttl, retention)
            self._data[key] = (now + self.ttl, now + servable, value)
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

        def cache_refresh(*args: Any, **kwargs: Any) -> bool:
            """Re-fetch NOW and overwrite the entry. The warm loop's write path.

            A plain call cannot warm anything that is still fresh - it returns
            the cached value and queries nothing - so a loop built out of plain
            calls only ever refreshes on the pass that happens to observe a
            stale entry. That is why the old warm interval had to sit under the
            shortest TTL. This forces the query and stores with the warm
            retention floor, which decouples "how often we re-query" from "how
            long a value stays servable".

            Single-flight against the SWR latch: returns False without querying
            when a refresh for this key is already in flight. Fetch failures
            propagate (the caller counts them) and leave the old entry intact.
            """
            key = (args, tuple(sorted(kwargs.items())))
            if not cache.try_begin_refresh(key):
                return False
            try:
                version = cache.version
                cache.put(key, fn(*args, **kwargs), version, retention=warm_retention())
            finally:
                cache.end_refresh(key)
            return True

        def cache_version() -> int:
            """The cache's current clear/evict counter.

            A ``cache_prime`` caller captures this BEFORE its shared fetch and
            hands it back, so a write that clears the cache mid-fetch discards
            the prime - the same read-your-writes guarantee an ordinary
            in-flight fetch gets.
            """
            return cache.version

        def cache_prime(
            value: Any, *args: Any, version: int | None = None, **kwargs: Any
        ) -> bool:
            """Store `value` under this call signature WITHOUT calling `fn`.

            The warm loop's bulk write path: one fleet-wide query answers every
            well's key, so the loop can fill ~90 entries for 1 statement instead
            of 90. The entry is indistinguishable from one `cache_refresh`
            wrote - same key, same warm retention floor - so the request path
            reads it exactly as it would a warmed per-well fetch.

            Single-flight against the SWR latch like `cache_refresh`: returns
            False without storing when a refresh for this key is in flight
            (that refresh is about to store a value fetched at least as late as
            this one). `version` is the counter captured before the caller's
            shared fetch; omitted, the current one is used, which only guards
            against a clear racing the store itself.

            Returns:
                True when the value was stored.
            """
            key = (args, tuple(sorted(kwargs.items())))
            if not cache.try_begin_refresh(key):
                return False
            try:
                stamp = cache.version if version is None else version
                return cache.put(key, value, stamp, retention=warm_retention())
            finally:
                cache.end_refresh(key)

        def cache_has(*args: Any, **kwargs: Any) -> bool:
            """True when an entry for this call signature is servable (fresh
            OR stale) - a peek that never queries. Lets a caller derive a
            narrower result from a broader cached one instead of re-fetching."""
            state, _value = cache.get((args, tuple(sorted(kwargs.items()))))
            return state in ("fresh", "stale")

        wrapper.cache_refresh = cache_refresh  # type: ignore[attr-defined]
        wrapper.cache_prime = cache_prime  # type: ignore[attr-defined]
        wrapper.cache_version = cache_version  # type: ignore[attr-defined]
        wrapper.cache_evict = cache_evict  # type: ignore[attr-defined]
        wrapper.cache_has = cache_has  # type: ignore[attr-defined]
        wrapper.cache_clear = cache.clear  # type: ignore[attr-defined]
        wrapper._cache = cache  # type: ignore[attr-defined] - test/ops introspection
        return wrapper  # type: ignore[return-value]

    return decorate


def refresher(fn: Any, *args: Any) -> Callable[[], bool]:
    """A zero-arg thunk that forces `fn` to re-query and overwrite its entry.

    The warm loop's target lists are built from these, because a plain callable
    short-circuits on a fresh entry and warms nothing. Raises TypeError up
    front when handed something that is not ttl_cache-decorated, so a target
    list can never silently degrade into a list of no-ops.
    """
    cache_refresh = getattr(fn, "cache_refresh", None)
    if cache_refresh is None:
        raise TypeError(f"{getattr(fn, '__name__', fn)} is not ttl_cache-decorated")
    return partial(cache_refresh, *args)


def clear_all_caches() -> int:
    """Testing/ops hook. Returns the number of caches cleared."""
    with _REGISTRY_LOCK:
        for cache in _REGISTRY:
            cache.clear()
        return len(_REGISTRY)
