"""server.cache ttl_cache contract: TTL semantics, stale-while-revalidate,
failure handling, and the clear() version guard that protects the write
endpoints' read-your-writes behavior.
"""

from __future__ import annotations

import threading
import time

from server.cache import ttl_cache


def _wait_until(pred, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_fresh_hit_and_expiry_to_miss():
    calls = []

    @ttl_cache(ttl=0.15)
    def fn(x):
        calls.append(x)
        return x * 2

    assert fn(3) == 6
    assert fn(3) == 6
    assert calls == [3]  # fresh hit

    # Beyond fresh + grace (2 * ttl) the entry is gone: a blocking recompute.
    time.sleep(0.35)
    assert fn(3) == 6
    assert calls == [3, 3]


def test_stale_serves_immediately_and_refreshes_in_background():
    calls = []
    gate = threading.Event()

    @ttl_cache(ttl=0.15)
    def fn():
        calls.append(1)
        if len(calls) > 1:
            gate.wait(2.0)  # make the refresh observable
            return "refreshed"
        return "first"

    assert fn() == "first"
    time.sleep(0.2)  # expired, within grace

    t0 = time.monotonic()
    assert fn() == "first"  # stale served instantly, not blocked on the gate
    assert time.monotonic() - t0 < 0.1

    gate.set()
    assert _wait_until(lambda: fn() == "refreshed")
    assert len(calls) == 2  # single-flight: repeated stale reads did not stack


def test_failed_refresh_keeps_serving_stale_and_retries():
    calls = []

    @ttl_cache(ttl=0.15)
    def fn():
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("databricks blip")
        return len(calls)

    assert fn() == 1
    time.sleep(0.2)
    assert fn() == 1  # stale; refresh #2 fails in background
    assert _wait_until(lambda: len(calls) >= 2)
    time.sleep(0.05)  # let the failed refresh clear its latch
    assert fn() == 1  # still stale, schedules refresh #3
    assert _wait_until(lambda: fn() == 3)


def test_failures_never_cached():
    calls = []

    @ttl_cache(ttl=10)
    def fn():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return "ok"

    try:
        fn()
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass
    assert fn() == "ok"
    assert len(calls) == 2


def test_clear_version_guard_blocks_pre_clear_fetch():
    """A fetch that began before clear() must not store its result after -
    the read-your-writes contract of the save/clear-pin/lock endpoints."""
    started = threading.Event()
    release = threading.Event()
    results = []

    @ttl_cache(ttl=10)
    def fn():
        started.set()
        release.wait(2.0)
        return "pre-write"

    def slow_reader():
        results.append(fn())

    t = threading.Thread(target=slow_reader)
    t.start()
    assert started.wait(2.0)
    fn.cache_clear()  # the write landed while the fetch was in flight
    release.set()
    t.join(2.0)
    assert results == ["pre-write"]  # caller still gets its computed value

    # But the cache did NOT keep it: the next read recomputes.
    calls = fn._cache._data
    assert len(calls) == 0


def test_evict_drops_one_key_and_keeps_the_rest():
    """A save on ONE well must not cold-start the fleet: cache_evict takes the
    saved well's entry and leaves every other well's cached read alone."""
    calls = []

    @ttl_cache(ttl=10, maxsize=8)
    def fn(well):
        calls.append(well)
        return f"saved:{well}"

    fn("MPB-28")
    fn("MPC-45")
    assert calls == ["MPB-28", "MPC-45"]

    fn.cache_evict("MPB-28")

    assert fn("MPC-45") == "saved:MPC-45"  # untouched, still a hit
    assert calls == ["MPB-28", "MPC-45"]
    assert fn("MPB-28") == "saved:MPB-28"  # evicted, recomputed
    assert calls == ["MPB-28", "MPC-45", "MPB-28"]


def test_evict_version_guard_blocks_a_pre_write_fetch():
    """Same read-your-writes guarantee clear() gives: a fetch already in
    flight when the write lands cannot store its stale result afterwards."""
    started = threading.Event()
    release = threading.Event()
    results = []

    @ttl_cache(ttl=10)
    def fn(well):
        started.set()
        release.wait(2.0)
        return "pre-write"

    t = threading.Thread(target=lambda: results.append(fn("MPB-28")))
    t.start()
    assert started.wait(2.0)
    fn.cache_evict("MPB-28")
    release.set()
    t.join(2.0)

    assert results == ["pre-write"]
    assert len(fn._cache._data) == 0


def test_kwargs_participate_in_key():
    calls = []

    @ttl_cache(ttl=10)
    def fn(a, b=1):
        calls.append((a, b))
        return a + b

    assert fn(1, b=2) == 3
    assert fn(1, b=3) == 4
    assert len(calls) == 2
