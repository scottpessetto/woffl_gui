"""server.cache ttl_cache contract: TTL semantics, stale-while-revalidate,
failure handling, the clear() version guard that protects the write endpoints'
read-your-writes behavior, and the warm loop's forced-overwrite write path.
"""

from __future__ import annotations

import threading
import time

import pytest

from server.cache import refresher, set_warm_retention, ttl_cache


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


def test_clear_drops_every_entry_and_recomputes():
    """cache_clear is the write-side invalidate-all (the pad board's path):
    every key gone, the very next calls recompute - no TTL wait."""
    calls = []

    @ttl_cache(60.0)
    def fn(well):
        calls.append(well)
        return f"saved:{well}"

    fn("MPB-28")
    fn("MPC-45")
    fn("MPB-28")  # fresh hit
    assert calls == ["MPB-28", "MPC-45"]

    fn.cache_clear()

    assert fn("MPB-28") == "saved:MPB-28"
    assert fn("MPC-45") == "saved:MPC-45"
    assert calls == ["MPB-28", "MPC-45", "MPB-28", "MPC-45"]


# ---------------------------------------------------------------------------
# cache_refresh + the warm retention floor (the warmup's write path)
# ---------------------------------------------------------------------------


def test_a_plain_call_cannot_warm_a_fresh_entry_but_refresh_can():
    """Why the warm loop needs its own write path at all: a plain call on a
    fresh entry returns the cached value and queries NOTHING, so a loop built
    out of plain calls refreshes on the TTL's schedule, not its own."""
    calls = []

    @ttl_cache(60.0)
    def fn():
        calls.append(1)
        return len(calls)

    assert fn() == 1
    assert fn() == 1 and len(calls) == 1  # fresh: no re-query

    assert fn.cache_refresh() is True
    assert len(calls) == 2, "cache_refresh must re-query regardless of freshness"
    assert fn() == 2, "and the new value must have replaced the old one"


def test_refresh_is_single_flight_against_a_background_swr_refresh():
    """A warm pass landing on a key the SWR pool is already refreshing must not
    duplicate the warehouse query."""
    calls = []

    @ttl_cache(60.0)
    def fn():
        calls.append(1)
        return len(calls)

    fn()
    fn._cache.try_begin_refresh(((), ()))  # pretend a stale read is refreshing
    try:
        assert fn.cache_refresh() is False
        assert len(calls) == 1
    finally:
        fn._cache.end_refresh(((), ()))

    assert fn.cache_refresh() is True and len(calls) == 2


def test_a_failed_refresh_leaves_the_previous_value_intact():
    """The whole point of overwriting rather than clearing: a warehouse blip
    must degrade to "slightly stale", never to "cold and blocking"."""
    calls = []

    @ttl_cache(60.0)
    def fn():
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError("warehouse down")
        return "good"

    assert fn() == "good"
    with pytest.raises(RuntimeError):
        fn.cache_refresh()
    assert fn() == "good", "the old entry must survive a failed refresh"


def test_the_retention_floor_keeps_a_warmed_entry_servable_past_2x_ttl():
    """Without this, a 6 h warm cadence over a 1 h TTL would delete every entry
    two hours in and hand the next reader a cold, blocking query."""
    warmed_calls = []

    @ttl_cache(ttl=0.05)
    def warmed():
        warmed_calls.append(1)
        return "value"

    set_warm_retention(30.0)
    try:
        assert warmed.cache_refresh() is True
        time.sleep(0.2)  # well past fresh + 2 x ttl
        # Served from the entry, not recomputed inline: the read never blocks.
        assert warmed() == "value"
        assert len(warmed_calls) == 1
    finally:
        set_warm_retention(0.0)


def test_an_unwarmed_entry_still_expires_into_a_blocking_recompute():
    """The floor applies ONLY to what the warm path wrote - ordinary caching
    semantics are untouched for everything else."""
    calls = []

    @ttl_cache(ttl=0.05)
    def fn():
        calls.append(1)
        return len(calls)

    set_warm_retention(30.0)  # set, but this entry is not warm-written
    try:
        assert fn() == 1
        time.sleep(0.2)
        assert fn() == 2
    finally:
        set_warm_retention(0.0)


# ---------------------------------------------------------------------------
# cache_prime - one fleet query filling every per-well key
# ---------------------------------------------------------------------------


def test_a_primed_entry_is_served_without_ever_calling_the_function():
    """Why it exists: one fleet-wide statement already holds every well's rows,
    so filling ~90 per-well keys must not cost ~90 warehouse queries."""
    calls = []

    @ttl_cache(60.0)
    def fn(well):
        calls.append(well)
        return f"queried:{well}"

    assert fn.cache_prime("from the fleet frame", "MPB-28") is True

    assert fn("MPB-28") == "from the fleet frame"
    assert calls == [], "a primed key must never reach the fetcher"
    # And only that key: priming is not a blanket fill.
    assert fn("MPC-45") == "queried:MPC-45"
    assert calls == ["MPC-45"]


def test_a_primed_entry_carries_the_warm_retention_floor():
    """A primed entry is the warm loop's own write, so it gets the same floor
    cache_refresh gives - otherwise the fleet warm would be deleted one TTL
    later and the next reader would pay the per-well query anyway."""
    calls = []

    @ttl_cache(ttl=0.05)
    def fn(well):
        calls.append(well)
        return "queried"

    set_warm_retention(30.0)
    try:
        assert fn.cache_prime("primed", "MPB-28") is True
        time.sleep(0.2)  # well past fresh + 2 x ttl
        assert fn("MPB-28") == "primed"
        assert calls == []
    finally:
        set_warm_retention(0.0)


def test_a_clear_between_capture_and_prime_discards_the_prime():
    """Read-your-writes, extended to the fleet path: the value was fetched
    before the write landed, so it must not be stored after it - exactly what
    the version guard does for an ordinary in-flight fetch."""
    calls = []

    @ttl_cache(60.0)
    def fn(well):
        calls.append(well)
        return "post-write"

    version = fn.cache_version()  # captured before the fleet fetch
    fn.cache_clear()  # a save landed while that fetch was in flight

    assert fn.cache_prime("pre-write", "MPB-28", version=version) is False
    assert fn("MPB-28") == "post-write"
    assert calls == ["MPB-28"]


def test_a_prime_defers_to_an_in_flight_refresh():
    """Single-flight, like cache_refresh: the refresh already running is about
    to store a value fetched no earlier than this one."""

    @ttl_cache(60.0)
    def fn(well):
        return "queried"

    key = (("MPB-28",), ())
    fn._cache.try_begin_refresh(key)
    try:
        assert fn.cache_prime("primed", "MPB-28") is False
    finally:
        fn._cache.end_refresh(key)

    assert fn.cache_prime("primed", "MPB-28") is True
    assert fn("MPB-28") == "primed"


def test_refresher_refuses_an_uncached_function():
    """A warm target list that silently became a list of no-ops is the failure
    mode this guards: the loop would report success and warm nothing."""
    with pytest.raises(TypeError):
        refresher(lambda: None)


def test_refresher_binds_the_arguments_of_the_key_it_warms():
    calls = []

    @ttl_cache(60.0)
    def fn(months):
        calls.append(months)
        return months

    fn(6)
    refresher(fn, 12)()

    assert calls == [6, 12]
    assert fn(12) == 12 and calls == [6, 12], "the 12-month key is warm now"
