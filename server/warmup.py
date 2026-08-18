"""Fleet cache warmup - keep every well's Databricks pulls pre-paid.

The problem this exists to fix: the old startup warm covered the FLEET-WIDE
fetches (chars, PF, JP history, well tests) and nothing per-well, so the first
engineer to open any given well still paid that well's two warehouse queries
(``history.extended_tests`` + ``history.bhp_daily``, 24 h TTL each). That is
exactly the reported symptom - a well is slow once and instant afterwards.

Two distinct causes, both handled here:

1. **Nothing per-well was warmed.** ``run_pass`` walks the whole well list and
   fills the per-well caches, so no user is ever the one who pays.
2. **A one-shot warm decays.** ``ttl_cache`` serves a stale entry for one extra
   TTL and then DELETES it (``server/cache.py`` ``put`` / ``get``), at which
   point the next reader blocks on a cold miss. On a server that has been up for
   days the 1 h tier (chars, PF, surveys, Well Sort) is therefore cold again
   between sessions.

How the loop keeps its promise (rewritten 2026-08-18, "cache everything every
6 hours, overwriting the previous copy"):

* **A pass OVERWRITES.** Every target is ``fn.cache_refresh(...)``, not a plain
  call. A plain call returns a fresh entry and queries nothing, so the old loop
  only ever re-queried on the pass that happened to catch an entry already
  stale - the cadence was really the TTL, and the interval had to sit under the
  shortest TTL it protected to avoid the delete cliff.
* **A warmed entry cannot go cold between passes.** ``cache_refresh`` stores
  with ``cache.set_warm_retention(retention_sec())`` = two intervals, so even a
  pass that fails outright leaves the previous value servable. Reads past the
  TTL still get the stale value plus a background SWR refresh; what they never
  get is a blocking cold query.
* **The day boundary gets its own pass.** ``history._query_window`` puts
  today's date in the per-well cache keys, so every per-well entry is a NEW key
  after local midnight. ``_next_wait`` therefore never sleeps past midnight -
  otherwise the first engineer of the morning pays exactly the cold query this
  module exists to prevent.

Design notes:

* **Own threads, not the SWR pool.** ``cache._refresh_pool`` is 2 threads shared
  with every stale-read refresh in the app; driving ~90 wells through it would
  serialize the warmup behind ordinary traffic and vice versa.
* **Small, long-lived pool.** ``databricks_client._CONN_LOCAL`` is a
  THREAD-LOCAL warehouse connection, so a thread per well would pay ~90
  handshakes. ``WOFFL_WARM_WORKERS`` is a warehouse-concurrency knob (default 3),
  deliberately separate from ``WOFFL_MAX_WORKERS`` (a CPU/ProcessPool cap) -
  this work is pure I/O wait.
* **Never raises.** A warmer that propagated would kill the loop and leave the
  process permanently cold. Every failure is logged and counted.
* **One process.** ``app.yaml`` runs a single bare ``uvicorn server.main:app``,
  so one pass warms the whole server. If ``--workers N`` is ever added, caches
  and this loop multiply per process.

Env knobs:

* ``WOFFL_WARM_INTERVAL_SEC`` - seconds between passes (default 21600 = 6 h;
  ``0`` means a single pass and no loop).
* ``WOFFL_WARM_WORKERS`` - concurrent warehouse connections (default 3, 1..8).
* ``WOFFL_WARM_WELLS`` - falsy skips the per-well pass (fleet frames only);
  useful for local ``uvicorn --reload``, where every restart would otherwise
  re-walk the fleet.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

log = logging.getLogger("woffl.web.warmup")

# 6 h: the data behind these caches (well tests, allocation, the JP tracker,
# chars) moves daily at most. The interval is NO LONGER bounded by the shortest
# TTL - `retention_sec` keeps warmed entries servable across passes - but it is
# still bounded by the day boundary, see `_next_wait`.
_DEFAULT_INTERVAL_SEC = 21_600
# Warmed entries stay servable for two intervals, so one failed pass cannot
# leave a cold entry behind.
_RETENTION_INTERVALS = 2
# A pass lands this long after local midnight, re-keying the per-well caches.
_DAY_ROLL_GRACE_SEC = 120.0
_DEFAULT_WORKERS = 3
_MAX_WORKERS = 8
# Enough to diagnose a bad pass without dumping the whole fleet into a log line.
_FAILURE_SAMPLE = 10


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return max(lo, min(int(raw), hi))
    except (TypeError, ValueError):
        log.warning("ignoring non-integer %s=%r, using %d", name, raw, default)
        return default


def interval_sec() -> int:
    """Seconds between passes; 0 means one pass then stop."""
    return _env_int("WOFFL_WARM_INTERVAL_SEC", _DEFAULT_INTERVAL_SEC, 0, 86_400)


def retention_sec() -> float:
    """How long a warm-written entry must stay servable.

    Two intervals: a pass that fails outright (warehouse restart, expired
    token) still leaves the previous value in place for the next one, so a
    failed warm degrades to "slightly stale" instead of "cold and blocking".
    0 when the loop is single-pass - nothing is promising to come back, so
    entries keep the plain 2 x TTL grace.
    """
    every = interval_sec()
    return 0.0 if every <= 0 else float(every * _RETENTION_INTERVALS)


def _next_wait(now: Optional[datetime] = None) -> float:
    """Seconds to sleep before the next pass; 0 means stop after this one.

    Capped at the next local midnight plus a short grace because
    ``history._query_window`` stamps today's date into the per-well cache keys:
    after the day rolls, every per-well entry is a key nobody has ever filled,
    and no retention floor can help a key that does not exist yet.
    """
    every = interval_sec()
    if every <= 0:
        return 0.0
    now = now or datetime.now()
    tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    until_roll = (tomorrow - now).total_seconds() + _DAY_ROLL_GRACE_SEC
    return min(float(every), until_roll)


def workers() -> int:
    """Concurrent warehouse connections the warmup may hold."""
    return _env_int("WOFFL_WARM_WORKERS", _DEFAULT_WORKERS, 1, _MAX_WORKERS)


def wells_enabled() -> bool:
    """False skips the per-well pass and warms only the fleet frames."""
    return os.environ.get("WOFFL_WARM_WELLS", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


def _warm_saved_ipr() -> None:
    from woffl.gui import ipr_anchor

    ipr_anchor.warm_saved_ipr_cache()


def _warm_prop_write_meta() -> None:
    from woffl.assembly import prop_hist_client

    prop_hist_client.fetch_prop_xref()
    prop_hist_client._fetch_enthid_groups()


def _warm_jp_history() -> None:
    """Re-pull the JP tracker, mirroring datasources.jp_history()'s preference.

    Warming the ``_safe`` wrapper would be a no-op while the entry is fresh, and
    warming only the Databricks side would leave the xlsx fallback cold on a day
    the tracker is unreachable. Raises only when BOTH sources fail - that is the
    condition under which the app has no JP history at all.
    """
    from server.cache import refresher
    from server.services import datasources

    try:
        refresher(datasources._jp_history_databricks)()
    except Exception as exc:  # noqa: BLE001 - the fallback is the app's read path too
        log.warning("warm: JP tracker pull failed, warming the xlsx fallback: %s", exc)
        refresher(datasources._jp_history_excel)()


def fleet_targets() -> list[tuple[str, Callable[[], Any]]]:
    """(label, thunk) for every fleet-wide pull a cold request would block on.

    Every thunk is a ``cache.refresher`` - a FORCED re-query that overwrites the
    entry - so a pass genuinely re-caches the fleet on its own cadence. Calling
    the service functions directly (what this used to do) returns fresh entries
    untouched, which made the TTL, not the interval, the real refresh clock.

    Imports are local: this module is imported at app construction and the
    service modules drag in pandas plus the Databricks client.
    """
    from server.cache import refresher
    from server.config import DEFAULT_TEST_MONTHS
    from server.services import calibration_points, datasources, evidence
    from server.services import tests as tests_svc
    from server.services import well_sort as well_sort_svc

    targets: list[tuple[str, Callable[[], Any]]] = [
        ("well_characteristics", refresher(datasources.well_chars)),
        ("pf_latest", refresher(datasources.pf_latest)),
        # Also the input the per-well pass needs to find each well's installs.
        ("jp_history", _warm_jp_history),
        # Two lookback windows are live in-tree: the router/sidebar default and
        # the 12 months evidence._min_test_bhp asks for. Warming only the first
        # left every /response-history request paying a fleet query.
        ("well_tests_6mo", refresher(tests_svc.fetch_all_well_tests, DEFAULT_TEST_MONTHS)),
        ("well_tests_12mo", refresher(tests_svc.fetch_all_well_tests, 12)),
        # 365 days of fleet pressure - the single biggest query in the app, and
        # every /response-history call blocks on it.
        ("fleet_pressure_daily", refresher(evidence._fleet_pressure_daily)),
        # 365 days of fleet power-fluid volume - behind the calibration points.
        ("fleet_pf_volume", refresher(calibration_points._fleet_pf_volume)),
        # One prop_hist snapshot that makes every per-well load_saved_ipr local.
        ("saved_ipr", _warm_saved_ipr),
        # prop_hist write metadata: the first save of a process pays it inline
        # (~0.5 s) before its INSERT.
        ("prop_write_meta", _warm_prop_write_meta),
    ]
    targets.extend(well_sort_svc.warm_targets())
    return targets


def well_universe() -> list[str]:
    """Every well name the per-well pass walks."""
    from server.services import wells as wells_svc

    payload = wells_svc.list_wells()
    return [w["name"] for w in payload.get("wells", []) if w.get("name")]


def warm_one_well(well: str) -> None:
    """Pre-pay everything a first request for ``well`` would block on.

    ``history.warm_well`` covers the two per-well warehouse queries (the real
    cost); ``datasources.survey`` covers the local deviation-survey CSV parse
    behind /wells/{name}/profile. The profile payload itself is NOT warmed - its
    cache key carries the client's jpump_tvd/field_model, so warming a guessed
    triple would fill an entry no request reads.
    """
    from server.cache import refresher
    from server.services import datasources
    from server.services import history as history_svc

    history_svc.warm_well(well)
    # Local CSV parse, so cheap - but forced like everything else, so the entry
    # carries the retention floor instead of being deleted one TTL later.
    refresher(datasources.survey, well)()


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

_LOCK = threading.Lock()
_STOP = threading.Event()
_THREAD: Optional[threading.Thread] = None

_STATUS: dict[str, Any] = {
    "running": False,
    "passes": 0,
    "interval_sec": None,
    "retention_sec": None,
    "workers": None,
    "wells_enabled": None,
    "started_at": None,
    "last_pass_at": None,
    "last_pass_sec": None,
    "fleet_total": 0,
    "fleet_ok": 0,
    "fleet_failed": [],
    "wells_total": 0,
    "wells_ok": 0,
    "wells_failed": 0,
    "wells_failed_sample": [],
}


def status() -> dict[str, Any]:
    """Snapshot of the warmup's progress (ops endpoint + tests)."""
    with _LOCK:
        snap = dict(_STATUS)
    snap["fleet_failed"] = list(snap["fleet_failed"])
    snap["wells_failed_sample"] = list(snap["wells_failed_sample"])
    return snap


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Passes
# ---------------------------------------------------------------------------


def _run_labelled(
    targets: list[tuple[str, Callable[[], Any]]], n: int, kind: str
) -> list[str]:
    """Run every thunk on a bounded pool; return the labels that raised."""
    failed: list[str] = []
    if not targets:
        return failed
    with ThreadPoolExecutor(max_workers=n, thread_name_prefix=f"warm-{kind}") as pool:
        futures = {pool.submit(fn): label for label, fn in targets}
        for fut in as_completed(futures):
            label = futures[fut]
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001 - warmers must never raise
                failed.append(label)
                log.warning("warm %s failed (%s): %s", kind, label, exc)
    return failed


def run_pass() -> dict[str, Any]:
    """One full warm: fleet frames first (the per-well pass reads the JP-history
    frame they fill), then every well. Never raises; returns the new status."""
    from server.cache import set_warm_retention

    t0 = time.monotonic()
    n = workers()
    # Set per pass, not once at startup: this is what makes every entry the
    # pass writes outlive the gap to the next pass, and it re-reads the env so
    # an operator changing the interval does not leave a stale floor behind.
    set_warm_retention(retention_sec())

    try:
        targets = fleet_targets()
    except Exception as exc:  # noqa: BLE001 - a bad import must not kill the loop
        log.warning("warm: could not build fleet targets: %s", exc)
        targets = []
    fleet_failed = _run_labelled(targets, n, "fleet")

    wells: list[str] = []
    wells_ok = 0
    wells_failed: list[str] = []
    if wells_enabled():
        try:
            wells = well_universe()
        except Exception as exc:  # noqa: BLE001
            log.warning("warm: well universe unavailable: %s", exc)
        if wells:
            with ThreadPoolExecutor(
                max_workers=n, thread_name_prefix="warm-well"
            ) as pool:
                futures = {pool.submit(warm_one_well, w): w for w in wells}
                for fut in as_completed(futures):
                    well = futures[fut]
                    try:
                        fut.result()
                        wells_ok += 1
                    except Exception as exc:  # noqa: BLE001
                        wells_failed.append(well)
                        log.warning("warm well failed (%s): %s", well, exc)

    elapsed = time.monotonic() - t0
    with _LOCK:
        _STATUS.update(
            passes=_STATUS["passes"] + 1,
            interval_sec=interval_sec(),
            retention_sec=retention_sec(),
            workers=n,
            wells_enabled=wells_enabled(),
            last_pass_at=_now(),
            last_pass_sec=round(elapsed, 1),
            fleet_total=len(targets),
            fleet_ok=len(targets) - len(fleet_failed),
            fleet_failed=list(fleet_failed),
            wells_total=len(wells),
            wells_ok=wells_ok,
            wells_failed=len(wells_failed),
            wells_failed_sample=wells_failed[:_FAILURE_SAMPLE],
        )
    log.info(
        "cache warm pass done in %.1fs: fleet %d/%d, wells %d/%d",
        elapsed,
        len(targets) - len(fleet_failed),
        len(targets),
        wells_ok,
        len(wells),
    )
    return status()


def _loop() -> None:
    try:
        while True:
            run_pass()
            wait = _next_wait()
            if wait <= 0:
                log.info("warm: WOFFL_WARM_INTERVAL_SEC=0, single pass only")
                return
            log.info("warm: next pass in %.0fs", wait)
            if _STOP.wait(wait):
                return
    finally:
        with _LOCK:
            _STATUS["running"] = False


def start() -> bool:
    """Kick the warm loop off on one daemon thread. Idempotent per process:
    returns False when a loop is already running."""
    global _THREAD
    with _LOCK:
        if _THREAD is not None and _THREAD.is_alive():
            return False
        _STOP.clear()
        _STATUS.update(
            running=True,
            started_at=_now(),
            interval_sec=interval_sec(),
            retention_sec=retention_sec(),
            workers=workers(),
            wells_enabled=wells_enabled(),
        )
        _THREAD = threading.Thread(target=_loop, daemon=True, name="warm-loop")
        _THREAD.start()
    return True


def stop(timeout: float = 0.0) -> None:
    """Ask the loop to finish after its current pass (lifespan shutdown, tests).
    An in-flight pass is not interrupted - it holds warehouse connections."""
    _STOP.set()
    thread = _THREAD
    if thread is not None and timeout > 0:
        thread.join(timeout)
