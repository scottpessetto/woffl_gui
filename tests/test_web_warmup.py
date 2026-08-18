"""server.warmup - the fleet cache warmup that removes the cold first load.

The symptom this fixes: the old startup warm covered fleet-wide fetches only,
so the first engineer to open a well paid that well's two Databricks queries
(history.extended_tests + history.bhp_daily), and a one-shot warm decayed once
ttl_cache deleted entries past 2 x TTL - a server up for days was cold again.

The load-bearing guarantees pinned here:

* ``history.warm_well`` fills the EXACT cache keys ``jp_history_payload`` reads.
  A key mismatch would leave the warmup filling entries nobody looks at, and
  nothing else in the system would notice.
* A pass never raises, and one failing target never stops the rest.
* Per-well cache maxsize exceeds two days' worth of the fleet (the keys carry
  today's date and roll over at midnight).

No Databricks: every target and fetcher is monkeypatched.
"""

from __future__ import annotations

import threading
import time

import pandas as pd
import pytest

from server import warmup
from server.services import database as database_svc
from server.services import datasources as datasources_svc
from server.services import history as history_svc
from server.services import tests as tests_svc

FLEET = 90  # docs/ipr_model_review.md; 91 deviation surveys on disk


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        "WOFFL_WARM_INTERVAL_SEC",
        "WOFFL_WARM_WORKERS",
        "WOFFL_WARM_WELLS",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _quiet_loop():
    """Never leave a warm loop running into the next test."""
    yield
    warmup.stop(timeout=5.0)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_a_warmed_entry_outlives_the_gap_between_passes():
    """The invariant that replaced "interval must be under the shortest TTL":
    a pass forces an overwrite and stores with a retention floor, so the
    cadence is free to be longer than any TTL it protects."""
    from server import config

    assert warmup.interval_sec() == 21_600  # 6 h
    assert warmup.retention_sec() > warmup.interval_sec()
    assert warmup.retention_sec() > 2 * config.TTL_CHARS
    assert warmup.workers() == 3
    assert warmup.wells_enabled() is True


def test_a_single_pass_run_keeps_the_plain_swr_grace(monkeypatch):
    """Nothing is promising to come back, so nothing may claim retention."""
    monkeypatch.setenv("WOFFL_WARM_INTERVAL_SEC", "0")
    assert warmup.retention_sec() == 0.0


def test_the_loop_never_sleeps_past_the_day_boundary():
    """Per-well cache keys carry today's date (history._query_window), so every
    per-well entry is a brand-new key after midnight - a key no retention floor
    can protect. A pass must land just after the roll."""
    from datetime import datetime

    # 20:00 with a 6 h interval would otherwise wake at 02:00, four hours late.
    wait = warmup._next_wait(datetime(2026, 8, 18, 20, 0, 0))
    assert wait == pytest.approx(4 * 3600 + 120)
    # Mid-morning is nowhere near the roll, so the plain interval stands.
    assert warmup._next_wait(datetime(2026, 8, 18, 9, 0, 0)) == 21_600


@pytest.mark.parametrize(
    "raw,expected",
    [("60", 60), ("0", 0), ("-5", 0), ("999999", 86_400), ("banana", 21_600), ("", 21_600)],
)
def test_interval_env_is_clamped_not_trusted(monkeypatch, raw, expected):
    monkeypatch.setenv("WOFFL_WARM_INTERVAL_SEC", raw)
    assert warmup.interval_sec() == expected


@pytest.mark.parametrize("raw,expected", [("1", 1), ("8", 8), ("50", 8), ("0", 1), ("x", 3)])
def test_worker_env_is_clamped_to_the_warehouse_ceiling(monkeypatch, raw, expected):
    monkeypatch.setenv("WOFFL_WARM_WORKERS", raw)
    assert warmup.workers() == expected


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", "OFF"])
def test_wells_pass_can_be_switched_off(monkeypatch, raw):
    monkeypatch.setenv("WOFFL_WARM_WELLS", raw)
    assert warmup.wells_enabled() is False


# ---------------------------------------------------------------------------
# Cache sizing: a fleet-wide warm must not evict itself
# ---------------------------------------------------------------------------


def test_per_well_caches_hold_two_days_of_the_fleet():
    """extended_tests / bhp_daily key on (well, start, TODAY), so at midnight
    every well gets a second key. At maxsize 128 the day's fresh entries evicted
    each other while yesterday's were still resident."""
    for fn in (history_svc.extended_tests, history_svc.bhp_daily):
        assert fn._cache.maxsize >= 2 * FLEET


def test_fleet_sized_caches_can_hold_every_well():
    assert database_svc._prop_history._cache.maxsize >= FLEET
    assert datasources_svc.survey._cache.maxsize >= FLEET


def test_fleet_test_window_cache_holds_every_live_window():
    """Two windows are live in-tree (6 and 12 months) and the router accepts
    1..60; at maxsize 4 a few ad-hoc requests evicted a 24 h fleet query."""
    assert tests_svc.fetch_all_well_tests._cache.maxsize >= 8


# ---------------------------------------------------------------------------
# The key-agreement invariant
# ---------------------------------------------------------------------------


def _jp_frame(well: str = "MPB-28") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Well Name": [well, well, "MPS-05"],
            "Date Set": pd.to_datetime(["2024-03-04", "2025-11-20", "2025-01-01"]),
            "Nozzle Number": [12.0, 13.0, 11.0],
            "Throat Ratio": ["B", "C", "A"],
        }
    )


@pytest.fixture
def recorded_keys(monkeypatch):
    """Record the (db_name, start, end) triples handed to the two fetchers.

    The stub answers BOTH entry points, because the two paths under test use
    different ones: the request path calls the fetcher, the warm path calls its
    ``cache_refresh`` (a forced overwrite - a plain call would return a fresh
    entry and warm nothing). They must still ask for the identical triple.
    """
    seen: list[tuple] = []

    class _Recorder:
        def __init__(self, label: str) -> None:
            self.label = label

        def __call__(self, db_name, start, end):
            seen.append((self.label, db_name, start, end))
            return pd.DataFrame()

        def cache_refresh(self, db_name, start, end):
            self(db_name, start, end)
            return True

    monkeypatch.setattr(history_svc, "extended_tests", _Recorder("extended_tests"))
    monkeypatch.setattr(history_svc, "bhp_daily", _Recorder("bhp_daily"))
    monkeypatch.setattr(
        history_svc.datasources,
        "jp_history_safe",
        lambda: (_jp_frame(), "databricks"),
    )
    return seen


def test_warm_well_fills_exactly_the_keys_the_request_path_reads(recorded_keys):
    assert history_svc.warm_well("MPB-28") is True
    warmed = sorted(recorded_keys)
    recorded_keys.clear()

    history_svc.jp_history_payload("MPB-28")
    requested = sorted(recorded_keys)

    assert warmed == requested
    # And the window really is earliest-install to today, in the DB name form.
    assert warmed[0][1] == "B-028"
    assert warmed[0][2] == "2024-03-04"


def test_warm_well_skips_a_well_with_no_dated_install(recorded_keys):
    assert history_svc.warm_well("MPX-99") is False
    assert recorded_keys == []


def test_warm_well_raises_when_no_jp_history_source_is_reachable(monkeypatch):
    monkeypatch.setattr(
        history_svc.datasources, "jp_history_safe", lambda: (None, None)
    )
    with pytest.raises(RuntimeError):
        history_svc.warm_well("MPB-28")


# ---------------------------------------------------------------------------
# run_pass
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_pass(monkeypatch):
    """Replace the real targets with recording stubs."""
    state = {
        "fleet_calls": [],
        "well_calls": [],
        "fleet_boom": set(),
        "well_boom": set(),
        "wells": [f"MPB-{i:02d}" for i in range(1, 6)],
    }

    def _fleet():
        out = []
        for label in ("alpha", "beta", "gamma"):

            def _fn(label=label):
                state["fleet_calls"].append(label)
                if label in state["fleet_boom"]:
                    raise RuntimeError(f"{label} down")

            out.append((label, _fn))
        return out

    def _one(well):
        state["well_calls"].append(well)
        if well in state["well_boom"]:
            raise RuntimeError(f"{well} down")

    monkeypatch.setattr(warmup, "fleet_targets", _fleet)
    monkeypatch.setattr(warmup, "well_universe", lambda: list(state["wells"]))
    monkeypatch.setattr(warmup, "warm_one_well", _one)
    return state


def test_a_pass_warms_every_fleet_target_and_every_well(stub_pass):
    st = warmup.run_pass()

    assert sorted(stub_pass["fleet_calls"]) == ["alpha", "beta", "gamma"]
    assert sorted(stub_pass["well_calls"]) == sorted(stub_pass["wells"])
    assert st["fleet_ok"] == 3 and st["fleet_failed"] == []
    assert st["wells_total"] == 5 and st["wells_ok"] == 5 and st["wells_failed"] == 0
    assert st["last_pass_sec"] is not None


def test_one_broken_target_never_stops_the_rest(stub_pass):
    stub_pass["fleet_boom"] = {"beta"}
    stub_pass["well_boom"] = {"MPB-03"}

    st = warmup.run_pass()

    assert sorted(stub_pass["fleet_calls"]) == ["alpha", "beta", "gamma"]
    assert st["fleet_failed"] == ["beta"] and st["fleet_ok"] == 2
    assert st["wells_ok"] == 4
    assert st["wells_failed"] == 1 and st["wells_failed_sample"] == ["MPB-03"]


def test_a_dead_well_list_still_leaves_the_fleet_warm(stub_pass, monkeypatch):
    def _boom():
        raise RuntimeError("chars unavailable")

    monkeypatch.setattr(warmup, "well_universe", _boom)

    st = warmup.run_pass()

    assert st["fleet_ok"] == 3
    assert st["wells_total"] == 0 and st["wells_ok"] == 0


def test_wells_disabled_warms_the_fleet_only(stub_pass, monkeypatch):
    monkeypatch.setenv("WOFFL_WARM_WELLS", "0")

    st = warmup.run_pass()

    assert st["fleet_ok"] == 3
    assert stub_pass["well_calls"] == []
    assert st["wells_total"] == 0 and st["wells_enabled"] is False


# ---------------------------------------------------------------------------
# Targets and the loop
# ---------------------------------------------------------------------------


def test_fleet_targets_cover_the_frames_every_cold_request_blocks_on():
    labels = [label for label, _ in warmup.fleet_targets()]

    for required in (
        "well_characteristics",
        "pf_latest",
        "jp_history",
        "well_tests_6mo",
        # evidence._min_test_bhp asks for 12 months; warming only 6 left every
        # /response-history request paying a fleet query.
        "well_tests_12mo",
        "fleet_pressure_daily",
        "fleet_pf_volume",
        "saved_ipr",
        "prop_write_meta",
        # Well Sort owns these; the warmup must not have to guess them.
        "shut_in_history",
        "producers",
        "producer_catalog",
        "last_tests_ever",
        "recent_tests",
    ):
        assert required in labels, required
    assert len(labels) == len(set(labels)), "duplicate warm label"


def test_well_universe_reads_the_characteristics_list(monkeypatch):
    from server.services import wells as wells_svc

    monkeypatch.setattr(
        wells_svc,
        "list_wells",
        lambda: {
            "wells": [{"name": "MPB-28"}, {"name": ""}, {"name": "MPS-05"}],
            "source": "csv_fallback",
        },
    )
    assert warmup.well_universe() == ["MPB-28", "MPS-05"]


def test_warm_one_well_covers_the_queries_and_the_local_survey(monkeypatch):
    called = []

    class _Survey:
        """The survey cache is warmed through cache_refresh, not a plain call -
        a plain call would return a still-fresh entry and refresh nothing."""

        def cache_refresh(self, well):
            called.append(("survey", well))
            return True

    monkeypatch.setattr(history_svc, "warm_well", lambda w: called.append(("jp", w)))
    monkeypatch.setattr(datasources_svc, "survey", _Survey())

    warmup.warm_one_well("MPS-05")

    assert called == [("jp", "MPS-05"), ("survey", "MPS-05")]


def test_the_loop_repeats_so_a_warm_entry_never_ages_out(monkeypatch):
    """A one-shot warm decays: ttl_cache deletes an entry past 2 x TTL and the
    next reader blocks. The loop is what keeps a days-old server warm."""
    passes = threading.Semaphore(0)
    monkeypatch.setenv("WOFFL_WARM_INTERVAL_SEC", "60")  # clamped floor, then stubbed
    monkeypatch.setattr(warmup, "interval_sec", lambda: 0.01)
    monkeypatch.setattr(warmup, "run_pass", lambda: passes.release() or {})

    assert warmup.start() is True
    assert passes.acquire(timeout=5.0)
    assert passes.acquire(timeout=5.0), "loop must run more than one pass"
    assert warmup.start() is False, "start must be idempotent per process"

    warmup.stop(timeout=5.0)
    deadline = time.monotonic() + 5.0
    while warmup.status()["running"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert warmup.status()["running"] is False


def test_interval_zero_runs_a_single_pass(monkeypatch):
    calls = []
    monkeypatch.setenv("WOFFL_WARM_INTERVAL_SEC", "0")
    monkeypatch.setattr(warmup, "run_pass", lambda: calls.append(1) or {})

    assert warmup.start() is True
    deadline = time.monotonic() + 5.0
    while warmup.status()["running"] and time.monotonic() < deadline:
        time.sleep(0.01)

    assert calls == [1]
    assert warmup.status()["running"] is False


# ---------------------------------------------------------------------------
# End-to-end wiring: lifespan -> loop -> ops endpoint
# ---------------------------------------------------------------------------


def test_lifespan_starts_the_warmup_and_meta_reports_progress(monkeypatch):
    """The whole chain, exercised through the real ASGI app: entering the
    lifespan starts the loop, and GET /api/meta/warmup is how an operator sees
    whether the fleet is warm."""
    from fastapi.testclient import TestClient

    from server.main import app

    monkeypatch.setenv("WOFFL_WARM_INTERVAL_SEC", "0")  # one pass, no loop
    monkeypatch.setenv("WOFFL_WARM_WELLS", "0")  # no fleet walk in a test
    monkeypatch.setattr(warmup, "fleet_targets", lambda: [("stub", lambda: None)])

    # `passes` is a process-lifetime counter, so measure the delta.
    before = warmup.status()["passes"]
    with TestClient(app) as client:
        deadline = time.monotonic() + 10.0
        body = client.get("/api/meta/warmup").json()
        while body["passes"] <= before and time.monotonic() < deadline:
            time.sleep(0.02)
            body = client.get("/api/meta/warmup").json()

    assert body["passes"] == before + 1
    assert body["fleet_total"] == 1 and body["fleet_ok"] == 1
    assert body["fleet_failed"] == []
    assert body["wells_enabled"] is False
    assert body["interval_sec"] == 0
    assert body["last_pass_at"] and body["last_pass_sec"] is not None
