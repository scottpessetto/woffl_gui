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

import pathlib
import threading
import time

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from server import warmup
from server.cache import clear_all_caches
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
    # The default itself, not a literal: it is a tuning knob (raised 3 -> 6
    # to halve the post-deploy cold window) and this test is about the
    # retention invariant, not the number.
    assert warmup.workers() == warmup._DEFAULT_WORKERS
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


@pytest.mark.parametrize(
    "raw,expected",
    [("1", 1), ("8", 8), ("50", 8), ("0", 1), ("x", warmup._DEFAULT_WORKERS)],
)
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
# history.warm_fleet - TWO statements where the fan-out ran ~180
#
# The SQL warehouse bills per wake window, not per statement, so the per-well
# fan-out's real cost was holding the warehouse up for minutes on every pass to
# fetch rows one wide query already contains. These pin the property that makes
# the swap safe: a primed entry is INDISTINGUISHABLE from what the per-well
# query would have cached.
# ---------------------------------------------------------------------------


def _today() -> str:
    """The `end` half of every per-well key (history._query_window)."""
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d")


def _fleet_jp_frame() -> pd.DataFrame:
    """Three wells with dated installs; MPX-99 (absent) is the skip case."""
    return pd.DataFrame(
        {
            "Well Name": ["MPB-28", "MPS-05", "MPC-45"],
            "Date Set": pd.to_datetime(["2024-03-04", "2025-01-01", "2025-02-02"]),
        }
    )


def _fleet_tests_frame() -> pd.DataFrame:
    """_FLEET_TEST_QUERY's shape. The S-005 2024-06-15 row is inside the FLEET
    window but before that well's own install, so its slice must drop it."""
    return pd.DataFrame(
        {
            "well_name": ["B-028", "B-028", "S-005", "S-005"],
            "wt_date": pd.to_datetime(
                ["2024-06-01", "2025-06-01", "2024-06-15", "2025-03-01"]
            ),
            "oil_rate": [120.0, 110.0, 80.0, 75.0],
            "fwat_rate": [900.0, 950.0, 500.0, 520.0],
            "lift_wat": [1500.0, 1500.0, 1200.0, 1200.0],
            "bhp": [1450.0, 1400.0, 1350.0, 1300.0],
            "pf_tubing_prs": [3000.0, 3050.0, 2900.0, 2950.0],
            "pf_inn_ann_prs": [50.0, 55.0, 45.0, 40.0],
        }
    )


def _fleet_bhp_frame() -> pd.DataFrame:
    """_FLEET_BHP_QUERY's shape - well_name is the join-back column the primed
    per-well frame must NOT carry (bhp_daily selects tag_date + bhp only)."""
    return pd.DataFrame(
        {
            "well_name": ["B-028", "B-028", "S-005", "S-005"],
            "tag_date": pd.to_datetime(
                ["2024-06-01", "2025-06-02", "2024-06-15", "2025-03-01"]
            ),
            "bhp": [1450.0, 1402.5, 1350.0, 1300.0],
        }
    )


def _boom(_sql):
    raise AssertionError("the warehouse must not be touched here")


@pytest.fixture
def fleet_queries(monkeypatch):
    """Answer the two fleet statements from synthetic frames; record the SQL."""
    from woffl.assembly import databricks_client

    state: dict = {"sql": []}

    def _execute(query, *_args, **_kwargs):
        state["sql"].append(query)
        if "vwt_map" in query:  # the fleet BHP join-back
            return _fleet_bhp_frame()
        return _fleet_tests_frame()

    state["execute"] = _execute
    monkeypatch.setattr(databricks_client, "execute_query", _execute)
    monkeypatch.setattr(
        history_svc.datasources, "jp_history_safe", lambda: (_fleet_jp_frame(), "databricks")
    )
    clear_all_caches()
    yield state
    clear_all_caches()


def test_warm_fleet_primes_what_the_per_well_query_would_have_cached(
    fleet_queries, monkeypatch
):
    """The load-bearing equality: the frame a request reads after a fleet warm
    is the frame it would have read after the per-well warm - same columns,
    same dtypes, same index, same order."""
    from woffl.assembly import databricks_client

    today = _today()
    raw_tests = _fleet_tests_frame()
    raw_bhp = _fleet_bhp_frame()
    # Exactly the rows B-028's own two queries would have returned.
    b28_tests = raw_tests[raw_tests["well_name"] == "B-028"].reset_index(drop=True)
    b28_bhp = (
        raw_bhp[raw_bhp["well_name"] == "B-028"]
        .drop(columns=["well_name"])
        .reset_index(drop=True)
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(databricks_client, "execute_query", lambda _sql: b28_tests.copy())
        expect_tests = history_svc.extended_tests("B-028", "2024-03-04", today)
        mp.setattr(databricks_client, "execute_query", lambda _sql: b28_bhp.copy())
        expect_bhp = history_svc.bhp_daily("B-028", "2024-03-04", today)
    clear_all_caches()  # drop those reference entries; warm from the fleet now

    summary = history_svc.warm_fleet(["MPB-28", "MPS-05", "MPC-45", "MPX-99"])

    assert summary == {"wells": 3, "skipped": 1, "statements": 2}
    assert len(fleet_queries["sql"]) == 2, "two statements for the whole fleet"

    # A plain call now: served from the primed entry, never from the warehouse.
    monkeypatch.setattr(databricks_client, "execute_query", _boom)
    assert_frame_equal(history_svc.extended_tests("B-028", "2024-03-04", today), expect_tests)
    assert_frame_equal(history_svc.bhp_daily("B-028", "2024-03-04", today), expect_bhp)


def test_each_well_is_sliced_to_its_own_window_not_the_fleets(fleet_queries, monkeypatch):
    """The fleet pull starts at the EARLIEST install anywhere, so a younger
    well's frame would otherwise carry rows from before its first pump."""
    from woffl.assembly import databricks_client

    today = _today()
    history_svc.warm_fleet(["MPB-28", "MPS-05"])

    monkeypatch.setattr(databricks_client, "execute_query", _boom)
    tests = history_svc.extended_tests("S-005", "2025-01-01", today)
    bhp = history_svc.bhp_daily("S-005", "2025-01-01", today)

    assert list(tests["WtDate"]) == [pd.Timestamp("2025-03-01")]
    assert list(tests["well"]) == ["MPS-05"], "the well column is normalized as before"
    assert list(bhp["tag_date"]) == [pd.Timestamp("2025-03-01")]
    assert list(bhp.columns) == ["tag_date", "bhp"], "the join-back column is dropped"


def test_the_fleet_sql_names_every_well_and_starts_at_the_earliest_install(fleet_queries):
    history_svc.warm_fleet(["MPB-28", "MPS-05", "MPC-45"])

    tests_sql, bhp_sql = fleet_queries["sql"]
    for quoted in ("'B-028'", "'S-005'", "'C-045'"):
        assert quoted in tests_sql, quoted
        assert quoted in bhp_sql, quoted
    # The outer window is the minimum per-well start, not each well's own.
    assert "2024-03-04" in tests_sql and "2024-03-04" in bhp_sql


def test_a_well_with_no_fleet_rows_is_primed_with_an_empty_frame(
    fleet_queries, monkeypatch
):
    """An empty frame is exactly what C-045's own two queries would have
    returned, so priming it is what keeps the request path off the warehouse."""
    from woffl.assembly import databricks_client

    today = _today()
    history_svc.warm_fleet(["MPB-28", "MPS-05", "MPC-45"])

    assert history_svc.extended_tests.cache_has("C-045", "2025-02-02", today) is True
    assert history_svc.bhp_daily.cache_has("C-045", "2025-02-02", today) is True

    monkeypatch.setattr(databricks_client, "execute_query", _boom)
    assert history_svc.extended_tests("C-045", "2025-02-02", today).empty
    assert history_svc.bhp_daily("C-045", "2025-02-02", today).empty


def test_warm_fleet_propagates_a_query_failure_and_primes_nothing(
    fleet_queries, monkeypatch
):
    """Failures are never cached (the caller falls back to the per-well path);
    a half-primed cache would be worse than a cold one."""
    from woffl.assembly import databricks_client

    def _down(_sql):
        raise RuntimeError("warehouse asleep")

    monkeypatch.setattr(databricks_client, "execute_query", _down)

    with pytest.raises(RuntimeError):
        history_svc.warm_fleet(["MPB-28", "MPS-05"])

    today = _today()
    assert history_svc.extended_tests.cache_has("B-028", "2024-03-04", today) is False
    assert history_svc.bhp_daily.cache_has("B-028", "2024-03-04", today) is False


def test_warm_fleet_skips_a_well_with_no_dated_install(fleet_queries):
    """Same skip rule as warm_well - and a skipped well must not widen the IN
    list either, or the fleet query pulls rows nobody will ever read."""
    summary = history_svc.warm_fleet(["MPX-99"])

    assert summary == {"wells": 0, "skipped": 1, "statements": 0}
    assert fleet_queries["sql"] == [], "nothing to warm means no statement at all"


# ---------------------------------------------------------------------------
# run_pass
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_pass(monkeypatch):
    """Replace the real targets with recording stubs."""
    state = {
        "fleet_calls": [],
        "well_calls": [],
        "skip_flags": [],
        "history_calls": [],
        "fleet_boom": set(),
        "well_boom": set(),
        "history_boom": False,
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

    def _one(well, skip_history=False):
        state["well_calls"].append(well)
        state["skip_flags"].append(skip_history)
        if well in state["well_boom"]:
            raise RuntimeError(f"{well} down")

    def _fleet_history(wells):
        state["history_calls"].append(list(wells))
        if state["history_boom"]:
            raise RuntimeError("warehouse asleep")
        return {"wells": len(wells), "skipped": 0, "statements": 2}

    monkeypatch.setattr(warmup, "fleet_targets", _fleet)
    monkeypatch.setattr(warmup, "well_universe", lambda: list(state["wells"]))
    monkeypatch.setattr(warmup, "warm_one_well", _one)
    monkeypatch.setattr(warmup, "warm_fleet_history", _fleet_history)
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


def test_a_pass_warms_the_fleet_history_in_one_go_and_counts_its_statements(stub_pass):
    st = warmup.run_pass()

    assert stub_pass["history_calls"] == [stub_pass["wells"]], "once, for all wells"
    assert st["fleet_history_ok"] is True
    # Every well still runs (the local survey parse), but with history skipped.
    assert stub_pass["skip_flags"] == [True] * len(stub_pass["wells"])
    # 3 fleet targets + 2 fleet history statements - NOT 3 + 2 x 5.
    assert st["statements"] == 5


def test_a_failed_fleet_history_pull_falls_back_to_the_per_well_queries(stub_pass):
    stub_pass["history_boom"] = True

    st = warmup.run_pass()

    assert st["fleet_history_ok"] is False
    assert sorted(stub_pass["well_calls"]) == sorted(stub_pass["wells"])
    assert stub_pass["skip_flags"] == [False] * len(stub_pass["wells"])
    assert st["statements"] == 3 + 2 * len(stub_pass["wells"])
    assert st["wells_ok"] == len(stub_pass["wells"]), "freshness is never the cost"


@pytest.fixture
def stub_history(monkeypatch):
    """The real run_pass -> warm_fleet_history -> history seam, with only the
    two history entry points and the survey parse stubbed."""
    calls: dict = {"fleet": [], "well": [], "survey": [], "boom": False}

    def _fleet(wells):
        calls["fleet"].append(list(wells))
        if calls["boom"]:
            raise RuntimeError("warehouse asleep")
        return {"wells": len(wells), "skipped": 0, "statements": 2}

    class _Survey:
        def cache_refresh(self, well):
            calls["survey"].append(well)
            return True

    monkeypatch.setattr(history_svc, "warm_fleet", _fleet)
    monkeypatch.setattr(history_svc, "warm_well", lambda w: calls["well"].append(w))
    monkeypatch.setattr(datasources_svc, "survey", _Survey())
    monkeypatch.setattr(warmup, "fleet_targets", list)
    monkeypatch.setattr(warmup, "well_universe", lambda: ["MPB-28", "MPS-05"])
    return calls


def test_a_pass_calls_warm_fleet_once_and_warm_well_never(stub_history):
    """DATA-9: the per-well fan-out was 2 queries x ~90 wells x 5 passes a day.
    On the happy path warm_well must not run at all."""
    st = warmup.run_pass()

    assert stub_history["fleet"] == [["MPB-28", "MPS-05"]]
    assert stub_history["well"] == []
    assert sorted(stub_history["survey"]) == ["MPB-28", "MPS-05"], "local parse still runs"
    assert st["statements"] == 2 and st["wells_ok"] == 2


def test_the_per_well_fan_out_returns_when_the_fleet_pull_fails(stub_history):
    stub_history["boom"] = True

    st = warmup.run_pass()

    assert len(stub_history["fleet"]) == 1
    assert sorted(stub_history["well"]) == ["MPB-28", "MPS-05"]
    assert st["fleet_history_ok"] is False and st["statements"] == 4


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
        # calibration_points.points_for_well asks for 24. This one WAS missing:
        # the list was hand-written here and drifted from the call sites, so
        # the first request touching calibration points paid a cold fleet
        # query. See test_every_live_lookback_window_is_warmed below.
        "well_tests_24mo",
        # /api/wells is cached now, so it needs the retention floor too.
        "well_list",
        "surveyed_wells",
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


def test_every_live_lookback_window_is_warmed():
    """The warm list must cover every months value the code actually asks for.

    fetch_all_well_tests caches PER WINDOW and each miss is a full-fleet
    query. The list used to be hand-maintained in fleet_targets and silently
    fell behind the call sites (24 months went unwarmed). It is now generated
    from config.WARM_TEST_MONTHS, so this test guards the remaining risk: a
    new call site picking a window nobody added to that tuple.
    """
    import re

    from server import config

    labels = {label for label, _ in warmup.fleet_targets()}
    for months in config.WARM_TEST_MONTHS:
        assert f"well_tests_{months}mo" in labels, months

    # Every months literal handed to the test fetchers anywhere under server/.
    sources = (pathlib.Path(__file__).resolve().parent.parent / "server").rglob("*.py")
    asked: set[int] = set()
    call = re.compile(r"tests_for_well\(\s*[^,]+,\s*(\d+)|fetch_all_well_tests\(\s*(\d+)")
    for path in sources:
        for lit in call.findall(path.read_text(encoding="utf-8")):
            asked.add(int(lit[0] or lit[1]))

    missing = asked - set(config.WARM_TEST_MONTHS)
    assert not missing, (
        f"lookback window(s) {sorted(missing)} are requested in server/ but not in "
        "config.WARM_TEST_MONTHS, so the first request that needs one pays a "
        "cold full-fleet query"
    )


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
