"""Pad-wide warm-up of the saved-IPR cache (`ipr_anchor.warm_saved_ipr_cache`).

Measured 2026-08-04: a Databricks round trip costs 150 ms minimum even warm.
`load_saved_ipr` is one round trip PER WELL, and a pad review walks ~20 wells
(four pads' worth on the CFP page), so the walk used to cost seconds of pure
latency before the engineer touched anything. One window-function query covers
every well — the same shape `review_persistence._latest_props` already runs on
every pad render.

These pin the properties that make that substitution safe: the warm-up must
produce EXACTLY what the per-well read would, must never clobber fresher data,
and must fail soft.
"""

from datetime import datetime, timezone

import pandas as pd
import pytest

from woffl.gui import ipr_anchor as ia

T1 = datetime(2026, 8, 3, 21, 25, tzinfo=timezone.utc)
T0 = datetime(2026, 4, 16, tzinfo=timezone.utc)


def _row(well, pid, value, at=T1, user="scott"):
    return {
        "well_name": well, "prop_id": pid, "prop_value": value,
        "entry_datetime": at, "entry_user": user,
    }


BULK = pd.DataFrame(
    [
        # B-028: a full saved curve + a pin + a WC lock
        _row("B-028", "ipr_qwf_liq", 2135.0),
        _row("B-028", "ipr_pwf", 1141.0),
        _row("B-028", "form_wc", 0.83),
        _row("B-028", "ipr_wt_uid", -3591520.0),
        _row("B-028", "form_wc_lock", 1.0),
        # B-030: friction only, no curve — must still be a record
        _row("B-030", "jpfric_entry", 0.12, at=T0, user="ka9612"),
        # B-032: a bare pin and nothing else — the case that used to vanish
        _row("B-032", "ipr_wt_uid", 4242.0),
        # C-002: a NULL (un-pinned) tombstone — not a pin
        _row("C-002", "ipr_wt_uid", float("nan")),
    ]
)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    ia._saved_ipr_cache.clear()
    monkeypatch.setattr(ia, "_warmed_at", 0.0)
    monkeypatch.setattr(
        "woffl.assembly.databricks_client.execute_query", lambda sql: BULK
    )
    yield
    ia._saved_ipr_cache.clear()


class TestWarmUp:
    def test_one_query_populates_every_well(self):
        calls = []

        def _q(sql):
            calls.append(sql)
            return BULK

        import woffl.assembly.databricks_client as dbc

        dbc.execute_query = _q
        n = ia.warm_saved_ipr_cache()
        assert n == 4 and len(calls) == 1, "must be ONE query for all wells"
        assert set(ia._saved_ipr_cache) == {"MPB-28", "MPB-30", "MPB-32", "MPC-02"}

    def test_warmed_record_matches_what_the_per_well_read_builds(self):
        """The two paths share `_assemble_saved_ipr`, so they cannot drift —
        this asserts the shared assembly actually is what's cached."""
        ia.warm_saved_ipr_cache()
        expected = ia._assemble_saved_ipr(
            {r["prop_id"]: r for _, r in BULK.iterrows()
             if r["well_name"] == "B-028"}
        )
        assert ia._saved_ipr_cache["MPB-28"] == expected

    def test_curve_pin_and_lock_all_survive(self):
        ia.warm_saved_ipr_cache()
        rec = ia._saved_ipr_cache["MPB-28"]
        assert rec["values"]["qwf_liq"] == 2135.0
        assert rec["values"]["pwf"] == 1141.0
        assert rec["pin_value"] == -3591520.0 and rec["pin_user"] == "scott"
        assert rec["locks"]["form_wc"] is True

    def test_a_pin_only_well_is_a_record_not_none(self):
        """A 📌-only well has no curve, no friction, no locks. Before the pin
        was counted it returned None and silently lost its anchor."""
        ia.warm_saved_ipr_cache()
        rec = ia._saved_ipr_cache["MPB-32"]
        assert rec is not None and rec["pin_value"] == 4242.0
        assert rec["values"] == {}

    def test_a_null_tombstone_is_not_a_pin(self):
        """NULL prop_value = un-pinned. With nothing else stored, the well has
        no record at all."""
        ia.warm_saved_ipr_cache()
        assert ia._saved_ipr_cache["MPC-02"] is None

    def test_friction_only_well_survives_without_a_curve(self):
        ia.warm_saved_ipr_cache()
        rec = ia._saved_ipr_cache["MPB-30"]
        assert rec is not None and rec["friction"] == {"ken": 0.12}
        assert rec["values"] == {} and rec["saved_at"] is None


class TestSafety:
    def test_never_clobbers_a_fresher_per_well_entry(self):
        """A save clears one well then re-reads it. The warm-up must not
        overwrite that with its older snapshot."""
        ia._saved_ipr_cache["MPB-28"] = {"values": {"qwf_liq": 999.0}}
        ia.warm_saved_ipr_cache()
        assert ia._saved_ipr_cache["MPB-28"] == {"values": {"qwf_liq": 999.0}}

    def test_ttl_makes_the_repeat_calls_free(self):
        """Both pad pages call this on EVERY rerun; the CFP page covers four
        pads. Without the guard that would be a query per rerun."""
        calls = []
        import woffl.assembly.databricks_client as dbc

        dbc.execute_query = lambda sql: (calls.append(1), BULK)[1]
        assert ia.warm_saved_ipr_cache() == 4
        assert ia.warm_saved_ipr_cache() == 0
        assert ia.warm_saved_ipr_cache() == 0
        assert len(calls) == 1

    def test_force_bypasses_the_ttl(self):
        ia.warm_saved_ipr_cache()
        ia._saved_ipr_cache.clear()
        assert ia.warm_saved_ipr_cache(force=True) == 4

    def test_failure_is_soft_and_retries_next_render(self):
        """A failed warm-up must not stamp the TTL, or one blip would disable
        it for five minutes."""
        import woffl.assembly.databricks_client as dbc

        def _boom(sql):
            raise RuntimeError("warehouse down")

        dbc.execute_query = _boom
        assert ia.warm_saved_ipr_cache() == 0
        assert ia._saved_ipr_cache == {}

        dbc.execute_query = lambda sql: BULK
        assert ia.warm_saved_ipr_cache() == 4, "must retry, not stay quiet"

    def test_empty_result_does_not_poison_the_cache(self):
        import woffl.assembly.databricks_client as dbc

        dbc.execute_query = lambda sql: pd.DataFrame()
        assert ia.warm_saved_ipr_cache() == 0
        assert ia._saved_ipr_cache == {}
