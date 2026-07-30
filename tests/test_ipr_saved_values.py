"""Saved IPR values — the single-well half of prop_hist Phase 2 (ipr_anchor).

Scott's goal sentence: opening a well keeps "the curve and rate the engineer
saw fit". These pin the Solver-side save (oil→liquid conversion, the 0.99 WC
cap, None-skipping) and the open-side load (requires qwf AND pwf, memoized,
fail-soft) plus the precedence rule against the anchor pin — latest timestamp
wins, ties to the VALUES (one Save writes both, and the values were seeded
from that anchor anyway).
"""

from datetime import datetime, timezone

import pandas as pd
import pytest

from woffl.gui import ipr_anchor as ia


@pytest.fixture(autouse=True)
def _clear_cache():
    ia.clear_saved_ipr_cache()
    yield
    ia.clear_saved_ipr_cache()


def _ts(s):
    return pd.Timestamp(s, tz="UTC")


# ── the precedence rule ─────────────────────────────────────────────────────


class TestSavedWins:
    def test_no_pin_means_values_win(self):
        assert ia.saved_wins(_ts("2026-07-30"), None) is True

    def test_newer_pin_wins(self):
        assert ia.saved_wins(_ts("2026-07-01"), _ts("2026-07-30")) is False

    def test_newer_values_win(self):
        assert ia.saved_wins(_ts("2026-07-30"), _ts("2026-07-01")) is True

    def test_tie_goes_to_the_values(self):
        """One Save click writes both — the values ARE the anchor's seed."""
        t = _ts("2026-07-30 12:00:00")
        assert ia.saved_wins(t, t) is True

    def test_no_values_never_win(self):
        assert ia.saved_wins(None, None) is False


class TestDefaultSigShared:
    def test_solver_aliases_the_same_object(self):
        """The sidebar seeds saved values UNDER the Solver's default anchor
        sig (the manual-edit affordance) — the two must be the same constant
        or the Solver's sync would clobber restored values on first render."""
        from woffl.gui.tabs.jetpump_solver import _IPR_SIDEBAR_DEFAULT_SIG

        assert _IPR_SIDEBAR_DEFAULT_SIG is ia.IPR_SIDEBAR_DEFAULT_SIG


# ── save: the sidebar's oil-based values → prop_hist ────────────────────────


class TestSaveIprValues:
    @pytest.fixture
    def pushes(self, monkeypatch):
        captured = []
        monkeypatch.setattr(
            ia, "push_prop", lambda w, p, v, u: captured.append((w, p, v, u))
        )
        monkeypatch.setattr(ia, "resolve_entry_user", lambda: "scott")
        return captured

    def test_oil_converts_to_total_liquid(self, pushes):
        n, msg = ia.save_ipr_values(
            "MPB-28", qwf_oil=300.0, pwf=900.0, res_pres=1800.0,
            form_wc=0.5, form_gor=250.0, surf_pres=210.0,
        )
        vals = {p: v for (_, p, v, _) in pushes}
        assert vals["ipr_qwf_liq"] == pytest.approx(600.0)  # 300 / (1 - 0.5)
        assert vals["ipr_pwf"] == 900.0
        assert vals["resvr_press"] == 1800.0  # the curve travels whole
        assert vals["surf_press"] == 210.0
        assert n == 6 and msg.startswith("💾")

    def test_wc_caps_at_099_so_conversion_cannot_degenerate(self, pushes):
        ia.save_ipr_values(
            "MPB-28", qwf_oil=10.0, pwf=900.0, res_pres=1800.0,
            form_wc=1.0, form_gor=250.0,
        )
        vals = {p: v for (_, p, v, _) in pushes}
        assert vals["form_wc"] == pytest.approx(0.99)
        assert vals["ipr_qwf_liq"] == pytest.approx(10.0 / 0.01)

    def test_missing_surface_pressure_is_skipped_not_nulled(self, pushes):
        n, _ = ia.save_ipr_values(
            "MPB-28", qwf_oil=300.0, pwf=900.0, res_pres=1800.0,
            form_wc=0.5, form_gor=250.0, surf_pres=None,
        )
        assert n == 5
        assert not any(p == "surf_press" for (_, p, _, _) in pushes)

    def test_failure_returns_the_prefixed_message(self, monkeypatch):
        def boom(w, p, v, u):
            raise RuntimeError("gate closed")

        monkeypatch.setattr(ia, "push_prop", boom)
        monkeypatch.setattr(ia, "resolve_entry_user", lambda: "scott")
        n, msg = ia.save_ipr_values(
            "MPB-28", qwf_oil=300.0, pwf=900.0, res_pres=1800.0,
            form_wc=0.5, form_gor=250.0,
        )
        assert n == 0 and msg.startswith(ia.VALUES_SAVE_FAILURE_PREFIX)

    def test_save_clears_the_load_memo(self, pushes):
        ia._saved_ipr_cache["MPB-28"] = {"stale": True}
        ia.save_ipr_values(
            "MPB-28", qwf_oil=300.0, pwf=900.0, res_pres=1800.0,
            form_wc=0.5, form_gor=250.0,
        )
        assert "MPB-28" not in ia._saved_ipr_cache


# ── load: latest values + the pin timestamp, one query, memoized ────────────


def _hist_df(rows):
    return pd.DataFrame(
        rows, columns=["prop_id", "prop_value", "entry_datetime", "entry_user"]
    )


class TestLoadSavedIpr:
    def _wire(self, monkeypatch, df):
        import woffl.assembly.databricks_client as dbc
        import woffl.assembly.prop_hist_client as phc

        monkeypatch.setattr(phc, "_resolve_enthid", lambda w: 12345)
        monkeypatch.setattr(dbc, "execute_query", lambda sql: df)

    def test_loads_values_timestamp_user_and_pin(self, monkeypatch):
        self._wire(
            monkeypatch,
            _hist_df(
                [
                    ("ipr_qwf_liq", 600.0, _ts("2026-07-30 10:00"), "scott"),
                    ("ipr_pwf", 900.0, _ts("2026-07-30 10:00"), "scott"),
                    ("form_wc", 0.5, _ts("2026-07-30 10:00"), "scott"),
                    ("form_gor", 250.0, _ts("2026-07-30 10:00"), "scott"),
                    ("surf_press", 210.0, _ts("2026-07-30 10:00"), "scott"),
                    ("resvr_press", 1800.0, _ts("2026-04-16"), "ka9612"),
                    ("ipr_wt_uid", -3587790.0, _ts("2026-07-21"), "scott"),
                ]
            ),
        )
        info = ia.load_saved_ipr("MPB-28")
        assert info["values"]["qwf_liq"] == 600.0
        assert info["values"]["res_pres"] == 1800.0
        assert info["saved_at"] == _ts("2026-07-30 10:00")
        assert info["saved_by"] == "scott"
        assert info["pin_at"] == _ts("2026-07-21")
        assert ia.saved_wins(info["saved_at"], info["pin_at"])

    def test_canonical_resvr_press_alone_is_not_a_saved_curve(self, monkeypatch):
        """Every well has resvr_press from the bulk load — that must not read
        as 'the engineer saved an IPR here'."""
        self._wire(
            monkeypatch,
            _hist_df([("resvr_press", 1800.0, _ts("2026-04-16"), "ka9612")]),
        )
        assert ia.load_saved_ipr("MPB-28") is None

    def test_canon_timestamp_does_not_date_the_saved_set(self, monkeypatch):
        """resvr_press written later (e.g. by a pad-review canon push) must
        not make stale IPR values look newer than a fresh pin."""
        self._wire(
            monkeypatch,
            _hist_df(
                [
                    ("ipr_qwf_liq", 600.0, _ts("2026-07-01"), "scott"),
                    ("ipr_pwf", 900.0, _ts("2026-07-01"), "scott"),
                    ("resvr_press", 1650.0, _ts("2026-07-30"), "scott"),
                    ("ipr_wt_uid", -3587790.0, _ts("2026-07-15"), "scott"),
                ]
            ),
        )
        info = ia.load_saved_ipr("MPB-28")
        assert info["saved_at"] == _ts("2026-07-01")
        assert not ia.saved_wins(info["saved_at"], info["pin_at"])  # pin wins

    def test_nulled_pin_row_does_not_count_as_a_pin(self, monkeypatch):
        self._wire(
            monkeypatch,
            _hist_df(
                [
                    ("ipr_qwf_liq", 600.0, _ts("2026-07-01"), "scott"),
                    ("ipr_pwf", 900.0, _ts("2026-07-01"), "scott"),
                    ("ipr_wt_uid", float("nan"), _ts("2026-07-15"), "scott"),
                ]
            ),
        )
        info = ia.load_saved_ipr("MPB-28")
        assert info["pin_at"] is None
        assert ia.saved_wins(info["saved_at"], info["pin_at"])

    def test_memoized_per_well(self, monkeypatch):
        calls = []

        import woffl.assembly.databricks_client as dbc
        import woffl.assembly.prop_hist_client as phc

        monkeypatch.setattr(phc, "_resolve_enthid", lambda w: 12345)

        def q(sql):
            calls.append(sql)
            return _hist_df(
                [
                    ("ipr_qwf_liq", 600.0, _ts("2026-07-01"), "scott"),
                    ("ipr_pwf", 900.0, _ts("2026-07-01"), "scott"),
                ]
            )

        monkeypatch.setattr(dbc, "execute_query", q)
        ia.load_saved_ipr("MPB-28")
        ia.load_saved_ipr("MPB-28")
        assert len(calls) == 1

    def test_query_failure_fails_soft_to_none(self, monkeypatch):
        import woffl.assembly.databricks_client as dbc
        import woffl.assembly.prop_hist_client as phc

        monkeypatch.setattr(phc, "_resolve_enthid", lambda w: 12345)
        monkeypatch.setattr(
            dbc, "execute_query",
            lambda sql: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        assert ia.load_saved_ipr("MPB-28") is None
