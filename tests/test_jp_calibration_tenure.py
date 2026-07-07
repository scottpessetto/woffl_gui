"""Regression tests for the pump-tenure fixes in Scott's Tools calibration.

Covers two review findings (docs/code_review_2026-07-01.md):

  - P1-29: ``jp_calibration.py`` / ``pf_scenario.py`` used to pair a
    friction-coef fit (or a gaugeless BHP back-calc) with TODAY's current
    pump instead of the pump actually installed on the historical test's
    date. Fixed via ``get_pump_at_date`` (mirrors ``jp_fric_trend.py`` /
    ``jp_washout.py``), plus a pump-changed guard that flags/soft-blocks the
    fit when the pump has since been changed out. Includes a JPCO
    same-day-pull+set regression per the CLAUDE.md gotcha: tenure must be
    derived from ``Date Set`` -> next ``Date Set``, never from ``Date
    Pulled`` (which lags in the tracker).

  - P1-35: the BHP calibration target and the WHP solver boundary used to
    be picked INDEPENDENTLY ("latest available" each), which could silently
    pull from two different tests/days. Fixed by coupling both to the SAME
    test row.
"""

import numpy as np
import pandas as pd
import pytest

from woffl.gui.scotts_tools import jp_calibration as jc
from woffl.gui.scotts_tools import pf_scenario as pfs

# ── shared fixtures ─────────────────────────────────────────────────────────


def _jp_hist_with_jpco(well: str = "MPB-30") -> pd.DataFrame:
    """Three installs on one well, including a JPCO same-day pull+set where
    the OLD pump's logged ``Date Pulled`` LAGS the new pump's ``Date Set`` by
    19 days — exactly the tracker gotcha CLAUDE.md warns about. A tenure
    rule that (wrongly) leaned on ``Date Pulled`` would keep attributing
    early/mid-January 2024 tests to the OLD pump; the correct
    Date-Set-to-next-Date-Set rule attributes them to the pump set on
    2024-01-01.
    """
    return pd.DataFrame(
        {
            "Well Name": [well, well, well],
            "Date Set": pd.to_datetime(["2023-01-15", "2024-01-01", "2025-06-15"]),
            "Date Pulled": pd.to_datetime([None, "2024-01-20", None]),
            "Nozzle Number": [10, 12, 13],
            "Throat Ratio": ["A", "B", "C"],
            "Tubing Diameter": [4.5, 4.5, 4.5],
        }
    )


def _jp_hist_no_change(well: str = "MPB-31") -> pd.DataFrame:
    """Single install — the pump at any test date equals the current pump."""
    return pd.DataFrame(
        {
            "Well Name": [well],
            "Date Set": pd.to_datetime(["2023-05-01"]),
            "Date Pulled": pd.to_datetime([None]),
            "Nozzle Number": [10],
            "Throat Ratio": ["C"],
            "Tubing Diameter": [4.5],
        }
    )


class FakeSt:
    """Minimal streamlit stand-in exposing a real dict session_state."""

    def __init__(self, session_state=None):
        self.session_state = session_state or {}


# ── jp_calibration._resolve_pump_for_test ───────────────────────────────────


class TestResolvePumpForTestJpCalibration:
    def test_pump_changed_uses_test_date_pump_not_current(self):
        hist = _jp_hist_with_jpco()
        # Between the 2024-01-01 and 2025-06-15 installs -> at-test pump is
        # 12B, current pump is 13C.
        pump, changed = jc._resolve_pump_for_test(
            hist, "MPB-30", pd.Timestamp("2024-05-01")
        )
        assert pump["nozzle_no"] == "12"
        assert pump["throat_ratio"] == "B"
        assert changed is True

    def test_jpco_same_day_ignores_date_pulled(self):
        """Test date falls AFTER the new pump's Date Set but BEFORE the old
        pump's (lagging) logged Date Pulled — must resolve to the NEW pump.
        """
        hist = _jp_hist_with_jpco()
        pump, changed = jc._resolve_pump_for_test(
            hist, "MPB-30", pd.Timestamp("2024-01-10")
        )
        assert pump["nozzle_no"] == "12"
        assert pump["throat_ratio"] == "B"
        assert changed is True  # differs from current (13C)

    def test_no_change_since_test(self):
        hist = _jp_hist_no_change()
        pump, changed = jc._resolve_pump_for_test(
            hist, "MPB-31", pd.Timestamp("2024-05-01")
        )
        assert pump["nozzle_no"] == "10"
        assert pump["throat_ratio"] == "C"
        assert changed is False

    def test_test_date_before_first_install_falls_back_to_current(self):
        hist = _jp_hist_with_jpco()
        pump, changed = jc._resolve_pump_for_test(
            hist, "MPB-30", pd.Timestamp("2020-01-01")
        )
        # get_pump_at_date returns None (predates first install); falls back
        # to the current pump (13C). Since pump_at IS current_pump in this
        # fallback, there's nothing to flag as "changed" — the guard only
        # fires when we have two DIFFERENT resolved pumps to compare.
        assert pump["nozzle_no"] == "13"
        assert changed is False

    def test_unknown_well_returns_none(self):
        hist = _jp_hist_with_jpco()
        pump, changed = jc._resolve_pump_for_test(
            hist, "FAKE-99", pd.Timestamp("2024-05-01")
        )
        assert pump is None
        assert changed is False

    def test_accepts_precomputed_current_pump(self):
        hist = _jp_hist_with_jpco()
        current = {"nozzle_no": "13", "throat_ratio": "C"}
        pump, changed = jc._resolve_pump_for_test(
            hist, "MPB-30", pd.Timestamp("2024-05-01"), current_pump=current
        )
        assert pump["nozzle_no"] == "12"
        assert changed is True


# ── jp_calibration._latest_bhp_whp_paired_per_well (P1-35) ─────────────────


class TestLatestBhpWhpPaired:
    def test_couples_bhp_and_whp_from_same_row(self, monkeypatch):
        # An older test has BOTH BHP and WHP; the latest test has BHP but NO
        # WHP reading. The independent-pick bug would return whp=240 (from
        # the older row) paired with bhp=900 (from the newer row) — two
        # different test days. The fix must keep whp=None (the newer row's
        # own value), not back-fill from the older row.
        raw = pd.DataFrame(
            {
                "well": ["MPB-30", "MPB-30"],
                "WtDate": pd.to_datetime(["2024-01-01", "2024-05-01"]),
                "BHP": [850.0, 900.0],
                "whp": [240.0, np.nan],
            }
        )
        monkeypatch.setattr(jc, "fetch_well_tests_raw", lambda months_back: raw)

        result = jc._latest_bhp_whp_paired_per_well(6)

        assert result["MPB-30"]["bhp"] == 900.0
        assert result["MPB-30"]["whp"] is None
        assert result["MPB-30"]["date"] == pd.Timestamp("2024-05-01")

    def test_keeps_whp_when_latest_row_has_it(self, monkeypatch):
        raw = pd.DataFrame(
            {
                "well": ["MPB-31"],
                "WtDate": pd.to_datetime(["2024-05-01"]),
                "BHP": [800.0],
                "whp": [225.0],
            }
        )
        monkeypatch.setattr(jc, "fetch_well_tests_raw", lambda months_back: raw)

        result = jc._latest_bhp_whp_paired_per_well(6)
        assert result["MPB-31"] == {
            "bhp": 800.0,
            "whp": 225.0,
            "date": pd.Timestamp("2024-05-01"),
        }

    def test_empty_when_no_bhp_rows(self, monkeypatch):
        monkeypatch.setattr(
            jc, "fetch_well_tests_raw", lambda months_back: pd.DataFrame()
        )
        assert jc._latest_bhp_whp_paired_per_well(6) == {}


# ── jp_calibration._build_calibration_input_table (integration) ────────────


class TestBuildCalibrationInputTable:
    def _patch_common(self, monkeypatch, raw, hist):
        monkeypatch.setattr(jc, "fetch_well_tests_raw", lambda months_back: raw)
        monkeypatch.setattr(
            jc, "load_well_characteristics", lambda: pd.DataFrame({"Well": []})
        )
        monkeypatch.setattr(jc, "live_pf_for_seed", lambda wn: None)
        monkeypatch.setattr(jc, "st", FakeSt(session_state={"jp_history_df": hist}))

    def test_pump_changed_row_flagged_and_include_defaults_off(self, monkeypatch):
        hist = _jp_hist_with_jpco()
        raw = pd.DataFrame(
            {
                "well": ["MPB-30"],
                "WtDate": pd.to_datetime(["2024-05-01"]),
                "BHP": [900.0],
                "whp": [np.nan],
            }
        )
        self._patch_common(monkeypatch, raw, hist)

        built = jc._build_calibration_input_table(6)
        assert built is not None
        df, pump_info = built
        row = df.iloc[0]

        assert row["Well"] == "MPB-30"
        # Modeled with the pump installed AT the test date (12B), not the
        # current pump (13C) — this is the P1-29 fix.
        assert row["Pump"] == "12B"
        assert row["Pump changed"] == True  # noqa: E712 (numpy bool from pandas)
        # Soft guard: stale-pump rows default OUT of the run.
        assert row["Include"] == False  # noqa: E712
        # WHP paired from the SAME row as BHP (P1-35) — this row's whp is
        # NaN, so it must be None, not silently borrowed from another test.
        assert pd.isna(row["WHP (psi)"]) or row["WHP (psi)"] is None

        assert pump_info["MPB-30"]["nozzle"] == "12"
        assert pump_info["MPB-30"]["throat"] == "B"
        assert pump_info["MPB-30"]["pump_changed"] is True

    def test_pump_unchanged_row_include_defaults_on(self, monkeypatch):
        hist = _jp_hist_no_change()
        raw = pd.DataFrame(
            {
                "well": ["MPB-31"],
                "WtDate": pd.to_datetime(["2024-05-01"]),
                "BHP": [800.0],
                "whp": [222.0],
            }
        )
        self._patch_common(monkeypatch, raw, hist)

        built = jc._build_calibration_input_table(6)
        assert built is not None
        df, pump_info = built
        row = df.iloc[0]

        assert row["Pump"] == "10C"
        assert row["Pump changed"] == False  # noqa: E712
        assert row["Include"] == True  # noqa: E712
        assert row["WHP (psi)"] == 222
        assert pump_info["MPB-31"]["pump_changed"] is False

    def test_no_bhp_wells_returns_none(self, monkeypatch):
        self._patch_common(monkeypatch, pd.DataFrame(), _jp_hist_no_change())
        assert jc._build_calibration_input_table(6) is None


# ── pf_scenario._resolve_pump_for_test (duplicated helper) ──────────────────


class TestResolvePumpForTestPfScenario:
    def test_pump_changed_uses_test_date_pump(self):
        hist = _jp_hist_with_jpco()
        pump, changed = pfs._resolve_pump_for_test(
            hist, "MPB-30", pd.Timestamp("2024-05-01")
        )
        assert pump["nozzle_no"] == "12"
        assert pump["throat_ratio"] == "B"
        assert changed is True

    def test_jpco_same_day_ignores_date_pulled(self):
        hist = _jp_hist_with_jpco()
        pump, changed = pfs._resolve_pump_for_test(
            hist, "MPB-30", pd.Timestamp("2024-01-10")
        )
        assert pump["nozzle_no"] == "12"
        assert changed is True


# ── pf_scenario._latest_bhp_with_date_per_well (P1-29 plumbing) ─────────────


class TestLatestBhpWithDatePfScenario:
    def test_returns_bhp_and_date_together(self, monkeypatch):
        raw = pd.DataFrame(
            {
                "well": ["MPB-30", "MPB-30"],
                "WtDate": pd.to_datetime(["2024-01-01", "2024-05-01"]),
                "BHP": [850.0, 900.0],
            }
        )
        monkeypatch.setattr(pfs, "fetch_well_tests_raw", lambda months_back: raw)

        result = pfs._latest_bhp_with_date_per_well(6)
        bhp, date = result["MPB-30"]
        assert bhp == 900.0
        assert date == pd.Timestamp("2024-05-01")


# ── pf_scenario._estimate_gaugeless_ipr pump resolution (P1-29) ────────────


class TestEstimateGaugelessIprPumpResolution:
    def test_back_calc_uses_pump_at_test_date_and_flags_change(self, monkeypatch):
        hist = _jp_hist_with_jpco()
        raw = pd.DataFrame(
            {
                "well": ["MPB-30"],
                "WtDate": pd.to_datetime(["2024-05-01"]),
                "BHP": [np.nan],
                "WtOilVol": [100.0],
                "WtWaterVol": [80.0],
                "WtTotalFluid": [180.0],
                "form_wc": [np.nan],
                "fgor": [np.nan],
                "whp": [230.0],
            }
        )
        monkeypatch.setattr(pfs, "fetch_well_tests_raw", lambda months_back: raw)

        captured = {}

        def fake_estimate_bhp(qoil_std, wc, fgor, pwh, tsu, ppf_surf, jpump, *a, **kw):
            # Record which pump the physics back-calc was actually built
            # against — must be the test-date pump (12B), not current (13C).
            captured["nozzle"] = jpump.noz_no
            captured["throat"] = jpump.rat_ar
            return 1000.0

        monkeypatch.setattr(pfs, "_estimate_bhp", fake_estimate_bhp)

        chars = {"MPB-30": {"is_sch": True, "form_temp": 75.0, "JP_MD": 4000.0}}
        result = pfs._estimate_gaugeless_ipr(["MPB-30"], 6, 3400.0, hist, chars)

        assert captured["nozzle"] == "12"
        assert captured["throat"] == "B"
        assert result["MPB-30"]["_pump_changed"] is True

    def test_back_calc_no_flag_when_pump_unchanged(self, monkeypatch):
        hist = _jp_hist_no_change()
        raw = pd.DataFrame(
            {
                "well": ["MPB-31"],
                "WtDate": pd.to_datetime(["2024-05-01"]),
                "BHP": [np.nan],
                "WtOilVol": [90.0],
                "WtWaterVol": [70.0],
                "WtTotalFluid": [160.0],
                "form_wc": [np.nan],
                "fgor": [np.nan],
                "whp": [210.0],
            }
        )
        monkeypatch.setattr(pfs, "fetch_well_tests_raw", lambda months_back: raw)
        monkeypatch.setattr(pfs, "_estimate_bhp", lambda *a, **kw: 900.0)

        chars = {"MPB-31": {"is_sch": True, "form_temp": 75.0, "JP_MD": 4000.0}}
        result = pfs._estimate_gaugeless_ipr(["MPB-31"], 6, 3400.0, hist, chars)

        assert result["MPB-31"]["_pump_changed"] is False
