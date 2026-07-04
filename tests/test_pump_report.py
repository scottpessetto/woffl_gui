"""Tests for woffl.assembly.pump_report — the Pump Report Card analysis.

Fixtures mirror real field shapes found in the 2026-07-03 feasibility probe:
B-28's same-pump JPCO churn (12B ×5 = one era), S-17's throat-only tracker
rows, MPE-24's WC confounder (same pump, 62% → 88% WC across eras).
"""

import numpy as np
import pandas as pd
import pytest

from woffl.assembly.pump_report import (
    MIN_TESTS_TO_RANK,
    build_pump_eras,
    build_report,
    build_verdict,
    default_good_thresholds,
    era_metrics,
    format_pump,
    rank_eras,
)

END = pd.Timestamp("2026-07-03")


def _jp(rows):
    return pd.DataFrame(
        [
            {
                "Well Name": "MPX-1",
                "Date Set": pd.Timestamp(d),
                "Nozzle Number": nz,
                "Throat Ratio": th,
            }
            for d, nz, th in rows
        ]
    )


def _tests(rows):
    """rows: (date, oil, water, lift, bhp, pf)."""
    return pd.DataFrame(
        [
            {
                "well": "MPX-1",
                "WtDate": pd.Timestamp(d),
                "WtOilVol": oil,
                "WtWaterVol": wat,
                "lift_wat": lift,
                "BHP": bhp,
                "pf_press": pf,
            }
            for d, oil, wat, lift, bhp, pf in rows
        ]
    )


# ── format_pump ────────────────────────────────────────────────────────────


class TestFormatPump:
    def test_standard(self):
        assert format_pump(12, "B") == "12B"

    def test_float_nozzle(self):
        assert format_pump("12.0", "B") == "12B"

    def test_missing_nozzle_s17_style(self):
        assert format_pump(np.nan, "D") == "D"

    def test_both_missing(self):
        assert format_pump(np.nan, np.nan) == "?"


# ── era building ───────────────────────────────────────────────────────────


class TestBuildPumpEras:
    def test_same_pump_resets_coalesce(self):
        # B-28 shape: 12B set 3x (JPCO churn), then 13A, then 13B twice.
        hist = _jp(
            [
                ("2023-12-07", 12, "B"),
                ("2023-12-12", 12, "B"),
                ("2024-05-01", 12, "B"),
                ("2024-11-11", 13, "A"),
                ("2025-10-26", 13, "B"),
                ("2025-11-25", 13, "B"),
            ]
        )
        eras = build_pump_eras(hist, end_date=END)
        assert [e["pump"] for e in eras] == ["12B", "13A", "13B"]
        assert [e["installs"] for e in eras] == [3, 1, 2]
        assert eras[0]["start"] == pd.Timestamp("2023-12-07")
        assert eras[0]["end"] == pd.Timestamp("2024-11-11")
        assert eras[0]["days"] == 340
        assert eras[-1]["active"] and not eras[0]["active"]
        assert eras[-1]["end"] == END

    def test_alternating_pumps_do_not_merge(self):
        hist = _jp([("2024-01-01", 12, "C"), ("2024-06-01", 10, "C"), ("2024-08-01", 12, "C")])
        eras = build_pump_eras(hist, end_date=END)
        assert [e["pump"] for e in eras] == ["12C", "10C", "12C"]

    def test_empty_and_undated(self):
        assert build_pump_eras(pd.DataFrame()) == []
        hist = _jp([("2024-01-01", 12, "B")])
        hist.loc[0, "Date Set"] = pd.NaT
        assert build_pump_eras(hist, end_date=END) == []


# ── per-era metrics ────────────────────────────────────────────────────────


class TestEraMetrics:
    def _era(self):
        return {
            "pump": "13A",
            "start": pd.Timestamp("2024-01-01"),
            "end": pd.Timestamp("2025-01-01"),
            "days": 366,
            "installs": 1,
            "active": False,
        }

    def test_medians_and_efficiency(self):
        tests = _tests(
            [
                ("2024-02-01", 400, 600, 2000, 1200, 2400),
                ("2024-04-01", 500, 500, 2000, 1150, 2500),
                ("2024-06-01", 450, 550, 1800, 1180, 2450),
                ("2025-02-01", 999, 1, 1, 1, 9999),  # outside era — ignored
            ]
        )
        m = era_metrics(self._era(), tests, None, good_oil=430, good_bhp=None)
        assert m["n_tests"] == 3
        assert m["med_oil"] == pytest.approx(450)
        assert m["med_wc"] == pytest.approx(0.55, abs=0.01)
        assert m["med_pf"] == pytest.approx(2450)
        assert m["n_pf"] == 3
        assert m["med_bhp"] == pytest.approx(1180)
        assert m["oil_per_pf"] == pytest.approx(450 / 1800, rel=1e-6)

    def test_good_state_from_tests(self):
        # good ≥ 400: good, good, bad, good → longest run = first two (59 days)
        tests = _tests(
            [
                ("2024-02-01", 420, 600, 2000, None, None),
                ("2024-03-31", 410, 600, 2000, None, None),
                ("2024-05-01", 300, 700, 2000, None, None),
                ("2024-06-01", 450, 550, 2000, None, None),
            ]
        )
        m = era_metrics(self._era(), tests, None, good_oil=400, good_bhp=None)
        assert m["good_test_frac"] == pytest.approx(0.75)
        assert m["good_run_days"] == 59

    def test_good_bhp_from_daily_series(self):
        daily = pd.DataFrame(
            {
                "tag_date": pd.date_range("2024-02-01", periods=10, freq="D"),
                "bhp": [1100] * 6 + [1400] * 2 + [1100] * 2,
            }
        )
        tests = _tests([("2024-02-01", 400, 600, 2000, 1100, 2400)])
        m = era_metrics(self._era(), tests, daily, good_oil=None, good_bhp=1200)
        assert m["good_bhp_frac"] == pytest.approx(0.8)
        assert m["good_bhp_run_days"] == 5  # first 6 consecutive days span 5

    def test_missing_pf_column_tolerated(self):
        tests = _tests([("2024-02-01", 400, 600, 2000, 1100, None)]).drop(
            columns=["pf_press"]
        )
        m = era_metrics(self._era(), tests, None, good_oil=None, good_bhp=None)
        assert m["med_pf"] is None
        assert m["n_pf"] == 0

    def test_wc_drift(self):
        rows = [
            (f"2024-{mm:02d}-01", 400, 400 * wc / (1 - wc), 2000, None, None)
            for mm, wc in zip(range(1, 10), [0.5] * 3 + [0.6] * 3 + [0.8] * 3)
        ]
        m = era_metrics(self._era(), _tests(rows), None, None, None)
        assert m["wc_drift"] == pytest.approx(0.3, abs=0.02)


# ── thresholds ─────────────────────────────────────────────────────────────


class TestDefaultThresholds:
    def test_seeds_from_history(self):
        tests = _tests(
            [(f"2024-0{i}-01", oil, 500, 2000, 1200, None) for i, oil in
             enumerate([100, 200, 300, 400, 500, 600, 700, 800], start=1)]
        )
        daily = pd.DataFrame(
            {"tag_date": pd.date_range("2024-01-01", periods=5), "bhp": [1111] * 5}
        )
        good_oil, good_bhp = default_good_thresholds(tests, daily)
        # P75 of 100..800 = 625 → 75% = 468.75 → 470
        assert good_oil == 470
        assert good_bhp == 1100  # 1111 → nearest 25

    def test_bhp_falls_back_to_tests(self):
        tests = _tests([("2024-01-01", 400, 500, 2000, 1250, None)])
        good_oil, good_bhp = default_good_thresholds(tests, None)
        assert good_bhp == 1250

    def test_no_data(self):
        assert default_good_thresholds(_tests([]), None) == (None, None)


# ── ranking + verdict ──────────────────────────────────────────────────────


def _metric(pump, med_oil, n_tests=10, days=300, **over):
    base = {
        "pump": pump,
        "start": pd.Timestamp("2024-01-01"),
        "end": pd.Timestamp("2024-10-27"),
        "days": days,
        "installs": 1,
        "active": False,
        "n_tests": n_tests,
        "med_oil": med_oil,
        "p90_oil": med_oil * 1.2 if med_oil else None,
        "oil_drift": None,
        "med_wc": 0.6,
        "wc_drift": None,
        "med_pf": 2500.0,
        "n_pf": n_tests,
        "med_bhp": 1200.0,
        "oil_per_pf": 0.2,
        "good_test_frac": 0.8,
        "good_run_days": 100,
        "good_bhp_frac": 0.7,
        "good_bhp_run_days": 90,
    }
    base.update(over)
    return base


class TestRankAndVerdict:
    def test_higher_oil_wins_all_else_equal(self):
        ranked = rank_eras([_metric("10B", 300), _metric("13A", 500)])
        assert ranked[0]["pump"] == "13A"
        assert ranked[0]["score"] > ranked[1]["score"]

    def test_thin_era_not_ranked(self):
        ranked = rank_eras(
            [_metric("13A", 500), _metric("9B", 900, n_tests=MIN_TESTS_TO_RANK - 1)]
        )
        assert ranked[0]["pump"] == "13A"  # 9B unrankable despite higher oil
        assert ranked[-1]["score"] is None and not ranked[-1]["ranked"]

    def test_missing_components_renormalized(self):
        # No PF/BHP evidence at all — still gets a score from oil+stability(tests)+longevity.
        m = _metric(
            "12B", 400, med_pf=None, n_pf=0, oil_per_pf=None,
            good_bhp_frac=None, good_bhp_run_days=None,
        )
        ranked = rank_eras([m])
        assert ranked[0]["score"] is not None

    def test_wc_shift_caveat(self):
        # MPE-24 shape: best era at 62% WC, runner-up at 88%.
        ranked = rank_eras(
            [_metric("12C", 279, med_wc=0.62), _metric("10C", 109, med_wc=0.88)]
        )
        v = build_verdict(ranked, "MPE-24")
        assert v["best"]["pump"] == "12C"
        assert any("water cut shifted" in c for c in v["caveats"])

    def test_no_pf_era_caveat(self):
        ranked = rank_eras(
            [_metric("13A", 500), _metric("9B", 300, n_pf=0, med_pf=None, oil_per_pf=None)]
        )
        v = build_verdict(ranked, "MPX-1")
        assert any("pre-dates" in c and "9B" in c for c in v["caveats"])

    def test_small_n_caveat(self):
        ranked = rank_eras([_metric("13A", 500, n_tests=3)])
        v = build_verdict(ranked, "MPX-1")
        assert any("thin evidence" in c for c in v["caveats"])

    def test_active_era_close_caveat(self):
        ranked = rank_eras(
            [
                _metric("13A", 500, days=349),
                _metric("13B", 480, days=60, active=True, good_run_days=40),
            ]
        )
        v = build_verdict(ranked, "MPB-28")
        assert v["best"]["pump"] == "13A"
        assert any("current pump" in c for c in v["caveats"])

    def test_verdict_none_when_nothing_ranked(self):
        ranked = rank_eras([_metric("13A", 500, n_tests=1)])
        assert build_verdict(ranked, "MPX-1") is None


# ── end-to-end ─────────────────────────────────────────────────────────────


class TestBuildReport:
    def test_end_to_end(self):
        hist = _jp(
            [
                ("2024-01-01", 12, "B"),
                ("2024-03-01", 12, "B"),
                ("2024-09-01", 13, "A"),
            ]
        )
        rows = []
        # 12B era: mediocre oil; 13A era: better oil.
        for mm in (1, 3, 5, 7):
            rows.append((f"2024-{mm:02d}-15", 300, 700, 2000, 1250, 2400))
        for mm in (9, 10, 11, 12):
            rows.append((f"2024-{mm:02d}-15", 480, 520, 1900, 1150, 2450))
        ranked, verdict = build_report(
            hist, _tests(rows), None, good_oil=400, good_bhp=None, end_date=END
        )
        assert [m["pump"] for m in ranked][0] == "13A"
        assert ranked[0]["installs"] == 1 and ranked[-1]["installs"] == 2
        assert verdict is not None
        assert verdict["best"]["pump"] == "13A"
        assert any("BOPD median" in r for r in verdict["reasons"])

    def test_no_tests(self):
        hist = _jp([("2024-01-01", 12, "B")])
        ranked, verdict = build_report(hist, _tests([]), None, 400, None, END)
        assert ranked == [] and verdict is None
