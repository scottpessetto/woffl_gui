"""server/services/calibration_points.py - multipoint fit-set builder.

All synthetic (no network): points_for_well is PURE, and the pad_points
test monkeypatches the fleet-frame / jp-history / test fetchers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import server.services.calibration_points as cp
from server.services.calibration_points import (
    DAILY_WEIGHT,
    MAX_FIT_POINTS,
    TEST_WEIGHT,
    pad_points,
    points_for_well,
)

_WELL = "MPM-64"
_ERA = pd.Timestamp("2026-06-01")


def _jp_hist(date_set=_ERA):
    """Two installs - the current era must resolve to the LATEST Date Set."""
    return pd.DataFrame(
        [
            {"Well Name": _WELL, "Date Set": pd.Timestamp("2025-01-15"),
             "Nozzle Number": 10, "Throat Ratio": "A"},
            {"Well Name": _WELL, "Date Set": date_set,
             "Nozzle Number": 12, "Throat Ratio": "B"},
        ]
    )


def _daily_frame(days_ppf_bhp, tubing=250.0):
    """(date, ppf, bhp) rows in the fleet-pressure shape (reverse circ:
    annulus carries PF; 250 psi tubing is below the valid-PF floor)."""
    return pd.DataFrame(
        [
            {"well": _WELL, "sample_date": pd.Timestamp(d), "tubing_prs": tubing,
             "inn_ann_prs": ppf, "btmhole_prs": bhp}
            for d, ppf, bhp in days_ppf_bhp
        ]
    )


def _pf_frame(days_rate):
    return pd.DataFrame(
        [
            {"well": _WELL, "pfdate": pd.Timestamp(d), "pwr_fld_net": rate}
            for d, rate in days_rate
        ]
    )


def _test_row(date, bhp=350.0, lift=2500.0, whp=120.0, ppf=3200.0,
              qtot=1800.0, oil=900.0, water=900.0, wc=0.5, fgor=400.0):
    return {"WtDate": pd.Timestamp(date), "BHP": bhp, "lift_wat": lift,
            "whp": whp, "pf_press": ppf, "WtTotalFluid": qtot,
            "WtOilVol": oil, "WtWaterVol": water, "form_wc": wc, "fgor": fgor}


def _spread_days(start, n, ppf_lo=2800.0, ppf_hi=3600.0, bhp=320.0, rate=2400.0):
    """n flowing daily rows with ppf swept across [ppf_lo, ppf_hi]."""
    dates = [pd.Timestamp(start) + pd.Timedelta(days=i) for i in range(n)]
    ppfs = np.linspace(ppf_lo, ppf_hi, n)
    daily = _daily_frame([(d, p, bhp) for d, p in zip(dates, ppfs)])
    pf = _pf_frame([(d, rate) for d in dates])
    return daily, pf


def _build(daily, pf, tests=None, **kw):
    tests_df = pd.DataFrame(tests) if tests is not None else None
    return points_for_well(
        _WELL, jp_hist=_jp_hist(), tests_df=tests_df,
        daily_df=daily, pf_df=pf, **kw,
    )


# ---------------------------------------------------------------------------
# Era gating + filter chain
# ---------------------------------------------------------------------------


def test_era_gating_drops_pre_era_daily_and_test_rows():
    daily, pf = _spread_days("2026-06-05", 12)
    # pre-era daily row (would otherwise pass every filter)
    daily = pd.concat(
        [daily, _daily_frame([("2026-05-20", 3000.0, 330.0)])], ignore_index=True
    )
    pf = pd.concat([pf, _pf_frame([("2026-05-20", 2400.0)])], ignore_index=True)
    tests = [_test_row("2026-05-15"), _test_row("2026-06-10")]

    res = _build(daily, pf, tests)
    assert res["era_start"] == "2026-06-01"
    assert res["pump"] == {"nozzle": "12", "throat": "B", "date_set": "2026-06-01"}
    assert res["n_test"] == 1  # the 05-15 test predates the era
    dates = [p["date"] for p in res["points"]]
    assert "2026-05-20" not in dates
    assert "2026-05-15" not in dates
    assert res["refusal"] is None


def test_shut_in_day_low_pf_rate_dropped():
    daily, pf = _spread_days("2026-06-05", 12)
    # a 13th day flowing on pressure but with pf_rate 300 -> shut-in, dropped
    daily = pd.concat(
        [daily, _daily_frame([("2026-06-20", 3300.0, 320.0)])], ignore_index=True
    )
    pf = pd.concat([pf, _pf_frame([("2026-06-20", 300.0)])], ignore_index=True)

    res = _build(daily, pf)
    assert "2026-06-20" not in [p["date"] for p in res["points"]]
    assert res["n_daily"] == 12


def test_glitch_bhp_and_bad_ppf_days_dropped():
    daily, pf = _spread_days("2026-06-05", 12)
    extra = _daily_frame(
        [
            ("2026-06-21", 3200.0, 29.5),   # dead gauge (bhp <= 50)
            ("2026-06-22", 6000.0, 320.0),  # ppf above the 5500 header cap
            ("2026-06-23", 400.0, 320.0),   # no valid PF reading (< 800)
        ]
    )
    daily = pd.concat([daily, extra], ignore_index=True)
    pf = pd.concat(
        [pf, _pf_frame([("2026-06-21", 2400.0), ("2026-06-22", 2400.0),
                        ("2026-06-23", 2400.0)])],
        ignore_index=True,
    )

    res = _build(daily, pf)
    kept = [p["date"] for p in res["points"]]
    assert not {"2026-06-21", "2026-06-22", "2026-06-23"} & set(kept)
    assert res["n_daily"] == 12


# ---------------------------------------------------------------------------
# Nearest-test attachment + fallback
# ---------------------------------------------------------------------------


def test_nearest_test_within_30_days_attaches_and_fallback_beyond():
    daily, pf = _spread_days("2026-06-05", 12)  # 06-05 .. 06-16
    # far daily 40+ days after the only test -> fallback values
    daily = pd.concat(
        [daily, _daily_frame([("2026-08-01", 3500.0, 310.0)])], ignore_index=True
    )
    pf = pd.concat([pf, _pf_frame([("2026-08-01", 2400.0)])], ignore_index=True)
    tests = [_test_row("2026-06-10", qtot=2000.0, oil=1200.0, wc=0.4, fgor=300.0)]

    res = _build(
        daily, pf, tests,
        fallback_qtot=1500.0, fallback_wc=0.6, fallback_fgor=250.0,
    )
    near = next(p for p in res["points"] if p["date"] == "2026-06-05")
    assert near["kind"] == "daily"
    assert near["qtot"] == 2000.0
    assert near["oil"] == 1200.0
    assert near["wc"] == pytest.approx(0.4)
    assert near["fgor"] == 300.0

    far = next(p for p in res["points"] if p["date"] == "2026-08-01")
    assert far["qtot"] == 1500.0
    assert far["wc"] == pytest.approx(0.6)
    assert far["fgor"] == 250.0
    assert far["oil"] == pytest.approx(1500.0 * 0.4)  # qtot * (1 - wc)


def test_attach_picks_nearest_of_two_tests_and_computes_wc_when_missing():
    daily, pf = _spread_days("2026-06-05", 12)
    tests = [
        # nearest to the early dailies; no form_wc -> computed from volumes.
        # pf_press NaN keeps it OUT of the fit set but it still anchors.
        _test_row("2026-06-06", qtot=2000.0, oil=500.0, water=1500.0,
                  wc=np.nan, ppf=np.nan, fgor=111.0),
        _test_row("2026-06-18", qtot=1000.0, oil=800.0, water=200.0,
                  wc=0.2, fgor=999.0),
    ]

    res = _build(daily, pf, tests)
    assert res["n_test"] == 1  # only the 06-18 test survives the ppf gate
    early = next(p for p in res["points"] if p["date"] == "2026-06-05")
    assert early["wc"] == pytest.approx(1500.0 / 2000.0)  # computed water frac
    assert early["fgor"] == 111.0
    late = next(p for p in res["points"] if p["date"] == "2026-06-16")
    # 06-16 is 10 days from 06-06 but only 2 from 06-18 -> the LATER test
    assert late["wc"] == pytest.approx(0.2)
    assert late["fgor"] == 999.0


# ---------------------------------------------------------------------------
# Stratified cap
# ---------------------------------------------------------------------------


def test_stratified_cap_keeps_all_tests_and_ppf_endpoints():
    daily, pf = _spread_days("2026-06-05", 30, ppf_lo=2800.0, ppf_hi=3800.0)
    tests = [_test_row("2026-06-10"), _test_row("2026-06-25")]

    res = _build(daily, pf, tests)
    assert len(res["points"]) == MAX_FIT_POINTS
    assert res["n_test"] == 2
    assert res["n_daily"] == MAX_FIT_POINTS - 2
    daily_ppfs = [p["ppf"] for p in res["points"] if p["kind"] == "daily"]
    assert min(daily_ppfs) == pytest.approx(2800.0)  # endpoints survive
    assert max(daily_ppfs) == pytest.approx(3800.0)
    # spread is reported over the FULL usable set, not the capped subset
    assert res["ppf_spread"] == pytest.approx(1000.0)
    assert res["refusal"] is None


def test_cap_noop_when_under_limit():
    daily, pf = _spread_days("2026-06-05", 12)
    res = _build(daily, pf)
    assert res["n_daily"] == 12
    assert len(res["points"]) == 12


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_young_era_refusal_string_and_points_still_returned():
    daily, pf = _spread_days("2026-06-05", 5)
    res = _build(daily, pf)
    assert res["refusal"] == "young pump era - 5 usable points"
    assert len(res["points"]) == 5  # refused wells still carry their points


def test_low_spread_refusal_string():
    daily, pf = _spread_days("2026-06-05", 12, ppf_lo=3000.0, ppf_hi=3050.0)
    res = _build(daily, pf)
    assert res["refusal"] == "not identifiable - ppf spread 50 psi in this pump era"
    assert res["ppf_spread"] == pytest.approx(50.0)
    assert len(res["points"]) == 12


def test_no_pump_record_refusal():
    daily, pf = _spread_days("2026-06-05", 12)
    res = points_for_well(
        _WELL, jp_hist=pd.DataFrame(columns=["Well Name", "Date Set"]),
        tests_df=None, daily_df=daily, pf_df=pf,
    )
    assert res["refusal"] == "no current pump record in jp_history"
    assert res["points"] == []


# ---------------------------------------------------------------------------
# Weight / kind / pwh assignment
# ---------------------------------------------------------------------------


def test_weight_kind_and_pwh_assignment():
    daily, pf = _spread_days("2026-06-05", 12)
    tests = [
        _test_row("2026-06-10", whp=135.0),
        _test_row("2026-06-20", whp=np.nan),  # no whp -> surf_pres fallback
    ]

    res = _build(daily, pf, tests, surf_pres=95.0)
    by_kind = {}
    for p in res["points"]:
        by_kind.setdefault(p["kind"], []).append(p)

    assert all(p["weight"] == TEST_WEIGHT for p in by_kind["test"])
    assert all(p["weight"] == DAILY_WEIGHT for p in by_kind["daily"])
    assert all(p["pwh"] == 95.0 for p in by_kind["daily"])
    whps = sorted(p["pwh"] for p in by_kind["test"])
    assert whps == [95.0, 135.0]
    # daily points have no measured rate columns of their own
    assert all(p["pf_rate"] > 0 for p in res["points"])


def test_res_pres_buildup_filter():
    daily, pf = _spread_days("2026-06-05", 12, bhp=320.0)
    daily = pd.concat(
        [daily, _daily_frame([("2026-06-20", 3300.0, 1850.0)])], ignore_index=True
    )
    pf = pd.concat([pf, _pf_frame([("2026-06-20", 2400.0)])], ignore_index=True)

    res = _build(daily, pf, res_pres=1800.0)
    assert "2026-06-20" not in [p["date"] for p in res["points"]]


def test_steady_state_filter_drops_transient_excursion():
    # 15 flat days at 320 psi with a 3-day transient in the middle: shut-in
    # buildup peak (1800), decaying tail (1100), restart undershoot (200).
    # The centered 5-day rolling median holds the 320 baseline through the
    # excursion, so exactly those three days trip STEADY_STATE_TOL_PSI.
    dates = [pd.Timestamp("2026-06-05") + pd.Timedelta(days=i) for i in range(15)]
    ppfs = np.linspace(2800.0, 3600.0, 15)
    bhps = [320.0] * 15
    bhps[6], bhps[7], bhps[8] = 1800.0, 1100.0, 200.0
    daily = _daily_frame(list(zip(dates, ppfs, bhps)))
    pf = _pf_frame([(d, 2400.0) for d in dates])

    res = _build(daily, pf)
    kept = [p["date"] for p in res["points"]]
    assert not {"2026-06-11", "2026-06-12", "2026-06-13"} & set(kept)
    assert len(kept) == 12
    assert res["refusal"] is None


def test_steady_state_filter_keeps_short_series():
    # 2-day era: the rolling median needs >= 3 days in its window, so even
    # a wild jump between the only two days keeps both rows.
    daily = _daily_frame([("2026-06-05", 3000.0, 320.0),
                          ("2026-06-06", 3100.0, 1800.0)])
    pf = _pf_frame([("2026-06-05", 2400.0), ("2026-06-06", 2400.0)])

    res = _build(daily, pf)
    assert len(res["points"]) == 2


# ---------------------------------------------------------------------------
# pad_points (monkeypatched fetchers - fail-soft contract)
# ---------------------------------------------------------------------------


def test_pad_points_fail_soft_per_well(monkeypatch):
    daily, pf = _spread_days("2026-06-05", 12)
    tests_df = pd.DataFrame([_test_row("2026-06-10")])

    import server.services.datasources as datasources
    import server.services.evidence as evidence
    import server.services.tests as tests_svc

    monkeypatch.setattr(evidence, "_fleet_pressure_daily", lambda: daily)
    monkeypatch.setattr(cp, "_fleet_pf_volume", lambda: pf)
    monkeypatch.setattr(datasources, "jp_history_safe", lambda: (_jp_hist(), "tracker"))

    def _tests_for_well(well, months, cap=0):
        if well == "MPM-99":
            raise RuntimeError("boom")
        return tests_df if well == _WELL else None

    monkeypatch.setattr(tests_svc, "tests_for_well", _tests_for_well)

    out = pad_points([_WELL, "MPM-99", "MPM-28"], surf_pres={_WELL: 95.0})
    assert _WELL in out
    assert "MPM-99" not in out  # per-well exception -> absent, never fatal
    assert out[_WELL]["n_test"] == 1
    assert out[_WELL]["n_daily"] == 12
    # MPM-28 has no rows in either frame and no pump record for its name
    assert out["MPM-28"]["refusal"] == "no current pump record in jp_history"
