"""Tests for woffl.gui.scotts_tools.header_impact pure helpers.

Exercise the override (_chosen_method), the analog-donor math (_analog_doil),
the sonic-aware verdict (_verdict), and header_trend's within-day fit +
classifier on synthetic inputs. No Databricks or Streamlit runtime needed.

Run with:
    python -m pytest tests/test_header_impact.py
"""

import numpy as np
import pandas as pd
import pytest

from woffl.gui.scotts_tools import header_impact as hi
from woffl.gui.scotts_tools import header_trend as ht


# ── _chosen_method (the empirical override) ──────────────────────────────────


def test_chosen_method_physics_keeps_physics():
    val, lab = hi._chosen_method(100.0, 30.0, "physics")
    assert val == 100.0
    assert lab == "physics"


def test_chosen_method_empirical_uses_emp_when_available():
    val, lab = hi._chosen_method(100.0, 30.0, "empirical")
    assert val == 30.0
    assert lab == "empirical"


def test_chosen_method_empirical_falls_back_to_physics_when_no_emp():
    val, lab = hi._chosen_method(100.0, float("nan"), "empirical")
    assert val == 100.0
    assert lab == "physics (emp N/A)"


def test_chosen_method_auto_keeps_physics_when_agree():
    # |100 - 110| = 10 < max(20, 0.3*100=30) → physics
    val, lab = hi._chosen_method(100.0, 110.0, "auto")
    assert val == 100.0
    assert lab == "physics"


def test_chosen_method_auto_flips_to_empirical_when_disagree():
    # |100 - 20| = 80 > 30 → empirical override
    val, lab = hi._chosen_method(100.0, 20.0, "auto")
    assert val == 20.0
    assert lab == "empirical (override)"


def test_chosen_method_nan_physics_uses_empirical_for_nonjp():
    # Non-JP wells have no physics; method becomes irrelevant.
    val, lab = hi._chosen_method(float("nan"), 42.0, "physics")
    assert val == 42.0
    assert lab == "empirical"


def test_chosen_method_nan_both_returns_nan():
    val, lab = hi._chosen_method(float("nan"), float("nan"), "auto")
    assert np.isnan(val)
    assert lab == "—"


# ── _analog_doil (gaugeless ESP via borrow-donor-drawdown) ───────────────────


def test_analog_doil_header_drop_gives_uplift_for_positive_slope():
    # Donor: slope 0.8, drawdown 540/1800 = 0.30. Target: 300 BOPD, ResP 1800.
    pwf, dbhp, doil = hi._analog_doil(
        donor_slope=0.8, donor_pwf=540.0, donor_res_pres=1800.0,
        target_qoil=300.0, target_res_pres=1800.0, delta_p=-50.0,
    )
    assert pwf == pytest.approx(540.0, rel=0.05)  # target inherits donor's drawdown
    assert dbhp < 0  # header drop → lower BHP
    assert doil > 0  # → more oil


def test_analog_doil_header_rise_gives_loss():
    _, dbhp, doil = hi._analog_doil(0.8, 540.0, 1800.0, 300.0, 1800.0, +50.0)
    assert dbhp > 0  # BHP rises
    assert doil < 0  # oil falls


def test_analog_doil_zero_delta_no_change():
    _, dbhp, doil = hi._analog_doil(0.8, 540.0, 1800.0, 300.0, 1800.0, 0.0)
    assert dbhp == 0.0
    assert doil == pytest.approx(0.0, abs=0.01)


# ── _verdict (combines physics sonic flag + empirical class) ─────────────────


def test_verdict_sonic_now_overrides_everything():
    # Already sonic at baseline → header lever won't help regardless of empirical.
    assert hi._verdict(True, True, 0.0, "responsive", True) == "sonic-decoupled"


def test_verdict_chokes_when_lowered():
    # Not sonic at baseline, sonic at scenario → partial response then choke.
    assert hi._verdict(False, True, 50.0, "responsive", True) == "chokes when lowered"


def test_verdict_no_response_when_delta_oil_tiny():
    assert hi._verdict(False, False, 0.5, "responsive", True) == "no response"


def test_verdict_responsive_agree_when_both_say_so():
    assert hi._verdict(False, False, 50.0, "responsive", True) == "responsive ✓"


def test_verdict_disagree_when_physics_says_respond_but_field_flat():
    assert hi._verdict(False, False, 50.0, "slugging", True) == "disagree — check"


def test_verdict_responsive_physics_when_no_empirical():
    # Several "empirical absent" paths should all collapse to responsive (physics).
    for emp in (None, "no tag", "no data", "insufficient"):
        assert hi._verdict(False, False, 50.0, emp, True) == "responsive (physics)"
    # compare_emp off → don't rely on empirical regardless
    assert hi._verdict(False, False, 50.0, "slugging", False) == "responsive (physics)"


# ── header_trend.classify_response ────────────────────────────────────────────


def _make_fit(mean_slope=0.8, mean_r2=0.9, n_days=30, n_good_days=25):
    """Hand-build a WithinDayFit for classification tests."""
    return ht.WithinDayFit(
        y_name="BHP", x_name="WHP",
        mean_slope=mean_slope, median_slope=mean_slope, slope_std=0.05,
        n_days=n_days, n_good_days=n_good_days, mean_r2=mean_r2,
        daily=pd.DataFrame(),
    )


def test_classify_responsive_when_enough_good_days_and_fraction():
    fit = _make_fit(n_days=30, n_good_days=20)  # 67% good
    assert ht.classify_response(fit) == "responsive"


def test_classify_slugging_when_few_good_days():
    fit = _make_fit(n_days=30, n_good_days=3)  # 10% good
    assert ht.classify_response(fit) == "slugging"


def test_classify_insufficient_when_few_fittable_days():
    fit = _make_fit(n_days=2, n_good_days=2)  # too few days
    assert ht.classify_response(fit) == "insufficient"


def test_classify_insufficient_when_fit_is_none():
    assert ht.classify_response(None) == "insufficient"


# ── header_trend.fit_within_day (the within-day slope engine) ────────────────


def _synthetic_responsive_df(planted_slope=0.8, days=20, noise=2.0, seed=0):
    """Hourly synthetic data with a real ~planted_slope BHP↔HeaderP coupling."""
    rng = np.random.default_rng(seed)
    idx, B, H, W = [], [], [], []
    for d in range(days):
        base = 200 + 0.3 * d
        for h in range(24):
            hp = base + 15 * np.sin(h / 24 * 2 * np.pi) + rng.normal(0, 1)
            W.append(hp + 30 + rng.normal(0, 1))
            H.append(hp)
            B.append(planted_slope * hp + 60 + rng.normal(0, noise))
            idx.append(pd.Timestamp("2025-01-01") + pd.Timedelta(days=d, hours=h))
    return pd.DataFrame({"BHP": B, "HeaderP": H, "WHP": W}, index=pd.DatetimeIndex(idx))


def test_fit_within_day_recovers_planted_slope():
    df = _synthetic_responsive_df(planted_slope=0.8, days=20)
    fit = ht.fit_within_day(df, y_name="BHP", x_name="HeaderP")
    assert fit is not None
    assert fit.mean_slope == pytest.approx(0.8, abs=0.05)
    assert ht.classify_response(fit) == "responsive"


def test_fit_within_day_slugging_well_has_few_good_days():
    rng = np.random.default_rng(1)
    df = _synthetic_responsive_df(days=20)
    df["BHP"] = 500 + rng.normal(0, 40, size=len(df))  # decouple BHP from header
    fit = ht.fit_within_day(df, y_name="BHP", x_name="HeaderP")
    assert fit.n_good_days <= 3  # the r²+band filters reject noise days
    assert ht.classify_response(fit) == "slugging"


def test_fit_within_day_returns_none_when_columns_absent():
    df = pd.DataFrame({"BHP": [1, 2, 3]}, index=pd.date_range("2025-01-01", periods=3))
    assert ht.fit_within_day(df, y_name="BHP", x_name="HeaderP") is None


# ── v2 review figure builders (Plotly construction, no Streamlit) ────────────


def test_response_curve_fig_builds_pad_and_field_traces():
    agg = {"deltas": [-50, 0, 50], "pads": {"G": [10.0, 0.0, -5.0]}, "ALL": [10.0, 0.0, -5.0]}
    fig = hi._response_curve_fig(agg)
    assert fig is not None
    names = [t.name for t in fig.data]
    assert "Pad G" in names and "Field (ALL)" in names


def test_response_curve_fig_single_pad_filter():
    agg = {"deltas": [-50, 0, 50], "pads": {"G": [10.0, 0.0, -5.0], "H": [1.0, 0.0, -1.0]}, "ALL": [11.0, 0.0, -6.0]}
    fig = hi._response_curve_fig(agg, pads_to_show=["G"])
    names = [t.name for t in fig.data]
    assert "Pad G" in names and "Pad H" not in names and "Field (ALL)" not in names


def test_response_curve_fig_empty_returns_none():
    assert hi._response_curve_fig({}) is None


def test_ipr_grid_fig_builds_curve_and_operating_points():
    pad_df = pd.DataFrame(
        {
            "Well": ["MPG-01"],
            "Oil now (BOPD)": [300.0],
            "BHP now (psi)": [800.0],
            "Oil scen (BOPD)": [340.0],
            "BHP scen (psi)": [740.0],
        }
    )
    ipr_rows = {"MPG-01": {"res_pres": 1800.0, "qwf": 1000.0, "pwf": 800.0, "form_wc": 0.5}}
    fig = hi._ipr_grid_fig(pad_df, ipr_rows)
    assert fig is not None
    # IPR curve + operating point (●) + scenario point (✕) = 3 traces.
    assert len(fig.data) == 3


def test_ipr_grid_fig_none_when_no_jp_rows():
    pad_df = pd.DataFrame({"Well": ["MPL-09"], "Oil now (BOPD)": [80.0], "BHP now (psi)": [np.nan]})
    assert hi._ipr_grid_fig(pad_df, ipr_rows={}) is None


def test_ipr_grid_fig_overlays_test_points_with_hover():
    pad_df = pd.DataFrame(
        {
            "Well": ["MPG-01"], "Oil now (BOPD)": [300.0], "BHP now (psi)": [800.0],
            "Oil scen (BOPD)": [340.0], "BHP scen (psi)": [740.0],
        }
    )
    ipr_rows = {"MPG-01": {"res_pres": 1800.0, "qwf": 1000.0, "pwf": 800.0, "form_wc": 0.5}}
    test_df = pd.DataFrame(
        {
            "well": ["MPG-01", "MPG-01"],
            "BHP": [780.0, 820.0],
            "WtOilVol": [310.0, 290.0],
            "WtWaterVol": [100.0, 110.0],
            "WtTotalFluid": [410.0, 400.0],
            "WtDate": pd.to_datetime(["2026-01-01", "2026-02-01"]),
            "fgor": [250.0, 240.0],
            "whp": [210.0, 215.0],
        }
    )
    fig = hi._ipr_grid_fig(pad_df, ipr_rows, test_df)
    # IPR curve + operating point + scenario point + test-points = 4 traces
    assert len(fig.data) == 4
    test_trace = [t for t in fig.data if getattr(t, "name", "") == "MPG-01 tests"]
    assert test_trace and "Oil: 310 BOPD" in test_trace[0].text[0]


# ── non-JP contribution to the response curve ────────────────────────────────


def test_add_nonjp_curve_populates_for_responsive():
    ipr_rows = {"MPL-03": {"res_pres": 1800.0, "qwf": 200.0, "pwf": 700.0, "form_wc": 0.0}}
    rrow = {"Pad": "L", "Emp dBHP/dWHP": 0.8, "Emp class": "responsive"}
    curve: dict = {}
    hi._add_nonjp_curve("MPL-03", rrow, ipr_rows, curve, (-100, -50, 0, 50))
    assert "MPL-03" in curve
    assert curve["MPL-03"]["pad"] == "L"
    assert len(curve["MPL-03"]["oil"]) == 4
    # a header drop (negative Δ) lowers BHP → more oil than the +50 point
    assert curve["MPL-03"]["oil"][0] > curve["MPL-03"]["oil"][-1]


def test_add_nonjp_curve_skips_unresponsive():
    ipr_rows = {"MPL-03": {"res_pres": 1800.0, "qwf": 200.0, "pwf": 700.0, "form_wc": 0.0}}
    rrow = {"Pad": "L", "Emp dBHP/dWHP": 0.1, "Emp class": "slugging"}
    curve: dict = {}
    hi._add_nonjp_curve("MPL-03", rrow, ipr_rows, curve, (-50, 0, 50))
    assert curve == {}


# ── PDF report (matplotlib, headless) ────────────────────────────────────────


def _report_results_df():
    return pd.DataFrame(
        {
            "Well": ["MPG-01", "MPG-02"],
            "Pad": ["G", "G"],
            "Lift": ["JP", "JP"],
            "Verdict": ["responsive ✓", "sonic-decoupled"],
            "Sonic now": [False, True],
            "Oil now (BOPD)": [100.0, 200.0],
            "Oil scen (BOPD)": [140.0, 200.0],
            "BHP now (psi)": [800.0, 600.0],
            "BHP scen (psi)": [740.0, 600.0],
            "ΔOil (BOPD)": [40.0, 0.0],
            "Emp ΔOil (BOPD)": [35.0, np.nan],
            "Chosen ΔOil (BOPD)": [40.0, 0.0],
            "Method used": ["physics", "physics"],
            "Emp class": ["responsive", "no data"],
        }
    )


def test_build_report_pdf_returns_valid_pdf_bytes():
    from woffl.gui.scotts_tools import header_report as hr

    ipr_rows = {
        "MPG-01": {"res_pres": 1800.0, "qwf": 1000.0, "pwf": 800.0, "form_wc": 0.5},
        "MPG-02": {"res_pres": 1700.0, "qwf": 1200.0, "pwf": 600.0, "form_wc": 0.4},
    }
    curve = {"deltas": [-50, 0, 50], "wells": {"MPG-01": {"pad": "G", "oil": [110, 100, 95]}}}
    pdf = hr.build_report_pdf(
        _report_results_df(), -50, ipr_rows=ipr_rows, test_oil={"MPG-01": 95.0}, curve=curve
    )
    assert pdf[:4] == b"%PDF"
    assert len(pdf) > 1500


def test_build_report_pdf_empty_still_valid():
    from woffl.gui.scotts_tools import header_report as hr

    pdf = hr.build_report_pdf(pd.DataFrame(), -50)
    assert pdf[:4] == b"%PDF"


def _synthetic_well_dfs_fits():
    from woffl.gui.scotts_tools import header_trend as ht

    idx = pd.date_range("2026-01-01", periods=48, freq="h")
    wdf = pd.DataFrame(
        {
            "WHP": np.linspace(200, 260, 48),
            "HeaderP": np.linspace(180, 210, 48),
            "BHP": np.linspace(520, 560, 48),
        },
        index=idx,
    )
    empty_daily = pd.DataFrame(
        columns=["day", "slope", "intercept", "r2", "n", "x_min", "x_max", "good"]
    )
    fit = ht.WithinDayFit("BHP", "WHP", 0.7, 0.7, 0.05, 30, 20, 0.85, empty_daily)
    return {"MPG-01": wdf}, {"MPG-01": {"BHP~WHP": fit}}


def test_build_report_pdf_with_correlation_grid_matplotlib():
    from woffl.gui.scotts_tools import header_report as hr

    well_dfs, fits = _synthetic_well_dfs_fits()
    pdf = hr.build_report_pdf(_report_results_df(), -50, well_dfs=well_dfs, fits=fits)
    assert pdf[:4] == b"%PDF"
    assert len(pdf) > 1500


def test_corr_grid_and_well_fit_mpl_render_png_no_kaleido():
    from woffl.gui.scotts_tools import header_report as hr

    well_dfs, fits = _synthetic_well_dfs_fits()
    cg = hr.corr_grid_mpl(well_dfs, fits, ["MPG-01"], "WHP")
    assert hr.fig_to_png_bytes(cg)[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic
    wf = hr.well_fit_mpl("MPG-01", well_dfs["MPG-01"], fits["MPG-01"])
    assert hr.fig_to_png_bytes(wf)[:8] == b"\x89PNG\r\n\x1a\n"


def test_build_per_well_pdf_one_page_each():
    from woffl.gui.scotts_tools import header_report as hr

    well_dfs, fits = _synthetic_well_dfs_fits()
    ipr_rows = {
        "MPG-01": {"res_pres": 1800.0, "qwf": 1000.0, "pwf": 800.0, "form_wc": 0.5},
        "MPG-02": {"res_pres": 1700.0, "qwf": 1200.0, "pwf": 600.0, "form_wc": 0.4},
    }
    # MPG-02 has no trend → its correlation panel shows the "no data" note, not a crash.
    pdf = hr.build_per_well_pdf(
        _report_results_df(), ipr_rows=ipr_rows, well_dfs=well_dfs, fits=fits
    )
    assert pdf[:4] == b"%PDF"
    assert len(pdf) > 1500


def test_build_per_well_pdf_empty_still_valid():
    from woffl.gui.scotts_tools import header_report as hr

    assert hr.build_per_well_pdf(pd.DataFrame())[:4] == b"%PDF"
