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
