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
        donor_slope=0.8,
        donor_pwf=540.0,
        donor_res_pres=1800.0,
        target_qoil=300.0,
        target_res_pres=1800.0,
        delta_p=-50.0,
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


# ── P1-21 / P1-22 / P1-25 wiring (header_engine helpers → header_impact) ─────
#
# header_engine.solver_error_note / clamp_scenario_whp / physics_slope /
# recent_bhp_anchor were written but had zero callers until this wiring. These
# tests exercise the actual call sites in header_impact.py (_solve_at_whp,
# _solve_nonjp_row, and the main run loop's clamp→slope→verdict composition),
# not just the pure helpers in isolation (see test_header_engine.py for those).


def test_helpers_are_reexported_from_header_engine_not_redefined():
    # header_impact must import the SAME function objects header_engine
    # defines, not a local re-implementation that could silently drift.
    from woffl.gui.scotts_tools import header_engine as he

    assert hi.clamp_scenario_whp is he.clamp_scenario_whp
    assert hi.physics_slope is he.physics_slope
    assert hi.solver_error_note is he.solver_error_note
    assert hi.recent_bhp_anchor is he.recent_bhp_anchor


class _FakeBatch:
    """Stand-in for BatchPump — returns a pre-built per-row result frame so
    _solve_at_whp's handling of the "error" column can be tested without a
    real jet-pump physics solve."""

    def __init__(self, df, **_kwargs):
        self._df = df

    def batch_run(self, jetpumps):
        return self._df


def _fake_wc(**overrides):
    from woffl.assembly.network_optimizer import WellConfig

    kwargs = dict(
        well_name="MPTEST-01", res_pres=1800.0, form_temp=120.0, jpump_tvd=6000.0
    )
    kwargs.update(overrides)
    return WellConfig(**kwargs)


def test_solve_at_whp_surfaces_batch_run_error_p1_21(monkeypatch):
    # P1-21: a failed solve (BatchPump's per-row "error", not "na") must be
    # readable off _solve_at_whp's return so the caller can pass it to _verdict.
    df = pd.DataFrame(
        [
            {
                "nozzle": "12",
                "throat": "B",
                "sonic_status": False,
                "mach_te": np.nan,
                "psu_solv": np.nan,
                "qoil_std": np.nan,
                "form_wat": np.nan,
                "lift_wat": np.nan,
                "totl_wat": np.nan,
                "form_wor": np.nan,
                "totl_wor": np.nan,
                "error": "ConvergenceError('no zero crossing')",
            }
        ]
    )
    monkeypatch.setattr(hi, "BatchPump", lambda **kw: _FakeBatch(df))
    wc = _fake_wc()
    result = hi._solve_at_whp(
        wc, (None, None, None, None, None), "12", "B", 250.0, 3000.0
    )
    assert result["error"] == "ConvergenceError('no zero crossing')"
    assert np.isnan(result["oil"]) and np.isnan(result["psu"])


def test_solve_at_whp_converged_error_is_na(monkeypatch):
    df = pd.DataFrame(
        [
            {
                "nozzle": "12",
                "throat": "B",
                "sonic_status": False,
                "mach_te": 0.3,
                "psu_solv": 520.0,
                "qoil_std": 110.0,
                "form_wat": 60.0,
                "lift_wat": 400.0,
                "totl_wat": 460.0,
                "form_wor": 0.55,
                "totl_wor": 4.18,
                "error": "na",
            }
        ]
    )
    monkeypatch.setattr(hi, "BatchPump", lambda **kw: _FakeBatch(df))
    wc = _fake_wc()
    result = hi._solve_at_whp(
        wc, (None, None, None, None, None), "12", "B", 250.0, 3000.0
    )
    assert result["error"] == "na"
    assert result["oil"] == pytest.approx(110.0)
    # solver_error_note treats "na" as converged — no false "model failed".
    assert hi.solver_error_note(result["error"], "na") is None


def test_solve_at_whp_empty_batch_result_flagged_as_error(monkeypatch):
    monkeypatch.setattr(hi, "BatchPump", lambda **kw: _FakeBatch(pd.DataFrame()))
    wc = _fake_wc()
    result = hi._solve_at_whp(
        wc, (None, None, None, None, None), "12", "B", 250.0, 3000.0
    )
    assert result["error"] == "empty batch result"
    assert hi.solver_error_note(result["error"], "na") == "empty batch result"


def test_solve_at_whp_error_drives_model_failed_verdict():
    # Mirrors the main run loop's composition: solver_error_note(res_now.error,
    # res_scen.error) feeds _verdict(..., error=...) so a non-converging solve
    # renders "model failed", never a confident sonic/no-response misreading.
    res_now = dict(
        oil=np.nan,
        pf_rate=np.nan,
        psu=np.nan,
        sonic=False,
        mach=np.nan,
        error="ConvergenceError('no zero crossing')",
    )
    res_scen = dict(
        oil=np.nan, pf_rate=np.nan, psu=np.nan, sonic=False, mach=np.nan, error="na"
    )
    err = hi.solver_error_note(res_now["error"], res_scen["error"])
    delta_oil = res_scen["oil"] - res_now["oil"]  # NaN — the pre-fix "no response" trap
    verdict = hi._verdict(
        res_now["sonic"], res_scen["sonic"], delta_oil, None, True, error=err
    )
    assert verdict.startswith(hi.MODEL_FAILED_PREFIX)
    assert "ConvergenceError" in verdict


def test_scenario_whp_clamp_feeds_the_same_value_to_slope_p1_22():
    # Mirrors render_tab()'s main solve loop: the DISPLAYED "WHP scen" and the
    # divisor in the physics-slope math must be the SAME clamped value — never
    # the raw (unclamped) whp_now + delta_p.
    whp_now, delta_p = 100.0, -500.0
    whp_scen, clamped = hi.clamp_scenario_whp(whp_now, delta_p, floor=hi._WHP_FLOOR)
    assert clamped is True
    assert whp_scen == pytest.approx(
        hi._WHP_FLOOR
    )  # what the UI displays as "WHP scen"

    psu_now, psu_scen = 500.0, 420.0
    slope = hi.physics_slope(psu_now, psu_scen, whp_now, whp_scen)
    assert slope == pytest.approx((psu_scen - psu_now) / (whp_scen - whp_now))
    # The pre-fix bug divided by the raw requested delta_p instead — confirm
    # that gives a materially different (understated) slope.
    wrong = (psu_scen - psu_now) / delta_p
    assert abs(slope) > abs(wrong)


def test_scenario_whp_unclamped_matches_legacy_behavior():
    # When the scenario WHP doesn't hit the floor, the clamp is a no-op and the
    # slope reduces to the old (psu_scen - psu_now) / delta_p.
    whp_now, delta_p = 250.0, -50.0
    whp_scen, clamped = hi.clamp_scenario_whp(whp_now, delta_p, floor=hi._WHP_FLOOR)
    assert clamped is False
    psu_now, psu_scen = 500.0, 460.0
    slope = hi.physics_slope(psu_now, psu_scen, whp_now, whp_scen)
    assert slope == pytest.approx((psu_scen - psu_now) / delta_p)


def test_solve_nonjp_row_bhp_anchor_ignores_tail_sentinel_p1_25():
    # P1-25: the last raw historian reading is a dead-gauge sentinel (12 psi);
    # the prior 23 readings are a clean plateau at 500. The OLD ad-hoc pick
    # (bhp_s.iloc[-1]) would have anchored on 12; recent_bhp_anchor must not.
    trend = pd.DataFrame({"BHP": [500.0] * 23 + [12.0]})
    row = pd.Series(
        {
            "Pad": "L",
            "Lift": "ESP",
            "Formation": "Schrader",
            "Oil (BOPD)": 200.0,
            "ResP (psi)": 1800.0,
            "WHP now (psi)": 250.0,
        }
    )
    out = hi._solve_nonjp_row(
        "MPL-01",
        row,
        emp_fits={},
        emp_well_dfs={"MPL-01": trend},
        delta_p=-50.0,
    )
    assert out["BHP now (psi)"] == pytest.approx(500.0)
    assert "unscreened" not in out["IPR src"]


def test_solve_nonjp_row_bhp_anchor_flags_unscreened_fallback():
    # When NOTHING passes the screen, recent_bhp_anchor still returns a value
    # (median of the raw tail) but flags it — _solve_nonjp_row must surface
    # that as a suspect-anchor note rather than presenting it as clean.
    trend = pd.DataFrame({"BHP": [10.0, 20.0, 5.0, 15.0]})
    row = pd.Series(
        {
            "Pad": "L",
            "Lift": "ESP",
            "Formation": "Schrader",
            "Oil (BOPD)": 200.0,
            "ResP (psi)": 1800.0,
            "WHP now (psi)": 250.0,
        }
    )
    out = hi._solve_nonjp_row(
        "MPL-01",
        row,
        emp_fits={},
        emp_well_dfs={"MPL-01": trend},
        delta_p=-50.0,
    )
    assert out["BHP now (psi)"] == pytest.approx(np.median([10.0, 20.0, 5.0, 15.0]))
    assert "unscreened" in out["IPR src"]


def test_solve_nonjp_row_no_trend_still_gaugeless():
    # No historian trend at all → still the pre-existing "gaugeless" path
    # (recent_bhp_anchor(None) → (nan, False), same as before the wiring).
    row = pd.Series(
        {
            "Pad": "L",
            "Lift": "ESP",
            "Formation": "Schrader",
            "Oil (BOPD)": 200.0,
            "ResP (psi)": 1800.0,
            "WHP now (psi)": 250.0,
        }
    )
    out = hi._solve_nonjp_row(
        "MPL-02", row, emp_fits={}, emp_well_dfs={}, delta_p=-50.0
    )
    assert out["Verdict"] == "gaugeless — use Analog"
    assert np.isnan(out["BHP now (psi)"])


# ── _solve_nonjp_row: P1-23 (no-data/insufficient must not book a confident
#    ΔOil = 0 mislabeled "slugging") ──────────────────────────────────────────


def test_solve_nonjp_row_no_data_reflects_emp_class_and_excludes_uplift_p1_23():
    # A real BHP trend exists (bhp_now resolves, so this ISN'T the gaugeless
    # path) but there's no historian fit at all (emp_fits={}) → emp_class
    # falls to "no data". Pre-fix this silently became verdict "slugging"
    # with a confident emp_doil = 0.0 — a data-absence case masquerading as
    # "the well doesn't respond", inflating field uplift understatement.
    trend = pd.DataFrame({"BHP": [500.0] * 24})
    row = pd.Series(
        {
            "Pad": "L",
            "Lift": "ESP",
            "Formation": "Schrader",
            "Oil (BOPD)": 200.0,
            "ResP (psi)": 1800.0,
            "WHP now (psi)": 250.0,
        }
    )
    out = hi._solve_nonjp_row(
        "MPL-05",
        row,
        emp_fits={},
        emp_well_dfs={"MPL-05": trend},
        delta_p=-50.0,
    )
    assert out["Emp class"] == "no data"
    assert out["Verdict"] == "no data"  # NOT "slugging"
    assert np.isnan(out["Emp ΔOil (BOPD)"])
    assert np.isnan(out["Oil scen (BOPD)"])
    # The real aggregation path (_render_results → _chosen_method) must drop
    # this well out of the field-uplift sum rather than booking a 0: physics
    # ΔOil is NaN for non-JP rows, so with emp NaN too the chosen value is
    # NaN and pandas .sum() (skipna=True) excludes it.
    val, label = hi._chosen_method(float("nan"), out["Emp ΔOil (BOPD)"], "auto")
    assert np.isnan(val)
    assert label == "—"


def test_solve_nonjp_row_insufficient_reflects_emp_class_and_excludes_uplift_p1_23():
    # A fit exists but with too few fittable days (n_days=2 < min_fittable=5)
    # → classify_response returns "insufficient". Same bug: pre-fix this
    # collapsed to verdict "insufficient" via the ternary's happy accident,
    # but STILL booked a confident emp_doil = 0.0 that fed the uplift sum.
    fit = _make_fit(n_days=2, n_good_days=1)
    trend = pd.DataFrame({"BHP": [500.0] * 24})
    row = pd.Series(
        {
            "Pad": "L",
            "Lift": "ESP",
            "Formation": "Schrader",
            "Oil (BOPD)": 200.0,
            "ResP (psi)": 1800.0,
            "WHP now (psi)": 250.0,
        }
    )
    out = hi._solve_nonjp_row(
        "MPL-06",
        row,
        emp_fits={"MPL-06": {"BHP~WHP": fit}},
        emp_well_dfs={"MPL-06": trend},
        delta_p=-50.0,
    )
    assert out["Emp class"] == "insufficient"
    assert out["Verdict"] == "insufficient data"
    assert np.isnan(out["Emp ΔOil (BOPD)"])
    assert np.isnan(out["Oil scen (BOPD)"])


def test_solve_nonjp_row_slugging_still_books_confident_zero():
    # Regression guard: a GENUINE slugging diagnosis (enough fittable days,
    # but few clean/consistent ones) is a real physical answer, not a data
    # gap — unlike no-data/insufficient above, it must keep booking a
    # confident ΔOil = 0 and feed the uplift sum as a real zero.
    fit = _make_fit(n_days=30, n_good_days=2)
    trend = pd.DataFrame({"BHP": [500.0] * 24})
    row = pd.Series(
        {
            "Pad": "L",
            "Lift": "ESP",
            "Formation": "Schrader",
            "Oil (BOPD)": 200.0,
            "ResP (psi)": 1800.0,
            "WHP now (psi)": 250.0,
        }
    )
    out = hi._solve_nonjp_row(
        "MPL-07",
        row,
        emp_fits={"MPL-07": {"BHP~WHP": fit}},
        emp_well_dfs={"MPL-07": trend},
        delta_p=-50.0,
    )
    assert out["Emp class"] == "slugging"
    assert out["Verdict"] == "slugging"
    assert out["Emp ΔOil (BOPD)"] == 0.0
    assert out["Oil scen (BOPD)"] == pytest.approx(200.0)


# ── _classify_lift: P1-26 (recent ESP amps must terminate a stale JP
#    tracker record) ───────────────────────────────────────────────────────


def _jp_hist_row(well: str, date_set, nozzle="12", throat="B"):
    return pd.DataFrame(
        [
            {
                "Well Name": well,
                "Date Set": pd.Timestamp(date_set),
                "Nozzle Number": nozzle,
                "Throat Ratio": throat,
                "Tubing Diameter": 2.375,
            }
        ]
    )


def test_classify_lift_recent_esp_amps_overrides_stale_jp_install_p1_26():
    # The JP tracker's last install is years old; if the tracker never
    # recorded the pull (JPCOs are same-day pull+set per CLAUDE.md, and the
    # tracker is unreliable), a well long since converted to ESP would keep
    # modeling as JP forever without this terminator.
    jp_hist = _jp_hist_row("MPB-99", "2019-03-01")
    ov_row = pd.Series({"esp_amps": 42.0, "lift_gas": 0.0})
    assert hi._classify_lift("MPB-99", jp_hist, ov_row) == "ESP"


def test_classify_lift_current_jp_well_unaffected():
    # No ESP amps evidence → a real current JP well still classifies JP.
    jp_hist = _jp_hist_row("MPB-98", "2026-01-01")
    ov_row = pd.Series({"esp_amps": np.nan, "lift_gas": 0.0})
    assert hi._classify_lift("MPB-98", jp_hist, ov_row) == "JP"


def test_classify_lift_zero_esp_amps_does_not_override_jp():
    # esp_amps == 0 (present but not flowing current) must not be treated as
    # ESP evidence — only a strictly positive reading proves a running ESP.
    jp_hist = _jp_hist_row("MPB-97", "2020-06-15")
    ov_row = pd.Series({"esp_amps": 0.0, "lift_gas": 0.0})
    assert hi._classify_lift("MPB-97", jp_hist, ov_row) == "JP"


def test_classify_lift_no_jp_esp_amps_still_esp():
    # Pre-existing behavior: no JP install at all + esp_amps > 0 → ESP.
    jp_hist = pd.DataFrame(
        columns=["Well Name", "Date Set", "Nozzle Number", "Throat Ratio"]
    )
    ov_row = pd.Series({"esp_amps": 30.0, "lift_gas": 0.0})
    assert hi._classify_lift("MPB-96", jp_hist, ov_row) == "ESP"


def test_classify_lift_gas_lift_and_flowing_unaffected():
    jp_hist = pd.DataFrame(
        columns=["Well Name", "Date Set", "Nozzle Number", "Throat Ratio"]
    )
    gl_row = pd.Series({"esp_amps": np.nan, "lift_gas": 15.0})
    assert hi._classify_lift("MPB-95", jp_hist, gl_row) == "gas-lift"
    flow_row = pd.Series({"esp_amps": np.nan, "lift_gas": np.nan})
    assert hi._classify_lift("MPB-94", jp_hist, flow_row) == "flowing"


# ── header_trend.classify_response ────────────────────────────────────────────


def _make_fit(mean_slope=0.8, mean_r2=0.9, n_days=30, n_good_days=25):
    """Hand-build a WithinDayFit for classification tests."""
    return ht.WithinDayFit(
        y_name="BHP",
        x_name="WHP",
        mean_slope=mean_slope,
        median_slope=mean_slope,
        slope_std=0.05,
        n_days=n_days,
        n_good_days=n_good_days,
        mean_r2=mean_r2,
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
    agg = {
        "deltas": [-50, 0, 50],
        "pads": {"G": [10.0, 0.0, -5.0]},
        "ALL": [10.0, 0.0, -5.0],
    }
    fig = hi._response_curve_fig(agg)
    assert fig is not None
    names = [t.name for t in fig.data]
    assert "Pad G" in names and "Field (ALL)" in names


def test_response_curve_fig_single_pad_filter():
    agg = {
        "deltas": [-50, 0, 50],
        "pads": {"G": [10.0, 0.0, -5.0], "H": [1.0, 0.0, -1.0]},
        "ALL": [11.0, 0.0, -6.0],
    }
    fig = hi._response_curve_fig(agg, pads_to_show=["G"])
    names = [t.name for t in fig.data]
    assert "Pad G" in names and "Pad H" not in names and "Field (ALL)" not in names


def test_response_curve_fig_empty_returns_none():
    assert hi._response_curve_fig({}) is None


# ── non-JP contribution to the response curve ────────────────────────────────


def test_add_nonjp_curve_populates_for_responsive():
    ipr_rows = {
        "MPL-03": {"res_pres": 1800.0, "qwf": 200.0, "pwf": 700.0, "form_wc": 0.0}
    }
    rrow = {"Pad": "L", "Emp dBHP/dWHP": 0.8, "Emp class": "responsive"}
    curve: dict = {}
    hi._add_nonjp_curve("MPL-03", rrow, ipr_rows, curve, (-100, -50, 0, 50))
    assert "MPL-03" in curve
    assert curve["MPL-03"]["pad"] == "L"
    assert len(curve["MPL-03"]["oil"]) == 4
    # a header drop (negative Δ) lowers BHP → more oil than the +50 point
    assert curve["MPL-03"]["oil"][0] > curve["MPL-03"]["oil"][-1]


def test_add_nonjp_curve_skips_unresponsive():
    ipr_rows = {
        "MPL-03": {"res_pres": 1800.0, "qwf": 200.0, "pwf": 700.0, "form_wc": 0.0}
    }
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
    curve = {
        "deltas": [-50, 0, 50],
        "wells": {"MPG-01": {"pad": "G", "oil": [110, 100, 95]}},
    }
    pdf = hr.build_report_pdf(
        _report_results_df(),
        -50,
        ipr_rows=ipr_rows,
        test_oil={"MPG-01": 95.0},
        curve=curve,
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
