"""Tests for woffl.gui.scotts_tools.header_engine pure helpers.

Per-pad power-fluid pressure resolution + the moved ΔOil picker / verdict.
No Streamlit or Databricks runtime needed.

Run with:
    python -m pytest tests/test_header_engine.py
"""

import numpy as np
import pandas as pd
import pytest

from woffl.gui.scotts_tools import header_engine as he


# ── resolve_pad_pf (per-pad PF, distinct G/H/I/J) ────────────────────────────

_DEFAULTS = {"G": 2200, "H": 3400, "I": 3400, "J": 2200}


def _default_fn(pad):
    return _DEFAULTS.get(pad, 3400)


def test_resolve_pad_pf_seeds_from_defaults_when_no_stored():
    out = he.resolve_pad_pf(["G", "H", "I", "J"], None, _default_fn)
    assert out == {"G": 2200, "H": 3400, "I": 3400, "J": 2200}


def test_resolve_pad_pf_preserves_stored_edits():
    # The whole point of v2: let the user distinguish H from I.
    stored = {"H": 3600, "I": 3100}
    out = he.resolve_pad_pf(["G", "H", "I"], stored, _default_fn)
    assert out["H"] == 3600
    assert out["I"] == 3100
    assert out["G"] == 2200  # newly selected pad seeded from default


def test_resolve_pad_pf_drops_deselected_pads():
    stored = {"G": 2200, "H": 3400, "I": 3400}
    out = he.resolve_pad_pf(["G"], stored, _default_fn)
    assert out == {"G": 2200}


def test_resolve_pad_pf_coerces_to_int():
    out = he.resolve_pad_pf(["G"], {"G": 2250.7}, _default_fn)
    assert out["G"] == 2251
    assert isinstance(out["G"], int)


def test_resolve_pad_pf_bad_value_falls_back_to_default():
    out = he.resolve_pad_pf(["G"], {"G": "oops"}, _default_fn)
    assert out["G"] == 2200


def test_resolve_pad_pf_empty_pads_returns_empty():
    assert he.resolve_pad_pf([], {"G": 2200}, _default_fn) == {}


# ── pf_map_from_selected (per-well PF for the gaugeless back-calc) ────────────


def test_pf_map_from_selected_picks_up_jp_pf_skips_nonjp():
    df = pd.DataFrame(
        {
            "Well": ["MPG-01", "MPG-02", "MPH-05"],
            "Lift": ["JP", "ESP", "JP"],
            "PF held (psi)": [2200, None, 3400],
        }
    )
    out = he.pf_map_from_selected(df)
    assert out == {"MPG-01": 2200.0, "MPH-05": 3400.0}  # ESP (None PF) skipped


def test_pf_map_from_selected_handles_missing_column():
    df = pd.DataFrame({"Well": ["MPG-01"], "Lift": ["JP"]})
    assert he.pf_map_from_selected(df) == {}


def test_pf_map_from_selected_none_returns_empty():
    assert he.pf_map_from_selected(None) == {}


# ── moved helpers still resolve from the engine module ───────────────────────


def test_chosen_method_importable_from_engine():
    val, lab = he._chosen_method(100.0, 20.0, "auto")
    assert val == 20.0
    assert lab == "empirical (override)"


def test_verdict_importable_from_engine():
    assert he._verdict(True, True, 0.0, "responsive", True) == "sonic-decoupled"


# ── sensitivity / sense-check / bias / curve (v2 / M2) ───────────────────────


def _results_df():
    """Synthetic header-impact results: pad G (1 responsive + 1 sonic), pad H
    (1 responsive-physics + 1 no-response ESP)."""
    return pd.DataFrame(
        {
            "Well": ["MPG-01", "MPG-02", "MPH-01", "MPH-02"],
            "Pad": ["G", "G", "H", "H"],
            "Lift": ["JP", "JP", "JP", "ESP"],
            "Verdict": ["responsive ✓", "sonic-decoupled", "responsive (physics)", "no response"],
            "Sonic now": [False, True, False, False],
            "Oil now (BOPD)": [100.0, 200.0, 150.0, 80.0],
            "ΔOil (BOPD)": [50.0, 0.0, 30.0, np.nan],
            "Emp ΔOil (BOPD)": [35.0, np.nan, np.nan, 10.0],
            "Chosen ΔOil (BOPD)": [50.0, 0.0, 30.0, 10.0],
            "Method used": ["physics", "physics", "physics", "empirical"],
            "Emp class": ["responsive", "no data", "no data", "responsive"],
        }
    )


def test_verdict_bucket():
    assert he.verdict_bucket("responsive ✓") == "responsive"
    assert he.verdict_bucket("sonic-decoupled") == "sonic"
    assert he.verdict_bucket("responsive (physics)", sonic_now=True) == "sonic"  # flag wins
    assert he.verdict_bucket("no response") == "no-response"
    assert he.verdict_bucket("disagree — check") == "other"


def test_summarize_sensitivity_per_pad_and_overall():
    per_pad, overall = he.summarize_sensitivity(_results_df(), delta_p=-50)
    g = per_pad[per_pad["Pad"] == "G"].iloc[0]
    assert g["Wells"] == 2
    assert g["Responsive"] == 1 and g["Sonic"] == 1 and g["No-response"] == 0
    assert g["Oil now (BOPD)"] == 300.0
    assert g["ΔOil (BOPD)"] == 50.0
    assert g["BOPD per 100 psi"] == pytest.approx(100.0)  # 50 BOPD over 50 psi → 100/100psi
    h = per_pad[per_pad["Pad"] == "H"].iloc[0]
    assert h["Responsive"] == 1 and h["No-response"] == 1
    assert h["ΔOil (BOPD)"] == 40.0
    assert overall["Wells"] == 4
    assert overall["ΔOil (BOPD)"] == 90.0
    assert overall["BOPD per 100 psi"] == pytest.approx(180.0)


def test_summarize_sensitivity_zero_delta_nan_norm():
    _, overall = he.summarize_sensitivity(_results_df(), delta_p=0)
    assert np.isnan(overall["BOPD per 100 psi"])


def test_summarize_sensitivity_empty():
    per_pad, overall = he.summarize_sensitivity(pd.DataFrame(), delta_p=-50)
    assert per_pad.empty and overall == {}


def test_sense_check_table_residual():
    out = he.sense_check_table(_results_df(), {"MPG-01": 95.0, "MPG-02": 210.0})
    r = out[out["Well"] == "MPG-01"].iloc[0]
    assert r["Oil now modeled (BOPD)"] == 100.0
    assert r["Oil now measured (BOPD)"] == 95.0
    assert r["Oil residual (BOPD)"] == pytest.approx(5.0)
    assert r["Oil residual %"] == pytest.approx(5.0 / 95.0 * 100.0)
    # No measured oil → residual % is NaN, not a divide error.
    r3 = out[out["Well"] == "MPH-01"].iloc[0]
    assert np.isnan(r3["Oil residual %"])


def test_bias_by_pad_flags_optimistic():
    out = he.bias_by_pad(_results_df())
    # Only MPG-01 qualifies (responsive, both ΔOils present, phys≠0): 35/50 = 0.7.
    g = out[out["Pad"] == "G"].iloc[0]
    assert g["Emp/Phys bias"] == pytest.approx(0.7)
    assert g["Flag"] == "physics optimistic (<0.8)"
    assert (out["Pad"] == "ALL").any()


def test_bias_by_pad_empty_when_none_qualify():
    df = _results_df()
    df["Emp ΔOil (BOPD)"] = np.nan  # nothing to compare
    assert he.bias_by_pad(df).empty


def test_aggregate_response_curve_relative_to_zero():
    curve = {
        "deltas": [-100, -50, 0, 50],
        "wells": {
            "MPG-01": {"pad": "G", "oil": [120, 110, 100, 95]},
            "MPG-02": {"pad": "G", "oil": [205, 202, 200, 199]},
            "MPH-01": {"pad": "H", "oil": [160, 155, 150, 148]},
        },
    }
    res = he.aggregate_response_curve(curve)
    assert res["deltas"] == [-100, -50, 0, 50]
    assert res["pads"]["G"] == pytest.approx([25.0, 12.0, 0.0, -6.0])
    assert res["pads"]["H"] == pytest.approx([10.0, 5.0, 0.0, -2.0])
    assert res["ALL"] == pytest.approx([35.0, 17.0, 0.0, -8.0])


def test_aggregate_response_curve_skips_mismatched_length():
    curve = {
        "deltas": [-50, 0, 50],
        "wells": {"A": {"pad": "G", "oil": [10, 8]}},  # wrong length → skipped
    }
    res = he.aggregate_response_curve(curve)
    assert res["pads"] == {} and res["ALL"] == [0.0, 0.0, 0.0]


def test_aggregate_response_curve_empty():
    assert he.aggregate_response_curve(None) == {}
    assert he.aggregate_response_curve({}) == {}


# ── like-wells donor assignment (G3) ─────────────────────────────────────────

_META = {
    "MPG-01": {"pad": "G", "lift": "JP", "formation": "Schrader"},
    "MPG-02": {"pad": "G", "lift": "JP", "formation": "Schrader"},
    "MPG-03": {"pad": "G", "lift": "ESP", "formation": "Schrader"},
    "MPH-01": {"pad": "H", "lift": "JP", "formation": "Kuparuk"},
}


def test_donor_tokens_shape():
    toks = he.donor_tokens(["MPH-01", "MPG-01"])
    assert toks[:5] == [
        he.OWN_TOKEN, he.GROUP_PADLIFT, he.GROUP_PADFORMATION,
        he.GROUP_FORMATION, he.GROUP_LIFT,
    ]
    assert toks[5:] == ["MPG-01", "MPH-01"]  # wells sorted after the tokens


def test_donor_member_wells_own_is_none():
    assert he.donor_member_wells("MPG-01", he.OWN_TOKEN, _META) is None
    assert he.donor_member_wells("MPG-01", "", _META) is None


def test_donor_member_wells_group_padlift():
    m = he.donor_member_wells("MPG-01", he.GROUP_PADLIFT, _META)
    assert sorted(m) == ["MPG-01", "MPG-02"]  # G + JP only (not the G ESP, not H)


def test_donor_member_wells_group_formation():
    m = he.donor_member_wells("MPG-01", he.GROUP_FORMATION, _META)
    assert sorted(m) == ["MPG-01", "MPG-02", "MPG-03"]  # all Schrader, regardless of lift


def test_donor_member_wells_group_padformation():
    # Same pad AND formation — the default for wells with no own IPR test data.
    m = he.donor_member_wells("MPG-01", he.GROUP_PADFORMATION, _META)
    assert sorted(m) == ["MPG-01", "MPG-02", "MPG-03"]  # pad G + Schrader (any lift)


def test_donor_member_wells_group_lift():
    # All like-lift wells field-wide — the default for wells with no usable
    # WHP→BHP correlation (the widest analog pool, crosses pads).
    m = he.donor_member_wells("MPG-01", he.GROUP_LIFT, _META)
    assert sorted(m) == ["MPG-01", "MPG-02", "MPH-01"]  # all JP, across pads G + H


def test_donor_member_wells_specific_and_unknown():
    assert he.donor_member_wells("MPG-01", "MPH-01", _META) == ["MPH-01"]
    assert he.donor_member_wells("MPG-01", "MPX-99", _META) is None  # not selected


def test_average_vogel_rows_means():
    out = he.average_vogel_rows(
        [
            {"ResP": 1800, "qwf": 1000, "pwf": 800, "form_wc": 0.5, "fgor": 250},
            {"ResP": 1600, "qwf": 800, "pwf": 600, "form_wc": 0.3, "fgor": 150},
        ]
    )
    assert out["ResP"] == pytest.approx(1700)
    assert out["qwf"] == pytest.approx(900)
    assert out["pwf"] == pytest.approx(700)
    assert out["form_wc"] == pytest.approx(0.4)
    assert out["fgor"] == pytest.approx(200)


def test_average_vogel_rows_requires_core_keys():
    assert he.average_vogel_rows([{"form_wc": 0.5}]) is None  # no ResP/qwf/pwf
    assert he.average_vogel_rows([]) is None
    assert he.average_vogel_rows(None) is None


def test_average_slope():
    assert he.average_slope([0.8, 0.6, None, np.nan]) == pytest.approx(0.7)
    assert he.average_slope([]) is None


def test_describe_donor():
    assert he.describe_donor(he.OWN_TOKEN) == "own"
    assert "pad+lift" in he.describe_donor(he.GROUP_PADLIFT, 3)
    assert "pad+formation" in he.describe_donor(he.GROUP_PADFORMATION, 3)
    assert "lift" in he.describe_donor(he.GROUP_LIFT, 5)
    assert he.describe_donor("MPG-02") == "donor MPG-02"


def test_corr_display_plan():
    df = pd.DataFrame(
        {
            "Well": ["A", "B", "C", "D"],
            "Corr donor": ["own", "donor B", "group pad+lift (n=3)", "own"],
            "Emp dBHP/dWHP": [0.8, np.nan, 0.6, 0.5],
        }
    )
    plan = he.corr_display_plan(df, available_wells=["A", "B"])
    assert plan["A"]["source"] == "A" and plan["A"]["note"] == ""        # own + has trend
    assert plan["B"]["source"] == "B" and "using B corr" in plan["B"]["note"]  # donor B present
    assert plan["C"]["source"] is None and "group" in plan["C"]["note"]  # group avg → note only
    assert plan["D"]["source"] is None                                   # own but no trend


# ── model-vs-observed response sense check ───────────────────────────────────


def test_slope_agreement_confirmed():
    # obs within the relative band of the model → confirmed
    label, ratio = he.slope_agreement(0.80, 0.82)
    assert label == "confirmed ✓"
    assert ratio == pytest.approx(0.82 / 0.80)


def test_slope_agreement_over_predicts():
    # model expects a big move, field stayed flatter → model over-predicts
    label, ratio = he.slope_agreement(0.80, 0.20)
    assert label == "model over-predicts"
    assert ratio == pytest.approx(0.25)


def test_slope_agreement_under_predicts():
    label, _ = he.slope_agreement(0.40, 0.90)
    assert label == "model under-predicts"


def test_slope_agreement_abs_tol_near_zero():
    # both near zero, within the absolute band → confirmed even though ratio is wild
    label, _ = he.slope_agreement(0.05, 0.12, rel_tol=0.30, abs_tol=0.15)
    assert label == "confirmed ✓"


def test_slope_agreement_missing():
    assert he.slope_agreement(np.nan, np.nan)[0] == "no data"
    assert he.slope_agreement(0.8, np.nan)[0] == "no observed slope"
    assert he.slope_agreement(np.nan, 0.8)[0] == "empirical only"


def test_sense_check_response_sonic_and_slope():
    df = pd.DataFrame(
        {
            "Well": ["A", "B", "C"],
            "Pad": ["G", "G", "G"],
            "Lift": ["JP", "JP", "JP"],
            "WHP now (psi)": [250, 240, 230],
            "BHP now (psi)": [500, 480, 460],
            "Phys dBHP/dWHP": [0.80, 0.70, 0.60],
            "Emp dBHP/dWHP": [0.82, 0.10, np.nan],
            "Emp class": ["responsive", "slugging", "no data"],
            "Sonic now": [False, True, False],
        }
    )
    out = he.sense_check_response(df).set_index("Well")
    assert out.loc["A", "Sense-check"] == "confirmed ✓"        # slopes agree
    assert out.loc["B", "Sense-check"] == "sonic-decoupled"     # sonic → reported separately
    assert "history flat — agrees" in out.loc["B", "Note"]      # slugging history agrees w/ choke
    assert out.loc["C", "Sense-check"] == "no observed slope"   # no history to confirm


def test_pad_updown_lever_asymmetry():
    # G-pad: dropping the header gains oil; raising it loses oil (and a 2nd well
    # adds in on the down side only). deltas symmetric around 0.
    curve = {
        "deltas": [-100, 0, 100],
        "wells": {
            "A": {"pad": "G", "oil": [1120.0, 1000.0, 910.0]},
            "B": {"pad": "G", "oil": [560.0, 500.0, 470.0]},
            "H1": {"pad": "H", "oil": [300.0, 300.0, 300.0]},
        },
    }
    down, up = he.pad_updown_lever(curve, "G", ref=100)
    assert down == pytest.approx((1120 + 560) - (1000 + 500))   # +180 going down
    assert up == pytest.approx((910 + 470) - (1000 + 500))      # -120 going up


def test_pad_updown_lever_missing_points():
    # ref=100 not in the swept deltas → (nan, nan)
    curve = {"deltas": [-50, 0, 50], "wells": {"A": {"pad": "G", "oil": [1, 2, 3]}}}
    down, up = he.pad_updown_lever(curve, "G", ref=100)
    assert np.isnan(down) and np.isnan(up)


def test_pad_updown_lever_empty():
    for arg in (None, {}):
        d, u = he.pad_updown_lever(arg, "G")
        assert np.isnan(d) and np.isnan(u)


# ── long-horizon back-test ───────────────────────────────────────────────────


def _tests_frame():
    # 6 monthly tests where the header was raised over the window: WHP & BHP climb,
    # liquid falls (textbook back-pressure response).
    return pd.DataFrame(
        {
            "WtDate": pd.to_datetime(
                ["2025-06-01", "2025-07-01", "2025-08-01",
                 "2026-03-01", "2026-04-01", "2026-05-01"]
            ),
            "whp": [180, 185, 190, 280, 285, 290],
            "BHP": [450, 455, 460, 540, 545, 550],
            "WtOilVol": [900, 890, 895, 760, 755, 750],
            "WtWaterVol": [100, 100, 100, 100, 100, 100],
            "WtTotalFluid": [1000, 990, 995, 860, 855, 850],
        }
    )


def test_backtest_anchors_medians_and_deltas():
    a = he.backtest_anchors(_tests_frame(), frac=0.34)  # k = round(6*.34)=2
    assert a["k"] == 2 and a["n_tests"] == 6
    assert a["whp_then"] == pytest.approx(182.5)   # median(180,185)
    assert a["whp_now"] == pytest.approx(287.5)    # median(285,290)
    assert a["d_whp"] == pytest.approx(105.0)
    assert a["bhp_then"] == pytest.approx(452.5) and a["bhp_now"] == pytest.approx(547.5)
    assert a["d_bhp"] == pytest.approx(95.0)
    assert a["liquid_then"] == pytest.approx(995.0) and a["liquid_now"] == pytest.approx(852.5)
    assert a["d_liquid"] == pytest.approx(-142.5)   # liquid fell as header rose


def test_backtest_anchors_too_few():
    one = _tests_frame().iloc[:1]
    assert he.backtest_anchors(one) == {}


def test_backtest_anchors_liquid_falls_back_to_oil():
    df = _tests_frame().drop(columns=["WtTotalFluid"])
    a = he.backtest_anchors(df, frac=0.34)
    assert a["liquid_then"] == pytest.approx(895.0)  # = median oil (890,900)


def test_predict_dbhp_from_curve_interp():
    cw = {"whp": [150, 200, 250, 300, 350], "bhp": [400, 450, 500, 550, 600]}  # slope 1.0
    dbhp, extr = he.predict_dbhp_from_curve(cw, whp_then=182.5, whp_now=287.5)
    assert dbhp == pytest.approx(105.0)  # slope 1.0 over a 105-psi move
    assert extr is False


def test_predict_dbhp_from_curve_extrapolated_flag():
    cw = {"whp": [200, 250, 300], "bhp": [500, 550, 600]}
    _, extr = he.predict_dbhp_from_curve(cw, whp_then=150, whp_now=290)  # 150 < 200
    assert extr is True


def test_predict_dbhp_from_curve_empty():
    assert he.predict_dbhp_from_curve(None, 1, 2) == (pytest.approx(np.nan, nan_ok=True), False)
    assert np.isnan(he.predict_dbhp_from_curve({"whp": [1], "bhp": [2]}, 1, 2)[0])


def test_backpressure_consistency_classes():
    # WHP↑ BHP↑ liquid↓ → textbook
    assert he.backpressure_consistency(105, 95, -142) == "back-pressure-consistent ✓"
    # liquid fell but BHP flat → depletion-like (IPR shifted, not the operating point)
    assert he.backpressure_consistency(105, 3, -150) == "depletion-like / BHP flat"
    # BHP moved opposite the header → contradicts
    assert he.backpressure_consistency(105, -90, -50) == "contradicts (BHP vs WHP)"
    # header barely moved → nothing to test
    assert he.backpressure_consistency(5, 50, -50) == "header ~flat (n/a)"
    # gaugeless but liquid opposes the header → consistent on liquid alone
    assert he.backpressure_consistency(105, np.nan, -150) == "liquid consistent (no BHP)"
    # BHP tracked WHP but liquid rose (wrong way)
    assert he.backpressure_consistency(105, 90, 150) == "BHP tracks WHP; liquid wrong way"


# ── windowed Vogel pseudo-pr fits (Standing back-out) ────────────────────────


def _vogel_q(pwf, pr, qmax):
    r = np.asarray(pwf, float) / pr
    return qmax * (1.0 - 0.2 * r - 0.8 * r * r)


def test_fit_vogel_ipr_recovers_known_pr():
    # Generate clean Vogel points with a KNOWN pr & qmax over a good pwf spread,
    # then confirm the fitter backs them out.
    pr_true, qmax_true = 1800.0, 1000.0
    pwf = [300, 500, 700, 900, 1100, 1300, 1500]
    q = _vogel_q(pwf, pr_true, qmax_true)
    fit = he.fit_vogel_ipr(pwf, q)
    assert fit is not None
    assert fit["pr"] == pytest.approx(pr_true, abs=40)      # back out p_r within ~40 psi
    assert fit["qmax"] == pytest.approx(qmax_true, rel=0.03)
    assert fit["rmse"] < 5.0
    assert fit["pr_at_bound"] is False


def test_fit_vogel_ipr_soft_when_clustered():
    # Tightly clustered pwf → pr is weakly constrained; fitter still returns but
    # flags the small spread (caller should trust only the pr *trend*, not its value).
    pwf = [500, 505, 498, 502]
    q = _vogel_q(pwf, 1800.0, 1000.0)
    fit = he.fit_vogel_ipr(pwf, q)
    assert fit is not None
    assert fit["pwf_spread"] < 10.0


def test_fit_vogel_ipr_too_few():
    assert he.fit_vogel_ipr([500], [800]) is None
    assert he.fit_vogel_ipr([], []) is None


def test_windowed_ipr_fits_splits_and_tracks_depletion():
    # 9 monthly tests; pr declines 1850→1650 over the window. Expect ≥2 windows
    # whose fitted pr trends DOWN (depletion), each fit on a decent pwf spread.
    dates = pd.to_datetime([f"2025-{m:02d}-15" for m in range(6, 12)]
                           + [f"2026-{m:02d}-15" for m in range(1, 4)])
    n = len(dates)
    prs = np.linspace(1850, 1650, n)
    # vary pwf within each month so the windows have spread to fit against
    pwf = [400, 900, 1300, 450, 950, 1350, 500, 1000, 1400]
    q = [float(_vogel_q(p, pr, 1000.0)) for p, pr in zip(pwf, prs)]
    tw = pd.DataFrame({"WtDate": dates, "BHP": pwf, "WtOilVol": q})
    wins = he.windowed_ipr_fits(tw, min_tests=3, min_days=25, max_days=95)
    assert len(wins) >= 2
    assert all(w["mid"] is not None for w in wins)
    # pseudo-pr should trend downward across the ordered windows (depletion)
    pr_series = [w["pr"] for w in wins]
    assert pr_series[-1] < pr_series[0]


def test_windowed_ipr_fits_too_few():
    tw = pd.DataFrame({"WtDate": pd.to_datetime(["2025-06-01"]), "BHP": [500], "WtOilVol": [800]})
    assert he.windowed_ipr_fits(tw) == []


def test_pr_hi_for_formation():
    assert he.pr_hi_for_formation("Schrader") == 2200.0
    assert he.pr_hi_for_formation("Kuparuk") == 4200.0
    assert he.pr_hi_for_formation("???") == he.PR_MAX_DEFAULT  # unknown → default


def test_fit_vogel_ipr_respects_pr_cap():
    # Data generated from a HIGH true pr=2600, but capped at the Schrader ceiling
    # 2200 → the fit clamps to the bound and flags it (don't trust the absolute pr).
    pwf = [400, 800, 1200, 1600, 2000]
    q = _vogel_q(pwf, 2600.0, 1000.0)
    fit = he.fit_vogel_ipr(pwf, q, pr_hi=2200.0)
    assert fit is not None
    assert fit["pr"] <= 2200.0 + 1e-6
    assert fit["pr_at_bound"] is True


def test_fit_well_ipr_uses_all_points():
    # one fit to all points on TOTAL LIQUID; recovers pr from a wide spread, None if too few
    pr_true = 2000.0
    pwf = [300, 600, 900, 1200, 1500, 1700]
    df = pd.DataFrame({"BHP": pwf, "WtTotalFluid": _vogel_q(pwf, pr_true, 1100.0)})
    fit = he.fit_well_ipr(df, pr_hi=2200.0)
    assert fit is not None and fit["pr"] == pytest.approx(pr_true, abs=60)
    assert fit["n"] == 6
    assert he.fit_well_ipr(pd.DataFrame({"BHP": [500, 600], "WtTotalFluid": [800, 700]})) is None  # <3


def test_depletion_signature_depleting():
    # BHP and liquid fall together over the year → positive corr → depleting
    dates = pd.to_datetime([f"2025-{m:02d}-15" for m in range(6, 12)]
                           + ["2026-01-15", "2026-02-15"])
    bhp = [1400, 1350, 1300, 1200, 1150, 1100, 1000, 950]
    liq = [900, 870, 840, 760, 730, 700, 640, 600]
    sig = he.depletion_signature(pd.DataFrame({"WtDate": dates, "BHP": bhp, "WtTotalFluid": liq}))
    assert sig["verdict"] == "depleting"
    assert sig["corr"] > 0.35 and sig["bhp_per_mo"] < 0 and sig["rate_per_mo"] < 0


def test_depletion_signature_back_pressure():
    # liquid rises when BHP falls (Vogel inverse) → negative corr → back-pressure
    dates = pd.to_datetime([f"2025-{m:02d}-15" for m in range(6, 14 - 1)][:6])
    bhp = [500, 800, 1100, 600, 900, 1200]
    liq = [1000, 760, 520, 940, 700, 460]   # high liquid at low BHP
    sig = he.depletion_signature(pd.DataFrame({"WtDate": dates, "BHP": bhp, "WtTotalFluid": liq}))
    assert sig["verdict"] == "back-pressure"
    assert sig["corr"] < -0.35


def test_depletion_signature_improving():
    # BHP and liquid rise together materially → IPR rising, not depleting
    dates = pd.to_datetime([f"2025-{m:02d}-15" for m in range(6, 12)] + ["2026-01-15", "2026-02-15"])
    bhp = [900, 950, 1000, 1080, 1150, 1200, 1300, 1360]
    liq = [600, 640, 680, 760, 830, 880, 980, 1040]
    sig = he.depletion_signature(pd.DataFrame({"WtDate": dates, "BHP": bhp, "WtTotalFluid": liq}))
    assert sig["verdict"] == "improving"
    assert sig["corr"] > 0.35 and sig["rate_per_mo"] > 10


def test_depletion_signature_correlated_but_flat():
    # positive corr but only a tiny rate trend (< rate_eps) → flat/mixed, NOT depleting
    dates = pd.to_datetime([f"2025-{m:02d}-15" for m in range(6, 12)] + ["2026-01-15", "2026-02-15"])
    bhp = [1000, 1005, 998, 1003, 1001, 1006, 999, 1004]
    liq = [700, 703, 699, 702, 701, 704, 700, 703]      # ~flat
    sig = he.depletion_signature(pd.DataFrame({"WtDate": dates, "BHP": bhp, "WtTotalFluid": liq}))
    assert sig["verdict"] == "flat/mixed"


def test_depletion_signature_insufficient():
    sig = he.depletion_signature(pd.DataFrame({"WtDate": pd.to_datetime(["2025-06-01"]),
                                               "BHP": [500], "WtTotalFluid": [800]}))
    assert sig["verdict"] == "insufficient" and sig["n"] == 1


# ── deterministic per-well header-impact chain ───────────────────────────────


def test_estimate_header_impact_full_chain():
    # +100 header, dBHP/dWHP=0.8 → ΔBHP=+80; liquid IPR (qmax=2000, pr=2000) at BHP 900→980;
    # ΔOil = ΔLiquid × (1−WC).
    r = he.estimate_header_impact(d_whp=100, dbhp_dwhp=0.8, qmax=2000.0, pr=2000.0,
                                  bhp_now=900.0, wc=0.5)
    assert r["dbhp"] == pytest.approx(80.0)
    assert r["bhp_scen"] == pytest.approx(980.0)
    dliq = he.vogel_oil(980.0, 2000.0, 2000.0) - he.vogel_oil(900.0, 2000.0, 2000.0)
    assert r["dliquid"] == pytest.approx(dliq)
    assert r["doil"] == pytest.approx(dliq * 0.5)        # raising BHP lowers liquid → ΔOil < 0
    assert r["doil"] < 0


def test_estimate_header_impact_sonic_zero():
    # sonic JP: header can't reach BHP → everything 0 regardless of the slope passed
    r = he.estimate_header_impact(d_whp=100, dbhp_dwhp=0.8, qmax=2000.0, pr=2000.0,
                                  bhp_now=900.0, wc=0.5, sonic=True)
    assert r["dbhp"] == 0.0 and r["dliquid"] == pytest.approx(0.0) and r["doil"] == pytest.approx(0.0)


def test_estimate_header_impact_missing_inputs():
    # no resolved correlation → can't compute ΔBHP/ΔOil
    r = he.estimate_header_impact(d_whp=100, dbhp_dwhp=np.nan, qmax=2000.0, pr=2000.0,
                                  bhp_now=900.0, wc=0.5)
    assert np.isnan(r["dbhp"]) and np.isnan(r["doil"])
    # have correlation but no IPR → ΔBHP known, ΔOil unknown
    r2 = he.estimate_header_impact(d_whp=100, dbhp_dwhp=0.8, qmax=np.nan, pr=np.nan,
                                   bhp_now=900.0, wc=0.5)
    assert r2["dbhp"] == pytest.approx(80.0) and np.isnan(r2["doil"])


def test_estimate_header_impacts_donor_ladder():
    wells = {
        # A: own correlation + own IPR → high confidence
        "A": {"own_corr": 0.8, "sonic": False, "lift": "JP", "qmax": 2000.0, "pr": 2000.0,
              "pad": "G", "formation": "Schrader", "bhp_now": 900.0, "wc": 0.4},
        # B: no own corr, JP + physics sonic → ΔBHP forced 0 (no lever)
        "B": {"own_corr": None, "sonic": True, "lift": "JP", "qmax": 1800.0, "pr": 1900.0,
              "pad": "G", "formation": "Schrader", "bhp_now": 850.0, "wc": 0.5},
        # C: no own corr, not sonic → borrows the group-by-lift correlation (A's 0.8);
        #    no own IPR → borrows the pad+formation group IPR (A's qmax/pr)
        "C": {"own_corr": None, "sonic": False, "lift": "JP", "qmax": None, "pr": None,
              "pad": "G", "formation": "Schrader", "bhp_now": 880.0, "wc": 0.3},
    }
    out = he.estimate_header_impacts(wells, d_whp=100)
    assert out["A"]["conf"] == "high" and out["A"]["corr_src"] == "own"
    assert out["B"]["conf"] == "sonic" and out["B"]["dbhp"] == 0.0 and out["B"]["doil"] == pytest.approx(0.0)
    assert out["C"]["corr_src"] == "group" and out["C"]["corr"] == pytest.approx(0.8)
    assert out["C"]["ipr_src"] == "group" and out["C"]["conf"] == "med"
    assert out["A"]["doil"] < 0 and out["C"]["doil"] < 0   # +100 psi header → oil down


def test_estimate_header_impacts_specific_well_override():
    wells = {
        "A": {"own_corr": 0.9, "sonic": False, "lift": "JP", "qmax": 2000.0, "pr": 2000.0,
              "pad": "G", "formation": "Schrader", "bhp_now": 900.0, "wc": 0.4},
        # B: would auto-resolve to sonic(0), but user overrides correlation to well A
        "B": {"own_corr": None, "sonic": True, "lift": "JP", "qmax": 1800.0, "pr": 1900.0,
              "pad": "G", "formation": "Schrader", "bhp_now": 850.0, "wc": 0.5,
              "corr_donor": "A"},
    }
    out = he.estimate_header_impacts(wells, d_whp=100)
    assert out["B"]["corr_src"] == "donor" and out["B"]["corr"] == pytest.approx(0.9)
    assert out["B"]["dbhp"] != 0.0          # the override beat the sonic→0 default


def test_estimate_header_impacts_group_corr_override_for_ci():
    wells = {
        "A": {"own_corr": 0.8, "sonic": False, "lift": "JP", "qmax": 2000.0, "pr": 2000.0,
              "pad": "G", "formation": "Schrader", "bhp_now": 900.0, "wc": 0.0},
        "B": {"own_corr": None, "sonic": False, "lift": "JP", "qmax": 2000.0, "pr": 2000.0,
              "pad": "G", "formation": "Schrader", "bhp_now": 900.0, "wc": 0.0},
    }
    base = he.estimate_header_impacts(wells, 100)
    lo = he.estimate_header_impacts(wells, 100, group_corr_override={"JP": 0.4})
    # B borrows the group correlation; halving it halves its ΔBHP (and ~ΔOil magnitude)
    assert lo["B"]["corr"] == pytest.approx(0.4)
    assert abs(lo["B"]["doil"]) < abs(base["B"]["doil"])
    assert base["A"]["doil"] == lo["A"]["doil"]   # own-corr well A unaffected by the override


def test_vogel_oil_matches_curve():
    # qmax at pwf=0; ~0 at pwf=pr; clamps beyond pr
    assert he.vogel_oil(0.0, 1000.0, 1800.0) == pytest.approx(1000.0)
    assert he.vogel_oil(1800.0, 1000.0, 1800.0) == pytest.approx(0.0, abs=1e-6)
    assert he.vogel_oil(2500.0, 1000.0, 1800.0) == pytest.approx(0.0, abs=1e-6)  # clamped
    # round-trips a fit_vogel_ipr point
    assert he.vogel_oil(900.0, 1000.0, 1800.0) == pytest.approx(_vogel_q(900.0, 1800.0, 1000.0))
