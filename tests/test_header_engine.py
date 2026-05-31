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
    assert toks[:3] == [he.OWN_TOKEN, he.GROUP_PADLIFT, he.GROUP_FORMATION]
    assert toks[3:] == ["MPG-01", "MPH-01"]  # wells sorted after the tokens


def test_donor_member_wells_own_is_none():
    assert he.donor_member_wells("MPG-01", he.OWN_TOKEN, _META) is None
    assert he.donor_member_wells("MPG-01", "", _META) is None


def test_donor_member_wells_group_padlift():
    m = he.donor_member_wells("MPG-01", he.GROUP_PADLIFT, _META)
    assert sorted(m) == ["MPG-01", "MPG-02"]  # G + JP only (not the G ESP, not H)


def test_donor_member_wells_group_formation():
    m = he.donor_member_wells("MPG-01", he.GROUP_FORMATION, _META)
    assert sorted(m) == ["MPG-01", "MPG-02", "MPG-03"]  # all Schrader, regardless of lift


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
