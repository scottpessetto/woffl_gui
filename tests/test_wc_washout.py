"""Tests for woffl.gui.wc_washout — WC washout detection + suggestion sweep.

Field context (MPE-19, 2026-07): low-rate well, test showed 0% WC because the
PF return (~2,774 BWPD) swamps the separator and the allocation nets
formation water to ~0. Raising modeled WC to 65% matched BHP and oil. The
detector must flag that setup; the suggester must recover ~0.65 from a
solver whose BHP/oil respond to WC.

The module is pure (no Streamlit import), so no mocking is needed.
"""

import pytest

from woffl.gui.wc_washout import (
    SWEEP_HI,
    WcSuggestion,
    detect_wc_washout,
    suggest_water_cut,
)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def test_detects_e19_like_washout():
    """0% WC, produced fluid ~5% of PF, BHP way off → flag with the numbers."""
    flag = detect_wc_washout(
        form_wc=0.0,
        pf_rate=2774.0,
        produced_fluid=151.0,
        modeled_psu=200.0,
        actual_bhp=548.0,
        modeled_oil=158.0,
        actual_oil=151.0,
    )
    assert flag is not None
    assert flag.fluid_to_pf_ratio == pytest.approx(151.0 / 2774.0)
    assert flag.bhp_delta == pytest.approx(-348.0)


def test_healthy_wc_never_flags():
    flag = detect_wc_washout(
        form_wc=0.40,
        pf_rate=2774.0,
        produced_fluid=151.0,
        modeled_psu=200.0,
        actual_bhp=548.0,
    )
    assert flag is None


def test_high_fluid_to_pf_ratio_never_flags():
    """A real fluid stream (36% of PF) — the allocation can see the water."""
    flag = detect_wc_washout(
        form_wc=0.0,
        pf_rate=2774.0,
        produced_fluid=1000.0,
        modeled_psu=200.0,
        actual_bhp=548.0,
    )
    assert flag is None


def test_good_match_never_flags():
    """Near-zero WC + washout exposure but the model matches → no nag."""
    flag = detect_wc_washout(
        form_wc=0.0,
        pf_rate=2774.0,
        produced_fluid=151.0,
        modeled_psu=540.0,
        actual_bhp=548.0,
        modeled_oil=153.0,
        actual_oil=151.0,
    )
    assert flag is None


def test_no_mismatch_evidence_never_flags():
    """Without BHP or oil comparisons there's no sign of matching trouble."""
    flag = detect_wc_washout(
        form_wc=0.0,
        pf_rate=2774.0,
        produced_fluid=151.0,
    )
    assert flag is None


def test_oil_only_mismatch_flags():
    """No measured BHP, but oil is 32% high → still flags."""
    flag = detect_wc_washout(
        form_wc=0.02,
        pf_rate=2774.0,
        produced_fluid=151.0,
        modeled_oil=200.0,
        actual_oil=151.0,
    )
    assert flag is not None
    assert flag.bhp_delta is None
    assert flag.oil_delta_frac == pytest.approx((200.0 - 151.0) / 151.0)


# ---------------------------------------------------------------------------
# Suggestion sweep
# ---------------------------------------------------------------------------


def _linear_solver(wc: float):
    """Hero-tuple stub whose BHP and oil both hit their targets at wc=0.65:
    psu = 200 + 500·wc (→ 525), qoil = 100 + 80·wc (→ 152)."""
    psu = 200.0 + 500.0 * wc
    qoil = 100.0 + 80.0 * wc
    return (psu, False, qoil, 300.0, 2600.0, 0.2)


def test_sweep_recovers_matching_wc():
    s = suggest_water_cut(
        _linear_solver, target_oil=152.0, target_bhp=525.0, base_wc=0.0
    )
    assert isinstance(s, WcSuggestion)
    assert s.suggested_wc == pytest.approx(0.65, abs=0.011)
    assert s.matched_on == "BHP + oil"
    assert not s.bounded
    assert s.modeled_psu == pytest.approx(525.0, abs=6.0)
    assert s.modeled_oil == pytest.approx(152.0, abs=1.0)


def test_oil_only_match_is_labeled():
    s = suggest_water_cut(_linear_solver, target_oil=152.0, base_wc=0.0)
    assert s is not None
    assert s.matched_on == "oil only"
    assert s.suggested_wc == pytest.approx(0.65, abs=0.011)


def test_all_failures_returns_none():
    assert (
        suggest_water_cut(lambda wc: None, target_oil=152.0, target_bhp=525.0)
        is None
    )


def test_boundary_best_is_flagged_bounded():
    """Error still falling at the top of the sweep → bounded=True."""

    def solver(wc: float):
        return (1000.0 * wc, False, 152.0, 300.0, 2600.0, 0.2)

    s = suggest_water_cut(solver, target_oil=152.0, target_bhp=2000.0)
    assert s is not None
    assert s.suggested_wc == pytest.approx(SWEEP_HI, abs=0.011)
    assert s.bounded


def test_base_wc_is_included_in_sweep():
    s = suggest_water_cut(
        _linear_solver, target_oil=152.0, target_bhp=525.0, base_wc=0.33
    )
    assert s is not None
    assert any(p.wc == pytest.approx(0.33) for p in s.points)


def test_unusable_target_oil_returns_none():
    assert suggest_water_cut(_linear_solver, target_oil=0.0) is None
    assert suggest_water_cut(_linear_solver, target_oil=None) is None
