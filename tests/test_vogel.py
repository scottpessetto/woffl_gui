"""Pin/regression tests for the consolidated Vogel curve math (woffl.gui.vogel).

Code review finding R-10/R-11 ("Vogel math triplication"): three GUI call
sites re-derived the Vogel IPR polynomial independently --

1. ``woffl.assembly.ipr_analyzer`` -- the canonical single test-point forward
   derivation goes through the library's ``InFlow.vogel_qmax`` /
   ``InFlow.oil_flow(..., "vogel")`` inside ``compute_vogel_coefficients`` /
   ``generate_ipr_curves``.
2. ``woffl.gui.workflow_steps.step2_review_ipr._template_to_vogel_coeffs`` --
   re-derived the forward qmax formula by hand (``x*x`` instead of ``x**2``)
   to re-visualize a hand-edited, re-uploaded optimization template.
3. ``woffl.gui.optimization_viz.create_ipr_comparison_pdf`` -- re-derived
   both the forward curve AND an inverse rate -> pwf solve (for the PDF's
   "Current" operating-point marker) by hand.

``woffl.gui.vogel`` now holds the one implementation of the reusable
forward/inverse pieces (2) and (3) delegate to; (1)'s RP-fit search
(``_normalized_curve_sse`` / ``_calculate_global_sse`` /
``estimate_reservoir_pressure`` / ``_calculate_r_squared``) is deliberately
NOT touched or routed through these helpers -- see the CLAUDE.md gotcha "The
Vogel RP fit objective is axis-normalized" and ``vogel.py``'s module
docstring. That fit's regression guard lives in
``test_ipr_analyzer.py::TestEstimateReservoirPressure::
test_flat_cloud_does_not_rail_to_cap`` and must stay green independent of
this file.

This file pins:
  - ``woffl.gui.vogel``'s helpers against ``InFlow`` (site 1's canonical
    forward math) over a grid including edge cases (pwf=0, pwf=pres,
    pwf>pres).
  - ``_template_to_vogel_coeffs`` (site 2, post-refactor) against a literal
    reproduction of its ORIGINAL pre-refactor formula
    (``x = pwf/ResP; vogel_frac = 1 - 0.2*x - 0.8*x*x``) so the refactor is
    provably behavior-preserving, including the pwf>=ResP fallback-to-1.0
    guard.
  - The forward-curve and inverse-pwf math used inside
    ``create_ipr_comparison_pdf`` (site 3, post-refactor) against a literal
    reproduction of its ORIGINAL pre-refactor inline formulas (same
    ``**2`` form, and critically the same floating-point operation ORDER
    for the inverse: divide-then-multiply, ``(-0.2+sqrt(disc))/1.6*pres``,
    which is NOT bit-identical to multiply-then-divide).
  - A smoke/integration test that ``create_ipr_comparison_pdf`` still runs
    end-to-end and returns a valid PDF after being rewired onto
    ``woffl.gui.vogel``.
"""

import io
import math

import numpy as np
import pandas as pd
import pytest

from woffl.assembly.network_optimizer import OptimizationResult, WellConfig
from woffl.flow.inflow import InFlow
from woffl.gui.optimization_viz import create_ipr_comparison_pdf
from woffl.gui.vogel import vogel_fraction, vogel_pwf_from_rate, vogel_qmax, vogel_rate
from woffl.gui.workflow_steps.step2_review_ipr import _template_to_vogel_coeffs

# ---------------------------------------------------------------------------
# Shared grid of (qwf, pwf, pres) test points, including edge cases.
# ---------------------------------------------------------------------------

PRES_VALUES = [800.0, 1500.0, 1800.0, 3000.0]
QWF_VALUES = [50.0, 500.0, 1500.0]
PWF_FRACTIONS = [0.0, 0.25, 0.5, 0.75, 0.999]  # strictly below pres -> normal cases


def _below_pres_grid():
    """(qwf, pwf, pres) combos strictly below pres (the well-behaved region)."""
    for pres in PRES_VALUES:
        for frac in PWF_FRACTIONS:
            pwf = pres * frac
            for qwf in QWF_VALUES:
                yield qwf, pwf, pres


# ---------------------------------------------------------------------------
# Legacy formula replicas -- verbatim copies of the PRE-refactor inline code
# at each of the three original call sites, kept here purely as an
# independent oracle for the pin test. Do not "simplify" these; they exist
# to prove the real (refactored) code hasn't drifted.
# ---------------------------------------------------------------------------


def _legacy_step2_qmax_recent(
    pwf: pd.Series, res_p: pd.Series, qwf: pd.Series
) -> pd.Series:
    """Verbatim pre-refactor formula from
    step2_review_ipr._template_to_vogel_coeffs (see git history):

        x = coeffs["pwf"] / coeffs["ResP"]
        vogel_frac = 1 - 0.2 * x - 0.8 * x * x
        coeffs["QMax_recent"] = coeffs["qwf"] / vogel_frac.where(vogel_frac > 0, 1.0)
    """
    x = pwf / res_p
    vogel_frac = 1 - 0.2 * x - 0.8 * x * x
    return qwf / vogel_frac.where(vogel_frac > 0, 1.0)


def _legacy_optviz_forward_fraction(pwf: float, pres: float) -> float:
    """Verbatim pre-refactor formula from
    optimization_viz.create_ipr_comparison_pdf:

        vogel_frac = 1 - 0.2 * (pwf / pres) - 0.8 * (pwf / pres) ** 2
    """
    return 1 - 0.2 * (pwf / pres) - 0.8 * (pwf / pres) ** 2


def _legacy_optviz_oil_rates(
    pressures: np.ndarray, qmax_oil: float, pres: float
) -> np.ndarray:
    """Verbatim pre-refactor formula from optimization_viz.create_ipr_comparison_pdf:

    oil_rates = qmax_oil * (
        1 - 0.2 * (pressures / pres) - 0.8 * (pressures / pres) ** 2
    )
    """
    return qmax_oil * (1 - 0.2 * (pressures / pres) - 0.8 * (pressures / pres) ** 2)


def _legacy_optviz_current_bhp(actual_oil, qmax_oil, pres):
    """Verbatim pre-refactor formula from optimization_viz.create_ipr_comparison_pdf:

        if (
            current_bhp is None
            and actual_oil is not None
            and 0 < actual_oil < qmax_oil
        ):
            disc = 0.04 + 3.2 * (1 - actual_oil / qmax_oil)
            if disc >= 0:
                current_bhp = (-0.2 + math.sqrt(disc)) / 1.6 * pres

    Returns None where the original left current_bhp untouched (None).
    """
    if actual_oil is None or not (0 < actual_oil < qmax_oil):
        return None
    disc = 0.04 + 3.2 * (1 - actual_oil / qmax_oil)
    if disc >= 0:
        return (-0.2 + math.sqrt(disc)) / 1.6 * pres
    return None


# ===========================================================================
# 1. vogel.py vs InFlow (site 1's canonical forward math) -- including edges
# ===========================================================================


class TestVogelHelpersVsInFlow:
    def test_vogel_qmax_matches_inflow_below_pres(self):
        for qwf, pwf, pres in _below_pres_grid():
            assert vogel_qmax(qwf, pwf, pres) == InFlow.vogel_qmax(qwf, pwf, pres)

    def test_vogel_qmax_edge_pwf_zero(self):
        assert (
            vogel_qmax(500.0, 0.0, 1800.0)
            == InFlow.vogel_qmax(500.0, 0.0, 1800.0)
            == 500.0
        )

    def test_vogel_qmax_edge_pwf_equals_pres_raises_like_inflow(self):
        # ratio == 1 -> fraction == 0.0 exactly -> float division by zero.
        # vogel_qmax delegates straight to InFlow.vogel_qmax, so it must
        # fail the exact same way (not silently return inf/nan).
        with pytest.raises(ZeroDivisionError):
            InFlow.vogel_qmax(500.0, 1800.0, 1800.0)
        with pytest.raises(ZeroDivisionError):
            vogel_qmax(500.0, 1800.0, 1800.0)

    def test_vogel_qmax_edge_pwf_above_pres_matches_inflow_sign(self):
        # Beyond pres the fraction goes negative -> qmax goes negative too.
        # Both implementations must agree (neither silently clamps).
        qmax_a = vogel_qmax(500.0, 2000.0, 1800.0)
        qmax_b = InFlow.vogel_qmax(500.0, 2000.0, 1800.0)
        assert qmax_a == qmax_b
        assert qmax_a < 0

    def test_vogel_rate_matches_inflow_oil_flow_below_and_at_pres(self):
        for qwf, pwf_test, pres in _below_pres_grid():
            inflow = InFlow(qwf, pwf_test, pres)
            qmax = InFlow.vogel_qmax(qwf, pwf_test, pres)
            # Sample the curve at several pnew values, including pnew=0 and
            # pnew=pres exactly (allowed by InFlow.oil_flow: only pnew>pres
            # raises).
            for pnew in (0.0, pres * 0.5, pres * 0.999, pres):
                expected = inflow.oil_flow(pnew, "vogel")
                actual = vogel_rate(pnew, qmax, pres)
                assert actual == expected

    def test_vogel_rate_array_matches_scalar_loop(self):
        qwf, pwf_test, pres = 700.0, 550.0, 1800.0
        qmax = InFlow.vogel_qmax(qwf, pwf_test, pres)
        pressures = np.linspace(0, pres, 200)
        vec = vogel_rate(pressures, qmax, pres)
        inflow = InFlow(qwf, pwf_test, pres)
        for i, p in enumerate(pressures):
            assert vec[i] == inflow.oil_flow(float(p), "vogel")

    def test_vogel_fraction_zero_at_pwf_equals_pres(self):
        assert vogel_fraction(1800.0, 1800.0) == 0.0


# ===========================================================================
# 2. step2_review_ipr._template_to_vogel_coeffs (site 2, post-refactor) vs
#    the verbatim pre-refactor formula.
# ===========================================================================


class TestStep2TemplateToVogelCoeffs:
    def _build_template_df(self, rows):
        return pd.DataFrame(
            {
                "Well": [r[0] for r in rows],
                "res_pres": [r[1] for r in rows],
                "qwf_blpd": [r[2] for r in rows],
                "pwf": [r[3] for r in rows],
            }
        )

    def test_matches_legacy_formula_on_grid_including_edges(self):
        rows = []
        i = 0
        for qwf, pwf, pres in _below_pres_grid():
            rows.append((f"W{i}", pres, qwf, pwf))
            i += 1
        # Edge cases the original `.where(vogel_frac > 0, 1.0)` guard exists
        # for: pwf == ResP (fraction exactly 0) and pwf > ResP (negative
        # fraction) -- both must fall back to QMax_recent == qwf, not crash.
        rows.append(("EDGE-AT-RP", 1800.0, 500.0, 1800.0))
        rows.append(("EDGE-ABOVE-RP", 1800.0, 500.0, 1900.0))

        df = self._build_template_df(rows)
        result = _template_to_vogel_coeffs(df)

        legacy = _legacy_step2_qmax_recent(df["pwf"], df["res_pres"], df["qwf_blpd"])

        # Row order/filter: _template_to_vogel_coeffs keeps rows with ResP>0
        # (all of ours qualify) and resets the index.
        pd.testing.assert_series_equal(
            result["QMax_recent"].reset_index(drop=True),
            legacy.reset_index(drop=True),
            check_names=False,
        )

    def test_edge_pwf_at_or_above_resp_falls_back_to_qwf(self):
        df = self._build_template_df(
            [
                ("AT-RP", 1800.0, 733.0, 1800.0),
                ("ABOVE-RP", 1800.0, 733.0, 1950.0),
            ]
        )
        result = _template_to_vogel_coeffs(df)
        assert result.loc[result["Well"] == "AT-RP", "QMax_recent"].iloc[0] == 733.0
        assert result.loc[result["Well"] == "ABOVE-RP", "QMax_recent"].iloc[0] == 733.0

    def test_edge_pwf_zero(self):
        df = self._build_template_df([("ZERO-PWF", 1800.0, 733.0, 0.0)])
        result = _template_to_vogel_coeffs(df)
        assert result["QMax_recent"].iloc[0] == 733.0


# ===========================================================================
# 3. optimization_viz.create_ipr_comparison_pdf math (site 3, post-refactor,
#    now delegating to woffl.gui.vogel) vs the verbatim pre-refactor formula.
# ===========================================================================


class TestOptimizationVizVogelMath:
    def test_forward_fraction_matches_legacy_on_grid(self):
        for _, pwf, pres in _below_pres_grid():
            assert vogel_fraction(pwf, pres) == _legacy_optviz_forward_fraction(
                pwf, pres
            )

    def test_forward_curve_matches_legacy_on_grid(self):
        for qwf, pwf_test, pres in _below_pres_grid():
            frac = _legacy_optviz_forward_fraction(pwf_test, pres)
            if frac <= 0:
                continue
            qmax = qwf / frac
            qmax_oil = qmax * 0.5  # arbitrary form_wc = 0.5
            pressures = np.linspace(0, pres, 200)
            expected = _legacy_optviz_oil_rates(pressures, qmax_oil, pres)
            actual = vogel_rate(pressures, qmax_oil, pres)
            assert np.array_equal(actual, expected)

    def test_inverse_matches_legacy_including_order_sensitive_edges(self):
        cases = [
            (100.0, 500.0, 1800.0),  # normal, well below qmax
            (0.0, 500.0, 1800.0),  # rate == 0 -> legacy returns None (0 < 0 is False)
            (500.0, 500.0, 1800.0),  # rate == qmax -> boundary, legacy returns None
            (600.0, 500.0, 1800.0),  # rate > qmax -> legacy returns None
            (499.999999, 500.0, 1800.0),  # just under qmax
        ]
        for actual_oil, qmax_oil, pres in cases:
            expected = _legacy_optviz_current_bhp(actual_oil, qmax_oil, pres)
            actual = vogel_pwf_from_rate(actual_oil, qmax_oil, pres)
            assert actual == expected, (actual_oil, qmax_oil, pres)

    def test_inverse_none_rate_matches_legacy_none_guard(self):
        # actual_oil is None upstream (gaugeless well, no test) -> both must
        # leave current_bhp as None.
        assert vogel_pwf_from_rate(None, 500.0, 1800.0) is None


# ===========================================================================
# 4. Cross-site agreement: the same test point through all three paths.
# ===========================================================================


class TestCrossSiteAgreement:
    def test_all_three_paths_agree_on_qmax(self):
        for qwf, pwf, pres in _below_pres_grid():
            if pwf == 0:
                continue  # trivial fraction=1 case, not interesting here
            inflow_qmax = InFlow.vogel_qmax(qwf, pwf, pres)

            df = pd.DataFrame(
                {"Well": ["X"], "res_pres": [pres], "qwf_blpd": [qwf], "pwf": [pwf]}
            )
            step2_qmax = _template_to_vogel_coeffs(df)["QMax_recent"].iloc[0]

            frac = vogel_fraction(pwf, pres)
            optviz_qmax = qwf / frac if frac > 0 else qwf

            assert inflow_qmax == step2_qmax == optviz_qmax


# ===========================================================================
# 5. Integration smoke test: create_ipr_comparison_pdf still runs end-to-end
#    after being rewired onto woffl.gui.vogel.
# ===========================================================================


class _FakeOptimizer:
    def __init__(self, wells):
        self._wells = {w.well_name: w for w in wells}

    def get_well_by_name(self, name):
        return self._wells.get(name)


class TestCreateIprComparisonPdfSmoke:
    def _make_result(self, well_name="B-28"):
        return OptimizationResult(
            well_name=well_name,
            recommended_nozzle="12",
            recommended_throat="B",
            allocated_power_fluid=3000.0,
            predicted_oil_rate=450.0,
            predicted_formation_water=200.0,
            predicted_lift_water=3000.0,
            suction_pressure=650.0,
            marginal_oil_rate=0.12,
            sonic_status=False,
            mach_te=0.8,
        )

    def _make_well_config(self, well_name="B-28"):
        return WellConfig(
            well_name=well_name,
            res_pres=1800.0,
            form_temp=90.0,
            jpump_tvd=6000.0,
            form_wc=0.5,
            qwf=900.0,
            pwf=700.0,
        )

    def test_runs_and_produces_valid_pdf_with_measured_bhp(self):
        result = self._make_result()
        well_config = self._make_well_config()
        optimizer = _FakeOptimizer([well_config])

        pdf_bytes = create_ipr_comparison_pdf(
            results=[result],
            optimizer=optimizer,
            actual_oil_map={"B-28": 420.0},
            actual_pf_map={"B-28": 2900.0},
            actual_bhp_map={"B-28": 720.0},
            current_jp_map={"B-28": "12B"},
        )
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes[:4] == b"%PDF"

    def test_runs_and_produces_valid_pdf_with_inverse_vogel_bhp(self):
        """No measured BHP -- current_bhp must come from vogel_pwf_from_rate."""
        result = self._make_result()
        well_config = self._make_well_config()
        optimizer = _FakeOptimizer([well_config])

        pdf_bytes = create_ipr_comparison_pdf(
            results=[result],
            optimizer=optimizer,
            actual_oil_map={"B-28": 420.0},
            actual_pf_map={"B-28": 2900.0},
            actual_bhp_map={},  # forces the inverse-Vogel path
            current_jp_map={"B-28": "12B"},
        )
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes[:4] == b"%PDF"

    def test_runs_with_no_actual_data_gaugeless_well(self):
        result = self._make_result()
        well_config = self._make_well_config()
        optimizer = _FakeOptimizer([well_config])

        pdf_bytes = create_ipr_comparison_pdf(
            results=[result],
            optimizer=optimizer,
            actual_oil_map={},
            actual_pf_map={},
            actual_bhp_map={},
            current_jp_map={},
        )
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes[:4] == b"%PDF"
