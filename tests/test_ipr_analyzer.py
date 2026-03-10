"""Tests for woffl.assembly.ipr_analyzer module.

Covers all public and private functions with synthetic Vogel-curve data.
No database or Streamlit dependencies required.

Run with:
    python -m pytest tests/test_ipr_analyzer.py
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from woffl.assembly.ipr_analyzer import (
    _calculate_global_sse,
    _calculate_r_squared,
    compute_vogel_coefficients,
    estimate_reservoir_pressure,
    export_optimization_template,
    generate_ipr_curves,
)


# ---------------------------------------------------------------------------
# Helpers for generating synthetic Vogel data
# ---------------------------------------------------------------------------


def vogel_flow(bhp: float, pres: float, qmax: float) -> float:
    """Compute fluid rate from Vogel equation: q = qmax * (1 - 0.2*r - 0.8*r^2)."""
    r = bhp / pres
    return qmax * (1.0 - 0.2 * r - 0.8 * r**2)


def make_vogel_points(pres: float, qmax: float, bhp_values: np.ndarray) -> np.ndarray:
    """Return exact Vogel fluid rates for given BHP values."""
    return np.array([vogel_flow(b, pres, qmax) for b in bhp_values])


def make_merged_dataframe(
    well_name,
    bhp_values,
    fluid_values,
    dates=None,
    water_fracs=None,
):
    """Build a DataFrame matching the 'merged_data' schema expected by the module."""
    n = len(bhp_values)
    if dates is None:
        base = pd.Timestamp.now()
        dates = [base - pd.Timedelta(days=int(d)) for d in np.linspace(0, 365, n)]
    df = pd.DataFrame(
        {
            "well": well_name,
            "BHP": bhp_values,
            "WtTotalFluid": fluid_values,
            "WtDate": dates,
        }
    )
    if water_fracs is not None:
        df["WtWaterVol"] = np.array(water_fracs) * fluid_values
    return df


# ===================================================================
# 1. _calculate_global_sse
# ===================================================================


class TestCalculateGlobalSSE:
    """Tests for _calculate_global_sse."""

    def test_perfect_vogel_data_returns_near_zero_sse(self):
        """When data lies exactly on a Vogel curve, SSE should be near zero."""
        pres = 1500.0
        qmax = 800.0
        bhp_values = np.array([200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0])
        fluid_values = make_vogel_points(pres, qmax, bhp_values)

        sse = _calculate_global_sse(bhp_values, fluid_values, pres)
        assert sse == pytest.approx(0.0, abs=1e-3)

    def test_single_point_returns_inf(self):
        """With fewer than 2 points, SSE should be inf."""
        bhp = np.array([500.0])
        fluid = np.array([300.0])
        result = _calculate_global_sse(bhp, fluid, 1500.0)
        assert result == float("inf")

    def test_empty_arrays_returns_inf(self):
        """Empty inputs should return inf."""
        result = _calculate_global_sse(np.array([]), np.array([]), 1500.0)
        assert result == float("inf")

    def test_bhp_greater_equal_pres_incurs_penalty(self):
        """Points with bhp >= pres contribute a large penalty to SSE."""
        pres = 1000.0
        qmax = 500.0
        bhp_values = np.array([200.0, 400.0, 1000.0])
        fluid_values = np.array([
            vogel_flow(200.0, pres, qmax),
            vogel_flow(400.0, pres, qmax),
            50.0,
        ])

        sse = _calculate_global_sse(bhp_values, fluid_values, pres)
        assert sse >= 1e8

    def test_all_anchor_fluid_nonpositive_returns_inf(self):
        """If all anchor fluid values are <= 0, function should return inf."""
        bhp_values = np.array([200.0, 400.0, 600.0])
        fluid_values = np.array([0.0, -10.0, 0.0])

        result = _calculate_global_sse(bhp_values, fluid_values, 1500.0)
        assert result == float("inf")

    def test_all_anchors_invalid_bhp_returns_inf(self):
        """If every anchor has bhp >= pres, function should return inf."""
        pres = 500.0
        bhp_values = np.array([500.0, 600.0, 700.0])
        fluid_values = np.array([100.0, 80.0, 60.0])

        result = _calculate_global_sse(bhp_values, fluid_values, pres)
        assert result == float("inf")

    def test_noisy_data_returns_nonzero_sse(self):
        """Data with noise should produce positive but finite SSE."""
        pres = 1500.0
        qmax = 800.0
        bhp_values = np.array([200.0, 400.0, 600.0, 800.0, 1000.0])
        perfect_fluid = make_vogel_points(pres, qmax, bhp_values)

        rng = np.random.default_rng(42)
        noisy_fluid = perfect_fluid + rng.normal(0, 20, size=len(perfect_fluid))

        sse = _calculate_global_sse(bhp_values, noisy_fluid, pres)
        assert 0 < sse < float("inf")

    def test_minimum_across_anchors(self):
        """The function should return the minimum SSE across anchor choices."""
        pres = 1500.0
        qmax = 700.0
        bhp_values = np.array([300.0, 600.0, 900.0, 1200.0])
        fluid_values = make_vogel_points(pres, qmax, bhp_values)
        fluid_values[2] += 30.0

        sse = _calculate_global_sse(bhp_values, fluid_values, pres)
        assert 0 < sse < float("inf")


# ===================================================================
# 2. _calculate_r_squared
# ===================================================================


class TestCalculateRSquared:
    """Tests for _calculate_r_squared."""

    def test_perfect_fit_returns_r2_near_one(self):
        """Exact Vogel data should give R^2 very close to 1.0."""
        pres = 1500.0
        qmax = 800.0
        bhp_values = np.array([200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0])
        fluid_values = make_vogel_points(pres, qmax, bhp_values)

        anchor_bhp = bhp_values[0]
        anchor_fluid = fluid_values[0]

        r2 = _calculate_r_squared(bhp_values, fluid_values, pres, anchor_bhp, anchor_fluid)
        assert r2 == pytest.approx(1.0, abs=1e-6)

    def test_anchor_bhp_gte_pres_returns_zero(self):
        """If anchor_bhp >= pres, the function should return 0.0."""
        bhp_values = np.array([200.0, 400.0, 600.0])
        fluid_values = np.array([500.0, 300.0, 100.0])
        pres = 1000.0

        r2 = _calculate_r_squared(bhp_values, fluid_values, pres, 1000.0, 50.0)
        assert r2 == 0.0

        r2 = _calculate_r_squared(bhp_values, fluid_values, pres, 1100.0, 50.0)
        assert r2 == 0.0

    def test_fewer_than_two_points_returns_zero(self):
        """With fewer than 2 fluid values, R^2 should be 0.0."""
        r2 = _calculate_r_squared(np.array([500.0]), np.array([300.0]), 1500.0, 500.0, 300.0)
        assert r2 == 0.0

    def test_noisy_data_r2_less_than_one(self):
        """Noisy data should give R^2 < 1.0 but still positive for a reasonable fit."""
        pres = 1500.0
        qmax = 800.0
        bhp_values = np.array([200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0])
        perfect_fluid = make_vogel_points(pres, qmax, bhp_values)

        rng = np.random.default_rng(99)
        noisy_fluid = perfect_fluid + rng.normal(0, 15, size=len(perfect_fluid))

        anchor_bhp = bhp_values[2]
        anchor_fluid = noisy_fluid[2]

        r2 = _calculate_r_squared(bhp_values, noisy_fluid, pres, anchor_bhp, anchor_fluid)
        assert 0.0 < r2 < 1.0

    def test_any_anchor_gives_consistent_qmax(self):
        """When data is perfect, any anchor should produce R^2 = 1.0."""
        pres = 2000.0
        qmax = 600.0
        bhp_values = np.array([300.0, 700.0, 1100.0, 1500.0])
        fluid_values = make_vogel_points(pres, qmax, bhp_values)

        for i in range(len(bhp_values)):
            r2 = _calculate_r_squared(
                bhp_values, fluid_values, pres, bhp_values[i], fluid_values[i]
            )
            assert r2 == pytest.approx(1.0, abs=1e-6), f"Failed for anchor index {i}"


# ===================================================================
# 3. estimate_reservoir_pressure
# ===================================================================


class TestEstimateReservoirPressure:
    """Tests for estimate_reservoir_pressure."""

    def _make_well_data(self, well_name, true_pres, qmax, bhp_range, n_points=6):
        """Generate synthetic well test data on a known Vogel curve."""
        bhps = np.linspace(bhp_range[0], bhp_range[1], n_points)
        fluids = make_vogel_points(true_pres, qmax, bhps)
        return make_merged_dataframe(well_name, bhps, fluids)

    def test_finds_known_reservoir_pressure(self):
        """estimate_reservoir_pressure should find an RP close to the true value."""
        true_pres = 1400.0
        qmax = 700.0
        df = self._make_well_data("WELL-A", true_pres, qmax, (300, 1100), n_points=8)

        result = estimate_reservoir_pressure(df)

        assert "Optimal_RP" in result.columns
        assert "PI" in result.columns

        optimal_rp = result["Optimal_RP"].iloc[0]
        assert abs(optimal_rp - true_pres) < 50, (
            f"Optimal RP {optimal_rp} too far from true {true_pres}"
        )

    def test_pi_column_computed_correctly(self):
        """PI should equal WtTotalFluid / (Optimal_RP - BHP)."""
        true_pres = 1500.0
        qmax = 800.0
        df = self._make_well_data("WELL-B", true_pres, qmax, (200, 1000), n_points=5)

        result = estimate_reservoir_pressure(df)

        for _, row in result.iterrows():
            if pd.notna(row["PI"]) and pd.notna(row["Optimal_RP"]):
                expected_pi = row["WtTotalFluid"] / (row["Optimal_RP"] - row["BHP"])
                assert row["PI"] == pytest.approx(expected_pi, rel=1e-6)

    def test_multiple_wells_handled_independently(self):
        """Each well should get its own Optimal_RP value."""
        df_a = self._make_well_data("WELL-A", 1200.0, 500.0, (200, 800))
        df_b = self._make_well_data("WELL-B", 1600.0, 900.0, (300, 1200))
        df = pd.concat([df_a, df_b], ignore_index=True)

        result = estimate_reservoir_pressure(df)

        rp_a = result[result["well"] == "WELL-A"]["Optimal_RP"].iloc[0]
        rp_b = result[result["well"] == "WELL-B"]["Optimal_RP"].iloc[0]

        assert abs(rp_a - rp_b) > 100
        assert abs(rp_a - 1200) < 50
        assert abs(rp_b - 1600) < 50

    def test_schrader_max_pressure_limit(self):
        """Schrader wells should not search above max_pres_schrader."""
        true_pres = 1700.0
        qmax = 500.0
        max_schrader = 1800
        df = self._make_well_data("WELL-SCH", true_pres, qmax, (400, 1400))

        jp_chars = pd.DataFrame({"Well": ["WELL-SCH"], "is_sch": [True]})
        result = estimate_reservoir_pressure(df, max_pres_schrader=max_schrader, jp_chars=jp_chars)

        optimal_rp = result["Optimal_RP"].iloc[0]
        assert optimal_rp <= max_schrader

    def test_kuparuk_max_pressure_limit(self):
        """Kuparuk wells should use the higher max_pres_kuparuk limit."""
        true_pres = 2500.0
        qmax = 600.0
        max_kuparuk = 3000
        df = self._make_well_data("WELL-KUP", true_pres, qmax, (500, 2000))

        jp_chars = pd.DataFrame({"Well": ["WELL-KUP"], "is_sch": [False]})
        result = estimate_reservoir_pressure(
            df, max_pres_schrader=1800, max_pres_kuparuk=max_kuparuk, jp_chars=jp_chars
        )

        optimal_rp = result["Optimal_RP"].iloc[0]
        assert optimal_rp <= max_kuparuk
        assert abs(optimal_rp - true_pres) < 50

    def test_default_field_model_is_schrader(self):
        """Without jp_chars, all wells default to Schrader limits."""
        true_pres = 1600.0
        qmax = 500.0
        df = self._make_well_data("WELL-DEFAULT", true_pres, qmax, (400, 1300))

        result = estimate_reservoir_pressure(df, max_pres_schrader=1800)

        optimal_rp = result["Optimal_RP"].iloc[0]
        assert optimal_rp <= 1800

    def test_nan_bhp_rows_skipped(self):
        """Rows with NaN BHP should be handled gracefully."""
        bhps = np.array([200.0, 400.0, np.nan, 800.0, 1000.0])
        fluids = np.array([700.0, 500.0, np.nan, 200.0, 100.0])
        df = make_merged_dataframe("WELL-NAN", bhps, fluids)

        result = estimate_reservoir_pressure(df)
        assert pd.notna(result["Optimal_RP"].dropna().iloc[0])


# ===================================================================
# 4. compute_vogel_coefficients
# ===================================================================


class TestComputeVogelCoefficients:
    """Tests for compute_vogel_coefficients."""

    def _make_merged_with_rp(
        self, well_name, pres, qmax, bhp_range, n_points=5, water_frac=0.3
    ):
        """Create merged data with Optimal_RP already set."""
        bhps = np.linspace(bhp_range[0], bhp_range[1], n_points)
        fluids = make_vogel_points(pres, qmax, bhps)
        df = make_merged_dataframe(
            well_name, bhps, fluids, water_fracs=[water_frac] * n_points
        )
        df["Optimal_RP"] = pres
        return df

    def test_output_columns_exist(self):
        """Output DataFrame should contain all expected columns."""
        df = self._make_merged_with_rp("WELL-A", 1500.0, 800.0, (300, 1200))
        result = compute_vogel_coefficients(df)

        expected_cols = [
            "Well", "ResP", "QMax_recent", "QMax_lowest_bhp",
            "QMax_median", "qwf", "pwf", "form_wc", "num_tests",
            "most_recent_date", "R2",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_qmax_values_positive(self):
        """All QMax values should be positive for valid data."""
        df = self._make_merged_with_rp("WELL-A", 1500.0, 800.0, (300, 1200))
        result = compute_vogel_coefficients(df)

        assert len(result) == 1
        row = result.iloc[0]
        assert row["QMax_recent"] > 0
        assert row["QMax_lowest_bhp"] > 0
        assert row["QMax_median"] > 0

    def test_resp_modifier_shifts_resp(self):
        """resp_modifier should shift ResP by the given amount."""
        pres = 1500.0
        df = self._make_merged_with_rp("WELL-A", pres, 800.0, (300, 1200))

        modifier = 100
        result = compute_vogel_coefficients(df, resp_modifier=modifier)

        assert result.iloc[0]["ResP"] == pytest.approx(pres + modifier)

    def test_resp_modifier_zero_gives_base_pres(self):
        """With resp_modifier=0, ResP should equal the Optimal_RP."""
        pres = 1400.0
        df = self._make_merged_with_rp("WELL-A", pres, 700.0, (300, 1100))

        result = compute_vogel_coefficients(df, resp_modifier=0)
        assert result.iloc[0]["ResP"] == pytest.approx(pres)

    def test_form_wc_computed_from_water_vol(self):
        """form_wc should be WtWaterVol / WtTotalFluid from most recent test."""
        water_frac = 0.45
        df = self._make_merged_with_rp(
            "WELL-WC", 1500.0, 800.0, (300, 1200), water_frac=water_frac
        )

        result = compute_vogel_coefficients(df)
        assert result.iloc[0]["form_wc"] == pytest.approx(water_frac, abs=0.01)

    def test_num_tests_matches_input_rows(self):
        """num_tests should match the number of valid input rows."""
        n_points = 7
        df = self._make_merged_with_rp(
            "WELL-N", 1500.0, 800.0, (300, 1200), n_points=n_points
        )

        result = compute_vogel_coefficients(df)
        assert result.iloc[0]["num_tests"] == n_points

    def test_r2_for_perfect_data_is_high(self):
        """R^2 should be close to 1.0 for perfect Vogel data."""
        df = self._make_merged_with_rp("WELL-R2", 1500.0, 800.0, (300, 1200), n_points=6)

        result = compute_vogel_coefficients(df)
        assert result.iloc[0]["R2"] >= 0.99

    def test_multiple_wells(self):
        """compute_vogel_coefficients should produce one row per well."""
        df_a = self._make_merged_with_rp("WELL-A", 1200.0, 500.0, (200, 900))
        df_b = self._make_merged_with_rp("WELL-B", 1600.0, 900.0, (300, 1200))
        df = pd.concat([df_a, df_b], ignore_index=True)

        result = compute_vogel_coefficients(df)
        assert len(result) == 2
        wells = set(result["Well"].tolist())
        assert wells == {"WELL-A", "WELL-B"}

    def test_most_recent_date_populated(self):
        """most_recent_date should be a valid datetime."""
        df = self._make_merged_with_rp("WELL-D", 1500.0, 800.0, (300, 1200))
        result = compute_vogel_coefficients(df)
        assert pd.notna(result.iloc[0]["most_recent_date"])

    def test_default_wc_when_no_water_column(self):
        """If WtWaterVol column is absent, form_wc should default to 0.5."""
        bhps = np.array([300.0, 600.0, 900.0, 1200.0])
        fluids = make_vogel_points(1500.0, 800.0, bhps)
        df = make_merged_dataframe("WELL-NOWC", bhps, fluids)
        df["Optimal_RP"] = 1500.0

        result = compute_vogel_coefficients(df)
        assert result.iloc[0]["form_wc"] == pytest.approx(0.5, abs=0.01)


# ===================================================================
# 5. generate_ipr_curves
# ===================================================================


class TestGenerateIPRCurves:
    """Tests for generate_ipr_curves."""

    def _make_vogel_coeffs(self, wells_data):
        """Create a vogel_coeffs DataFrame for testing."""
        rows = []
        for wd in wells_data:
            rows.append(
                {
                    "Well": wd["Well"],
                    "ResP": wd["ResP"],
                    "QMax_recent": wd.get("QMax_recent", 800.0),
                    "QMax_lowest_bhp": wd.get("QMax_lowest_bhp", 850.0),
                    "QMax_median": wd.get("QMax_median", 820.0),
                    "qwf": wd["qwf"],
                    "pwf": wd["pwf"],
                    "form_wc": wd.get("form_wc", 0.3),
                    "num_tests": wd.get("num_tests", 5),
                    "most_recent_date": pd.Timestamp.now(),
                    "R2": wd.get("R2", 0.95),
                }
            )
        return pd.DataFrame(rows)

    def test_bhp_array_starts_at_zero(self):
        """bhp_array should start at 0."""
        coeffs = self._make_vogel_coeffs(
            [{"Well": "W1", "ResP": 1500, "qwf": 500, "pwf": 600}]
        )
        result = generate_ipr_curves(coeffs)

        assert "W1" in result
        assert result["W1"]["bhp_array"][0] == 0

    def test_bhp_array_ends_below_res_pres(self):
        """bhp_array should not exceed reservoir pressure."""
        res_p = 1500
        coeffs = self._make_vogel_coeffs(
            [{"Well": "W1", "ResP": res_p, "qwf": 500, "pwf": 600}]
        )
        result = generate_ipr_curves(coeffs)

        assert result["W1"]["bhp_array"][-1] < res_p

    def test_bhp_array_step_is_10(self):
        """bhp_array should use 10 psi steps."""
        coeffs = self._make_vogel_coeffs(
            [{"Well": "W1", "ResP": 1500, "qwf": 500, "pwf": 600}]
        )
        result = generate_ipr_curves(coeffs)

        bhp = result["W1"]["bhp_array"]
        diffs = np.diff(bhp)
        assert np.all(diffs == 10)

    def test_fluid_recent_values_positive(self):
        """All fluid_recent values should be positive (or zero at res_pres)."""
        coeffs = self._make_vogel_coeffs(
            [{"Well": "W1", "ResP": 1500, "qwf": 500, "pwf": 600}]
        )
        result = generate_ipr_curves(coeffs)

        fluids = result["W1"]["fluid_recent"]
        assert all(f >= 0 for f in fluids)

    def test_fluid_recent_decreases_with_bhp(self):
        """Fluid rate should decrease as BHP increases (Vogel property)."""
        coeffs = self._make_vogel_coeffs(
            [{"Well": "W1", "ResP": 1500, "qwf": 500, "pwf": 600}]
        )
        result = generate_ipr_curves(coeffs)

        fluids = result["W1"]["fluid_recent"]
        for i in range(len(fluids) - 1):
            assert fluids[i] >= fluids[i + 1]

    def test_res_pres_matches_input(self):
        """res_pres in output should match ResP from input."""
        coeffs = self._make_vogel_coeffs(
            [{"Well": "W1", "ResP": 1500, "qwf": 500, "pwf": 600}]
        )
        result = generate_ipr_curves(coeffs)

        assert result["W1"]["res_pres"] == 1500

    def test_qmax_recent_matches_input(self):
        """qmax_recent in output should match input QMax_recent."""
        coeffs = self._make_vogel_coeffs(
            [{"Well": "W1", "ResP": 1500, "qwf": 500, "pwf": 600, "QMax_recent": 900.0}]
        )
        result = generate_ipr_curves(coeffs)

        assert result["W1"]["qmax_recent"] == 900.0

    def test_multiple_wells_in_output(self):
        """Each well in input should appear in output dict."""
        coeffs = self._make_vogel_coeffs(
            [
                {"Well": "W1", "ResP": 1500, "qwf": 500, "pwf": 600},
                {"Well": "W2", "ResP": 2000, "qwf": 700, "pwf": 800},
            ]
        )
        result = generate_ipr_curves(coeffs)

        assert "W1" in result
        assert "W2" in result

    def test_invalid_resp_skipped(self):
        """Wells with NaN or zero ResP should be skipped."""
        coeffs = self._make_vogel_coeffs(
            [
                {"Well": "W-BAD", "ResP": 0, "qwf": 500, "pwf": 600},
                {"Well": "W-GOOD", "ResP": 1500, "qwf": 500, "pwf": 600},
            ]
        )
        result = generate_ipr_curves(coeffs)

        assert "W-BAD" not in result
        assert "W-GOOD" in result


# ===================================================================
# 6. export_optimization_template
# ===================================================================


class TestExportOptimizationTemplate:
    """Tests for export_optimization_template."""

    def _make_vogel_coeffs_df(self, n_wells=2):
        """Build a simple vogel_coeffs DataFrame."""
        rows = []
        for i in range(n_wells):
            rows.append(
                {
                    "Well": f"WELL-{chr(65 + i)}",
                    "ResP": 1500 + i * 100,
                    "QMax_recent": 800 + i * 50,
                    "QMax_lowest_bhp": 850 + i * 50,
                    "QMax_median": 820 + i * 50,
                    "qwf": 500 + i * 100,
                    "pwf": 600 + i * 50,
                    "form_wc": round(0.3 + i * 0.1, 3),
                    "num_tests": 5 + i,
                    "most_recent_date": pd.Timestamp.now(),
                    "R2": 0.95,
                }
            )
        return pd.DataFrame(rows)

    def test_output_columns_match_template(self):
        """Output should have all the expected columns."""
        coeffs = self._make_vogel_coeffs_df()
        result = export_optimization_template(coeffs, jp_chars_path="__nonexistent__.csv")

        expected_cols = [
            "Well", "res_pres", "form_temp", "JP_TVD", "JP_MD",
            "out_dia", "thick", "casing_od", "casing_thick",
            "form_wc", "form_gor", "field_model", "surf_pres",
            "qwf_bopd", "pwf", "comments",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_one_row_per_well(self):
        """Output should have one row per well in the input."""
        coeffs = self._make_vogel_coeffs_df(n_wells=3)
        result = export_optimization_template(coeffs, jp_chars_path="__nonexistent__.csv")
        assert len(result) == 3

    def test_res_pres_from_vogel_coeffs(self):
        """res_pres should come from the ResP column in vogel_coeffs."""
        coeffs = self._make_vogel_coeffs_df(n_wells=1)
        result = export_optimization_template(coeffs, jp_chars_path="__nonexistent__.csv")
        assert result.iloc[0]["res_pres"] == coeffs.iloc[0]["ResP"]

    def test_qwf_bopd_from_vogel_coeffs(self):
        """qwf_bopd should come from the qwf column in vogel_coeffs."""
        coeffs = self._make_vogel_coeffs_df(n_wells=1)
        result = export_optimization_template(coeffs, jp_chars_path="__nonexistent__.csv")
        assert result.iloc[0]["qwf_bopd"] == coeffs.iloc[0]["qwf"]

    def test_form_wc_from_vogel_coeffs(self):
        """form_wc should come from the vogel_coeffs."""
        coeffs = self._make_vogel_coeffs_df(n_wells=1)
        result = export_optimization_template(coeffs, jp_chars_path="__nonexistent__.csv")
        assert result.iloc[0]["form_wc"] == coeffs.iloc[0]["form_wc"]

    def test_default_field_model_schrader(self):
        """Without jp_chars, field_model should default to Schrader."""
        coeffs = self._make_vogel_coeffs_df(n_wells=1)
        result = export_optimization_template(coeffs, jp_chars_path="__nonexistent__.csv")
        assert result.iloc[0]["field_model"] == "Schrader"

    def test_with_jp_chars_csv(self):
        """When jp_chars.csv is provided, it should use the data from it."""
        coeffs = self._make_vogel_coeffs_df(n_wells=1)
        well_name = coeffs.iloc[0]["Well"]

        jp_chars_data = pd.DataFrame(
            {
                "Well": [well_name],
                "is_sch": [False],
                "form_temp": [170],
                "JP_TVD": [5500],
                "JP_MD": [6000],
                "out_dia": [3.5],
                "thick": [0.254],
            }
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as f:
            jp_chars_data.to_csv(f, index=False)
            tmp_path = f.name

        try:
            result = export_optimization_template(coeffs, jp_chars_path=tmp_path)

            row = result.iloc[0]
            assert row["field_model"] == "Kuparuk"
            assert row["form_temp"] == 170
            assert row["JP_TVD"] == 5500
            assert row["JP_MD"] == 6000
            assert row["out_dia"] == 3.5
            assert row["thick"] == 0.254
        finally:
            os.unlink(tmp_path)

    def test_comments_include_num_tests(self):
        """Comments should include the number of tests."""
        coeffs = self._make_vogel_coeffs_df(n_wells=1)
        result = export_optimization_template(coeffs, jp_chars_path="__nonexistent__.csv")
        num_tests = coeffs.iloc[0]["num_tests"]
        assert str(num_tests) in result.iloc[0]["comments"]

    def test_fixed_defaults(self):
        """Verify hardcoded defaults: casing_od, casing_thick, form_gor, surf_pres."""
        coeffs = self._make_vogel_coeffs_df(n_wells=1)
        result = export_optimization_template(coeffs, jp_chars_path="__nonexistent__.csv")

        row = result.iloc[0]
        assert row["casing_od"] == 6.875
        assert row["casing_thick"] == 0.5
        assert row["form_gor"] == 250
        assert row["surf_pres"] == 210


# ===================================================================
# Integration-style test: full pipeline
# ===================================================================


class TestFullPipeline:
    """Run a full pipeline from synthetic data through to IPR curves."""

    def test_end_to_end(self):
        """Walk through estimate_reservoir_pressure -> compute_vogel_coefficients
        -> generate_ipr_curves and verify consistency."""
        true_pres = 1400.0
        qmax = 700.0
        bhps = np.array([300.0, 500.0, 700.0, 900.0, 1100.0])
        fluids = make_vogel_points(true_pres, qmax, bhps)
        water_fracs = [0.35] * len(bhps)

        df = make_merged_dataframe("WELL-INT", bhps, fluids, water_fracs=water_fracs)

        # Step 1: estimate reservoir pressure
        df_with_rp = estimate_reservoir_pressure(df)
        optimal_rp = df_with_rp["Optimal_RP"].iloc[0]
        assert abs(optimal_rp - true_pres) < 50

        # Step 2: compute Vogel coefficients
        vogel_coeffs = compute_vogel_coefficients(df_with_rp, resp_modifier=0)
        assert len(vogel_coeffs) == 1
        assert vogel_coeffs.iloc[0]["Well"] == "WELL-INT"
        assert vogel_coeffs.iloc[0]["QMax_recent"] > 0
        assert vogel_coeffs.iloc[0]["form_wc"] == pytest.approx(0.35, abs=0.02)

        # Step 3: generate IPR curves
        ipr = generate_ipr_curves(vogel_coeffs)
        assert "WELL-INT" in ipr
        assert ipr["WELL-INT"]["bhp_array"][0] == 0
        assert len(ipr["WELL-INT"]["fluid_recent"]) > 0
        assert all(f >= 0 for f in ipr["WELL-INT"]["fluid_recent"])

        # Step 4: export template
        template = export_optimization_template(
            vogel_coeffs, jp_chars_path="__nonexistent__.csv"
        )
        assert len(template) == 1
        assert template.iloc[0]["Well"] == "WELL-INT"
        assert template.iloc[0]["res_pres"] == vogel_coeffs.iloc[0]["ResP"]
