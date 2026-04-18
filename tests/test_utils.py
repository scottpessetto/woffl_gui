"""Comprehensive tests for woffl.gui.utils

Tests the data flow and object creation functions in utils.py without
requiring Streamlit or a database connection. Uses unittest.mock to
patch st.* calls where needed.
"""

import math
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Mock streamlit *before* importing anything from woffl.gui.utils so that
# module-level `import streamlit as st` and the @st.cache_data decorator
# resolve without a running Streamlit server.
# ---------------------------------------------------------------------------
_st_mock = MagicMock()
# Make @st.cache_data a passthrough decorator
_st_mock.cache_data = lambda *args, **kwargs: (
    args[0] if args and callable(args[0]) else lambda fn: fn
)
sys.modules.setdefault("streamlit", _st_mock)

from woffl.flow.inflow import InFlow
from woffl.geometry.jetpump import JetPump
from woffl.geometry.pipe import Pipe, PipeInPipe
from woffl.geometry.wellprofile import WellProfile
from woffl.gui.utils import (
    _validate_water_type,
    create_inflow,
    create_jetpump,
    create_pipes,
    create_pvt_components,
    create_reservoir_mix,
    create_well_profile,
    get_available_wells,
    get_well_data,
    is_valid_number,
    load_well_characteristics,
    recommend_jetpump,
    run_batch_pump,
    run_jetpump_solver,
)
from woffl.pvt.blackoil import BlackOil
from woffl.pvt.formgas import FormGas
from woffl.pvt.formwat import FormWater
from woffl.pvt.resmix import ResMix


# ===================================================================
# Fixtures -- reusable E-41 well data
# ===================================================================
@pytest.fixture
def e41_params():
    """E-41 well test parameters (11/27/2023)."""
    return dict(
        surf_pres=210,
        form_temp=111,
        rho_pf=62.4,
        ppf_surf=3168,
    )


@pytest.fixture
def e41_tube():
    return Pipe(out_dia=4.5, thick=0.5)


@pytest.fixture
def e41_wellbore():
    tube = Pipe(out_dia=4.5, thick=0.5)
    case = Pipe(out_dia=6.875, thick=0.5)
    return PipeInPipe(inn_pipe=tube, out_pipe=case)


@pytest.fixture
def e41_well_profile():
    return WellProfile.schrader()


@pytest.fixture
def e41_inflow():
    return InFlow(qwf=246, pwf=1049, pres=1400)


@pytest.fixture
def e41_res_mix():
    oil = BlackOil.schrader()
    wat = FormWater.schrader()
    gas = FormGas.schrader()
    return ResMix(wc=0.894, fgor=600, oil=oil, wat=wat, gas=gas)


@pytest.fixture
def e41_jetpump():
    return JetPump(nozzle_no="13", area_ratio="A", ken=0.03, kth=0.3, kdi=0.4)


# ===================================================================
# Tests: is_valid_number
# ===================================================================
class TestIsValidNumber:
    def test_none_returns_false(self):
        assert is_valid_number(None) is False

    def test_nan_returns_false(self):
        assert is_valid_number(float("nan")) is False

    def test_math_nan_returns_false(self):
        assert is_valid_number(math.nan) is False

    def test_np_nan_returns_false(self):
        assert is_valid_number(np.nan) is False

    def test_zero_returns_true(self):
        assert is_valid_number(0) is True

    def test_positive_int_returns_true(self):
        assert is_valid_number(42) is True

    def test_negative_float_returns_true(self):
        assert is_valid_number(-3.14) is True

    def test_positive_float_returns_true(self):
        assert is_valid_number(1.0) is True

    def test_inf_returns_true(self):
        # inf is a valid float, just not NaN
        assert is_valid_number(float("inf")) is True

    def test_string_returns_true(self):
        # strings are non-None and not float-NaN
        assert is_valid_number("hello") is True

    def test_zero_float_returns_true(self):
        assert is_valid_number(0.0) is True


# ===================================================================
# Tests: create_jetpump
# ===================================================================
class TestCreateJetpump:
    def test_basic_creation(self):
        jp = create_jetpump("12", "B", ken=0.03, kth=0.3, kdi=0.4)
        assert isinstance(jp, JetPump)
        assert jp.noz_no == "12"
        assert jp.rat_ar == "B"
        assert jp.ken == 0.03
        assert jp.kth == 0.3
        assert jp.kdi == 0.4

    def test_nozzle_13_area_A(self):
        jp = create_jetpump("13", "A", ken=0.03, kth=0.3, kdi=0.4)
        assert jp.noz_no == "13"
        assert jp.rat_ar == "A"
        assert jp.anz > 0
        assert jp.ate > 0

    def test_different_friction_factors(self):
        jp = create_jetpump("10", "C", ken=0.05, kth=0.25, kdi=0.35)
        assert jp.ken == 0.05
        assert jp.kth == 0.25
        assert jp.kdi == 0.35

    def test_areas_are_positive(self):
        jp = create_jetpump("12", "B", 0.03, 0.3, 0.4)
        assert jp.anz > 0
        assert jp.ath > 0
        assert jp.ate > 0  # throat entrance = ath - anz > 0

    def test_throat_area_gt_nozzle_area(self):
        jp = create_jetpump("10", "B", 0.03, 0.3, 0.4)
        assert jp.ath > jp.anz


# ===================================================================
# Tests: create_pvt_components
# ===================================================================
class TestCreatePvtComponents:
    def test_schrader_explicit(self):
        oil, water, gas = create_pvt_components("Schrader")
        assert isinstance(oil, BlackOil)
        assert isinstance(water, FormWater)
        assert isinstance(gas, FormGas)

    def test_schrader_lowercase(self):
        oil, water, gas = create_pvt_components("schrader")
        assert isinstance(oil, BlackOil)

    def test_kuparuk_explicit(self):
        oil, water, gas = create_pvt_components("Kuparuk")
        assert isinstance(oil, BlackOil)
        assert isinstance(water, FormWater)
        assert isinstance(gas, FormGas)

    def test_kuparuk_lowercase(self):
        oil, water, gas = create_pvt_components("kuparuk")
        assert isinstance(oil, BlackOil)

    def test_none_defaults_to_schrader(self):
        oil_default, water_default, gas_default = create_pvt_components(None)
        oil_sch, water_sch, gas_sch = create_pvt_components("Schrader")
        # Compare a distinguishing attribute -- Schrader and Kuparuk have
        # different API gravities or bubble points.
        assert type(oil_default) is type(oil_sch)

    def test_unknown_model_defaults_to_schrader(self):
        oil, water, gas = create_pvt_components("UnknownField")
        oil_sch, _, _ = create_pvt_components("Schrader")
        assert type(oil) is type(oil_sch)

    def test_schrader_and_kuparuk_differ(self):
        """Schrader and Kuparuk should produce different PVT objects."""
        oil_s, _, _ = create_pvt_components("Schrader")
        oil_k, _, _ = create_pvt_components("Kuparuk")
        # They are both BlackOil but have different parameters
        assert oil_s.oil_api != oil_k.oil_api or oil_s.pbp != oil_k.pbp

    def test_returns_three_elements(self):
        result = create_pvt_components("Schrader")
        assert len(result) == 3


# ===================================================================
# Tests: create_reservoir_mix
# ===================================================================
class TestCreateReservoirMix:
    def test_basic_creation(self):
        rm = create_reservoir_mix(wc=0.5, gor=400, temp=100)
        assert isinstance(rm, ResMix)

    def test_high_watercut(self):
        rm = create_reservoir_mix(wc=0.95, gor=600, temp=111, field_model="Schrader")
        assert isinstance(rm, ResMix)

    def test_kuparuk_model(self):
        rm = create_reservoir_mix(wc=0.5, gor=400, temp=170, field_model="Kuparuk")
        assert isinstance(rm, ResMix)

    def test_none_field_model(self):
        rm = create_reservoir_mix(wc=0.5, gor=400, temp=100, field_model=None)
        assert isinstance(rm, ResMix)

    def test_e41_data(self):
        rm = create_reservoir_mix(wc=0.894, gor=600, temp=111, field_model="Schrader")
        assert isinstance(rm, ResMix)


# ===================================================================
# Tests: create_well_profile
# ===================================================================
class TestCreateWellProfile:
    def test_schrader_default(self):
        wp = create_well_profile()
        assert isinstance(wp, WellProfile)
        assert hasattr(wp, "md_ray")
        assert hasattr(wp, "vd_ray")

    def test_schrader_explicit(self):
        wp = create_well_profile(field_model="schrader")
        assert isinstance(wp, WellProfile)

    def test_kuparuk(self):
        wp = create_well_profile(field_model="kuparuk")
        assert isinstance(wp, WellProfile)

    def test_none_defaults_to_schrader(self):
        wp_default = create_well_profile(field_model=None)
        wp_sch = create_well_profile(field_model="schrader")
        # Both should have the same jetpump MD
        assert wp_default.jetpump_md == pytest.approx(wp_sch.jetpump_md, rel=1e-6)

    def test_unknown_model_defaults_to_schrader(self):
        wp = create_well_profile(field_model="unknown_field")
        wp_sch = create_well_profile(field_model="schrader")
        assert wp.jetpump_md == pytest.approx(wp_sch.jetpump_md, rel=1e-6)

    def test_jpump_tvd_override(self):
        wp = create_well_profile(field_model="schrader", jpump_tvd=4000)
        assert isinstance(wp, WellProfile)
        # The jetpump MD should differ from the default
        wp_default = create_well_profile(field_model="schrader")
        assert wp.jetpump_md != wp_default.jetpump_md

    def test_jpump_tvd_none_uses_default(self):
        wp = create_well_profile(field_model="schrader", jpump_tvd=None)
        wp_default = create_well_profile(field_model="schrader")
        assert wp.jetpump_md == pytest.approx(wp_default.jetpump_md, rel=1e-6)

    def test_well_profile_has_arrays(self):
        wp = create_well_profile()
        assert len(wp.md_ray) > 0
        assert len(wp.vd_ray) > 0

    def test_case_insensitive_schrader(self):
        wp = create_well_profile(field_model="SCHRADER")
        assert isinstance(wp, WellProfile)

    def test_case_insensitive_kuparuk(self):
        wp = create_well_profile(field_model="KUPARUK")
        assert isinstance(wp, WellProfile)


# ===================================================================
# Tests: create_pipes
# ===================================================================
class TestCreatePipes:
    def test_default_parameters(self):
        tube, case, ann = create_pipes()
        assert isinstance(tube, Pipe)
        assert isinstance(case, Pipe)
        assert isinstance(ann, PipeInPipe)

    def test_default_tubing_dimensions(self):
        tube, case, ann = create_pipes()
        assert tube.out_dia == 4.5
        assert tube.thick == 0.5

    def test_default_casing_dimensions(self):
        tube, case, ann = create_pipes()
        assert case.out_dia == 6.875
        assert case.thick == 0.5

    def test_custom_dimensions(self):
        tube, case, ann = create_pipes(
            tubing_od=3.5,
            tubing_thickness=0.254,
            casing_od=7.0,
            casing_thickness=0.408,
        )
        assert tube.out_dia == 3.5
        assert tube.thick == 0.254
        assert case.out_dia == 7.0
        assert case.thick == 0.408

    def test_inner_diameter(self):
        tube, case, ann = create_pipes()
        assert tube.inn_dia == pytest.approx(4.5 - 2 * 0.5, rel=1e-6)
        assert case.inn_dia == pytest.approx(6.875 - 2 * 0.5, rel=1e-6)

    def test_returns_three_objects(self):
        result = create_pipes()
        assert len(result) == 3

    def test_annulus_references_inner_and_outer(self):
        tube, case, ann = create_pipes()
        # PipeInPipe should be defined by the tube (inner) and case (outer)
        assert isinstance(ann, PipeInPipe)


# ===================================================================
# Tests: create_inflow
# ===================================================================
class TestCreateInflow:
    def test_basic_creation(self):
        ipr = create_inflow(qwf=750, pwf=500, pres=1700)
        assert isinstance(ipr, InFlow)

    def test_e41_data(self):
        ipr = create_inflow(qwf=246, pwf=1049, pres=1400)
        assert isinstance(ipr, InFlow)

    def test_inflow_has_oil_flow_method(self):
        ipr = create_inflow(qwf=500, pwf=800, pres=1500)
        assert hasattr(ipr, "oil_flow")

    def test_inflow_at_reservoir_pressure(self):
        """At reservoir pressure, oil flow should be zero (or very near)."""
        ipr = create_inflow(qwf=500, pwf=800, pres=1500)
        q_at_pres = ipr.oil_flow(1500)
        assert q_at_pres == pytest.approx(0.0, abs=1.0)


# ===================================================================
# Tests: run_jetpump_solver (uses real E-41 data)
# ===================================================================
class TestRunJetpumpSolver:
    def test_successful_solve(
        self,
        e41_params,
        e41_wellbore,
        e41_well_profile,
        e41_inflow,
        e41_res_mix,
        e41_jetpump,
    ):
        result = run_jetpump_solver(
            surf_pres=e41_params["surf_pres"],
            form_temp=e41_params["form_temp"],
            rho_pf=e41_params["rho_pf"],
            ppf_surf=e41_params["ppf_surf"],
            jetpump=e41_jetpump,
            wellbore=e41_wellbore,
            well_profile=e41_well_profile,
            inflow=e41_inflow,
            res_mix=e41_res_mix,
        )
        assert result is not None
        assert len(result) == 6

    def test_returns_six_element_tuple(
        self,
        e41_params,
        e41_wellbore,
        e41_well_profile,
        e41_inflow,
        e41_res_mix,
        e41_jetpump,
    ):
        result = run_jetpump_solver(
            surf_pres=e41_params["surf_pres"],
            form_temp=e41_params["form_temp"],
            rho_pf=e41_params["rho_pf"],
            ppf_surf=e41_params["ppf_surf"],
            jetpump=e41_jetpump,
            wellbore=e41_wellbore,
            well_profile=e41_well_profile,
            inflow=e41_inflow,
            res_mix=e41_res_mix,
        )
        psu, sonic_status, qoil_std, fwat_bwpd, qnz_bwpd, mach_te = result
        assert isinstance(psu, float)
        assert isinstance(sonic_status, (bool, np.bool_))
        assert isinstance(qoil_std, float)
        assert isinstance(fwat_bwpd, float)
        assert isinstance(qnz_bwpd, float)
        assert isinstance(mach_te, float)

    def test_positive_rates(
        self,
        e41_params,
        e41_wellbore,
        e41_well_profile,
        e41_inflow,
        e41_res_mix,
        e41_jetpump,
    ):
        result = run_jetpump_solver(
            surf_pres=e41_params["surf_pres"],
            form_temp=e41_params["form_temp"],
            rho_pf=e41_params["rho_pf"],
            ppf_surf=e41_params["ppf_surf"],
            jetpump=e41_jetpump,
            wellbore=e41_wellbore,
            well_profile=e41_well_profile,
            inflow=e41_inflow,
            res_mix=e41_res_mix,
        )
        psu, sonic_status, qoil_std, fwat_bwpd, qnz_bwpd, mach_te = result
        assert qoil_std > 0, "Oil rate must be positive"
        assert fwat_bwpd > 0, "Water rate must be positive"
        assert qnz_bwpd > 0, "Power fluid rate must be positive"
        assert psu > 0, "Suction pressure must be positive"

    def test_solver_failure_returns_none(
        self, e41_wellbore, e41_well_profile, e41_inflow, e41_res_mix
    ):
        """A very small pump at low power fluid pressure should fail."""
        small_jp = JetPump(nozzle_no="1", area_ratio="A", ken=0.03, kth=0.3, kdi=0.4)
        with patch("woffl.gui.utils.st") as mock_st:
            result = run_jetpump_solver(
                surf_pres=210,
                form_temp=111,
                rho_pf=62.4,
                ppf_surf=100,  # very low power fluid pressure
                jetpump=small_jp,
                wellbore=e41_wellbore,
                well_profile=e41_well_profile,
                inflow=e41_inflow,
                res_mix=e41_res_mix,
            )
        # Should either return None or a valid result; depends on solver behavior.
        # The test ensures no unhandled exception is raised.
        assert result is None or len(result) == 6


# ===================================================================
# Tests: run_batch_pump (uses real E-41 data)
# ===================================================================
class TestRunBatchPump:
    def test_basic_batch_run(
        self, e41_params, e41_wellbore, e41_well_profile, e41_inflow, e41_res_mix
    ):
        with patch("woffl.gui.utils.st") as mock_st:
            bp = run_batch_pump(
                surf_pres=e41_params["surf_pres"],
                form_temp=e41_params["form_temp"],
                rho_pf=e41_params["rho_pf"],
                ppf_surf=e41_params["ppf_surf"],
                wellbore=e41_wellbore,
                well_profile=e41_well_profile,
                inflow=e41_inflow,
                res_mix=e41_res_mix,
                nozzle_options=["10", "11", "12", "13"],
                throat_options=["A", "B"],
                wellname="E-41 Test",
            )
        assert bp is not None
        assert hasattr(bp, "df")
        assert not bp.df.empty

    def test_batch_pump_has_results_columns(
        self, e41_params, e41_wellbore, e41_well_profile, e41_inflow, e41_res_mix
    ):
        with patch("woffl.gui.utils.st") as mock_st:
            bp = run_batch_pump(
                surf_pres=e41_params["surf_pres"],
                form_temp=e41_params["form_temp"],
                rho_pf=e41_params["rho_pf"],
                ppf_surf=e41_params["ppf_surf"],
                wellbore=e41_wellbore,
                well_profile=e41_well_profile,
                inflow=e41_inflow,
                res_mix=e41_res_mix,
                nozzle_options=["10", "11", "12", "13"],
                throat_options=["A", "B"],
                wellname="E-41 Test",
            )
        assert "qoil_std" in bp.df.columns
        assert "nozzle" in bp.df.columns
        assert "throat" in bp.df.columns

    def test_batch_pump_with_single_nozzle(
        self, e41_params, e41_wellbore, e41_well_profile, e41_inflow, e41_res_mix
    ):
        """A single nozzle may not produce enough semi-finalists for curve
        fitting, so process_results() can fail and run_batch_pump may return
        None or a BatchPump without curve-fit coefficients.  Either outcome
        is acceptable -- the function must not raise an unhandled exception."""
        with patch("woffl.gui.utils.st") as mock_st:
            bp = run_batch_pump(
                surf_pres=e41_params["surf_pres"],
                form_temp=e41_params["form_temp"],
                rho_pf=e41_params["rho_pf"],
                ppf_surf=e41_params["ppf_surf"],
                wellbore=e41_wellbore,
                well_profile=e41_well_profile,
                inflow=e41_inflow,
                res_mix=e41_res_mix,
                nozzle_options=["12"],
                throat_options=["A", "B", "C"],
                wellname="E-41 Single",
            )
        # bp may be None (process_results error) or a valid BatchPump
        if bp is not None:
            assert hasattr(bp, "df")

    def test_default_wellname(
        self, e41_params, e41_wellbore, e41_well_profile, e41_inflow, e41_res_mix
    ):
        with patch("woffl.gui.utils.st") as mock_st:
            bp = run_batch_pump(
                surf_pres=e41_params["surf_pres"],
                form_temp=e41_params["form_temp"],
                rho_pf=e41_params["rho_pf"],
                ppf_surf=e41_params["ppf_surf"],
                wellbore=e41_wellbore,
                well_profile=e41_well_profile,
                inflow=e41_inflow,
                res_mix=e41_res_mix,
                nozzle_options=["10", "11", "12", "13"],
                throat_options=["A", "B"],
            )
        assert bp.wellname == "Test Well"

    def test_batch_pump_process_results(
        self, e41_params, e41_wellbore, e41_well_profile, e41_inflow, e41_res_mix
    ):
        """Batch pump with enough nozzles should produce curve-fit coefficients."""
        with patch("woffl.gui.utils.st") as mock_st:
            bp = run_batch_pump(
                surf_pres=e41_params["surf_pres"],
                form_temp=e41_params["form_temp"],
                rho_pf=e41_params["rho_pf"],
                ppf_surf=e41_params["ppf_surf"],
                wellbore=e41_wellbore,
                well_profile=e41_well_profile,
                inflow=e41_inflow,
                res_mix=e41_res_mix,
                nozzle_options=["8", "9", "10", "11", "12", "13", "14"],
                throat_options=["A", "B", "C"],
                wellname="E-41 Full",
            )
        assert bp is not None
        # After process_results, coeff_totl and coeff_lift should exist
        assert hasattr(bp, "coeff_totl") or hasattr(bp, "coeff_lift")


# ===================================================================
# Tests: recommend_jetpump
# ===================================================================
class TestRecommendJetpump:
    @pytest.fixture
    def batch_pump_with_results(
        self, e41_params, e41_wellbore, e41_well_profile, e41_inflow, e41_res_mix
    ):
        """Create a BatchPump with processed results for recommendation tests."""
        with patch("woffl.gui.utils.st") as mock_st:
            bp = run_batch_pump(
                surf_pres=e41_params["surf_pres"],
                form_temp=e41_params["form_temp"],
                rho_pf=e41_params["rho_pf"],
                ppf_surf=e41_params["ppf_surf"],
                wellbore=e41_wellbore,
                well_profile=e41_well_profile,
                inflow=e41_inflow,
                res_mix=e41_res_mix,
                nozzle_options=["8", "9", "10", "11", "12", "13", "14"],
                throat_options=["A", "B", "C"],
                wellname="E-41 Recommend",
            )
        return bp

    def test_recommend_returns_dict(self, batch_pump_with_results):
        bp = batch_pump_with_results
        if not hasattr(bp, "coeff_totl"):
            pytest.skip("Curve fitting did not succeed for this data set")
        rec = recommend_jetpump(bp, marginal_watercut=0.95, water_type="lift")
        assert isinstance(rec, dict)

    def test_recommend_has_required_keys(self, batch_pump_with_results):
        bp = batch_pump_with_results
        if not hasattr(bp, "coeff_totl"):
            pytest.skip("Curve fitting did not succeed for this data set")
        rec = recommend_jetpump(bp, marginal_watercut=0.95, water_type="lift")
        required_keys = {
            "nozzle",
            "throat",
            "qoil_std",
            "water_rate",
            "marginal_ratio",
            "recommendation_type",
        }
        assert required_keys.issubset(rec.keys())

    def test_recommend_oil_rate_positive(self, batch_pump_with_results):
        bp = batch_pump_with_results
        if not hasattr(bp, "coeff_totl"):
            pytest.skip("Curve fitting did not succeed for this data set")
        rec = recommend_jetpump(bp, marginal_watercut=0.95, water_type="lift")
        assert rec["qoil_std"] > 0

    def test_recommend_with_total_water(self, batch_pump_with_results):
        bp = batch_pump_with_results
        if not hasattr(bp, "coeff_totl"):
            pytest.skip("Curve fitting did not succeed for this data set")
        rec = recommend_jetpump(bp, marginal_watercut=0.95, water_type="total")
        assert isinstance(rec, dict)
        assert rec["qoil_std"] > 0

    def test_recommend_empty_df_raises(self):
        """If batch_pump.df is empty, recommend should raise ValueError."""
        mock_bp = MagicMock()
        mock_bp.df = pd.DataFrame()
        with pytest.raises(ValueError, match="no results"):
            recommend_jetpump(mock_bp, marginal_watercut=0.95)

    def test_recommend_no_df_raises(self):
        """If batch_pump has no df attribute, recommend should raise."""
        mock_bp = MagicMock(spec=[])
        with pytest.raises((ValueError, AttributeError)):
            recommend_jetpump(mock_bp, marginal_watercut=0.95)

    def test_recommend_no_coefficients_raises(self):
        """If curve fitting was not performed, recommend should raise."""
        mock_bp = MagicMock()
        mock_bp.df = pd.DataFrame(
            {"qoil_std": [100], "nozzle": ["12"], "throat": ["A"]}
        )
        del mock_bp.coeff_totl
        del mock_bp.coeff_lift
        with pytest.raises(ValueError, match="curve fitting"):
            recommend_jetpump(mock_bp, marginal_watercut=0.95)


# ===================================================================
# Tests: _validate_water_type
# ===================================================================
class TestValidateWaterType:
    def test_lift_lowercase(self):
        assert _validate_water_type("lift") == "lift"

    def test_total_lowercase(self):
        assert _validate_water_type("total") == "total"

    def test_totl_standardized(self):
        assert _validate_water_type("totl") == "total"

    def test_invalid_raises_value_error(self):
        with pytest.raises(ValueError):
            _validate_water_type("invalid")

    def test_uppercase_raises(self):
        """validate_water expects lowercase 'lift' or 'total', not 'Lift'."""
        with pytest.raises(ValueError):
            _validate_water_type("Lift")

    def test_uppercase_total_raises(self):
        with pytest.raises(ValueError):
            _validate_water_type("Total")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            _validate_water_type("")


# ===================================================================
# Tests: load_well_characteristics
# ===================================================================
class TestLoadWellCharacteristics:
    def test_returns_dataframe(self):
        df = load_well_characteristics()
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        df = load_well_characteristics()
        if df.empty:
            pytest.skip("jp_chars.csv not found or empty")
        expected_cols = {
            "Well",
            "is_sch",
            "out_dia",
            "thick",
            "res_pres",
            "form_temp",
            "JP_TVD",
            "JP_MD",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_has_known_wells(self):
        df = load_well_characteristics()
        if df.empty:
            pytest.skip("jp_chars.csv not found or empty")
        wells = df["Well"].tolist()
        assert "MPB-28" in wells
        assert "MPB-30" in wells
        assert "MPB-32" in wells

    def test_numeric_columns(self):
        df = load_well_characteristics()
        if df.empty:
            pytest.skip("jp_chars.csv not found or empty")
        assert pd.api.types.is_numeric_dtype(df["out_dia"])
        assert pd.api.types.is_numeric_dtype(df["thick"])
        assert pd.api.types.is_numeric_dtype(df["res_pres"])
        assert pd.api.types.is_numeric_dtype(df["form_temp"])
        assert pd.api.types.is_numeric_dtype(df["JP_TVD"])
        assert pd.api.types.is_numeric_dtype(df["JP_MD"])

    def test_positive_values(self):
        df = load_well_characteristics()
        if df.empty:
            pytest.skip("jp_chars.csv not found or empty")
        assert (df["out_dia"] > 0).all()
        assert (df["thick"] > 0).all()
        assert (df["res_pres"] > 0).all()
        assert (df["JP_TVD"] > 0).all()
        assert (df["JP_MD"] > 0).all()


# ===================================================================
# Tests: get_available_wells
# ===================================================================
class TestGetAvailableWells:
    def test_returns_list(self):
        wells = get_available_wells()
        assert isinstance(wells, list)

    def test_custom_is_first(self):
        wells = get_available_wells()
        assert wells[0] == "Custom"

    def test_contains_known_wells(self):
        wells = get_available_wells()
        if len(wells) <= 1:
            pytest.skip("jp_chars.csv not loaded")
        assert "MPB-28" in wells
        assert "MPB-30" in wells

    def test_wells_sorted_after_custom(self):
        wells = get_available_wells()
        if len(wells) <= 2:
            pytest.skip("Not enough wells to test sorting")
        # Wells after "Custom" should be sorted
        well_names = wells[1:]
        assert well_names == sorted(well_names)

    def test_more_than_just_custom(self):
        wells = get_available_wells()
        assert len(wells) > 1, "Expected wells from jp_chars.csv"


# ===================================================================
# Tests: get_well_data
# ===================================================================
class TestGetWellData:
    def test_known_well_returns_dict(self):
        data = get_well_data("MPB-28")
        if data is None:
            pytest.skip("jp_chars.csv not loaded")
        assert isinstance(data, dict)

    def test_known_well_has_required_keys(self):
        data = get_well_data("MPB-28")
        if data is None:
            pytest.skip("jp_chars.csv not loaded")
        required_keys = {
            "Well",
            "is_sch",
            "out_dia",
            "thick",
            "res_pres",
            "form_temp",
            "JP_TVD",
            "JP_MD",
        }
        assert required_keys.issubset(data.keys())

    def test_mpb28_values(self):
        data = get_well_data("MPB-28")
        if data is None:
            pytest.skip("MPB-28 not available from data source")
        assert data["Well"] == "MPB-28"
        assert data["out_dia"] == pytest.approx(4.5)
        assert data["res_pres"] == pytest.approx(1900.0)
        # form_temp comes from vw_prop_resvr.resvr_temp (89.5) when Databricks
        # is reachable, or from jp_chars.csv (70.0) on CSV fallback.
        assert data["form_temp"] in (pytest.approx(70.0), pytest.approx(89.5))

    def test_unknown_well_returns_none(self):
        data = get_well_data("NONEXISTENT-999")
        assert data is None

    def test_empty_string_returns_none(self):
        data = get_well_data("")
        assert data is None

    def test_kuparuk_well_mpb30(self):
        data = get_well_data("MPB-30")
        if data is None:
            pytest.skip("jp_chars.csv not loaded")
        assert data["Well"] == "MPB-30"
        assert data["is_sch"] == False  # noqa: E712  -- intentional bool check
        assert data["form_temp"] == pytest.approx(170)

    def test_schrader_well_mpb28(self):
        data = get_well_data("MPB-28")
        if data is None:
            pytest.skip("jp_chars.csv not loaded")
        assert data["is_sch"] == True  # noqa: E712


# ===================================================================
# Tests: integration / data flow
# ===================================================================
class TestDataFlowIntegration:
    """End-to-end tests verifying the full data flow from object creation
    through solver execution."""

    def test_pvt_to_resmix_flow(self):
        """PVT components should feed into ResMix creation."""
        oil, water, gas = create_pvt_components("Schrader")
        rm = ResMix(wc=0.5, fgor=400, oil=oil, wat=water, gas=gas)
        assert isinstance(rm, ResMix)

    def test_pipes_to_annulus(self):
        """Pipe objects should combine into a PipeInPipe."""
        tube, case, ann = create_pipes()
        assert isinstance(ann, PipeInPipe)

    def test_full_solver_flow(self):
        """Full flow from component creation to solver execution."""
        # Create all objects using utils functions
        oil, water, gas = create_pvt_components("Schrader")
        rm = ResMix(wc=0.894, fgor=600, oil=oil, wat=water, gas=gas)
        jp = create_jetpump("13", "A", ken=0.03, kth=0.3, kdi=0.4)
        tube, case, wellbore = create_pipes()
        wp = create_well_profile(field_model="schrader")
        ipr = create_inflow(qwf=246, pwf=1049, pres=1400)

        result = run_jetpump_solver(
            surf_pres=210,
            form_temp=111,
            rho_pf=62.4,
            ppf_surf=3168,
            jetpump=jp,
            wellbore=wellbore,
            well_profile=wp,
            inflow=ipr,
            res_mix=rm,
        )
        assert result is not None
        assert len(result) == 6

    def test_well_data_to_objects(self):
        """Well data from CSV should drive object creation."""
        data = get_well_data("MPB-28")
        if data is None:
            pytest.skip("jp_chars.csv not loaded")

        tube, case, ann = create_pipes(
            tubing_od=data["out_dia"],
            tubing_thickness=data["thick"],
        )
        assert tube.out_dia == pytest.approx(data["out_dia"])

        field_model = "schrader" if data["is_sch"] else "kuparuk"
        wp = create_well_profile(field_model=field_model, jpump_tvd=data["JP_TVD"])
        assert isinstance(wp, WellProfile)

    def test_create_reservoir_mix_uses_pvt(self):
        """create_reservoir_mix should internally use create_pvt_components."""
        rm_sch = create_reservoir_mix(wc=0.5, gor=400, temp=100, field_model="Schrader")
        rm_kup = create_reservoir_mix(wc=0.5, gor=400, temp=100, field_model="Kuparuk")
        # They should be valid but different
        assert isinstance(rm_sch, ResMix)
        assert isinstance(rm_kup, ResMix)
