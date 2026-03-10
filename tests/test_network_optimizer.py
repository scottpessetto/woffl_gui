"""Tests for WellConfig, PowerFluidConstraint, OptimizationResult, and NetworkOptimizer."""

import pandas as pd
import pytest

from woffl.assembly.network_optimizer import (
    NetworkOptimizer,
    OptimizationResult,
    PowerFluidConstraint,
    WellConfig,
    create_well_template_csv,
    validate_well_config,
)


# ── WellConfig validation ──────────────────────────────────────────────────


class TestWellConfig:
    def test_valid_schrader(self):
        wc = WellConfig(well_name="TestWell", res_pres=1500, form_temp=70, jpump_tvd=4000)
        assert wc.field_model == "Schrader"

    def test_valid_kuparuk(self):
        wc = WellConfig(
            well_name="TestWell", res_pres=2500, form_temp=170,
            jpump_tvd=6500, field_model="Kuparuk",
        )
        assert wc.field_model == "Kuparuk"

    def test_jpump_md_defaults_to_tvd(self):
        wc = WellConfig(well_name="TestWell", res_pres=1500, form_temp=70, jpump_tvd=4000)
        assert wc.jpump_md == 4000

    def test_jpump_md_explicit(self):
        wc = WellConfig(
            well_name="TestWell", res_pres=1500, form_temp=70,
            jpump_tvd=4000, jpump_md=4500,
        )
        assert wc.jpump_md == 4500

    def test_invalid_field_model(self):
        with pytest.raises(ValueError, match="field_model"):
            WellConfig(well_name="X", res_pres=1500, form_temp=70, jpump_tvd=4000, field_model="BadModel")

    def test_res_pres_too_low(self):
        with pytest.raises(ValueError, match="res_pres"):
            WellConfig(well_name="X", res_pres=100, form_temp=70, jpump_tvd=4000)

    def test_res_pres_too_high(self):
        with pytest.raises(ValueError, match="res_pres"):
            WellConfig(well_name="X", res_pres=6000, form_temp=70, jpump_tvd=4000)

    def test_form_temp_too_low(self):
        with pytest.raises(ValueError, match="form_temp"):
            WellConfig(well_name="X", res_pres=1500, form_temp=20, jpump_tvd=4000)

    def test_form_temp_too_high(self):
        with pytest.raises(ValueError, match="form_temp"):
            WellConfig(well_name="X", res_pres=1500, form_temp=400, jpump_tvd=4000)

    def test_jpump_tvd_too_low(self):
        with pytest.raises(ValueError, match="jpump_tvd"):
            WellConfig(well_name="X", res_pres=1500, form_temp=70, jpump_tvd=1000)

    def test_jpump_tvd_too_high(self):
        with pytest.raises(ValueError, match="jpump_tvd"):
            WellConfig(well_name="X", res_pres=1500, form_temp=70, jpump_tvd=9000)

    def test_form_wc_negative(self):
        with pytest.raises(ValueError, match="form_wc"):
            WellConfig(well_name="X", res_pres=1500, form_temp=70, jpump_tvd=4000, form_wc=-0.1)

    def test_form_wc_above_one(self):
        with pytest.raises(ValueError, match="form_wc"):
            WellConfig(well_name="X", res_pres=1500, form_temp=70, jpump_tvd=4000, form_wc=1.1)

    def test_boundary_values_valid(self):
        """Boundary values should be accepted."""
        wc = WellConfig(
            well_name="X", res_pres=400, form_temp=32, jpump_tvd=2500, form_wc=0.0,
        )
        assert wc.res_pres == 400


# ── PowerFluidConstraint validation ─────────────────────────────────────────


class TestPowerFluidConstraint:
    def test_valid(self):
        pf = PowerFluidConstraint(total_rate=5000, pressure=3000)
        assert pf.rho_pf == 62.4

    def test_rate_zero(self):
        with pytest.raises(ValueError, match="total_rate"):
            PowerFluidConstraint(total_rate=0, pressure=3000)

    def test_rate_negative(self):
        with pytest.raises(ValueError, match="total_rate"):
            PowerFluidConstraint(total_rate=-100, pressure=3000)

    def test_pressure_too_low(self):
        with pytest.raises(ValueError, match="pressure"):
            PowerFluidConstraint(total_rate=5000, pressure=500)

    def test_pressure_too_high(self):
        with pytest.raises(ValueError, match="pressure"):
            PowerFluidConstraint(total_rate=5000, pressure=6000)

    def test_rho_too_low(self):
        with pytest.raises(ValueError, match="rho_pf"):
            PowerFluidConstraint(total_rate=5000, pressure=3000, rho_pf=40)

    def test_rho_too_high(self):
        with pytest.raises(ValueError, match="rho_pf"):
            PowerFluidConstraint(total_rate=5000, pressure=3000, rho_pf=80)

    def test_boundary_pressure(self):
        pf = PowerFluidConstraint(total_rate=100, pressure=1000)
        assert pf.pressure == 1000
        pf = PowerFluidConstraint(total_rate=100, pressure=5000)
        assert pf.pressure == 5000

    def test_boundary_rho(self):
        pf = PowerFluidConstraint(total_rate=100, pressure=3000, rho_pf=50)
        assert pf.rho_pf == 50
        pf = PowerFluidConstraint(total_rate=100, pressure=3000, rho_pf=70)
        assert pf.rho_pf == 70


# ── OptimizationResult properties ──────────────────────────────────────────


class TestOptimizationResult:
    @staticmethod
    def _make_result(**overrides):
        defaults = dict(
            well_name="TestWell",
            recommended_nozzle="12",
            recommended_throat="B",
            allocated_power_fluid=500,
            predicted_oil_rate=200,
            predicted_formation_water=100,
            predicted_lift_water=500,
            suction_pressure=1100,
            marginal_oil_rate=0.4,
            sonic_status=True,
            mach_te=1.05,
        )
        defaults.update(overrides)
        return OptimizationResult(**defaults)

    def test_predicted_total_water(self):
        r = self._make_result(predicted_formation_water=100, predicted_lift_water=500)
        assert r.predicted_total_water == 600

    def test_total_watercut(self):
        r = self._make_result(predicted_oil_rate=200, predicted_formation_water=100, predicted_lift_water=500)
        expected = 600 / (200 + 600)  # 0.75
        assert r.total_watercut == pytest.approx(expected)

    def test_total_watercut_zero_rates(self):
        r = self._make_result(predicted_oil_rate=0, predicted_formation_water=0, predicted_lift_water=0)
        assert r.total_watercut == 0.0


# ── NetworkOptimizer basic methods ──────────────────────────────────────────


class TestNetworkOptimizer:
    @staticmethod
    def _make_optimizer():
        wells = [
            WellConfig(well_name="WellA", res_pres=1500, form_temp=70, jpump_tvd=4000),
            WellConfig(well_name="WellB", res_pres=1600, form_temp=80, jpump_tvd=4200),
        ]
        pf = PowerFluidConstraint(total_rate=5000, pressure=3000)
        return NetworkOptimizer(wells, pf, ["10", "11", "12"], ["A", "B", "C"])

    def test_get_well_by_name_found(self):
        opt = self._make_optimizer()
        w = opt.get_well_by_name("WellA")
        assert w is not None
        assert w.well_name == "WellA"

    def test_get_well_by_name_not_found(self):
        opt = self._make_optimizer()
        assert opt.get_well_by_name("NoSuchWell") is None

    def test_calculate_field_metrics_empty(self):
        opt = self._make_optimizer()
        assert opt.calculate_field_metrics() == {}

    def test_calculate_field_metrics_with_results(self):
        opt = self._make_optimizer()
        results = [
            OptimizationResult(
                "WellA", "12", "B", 500, 200, 100, 500, 1100, 0.4, True, 1.05,
            ),
            OptimizationResult(
                "WellB", "11", "A", 400, 150, 80, 400, 1050, 0.3, False, 0.9,
            ),
        ]
        metrics = opt.calculate_field_metrics(results)
        assert metrics["total_oil_rate"] == 350
        assert metrics["num_wells"] == 2
        assert metrics["num_sonic"] == 1

    def test_to_dataframe_empty(self):
        opt = self._make_optimizer()
        df = opt.to_dataframe()
        assert df.empty

    def test_to_dataframe_columns(self):
        opt = self._make_optimizer()
        results = [
            OptimizationResult(
                "WellA", "12", "B", 500, 200, 100, 500, 1100, 0.4, True, 1.05,
            ),
        ]
        df = opt.to_dataframe(results)
        assert "Well" in df.columns
        assert "Oil Rate (BOPD)" in df.columns
        assert len(df) == 1


# ── validate_well_config ───────────────────────────────────────────────────


class TestValidateWellConfig:
    def test_valid_config(self):
        wc = WellConfig(well_name="X", res_pres=1500, form_temp=70, jpump_tvd=4000)
        is_valid, errors = validate_well_config(wc)
        assert is_valid
        assert errors == []

    def test_deep_well_small_tubing_warning(self):
        wc = WellConfig(
            well_name="X", res_pres=1500, form_temp=70, jpump_tvd=6500, tubing_od=2.5,
        )
        is_valid, errors = validate_well_config(wc)
        assert not is_valid
        assert any("tubing" in e.lower() for e in errors)

    def test_high_wc_low_gor_warning(self):
        wc = WellConfig(
            well_name="X", res_pres=1500, form_temp=70, jpump_tvd=4000,
            form_wc=0.96, form_gor=50,
        )
        is_valid, errors = validate_well_config(wc)
        assert not is_valid
        assert any("watercut" in e.lower() for e in errors)


# ── create_well_template_csv ───────────────────────────────────────────────


class TestCreateWellTemplateCSV:
    def test_non_empty(self):
        csv = create_well_template_csv()
        assert len(csv) > 0

    def test_has_header_columns(self):
        csv = create_well_template_csv()
        header = csv.split("\n")[0]
        for col in ["Well", "res_pres", "form_temp", "JP_TVD", "field_model", "qwf_bopd", "pwf"]:
            assert col in header
