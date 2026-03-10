"""Tests for SimulationParams dataclass and module constants."""

import pytest

from woffl.gui.params import NOZZLE_OPTIONS, THROAT_OPTIONS, SimulationParams


class TestModuleConstants:
    def test_nozzle_options(self):
        assert NOZZLE_OPTIONS == ["8", "9", "10", "11", "12", "13", "14", "15"]

    def test_throat_options(self):
        assert THROAT_OPTIONS == ["X", "A", "B", "C", "D", "E"]


class TestSimulationParamsDefaults:
    def test_default_instantiation(self):
        params = SimulationParams()
        assert params is not None

    def test_is_dataclass(self):
        assert hasattr(SimulationParams, "__dataclass_fields__")

    def test_jetpump_defaults(self):
        p = SimulationParams()
        assert p.nozzle_no == "12"
        assert p.area_ratio == "B"
        assert p.ken == 0.03
        assert p.kth == 0.3
        assert p.kdi == 0.4

    def test_pipe_defaults(self):
        p = SimulationParams()
        assert p.tubing_od == 4.5
        assert p.tubing_thickness == 0.5
        assert p.casing_od == 6.875
        assert p.casing_thickness == 0.5

    def test_formation_defaults(self):
        p = SimulationParams()
        assert p.form_wc == 0.50
        assert p.form_gor == 250
        assert p.form_temp == 70
        assert p.field_model == "Schrader"

    def test_well_defaults(self):
        p = SimulationParams()
        assert p.surf_pres == 210
        assert p.jpump_tvd == 4065
        assert p.rho_pf == 62.4
        assert p.ppf_surf == 3168

    def test_inflow_defaults(self):
        p = SimulationParams()
        assert p.qwf == 750
        assert p.pwf == 500
        assert p.pres == 1700

    def test_batch_defaults(self):
        p = SimulationParams()
        assert p.nozzle_batch_options == ["9", "10", "11", "12", "13", "14", "15"]
        assert p.throat_batch_options == ["A", "B", "C", "D"]
        assert p.water_type == "total"
        assert p.marginal_watercut == 0.94

    def test_power_fluid_range_defaults(self):
        p = SimulationParams()
        assert p.power_fluid_min == 1800
        assert p.power_fluid_max == 3600
        assert p.power_fluid_step == 200

    def test_well_selection_defaults(self):
        p = SimulationParams()
        assert p.selected_well == "Custom"
        assert p.well_data is None


class TestSimulationParamsCustom:
    def test_custom_values(self):
        p = SimulationParams(nozzle_no="10", area_ratio="C", pres=2000)
        assert p.nozzle_no == "10"
        assert p.area_ratio == "C"
        assert p.pres == 2000

    def test_well_data_accepts_dict(self):
        data = {"Well": "MPB-28", "res_pres": 1900}
        p = SimulationParams(well_data=data)
        assert p.well_data == data

    def test_mutable_defaults_independent(self):
        p1 = SimulationParams()
        p2 = SimulationParams()
        p1.nozzle_batch_options.append("8")
        assert "8" not in p2.nozzle_batch_options

    def test_throat_batch_independent(self):
        p1 = SimulationParams()
        p2 = SimulationParams()
        p1.throat_batch_options.append("E")
        assert "E" not in p2.throat_batch_options
