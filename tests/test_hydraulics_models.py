"""Alternative hydraulics: independent limits, domains and real solver dispatch.

These checks establish numerical/implementation consistency, not field accuracy.
"""
from copy import deepcopy
import math

import numpy as np
import pytest
from pydantic import ValidationError

from server import schemas, surface_cache
from server.services import solve
from woffl.assembly import solopump
from woffl.assembly.network_optimizer import WellConfig
from woffl.flow import hydraulics as h, outflow
from woffl.flow.errors import HydraulicsDomainError
from woffl.flow.inflow import InFlow
from woffl.geometry import JetPump, Pipe, PipeInPipe, WellProfile
from woffl.pvt import BlackOil, FormGas, FormWater, ResMix

ALTERNATIVES = ("hagedorn_brown", "drift_flux")


class ConstantLiquid:
    """Incompressible, gas-free limit with independently specified phase data."""
    tension = .004
    def condition(self, pressure, temperature):
        self.press, self.temp = pressure, temperature
        return self
    def insitu_volm_flow(self, rate):
        return 0., rate, 0.  # test-only volume flow, ft3/s
    def rho_two(self): return 62.4, .1
    def visc_two(self): return 1., .01
    def nslh(self): return 1.


@pytest.mark.parametrize("model", ALTERNATIVES)
def test_single_phase_matches_hydrostatics_and_hagen_poiseuille(model):
    """Laminar liquid: dp/dL = 32 mu U / D^2, independent of holdup closures."""
    diameter = 3.5
    area = math.pi * (diameter / 12)**2 / 4
    velocity = .03  # ft/s; Reynolds < 2300
    fluid = ConstantLiquid()
    ds, df, hl = h.alternative_diff_press(100, 80, diameter, area, .004,
                                         100, 75, area*velocity, fluid, model)
    expected_friction = 32*.001*(velocity*.3048)*(100*.3048)/(diameter*.0254)**2 / h.PA_PER_PSI
    assert ds == pytest.approx(62.4*75/144, rel=1e-7)
    assert df == pytest.approx(expected_friction, rel=.001)
    assert hl == 1.
    reverse = h.alternative_diff_press(100, 80, diameter, area, .004,
                                       -100, -75, area*velocity, fluid, model)
    assert reverse == pytest.approx((-ds, -df, hl))
    wide = h.alternative_diff_press(100, 80, diameter, 2*area, .004,
                                    100, 75, area*velocity, fluid, model)
    assert wide[1] == pytest.approx(df/2)  # actual area independent of hydraulic diameter
    assert (fluid.press, fluid.temp) == (100, 80)


def test_griffith_bubble_branch_conserves_flux_with_prescribed_slip():
    vsl, vsg = .3, .01
    hl, bubble = h.hagedorn_brown_holdup(vsl, vsg, .1, 1000, .001, .06, 2e6)
    assert bubble and vsl/(vsl+vsg) < hl < 1
    # Griffith's actual gas velocity exceeds actual liquid velocity by 0.8 ft/s.
    assert vsg/(1-hl)-vsl/hl == pytest.approx(.8*.3048, rel=1e-10)


@pytest.mark.parametrize("model", ALTERNATIVES)
def test_gas_acceleration_matches_ideal_gas_momentum_and_restores_pvt(model):
    class IdealGas(ConstantLiquid):
        def rho_two(self): return 62.4, 1.5*(self.press+14.7)/214.7
        def insitu_volm_flow(self, mass_rate): return 0., 0., mass_rate/self.rho_two()[1]
        def nslh(self): return 0.
    fluid = IdealGas()
    area = .07
    ds, _, hl = h.alternative_diff_press(200, 80, 3.5, area, .004, 100, 100,
                                         1.5*area*100, fluid, model)
    # Constant mass flux, rho proportional to absolute pressure: dJ/dP = -J/P.
    denominator = 1-1.5*100**2/(32.174*144*214.7)
    assert ds == pytest.approx((1.5*100/144)/denominator, rel=1e-8)
    assert hl == 0 and (fluid.press, fluid.temp) == (200, 80)
    with pytest.raises(HydraulicsDomainError, match="acceleration limit"):
        h.alternative_diff_press(200, 80, 3.5, area, .004, 100, 100,
                                  1.5*area*1000, fluid, model)
    assert (fluid.press, fluid.temp) == (200, 80)


def test_pan_kutateladze_against_published_figure_and_large_diameter_limit():
    # Pan et al. LBNL-4291E Fig. 1: approximate measured chart locations.
    # Chart-reading tolerance, deliberately not false precision from digitizing.
    for diameter, measured in ((18, 2.43), (56, 3.03), (89, 3.20)):
        assert h.kutateladze(diameter**2) == pytest.approx(measured, abs=.15)
    assert h.kutateladze(1e20) == pytest.approx(.008**-.25, rel=1e-7)
    assert 0 < h.kutateladze(1e-24) < 1e-5


@pytest.mark.parametrize("incline", [0., 30., 60., 90.])
@pytest.mark.parametrize("vsl,vsg", [(1., .001), (.1, 1.), (3., 10.), (.001, 20.)])
def test_drift_flux_holdup_has_physical_phase_fractions(incline, vsl, vsg):
    hl = h.drift_flux_holdup(vsl, vsg, .09, 970, 20, .04, incline)
    assert vsl/(vsl+vsg)-1e-12 <= hl < 1
    assert math.isfinite(vsl/hl) and math.isfinite(vsg/(1-hl))


def test_phase_limits_and_invalid_models_are_explicit():
    assert h.drift_flux_holdup(1, 0, .1, 1000, 10, .05, 90) == 1
    assert h.drift_flux_holdup(0, 1, .1, 1000, 10, .05, 90) == 0
    assert h.hagedorn_brown_holdup(1, 0, .1, 1000, .001, .05, 1e6)[0] == 1
    assert h.hagedorn_brown_holdup(0, 1, .1, 1000, .001, .05, 1e6)[0] == 0
    for model in ("tulsa", "typo"):
        with pytest.raises(HydraulicsDomainError):
            h.validate_model(model)
        with pytest.raises(ValidationError):
            schemas.SimParams(hydraulics_model=model)


@pytest.mark.parametrize("model", ALTERNATIVES)
@pytest.mark.parametrize("changes", [{"height": -25}, {"pin": -14.7}, {"area": 0},
                                    {"qoil_std": -1}, {"length": float("nan")}])
def test_invalid_or_downhill_segments_fail_without_model_fallback(model, changes):
    args = dict(pin=100, tin=80, hyd_dia=3.5, area=.07, abs_ruff=.004,
                length=100, height=50, qoil_std=.1, prop=ConstantLiquid(), model=model)
    args.update(changes)
    with pytest.raises(HydraulicsDomainError):
        h.alternative_diff_press(**args)


def mixture():
    return ResMix(.894, 600, BlackOil.schrader(), FormWater.schrader(), FormGas.schrader())


def well_objects():
    return (PipeInPipe(Pipe(4.5, .5), Pipe(6.875, .5)), WellProfile.schrader(),
            InFlow(246, 1049, 1400), mixture(), FormWater.schrader())


def test_default_beggs_is_unchanged_and_alternatives_change_real_traverse():
    bore, profile, _, fluid, _ = well_objects()
    default = outflow.production_top_down_press(210, 80, 246, deepcopy(fluid), bore, profile)
    explicit = outflow.production_top_down_press(210, 80, 246, deepcopy(fluid), bore, profile, model="beggs")
    for left, right in zip(default, explicit):
        np.testing.assert_array_equal(left, right)
    for model in ALTERNATIVES:
        result = outflow.production_top_down_press(210, 80, 246, deepcopy(fluid), bore, profile, model=model)
        assert np.isfinite(result[1]).all()
        assert np.all(np.diff(result[1]) > 0)
        assert abs(result[1][-1]-default[1][-1]) > 5


@pytest.mark.parametrize("model", ALTERNATIVES)
def test_selected_model_reaches_solver_and_closes_its_discharge_balance(monkeypatch, model):
    seen = []
    original = outflow.production_top_down_press
    def tracked(*args, **kwargs):
        seen.append(kwargs.get("model", "beggs"))
        return original(*args, **kwargs)
    monkeypatch.setattr(outflow, "production_top_down_press", tracked)
    args = (210, 80, 3168, JetPump("12", "C"), *well_objects())
    psu, sonic, oil, _, pf, _ = solopump.jetpump_solver(*args, hydraulics_model=model)
    assert seen and set(seen) == {model}
    assert not sonic and oil > 0 and pf > 0
    residual = solopump.discharge_residual(psu, *args, hydraulics_model=model)[0]
    assert abs(residual) < 5


@pytest.mark.parametrize("model", ["beggs", *ALTERNATIVES])
def test_model_version_and_surface_cache_separate_equal_hardware(model):
    config = WellConfig("Custom", res_pres=1600, form_temp=80, jpump_tvd=4000, hydraulics_model=model)
    bb = deepcopy(config)
    bb.hydraulics_model = "beggs"
    assert (surface_cache._key(config, 3000, ["12"], ["C"]) ==
            surface_cache._key(bb, 3000, ["12"], ["C"])) is (model == "beggs")
    assert schemas.SimParams(hydraulics_model=model).to_simulation_params("Custom").hydraulics_model == model
    assert h.physics_model(model).startswith("entry-energy-v2")


@pytest.mark.parametrize("model", ["beggs", *ALTERNATIVES])
@pytest.mark.parametrize("water_mode", [False, True])
def test_pressure_profile_schema_and_mixture_match_solver(monkeypatch, model, water_mode):
    observed = []
    original = outflow.production_top_down_press
    def tracked(*args, **kwargs):
        result = original(*args, **kwargs)
        observed.append((args[2], args[3].wat.wat_sg, result[1][-1], kwargs.get("model", "beggs")))
        return result
    monkeypatch.setattr(outflow, "production_top_down_press", tracked)
    params = schemas.SimParams(hydraulics_model=model, wat_sg=1.08, rho_pf=60.,
                              model_as_water=water_mode, form_wc=1. if water_mode else .8)
    result = schemas.PressureProfileResponse(**solve.pressure_profile("Custom", params))
    assert result.hydraulics_model == model and result.physics_model == h.physics_model(model)
    assert len(observed) > 1 and {r[3] for r in observed} == {model}
    # Final solver evaluation and plotted return column carry the same mixture/rate.
    assert observed[-1][:3] == pytest.approx(observed[-2][:3])


def test_comparison_scores_common_cases_without_hiding_failed_predictions():
    from tools.hydraulics_benchmark import summarize
    def challenge(predictions):
        return [{"well": "A", "prediction_start": "2026-01-01", "different_size": True,
                 "rows": [{"date": "2026-01-02", "wt_uid": i,
                           "predictions": {m: p for m in ("frozen_composition", "measured_composition")}}
                          for i, p in enumerate(predictions)]}]
    reports = {"beggs": challenge([{"bhp_error": 10}, {"bhp_error": 100}]),
               "drift_flux": challenge([{"bhp_error": 20}, {"error": "no solution"}])}
    summary = summarize(reports)["frozen_composition"]
    assert summary["beggs"]["all"]["solved"] == 2
    assert summary["drift_flux"]["all"]["failed"] == 1
    assert summary["beggs"]["common"]["bhp_error"]["rms"] == 10
    assert summary["drift_flux"]["common"]["bhp_error"]["rms"] == 20
