"""Independent regressions for the 2026-09-08 fluid-property follow-up."""
import pytest

from server import schemas
from server.services import factories, optimizer_runs, solve
from woffl.assembly.network_optimizer import NetworkOptimizer, PowerFluidConstraint, _simulate_single_well


@pytest.mark.parametrize("rho_pf", [62.4, 66.0])
def test_explicit_lift_density_agrees_across_solve_paths(rho_pf):
    params = schemas.SimParams(nozzle_no="12", area_ratio="B", nozzle_batch_options=["12"],
        throat_batch_options=["B"], form_wc=.8, form_gor=1200, qwf=500, pres=1700,
        form_temp=100, ppf_surf=3168, tubing_thickness=.5, wat_sg=1.15, rho_pf=rho_pf)
    config = optimizer_runs._config_from_seeds("Custom", "M", params.model_dump())
    config.jpump_md = factories.build_sim_objects(params, "Custom")[4].jetpump_md
    single = solve.solve_single("Custom", params)
    batch = solve.run_batch("Custom", params)["rows"][0]
    network = _simulate_single_well(config, 3168., ["12"], ["B"]).df.iloc[0]
    for one, many in (("psu", "psu_solv"), ("qnz_bwpd", "lift_wat"), ("qoil_std", "qoil_std")):
        assert single[one] == batch[many] == network[many]
    *_, mixture, power = NetworkOptimizer._create_well_objects(config)
    assert power is not mixture.wat
    assert power.condition(0, 60).density == pytest.approx(rho_pf)
    assert mixture.wat.wat_sg == 1.15


def test_constraint_density_reaches_workers_without_mutating_input():
    cfg = optimizer_runs._config_from_seeds("Custom", "M", {})
    cfg.rho_pf = None
    opt = NetworkOptimizer([cfg], PowerFluidConstraint(10000, 3000, 67), ["12"], ["B"])
    assert cfg.rho_pf is None
    assert opt.wells[0].rho_pf == 67
    assert NetworkOptimizer._create_well_objects(opt.wells[0])[-1].density == pytest.approx(67)


def test_lift_density_changes_nozzle_prediction():
    base = schemas.SimParams(form_wc=.8, form_gor=600, qwf=800, pres=1700, rho_pf=62.4)
    a = solve.solve_single("Custom", base)
    b = solve.solve_single("Custom", base.model_copy(update={"rho_pf": 67.0}))
    assert abs(a["qnz_bwpd"] - b["qnz_bwpd"]) > 1.0


@pytest.mark.parametrize("p,t,volume", [(3, 300, .00100215168), (80, 300, .000971180894), (3, 500, .00120241800)])
def test_if97_published_liquid_reference(p, t, volume):
    from woffl.pvt.water_properties import liquid_properties
    assert 1/liquid_properties(p, t)[0] == pytest.approx(volume, rel=4e-9)


@pytest.mark.parametrize("t,rho,cp", [(298.15, 998, .889735100), (298.15, 1200, 1.437649467), (373.15, 1000, .307883622)])
def test_iapws_published_viscosity_reference(t, rho, cp):
    from woffl.pvt.water_properties import viscosity_cp
    assert viscosity_cp(t, rho) == pytest.approx(cp, abs=6e-10)


@pytest.mark.parametrize("pressure,temp", [(100, 60), (1500, 100), (5000, 180)])
def test_water_density_derivative_and_standard_mass(pressure, temp):
    from woffl.pvt import FormWater
    water = FormWater(1.04)
    rho = water.condition(pressure, temp).density
    cw = water.compress
    drho = (water.condition(pressure+.01, temp).density - water.condition(pressure-.01, temp).density)/.02
    assert drho/rho == pytest.approx(cw, rel=1e-7)
    assert water.condition(pressure, temp).volume_factor()*rho == pytest.approx(1.04*62.4)


@pytest.mark.parametrize("cap", [0., 50., 150., None])
def test_undersaturated_oil_compression_closes_its_density_derivative(cap):
    from woffl.pvt import BlackOil
    import math
    oil = BlackOil.schrader().condition(2200, 100, rs_max=cap)
    pb = oil.effective_bubblepoint
    previous = None
    for p in [pb+1, pb+100, pb+600]:
        oil.condition(p, 100, rs_max=cap)
        rho, co, rs = oil.density, oil.compress, oil.gas_solubility()
        if previous:
            assert rho > previous
        previous = rho
        plus = oil.condition(p+.01, 100, rs_max=cap).oil_fvf()
        minus = oil.condition(p-.01, 100, rs_max=cap).oil_fvf()
        assert -(math.log(plus)-math.log(minus))/.02 == pytest.approx(co, rel=2e-6)
        if cap is not None:
            assert rs <= cap
    if cap:
        left = oil.condition(pb-1e-4, 100, rs_max=cap).oil_fvf()
        right = oil.condition(pb+1e-4, 100, rs_max=cap).oil_fvf()
        assert left == pytest.approx(right, rel=1e-7)


def test_nozzle_energy_and_discharge_water_mass_are_conserved():
    from scipy.integrate import quad
    from copy import deepcopy
    from woffl.flow import jetflow, singlephase
    from woffl.pvt import BlackOil, FormWater, FormGas, ResMix
    water = FormWater(1.08)
    v, q = jetflow.water_nozzle(4500, 700, 100, .01, .002, water)
    independent = quad(lambda p: 144*32.174/FormWater(1.08).condition(p, 100).density, 700, 4500)[0]
    assert 1.01*v*v/2 == pytest.approx(independent, rel=1e-8)
    assert q*water.density_std == pytest.approx(singlephase.ft3s_to_bpd(.002*v)*water.density)
    inlet = ResMix(.8, 600, BlackOil.schrader(), FormWater(1.02), FormGas.schrader()).condition(1000, 100)
    before = deepcopy(inlet)
    _, fwat, mix = jetflow.throat_mixture(100, q, inlet, water)
    expected_water_mass = singlephase.bpd_to_ft3s(fwat*inlet.wat.density_std + q*water.density_std)
    for pressure in (100, 1000, 3000):
        assert mix.condition(pressure, 100).insitu_mass_flow(100)[1] == pytest.approx(expected_water_mass, rel=1e-12)
    assert inlet.press == before.press
    assert inlet.oil.press == before.oil.press


@pytest.mark.parametrize("flowpath", ["tubing", "annulus"])
def test_lift_column_velocity_preserves_standard_mass(flowpath, monkeypatch):
    from woffl.flow import outflow, singlephase
    from woffl.pvt import FormWater
    from tests.asm_helper import wbore, profile

    water = FormWater(1.04)
    area = wbore.tube_area if flowpath == "tubing" else wbore.ann_area
    # Inspect the mass transported by the velocity supplied to the friction
    # calculation, independent of how its volume conversion is implemented.
    monkeypatch.setattr(singlephase, "diff_press_friction",
                        lambda ff, rho, vel, diameter, length: rho * vel * area)
    mass = outflow.powerfluid_top_down_friction(4000., 180., 3000., water, wbore, profile, flowpath)
    assert mass == pytest.approx(singlephase.bpd_to_ft3s(3000.) * water.density_std, rel=1e-12)
