"""Independent conservation/limit checks for the production entry balance."""
from copy import deepcopy
import math

import numpy as np
import pytest
from scipy.integrate import quad

from woffl.flow import entry_energy as ee, jetflow as jf, jetplot as jp
from woffl.flow.inflow import InFlow
from woffl.geometry import JetPump
from woffl.pvt import BlackOil, FormWater, FormGas, ResMix
from tests.test_critical_mach_study import IsothermalGas, ConstantInflow


@pytest.mark.parametrize("ken", [0., .03, .4])
def test_production_limit_matches_analytic_isothermal_gas(ken):
    gas, ipr, area = IsothermalGas(), ConstantInflow(), .001
    pcrit = gas.mass_flow/area*math.sqrt((1+ken)/(ee.GC_PSI*gas.b))
    psu, rate, book = ee.suction_limit(100., ken, area, ipr, gas)
    assert psu+14.7 == pytest.approx(math.sqrt(math.e)*pcrit, abs=.01)
    assert book.limit_reason == "energy_minimum"
    assert book.limit_pressure+14.7 == pytest.approx(pcrit, abs=.05)
    assert book.dete_zero()[3] == pytest.approx(1/math.sqrt(1+ken), abs=.0002)
    assert abs(book.minimum_energy) < 1e-6*book.kde[0]


def test_pressure_bound_is_not_a_sonic_limit(monkeypatch):
    from woffl.assembly import solopump as so
    from tests.asm_helper import wbore, profile
    fluid = ResMix(1., 0., BlackOil.schrader(), FormWater.schrader(), FormGas.schrader(), model_as_water=True)
    ipr, pump = InFlow(500., 500., 1700.), JetPump("12", "B")
    _, _, book = ee.suction_limit(100., pump.ken, pump.ate, ipr, fluid)
    assert book.limit_reason == "pressure_bound"
    assert book.limit_pressure == 50.
    # Positive available discharge at a NUMERICAL bound must not say sonic.
    monkeypatch.setattr(so, "_residual_walk_inward", lambda start, *a, **kw: (start, 100., (0., 500., 2000., .01)))
    answer = so.jetpump_solver(210., 100., 3168., pump, wbore, profile, ipr, fluid, FormWater.schrader())
    assert answer[1] is False


def test_two_walks_and_diagnostics_share_full_energy_curve():
    fluid = ResMix(.8, 1200., BlackOil.schrader(), FormWater.schrader(), FormGas.schrader())
    pump, ipr = JetPump("12", "B"), InFlow(100., 500., 1700.)
    args = (1200., 100., pump.ken, pump.ate, ipr, fluid)
    _, operating = jf.throat_entry_zero_tde(*args)
    _, diagnostic = jp.throat_entry_book(*args)
    with pytest.warns(DeprecationWarning):
        _, _, limit = jf.throat_entry_mach_one(*args, mach_crit=2.)
    for name in ("prs", "kde", "ede", "tde", "vel", "rho"):
        assert getattr(operating, name) == getattr(limit, name) == getattr(diagnostic, name)
    assert operating.dete_zero() == limit.dete_zero()
    mass_flux = np.array(operating.rho)*operating.vel
    assert np.ptp(mass_flux) < 1e-10*mass_flux[0]
    np.testing.assert_allclose(operating.kde, (1+pump.ken)*np.array(operating.vel)**2/2, rtol=1e-14)
    # Integrate actual PVT independently of the interpolant's antiderivative.
    physical = deepcopy(fluid)
    integral = quad(lambda p: ee.GC_PSI/physical.condition(p, 100.).rho_mix(),
                    operating.prs[0], operating.prs[-1], epsabs=.005)[0]
    assert operating.ede[-1] == pytest.approx(integral, rel=2e-5)


@pytest.mark.parametrize("mach", [1., 1.1, 1.5, 2., 2.5])
def test_gas_rich_well_remains_choked_and_solvable(mach):
    from woffl.assembly import solopump as so
    from tests.asm_helper import wbore, profile
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        answer = so.jetpump_solver(210., 100., 3168., JetPump("12", "B"), wbore, profile,
            InFlow(100., 500., 1700.),
            ResMix(.8, 1200., BlackOil.schrader(), FormWater.schrader(), FormGas.schrader()),
            FormWater.schrader(), mach_crit=mach)
    assert answer[1] is True
    assert answer[0] == pytest.approx(327.83308, abs=.05)
    assert answer[2] == pytest.approx(106.85, abs=.05)


def test_pvt_refinement_preserves_limit_and_reports_turning_point(monkeypatch):
    values = []
    for step in (10., 5., 2.5):
        monkeypatch.setattr(ee, "PVT_STEP", step)
        psu, _, book = ee.suction_limit(80., .03, JetPump("12", "B").ate,
            InFlow(246., 1049., 1400.),
            ResMix(.894, 600., BlackOil.schrader(), FormWater.schrader(), FormGas.schrader()))
        values.append(psu)
        assert book.limit_reason == "energy_minimum"
        assert abs(book.grad[-1]) < 1e-6
        assert abs(book.minimum_energy/book.kde[0]) < 1e-6
    assert max(values)-min(values) < .02


def test_material_path_reuse_is_scoped_and_does_not_mutate_inputs():
    ipr = InFlow(100., 500., 1700.)
    fluid = ResMix(.8, 600., BlackOil.schrader(), FormWater.schrader(), FormGas.schrader()).condition(800., 100.)
    @ee.scoped_paths
    def operation():
        first = ee.material_path(100., ipr, fluid)
        assert ee.material_path(100., ipr, fluid) is first
        return first
    assert operation() is not operation()
    assert fluid.press == 800.


def test_reachable_limit_does_not_jump_to_a_deeper_energy_minimum():
    from scipy.interpolate import PchipInterpolator
    path = ee.MaterialPath.__new__(ee.MaterialPath)
    path.pressure_max = 950.
    path.grid = np.arange(50., 1000., 100.)
    path.mass_per_rate = 1.
    path.volume = PchipInterpolator(path.grid, [.25, .23, .22, .16, .14, .13, .07, .05, .04, .035])
    path.dv = path.volume.derivative()
    path.work = path.volume.antiderivative()
    balance = path.balance(950., 1., 0., 1/math.sqrt(ee.GC_PSI/.0004))
    pmin, reason = balance.limit()
    assert reason == "energy_minimum"
    assert 640. < pmin < 650.
    # The later, deeper minimum is unreachable along the first stable branch.
    assert balance.energy(345.) < balance.energy(pmin)
    assert balance.derivative(pmin-1) < 0 < balance.derivative(pmin+1)


@pytest.mark.parametrize("wear", [1., 1.1])
def test_solver_batch_and_network_agree_for_identical_physical_inputs(wear):
    from server import schemas
    from server.services import solve, factories, optimizer_runs
    from woffl.assembly.network_optimizer import _simulate_single_well
    params = schemas.SimParams(nozzle_no="12", area_ratio="B", nozzle_batch_options=["12"],
        throat_batch_options=["B"], form_wc=.8, form_gor=1200, qwf=500, pres=1700,
        form_temp=100, ppf_surf=3168, tubing_thickness=.5, nozzle_area_factor=wear)
    config = optimizer_runs._config_from_seeds("Custom", "M", params.model_dump())
    # The API chooses a template by TVD; Network accepts measured depth.
    config.jpump_md = factories.build_sim_objects(params, "Custom")[4].jetpump_md
    single = solve.solve_single("Custom", params)
    batch = solve.run_batch("Custom", params)["rows"][0]
    network = _simulate_single_well(config, 3168., ["12"], ["B"]).df.iloc[0]
    for one, many in (("psu", "psu_solv"), ("qoil_std", "qoil_std"),
                      ("qnz_bwpd", "lift_wat"), ("mach_te", "mach_te")):
        assert single[one] == batch[many] == network[many]
    assert single["sonic_status"] == batch["sonic_status"] == network["sonic_status"]
