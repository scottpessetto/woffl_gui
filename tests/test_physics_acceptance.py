"""Physical rules and independent solutions, not regenerated output snapshots."""
from copy import deepcopy
from itertools import product
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("rho,depth", [(62.4, 4000.), (65., 6500.)])
def test_hydrostatic_column_hand_calculation(rho, depth):
    from woffl.flow.singlephase import diff_press_static
    # lbf/ft2 divided by 144 in2/ft2, positive downward head.
    assert diff_press_static(rho, depth) == pytest.approx(rho * depth / 144)


@pytest.mark.parametrize("loss", [0., .01, .2])
def test_nozzle_bernoulli_balance(loss):
    from woffl.flow.jetflow import nozzle_velocity
    rho, inlet, outlet = 63.5, 3200., 600.
    velocity = nozzle_velocity(inlet, outlet, loss, rho)
    # Pressure work equals kinetic energy plus the nozzle loss.
    pressure_work = (inlet-outlet) * 144 * 32.174 / rho
    assert (1+loss) * velocity**2 / 2 == pytest.approx(pressure_work, rel=1e-12)


def test_unit_mach_energy_walks_agree():
    from tools.physics_qualification import energy_closure
    assert energy_closure(1.)["qualified"]


@pytest.mark.parametrize("mach", [1.5, 2., 2.5])
def test_critical_mach_energy_walks_agree(mach):
    from tools.physics_qualification import energy_closure
    assert energy_closure(mach)["qualified"]


def test_throat_energy_refines_under_smaller_pressure_steps(monkeypatch):
    from woffl.flow import jetflow as jf
    from woffl.flow.inflow import InFlow
    from woffl.geometry.jetpump import JetPump
    from woffl.pvt import BlackOil, FormGas, FormWater, ResMix
    fluid = ResMix(.8, 250., BlackOil.schrader(), FormWater.schrader(), FormGas.schrader())
    pump = JetPump("12", "B")
    ipr = InFlow(100., 500., 1700.)
    pressures = []
    for step in [25., 12.5, 6.25, 3.125]:
        from woffl.flow import entry_energy
        monkeypatch.setattr(entry_energy, "PVT_STEP", step)
        _, book = jf.throat_entry_zero_tde(1200., 100., pump.ken, pump.ate, ipr, deepcopy(fluid))
        pressures.append(book.dete_zero()[0])
    assert abs(pressures[-1]-pressures[-2]) < abs(pressures[1]-pressures[0])
    assert abs(pressures[-1]-pressures[-2]) < 1.0


@pytest.mark.parametrize("water_key", ["lift_wat", "totl_wat"])
@pytest.mark.parametrize("price", [0., .25, 1.])
def test_small_allocations_match_exhaustive_enumeration(water_key, price):
    from woffl.assembly.optimization_algorithms import milp_optimization, mckp_optimization
    rng = np.random.default_rng(907)
    for _ in range(8):
        frames = {}
        for name in ["A", "B", "C"]:
            oil = rng.integers(1, 70, 3).astype(float)
            pf = rng.integers(1, 100, 3).astype(float)
            water = rng.integers(0, 50, 3).astype(float)
            frames[name] = pd.DataFrame(dict(nozzle=["10", "11", "12"], throat=["B"]*3,
                qoil_std=oil, lift_wat=pf, form_wat=water, totl_wat=pf+water))
        capacity = 150.
        def performance(name, nozzle, throat):
            selected = frames[name][frames[name].nozzle == nozzle]
            if selected.empty:
                return None
            r = selected.iloc[0]
            return dict(oil_rate=r.qoil_std, lift_water=r.lift_wat, formation_water=r.form_wat,
                total_water=r.totl_wat, suction_pressure=500., sonic_status=False, mach_te=.5,
                marginal_oil_lift_water=1., marginal_oil_total_water=1.)
        feasible = []
        for choices in product([-1, 0, 1, 2], repeat=3):
            oil = water = 0.
            for name, choice in zip(frames, choices):
                if choice >= 0:
                    r = frames[name].iloc[choice]
                    oil += r.qoil_std
                    water += r[water_key]
            if water <= capacity:
                feasible.append((oil-price*water, oil))
        best = max(feasible)
        for solve in [milp_optimization, mckp_optimization]:
            opt = NS(wells=[NS(well_name=n) for n in frames],
                batch_results={n: NS(df=df) for n, df in frames.items()},
                power_fluid=NS(total_rate=capacity), water_price=price,
                get_pump_performance=performance)
            results = solve(opt, water_key)
            oil = sum(r.predicted_oil_rate for r in results)
            water = sum(r.predicted_lift_water + (r.predicted_formation_water if water_key == "totl_wat" else 0.) for r in results)
            assert water <= capacity
            assert oil-price*water == pytest.approx(best[0])
            if price > 0:
                assert oil == pytest.approx(best[1])
