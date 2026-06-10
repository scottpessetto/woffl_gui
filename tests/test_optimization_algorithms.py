"""Tests for optimization_algorithms — water_key stream selection.

POPS pads constrain different water streams: S/H/I pad pumps see lift
water only, M/F/E handle lift + formation. milp_optimization must budget
whichever stream water_key names.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from woffl.assembly.optimization_algorithms import milp_optimization, optimize


def _stub_optimizer(budget: float):
    """Duck-typed NetworkOptimizer with one well and two configs.

    Config A: more oil, lift 500 / total 2500.
    Config B: less oil, lift 300 / total 1800.
    """
    perfs = {
        ("12", "A"): {
            "oil_rate": 100.0,
            "formation_water": 2000.0,
            "lift_water": 500.0,
            "total_water": 2500.0,
            "suction_pressure": 900.0,
            "sonic_status": False,
            "mach_te": 0.5,
            "marginal_oil_lift_water": 0.2,
            "marginal_oil_total_water": 0.04,
        },
        ("12", "B"): {
            "oil_rate": 80.0,
            "formation_water": 1500.0,
            "lift_water": 300.0,
            "total_water": 1800.0,
            "suction_pressure": 950.0,
            "sonic_status": False,
            "mach_te": 0.4,
            "marginal_oil_lift_water": 0.27,
            "marginal_oil_total_water": 0.044,
        },
    }
    df = pd.DataFrame(
        {
            "nozzle": ["12", "12"],
            "throat": ["A", "B"],
            "qoil_std": [100.0, 80.0],
        }
    )
    opt = SimpleNamespace(
        wells=[SimpleNamespace(well_name="MPX-1")],
        batch_results={"MPX-1": SimpleNamespace(df=df)},
        power_fluid=SimpleNamespace(total_rate=budget),
        optimization_results=None,
    )
    opt.get_pump_performance = lambda wn, noz, thr: perfs.get((noz, thr))
    return opt


def test_lift_key_budgets_lift_water_only():
    # Budget 600 admits config A on lift water (500) even though its total
    # water (2500) is huge — formation water must NOT count.
    opt = _stub_optimizer(budget=600.0)
    results = milp_optimization(opt, water_key="lift_wat")
    assert len(results) == 1
    assert results[0].recommended_throat == "A"
    assert results[0].marginal_oil_rate == pytest.approx(0.2)


def test_total_key_budgets_lift_plus_formation():
    # Same configs, budget 2000 on TOTAL water: A (2500) violates, B (1800)
    # fits — the full-POPS constraint changes the chosen pump.
    opt = _stub_optimizer(budget=2000.0)
    results = milp_optimization(opt, water_key="totl_wat")
    assert len(results) == 1
    assert results[0].recommended_throat == "B"
    # marginal oil now reported per barrel of TOTAL water
    assert results[0].marginal_oil_rate == pytest.approx(0.044)


def test_default_water_key_is_lift():
    opt = _stub_optimizer(budget=600.0)
    results = milp_optimization(opt)
    assert results[0].recommended_throat == "A"


def test_unknown_water_key_raises():
    opt = _stub_optimizer(budget=600.0)
    with pytest.raises(ValueError, match="water_key"):
        milp_optimization(opt, water_key="form_wat")


def test_dispatcher_passes_water_key():
    opt = _stub_optimizer(budget=2000.0)
    results = optimize(opt, method="milp", water_key="totl_wat")
    assert results[0].recommended_throat == "B"
