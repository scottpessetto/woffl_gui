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


class TestWaterPriceObjective:
    """docs/optimization_redesign_2026-09.md §1: one λ in the objective of
    BOTH solvers replaces the marginal-WC gate and the parsimony pass."""

    def test_lambda_zero_is_pure_oil(self):
        opt = _stub_optimizer(budget=10_000)
        opt.water_price = 0.0
        res = milp_optimization(opt)
        assert (res[0].recommended_nozzle, res[0].recommended_throat) == ("12", "A")
        assert opt.lambda_used == 0.0

    def test_lambda_prices_the_incremental_barrel(self):
        """A -> B trades 20 BOPD for 200 BPD of lift water: ratio 0.10. Any
        λ above 0.10 must prefer B (parsimony without a threshold), any λ
        below must keep A."""
        opt = _stub_optimizer(budget=10_000)
        opt.water_price = 0.15
        res = milp_optimization(opt)
        assert res[0].recommended_throat == "B"
        opt.water_price = 0.05
        res = milp_optimization(opt)
        assert res[0].recommended_throat == "A"

    def test_lambda_can_shut_a_well_in(self):
        """When no config's oil covers its priced water, the well is shut in."""
        opt = _stub_optimizer(budget=10_000)
        opt.water_price = 1.0  # 100 BOPD never pays for 300+ BPD
        assert milp_optimization(opt) == []

    def test_legacy_marginal_wc_maps_to_the_same_price(self):
        from woffl.assembly.optimization_algorithms import marginal_wc_to_lambda, water_price

        assert marginal_wc_to_lambda(1.0) == 0.0
        assert marginal_wc_to_lambda(0.9) == pytest.approx(1.0 / 9.0)
        opt = _stub_optimizer(budget=10_000)
        opt.marginal_watercut = 0.9  # gate: r < 0.111 excluded -> A (0.2) and B ok
        assert water_price(opt) == pytest.approx(1.0 / 9.0)
        opt.water_price = 0.3
        assert water_price(opt) == 0.3  # explicit price wins

    def test_derive_lambda_is_the_crossing_ratio(self):
        from woffl.assembly.optimization_algorithms import derive_lambda

        # frontier: (300 BPD, 80) then (500 BPD, 100): segments 0.267, 0.10
        opt = _stub_optimizer(budget=0)
        df = pd.DataFrame(
            {"nozzle": ["12", "12"], "throat": ["B", "A"], "qoil_std": [80.0, 100.0], "lift_wat": [300.0, 500.0]}
        )
        lam, slack = derive_lambda({"MPX-1": df}, cap=400.0)
        assert not slack and lam == pytest.approx(0.10)  # the 2nd segment crosses 400
        lam, slack = derive_lambda({"MPX-1": df}, cap=10_000.0)
        assert slack and lam == 0.0

    def test_milp_and_mckp_agree_on_the_priced_objective(self):
        from woffl.assembly.optimization_algorithms import mckp_optimization

        opt = _stub_optimizer(budget=10_000)
        opt.water_price = 0.15
        df = opt.batch_results["MPX-1"].df
        df["lift_wat"] = [500.0, 300.0]
        df["form_wat"] = [2000.0, 1500.0]
        df["totl_wat"] = [2500.0, 1800.0]
        df["error"] = "na"
        a = milp_optimization(opt)
        b = mckp_optimization(opt)
        assert [(r.recommended_nozzle, r.recommended_throat) for r in a] == [
            (r.recommended_nozzle, r.recommended_throat) for r in b
        ]


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
