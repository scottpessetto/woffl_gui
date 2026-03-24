"""Tests for optimization algorithms: milp, mckp, and dispatcher."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from woffl.assembly.network_optimizer import (
    NetworkOptimizer,
    OptimizationResult,
    PowerFluidConstraint,
    WellConfig,
)
from woffl.assembly.optimization_algorithms import (
    mckp_optimization,
    milp_optimization,
    optimize,
)

# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_mock_batch_df(configs):
    """Create a DataFrame mimicking BatchPump.df.

    configs: list of dicts with keys: nozzle, throat, qoil_std, lift_wat, form_wat
    """
    rows = []
    for c in configs:
        rows.append(
            {
                "nozzle": c["nozzle"],
                "throat": c["throat"],
                "qoil_std": c["qoil_std"],
                "lift_wat": c["lift_wat"],
                "form_wat": c["form_wat"],
                "totl_wat": c["lift_wat"] + c["form_wat"],
                "psu_solv": 1100.0,
                "sonic_status": True,
                "mach_te": 1.05,
                "molwr": 0.3,
                "motwr": 0.2,
                "semi": True,
                "error": "",
            }
        )
    return pd.DataFrame(rows)


def _make_optimizer_with_results():
    """Create a NetworkOptimizer with pre-loaded batch results for 2 wells."""
    wells = [
        WellConfig(well_name="WellA", res_pres=1500, form_temp=70, jpump_tvd=4000),
        WellConfig(well_name="WellB", res_pres=1600, form_temp=80, jpump_tvd=4200),
    ]
    pf = PowerFluidConstraint(total_rate=1200, pressure=3000)
    opt = NetworkOptimizer(wells, pf, ["10", "11"], ["A", "B"])

    # Mock batch results for WellA
    mock_bp_a = MagicMock()
    mock_bp_a.wellname = "WellA"
    mock_bp_a.df = _make_mock_batch_df(
        [
            {
                "nozzle": "10",
                "throat": "A",
                "qoil_std": 200,
                "lift_wat": 500,
                "form_wat": 100,
            },
            {
                "nozzle": "10",
                "throat": "B",
                "qoil_std": 180,
                "lift_wat": 400,
                "form_wat": 90,
            },
            {
                "nozzle": "11",
                "throat": "A",
                "qoil_std": 250,
                "lift_wat": 700,
                "form_wat": 120,
            },
            {
                "nozzle": "11",
                "throat": "B",
                "qoil_std": 220,
                "lift_wat": 600,
                "form_wat": 110,
            },
        ]
    )

    # Mock batch results for WellB
    mock_bp_b = MagicMock()
    mock_bp_b.wellname = "WellB"
    mock_bp_b.df = _make_mock_batch_df(
        [
            {
                "nozzle": "10",
                "throat": "A",
                "qoil_std": 150,
                "lift_wat": 450,
                "form_wat": 80,
            },
            {
                "nozzle": "10",
                "throat": "B",
                "qoil_std": 130,
                "lift_wat": 350,
                "form_wat": 70,
            },
            {
                "nozzle": "11",
                "throat": "A",
                "qoil_std": 170,
                "lift_wat": 550,
                "form_wat": 90,
            },
            {
                "nozzle": "11",
                "throat": "B",
                "qoil_std": 160,
                "lift_wat": 500,
                "form_wat": 85,
            },
        ]
    )

    opt.batch_results = {"WellA": mock_bp_a, "WellB": mock_bp_b}
    return opt


# ── optimize dispatcher ────────────────────────────────────────────────────


class TestOptimizeDispatcher:
    def test_milp(self):
        opt = _make_optimizer_with_results()
        results = optimize(opt, method="milp")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_mckp(self):
        opt = _make_optimizer_with_results()
        results = optimize(opt, method="mckp")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_default_is_milp(self):
        opt = _make_optimizer_with_results()
        results = optimize(opt)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_unknown_method_raises(self):
        opt = _make_optimizer_with_results()
        with pytest.raises(ValueError, match="Unknown optimization method"):
            optimize(opt, method="unknown")


# ── milp_optimization ─────────────────────────────────────────────────────


class TestMilpOptimization:
    def test_returns_results(self):
        opt = _make_optimizer_with_results()
        results = milp_optimization(opt)
        assert len(results) > 0
        assert all(isinstance(r, OptimizationResult) for r in results)

    def test_sets_optimization_results(self):
        opt = _make_optimizer_with_results()
        milp_optimization(opt)
        assert opt.optimization_results is not None

    def test_total_pf_within_constraint(self):
        opt = _make_optimizer_with_results()
        results = milp_optimization(opt)
        total_pf = sum(r.allocated_power_fluid for r in results)
        assert total_pf <= opt.power_fluid.total_rate + 1

    def test_no_batch_results_raises(self):
        wells = [WellConfig(well_name="X", res_pres=1500, form_temp=70, jpump_tvd=4000)]
        pf = PowerFluidConstraint(total_rate=5000, pressure=3000)
        opt = NetworkOptimizer(wells, pf, ["10"], ["A"])
        with pytest.raises(ValueError, match="batch simulations"):
            milp_optimization(opt)

    def test_result_oil_rates_positive(self):
        opt = _make_optimizer_with_results()
        results = milp_optimization(opt)
        for r in results:
            assert r.predicted_oil_rate > 0

    def test_one_pump_per_well(self):
        """MILP should assign at most one pump config per well."""
        opt = _make_optimizer_with_results()
        results = milp_optimization(opt)
        well_names = [r.well_name for r in results]
        assert len(well_names) == len(set(well_names))

    def test_tight_budget_picks_best_combo(self):
        """With a tight PF budget, MILP should pick the highest-oil configs that fit."""
        wells = [
            WellConfig(well_name="W1", res_pres=1500, form_temp=70, jpump_tvd=4000),
            WellConfig(well_name="W2", res_pres=1600, form_temp=80, jpump_tvd=4200),
        ]
        # Budget only allows one pump (~500 bbl)
        pf = PowerFluidConstraint(total_rate=550, pressure=3000)
        opt = NetworkOptimizer(wells, pf, ["10", "11"], ["A", "B"])

        mock_bp_1 = MagicMock()
        mock_bp_1.df = _make_mock_batch_df(
            [
                {
                    "nozzle": "10",
                    "throat": "A",
                    "qoil_std": 300,
                    "lift_wat": 500,
                    "form_wat": 100,
                },
                {
                    "nozzle": "10",
                    "throat": "B",
                    "qoil_std": 100,
                    "lift_wat": 200,
                    "form_wat": 50,
                },
            ]
        )
        mock_bp_2 = MagicMock()
        mock_bp_2.df = _make_mock_batch_df(
            [
                {
                    "nozzle": "10",
                    "throat": "A",
                    "qoil_std": 280,
                    "lift_wat": 500,
                    "form_wat": 90,
                },
                {
                    "nozzle": "10",
                    "throat": "B",
                    "qoil_std": 90,
                    "lift_wat": 200,
                    "form_wat": 40,
                },
            ]
        )
        opt.batch_results = {"W1": mock_bp_1, "W2": mock_bp_2}

        results = milp_optimization(opt)
        total_pf = sum(r.allocated_power_fluid for r in results)
        assert total_pf <= 550
        # Should pick W1/10A (300 oil, 500 pf) — best single pump within budget
        assert any(r.well_name == "W1" and r.predicted_oil_rate == 300 for r in results)


# ── mckp_optimization ─────────────────────────────────────────────────────


class TestMckpOptimization:
    def test_returns_results(self):
        opt = _make_optimizer_with_results()
        results = mckp_optimization(opt)
        assert len(results) > 0
        assert all(isinstance(r, OptimizationResult) for r in results)

    def test_sets_optimization_results(self):
        opt = _make_optimizer_with_results()
        mckp_optimization(opt)
        assert opt.optimization_results is not None

    def test_total_pf_within_constraint(self):
        opt = _make_optimizer_with_results()
        results = mckp_optimization(opt)
        total_pf = sum(r.allocated_power_fluid for r in results)
        assert total_pf <= opt.power_fluid.total_rate + 1

    def test_no_batch_results_raises(self):
        wells = [WellConfig(well_name="X", res_pres=1500, form_temp=70, jpump_tvd=4000)]
        pf = PowerFluidConstraint(total_rate=5000, pressure=3000)
        opt = NetworkOptimizer(wells, pf, ["10"], ["A"])
        with pytest.raises(ValueError, match="batch simulations"):
            mckp_optimization(opt)

    def test_result_oil_rates_positive(self):
        opt = _make_optimizer_with_results()
        results = mckp_optimization(opt)
        for r in results:
            assert r.predicted_oil_rate > 0

    def test_one_pump_per_well(self):
        """MCKP should assign at most one pump config per well."""
        opt = _make_optimizer_with_results()
        results = mckp_optimization(opt)
        well_names = [r.well_name for r in results]
        assert len(well_names) == len(set(well_names))

    def test_milp_and_mckp_agree_on_simple_case(self):
        """Both methods should find the same optimum on a simple problem."""
        opt_milp = _make_optimizer_with_results()
        milp_results = milp_optimization(opt_milp)
        milp_oil = sum(r.predicted_oil_rate for r in milp_results)

        opt_mckp = _make_optimizer_with_results()
        mckp_results = mckp_optimization(opt_mckp)
        mckp_oil = sum(r.predicted_oil_rate for r in mckp_results)

        # On a simple 2-well problem, both should find the same optimum
        assert milp_oil == pytest.approx(mckp_oil, rel=0.01)
