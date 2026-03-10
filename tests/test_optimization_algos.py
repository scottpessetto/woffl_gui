"""Tests for optimization algorithms: greedy, proportional, and dispatcher."""

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
    greedy_optimization,
    optimize,
    simple_proportional_allocation,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_mock_batch_df(configs):
    """Create a DataFrame mimicking BatchPump.df.

    configs: list of dicts with keys: nozzle, throat, qoil_std, lift_wat, form_wat
    """
    rows = []
    for c in configs:
        rows.append({
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
        })
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
    mock_bp_a.df = _make_mock_batch_df([
        {"nozzle": "10", "throat": "A", "qoil_std": 200, "lift_wat": 500, "form_wat": 100},
        {"nozzle": "10", "throat": "B", "qoil_std": 180, "lift_wat": 400, "form_wat": 90},
        {"nozzle": "11", "throat": "A", "qoil_std": 250, "lift_wat": 700, "form_wat": 120},
        {"nozzle": "11", "throat": "B", "qoil_std": 220, "lift_wat": 600, "form_wat": 110},
    ])

    # Mock batch results for WellB
    mock_bp_b = MagicMock()
    mock_bp_b.df = _make_mock_batch_df([
        {"nozzle": "10", "throat": "A", "qoil_std": 150, "lift_wat": 450, "form_wat": 80},
        {"nozzle": "10", "throat": "B", "qoil_std": 130, "lift_wat": 350, "form_wat": 70},
        {"nozzle": "11", "throat": "A", "qoil_std": 170, "lift_wat": 550, "form_wat": 90},
        {"nozzle": "11", "throat": "B", "qoil_std": 160, "lift_wat": 500, "form_wat": 85},
    ])

    opt.batch_results = {"WellA": mock_bp_a, "WellB": mock_bp_b}
    return opt


# ── optimize dispatcher ────────────────────────────────────────────────────


class TestOptimizeDispatcher:
    def test_greedy(self):
        opt = _make_optimizer_with_results()
        results = optimize(opt, method="greedy")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_proportional(self):
        opt = _make_optimizer_with_results()
        results = optimize(opt, method="proportional")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_unknown_method_raises(self):
        opt = _make_optimizer_with_results()
        with pytest.raises(ValueError, match="Unknown optimization method"):
            optimize(opt, method="unknown")


# ── greedy_optimization ─────────────────────────────────────────────────────


class TestGreedyOptimization:
    def test_returns_results(self):
        opt = _make_optimizer_with_results()
        results = greedy_optimization(opt)
        assert len(results) > 0
        assert all(isinstance(r, OptimizationResult) for r in results)

    def test_sets_optimization_results(self):
        opt = _make_optimizer_with_results()
        greedy_optimization(opt)
        assert opt.optimization_results is not None

    def test_total_pf_within_constraint(self):
        opt = _make_optimizer_with_results()
        results = greedy_optimization(opt)
        total_pf = sum(r.allocated_power_fluid for r in results)
        assert total_pf <= opt.power_fluid.total_rate + 1  # small tolerance

    def test_no_batch_results_raises(self):
        wells = [WellConfig(well_name="X", res_pres=1500, form_temp=70, jpump_tvd=4000)]
        pf = PowerFluidConstraint(total_rate=5000, pressure=3000)
        opt = NetworkOptimizer(wells, pf, ["10"], ["A"])
        with pytest.raises(ValueError, match="batch simulations"):
            greedy_optimization(opt)

    def test_result_oil_rates_positive(self):
        opt = _make_optimizer_with_results()
        results = greedy_optimization(opt)
        for r in results:
            assert r.predicted_oil_rate > 0


# ── simple_proportional_allocation ──────────────────────────────────────────


class TestSimpleProportionalAllocation:
    def test_returns_results(self):
        opt = _make_optimizer_with_results()
        results = simple_proportional_allocation(opt)
        assert len(results) > 0

    def test_sets_optimization_results(self):
        opt = _make_optimizer_with_results()
        simple_proportional_allocation(opt)
        assert opt.optimization_results is not None

    def test_no_batch_results_raises(self):
        wells = [WellConfig(well_name="X", res_pres=1500, form_temp=70, jpump_tvd=4000)]
        pf = PowerFluidConstraint(total_rate=5000, pressure=3000)
        opt = NetworkOptimizer(wells, pf, ["10"], ["A"])
        with pytest.raises(ValueError, match="batch simulations"):
            simple_proportional_allocation(opt)

    def test_result_well_names(self):
        opt = _make_optimizer_with_results()
        results = simple_proportional_allocation(opt)
        well_names = {r.well_name for r in results}
        assert well_names.issubset({"WellA", "WellB"})
