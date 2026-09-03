"""Regression tests for the water price in the optimizers.

History: the marginal-watercut knob was collected by the pad pages, stored on
NetworkOptimizer and never read (P0-6); it was then enforced as a PRUNE on
each config's marginal oil-water ratio. The 2026-09 redesign
(docs/optimization_redesign_2026-09.md) replaced the prune with a PRICE: both
MILP and MCKP maximize oil − λ·water over every converged config, with
λ = (1 − wc) / wc when the legacy gate is what the caller supplied. Nothing
is pruned any more; a config that does not pay for its water simply loses
to one that does, and a well with no paying config is shut in (Σ x ≤ 1).
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from woffl.assembly.network_optimizer import (
    NetworkOptimizer,
    PowerFluidConstraint,
    WellConfig,
)
from woffl.assembly.optimization_algorithms import (
    _over_marginal_wc,
    marginal_wc_to_lambda,
    mckp_optimization,
    milp_optimization,
    water_price,
)


def _batch_df(rows):
    """BatchPump.df stand-in. rows: (nozzle, throat, qoil, lift, form, molwr, motwr)."""
    out = []
    for nozzle, throat, qoil, lift, form, molwr, motwr in rows:
        out.append(
            {
                "nozzle": nozzle,
                "throat": throat,
                "qoil_std": qoil,
                "lift_wat": lift,
                "form_wat": form,
                "totl_wat": lift + form,
                "psu_solv": 1100.0,
                "sonic_status": False,
                "mach_te": 0.5,
                "molwr": molwr,
                "motwr": motwr,
                "semi": True,
                "error": "",
            }
        )
    return pd.DataFrame(out)


def _optimizer(rows_by_well, marginal_watercut, total_rate=50000):
    wells = [
        WellConfig(well_name=wn, res_pres=1500, form_temp=70, jpump_tvd=4000)
        for wn in rows_by_well
    ]
    pf = PowerFluidConstraint(total_rate=total_rate, pressure=3000)
    opt = NetworkOptimizer(
        wells, pf, ["10", "11"], ["A", "B"], marginal_watercut=marginal_watercut
    )
    opt.batch_results = {}
    for wn, rows in rows_by_well.items():
        bp = MagicMock()
        bp.wellname = wn
        bp.df = _batch_df(rows)
        opt.batch_results[wn] = bp
    return opt


# WellA: four configs; 10B makes the most oil AND the most priced objective
# at the 0.94 gate's price (λ = 0.0638 BOPD/BPD): 240 − 57.4 = 182.6 vs
# 10A 168.1, 11A 171.7, 11B 163.5. The NaN / negative MARGINAL columns no
# longer matter - the price acts on totals.
WELL_A = [
    ("10", "A", 200.0, 500.0, 100.0, 0.30, 0.25),
    ("10", "B", 240.0, 900.0, 110.0, 0.05, 0.04),
    ("11", "A", 210.0, 600.0, 105.0, np.nan, np.nan),
    ("11", "B", 205.0, 650.0, 102.0, -0.10, -0.10),
]
# WellB: no config pays for its water at the 0.94 price
# (50 − 0.0638·800 = −1.1; 55 − 0.0638·900 = −2.4) -> shut in.
WELL_B = [
    ("10", "A", 50.0, 800.0, 40.0, 0.02, 0.02),
    ("10", "B", 55.0, 900.0, 45.0, 0.01, 0.01),
]

LAM_094 = marginal_wc_to_lambda(0.94)


def _objective(rows, lam, water="lift"):
    best = None
    for nozzle, throat, qoil, lift, form, _m1, _m2 in rows:
        w = lift if water == "lift" else lift + form
        obj = qoil - lam * w
        if best is None or obj > best[0]:
            best = (obj, nozzle, throat)
    return best


class TestOverMarginalWc:
    # the helper survives for callers that still classify a single pump
    def test_conversion_matches_recommender(self):
        assert not _over_marginal_wc(0.30, 0.94)
        assert _over_marginal_wc(0.05, 0.94)

    def test_nan_and_none_fail_open(self):
        assert not _over_marginal_wc(None, 0.5)
        assert not _over_marginal_wc(float("nan"), 0.5)

    def test_nonpositive_marginal_is_over_any_threshold(self):
        assert _over_marginal_wc(0.0, 0.99)
        assert _over_marginal_wc(-0.5, 0.99)


class TestPriceFromGate:
    def test_legacy_gate_maps_to_a_price(self):
        assert marginal_wc_to_lambda(1.0) == 0.0
        assert marginal_wc_to_lambda(0.94) == pytest.approx(0.06 / 0.94)
        assert marginal_wc_to_lambda(0.5) == pytest.approx(1.0)

    def test_explicit_price_wins_over_the_gate(self):
        opt = _optimizer({"WellA": WELL_A}, marginal_watercut=0.94)
        assert water_price(opt) == pytest.approx(LAM_094)
        opt.water_price = 0.5
        assert water_price(opt) == pytest.approx(0.5)


@pytest.mark.parametrize("solve", [milp_optimization, mckp_optimization], ids=["milp", "mckp"])
class TestPricedObjective:
    def test_price_decides_not_a_prune(self, solve):
        opt = _optimizer({"WellA": WELL_A}, marginal_watercut=0.94)
        (r,) = solve(opt)
        _obj, nozzle, throat = _objective(WELL_A, LAM_094)
        assert (r.recommended_nozzle, r.recommended_throat) == (nozzle, throat) == ("10", "B")
        assert opt.mwc_excluded == {}
        assert getattr(opt, "mwc_excluded_wells", []) == []
        assert opt.lambda_used == pytest.approx(LAM_094)

    def test_threshold_1_0_is_free_water(self, solve):
        opt = _optimizer({"WellA": WELL_A}, marginal_watercut=1.0)
        (r,) = solve(opt)
        # λ = 0: the highest-oil config wins
        assert (r.recommended_nozzle, r.recommended_throat) == ("10", "B")
        assert opt.lambda_used == 0.0

    def test_well_with_no_paying_config_is_shut_in(self, solve):
        opt = _optimizer({"WellA": WELL_A, "WellB": WELL_B}, marginal_watercut=0.94)
        results = solve(opt)
        assert [r.well_name for r in results] == ["WellA"]
        # nothing was pruned - the knapsack CHOSE to leave WellB off
        assert opt.mwc_excluded == {}

    def test_steep_price_shuts_every_well_in_without_crashing(self, solve):
        # λ = 1 BOPD/BPD: 200 − 500 < 0 for every WellA config
        opt = _optimizer({"WellA": WELL_A}, marginal_watercut=0.5)
        assert solve(opt) == []

    def test_price_moves_the_pick(self, solve):
        # a stiff enough price prefers the LEAST water config that still pays:
        # at λ = 0.25, 10A = 200 − 125 = 75 beats 10B = 240 − 225 = 15
        opt = _optimizer({"WellA": WELL_A}, marginal_watercut=0.8)  # λ = 0.25
        (r,) = solve(opt)
        assert (r.recommended_nozzle, r.recommended_throat) == ("10", "A")


class TestSolverAgreement:
    def test_milp_and_mckp_pick_the_same_plan(self):
        for wc in (1.0, 0.94, 0.8, 0.5):
            a = _optimizer({"WellA": WELL_A, "WellB": WELL_B}, marginal_watercut=wc)
            b = _optimizer({"WellA": WELL_A, "WellB": WELL_B}, marginal_watercut=wc)
            ra = {(r.well_name, r.recommended_nozzle, r.recommended_throat) for r in milp_optimization(a)}
            rb = {(r.well_name, r.recommended_nozzle, r.recommended_throat) for r in mckp_optimization(b)}
            assert ra == rb, wc


class TestMckpHousekeeping:
    def test_cached_batch_df_not_mutated(self):
        opt = _optimizer({"WellA": WELL_A}, marginal_watercut=0.94)
        before = opt.batch_results["WellA"].df.copy()
        mckp_optimization(opt)
        pd.testing.assert_frame_equal(opt.batch_results["WellA"].df, before)

    def test_marginal_oil_follows_water_key(self):
        # Pre-fix, a totl_wat run still reported lift-water marginals.
        opt = _optimizer({"WellA": WELL_A[:1]}, marginal_watercut=1.0)
        (r_lift,) = mckp_optimization(opt, water_key="lift_wat")
        assert r_lift.marginal_oil_rate == pytest.approx(0.30)
        opt2 = _optimizer({"WellA": WELL_A[:1]}, marginal_watercut=1.0)
        (r_totl,) = mckp_optimization(opt2, water_key="totl_wat")
        assert r_totl.marginal_oil_rate == pytest.approx(0.25)

    def test_unknown_water_key_raises(self):
        opt = _optimizer({"WellA": WELL_A}, marginal_watercut=1.0)
        with pytest.raises(ValueError, match="water_key"):
            mckp_optimization(opt, water_key="bogus")
