"""Regression tests for marginal-watercut enforcement in the optimizers (P0-6).

The knob was collected by Step 3 / the pad pages, stored on NetworkOptimizer,
and never read — a dead knob whose SI info-boxes attributed shut-ins to an
impossible cause. Now both MILP and MCKP prune configs whose marginal watercut
(1/(1 + marginal oil-water ratio) — the single-well recommender's conversion)
exceeds the threshold. At the pad pages' default of 1.0 the gate is off and
behavior is identical to the legacy unenforced path.
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
    mckp_optimization,
    milp_optimization,
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


# WellA: 10A economic marginal (0.3 -> wc 0.77); 10B marginal wc 0.952 (over
# 0.94); 11A NaN marginal (fail-open: kept); 11B negative marginal (over).
WELL_A = [
    ("10", "A", 200.0, 500.0, 100.0, 0.30, 0.25),
    ("10", "B", 240.0, 900.0, 110.0, 0.05, 0.04),
    ("11", "A", 210.0, 600.0, 105.0, np.nan, np.nan),
    ("11", "B", 205.0, 650.0, 102.0, -0.10, -0.10),
]
# WellB: every config over the 0.94 threshold.
WELL_B = [
    ("10", "A", 50.0, 800.0, 40.0, 0.02, 0.02),
    ("10", "B", 55.0, 900.0, 45.0, 0.01, 0.01),
]


class TestOverMarginalWc:
    def test_conversion_matches_recommender(self):
        # marginal wc = 1/(1+r): r=0.3 -> 0.769, r=0.05 -> 0.952
        assert not _over_marginal_wc(0.30, 0.94)
        assert _over_marginal_wc(0.05, 0.94)

    def test_nan_and_none_fail_open(self):
        assert not _over_marginal_wc(None, 0.5)
        assert not _over_marginal_wc(float("nan"), 0.5)

    def test_nonpositive_marginal_is_over_any_threshold(self):
        assert _over_marginal_wc(0.0, 0.99)
        assert _over_marginal_wc(-0.5, 0.99)


class TestMilpEnforcement:
    def test_threshold_prunes_uneconomic_configs(self):
        opt = _optimizer({"WellA": WELL_A}, marginal_watercut=0.94)
        results = milp_optimization(opt)
        (r,) = results
        # 10B (highest oil) is pruned (wc 0.952); NaN-marginal 11A is kept and
        # wins as the best remaining config.
        assert (r.recommended_nozzle, r.recommended_throat) == ("11", "A")
        assert opt.mwc_excluded == {"WellA": 2}  # 10B + 11B
        assert opt.mwc_excluded_wells == []

    def test_threshold_1_0_is_legacy_behavior(self):
        opt = _optimizer({"WellA": WELL_A}, marginal_watercut=1.0)
        results = milp_optimization(opt)
        (r,) = results
        # Gate off: the highest-oil config (10B) wins, negative marginal kept.
        assert (r.recommended_nozzle, r.recommended_throat) == ("10", "B")
        assert opt.mwc_excluded == {}

    def test_fully_excluded_well_is_recorded(self):
        opt = _optimizer({"WellA": WELL_A, "WellB": WELL_B}, marginal_watercut=0.94)
        results = milp_optimization(opt)
        assert [r.well_name for r in results] == ["WellA"]
        assert opt.mwc_excluded_wells == ["WellB"]
        assert opt.mwc_excluded["WellB"] == 2


class TestMckpEnforcement:
    def test_threshold_prunes_uneconomic_configs(self):
        opt = _optimizer({"WellA": WELL_A}, marginal_watercut=0.94)
        results = mckp_optimization(opt)
        (r,) = results
        assert (r.recommended_nozzle, r.recommended_throat) == ("11", "A")
        assert opt.mwc_excluded == {"WellA": 2}

    def test_threshold_1_0_is_legacy_behavior(self):
        opt = _optimizer({"WellA": WELL_A}, marginal_watercut=1.0)
        results = mckp_optimization(opt)
        (r,) = results
        assert (r.recommended_nozzle, r.recommended_throat) == ("10", "B")

    def test_fully_excluded_well_skipped_not_crashed(self):
        opt = _optimizer({"WellA": WELL_A, "WellB": WELL_B}, marginal_watercut=0.94)
        results = mckp_optimization(opt)
        assert [r.well_name for r in results] == ["WellA"]
        assert opt.mwc_excluded_wells == ["WellB"]

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
