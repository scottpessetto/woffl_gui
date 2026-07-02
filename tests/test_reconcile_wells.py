"""Tests for reconcile_wells — the per-well configured→simulated→allocated
accounting (P0-5). A well must never silently vanish from a pad/field plan."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from woffl.assembly.network_optimizer import (
    NetworkOptimizer,
    OptimizationResult,
    PowerFluidConstraint,
    WellConfig,
    reconcile_wells,
)


def _df(rows):
    """rows: (nozzle, throat, qoil_std, error)."""
    return pd.DataFrame(
        [
            {
                "nozzle": n,
                "throat": t,
                "qoil_std": q,
                "lift_wat": 500.0,
                "form_wat": 100.0,
                "totl_wat": 600.0,
                "psu_solv": 1100.0,
                "sonic_status": False,
                "mach_te": 0.5,
                "molwr": 0.3,
                "motwr": 0.25,
                "semi": q == q,
                "error": e,
            }
            for n, t, q, e in rows
        ]
    )


def _result(wn):
    return OptimizationResult(
        well_name=wn,
        recommended_nozzle="10",
        recommended_throat="A",
        allocated_power_fluid=500.0,
        predicted_oil_rate=200.0,
        predicted_formation_water=100.0,
        predicted_lift_water=500.0,
        suction_pressure=1100.0,
        marginal_oil_rate=0.3,
        sonic_status=False,
        mach_te=0.5,
    )


def _optimizer(dfs_by_well):
    wells = [
        WellConfig(well_name=wn, res_pres=1500, form_temp=70, jpump_tvd=4000)
        for wn in dfs_by_well
    ]
    pf = PowerFluidConstraint(total_rate=5000, pressure=3000)
    opt = NetworkOptimizer(wells, pf, ["10"], ["A"])
    for wn, df in dfs_by_well.items():
        if df is None:
            continue  # simulate "never ran"
        bp = MagicMock()
        bp.wellname = wn
        bp.df = df
        opt.batch_results[wn] = bp
    return opt


GOOD = _df([("10", "A", 200.0, ""), ("10", "B", 180.0, "")])
ALL_FAILED = _df(
    [
        ("10", "A", np.nan, "ConvergenceError('psu did not converge')"),
        ("10", "B", np.nan, "ThroatEntryNoSolution('no zero crossing')"),
    ]
)
PARTIAL = _df([("10", "A", 150.0, ""), ("10", "B", np.nan, "JetPumpError('pni<=pte')")])


class TestReconcileWells:
    def test_all_allocated(self):
        opt = _optimizer({"W1": GOOD, "W2": PARTIAL})
        recon = reconcile_wells(opt, [_result("W1"), _result("W2")])
        assert list(recon["Status"]) == ["allocated", "allocated"]
        w2 = recon[recon["Well"] == "W2"].iloc[0]
        assert w2["Configs OK"] == 1 and w2["Configs Failed"] == 1

    def test_failed_simulation_named_with_first_error(self):
        opt = _optimizer({"W1": GOOD, "W2": ALL_FAILED})
        recon = reconcile_wells(opt, [_result("W1")])
        w2 = recon[recon["Well"] == "W2"].iloc[0]
        assert w2["Status"] == "failed simulation"
        assert "ConvergenceError" in w2["Detail"]

    def test_milp_implicit_shutin_is_not_allocated(self):
        opt = _optimizer({"W1": GOOD, "W2": GOOD.copy()})
        recon = reconcile_wells(opt, [_result("W1")])
        w2 = recon[recon["Well"] == "W2"].iloc[0]
        assert w2["Status"] == "not allocated"
        assert "budget" in w2["Detail"]

    def test_marginal_wc_exclusion_reported(self):
        opt = _optimizer({"W1": GOOD, "W2": GOOD.copy()})
        opt.mwc_excluded_wells = ["W2"]
        opt.mwc_excluded = {"W2": 2}
        recon = reconcile_wells(opt, [_result("W1")])
        w2 = recon[recon["Well"] == "W2"].iloc[0]
        assert w2["Status"] == "above marginal WC"
        assert "2 viable config(s)" in w2["Detail"]

    def test_mckp_skip_reported(self):
        opt = _optimizer({"W1": GOOD, "W2": GOOD.copy()})
        opt.mckp_skipped = ["W2"]
        recon = reconcile_wells(opt, [_result("W1")])
        assert recon[recon["Well"] == "W2"].iloc[0]["Status"] == "no semi-finalists"

    def test_pre_optimize_view(self):
        opt = _optimizer({"W1": GOOD, "W2": ALL_FAILED})
        recon = reconcile_wells(opt, [])
        statuses = dict(zip(recon["Well"], recon["Status"]))
        assert statuses == {"W1": "simulated", "W2": "failed simulation"}

    def test_never_simulated_reported(self):
        opt = _optimizer({"W1": GOOD, "W2": None})
        recon = reconcile_wells(opt, [_result("W1")])
        assert recon[recon["Well"] == "W2"].iloc[0]["Status"] == "not simulated"

    def test_problem_rows_sort_first(self):
        opt = _optimizer({"A_ok": GOOD, "B_bad": ALL_FAILED})
        recon = reconcile_wells(opt, [_result("A_ok")])
        assert recon.iloc[0]["Well"] == "B_bad"

    def test_every_configured_well_has_a_row(self):
        opt = _optimizer({"W1": GOOD, "W2": ALL_FAILED, "W3": None})
        recon = reconcile_wells(opt, [])
        assert set(recon["Well"]) == {"W1", "W2", "W3"}
