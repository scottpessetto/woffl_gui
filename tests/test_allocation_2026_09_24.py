"""Patches 47-50 (2026-09-24 allocation review): solver deadline, HiGHS thread
state, single-solve tie-break, linear candidate building and malformed
installed identities. Synthetic frames only; no Databricks, no network."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ortools.sat.python import cp_model

from woffl.assembly import network
from woffl.assembly.network import (
    AllocationDeadline, AllocationError, TIE_BREAK_REL, optimize_jet_pumps, solve_milp_choices,
)
from woffl.assembly.network_optimizer import NetworkOptimizer
from woffl.assembly import optimization_algorithms as oa
from woffl.assembly.pump_candidates import scoped_pumps

ROOT = Path(__file__).resolve().parents[1]


def _frame(options, pump_states=None):
    """options: (oil, water) pairs -> a candidate table (BOPD, BPD)."""
    states = pump_states or ["installed"] + ["replacement"] * (len(options) - 1)
    return pd.DataFrame(dict(nozzle="12", throat=[chr(65 + j) for j in range(len(options))],
                             pump_state=states[:len(options)], qoil_std=[o for o, _ in options],
                             lift_wat=[w for _, w in options], form_wat=0.0,
                             totl_wat=[w for _, w in options]))


def _correlated(wells, options, seed=11):
    """Strongly correlated multiple-choice knapsack (review r06/r07): HiGHS
    cannot prove optimality of the 30x50 case within minutes."""
    rng = np.random.default_rng(seed)
    names, cands = [], []
    for i in range(wells):
        water = np.sort(rng.integers(200, 4000, options)).astype(float) + rng.uniform(0, 1, options).round(6)
        oil = water * 0.1 + rng.integers(0, 4, options)
        cands.append(pd.DataFrame(dict(nozzle=[str(k) for k in range(options)], throat="A", qoil_std=oil,
                                       lift_wat=water, form_wat=0.0, totl_wat=water)))
        names.append(f"W{i}")
    return names, cands, wells * 1500.0 + 0.5


def _optimizer(frames, capacity, price=0.0, fast=True):
    """Duck-typed NetworkOptimizer over batch frames. ``fast`` binds the real
    lookup method (vectorized candidate path); otherwise a wrapper forces the
    per-row lookup that duck-typed optimizers keep."""
    opt = types.SimpleNamespace(
        wells=[types.SimpleNamespace(well_name=name) for name in frames],
        batch_results={name: types.SimpleNamespace(wellname=name, df=df) for name, df in frames.items()},
        power_fluid=types.SimpleNamespace(total_rate=capacity), water_price=price,
        required_wells=set(), optimization_results=None, allocation_status=None)
    opt.get_pump_performance = (types.MethodType(NetworkOptimizer.get_pump_performance, opt) if fast
                                else (lambda *a, **kw: NetworkOptimizer.get_pump_performance(opt, *a, **kw)))
    return opt


def _batch_row(nozzle, throat, state, oil, lift, form=100., psu=500., error="na", **extra):
    return dict(nozzle=nozzle, throat=throat, pump_state=state, qoil_std=oil, lift_wat=lift,
                form_wat=form, totl_wat=(lift + form) if lift == lift and form == form else np.nan,
                psu_solv=psu, sonic_status=False, mach_te=.5, error=error, semi=True, **extra)


# ---------------------------------------------------------------------------
# Patch 47: one wall-clock deadline per allocation call
# ---------------------------------------------------------------------------

def test_deadline_returns_qualified_incumbent_as_feasible_with_gap():
    names, cands, capacity = _correlated(30, 50)
    start = time.perf_counter()
    selected, status = solve_milp_choices(names, cands, capacity, "lift_wat", 0.0, time_limit_s=1.0)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0 + 3.0
    assert status["status"] == "feasible" and status["time_limit_reached"] is True
    assert status["time_limit_s"] == 1.0
    assert 0.0 < status["gap"] < 1e-3
    assert status["objective_bound"] >= status["objective"]
    assert status["selected_water_bpd"] <= capacity + status["resource_tolerance_bpd"]
    assert len(selected) == 30


def test_env_var_sets_the_allocation_budget_through_the_gui_adapter(monkeypatch):
    names, cands, capacity = _correlated(30, 50)
    frames = {name: df.assign(psu_solv=500., sonic_status=False, mach_te=.5, error="na")
              for name, df in zip(names, cands)}
    monkeypatch.setenv(network.TIME_LIMIT_ENV, "1.0")
    opt = _optimizer(frames, capacity)
    start = time.perf_counter()
    results = oa.optimize(opt, method="milp")
    assert time.perf_counter() - start < 1.0 + 3.0
    assert opt.allocation_status["status"] == "feasible"
    assert opt.allocation_status["time_limit_s"] == 1.0
    assert len(results) == 30
    monkeypatch.setenv(network.TIME_LIMIT_ENV, "not a number")
    assert AllocationDeadline().time_limit_s == network.DEFAULT_TIME_LIMIT_S
    assert AllocationDeadline(2.5).time_limit_s == 2.5
    with pytest.raises(ValueError):
        AllocationDeadline(0.0)


def test_milp_receives_remaining_budget_and_no_thread_option(monkeypatch):
    import scipy.optimize
    original = scipy.optimize.milp
    seen = []

    def spy(**kwargs):
        seen.append(dict(kwargs["options"]))
        return original(**kwargs)

    monkeypatch.setattr(scipy.optimize, "milp", spy)
    solve_milp_choices(["A"], [_frame([(100., 100.5)])], 1000., time_limit_s=7.0)
    assert len(seen) == 1
    assert "threads" not in seen[0]
    assert 0 < seen[0]["time_limit"] <= 7.0
    assert seen[0]["mip_rel_gap"] == 0.0 and seen[0]["mip_feasibility_tolerance"] == 1e-9


def test_milp_time_limit_without_incumbent_is_typed_unknown(monkeypatch):
    import scipy.optimize
    monkeypatch.setattr(scipy.optimize, "milp", lambda **kw: types.SimpleNamespace(
        status=1, success=False, x=None,
        message="Time limit reached. (HiGHS Status 13: model_status is Time limit reached)"))
    with pytest.raises(AllocationError) as caught:
        solve_milp_choices(["A"], [_frame([(100., 100.5)])], 1000., time_limit_s=1.0)
    assert caught.value.status == "unknown"
    assert caught.value.details["reason"] == "time_limit"
    assert caught.value.details["time_limit_s"] == 1.0


def test_cp_sat_and_its_milp_fallback_share_one_deadline(monkeypatch):
    budgets = []
    original_solve = cp_model.CpSolver.solve

    def spy_solve(self, model, *a, **kw):
        budgets.append(self.parameters.max_time_in_seconds)
        return original_solve(self, model, *a, **kw)

    monkeypatch.setattr(cp_model.CpSolver, "solve", spy_solve)
    deadline = AllocationDeadline(9.0)
    wells = [types.SimpleNamespace(wellname="A", df=_frame([(100., 100.)]))]
    optimize_jet_pumps(wells, 1000., allow_shutin=True, all_configs=True, deadline=deadline)
    assert len(budgets) == 1 and 0 < budgets[0] <= 9.0

    handed = []
    real_milp = network.solve_milp_choices
    monkeypatch.setattr(network, "solve_milp_choices",
                        lambda *a, **kw: handed.append(kw.get("deadline")) or real_milp(*a, **kw))
    # Fractional water routes the CP request to the MILP, on the same budget.
    fractional = [types.SimpleNamespace(wellname="A", df=_frame([(100., 100.001)]))]
    table = optimize_jet_pumps(fractional, 1000., allow_shutin=True, all_configs=True, deadline=deadline)
    assert handed == [deadline]
    assert table.attrs["allocation_status"]["time_limit_s"] == 9.0


def test_cp_sat_unknown_after_deadline_is_typed_time_limit(monkeypatch):
    monkeypatch.setattr(cp_model.CpSolver, "solve", lambda *a, **kw: cp_model.UNKNOWN)
    deadline = AllocationDeadline(1e-6)
    time.sleep(0.01)
    wells = [types.SimpleNamespace(wellname="A", df=_frame([(100., 100.)]))]
    with pytest.raises(AllocationError) as caught:
        optimize_jet_pumps(wells, 1000., allow_shutin=True, all_configs=True, deadline=deadline)
    assert caught.value.status == "unknown"
    assert caught.value.details["reason"] == "time_limit"


# ---------------------------------------------------------------------------
# Patch 47: HiGHS keeps one process-wide scheduler
# ---------------------------------------------------------------------------

_POISON_PROBE = r"""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
linprog(c=[-1.0, -2.0], A_ub=[[1.0, 1.0]], b_ub=[1.0], bounds=(0, 1), method="highs")
milp(c=np.array([-1.0, -2.0]), constraints=[LinearConstraint(np.array([[1.0, 1.0]]), -np.inf, 1.0)],
     bounds=Bounds(0, 1), integrality=np.ones(2))
from tests.test_allocation_qualification import make_optimizer
from woffl.assembly.optimization_algorithms import mckp_optimization, milp_optimization
for engine in (milp_optimization, mckp_optimization, milp_optimization):
    opt = make_optimizer({"A": [(100.0, 300.001), (150.0, 700.001)], "B": [(80.0, 200.001)]}, capacity=800.0)
    wells = sorted(r.well_name for r in engine(opt))
    assert wells == ["A", "B"], wells
    assert opt.allocation_status["status"] == "optimal", opt.allocation_status
print("ALLOCATORS_OK")
"""


def test_default_thread_highs_solve_first_does_not_poison_allocators():
    """Review r04b: a default-thread HiGHS call before an allocation used to
    make every later allocation MILP fail with 'HiGHS Status 0: Not Set'.
    Fresh interpreter: the HiGHS scheduler is process-global."""
    out = subprocess.run([sys.executable, "-c", _POISON_PROBE], cwd=str(ROOT), capture_output=True,
                         text=True, env={**os.environ, "PYTHONPATH": str(ROOT)}, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "ALLOCATORS_OK" in out.stdout


def test_milp_error_status_is_a_clear_typed_failure(monkeypatch):
    import scipy.optimize
    monkeypatch.setattr(scipy.optimize, "milp", lambda **kw: types.SimpleNamespace(
        status=4, success=False, x=None, message="(HiGHS Status 0: Not Set)"))
    with pytest.raises(AllocationError, match="no plan was produced") as caught:
        solve_milp_choices(["A"], [_frame([(100., 100.5)])], 1000.)
    assert caught.value.status == "error"
    assert caught.value.details["solver_message"] == "(HiGHS Status 0: Not Set)"


# ---------------------------------------------------------------------------
# Patch 48: one solve with bounded tie-break terms
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fraction", [0.0, 0.001])  # CP-SAT (exact) and MILP routes
def test_zero_price_equal_oil_prefers_least_water(fraction):
    """Review r01: at price 0 equal-oil alternatives and zero-oil pumps were
    arbitrary; the least-water plan is now selected."""
    d = fraction
    wells = [types.SimpleNamespace(wellname="A", df=_frame([(200., 900. + d), (200., 300. + d)])),
             types.SimpleNamespace(wellname="Z", df=_frame([(0., 800. + d)])),
             types.SimpleNamespace(wellname="C", df=_frame([(50., 500. + d), (50., 100. + d)]))]
    table = optimize_jet_pumps(wells, 5000., allow_shutin=True, all_configs=True)
    status = table.attrs["allocation_status"]
    assert status["solver"] == ("milp" if d else "cp-sat")
    assert status["oil_bopd"] == 250.
    assert status["selected_water_bpd"] == pytest.approx(400. + 2 * d)
    assert list(table.nozzle) == ["12", "off", "12"]
    assert "less water among equal oil" in status["tie_break"]
    names = [w.wellname for w in wells]
    selected, milp_status = solve_milp_choices(names, [w.df for w in wells], 5000.)
    assert milp_status["selected_water_bpd"] == pytest.approx(400. + 2 * d)
    assert set(selected) == {"A", "C"}


@pytest.mark.parametrize("fraction", [0.0, 0.001])
def test_equal_outcomes_keep_the_installed_pump(fraction):
    frame = pd.DataFrame(dict(nozzle=["13", "12"], throat=["C", "B"], pump_state=["replacement", "installed"],
                              qoil_std=[100., 100.], lift_wat=[500. + fraction] * 2, form_wat=0.,
                              totl_wat=[500. + fraction] * 2))
    table = optimize_jet_pumps([types.SimpleNamespace(wellname="A", df=frame)], 1000.,
                               allow_shutin=True, all_configs=True)
    assert table.pump_state.tolist() == ["installed"]
    selected, _ = solve_milp_choices(["A"], [frame], 1000.)
    assert selected == {"A": (0, 1)}


def test_priced_tie_break_distortion_stays_within_reported_bound():
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csc_array, vstack

    for seed in range(6):
        rng = np.random.default_rng(seed)
        names, cands = [], []
        for i in range(8):
            lift = np.sort(rng.uniform(300, 3500, 10))
            oil = np.maximum(rng.uniform(150, 600) * (1 - np.exp(-rng.uniform(5e-4, 2e-3) * lift))
                             + rng.normal(0, 3, 10), 0)
            cands.append(pd.DataFrame(dict(nozzle=[str(k) for k in range(10)], throat="A", qoil_std=oil,
                                           lift_wat=lift, form_wat=0., totl_wat=lift)))
            names.append(f"W{i}")
        price, capacity = .05, 8 * 1400.
        _, status = solve_milp_choices(names, cands, capacity, "lift_wat", price)
        oil = np.concatenate([df.qoil_std for df in cands])
        water = np.concatenate([df.lift_wat for df in cands])
        rows = np.repeat(np.arange(8), 10)
        A = vstack([csc_array((np.ones(80), (rows, np.arange(80))), shape=(8, 80)), csc_array(water.reshape(1, -1))])
        exact = milp(-(oil - price * water), integrality=np.ones(80), bounds=Bounds(0, 1),
                     constraints=[LinearConstraint(A, np.r_[np.zeros(8), -np.inf], np.r_[np.ones(8), capacity])],
                     options={"mip_rel_gap": 0.0})
        scale = sum(float((df.qoil_std + price * df.lift_wat).max()) for df in cands)
        assert status["status"] == "optimal"
        assert 0 < status["tie_break_bound"] <= 0.75 * TIE_BREAK_REL * scale
        assert status["objective"] >= -exact.fun - status["tie_break_bound"] - 1e-6
        assert status["objective_bound"] >= -exact.fun - 1e-6


def test_cp_sat_tie_break_is_exact_and_single_solve(monkeypatch):
    calls = []
    original = cp_model.CpSolver.solve
    monkeypatch.setattr(cp_model.CpSolver, "solve",
                        lambda self, model, *a, **kw: calls.append(1) or original(self, model, *a, **kw))
    # Priced objective ties (0 each); the tie-break takes the most oil.
    frame = _frame([(80., 80.), (100., 100.), (60., 60.)])
    table = optimize_jet_pumps([types.SimpleNamespace(wellname="W", df=frame)], 1000.,
                               allow_shutin=True, all_configs=True, water_price=1.)
    status = table.attrs["allocation_status"]
    assert calls == [1]
    assert table.qoil_std.tolist() == [100.]
    assert status["solver"] == "cp-sat" and status["status"] == "optimal"
    assert status["tie_break_bound"] == 0.0 and status["quantized_objective"] == 0.0
    assert status["objective_bound"] == pytest.approx(0.01)  # quantization only


# ---------------------------------------------------------------------------
# Patch 49: linear candidate building and separate duplicate accounting
# ---------------------------------------------------------------------------

def _parity_frames():
    rng = np.random.default_rng(24)
    frames = {}
    for w in range(6):
        rows = [_batch_row("12", "B", "installed", 300. + w, 900.)]
        # identical installed/clean outcome -> deduplicated, not excluded
        rows.append(_batch_row("12", "B", "replacement", 300. + w, 900.))
        for k, (nozzle, throat) in enumerate([("11", "A"), ("13", "C"), ("14", "D"), ("12", "C")]):
            rows.append(_batch_row(nozzle, throat, "replacement", float(rng.uniform(50, 400)),
                                   float(rng.uniform(200, 2000)), molwr=float(rng.uniform(0, .3)),
                                   motwr=np.nan if k == 1 else .05))
        rows.append(_batch_row("15", "E", "replacement", np.nan, np.nan, error="ValueError('x')"))
        rows.append(_batch_row("16", "E", "replacement", 10., -5.))           # negative water
        rows.append(_batch_row("9", "A", "replacement", float("inf"), 100.))  # nonfinite oil
        rows.append(_batch_row("10", "B", "replacement", 40., 100., error=""))  # blank success marker
        if w % 2:
            # first match is invalid: the per-row lookup skips the valid duplicate
            rows.append(_batch_row("8", "X", "replacement", np.nan, 50., error="na"))
            rows.append(_batch_row("8", "X", "replacement", 20., 50.))
            # pump_state None prefers an installed match of the same size
            rows.append(_batch_row("12", "B", None, 500., 1500.))
        frames[f"W{w}"] = pd.DataFrame(rows)
    no_state = frames["W0"].drop(columns=["pump_state", "molwr", "motwr"]).iloc[2:].copy()
    no_state.loc[len(no_state) + 10] = no_state.iloc[0]  # repeated size, no pump_state column
    frames["NS"] = no_state
    frames["EMPTY"] = frames["W1"].iloc[:0]
    frames["ALLBAD"] = pd.DataFrame([_batch_row("15", "E", "installed", np.nan, np.nan, error="bad")])
    return frames


def test_vectorized_candidates_match_the_per_row_lookup(monkeypatch):
    frames = _parity_frames()
    vectorized = []
    real_rows = oa._performance_rows
    monkeypatch.setattr(oa, "_performance_rows", lambda *a: vectorized.append(1) or real_rows(*a))
    fast = oa._allocation_candidates(_optimizer(frames, 5000.))
    assert len(vectorized) == len(frames)
    slow = oa._allocation_candidates(_optimizer(frames, 5000., fast=False))
    assert len(vectorized) == len(frames)  # duck-typed lookups stay per row
    assert fast[0] == slow[0]
    assert fast[2] == slow[2] and fast[3] == slow[3]
    for a, b in zip(fast[1], slow[1]):
        pd.testing.assert_frame_equal(a, b)
    assert sum(map(len, fast[1])) > 30
    for method in ("milp", "mckp"):
        for price in (0.0, .05):
            opts = [_optimizer(frames, 5000., price, fast=f) for f in (True, False)]
            plans = [[(r.well_name, r.recommended_nozzle, r.recommended_throat, r.pump_state,
                       r.predicted_oil_rate, r.suction_pressure, r.marginal_oil_rate)
                      for r in oa._allocate(opt, "lift_wat", method)] for opt in opts]
            assert plans[0] == plans[1]
            assert opts[0].allocation_status["objective"] == opts[1].allocation_status["objective"]


def test_real_network_optimizer_uses_the_vectorized_lookup(monkeypatch):
    opt = NetworkOptimizer.__new__(NetworkOptimizer)
    assert oa._standard_performance_lookup(opt)
    monkeypatch.setattr(opt, "get_pump_performance", lambda *a, **kw: None, raising=False)
    assert not oa._standard_performance_lookup(opt)


def test_identical_installed_and_clean_outcomes_are_deduplicated_not_excluded():
    frame = pd.DataFrame([_batch_row("12", "B", "installed", 100., 500.),
                          _batch_row("12", "B", "replacement", 100., 500.),
                          _batch_row("13", "C", "replacement", np.nan, 600., error="ValueError('x')")])
    for fast in (True, False):
        opt = _optimizer({"A": frame}, 1000., fast=fast)
        results = oa.milp_optimization(opt)
        assert [r.pump_state for r in results] == ["installed"]
        assert opt.allocation_status["excluded_candidates"] == {"A": 1}
        assert opt.allocation_status["deduplicated_candidates"] == {"A": 1}


# ---------------------------------------------------------------------------
# Patch 50: a malformed installed identity skips only that candidate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("identity", [("12.0", "B"), ("20", "E"), ("12", "F")])
def test_malformed_installed_identity_is_skipped_with_a_reason(identity):
    rejected = []
    pumps = scoped_pumps(["12"], ["A", "B"], identity, {"kth": .6}, rejected=rejected)
    assert [(p.noz_no, p.rat_ar, p.pump_state) for p in pumps] == [
        ("12", "A", "replacement"), ("12", "B", "replacement")]
    assert len(rejected) == 1 and rejected[0][:2] == identity
    assert "installed pump identity not in catalog" in rejected[0][2]
    assert len(scoped_pumps(["12"], ["B"], identity)) == 1  # no sink: still no raise


def test_malformed_installed_identity_does_not_abort_the_batch(monkeypatch):
    """Review r14: one bad tracker identity aborted every well's batch."""
    from woffl.assembly import network_optimizer as no
    from woffl.assembly.batchpump import BatchPump

    def fake_batch_run(self, jetpumps, debug=False):
        self.df = pd.DataFrame([_batch_row(jp.noz_no, jp.rat_ar, jp.pump_state, 100., 500.)
                                for jp in jetpumps])
        return self.df

    monkeypatch.setattr(BatchPump, "batch_run", fake_batch_run)
    monkeypatch.setattr(BatchPump, "process_results", lambda self: None)
    wells = [no.WellConfig(well_name=name, res_pres=1500, form_temp=80, jpump_tvd=4000, form_wc=.4,
                           qwf=900, pwf=700, pump_calibration_scoped=True,
                           installed_nozzle=noz, installed_throat=thr)
             for name, noz, thr in (("GOOD", "11", "B"), ("BAD", "12.0", "B"))]
    opt = no.NetworkOptimizer(wells, no.PowerFluidConstraint(total_rate=5000., pressure=3000.), ["12"], ["B"])
    opt.run_all_batch_simulations(max_workers=1)
    good, bad = opt.batch_results["GOOD"].df, opt.batch_results["BAD"].df
    assert good.pump_state.tolist() == ["installed", "replacement"]
    assert bad.pump_state.tolist() == ["installed", "replacement"]
    assert bad.loc[0, "nozzle"] == "12.0" and np.isnan(bad.loc[0, "qoil_std"])
    assert "installed pump identity not in catalog" in bad.loc[0, "error"]
    accounting = no.reconcile_wells(opt, oa.optimize(opt, "milp")).set_index("Well")
    assert accounting.loc["BAD", "Configs Failed"] == 1
    assert "installed pump identity not in catalog" in accounting.loc["BAD", "Detail"]
