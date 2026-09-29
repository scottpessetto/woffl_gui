"""Patch 51 (2026-09-24 performance): candidate subsets, one solve for an
installed pump identical to its clean twin, single-row reads, and the pad
sweep's bracket-dominance pruning (``WOFFL_SWEEP_PRUNE``).

Real physics only where a row's identity is the point (small grids); the
sweep tests drive the REAL ``run_optimization`` / ``NetworkOptimizer`` /
MILP / CP-SAT and the real S- and I-Pad plants on an analytic batch model,
so pruned and unpruned sweeps can be compared exactly and quickly. No
Databricks, no network.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import woffl.assembly.network_optimizer as no
import woffl.assembly.parallelism as parallelism
from woffl.assembly import compute_runtime
from woffl.assembly.pump_candidates import identical_twins, jetpump_key, pump_key, scoped_pumps
from woffl.gui import pad_optimize as po

NOZZLES = ["9", "10", "11", "12", "13", "14", "15"]
THROATS = ["A", "B", "C", "D"]


def _scoped_well(name="MPS-TEST", **kw):
    base = dict(well_name=name, res_pres=1700.0, form_temp=80.0, jpump_tvd=4000.0, form_wc=0.4,
                form_gor=300.0, qwf=1200.0, pwf=900.0, pump_calibration_scoped=True,
                installed_nozzle="12", installed_throat="B")
    base.update(kw)
    return no.WellConfig(**base)


# ---------------------------------------------------------------------------
# Library: twins and subsets (real physics, small grids)
# ---------------------------------------------------------------------------


def test_identical_twins_require_every_physical_input_equal():
    clean = scoped_pumps(["11", "12"], ["B"], ("12", "B"), {})
    assert [jetpump_key(jp) for jp in clean] == [("12", "B", "installed"), ("11", "B", "replacement"),
                                                  ("12", "B", "replacement")]
    assert identical_twins(clean) == {2: 0}
    # the reference values written out, or an area factor of exactly 1.0, are still twins
    assert identical_twins(scoped_pumps(["12"], ["B"], ("12", "B"),
                                        {"ken": .03, "kth": .3, "kdi": .4, "nozzle_area_factor": 1.0})) == {1: 0}
    # any fitted loss or area keeps the installed pump independent
    for coefs in ({"ken": .05}, {"kth": .31}, {"kdi": .45}, {"nozzle_area_factor": 1.05}):
        assert identical_twins(scoped_pumps(["12"], ["B"], ("12", "B"), coefs)) == {}
    assert identical_twins(scoped_pumps(["12"], ["B"], None, {})) == {}
    assert pump_key(12, "b", None) == ("12", "B", None)


def test_twin_is_solved_once_and_matches_separate_solves(monkeypatch):
    """Guard: the collapsed batch equals the batch that solves both twins."""
    import woffl.assembly.pump_candidates as pc
    import woffl.assembly.solopump as so

    calls = []
    real = so.jetpump_solver

    def counted(*a, **k):
        calls.append(a[3].noz_no + a[3].rat_ar)
        return real(*a, **k)

    monkeypatch.setattr(so, "jetpump_solver", counted)
    well = _scoped_well()
    grid = (["10", "11", "12", "13"], ["A", "B", "C"])
    collapsed = no._simulate_single_well(well, 3000.0, *grid)
    n_collapsed = len(calls)
    monkeypatch.setattr(pc, "identical_twins", lambda jps: {})
    calls.clear()
    separate = no._simulate_single_well(well, 3000.0, *grid)
    assert n_collapsed == len(calls) - 1 == 12
    assert collapsed.df.equals(separate.df)
    assert collapsed.df.pump_state.tolist() == ["installed"] + ["replacement"] * 12
    for attr in ("coeff_lift", "coeff_totl"):
        np.testing.assert_array_equal(getattr(collapsed, attr), getattr(separate, attr))


def test_fitted_installed_pump_is_still_solved_separately(monkeypatch):
    import woffl.assembly.solopump as so

    calls = []
    real = so.jetpump_solver
    monkeypatch.setattr(so, "jetpump_solver", lambda *a, **k: calls.append(1) or real(*a, **k))
    bp = no._simulate_single_well(_scoped_well(fnz_well=1.1), 3000.0, ["12"], ["B"])
    assert len(calls) == 2
    assert bp.df.loc[0, "lift_wat"] != bp.df.loc[1, "lift_wat"]


def test_subset_rows_equal_the_full_grid_rows():
    """Guard: a subset job solves only its keys, in grid order, and each row is
    the full grid's row (the per-set semi/marginal columns aside)."""
    well = _scoped_well(ken_well=0.05)
    full = no._simulate_single_well(well, 2900.0, ["10", "11", "12"], ["A", "B", "C"])
    keys = [("12", "B", "installed"), ("11", "C", "replacement"), ("10", "a", "replacement")]
    part = no._simulate_single_well(well, 2900.0, ["10", "11", "12"], ["A", "B", "C"], keys)
    cols = ["nozzle", "throat", "pump_state", "qoil_std", "lift_wat", "form_wat", "psu_solv",
            "sonic_status", "mach_te", "error"]
    want = full.df.set_index(["nozzle", "throat", "pump_state"]).loc[
        [("12", "B", "installed"), ("10", "A", "replacement"), ("11", "C", "replacement")]].reset_index()
    pd.testing.assert_frame_equal(part.df[cols].reset_index(drop=True), want[cols])


def test_subset_cache_keys(monkeypatch):
    from server import surface_cache

    well = _scoped_well()
    base = surface_cache._key(well, 3000.0, ["12"], ["B"])
    assert base == surface_cache._key(well, 3000.0, ["12"], ["B"], None)
    a = surface_cache._key(well, 3000.0, ["12"], ["B"], [("12", "B", "installed")])
    b = surface_cache._key(well, 3000.0, ["12"], ["B"], [("12", "B", "replacement")])
    assert len({base, a, b}) == 3
    assert a == surface_cache._key(well, 3000.0, ["12"], ["B"], (("12", "B", "installed"),) * 2)


def test_simulate_jobs_pads_mixed_job_arity_for_the_pool(monkeypatch):
    import concurrent.futures as cf

    seen = []
    monkeypatch.setattr(compute_runtime, "job_runner", None)
    monkeypatch.setattr(cf, "ProcessPoolExecutor", cf.ThreadPoolExecutor)
    monkeypatch.setattr(no, "_simulate_single_well",
                        lambda w, p, n, t, pumps=None: seen.append((p, pumps)) or SimpleNamespace(p=p))
    out = no.simulate_jobs([(None, 1.0, ["12"], ["B"]), (None, 2.0, ["12"], ["B"], [("12", "B", None)])],
                           max_workers=2)
    assert [o.p for o in out] == [1.0, 2.0]
    assert sorted(seen, key=lambda s: s[0]) == [(1.0, None), (2.0, [("12", "B", None)])]


# ---------------------------------------------------------------------------
# Analytic batch model for sweep tests (honours subsets, records calls)
# ---------------------------------------------------------------------------


_T_EFF = {"A": 1.0, "B": 1.06, "C": 0.99, "D": 0.9}
_T_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}


def _row(wc, header, nozzle, throat, state):
    i = int(wc.well_name.split("-")[1])
    qmax = 400.0 + 170.0 * i
    lift = 22.0 * int(nozzle) ** 2 * (1 + 0.12 * _T_IDX[throat]) * math.sqrt(header / 1000.0)
    eff = _T_EFF[throat] * (0.93 if state == "installed" and wc.kth_well else 1.0)
    fails = (int(nozzle) == 9 and header < 2400.0) or (int(nozzle) >= 14 and throat == "D")
    oil = qmax * (1 - math.exp(-lift * eff / (qmax * 9.0))) * (1 + 0.02 * (i % 3))
    form = oil * wc.form_wc / (1 - wc.form_wc)
    return {"nozzle": nozzle, "throat": throat, "sonic_status": False,
            "mach_te": np.nan if fails else 0.5, "psu_solv": np.nan if fails else 900.0 - oil / 10.0,
            "qoil_std": np.nan if fails else oil, "form_wat": np.nan if fails else form,
            "lift_wat": np.nan if fails else lift, "totl_wat": np.nan if fails else form + lift,
            "form_wor": np.nan, "totl_wor": np.nan, "error": "failed" if fails else "na",
            **({"pump_state": state} if wc.pump_calibration_scoped else {})}


@pytest.fixture
def model(monkeypatch):
    """Replace the per-well physics with the analytic model; record calls."""
    log = []

    def simulate(wc, pressure, nozzles, throats, pumps=None):
        header = wc.ppf_surf_well if wc.ppf_surf_well is not None else pressure
        jps = scoped_pumps(nozzles, throats, (wc.installed_nozzle, wc.installed_throat), {})
        if pumps is not None:
            wanted = {pump_key(*k) for k in pumps}
            jps = [jp for jp in jps if jetpump_key(jp) in wanted]
        log.append((wc.well_name, header, None if pumps is None else len(jps)))
        return SimpleNamespace(df=pd.DataFrame([_row(wc, header, jp.noz_no, jp.rat_ar, jp.pump_state)
                                                for jp in jps]))

    monkeypatch.setattr(no, "_simulate_single_well", simulate)
    monkeypatch.setattr(compute_runtime, "job_runner", None)
    monkeypatch.setattr(compute_runtime, "batch_runner", None)
    monkeypatch.setattr(parallelism, "worker_ceiling", lambda: 1)
    return log


def _wells(pad, n=8):
    return [_scoped_well(f"MP{pad}-{i:02d}", form_wc=round(0.2 + 0.08 * i, 3),
                         installed_nozzle=str(10 + i % 4), installed_throat=["B", "C", "B", "A"][i % 4],
                         kth_well=0.45 if i % 3 == 0 else None)
            for i in range(n)]


def _plant(pad):
    if pad == "S":
        from woffl.gui.s_pad_plant import PLANT
        return PLANT, 3
    from woffl.gui.i_pad_plant import PLANT
    return PLANT, None


def _run(pad, monkeypatch, prune, **kw):
    monkeypatch.setenv("WOFFL_SWEEP_PRUNE", "1" if prune else "0")
    plant, n_pumps = _plant(pad)
    return po.run_optimization(_wells(pad), plant, n_pumps, NOZZLES, THROATS, kw.pop("method", "milp"), None,
                               n_steps=11, **kw)


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    if isinstance(a, pd.DataFrame) or isinstance(b, pd.DataFrame):
        return isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame) and a.equals(b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


# ---------------------------------------------------------------------------
# Sweep pruning
# ---------------------------------------------------------------------------


def test_prune_order_brackets_every_pruned_point():
    for n in range(1, 22):
        order = po._prune_order(n)
        assert sorted(i for i, _ in order) == list(range(n))
        full = {i for i, f in order if f}
        assert {0, n - 1} <= full and (n < 3 or (n - 1) // 2 in full)
        done = []
        for i, f in order:
            if not f:
                assert any(d < i for d in done) and any(d > i for d in done)
            done.append(i)


def test_pruner_needs_one_rival_beating_by_the_margin_on_both_sides():
    pr = po._SweepPruner("lift_wat", margin=0.01)
    frame = lambda rows: SimpleNamespace(df=pd.DataFrame(
        [{"nozzle": n, "throat": t, "pump_state": "replacement", "qoil_std": q, "lift_wat": w, "error": e}
         for n, t, q, w, e in rows]))
    lo = {"W": frame([("12", "A", 100.0, 1000.0, "na"), ("12", "B", 99.5, 1005.0, "na"),
                      ("12", "C", 90.0, 1100.0, "na"), ("12", "D", 80.0, 1200.0, "bad")])}
    hi = {"W": frame([("12", "A", 120.0, 1200.0, "na"), ("12", "B", 119.0, 1205.0, "na"),
                      ("12", "C", 100.0, 1300.0, "na"), ("12", "D", 90.0, 1400.0, "na")])}
    assert pr.observe(2000.0, lo, {}) == 0 and pr.observe(3000.0, hi, {}) == 0
    keep, skipped = pr.plan(["W"], 2500.0)["W"]
    # C is beaten >= 1% on both sides by A and B; B only by < 1%; D failed at one side
    rivals = frozenset({("12", "A", "replacement"), ("12", "B", "replacement")})
    assert skipped == {("12", "C", "replacement"): rivals}
    assert keep == [("12", "A", "replacement"), ("12", "B", "replacement"), ("12", "D", "replacement")]
    assert pr.plan(["W"], 3500.0) == {} and pr.plan(["W"], 2000.0) == {}
    mid = {"W": frame([("12", "A", 110.0, 1100.0, "na"), ("12", "B", 109.0, 1105.0, "na"),
                       ("12", "D", 85.0, 1300.0, "na")])}
    assert pr.observe(2500.0, mid, {"W": (keep, skipped)}) == 1
    # the skipped candidate carries its common rivals into narrower brackets
    assert pr.records["W"][2500.0][("12", "C", "replacement")] == rivals
    assert pr.plan(["W"], 2250.0)["W"][1].keys() == {("12", "C", "replacement")}


@pytest.mark.parametrize("pad", ["S", "I"])
@pytest.mark.parametrize("case", ["default", "price", "mckp", "required"])
def test_pruned_sweep_matches_the_unpruned_sweep_exactly(pad, case, model, monkeypatch):
    """Guard: identical plan, meta, sweep and winning batch frames; fewer solves."""
    kw = {"default": {}, "price": {"water_price": 0.08}, "mckp": {"method": "mckp"},
          "required": {"required_wells": [f"MP{pad}-07"]}}[case]
    r0, o0, m0 = _run(pad, monkeypatch, False, **dict(kw))
    unpruned_log = list(model)
    model.clear()
    r1, o1, m1 = _run(pad, monkeypatch, True, **dict(kw))
    assert [asdict(r) for r in r0] == [asdict(r) for r in r1]
    assert _same({k: v for k, v in m0.items() if k != "sweep_pruning"},
                 {k: v for k, v in m1.items() if k != "sweep_pruning"})
    assert all(o1.batch_results[w].df.equals(o0.batch_results[w].df) for w in o0.batch_results)
    assert m0["sweep_pruning"] == {"enabled": False, "margin": None, "skipped_solves": 0,
                                   "total_solves": None, "winner_reruns": 0}
    assert m1["sweep_pruning"]["enabled"] and m1["sweep_pruning"]["skipped_solves"] > 0
    assert m1["sweep_pruning"]["margin"] == po._PRUNE_MARGIN
    # the kill switch path never asks for a sweep subset (only single-row reads)
    assert all(c[2] is None or c[2] == 1 for c in unpruned_log)
    rows = lambda log: sum(29 if c[2] is None else c[2] for c in log)
    assert rows(model) < rows(unpruned_log)
    if case == "required":
        assert f"MP{pad}-07" in {r.well_name for r in r1}


def test_kill_switch_restores_natural_order(model, monkeypatch):
    _run("I", monkeypatch, False)
    headers = [h for _w, h, n in model if n is None]
    firsts = list(dict.fromkeys(headers))
    assert firsts[:11] == sorted(firsts[:11])  # coarse sweep left to right
    model.clear()
    _run("I", monkeypatch, True)
    order = list(dict.fromkeys(h for _w, h, _n in model))
    lo, hi = min(order[:11]), max(order[:11])
    assert order[:2] == [lo, (lo + hi) / 2]  # an end and the middle first


@pytest.mark.parametrize("pad", ["S", "I"])
def test_pruned_winner_is_rerun_on_the_full_grid(pad, model, monkeypatch):
    """Guard: the returned optimizer holds complete frames at the reported
    search header even when the winning trial was a pruned one."""
    results, opt, meta = _run(pad, monkeypatch, True)
    assert all(len(bp.df) == 29 for bp in opt.batch_results.values())
    assert meta["sweep_pruning"]["winner_reruns"] in (0, 1)
    header = meta.get("search_header_psi", meta["header_psi"])
    at_header = [c for c in model if c[1] == header]
    if meta["sweep_pruning"]["winner_reruns"]:
        # pruned first, then only the skipped candidates: each solved once
        assert any(c[2] is not None and c[2] > 1 for c in at_header)
        assert sum(29 if c[2] is None else c[2] for c in at_header) == 29 * len(opt.batch_results)
    assert {r.well_name for r in results} <= set(opt.batch_results)


# ---------------------------------------------------------------------------
# Single-row reads
# ---------------------------------------------------------------------------


def test_forced_header_solves_only_the_held_row(model):
    wells = _wells("I", 3)
    held = {wells[0].well_name: (wells[0].installed_nozzle, wells[0].installed_throat),
            wells[1].well_name: ("14", "C"), wells[2].well_name: ("13", "B", "replacement")}
    out = po._model_at_forced_header(wells, 3000.0, held)
    assert [c[2] for c in model] == [1, 1, 1]
    assert all(v is not None for v in out.values())
    assert po._read_key(wells[0], held[wells[0].well_name])[2] == "installed"
    assert po._read_key(wells[1], ("14", "C")) == ("14", "C", "replacement")
    legacy = no.WellConfig(well_name="L-1", res_pres=1500, form_temp=70, jpump_tvd=4000)
    assert po._read_key(legacy, ("12", "B", "installed")) == ("12", "B", None)


def test_fixed_scenario_completes_a_failing_well_on_the_union_grid(model, monkeypatch):
    """Chosen and fallback both fail at the header: the well is re-run on the
    union grid so the best-feasible fallback sees the rows it always did."""
    from woffl.gui.i_pad_plant import PLANT

    wells = _wells("I", 3)
    # 15D and 14D never solve in the analytic model
    choices = {wells[0].well_name: ("15", "D"), wells[1].well_name: ("12", "B"), wells[2].well_name: None}
    fallback = {wells[0].well_name: ("14", "D")}
    per_well, _meta = po.evaluate_fixed_scenario(wells, PLANT, None, choices, fallback_choices=fallback)
    rows = {r["well"]: r for r in per_well}
    w0 = [c for c in model if c[0] == wells[0].well_name]
    assert any(c[2] == 2 for c in w0) and any(c[2] is None for c in w0)  # reads, then the union grid
    # the best feasible pump of the union grid {12, 14, 15} x {B, D}
    assert rows[wells[0].well_name]["pump"] == "15D✗→15B"
    assert all(c[2] == 1 for c in model if c[0] == wells[1].well_name)
    assert rows[wells[2].well_name]["pump"] == "SHUT IN"
    assert all(c[0] != wells[2].well_name for c in model)
