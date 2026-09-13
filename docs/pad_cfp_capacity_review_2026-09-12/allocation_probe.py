r"""Offline allocation audit; no physics runs, warehouse reads, or application edits.

From repository root:
  $env:PYTHONPATH='.'
  $env:WOFFL_MAX_WORKERS='1'
  .\venv\Scripts\python.exe build/optimization-algo-review/allocation_probe.py

Small synthetic candidate tables are passed through the actual MILP/MCKP and
choke selectors. Decimal exhaustive enumeration is the independent oracle.
"""

from copy import deepcopy
from dataclasses import asdict
from decimal import Decimal
from itertools import product
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import scipy
import scipy.optimize
from ortools.sat.python import cp_model
import ortools

from woffl.assembly.network import optimize_jet_pumps
from woffl.assembly.network_optimizer import NetworkOptimizer, reconcile_wells
from woffl.assembly.optimization_algorithms import (
    _frontier_segments, _pareto_frontier, _valid_configs, derive_lambda,
    milp_optimization, mckp_optimization,
)
from woffl.gui.pad_optimize import _choke_frontier, _trim_to_budget


def row(oil, water, nozzle="12", throat="A", state="installed", form=0.0):
    return dict(qoil_std=oil, lift_wat=water, form_wat=form,
                totl_wat=water + form, nozzle=nozzle, throat=throat,
                pump_state=state, psu_solv=500.0, sonic_status=False,
                mach_te=.5, molwr=0.0, motwr=0.0, error="na", semi=True)


def optimizer(groups, capacity, price):
    opt = SimpleNamespace(wells=[SimpleNamespace(well_name=k) for k in groups],
        batch_results={k: SimpleNamespace(df=pd.DataFrame(v), wellname=k)
                       for k, v in groups.items()},
        power_fluid=SimpleNamespace(total_rate=capacity), water_price=price,
        optimization_results=None)
    opt.get_pump_performance = lambda *a, **kw: NetworkOptimizer.get_pump_performance(opt, *a, **kw)
    return opt


def oracle(groups, capacity, price=0.0, optional=True, water_key="lift_wat"):
    """At-most-one or exactly-one, maximize priced value then oil then less water.

    Decimal(str(...)) avoids inheriting binary float capacity errors. Tertiary
    water preference is only for a deterministic oracle witness; comparisons
    require primary value, and secondary oil only when the app uses it.
    """
    names = list(groups)
    best = None
    for chosen in product(*[([None] if optional else []) + opts for opts in groups.values()]):
        oil = sum((Decimal(str(r["qoil_std"])) for r in chosen if r), Decimal(0))
        water = sum((Decimal(str(r[water_key])) for r in chosen if r), Decimal(0))
        if water > Decimal(str(capacity)):
            continue
        score = oil - Decimal(str(price)) * water
        key = score, oil, -water
        if best is None or key > best[0]:
            best = key, dict(objective=float(score), oil=float(oil), water=float(water),
                choices={n: (f"{r['nozzle']}{r['throat']}:{r['pump_state']}" if r else "off")
                         for n, r in zip(names, chosen)})
    return best[1] if best else None


def invoke(groups, capacity, price, method, water_key="lift_wat"):
    opt = optimizer(groups, capacity, price)
    diagnostics = []
    original_milp = scipy.optimize.milp
    original_solve = cp_model.CpSolver.solve
    def trace_milp(*a, **kw):
        out = original_milp(*a, **kw)
        diagnostics.append({k: getattr(out, k, None) for k in
                            ("status", "message", "success", "mip_gap", "mip_node_count")})
        return out
    def trace_cp(self, *a, **kw):
        status = original_solve(self, *a, **kw)
        diagnostics.append(dict(status=self.status_name(status),
                                objective=self.objective_value,
                                objective_bound=self.best_objective_bound))
        return status
    try:
        with patch.object(scipy.optimize, "milp", trace_milp), patch.object(cp_model.CpSolver, "solve", trace_cp):
            result = {"milp": milp_optimization, "mckp": mckp_optimization}[method](opt, water_key)
        oil = sum(r.predicted_oil_rate for r in result)
        water = sum(r.predicted_lift_water if water_key == "lift_wat" else r.predicted_total_water for r in result)
        return dict(objective=oil - price * water, oil=oil, water=water,
                    choices={r.well_name: f"{r.recommended_nozzle}{r.recommended_throat}:{r.pump_state}" for r in result},
                    feasible_original=Decimal(str(water)) <= Decimal(str(capacity)),
                    reconciliation=reconcile_wells(opt, result).to_dict("records"),
                    raw_solver_diagnostics_not_returned_by_app=diagnostics)
    except Exception as exc:
        return dict(error=f"{type(exc).__name__}: {exc}", raw_solver_diagnostics_not_returned_by_app=diagnostics)


def evaluate(name, groups, capacity, price=0.0, water_key="lift_wat"):
    try:
        expected = oracle(groups, capacity, price, water_key=water_key)
    except Exception as exc:
        expected = dict(error=f"{type(exc).__name__}: {exc}")
    return dict(name=name, inputs=dict(groups=groups, capacity=capacity, water_price=price, water_key=water_key),
                oracle=expected,
                milp=invoke(groups, capacity, price, "milp", water_key),
                mckp=invoke(groups, capacity, price, "mckp", water_key))


def main():
    cases = []
    cases.append(evaluate("hard_capacity_oil", {"A": [row(100, 1000)], "B": [row(61, 600)]}, 1000))
    auto_groups = {"A": [row(100, 1000)], "B": [row(61, 600)]}
    lam, slack = derive_lambda({k: pd.DataFrame(v) for k, v in auto_groups.items()}, 1000)
    auto = evaluate("auto_shadow_price_changes_discrete_oil_winner", auto_groups, 1000, lam)
    auto["auto_lambda"] = dict(value=lam, slack=slack,
        unpriced_capacity_oracle=oracle(auto_groups, 1000))
    cases.append(auto)
    cases.append(evaluate("exact_fit_subcent_capacity_single", {"A": [row(500, 1000.001)]}, 1000.001))
    cases.append(evaluate("exact_fit_subcent_capacity_two", {"A": [row(500, 1000.001)], "B": [row(100, 500.001)]}, 1500.002))
    cases.append(evaluate("capacity_just_below_required", {"A": [row(500, 1000.001)]}, 1000))
    cases.append(evaluate("zero_profit_prefers_production", {"A": [row(100, 1000)]}, 1000, .1))
    cases.append(evaluate("all_negative_profit_is_all_shutin", {"A": [row(100, 1000)], "B": [row(20, 500)]}, 2000, .2))
    cases.append(evaluate("zero_oil_and_negative_oil", {"A": [row(0, 100)], "B": [row(-2, 100)]}, 200))
    cases.append(evaluate("negative_water_not_rejected", {"A": [row(0, -100)], "B": [row(500, 200)]}, 100))
    cases.append(evaluate("nan_water_not_rejected", {"A": [row(100, float("nan"))]}, 1000))
    cases.append(evaluate("infinite_oil_not_rejected", {"A": [row(float("inf"), 100)]}, 1000))
    cases.append(evaluate("quantized_secondary_oil_changes_primary", {"A": [row(100.001, 100), row(1000, 1000, throat="B")]}, 1000, 1.0))
    cases.append(evaluate("identical_same_size_keeps_installed", {"A": [row(100, 1000, state="replacement"), row(100, 1000)]}, 2000))
    cases.append(evaluate("total_water_capacity", {"A": [row(100, 100, form=900), row(80, 300, throat="B", form=100)]}, 500, water_key="totl_wat"))

    # Resource scaling is conservative. Even an exact 0.01 input can acquire
    # one extra integer unit from ceil(binary_float*100).
    dust = next((i / 100 for i in range(100_000, 101_000)
                 if np.ceil((i / 100) * 100) > i), None)
    if dust is not None:
        cases.append(evaluate("exact_hundredth_binary_float_dust", {"A": [row(500, dust)]}, dust))

    # Direct API can require all wells, but GUI bridge always allows every well SI.
    mandatory_groups = {"Existing": [row(100, 1000)], "Future": [row(10, 500)]}
    opt = optimizer(mandatory_groups, 1000, 0)
    mandatory = dict(inputs=mandatory_groups, capacity=1000,
        exactly_one_oracle=oracle(mandatory_groups, 1000, optional=False),
        gui_all_optional=invoke(mandatory_groups, 1000, 0, "mckp"))
    try:
        optimize_jet_pumps(list(opt.batch_results.values()), 1000, allow_shutin=False, all_configs=True)
    except Exception as exc:
        mandatory["direct_exactly_one_error"] = f"{type(exc).__name__}: {exc}"

    # Flat/concave two-point staircases: lowest marginal oil/PF trim is not
    # an exact discrete knapsack when the first trim greatly overshoots deficit.
    choke_groups = {"A": [row(500, 5000)], "B": [row(150, 1000)]}
    wells = [dict(well=k, idx=0, opts=_choke_frontier([(3000, r["qoil_std"], r["lift_wat"], 500) for r in rs] + [(None, 0, 0, None)])) for k, rs in choke_groups.items()]
    original = deepcopy(wells)
    pf, oil, slope = _trim_to_budget(wells, 5000)
    choke = dict(inputs=original, capacity=5000, greedy=dict(oil=oil, water=pf, last_slope=slope, states=wells),
                 oracle=oracle(choke_groups, 5000),
                 milp=invoke(choke_groups, 5000, 0, "milp"))
    dust_wells = [dict(idx=0, opts=[(3000, 500, 5000.0, 500), (None, 0, 0, None)])]
    dust_pf, dust_oil, _ = _trim_to_budget(dust_wells, 5000.0 - 1e-10)
    choke["floating_point_dust"] = dict(budget=5000.0 - 1e-10, demand=5000.0, oil=dust_oil, water=dust_pf)

    # Pareto != upper concave hull. Pooling sorted adjacent segments can spend
    # a high-return later segment without spending its prerequisite first.
    nonconcave = {"A": [row(1, 1000), row(101, 1100, throat="B")]}
    frontier = _pareto_frontier(pd.DataFrame(nonconcave["A"]), "lift_wat")
    nonconcave_shadow = dict(inputs=nonconcave, frontier=frontier,
        segments=_frontier_segments(frontier), capacity=500,
        derived=derive_lambda({"A": pd.DataFrame(nonconcave["A"])}, 500),
        upper_concave_hull_shadow=101 / 1100)

    # Bounded exact regressions: 120 random small models, integers for resources
    # and exactly representable prices. Check actual float primary objective.
    rng = random.Random(20260912)
    random_summary = dict(seed=20260912, cases=120, mismatches=[], max_primary_gap={"milp": 0., "mckp": 0.})
    for idx in range(random_summary["cases"]):
        groups = {f"W{i}": [row(rng.randint(-10, 300), rng.randint(0, 800), throat=chr(65+j)) for j in range(3)] for i in range(3)}
        capacity = rng.randint(0, 2000)
        price = rng.choice([0., .125, .25, .5, 1.])
        expected = oracle(groups, capacity, price)
        for method in ("milp", "mckp"):
            result = invoke(groups, capacity, price, method)
            gap = expected["objective"] - result.get("objective", float("-inf"))
            random_summary["max_primary_gap"][method] = max(random_summary["max_primary_gap"][method], gap)
            if gap > .031 or not result.get("feasible_original", False):
                random_summary["mismatches"].append(dict(index=idx, method=method, inputs=groups,
                    capacity=capacity, price=price, oracle=expected, result=result))

    # Failure/status channels: no live timeout needed to prove current adapters
    # collapse a solver failure into an all-SI allocation / capacity diagnosis.
    failure_groups = {"A": [row(100, 100)]}
    with patch.object(scipy.optimize, "milp", return_value=SimpleNamespace(success=False, status=4, message="Synthetic solver error")):
        failed_milp = milp_optimization(optimizer(failure_groups, 1000, 0))
    with patch.object(cp_model.CpSolver, "solve", return_value=cp_model.UNKNOWN):
        failed_cp = mckp_optimization(optimizer(failure_groups, 1000, 0))
    output = dict(scope="offline synthetic allocation audit, no physics or live data", versions=dict(numpy=np.__version__, scipy=scipy.__version__, ortools=ortools.__version__),
        cases=cases, mandatory=mandatory, choke=choke, nonconcave_shadow=nonconcave_shadow,
        random_oracle=random_summary,
        injected_failure_reporting=dict(milp_solver_error_result=failed_milp, cp_unknown_result=failed_cp))
    target = Path(__file__).with_name("allocation_results.json")
    def json_safe(value):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        if isinstance(value, dict):
            return {k: json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(v) for v in value]
        return value
    target.write_text(json.dumps(json_safe(output), indent=2, allow_nan=False, default=str), encoding="utf-8")
    print(json.dumps(dict(result=str(target), named_cases=len(cases), random_oracle=random_summary,
                         auto_oil=dict(selected=auto["milp"].get("oil"), pure_oil=auto["auto_lambda"]["unpriced_capacity_oracle"]["oil"]),
                         choke_oil=dict(greedy=oil, oracle=choke["oracle"]["oil"])), indent=2))


if __name__ == "__main__":
    main()
