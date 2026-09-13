r"""Offline CFP algorithm counterexamples; no runtime writes or warehouse calls.

Run from the repo: .\venv\Scripts\python.exe build/optimization-algo-review/cfp_probes.py
PYTHONPATH must include the repo. Output is synthetic, not field validation.
"""
from itertools import product
import json
import math
from pathlib import Path
import random

from woffl.gui import cfp_moves as cm

GRID = [2300., 2500., 2600., 2700., 2800., 2880.]
P0 = 2800.


def option(oil, water, slope=0., grid=GRID):
    return {"_grid": grid, "oil": [oil+slope*(p-P0) for p in grid], "water": [water]*len(grid)}


def surface(well, online=True, current="A", options=None):
    return cm.WellSurface(well, "B", online, current, options or {})


def plant(s):
    return cm.anchor(s, psi_per_kbpd=15.)


def key_state(s):
    return {k: s[k] for k in ("pressure", "oil", "water", "feasible", "converged", "pressure_residual_psi", "choices")}


def pressure_floor():
    grid = [2500., 2600., 2700., 2800., 2880.]
    s = cm.Surfaces(grid, P0, {
        "Existing": surface("Existing", options={"A": option(100., 1000., grid=grid)}),
        "New": surface("New", False, None, {"B": option(500., 30000., grid=grid)}),
    })
    p = plant(s)
    state = cm.settle({"Existing": "A", "New": "B"}, s, p)
    raw = p.p0 + p.psi_per_kbpd*(p.baseline_water-state["water"])/1000
    assert state["feasible"] and state["pressure"] == 2500 and raw == 2350
    return {"state": key_state(state), "raw_anchored_pressure": raw,
            "unclipped_balance_error_psi": state["pressure"]-raw,
            "plan_gain": cm.moves_summary(s, p)["plan_gain"]}


def missing_current():
    s = cm.Surfaces(GRID, P0, {"OnlineNoCurrent": surface("OnlineNoCurrent", options={"B": option(100., 1000.)})})
    p = plant(s)
    summary = cm.moves_summary(s, p)
    assert summary["baseline"] == {"OnlineNoCurrent": "SI"}
    assert summary["today"]["oil"] == 0 and summary["plan_gain"] == 100
    return {"baseline": summary["baseline"], "today": summary["today"],
            "plan_gain": summary["plan_gain"], "actions": summary["plan"]["actions"]}


def coupled_only_pair():
    bol = option(300., 4000.)
    bol["oil"][:4] = [None]*4
    bol["water"][:4] = [None]*4
    s = cm.Surfaces(GRID, P0, {
        "Offset": surface("Offset", options={"A": option(100., 4000.)}),
        "New": surface("New", False, None, {"B": bol}),
    })
    p = plant(s)
    alone = cm.settle({"Offset": "A", "New": "B"}, s, p)
    paired = cm.settle({"Offset": "SI", "New": "B"}, s, p)
    pairs = cm.pair_moves(s, p)
    assert not alone["feasible"] and paired["feasible"] and not pairs
    return {"alone": key_state(alone), "paired": key_state(paired),
            "pairs_returned": pairs, "pair_oil_gain": paired["oil"]-100,
            "frontier_plan": key_state(cm.best_plan(cm.sweep_frontier(s, p), s.baseline_choices(), s))}


def unavailable_old_option_at_new_pressure():
    old = option(100., 1000.)
    old["oil"][:4] = [None]*4
    old["water"][:4] = [None]*4
    s = cm.Surfaces(GRID, P0, {"Resize": surface("Resize", options={"A": old, "B": option(200., 2000.)})})
    summary = cm.moves_summary(s, plant(s))
    move = next(m for m in summary["singles"] if m["to"] == "B")
    ws = s.wells[move["well"]]
    try:
        # Exact expression in optimizer_runs._run_cfp_job move enrichment.
        delta = cm.option_at(ws, move["to"], move["pressure_after"])[1] - cm.option_at(ws, move["from"], move["pressure_after"])[1]
    except TypeError as exc:
        return {"valid_move": move, "enrichment_error": f"{type(exc).__name__}: {exc}"}
    raise AssertionError(f"expected existing API enrichment to fail, got {delta}")


def anchor_above_margin():
    p0 = 2890.
    grid = [2590.+(2880.-2590.)*i/6 for i in range(7)]
    s = cm.Surfaces(grid, p0, {"Online": surface("Online", options={"A": option(91., 1000., .1, grid)})})
    try:
        plant(s)
    except ValueError as exc:
        runtime_error = str(exc)
    else:
        raise AssertionError("runtime grid excludes legal schema p0")
    extended = grid+[2890., 2900.]
    s = cm.Surfaces(extended, p0, {"Online": surface("Online", options={"A": option(91., 1000., .1, extended)})})
    summary = cm.moves_summary(s, plant(s))
    return {"p0_psi": p0, "runtime_grid_max": max(grid), "runtime_anchor_error": runtime_error,
            "with_extended_grid_today": summary["today"],
            "with_extended_grid_baseline": key_state(cm.settle(s.baseline_choices(), s, plant(s))),
            "actual_oil_at_p0": cm.option_at(s.wells["Online"], "A", p0)[0]}


def search_gap():
    rng = random.Random(20260912)
    worst = None
    baseline_regression = None
    singles_regression = None
    for trial in range(1000):
        specs = {}
        for w in ("A", "B", "C"):
            opts = {}
            for label in ("current", "alt1", "alt2"):
                oil = rng.randrange(150, 701, 25)
                wat = rng.randrange(1500, 18001, 500)
                slope = rng.randrange(0, 11)*oil/10000
                opts[label] = (oil, wat, slope)
            specs[w] = opts
        s = cm.Surfaces(GRID, P0, {w: surface(w, current="current", options={label: option(*values) for label, values in opts.items()}) for w, opts in specs.items()})
        p = plant(s)
        best = cm.best_plan(cm.sweep_frontier(s, p), s.baseline_choices(), s)
        exact = max((cm.settle(dict(zip(s.wells, labels)), s, p) for labels in product(*(ws.choice_labels() for ws in s.wells.values()))), key=lambda st: st["oil"])
        # Floor-clamp defect is separate; restrict this optimizer comparison
        # to a best choice whose raw plant balance stays within the grid.
        if exact["pressure"] <= min(GRID) or best["pressure"] <= min(GRID):
            continue
        gap = exact["oil"]-best["oil"]
        if worst is None or gap > worst["oil_gap"]:
            summary = cm.moves_summary(s, p)
            worst = {"trial": trial, "specs": specs, "oil_gap": gap,
                     "frontier": key_state(best), "enumerated": key_state(exact),
                     "baseline_oil": summary["today"]["oil"], "reported_plan_gain": summary["plan_gain"],
                     "best_single_gain": summary["singles"][0]["fleet_oil_delta"] if summary["singles"] else None}
        baseline = cm.settle(s.baseline_choices(), s, p)
        if best["oil"] < baseline["oil"] - 1e-6 and baseline_regression is None:
            baseline_regression = {"trial": trial, "specs": specs, "frontier": key_state(best), "baseline": key_state(baseline)}
        singles = cm.rank_single_moves(s, p)
        if singles and singles[0]["fleet_oil_delta"] > best["oil"]-baseline["oil"] + 1e-6 and singles_regression is None:
            singles_regression = {"trial": trial, "specs": specs, "frontier": key_state(best), "baseline": key_state(baseline), "single": singles[0]}
    return {"worst_gap": worst, "baseline_regression": baseline_regression, "better_single": singles_regression}


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    report = {"pressure_floor": pressure_floor(), "missing_current": missing_current(),
              "coupled_only_pair": coupled_only_pair(),
              "unavailable_old_option_at_new_pressure": unavailable_old_option_at_new_pressure(),
              "anchor_above_margin": anchor_above_margin(),
              "frontier_gap": search_gap()}
    target = Path(__file__).with_name("cfp_probes.json")
    target.write_text(json.dumps(clean(report), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"artifact": str(target), "cases": list(report),
                      "floor_balance_error_psi": report["pressure_floor"]["unclipped_balance_error_psi"],
                      "invented_missing_current_gain": report["missing_current"]["plan_gain"],
                      "enumerated_frontier_gap_bopd": report["frontier_gap"]["worst_gap"]["oil_gap"],
                      "below_baseline_reproduced": report["frontier_gap"]["baseline_regression"] is not None,
                      "better_single_reproduced": report["frontier_gap"]["better_single"] is not None}, indent=2))
