"""Jet Pump Network Solver

Add mutliple BatchPumps to a network and provide a shared resource. The shared
resource can be either lift water (power fluid) or total water.
"""

from decimal import Decimal
from math import fsum
import warnings

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from woffl.assembly.batchpump import BatchPump

SCALE = 100  # CP-SAT requires integers; multiply floats by this before rounding
RESOURCE_TOLERANCE = 1e-8  # BPD; floating arithmetic only, not plant headroom


# [LIBRARY change -> upstream PR to kwellis/woffl] Patch 46: typed outcomes,
# original-unit capacity qualification, and optional per-well online constraints.
class AllocationError(RuntimeError):
    """Allocation could not produce a qualified plan; never an implicit SI plan."""

    def __init__(self, message: str, status: str = "error", **details):
        super().__init__(message)
        self.status = status
        self.details = {**details, "status": status, "message": message}


def valid_rate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep finite, nonnegative oil/water rates in columns present; no mutation."""
    keep = pd.Series(True, index=df.index)
    for key in ("qoil_std", "lift_wat", "form_wat", "totl_wat"):
        if key in df:
            values = pd.to_numeric(df[key], errors="coerce")
            keep &= np.isfinite(values) & (values >= 0)
    return df.loc[keep]


def _validate_problem(names, candidates, capacity, price, required):
    if not np.isfinite(capacity) or capacity < 0:
        raise AllocationError("Water capacity must be finite and nonnegative", "error")
    if not np.isfinite(price) or price < 0:
        raise AllocationError("Water price must be finite and nonnegative", "error")
    if len(names) != len(set(names)):
        raise AllocationError("Allocation well names must be unique", "error")
    available = {name for name, df in zip(names, candidates) if not df.empty}
    missing = sorted(set(required) - available)
    if missing:
        raise AllocationError(
            "Required online wells have no valid allocation options: " + ", ".join(missing),
            "unsupported", required_wells=sorted(required), unsupported_wells=missing,
        )


def _qualified_selection(names, candidates, chosen, capacity, water_key, required):
    """Verify required selections and capacity in original BPD."""
    if not set(required).issubset(chosen):
        return False
    water = fsum(float(candidates[i].iloc[j][water_key]) for i, j in chosen.values())
    return water <= capacity + RESOURCE_TOLERANCE


def _selection_metrics(names, candidates, chosen, capacity, price, water_key):
    water = fsum(float(candidates[i].iloc[j][water_key]) for i, j in chosen.values())
    oil = fsum(float(candidates[i].iloc[j]["qoil_std"]) for i, j in chosen.values())
    return dict(objective=oil - price * water, oil_bopd=oil,
                selected_water_bpd=water, capacity_bpd=float(capacity),
                resource_tolerance_bpd=RESOURCE_TOLERANCE,
                selected_wells=sorted(chosen), shut_in_wells=sorted(set(names) - set(chosen)))


def solve_milp_choices(names, candidates, capacity, water_key="lift_wat", price=0.0,
                       required_wells=None):
    """MILP on candidate rows; return selected row indices and JSON-safe status.

    Used by the GUI and CP-SAT's precise-resource refinement. No well physics is
    recomputed. One native thread; objective is oil minus price*water.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csc_array, vstack

    required = set(required_wells or ())
    _validate_problem(names, candidates, capacity, price, required)
    records = [(i, j) for i, df in enumerate(candidates) for j in range(len(df))]
    if not records:
        return {}, dict(status="optimal", solver="milp", objective_bound=0.0, gap=0.0,
                        required_wells=sorted(required),
                        **_selection_metrics(names, candidates, {}, capacity, price, water_key))
    oil = np.array([float(candidates[i].iloc[j]["qoil_std"]) for i, j in records])
    water = np.array([float(candidates[i].iloc[j][water_key]) for i, j in records])
    c = -(oil - price * water)
    n = len(records)
    A_well = csc_array((np.ones(n), ([i for i, _ in records], list(range(n)))),
                       shape=(len(names), n))
    A = vstack([A_well, csc_array(water.reshape(1, -1))], format="csc")
    lower = np.array([1.0 if name in required else 0.0 for name in names] + [-np.inf])
    upper = np.array([1.0] * len(names) + [capacity])
    constraints = [LinearConstraint(A, lower, upper)]
    bounds = Bounds(np.zeros(n), np.ones(n))

    def run(objective, extra=()):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unrecognized options detected:.*",
                                    category=RuntimeWarning)
            try:
                return milp(c=objective, constraints=[*constraints, *extra],
                            bounds=bounds, integrality=np.ones(n),
                            options={"mip_rel_gap": 0.0, "threads": 1,
                                     "mip_feasibility_tolerance": 1e-9})
            except Exception as exc:
                raise AllocationError(f"MILP solver error: {exc}", "error", solver="milp") from exc

    def choice(result):
        x = getattr(result, "x", None)
        if x is None or len(x) != n or not np.all(np.isfinite(x)) or np.any(abs(x - np.rint(x)) > 1e-6) or np.any(np.asarray(x) < -1e-6) or np.any(np.asarray(x) > 1 + 1e-6):
            return None
        indices = np.flatnonzero(np.asarray(x) > .5)
        selected = {names[records[k][0]]: records[k] for k in indices}
        return selected if len(selected) == len(indices) else None

    # HiGHS has a feasibility tolerance. Exclude an out-of-budget binary
    # incumbent and re-solve rather than publishing a physically invalid plan.
    for refinements in range(20):
        result = run(c)
        status_code = int(getattr(result, "status", 4))
        selected = choice(result)
        status = {0: "optimal", 1: "feasible", 2: "infeasible", 3: "error", 4: "error"}.get(status_code, "unknown")
        if status not in ("optimal", "feasible") or selected is None:
            if status == "feasible":
                status = "unknown"
            elif status == "optimal":
                status = "error"
            raise AllocationError(f"MILP {status}: {getattr(result, 'message', 'no qualified incumbent')}",
                                  status, solver="milp", solver_status=status_code,
                                  required_wells=sorted(required))
        if _qualified_selection(names, candidates, selected, capacity, water_key, required):
            break
        indices = np.flatnonzero(np.asarray(result.x) > .5)
        if not len(indices) or not required.issubset(selected):
            raise AllocationError("MILP returned an invalid well allocation", "error", solver="milp")
        cut = np.zeros(n)
        cut[indices] = 1.0
        constraints.append(LinearConstraint(cut, -np.inf, len(indices) - 1))
    else:
        raise AllocationError("MILP could not qualify capacity after numerical refinement", "error", solver="milp")

    primary = float(c @ np.rint(result.x))
    raw_bound = getattr(result, "mip_dual_bound", None)
    raw_gap = getattr(result, "mip_gap", None)
    diagnostics = dict(status=status, solver="milp", solver_status=status_code,
                       message=str(getattr(result, "message", "")),
                       objective_bound=-float(raw_bound) if raw_bound is not None and np.isfinite(raw_bound) else None,
                       gap=float(raw_gap) if raw_gap is not None and np.isfinite(raw_gap) else None,
                       capacity_refinements=refinements, required_wells=sorted(required))
    if price > 0 and status == "optimal":
        try:
            tied = run(-oil, [LinearConstraint(c, primary, primary)])
            proposed = choice(tied)
            if getattr(tied, "status", None) in (0, 1) and proposed is not None and _qualified_selection(names, candidates, proposed, capacity, water_key, required) and float(c @ np.rint(tied.x)) <= primary + 1e-7:
                selected = proposed
            else:
                diagnostics["oil_tie_break"] = "primary solution retained"
        except AllocationError as exc:
            diagnostics["oil_tie_break"] = str(exc)
    diagnostics.update(_selection_metrics(names, candidates, selected, capacity, price, water_key))
    return selected, diagnostics


def optimize_jet_pumps(
    well_list: list[BatchPump],
    qpf_tot: float,
    water_key: str = "lift_wat",
    allow_shutin: bool = False,
    water_price: float = 0.0,
    all_configs: bool = False,
    required_wells=None,
) -> pd.DataFrame:
    """Allocate pumps under capacity, with optional per-well online requirements.

    Existing callers retain the DataFrame interface, semi-finalist default and
    off rows. ``required_wells`` forces those names online when ``allow_shutin``
    is true; otherwise every well must be online. The objective is oil (BOPD)
    minus ``water_price`` times the chosen water stream (BPD). Solver status,
    original-unit totals, bound and gap are in ``df.attrs['allocation_status']``.
    Failures raise :class:`AllocationError`, never a shutdown plan.
    """
    # [LIBRARY change -> upstream PR to kwellis/woffl] Patch 46.
    if water_key not in ("lift_wat", "totl_wat"):
        raise ValueError(f"Unknown water_key: {water_key}")
    names = [well.wellname for well in well_list]
    required = set(required_wells or ()) | (set() if allow_shutin else set(names))
    candidates = []
    excluded = {}
    for well in well_list:
        df = well.df
        if not all_configs:
            df = df[df["semi"]]
            if df.empty:
                raise ValueError(f"Well '{well.wellname}' has no semi-finalists")
        if "error" in df:
            err = df["error"]
            df = df[err.isna() | err.astype(str).str.strip().isin(("na", ""))]
        if "qoil_std" not in df or water_key not in df:
            raise AllocationError(f"Well '{well.wellname}' lacks oil/{water_key} rates", "unsupported")
        valid = valid_rate_rows(df).reset_index(drop=True)
        excluded[well.wellname] = int(len(well.df) - len(valid))
        candidates.append(valid)
    _validate_problem(names, candidates, qpf_tot, water_price, required)

    # Resource ceil/floor at .01 BPD can exclude an exact-fit allocation.
    # Refine fractional resources in original units; Decimal avoids binary
    # float dust even when inputs are exact hundredths.
    water_scaled = [[Decimal(str(float(w))) * SCALE for w in df[water_key]] for df in candidates]
    needs_precision = any(w != w.to_integral_value() for values in water_scaled for w in values)
    if needs_precision:
        selected, diagnostics = solve_milp_choices(names, candidates, qpf_tot, water_key,
                                                    water_price, required)
        diagnostics.update(requested_solver="cp-sat", refinement_reason="fractional resource coefficients")
    else:
        model = cp_model.CpModel()
        x = [[model.new_bool_var(f"w{i}_p{j}") for j in range(len(df))] for i, df in enumerate(candidates)]
        for name, well_vars in zip(names, x):
            if name in required:
                model.add_exactly_one(well_vars)
            else:
                model.add_at_most_one(well_vars)
        # Match the original-unit qualification tolerance. Without it, binary
        # dust just below an exact resource quantum can still shut an entire well.
        capacity_decimal = Decimal(str(float(qpf_tot))) + Decimal(str(RESOURCE_TOLERANCE))
        capacity_scaled = int((capacity_decimal * SCALE).to_integral_value(rounding="ROUND_FLOOR"))
        model.add(sum(int(w) * x[i][j] for i, values in enumerate(water_scaled) for j, w in enumerate(values)) <= capacity_scaled)
        primary = sum(int(np.floor((float(row["qoil_std"]) - water_price * float(row[water_key])) * SCALE)) * x[i][j]
                      for i, df in enumerate(candidates) for j, (_, row) in enumerate(df.iterrows()))
        model.maximize(primary)
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        try:
            status_code = solver.solve(model)
        except Exception as exc:
            raise AllocationError(f"CP-SAT solver error: {exc}", "error", solver="cp-sat") from exc
        status = {cp_model.OPTIMAL: "optimal", cp_model.FEASIBLE: "feasible", cp_model.INFEASIBLE: "infeasible", cp_model.MODEL_INVALID: "error", cp_model.UNKNOWN: "unknown"}.get(status_code, "unknown")
        if status not in ("optimal", "feasible"):
            raise AllocationError(f"CP-SAT allocation {status}", status, solver="cp-sat",
                                  solver_status=int(status_code), required_wells=sorted(required))
        selected = {names[i]: (i, j) for i, vars_ in enumerate(x) for j, var in enumerate(vars_) if solver.value(var)}
        bound = float(solver.best_objective_bound) / SCALE + len(names) / SCALE
        diagnostics = dict(status=status, solver="cp-sat", solver_status=int(status_code),
                           objective_bound=bound, quantized_objective=float(solver.objective_value) / SCALE,
                           objective_resolution_bopd=1 / SCALE, required_wells=sorted(required),
                           optimality_scope="integer priced objective; bound includes quantization")
        if water_price > 0 and status_code == cp_model.OPTIMAL:
            model.add(primary == solver.value(primary))
            model.maximize(sum(int(np.floor(float(oil) * SCALE)) * x[i][j]
                               for i, df in enumerate(candidates) for j, oil in enumerate(df["qoil_std"])))
            try:
                tied_status = solver.solve(model)
                if tied_status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                    selected = {names[i]: (i, j) for i, vars_ in enumerate(x) for j, var in enumerate(vars_) if solver.value(var)}
                else:
                    diagnostics["oil_tie_break"] = "primary solution retained"
            except Exception as exc:
                diagnostics["oil_tie_break"] = f"primary solution retained: {exc}"
        if not _qualified_selection(names, candidates, selected, qpf_tot, water_key, required):
            selected, diagnostics = solve_milp_choices(names, candidates, qpf_tot, water_key, water_price, required)
            diagnostics.update(requested_solver="cp-sat", refinement_reason="original-unit capacity qualification")
        else:
            diagnostics.update(_selection_metrics(names, candidates, selected, qpf_tot, water_price, water_key))
            diagnostics["gap"] = max(0.0, bound - diagnostics["objective"]) / max(1.0, abs(diagnostics["objective"]))

    results = []
    for i, name in enumerate(names):
        if name in selected:
            _, j = selected[name]
            row = candidates[i].iloc[j]
            results.append({"wellname": name, **{key: row[key] for key in
                ("nozzle", "throat", "qoil_std", "lift_wat", "form_wat", "totl_wat")},
                **({"pump_state": row["pump_state"]} if "pump_state" in row else {})})
        else:
            results.append(dict(wellname=name, nozzle="off", throat="off", qoil_std=0.0,
                                lift_wat=0.0, form_wat=0.0, totl_wat=0.0))
    output = pd.DataFrame(results) if results else pd.DataFrame(columns=
        ["wellname", "nozzle", "throat", "qoil_std", "lift_wat", "form_wat", "totl_wat"])
    diagnostics["excluded_candidates"] = excluded
    output.attrs["allocation_status"] = diagnostics
    return output
