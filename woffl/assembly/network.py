"""Jet Pump Network Solver

Add mutliple BatchPumps to a network and provide a shared resource. The shared
resource can be either lift water (power fluid) or total water.
"""

from decimal import Decimal
from math import floor, fsum
import os
import time
import warnings

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from woffl.assembly.batchpump import BatchPump

SCALE = 100  # CP-SAT requires integers; multiply floats by this before rounding
RESOURCE_TOLERANCE = 1e-8  # BPD; floating arithmetic only, not plant headroom

# [LIBRARY change -> upstream PR to kwellis/woffl] Patch 47: one wall-clock
# budget per allocation call, shared by every native solve it makes.
DEFAULT_TIME_LIMIT_S = 20.0  # s; primary, refinement and fallback solves together
TIME_LIMIT_ENV = "WOFFL_ALLOC_TIME_LIMIT_S"
_MIN_SOLVE_S = 0.05  # s; an expired budget still lets a solve report its state

# [LIBRARY change -> upstream PR to kwellis/woffl] Patch 48: tie-break terms
# may move the MILP's priced objective by less than this fraction of its
# gross scale (see _tie_break_terms).
TIE_BREAK_REL = 1e-6
_CP_EXACT_LIMIT = 2**53  # CP-SAT objective values are doubles; keep them exact


# [LIBRARY change -> upstream PR to kwellis/woffl] Patch 46: typed outcomes,
# original-unit capacity qualification, and optional per-well online constraints.
class AllocationError(RuntimeError):
    """Allocation could not produce a qualified plan; never an implicit SI plan."""

    def __init__(self, message: str, status: str = "error", **details):
        super().__init__(message)
        self.status = status
        self.details = {**details, "status": status, "message": message}


def resolve_time_limit(time_limit_s: float | None = None) -> float:
    """Wall-clock budget for one allocation call.

    Args:
        time_limit_s (float | None): Explicit budget, s. ``None`` reads the
            ``WOFFL_ALLOC_TIME_LIMIT_S`` environment variable, else uses
            ``DEFAULT_TIME_LIMIT_S``. An unusable environment value is ignored.

    Returns:
        float: Budget, s (finite and positive).

    Raises:
        ValueError: An explicit budget is not finite and positive.
    """
    # [LIBRARY change -> upstream PR to kwellis/woffl] Patch 47.
    if time_limit_s is not None:
        value = float(time_limit_s)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Allocation time limit must be finite and positive (s)")
        return value
    try:
        value = float(os.environ.get(TIME_LIMIT_ENV, "") or "nan")
    except ValueError:
        value = float("nan")
    return value if np.isfinite(value) and value > 0 else DEFAULT_TIME_LIMIT_S


class AllocationDeadline:
    """One wall-clock deadline shared by every solve of one allocation call.

    HiGHS receives the remaining time as ``time_limit`` and CP-SAT as
    ``max_time_in_seconds``; capacity-refinement re-solves and the CP-SAT ->
    MILP qualification fallback draw on the same budget.

    Args:
        time_limit_s (float | None): Budget, s; see :func:`resolve_time_limit`.
    """

    # [LIBRARY change -> upstream PR to kwellis/woffl] Patch 47.
    def __init__(self, time_limit_s: float | None = None) -> None:
        self.time_limit_s = resolve_time_limit(time_limit_s)
        self._end = time.monotonic() + self.time_limit_s

    def remaining(self) -> float:
        """Seconds left, never negative (float, s)."""
        return max(0.0, self._end - time.monotonic())

    def solver_seconds(self) -> float:
        """Budget to hand one native solve (float, s)."""
        return max(self.remaining(), _MIN_SOLVE_S)

    @property
    def expired(self) -> bool:
        """True once the budget is spent."""
        return time.monotonic() >= self._end


def _time_limit_error(solver: str, deadline: AllocationDeadline, required, **details):
    return AllocationError(
        f"{solver.upper()} time limit ({deadline.time_limit_s:g} s) reached without a "
        "capacity-qualified incumbent", "unknown", solver=solver, reason="time_limit",
        time_limit_s=deadline.time_limit_s, required_wells=sorted(required), **details)


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


# [LIBRARY change -> upstream PR to kwellis/woffl] Patch 49: whole-column
# reads replace one .iloc lookup per candidate.
def _rates(candidates, key) -> np.ndarray:
    """One rate column over every candidate table, in record order (float)."""
    parts = [np.asarray(df[key], dtype=float) for df in candidates if len(df)]
    return np.concatenate(parts) if parts else np.zeros(0)


def _installed_flags(candidates) -> np.ndarray:
    """1.0 for rows that keep the installed pump, in record order (float)."""
    parts = [(df["pump_state"] == "installed").to_numpy(dtype=float) if "pump_state" in df
             else np.zeros(len(df)) for df in candidates if len(df)]
    return np.concatenate(parts) if parts else np.zeros(0)


def _well_spans(sizes):
    """(start, stop) record slices per well."""
    stops = np.cumsum(sizes)
    return [(int(stop - size), int(stop)) for size, stop in zip(sizes, stops)]


def _tie_break_terms(sizes, oil, water, installed, price):
    """Tie-break objective terms for the MILP and their proven distortion bound.

    Among plans with equal priced objective the solver prefers more oil
    (``price > 0``) or, at ``price == 0`` where the objective is oil itself,
    less water; among plans still tied it keeps installed pumps. The terms are
    one weighted sum, so they act only on differences the solver resolves
    (HiGHS ``mip_abs_gap`` is 1e-6 in objective units); the installed bonus
    stays below half of any one well's smallest oil/water step.

    With ``S = sum_w max_k(|oil| + price*|water|)`` (BOPD) and ``B =
    TIE_BREAK_REL * max(1, S)``, the oil/water level spans at most ``B/2`` and
    the installed level ``B/4`` over any plan. If ``x*`` maximizes the primary
    objective ``F`` and ``x^`` maximizes ``F + T``, then ``F(x*) - F(x^) <=
    T(x^) - T(x*) <= sum_w [max(0, max_k T_wk) - min(0, min_k T_wk)]``, the
    returned ``bound`` (<= 0.75 B), before the solver's own gap.

    Args:
        sizes (list): Candidate count per well.
        oil (np.ndarray): Oil per record, BOPD.
        water (np.ndarray): Constrained water per record, BPD.
        installed (np.ndarray): 1.0 where the record keeps the installed pump.
        price (float): Water price, BOPD per BPD.

    Returns:
        tuple: ``(tie, bound, floor)`` - per-record terms (BOPD), the
        distortion bound (BOPD) and the smallest total over plans (BOPD).
    """
    spans = [(a, b) for a, b in _well_spans(sizes) if b > a]
    gross = np.abs(oil) + price * np.abs(water)
    budget = TIE_BREAK_REL * max(1.0, fsum(float(gross[a:b].max()) for a, b in spans))
    level = oil if price > 0 else -water
    level_scale = fsum(float(np.abs(level[a:b]).max()) for a, b in spans)
    eps_level = 0.5 * budget / level_scale if level_scale > 0 else 0.0
    wells_installed = sum(bool(installed[a:b].any()) for a, b in spans)
    tie = eps_level * level
    if wells_installed:
        # Keep the installed bonus below half of any one well's smallest
        # oil/water step (off included), so it only decides level ties.
        steps = [np.diff(np.unique(np.append(level[a:b], 0.0))) for a, b in spans]
        step = min((float(s[s > 0].min()) for s in steps if np.any(s > 0)), default=np.inf)
        eps_installed = 0.25 * budget / wells_installed
        if eps_level > 0 and np.isfinite(step):
            eps_installed = min(eps_installed, 0.5 * eps_level * step)
        tie = tie + eps_installed * installed
    high = fsum(max(0.0, float(tie[a:b].max())) for a, b in spans)
    low = fsum(min(0.0, float(tie[a:b].min())) for a, b in spans)
    return tie, high - low, low


def _tie_break_label(price):
    first = "more oil among equal priced objectives" if price > 0 else "less water among equal oil"
    return f"{first}, then installed pumps"


def solve_milp_choices(names, candidates, capacity, water_key="lift_wat", price=0.0,
                       required_wells=None, time_limit_s=None, deadline=None):
    """MILP on candidate rows; return selected row indices and JSON-safe status.

    Used by the GUI and CP-SAT's precise-resource refinement. No well physics is
    recomputed. The objective is oil minus price*water (BOPD) plus bounded
    tie-break terms solved in the same MILP (``tie_break_bound`` in the status).

    Args:
        time_limit_s (float | None): Budget for this call, s; see
            :func:`resolve_time_limit`. Ignored when ``deadline`` is given.
        deadline (AllocationDeadline | None): A caller's shared deadline.

    On a time limit with a capacity-qualified incumbent the status is
    ``feasible`` with its bound and gap; without one, :class:`AllocationError`
    with status ``unknown`` and reason ``time_limit``.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csc_array, vstack

    deadline = deadline if deadline is not None else AllocationDeadline(time_limit_s)
    required = set(required_wells or ())
    _validate_problem(names, candidates, capacity, price, required)
    sizes = [len(df) for df in candidates]
    n = int(sum(sizes))
    if not n:
        return {}, dict(status="optimal", solver="milp", objective_bound=0.0, gap=0.0,
                        tie_break=_tie_break_label(price), tie_break_bound=0.0,
                        time_limit_s=deadline.time_limit_s, time_limit_reached=False,
                        required_wells=sorted(required),
                        **_selection_metrics(names, candidates, {}, capacity, price, water_key))
    well_of = np.repeat(np.arange(len(names)), sizes)
    records = list(zip(well_of.tolist(), np.concatenate([np.arange(s) for s in sizes]).tolist()))
    oil = _rates(candidates, "qoil_std")
    water = _rates(candidates, water_key)
    primary = oil - price * water
    # [LIBRARY change -> upstream PR to kwellis/woffl] Patch 48: one solve with
    # bounded tie-break terms replaces the equality-constrained second MILP.
    tie, tie_bound, tie_floor = _tie_break_terms(sizes, oil, water, _installed_flags(candidates), price)
    c = -(primary + tie)
    A_well = csc_array((np.ones(n), (well_of, np.arange(n))), shape=(len(names), n))
    A = vstack([A_well, csc_array(water.reshape(1, -1))], format="csc")
    lower = np.array([1.0 if name in required else 0.0 for name in names] + [-np.inf])
    upper = np.array([1.0] * len(names) + [capacity])
    constraints = [LinearConstraint(A, lower, upper)]
    bounds = Bounds(np.zeros(n), np.ones(n))

    def run():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unrecognized options detected:.*",
                                    category=RuntimeWarning)
            try:
                # [LIBRARY change -> upstream PR to kwellis/woffl] Patch 47: no
                # "threads" option. HiGHS keeps one process-wide scheduler; a
                # thread count differing from an earlier default-thread solve
                # fails every later run ("HiGHS Status 0: Not Set").
                return milp(c=c, constraints=constraints, bounds=bounds, integrality=np.ones(n),
                            options={"mip_rel_gap": 0.0, "mip_feasibility_tolerance": 1e-9,
                                     "time_limit": deadline.solver_seconds()})
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
    time_limited = False
    for refinements in range(20):
        if refinements and deadline.expired:
            raise _time_limit_error("milp", deadline, required, capacity_refinements=refinements)
        result = run()
        status_code = int(getattr(result, "status", 4))
        message = str(getattr(result, "message", ""))
        time_limited = time_limited or (status_code == 1 and (
            "time limit" in message.lower() or deadline.expired))
        selected = choice(result)
        status = {0: "optimal", 1: "feasible", 2: "infeasible", 3: "error", 4: "error"}.get(status_code, "unknown")
        if status not in ("optimal", "feasible") or selected is None:
            if status == "feasible" and time_limited:
                raise _time_limit_error("milp", deadline, required, solver_status=status_code,
                                        solver_message=message)
            if status == "feasible":
                status = "unknown"
            elif status == "optimal":
                status = "error"
            text = (f"MILP solver error (no usable HiGHS result; no plan was produced): {message}"
                    if status == "error" else
                    f"MILP {status}: {message or 'no qualified incumbent'}")
            raise AllocationError(text, status, solver="milp", solver_status=status_code,
                                  solver_message=message, required_wells=sorted(required))
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

    # The dual bound covers primary + tie terms; subtracting the smallest
    # possible tie total keeps it a valid bound on the primary objective.
    raw_bound = getattr(result, "mip_dual_bound", None)
    bound = (-float(raw_bound) - tie_floor
             if raw_bound is not None and np.isfinite(raw_bound) else None)
    diagnostics = dict(status=status, solver="milp", solver_status=status_code, message=message,
                       objective_bound=bound, capacity_refinements=refinements,
                       required_wells=sorted(required), tie_break=_tie_break_label(price),
                       tie_break_bound=tie_bound, time_limit_s=deadline.time_limit_s,
                       time_limit_reached=time_limited)
    diagnostics.update(_selection_metrics(names, candidates, selected, capacity, price, water_key))
    diagnostics["gap"] = (max(0.0, bound - diagnostics["objective"]) / max(1.0, abs(diagnostics["objective"]))
                          if bound is not None else None)
    return selected, diagnostics


def _solve_cp_choices(names, candidates, water_scaled, capacity, water_key, price, required, deadline):
    """CP-SAT on exact-hundredth resources; one lexicographic integer solve.

    The integer objective is ``P*K1 + t``: ``P`` is the quantized priced
    objective and ``t`` packs the tie-break levels (oil when priced, else
    negative water; then installed pumps) with ``|t| < K1``, so ``P`` is
    optimized exactly before any tie-break. Levels that would push the
    objective beyond exact double range are dropped and reported.
    """
    # [LIBRARY change -> upstream PR to kwellis/woffl] Patch 48.
    sizes = [len(df) for df in candidates]
    offsets = [a for a, _ in _well_spans(sizes)]
    spans = [(a, b) for a, b in _well_spans(sizes) if b > a]
    oil = _rates(candidates, "qoil_std")
    water = _rates(candidates, water_key)
    primary = np.floor((oil - price * water) * SCALE).astype(np.int64).tolist()
    level = (np.floor(oil * SCALE).astype(np.int64).tolist() if price > 0
             else [-int(w) for values in water_scaled for w in values])
    installed = [int(v) for v in _installed_flags(candidates)]
    primary_span = sum(max(abs(p) for p in primary[a:b]) for a, b in spans)
    level_span = sum(max(0, *level[a:b]) - min(0, *level[a:b]) for a, b in spans)
    wells_installed = sum(1 for a, b in spans if any(installed[a:b]))
    tie_break = _tie_break_label(price)
    k2 = wells_installed + 1  # installed count per plan is 0..wells_installed
    if (primary_span + 1) * k2 * (level_span + 1) > _CP_EXACT_LIMIT:
        installed, k2 = [0] * len(installed), 1
        tie_break = tie_break.rsplit(",", 1)[0] + " (installed-pump level omitted: integer range)"
    k1 = k2 * (level_span + 1)  # exceeds the span of every tie total
    if (primary_span + 1) * k1 > _CP_EXACT_LIMIT:
        level, k1 = [0] * len(level), 1
        tie_break = "omitted: integer objective range"
    ties = [t * k2 + s for t, s in zip(level, installed)]
    composite = [p * k1 + t for p, t in zip(primary, ties)]
    tie_floor = sum(min(0, *ties[a:b]) for a, b in spans)

    model = cp_model.CpModel()
    x = [[model.new_bool_var(f"w{i}_p{j}") for j in range(len(df))] for i, df in enumerate(candidates)]
    for name, well_vars in zip(names, x):
        if name in required:
            model.add_exactly_one(well_vars)
        else:
            model.add_at_most_one(well_vars)
    flat = [var for well_vars in x for var in well_vars]
    # Match the original-unit qualification tolerance. Without it, binary
    # dust just below an exact resource quantum can still shut an entire well.
    capacity_decimal = Decimal(str(float(capacity))) + Decimal(str(RESOURCE_TOLERANCE))
    capacity_scaled = int((capacity_decimal * SCALE).to_integral_value(rounding="ROUND_FLOOR"))
    model.add(cp_model.LinearExpr.weighted_sum(flat, [int(w) for values in water_scaled for w in values])
              <= capacity_scaled)
    model.maximize(cp_model.LinearExpr.weighted_sum(flat, composite))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    # [LIBRARY change -> upstream PR to kwellis/woffl] Patch 47.
    solver.parameters.max_time_in_seconds = deadline.solver_seconds()
    try:
        status_code = solver.solve(model)
    except Exception as exc:
        raise AllocationError(f"CP-SAT solver error: {exc}", "error", solver="cp-sat") from exc
    status = {cp_model.OPTIMAL: "optimal", cp_model.FEASIBLE: "feasible", cp_model.INFEASIBLE: "infeasible", cp_model.MODEL_INVALID: "error", cp_model.UNKNOWN: "unknown"}.get(status_code, "unknown")
    if status not in ("optimal", "feasible"):
        if status == "unknown" and deadline.expired:
            raise _time_limit_error("cp-sat", deadline, required, solver_status=int(status_code))
        raise AllocationError(f"CP-SAT allocation {status}", status, solver="cp-sat",
                              solver_status=int(status_code), required_wells=sorted(required))
    selected = {names[i]: (i, j) for i, vars_ in enumerate(x) for j, var in enumerate(vars_) if solver.value(var)}
    quantized = sum(primary[offsets[i] + j] for i, j in selected.values())
    primary_bound = (floor(solver.best_objective_bound) - tie_floor) // k1
    diagnostics = dict(status=status, solver="cp-sat", solver_status=int(status_code),
                       objective_bound=primary_bound / SCALE + len(names) / SCALE,
                       quantized_objective=quantized / SCALE,
                       objective_resolution_bopd=1 / SCALE, required_wells=sorted(required),
                       optimality_scope="integer priced objective; bound includes quantization",
                       tie_break=tie_break, tie_break_bound=0.0,
                       time_limit_s=deadline.time_limit_s,
                       time_limit_reached=status_code == cp_model.FEASIBLE)
    return selected, diagnostics


def optimize_jet_pumps(
    well_list: list[BatchPump],
    qpf_tot: float,
    water_key: str = "lift_wat",
    allow_shutin: bool = False,
    water_price: float = 0.0,
    all_configs: bool = False,
    required_wells=None,
    time_limit_s: float | None = None,
    deadline: AllocationDeadline | None = None,
) -> pd.DataFrame:
    """Allocate pumps under capacity, with optional per-well online requirements.

    Existing callers retain the DataFrame interface, semi-finalist default and
    off rows. ``required_wells`` forces those names online when ``allow_shutin``
    is true; otherwise every well must be online. The objective is oil (BOPD)
    minus ``water_price`` times the chosen water stream (BPD). Solver status,
    original-unit totals, bound and gap are in ``df.attrs['allocation_status']``.
    Failures raise :class:`AllocationError`, never a shutdown plan.
    ``time_limit_s`` (s) or a caller's ``deadline`` bounds every solve of the
    call together; see :func:`solve_milp_choices` for time-limit outcomes.
    """
    # [LIBRARY change -> upstream PR to kwellis/woffl] Patch 46.
    if water_key not in ("lift_wat", "totl_wat"):
        raise ValueError(f"Unknown water_key: {water_key}")
    deadline = deadline if deadline is not None else AllocationDeadline(time_limit_s)
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
                                                    water_price, required, deadline=deadline)
        diagnostics.update(requested_solver="cp-sat", refinement_reason="fractional resource coefficients")
    else:
        selected, diagnostics = _solve_cp_choices(names, candidates, water_scaled, qpf_tot,
                                                  water_key, water_price, required, deadline)
        bound = diagnostics["objective_bound"]
        if not _qualified_selection(names, candidates, selected, qpf_tot, water_key, required):
            selected, diagnostics = solve_milp_choices(names, candidates, qpf_tot, water_key,
                                                        water_price, required, deadline=deadline)
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
