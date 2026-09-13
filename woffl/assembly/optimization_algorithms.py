"""Optimization Algorithms for Multi-Well Jet Pump Sizing

This module contains algorithms for optimizing jet pump sizing across multiple
wells subject to power fluid constraints.

Methods:
    milp: Mixed-integer linear programming via scipy (exact solver)
    mckp: Multi-choice knapsack via OR-Tools CP-SAT (Kaelin's upstream solver)
"""

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from woffl.assembly.network_optimizer import NetworkOptimizer, OptimizationResult

logger = logging.getLogger(__name__)

# Batch-df marginal oil-water-ratio column for each constrained stream.
_MARG_COLS = {"lift_wat": "molwr", "totl_wat": "motwr"}


def _over_marginal_wc(ratio, threshold: float) -> bool:
    """True when a config's marginal watercut exceeds ``threshold``.

    ``ratio`` is the batch df's marginal oil-water ratio (molwr/motwr — bbl oil
    per bbl water at the margin); marginal watercut = 1 / (1 + ratio), the same
    conversion the single-well recommender uses (utils.recommend_jetpump), so
    the multi-well gate and the Batch Run recommendation agree.

    NaN/missing ratios KEEP the config (fail open — a data-quality gap must not
    silently exclude configs); ratio <= 0 means the marginal barrel buys no
    oil, which is over any threshold < 1.
    """
    if ratio is None:
        return False
    try:
        f = float(ratio)
    except (TypeError, ValueError):
        return False
    if f != f:  # NaN
        return False
    if f <= 0.0:
        return True
    return (1.0 / (1.0 + f)) > threshold


def _marginal_wc_settings(optimizer: "NetworkOptimizer") -> tuple[float, bool]:
    """(threshold, active). The gate only engages below 1.0 so the default
    pad-page setting (1.0 = no economic cut) is bit-identical to the old
    unenforced behavior."""
    mwc = float(getattr(optimizer, "marginal_watercut", 1.0) or 1.0)
    return mwc, mwc < 1.0


# [LIBRARY change -> upstream PR to kwellis/woffl] water price λ in both
# solvers: marginal_wc_to_lambda / water_price / derive_lambda, the priced
# objective in milp_optimization and mckp_optimization, and _valid_configs
# accepting the blank error markers (docs/upstream_sync.md #36).
def marginal_wc_to_lambda(mwc: float) -> float:
    """The water price (bbl oil per bbl water) equivalent to a marginal
    water-cut gate: a pump whose incremental barrel is more than ``mwc``
    water has oil-per-water ratio r < (1 - mwc) / mwc. ``mwc >= 1`` (no
    gate) is a price of 0."""
    mwc = float(mwc)
    if mwc >= 1.0:
        return 0.0
    if mwc <= 0.0:
        return float("inf")
    return (1.0 - mwc) / mwc


def water_price(optimizer: "NetworkOptimizer") -> float:
    """The λ every solver prices machine water at, bbl oil per bbl water.

    ``optimizer.water_price`` when the run set one (the redesign's knob, see
    docs/optimization_redesign_2026-09.md); else the legacy
    ``marginal_watercut`` gate converted to the same units so an old caller
    gets the same economics through the objective instead of a filter.
    """
    lam = getattr(optimizer, "water_price", None)
    if lam is not None:
        value = float(lam)
        if not np.isfinite(value):
            raise ValueError("Water price must be finite")
        return max(0.0, value)
    mwc, active = _marginal_wc_settings(optimizer)
    return marginal_wc_to_lambda(mwc) if active else 0.0


def derive_lambda(
    batch_results: dict, cap: float, water_key: str = "lift_wat"
) -> tuple[float, bool]:
    """The budget's own shadow price λ* (bbl oil per bbl water), from the
    pad's pooled oil-per-water Pareto frontier: spend ``cap`` on the best
    marginal segments first; λ* is the ratio of the segment that crosses the
    budget. ``(0.0, True)`` when every segment fits (slack) or there is
    nothing to pool. Same pooling as ``derive_pad_marginal_wc``; this is the
    equal-slope price (Kanu 1981) the CFP engine already uses, so every pad
    engine now speaks the same units.
    """
    if cap is None or cap <= 0:
        return 0.0, True
    segments: list[tuple[float, float]] = []
    for bp in (batch_results or {}).values():
        df = bp if hasattr(bp, "columns") else getattr(bp, "df", None)
        if df is None or not hasattr(df, "columns"):
            continue
        frontier = _pareto_frontier(_valid_configs(df), water_key)
        segments.extend(_frontier_segments(frontier))
    if not segments:
        return 0.0, True
    segments.sort(key=lambda s: s[1], reverse=True)
    cumulative = 0.0
    for water_delta, ratio in segments:
        if cumulative + water_delta <= cap:
            cumulative += water_delta
            continue
        return max(0.0, float(ratio)), False
    return 0.0, True


# ---------------------------------------------------------------------------
# Pad-level marginal-WC auto-derivation + parsimony tie-break (pad optimizer)
# ---------------------------------------------------------------------------
#
# The pad optimizer's marginal-watercut gate above was always a hand-set
# threshold. ``derive_pad_marginal_wc`` instead reads it off the pad's OWN
# physical limits: pool every well's oil-per-water Pareto frontier, spend the
# plant's PF/water budget on the best ratios first, and the gate is the ratio
# of the segment that exhausts the budget — no ratio worse than that would
# have bought oil the plant can even deliver. ``apply_parsimony`` is a
# separate tie-break: among configs within ``threshold_bopd`` of the chosen
# well's oil, prefer the one that spends the least water (the field case that
# motivated this: a well upsized 13C->15B for ~2 BOPD at +1,500 BPD PF).


def _valid_configs(df: "pd.DataFrame", *, deduplicate: bool = True) -> "pd.DataFrame":
    """Batch-df rows the solver actually converged on (``error`` == "na" —
    the literal sentinel ``BatchPump._run_core`` writes on success). Falls
    back to ``qoil_std`` non-null when the "error" column isn't present
    (older/mocked frames), so this stays usable outside the full BatchPump
    pipeline. ``deduplicate=False`` counts successful simulations independently
    of identical hardware outcomes removed from the decision set. Never mutates
    ``df``."""
    if "error" in df.columns:
        err = df["error"]
        # "na" is the success sentinel; an empty / missing cell (mocked or
        # older frames) is also not an error. Anything else is a solver message.
        valid = df[err.isna() | err.astype(str).str.strip().isin(("na", ""))]
    else:
        valid = df
    # [LIBRARY change -> upstream PR to kwellis/woffl] Patch 46: a success
    # marker does not make NaN, infinite or negative production rates usable.
    from woffl.assembly.network import valid_rate_rows
    valid = valid_rate_rows(valid)
    # [LIBRARY change -> upstream PR to kwellis/woffl] no changeout for an
    # exactly identical modeled outcome. Keep both candidates in batch reports.
    if deduplicate and "pump_state" in valid:
        keys = [k for k in ("nozzle", "throat", "qoil_std", "lift_wat", "form_wat", "psu_solv", "sonic_status", "mach_te") if k in valid]
        valid = valid.sort_values("pump_state", kind="stable").drop_duplicates(keys)
    return valid


def _pareto_frontier(df: "pd.DataFrame", water_key: str) -> list[tuple[float, float]]:
    """(water, oil) Pareto frontier from a well's valid configs: sort by
    water ascending, keep a row only when its oil strictly exceeds every
    lower-water row's oil (ties in water are broken oil-descending first, so
    a tied lower-oil row can never sneak onto the frontier). Pure — never
    mutates ``df``."""
    if water_key not in df.columns or "qoil_std" not in df.columns:
        return []
    from woffl.assembly.network import valid_rate_rows
    pairs = valid_rate_rows(df)[[water_key, "qoil_std"]].dropna()
    if pairs.empty:
        return []
    ordered = pairs.sort_values(by=[water_key, "qoil_std"], ascending=[True, False])
    frontier: list[tuple[float, float]] = []
    best_oil = float("-inf")
    for water, oil in zip(ordered[water_key].tolist(), ordered["qoil_std"].tolist()):
        water, oil = float(water), float(oil)
        if oil > best_oil:
            frontier.append((water, oil))
            best_oil = oil
    return frontier


def _frontier_segments(
    frontier: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Marginal segments of the upper concave hull, anchored at shutdown.

    This is a continuous relaxation diagnostic, not the discrete allocation's
    exact marginal value. Concavification prevents pooling a high-return later
    step before buying its lower-return prerequisite. Free oil at zero water
    changes the origin's oil but consumes no budget.
    """
    # [LIBRARY change -> upstream PR to kwellis/woffl] Patch 46.
    hull = [(0.0, 0.0)]
    for water, oil in frontier:
        if water == 0:
            hull[0] = (0.0, max(hull[0][1], oil))
            continue
        while len(hull) >= 2:
            a, b = hull[-2:]
            if (b[1] - a[1]) * (water - b[0]) > (oil - b[1]) * (b[0] - a[0]):
                break
            hull.pop()
        hull.append((water, oil))
    segments = []
    prev_water, prev_oil = hull[0]
    for water, oil in hull[1:]:
        dw = water - prev_water
        if dw > 0:
            segments.append((dw, (oil - prev_oil) / dw))
        prev_water, prev_oil = water, oil
    return segments


def derive_pad_marginal_wc(
    batch_results: dict, cap: float, water_key: str = "lift_wat"
) -> tuple[float, bool]:
    """Derive the pad optimizer's marginal-watercut gate from the pad's OWN
    physical limits, instead of a hand-set threshold.

    Pools every well's oil-per-water marginal segments (see
    ``_frontier_segments``), sorts them best-ratio-first, and spends the
    plant's budget (``cap`` — e.g. ``plant.flow_window(n_pumps)[1]`` for the
    fixed-point coupling, or ``plant.budget_at_pressure(P, n_pumps)`` per
    trial header in the pressure sweep) on the best ratios first. The gate is
    ``1 / (1 + r*)`` where ``r*`` is the ratio of the segment that crosses
    the budget — the same oil-per-water -> watercut conversion
    ``_over_marginal_wc`` already uses, so the derived gate plugs straight
    into the existing MILP/MCKP enforcement unchanged.

    Args:
        batch_results: well_name -> object with a ``.df`` BatchPump frame
            (duck-typed — a plain dict of DataFrames also works, and a bare
            DataFrame value is accepted directly for tests).
        cap: the plant's PF/water budget at the header this derivation is
            for. ``cap <= 0`` returns ``(1.0, True)`` — see the note below.
        water_key: which water column pools the segments ("lift_wat" or
            "totl_wat"). Reads ``df[water_key]`` and ``df["qoil_std"]``
            directly — the batch df's ``molwr``/``motwr`` marginal-ratio
            columns are NOT needed: this derivation computes its own
            marginal ratios straight off each well's frontier, independent
            of whatever per-row marginal the batch sweep happened to fit.

    Returns:
        ``(gate, slack)``. ``gate`` is the marginal-watercut threshold to
        hand ``NetworkOptimizer.marginal_watercut``. ``slack`` is True when
        every pooled segment fit inside ``cap`` without a crossing (no
        well's economics needed trimming at this budget — gate is the
        pass-everything 1.0) or when there was nothing to pool at all (no
        well had a valid, non-degenerate config).

    ``cap <= 0`` is a degenerate-BUDGET case, not a "gate everything" case: a
    plant with no PF/water to give will already deliver zero through the
    MILP/MCKP water-budget CONSTRAINT regardless of the marginal-WC gate, so
    setting the gate to 1.0 (no additional economic pruning) here just avoids
    the reconciliation table misreporting "above marginal WC" for a well that
    was actually cut by the budget itself — the budget constraint does that
    job already.
    """
    if cap is None or cap <= 0:
        return 1.0, True

    segments: list[tuple[float, float]] = []
    for bp in (batch_results or {}).values():
        df = bp if hasattr(bp, "columns") else getattr(bp, "df", None)
        if df is None or not hasattr(df, "columns"):
            continue
        frontier = _pareto_frontier(_valid_configs(df), water_key)
        segments.extend(_frontier_segments(frontier))

    if not segments:
        return 1.0, True

    segments.sort(key=lambda s: s[1], reverse=True)

    cumulative = 0.0
    for water_delta, ratio in segments:
        if cumulative + water_delta <= cap:
            cumulative += water_delta
            continue
        return 1.0 / (1.0 + ratio), False

    return 1.0, True  # every segment fit inside the budget -> slack


def apply_parsimony(
    results: list,
    optimizer: "NetworkOptimizer",
    water_key: str = "lift_wat",
    threshold_bopd: float = 20.0,
) -> tuple[list, list[dict]]:
    """Parsimony tie-break: don't spend PF/water on noise-level oil gains.

    For each result, look among that well's OTHER valid batch configs for
    the one with the LEAST water among those giving up at most
    ``threshold_bopd`` oil vs. the chosen config (tie on water -> higher
    oil), and swap to it when found. Never swaps a non-sonic chosen config
    for a sonic one (don't trade a sonic warning in for a parsimony win).
    ``threshold_bopd <= 0`` disables the pass entirely.

    Args:
        results: the optimizer's per-well results (``OptimizationResult`` or
            anything duck-typed the same way: ``well_name`` /
            ``recommended_nozzle`` / ``recommended_throat`` /
            ``predicted_oil_rate`` / ``predicted_lift_water`` /
            ``predicted_total_water`` / ``sonic_status``).
        optimizer: the NetworkOptimizer the results came from (read for
            ``batch_results`` + ``get_pump_performance``).
        water_key: "lift_wat" or "totl_wat" — which water stream the
            "least water" comparison uses.
        threshold_bopd: max oil to give up for the swap. <= 0 disables.

    Returns:
        ``(new_results, swaps)``. ``new_results`` is a NEW list — the input
        list/objects are never mutated; wells that don't swap keep their
        original result object. ``swaps`` is a list of ``{well, from_pump,
        to_pump, oil_given_up, pf_saved}`` dicts, one per well that swapped
        (empty when none did).
    """
    if not results or threshold_bopd <= 0:
        return list(results or []), []

    from woffl.assembly.network_optimizer import OptimizationResult

    result_water_attr = {
        "lift_wat": "predicted_lift_water",
        "totl_wat": "predicted_total_water",
    }.get(water_key, "predicted_lift_water")
    perf_water_key = {"lift_wat": "lift_water", "totl_wat": "total_water"}.get(
        water_key, "lift_water"
    )
    marg_perf_key = {
        "lift_wat": "marginal_oil_lift_water",
        "totl_wat": "marginal_oil_total_water",
    }.get(water_key, "marginal_oil_lift_water")

    batch_results = getattr(optimizer, "batch_results", None) or {}
    new_results: list = []
    swaps: list[dict] = []

    for r in results:
        chosen_water = getattr(r, result_water_attr, None)
        if chosen_water is None:
            chosen_water = getattr(r, "predicted_lift_water", None)
        chosen_oil = getattr(r, "predicted_oil_rate", None)
        chosen_sonic = bool(getattr(r, "sonic_status", False))

        bp = batch_results.get(r.well_name)
        df = getattr(bp, "df", None) if bp is not None else None
        if (
            df is None
            or chosen_water is None
            or chosen_oil is None
            or water_key not in df.columns
        ):
            new_results.append(r)
            continue

        valid = _valid_configs(df)
        if valid.empty:
            new_results.append(r)
            continue

        candidates = valid[
            (valid[water_key] < chosen_water)
            & (valid["qoil_std"] >= chosen_oil - threshold_bopd)
        ]
        if not chosen_sonic and "sonic_status" in candidates.columns:
            candidates = candidates[~candidates["sonic_status"].astype(bool)]

        if candidates.empty:
            new_results.append(r)
            continue

        candidates = candidates.sort_values(
            by=[water_key, "qoil_std"], ascending=[True, False]
        )
        best = candidates.iloc[0]
        new_nozzle, new_throat = str(best["nozzle"]), str(best["throat"])

        perf = optimizer.get_pump_performance(r.well_name, new_nozzle, new_throat, **({"pump_state": best["pump_state"]} if "pump_state" in best else {}))
        if perf is None:
            new_results.append(r)
            continue

        new_r = OptimizationResult(
            well_name=r.well_name,
            recommended_nozzle=new_nozzle,
            recommended_throat=new_throat,
            allocated_power_fluid=perf["lift_water"],
            predicted_oil_rate=perf["oil_rate"],
            predicted_formation_water=perf["formation_water"],
            predicted_lift_water=perf["lift_water"],
            suction_pressure=perf["suction_pressure"],
            marginal_oil_rate=perf.get(marg_perf_key, 0.0),
            sonic_status=perf["sonic_status"],
            mach_te=perf["mach_te"],
                # [LIBRARY change -> upstream PR to kwellis/woffl]
                pump_state=perf.get("pump_state"),
        )
        new_results.append(new_r)
        swaps.append(
            {
                "well": r.well_name,
                "from_pump": f"{r.recommended_nozzle}{r.recommended_throat}",
                "to_pump": f"{new_nozzle}{new_throat}",
                "oil_given_up": chosen_oil - perf["oil_rate"],
                "pf_saved": chosen_water - perf[perf_water_key],
            }
        )

    return new_results, swaps


class _WellView:
    """Duck-typed BatchPump view (wellname + df) handed to the MCKP solver, so
    the marginal-watercut filter never mutates the cached BatchPump results."""

    __slots__ = ("wellname", "df")

    def __init__(self, wellname: str, df: "pd.DataFrame") -> None:
        self.wellname = wellname
        self.df = df


def _allocation_candidates(optimizer):
    """One validated candidate/performance table for both allocation engines."""
    from woffl.assembly.network import valid_rate_rows

    if not optimizer.batch_results:
        raise ValueError("Must run batch simulations before optimization")
    names, candidates, excluded = [], [], {}
    for well in optimizer.wells:
        name = well.well_name
        bp = optimizer.batch_results.get(name)
        rows = []
        if bp is not None:
            for _, row in _valid_configs(bp.df).iterrows():
                perf = optimizer.get_pump_performance(name, row["nozzle"], row["throat"],
                    **({"pump_state": row["pump_state"]} if "pump_state" in row else {}))
                if perf is None:
                    continue
                rows.append(dict(nozzle=row["nozzle"], throat=row["throat"],
                    pump_state=perf.get("pump_state"), qoil_std=perf["oil_rate"],
                    lift_wat=perf["lift_water"], form_wat=perf["formation_water"],
                    totl_wat=perf["total_water"], perf=perf))
        df = pd.DataFrame(rows, columns=["nozzle", "throat", "pump_state", "qoil_std",
                                        "lift_wat", "form_wat", "totl_wat", "perf"])
        df = valid_rate_rows(df).reset_index(drop=True)
        names.append(name)
        candidates.append(df)
        excluded[name] = int(len(bp.df) - len(df)) if bp is not None else 0
    return names, candidates, excluded


def _allocate(optimizer, water_key, method):
    # [LIBRARY change -> upstream PR to kwellis/woffl] Patch 46: common valid
    # candidates, required online wells and explicit successful/failure status.
    from woffl.assembly.network import AllocationError, optimize_jet_pumps, solve_milp_choices
    from woffl.assembly.network_optimizer import OptimizationResult

    if water_key not in _MARG_COLS:
        raise ValueError(f"Unknown water_key: {water_key}. Use 'lift_wat' or 'totl_wat'")
    optimizer.optimization_results = None
    optimizer.allocation_status = None
    if not optimizer.batch_results:
        raise ValueError("Must run batch simulations before optimization")
    optimizer.mwc_excluded = {}
    optimizer.mwc_excluded_wells = []
    optimizer.mckp_skipped = []
    required = set(getattr(optimizer, "required_wells", None) or ())
    excluded = {}
    try:
        names, candidates, excluded = _allocation_candidates(optimizer)
        optimizer.mckp_skipped = [name for name, df in zip(names, candidates) if df.empty]
        lam = water_price(optimizer)
        optimizer.lambda_used = lam
        if method == "milp":
            selected, status = solve_milp_choices(names, candidates, optimizer.power_fluid.total_rate,
                                                  water_key, lam, required)
        else:
            table = optimize_jet_pumps(
                well_list=[_WellView(name, df) for name, df in zip(names, candidates)],
                qpf_tot=optimizer.power_fluid.total_rate, water_key=water_key,
                allow_shutin=True, water_price=lam, all_configs=True, required_wells=required)
            status = dict(table.attrs.get("allocation_status") or {})
            selected = {}
            for _, row in table.iterrows():
                if row["nozzle"] == "off":
                    continue
                i = names.index(row["wellname"])
                df = candidates[i]
                mask = (df["nozzle"] == row["nozzle"]) & (df["throat"] == row["throat"])
                state = row.get("pump_state")
                if pd.notna(state):
                    mask &= df["pump_state"] == state
                hits = df.index[mask]
                if not len(hits):
                    raise AllocationError(f"Selected pump for {row['wellname']} has no matching performance", "error")
                selected[row["wellname"]] = (i, int(hits[0]))
        status.update(excluded_candidates=excluded,
                      unsupported_wells=[name for name, df in zip(names, candidates) if df.empty])
        optimizer.allocation_status = status
    except AllocationError as exc:
        optimizer.allocation_status = {**exc.details, "excluded_candidates": excluded}
        raise
    except Exception as exc:
        error = AllocationError(f"{method.upper()} allocation error: {exc}", "error", solver=method)
        optimizer.allocation_status = error.details
        raise error from exc

    marginal = "marginal_oil_lift_water" if water_key == "lift_wat" else "marginal_oil_total_water"
    results = []
    for name, (i, j) in selected.items():
        row = candidates[i].iloc[j]
        perf = row["perf"]
        results.append(OptimizationResult(
            well_name=name, recommended_nozzle=row["nozzle"], recommended_throat=row["throat"],
            allocated_power_fluid=perf["lift_water"], predicted_oil_rate=perf["oil_rate"],
            predicted_formation_water=perf["formation_water"], predicted_lift_water=perf["lift_water"],
            suction_pressure=perf["suction_pressure"], marginal_oil_rate=perf[marginal],
            sonic_status=perf["sonic_status"], mach_te=perf["mach_te"], pump_state=perf.get("pump_state")))
    optimizer.optimization_results = results
    return results


def milp_optimization(optimizer: "NetworkOptimizer", water_key: str = "lift_wat") -> list["OptimizationResult"]:
    """Original-unit MILP allocation; status is retained on the optimizer."""
    return _allocate(optimizer, water_key, "milp")


def mckp_optimization(optimizer: "NetworkOptimizer", water_key: str = "lift_wat") -> list["OptimizationResult"]:
    """CP-SAT allocation with precise-resource refinement and typed failure."""
    return _allocate(optimizer, water_key, "mckp")


def optimize(
    optimizer: "NetworkOptimizer", method: str = "milp", water_key: str = "lift_wat"
) -> list["OptimizationResult"]:
    """Main optimization dispatcher

    Args:
        optimizer: NetworkOptimizer instance
        method: Optimization method ('milp' or 'mckp')
        water_key: Constrained water stream — 'lift_wat' (PF-only budget)
            or 'totl_wat' (lift + formation, full-POPS pad pump limit)

    Returns:
        List of OptimizationResult objects

    Raises:
        ValueError: If method is not recognized
    """
    # [LIBRARY change -> upstream PR to kwellis/woffl] Standalone defaults
    # are no-op contexts; the host coordinates allocations with physics jobs.
    from woffl.assembly import compute_runtime
    with compute_runtime.cpu_slot(), compute_runtime.measure("compute.allocation"):
        if method == "milp":
            return milp_optimization(optimizer, water_key=water_key)
        elif method == "mckp":
            return mckp_optimization(optimizer, water_key=water_key)
        else:
            raise ValueError(f"Unknown optimization method: {method}. Use 'milp' or 'mckp'")
