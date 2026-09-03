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
        return max(0.0, float(lam))
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


def _valid_configs(df: "pd.DataFrame") -> "pd.DataFrame":
    """Batch-df rows the solver actually converged on (``error`` == "na" —
    the literal sentinel ``BatchPump._run_core`` writes on success). Falls
    back to ``qoil_std`` non-null when the "error" column isn't present
    (older/mocked frames), so this stays usable outside the full BatchPump
    pipeline. Never mutates ``df``."""
    if "error" in df.columns:
        err = df["error"]
        # "na" is the success sentinel; an empty / missing cell (mocked or
        # older frames) is also not an error. Anything else is a solver message.
        valid = df[err.isna() | err.astype(str).str.strip().isin(("na", ""))]
    else:
        valid = df
    if "qoil_std" in valid.columns:
        valid = valid[valid["qoil_std"].notna()]
    return valid


def _pareto_frontier(df: "pd.DataFrame", water_key: str) -> list[tuple[float, float]]:
    """(water, oil) Pareto frontier from a well's valid configs: sort by
    water ascending, keep a row only when its oil strictly exceeds every
    lower-water row's oil (ties in water are broken oil-descending first, so
    a tied lower-oil row can never sneak onto the frontier). Pure — never
    mutates ``df``."""
    if water_key not in df.columns or "qoil_std" not in df.columns:
        return []
    pairs = df[[water_key, "qoil_std"]].dropna()
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
    """(water_delta, oil_per_water) marginal segments along a frontier,
    anchored at the origin (a well producing nothing costs nothing). The
    first frontier point sitting exactly at water=0 contributes no segment —
    it's free oil, nothing to pool against the budget."""
    segments = []
    prev_water, prev_oil = 0.0, 0.0
    for water, oil in frontier:
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

        perf = optimizer.get_pump_performance(r.well_name, new_nozzle, new_throat)
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


def milp_optimization(
    optimizer: "NetworkOptimizer", water_key: str = "lift_wat"
) -> list["OptimizationResult"]:
    """Optimal allocation via mixed-integer linear programming.

    Formulates pump selection as a multiple-choice knapsack problem and
    solves it exactly using MILP.  Each well may be assigned at most one
    pump configuration (nozzle/throat).  The solver maximizes total oil
    production subject to the water-budget constraint.

    Args:
        optimizer: NetworkOptimizer instance with batch results already run
        water_key: Which water stream the budget constrains. "lift_wat"
            (power fluid only — PF-only POPS pads, where formation water
            passes through to the plant) or "totl_wat" (lift + formation —
            full-POPS pads whose pad pump handles the entire stream).

    Returns:
        List of OptimizationResult objects

    Raises:
        ValueError: If batch results haven't been run or water_key unknown
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csc_array
    from scipy.sparse import vstack as sp_vstack

    from woffl.assembly.network_optimizer import OptimizationResult

    if not optimizer.batch_results:
        raise ValueError("Must run batch simulations before optimization")

    try:
        perf_key = {"lift_wat": "lift_water", "totl_wat": "total_water"}[water_key]
        marg_key = {
            "lift_wat": "marginal_oil_lift_water",
            "totl_wat": "marginal_oil_total_water",
        }[water_key]
    except KeyError:
        raise ValueError(
            f"Unknown water_key: {water_key}. Use 'lift_wat' or 'totl_wat'"
        ) from None

    # ── Build decision-variable list ──────────────────────────────────────
    # Each variable x[k] ∈ {0,1} represents selecting config k for a well.
    configs: list[dict] = []
    well_names: list[str] = []

    # Every converged config is a candidate. The economics live in the
    # OBJECTIVE (oil - λ·water), not in a pre-filter: the old marginal-WC
    # gate read a per-row ratio that existed only on semifinalist rows and
    # failed open elsewhere (review 2026-09-01, OPT-A3), and it could not
    # re-spend the water it freed. See docs/optimization_redesign_2026-09.md.
    lam = water_price(optimizer)
    optimizer.mwc_excluded = {}
    optimizer.mwc_excluded_wells = []
    optimizer.lambda_used = lam

    for well in optimizer.wells:
        wn = well.well_name
        if wn not in optimizer.batch_results:
            continue
        batch_pump = optimizer.batch_results[wn]
        successful = _valid_configs(batch_pump.df)
        for _, row in successful.iterrows():
            perf = optimizer.get_pump_performance(wn, row["nozzle"], row["throat"])
            if perf is None:
                continue
            configs.append(
                {
                    "well_name": wn,
                    "nozzle": row["nozzle"],
                    "throat": row["throat"],
                    "perf": perf,
                }
            )
        if wn not in well_names:
            well_names.append(wn)

    if not configs:
        optimizer.optimization_results = []
        return []

    n = len(configs)
    n_wells = len(well_names)
    well_idx = {wn: i for i, wn in enumerate(well_names)}

    # ── Objective: maximize Σ (oil − λ·water)  →  minimize the negative ────
    c = np.array(
        [-(cfg["perf"]["oil_rate"] - lam * cfg["perf"][perf_key]) for cfg in configs]
    )

    # ── Constraint 1: at most one config per well ─────────────────────────
    row_ids, col_ids, vals = [], [], []
    for k, cfg in enumerate(configs):
        row_ids.append(well_idx[cfg["well_name"]])
        col_ids.append(k)
        vals.append(1.0)
    A_well = csc_array((vals, (row_ids, col_ids)), shape=(n_wells, n))

    # ── Constraint 2: Σ water[k]*x[k] ≤ budget ──────────────────────────
    # water_key picks the constrained stream: lift water (PF budget) or
    # lift + formation (full-POPS pad pump limit).
    pf_vals = np.array([[cfg["perf"][perf_key] for cfg in configs]])
    A_pf = csc_array(pf_vals)

    A = sp_vstack([A_well, A_pf], format="csc")
    b_upper = np.concatenate([np.ones(n_wells), [optimizer.power_fluid.total_rate]])
    b_lower = np.full(n_wells + 1, -np.inf)

    constraints = LinearConstraint(A, lb=b_lower, ub=b_upper)

    # ── Bounds & integrality ──────────────────────────────────────────────
    bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))
    integrality = np.ones(n)  # all binary

    result = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)

    if not result.success:
        optimizer.optimization_results = []
        return []

    # ── Extract selected configurations ───────────────────────────────────
    selected = np.where(result.x > 0.5)[0]

    results = []
    for k in selected:
        cfg = configs[k]
        perf = cfg["perf"]
        results.append(
            OptimizationResult(
                well_name=cfg["well_name"],
                recommended_nozzle=cfg["nozzle"],
                recommended_throat=cfg["throat"],
                allocated_power_fluid=perf["lift_water"],
                predicted_oil_rate=perf["oil_rate"],
                predicted_formation_water=perf["formation_water"],
                predicted_lift_water=perf["lift_water"],
                suction_pressure=perf["suction_pressure"],
                marginal_oil_rate=perf[marg_key],
                sonic_status=perf["sonic_status"],
                mach_te=perf["mach_te"],
            )
        )

    optimizer.optimization_results = results
    return results


def mckp_optimization(
    optimizer: "NetworkOptimizer", water_key: str = "lift_wat"
) -> list["OptimizationResult"]:
    """Optimal allocation via Multi-Choice Knapsack (OR-Tools CP-SAT).

    Bridges the GUI's NetworkOptimizer interface to
    network.optimize_jet_pumps() which uses the CP-SAT constraint solver.

    Each well picks exactly one jet pump from its semi-finalists to maximize
    total oil production subject to the power-fluid budget constraint.

    Args:
        optimizer: NetworkOptimizer instance with batch results already run

    Returns:
        List of OptimizationResult objects

    Raises:
        ValueError: If batch results haven't been run
    """
    from woffl.assembly.network import optimize_jet_pumps
    from woffl.assembly.network_optimizer import OptimizationResult

    if not optimizer.batch_results:
        raise ValueError("Must run batch simulations before optimization")

    try:
        marg_col = _MARG_COLS[water_key]
        marg_perf_key = {
            "lift_wat": "marginal_oil_lift_water",
            "totl_wat": "marginal_oil_total_water",
        }[water_key]
    except KeyError:
        raise ValueError(
            f"Unknown water_key: {water_key}. Use 'lift_wat' or 'totl_wat'"
        ) from None

    lam = water_price(optimizer)
    optimizer.lambda_used = lam

    # The SAME candidate set as the MILP: every converged config (not the
    # upstream "semi-finalist" subset, which kept only the best throat per
    # nozzle and so could never pick 12A over 12B). One bad well contributes
    # no configs rather than aborting the run.
    batch_pumps = []
    skipped = []
    for well in optimizer.wells:
        wn = well.well_name
        if wn not in optimizer.batch_results:
            continue
        df = _valid_configs(optimizer.batch_results[wn].df)
        if df.empty:
            skipped.append(wn)
            continue
        batch_pumps.append(_WellView(wn, df))

    optimizer.mwc_excluded = {}
    optimizer.mwc_excluded_wells = []
    optimizer.mckp_skipped = skipped

    if skipped:
        logger.warning(
            "MCKP: skipping %d well(s) with no converged config: %s",
            len(skipped),
            ", ".join(skipped),
        )

    if not batch_pumps:
        optimizer.optimization_results = []
        return []

    # Call upstream MCKP solver. allow_shutin=True matches the MILP's
    # "at most one" semantics (implicit shut-in when a well cannot be fed);
    # with exactly-one the model was INFEASIBLE at any tight budget and the
    # RuntimeError below escaped every caller, so one high-header trial
    # killed a whole I/M/E sweep (review 2026-09-01, OPT-A4). water_price and
    # all_configs make its objective and candidate set the MILP's.
    try:
        mckp_df = optimize_jet_pumps(
            well_list=batch_pumps,
            qpf_tot=optimizer.power_fluid.total_rate,
            water_key=water_key,
            allow_shutin=True,
            water_price=lam,
            all_configs=True,
        )
    except RuntimeError as exc:
        logger.warning("MCKP: no feasible allocation at this budget (%s); trial empty", exc)
        optimizer.optimization_results = []
        return []

    # Convert MCKP result DataFrame to OptimizationResult objects
    results = []
    for _, row in mckp_df.iterrows():
        well_name = row["wellname"]
        nozzle = str(row["nozzle"])
        throat = str(row["throat"])

        # Look up full performance from batch results
        perf = optimizer.get_pump_performance(well_name, nozzle, throat)
        if perf is None:
            continue

        results.append(
            OptimizationResult(
                well_name=well_name,
                recommended_nozzle=nozzle,
                recommended_throat=throat,
                allocated_power_fluid=perf["lift_water"],
                predicted_oil_rate=perf["oil_rate"],
                predicted_formation_water=perf["formation_water"],
                predicted_lift_water=perf["lift_water"],
                suction_pressure=perf["suction_pressure"],
                # Follow the constrained stream, matching the MILP path (a
                # totl_wat run used to report lift-water marginals here).
                marginal_oil_rate=perf[marg_perf_key],
                sonic_status=perf["sonic_status"],
                mach_te=perf["mach_te"],
            )
        )

    optimizer.optimization_results = results
    return results


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
    if method == "milp":
        return milp_optimization(optimizer, water_key=water_key)
    elif method == "mckp":
        return mckp_optimization(optimizer, water_key=water_key)
    else:
        raise ValueError(f"Unknown optimization method: {method}. Use 'milp' or 'mckp'")
