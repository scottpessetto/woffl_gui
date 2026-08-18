"""Knob sensitivity sweep - what each calibration knob does to the match.

Read-only diagnostic. Every knob the sidebar exposes is swept across a
defensible range around the CURRENT operating point and re-solved, so the
engineer can see which knobs actually move the four match quantities
(suction BHP, oil, liquid, power fluid) and which are dead weight.

No physics here: each point is a plain ``solve.solve_single`` call and
nothing is written back.

The finding this exists for: on a choked well ``jetpump_solver`` returns
``psu_minimize(tsu, ken, ate, ipr_su, prop_su)`` directly
(woffl/assembly/solopump.py:408), so power-fluid pressure, kth, kdi and
wellhead pressure produce bit-identical results across their whole range.
Those knobs come back ``inert=True`` instead of looking like live handles.

Cost: one solve is about 20 ms and the table is about 90 solves. Those
solves are independent, so the table fans out over the shared persistent
process pool (``server.pool``) in ONE flat batch across all knobs - keeping
the pool busy and letting the request thread wait on futures instead of
holding the GIL. It falls back to the identical serial loop whenever the
pool is unavailable.

The engineer can override any knob's range (``bounds`` on the request) when
they know which inputs are shaky on a given well, and can vary several knobs
TOGETHER through ``run_combine`` to answer the question one knob at a time
cannot: does any combination inside those ranges reach the measured test?
"""

from __future__ import annotations

import itertools
import logging
from math import isfinite, sqrt
from typing import Any, Callable, NamedTuple, Optional

from server import jobs, pool, schemas
from server.services import factories, solve

log = logging.getLogger("woffl.web.sensitivity")

# Points per continuous knob, inclusive of both range ends. The baseline
# value is inserted on top when the linear grid misses it, so a knob can
# report 8 points.
STEPS = 7

# An engineer-supplied override may ask for a different point count. Two is
# corners-only and fifteen is already more rows than the page can read.
STEPS_BOUNDS = (2, 15)

# Hard ceiling on one combined-permutations study. The study runs as a
# background job, so this is a resource-sanity bound, not a request-latency
# one: 10,000 solves is about 2 minutes of single-core solving and a few MB of
# JSON held in memory until the caller polls it. Past the cap the request is
# a mistake and the caller gets an error: quietly sampling a subset of a
# factorial would be a lie about what was explored.
MAX_COMBINE_RUNS = 10000

# How often the combine study publishes progress. At about 11 ms a solve
# that is roughly four ticks a second - enough for the count to look alive,
# rare enough to cost nothing.
_PROGRESS_EVERY = 25

# Grids below this stay serial even on a many-core host: ProcessPool spawn
# on Windows costs seconds per worker, which small factorials never win back.
_PARALLEL_MIN_RUNS = 50

# Test seam: swapped for ThreadPoolExecutor in tests so monkeypatched solves
# stay visible (a real child process cannot see them). None = ProcessPool.
_EXECUTOR_CLS: Optional[type] = None

# A knob counts as inert when NO swept point moves a match quantity by more
# than these. Set at the resolution an engineer would act on, comfortably
# above solver noise and well below anything worth re-matching for.
INERT_TOL = {
    "psu": 0.5,  # psi
    "qoil": 0.5,  # BOPD
    "qliq": 0.5,  # BLPD
    "qpf": 1.0,  # BWPD
}

METRICS = ("psu", "qoil", "qliq", "qpf")

# Exact wording is the point of this note - it names the branch that makes
# the inert knobs inert.
SONIC_NOTE = (
    "Baseline is on the choked-flow floor - power-fluid pressure, kth, kdi "
    "and wellhead pressure cannot move suction here."
)


class _Knob(NamedTuple):
    """One row of the frozen knob table.

    Attributes:
        id (str): Stable client-side key.
        label (str): Display name.
        field (str): SimParams field this knob writes.
        unit (str): Display unit, "" when unitless.
        kind (str): "mult" (low/high are multipliers of the baseline),
            "abs" (low/high are absolute endpoints), "delta" (low/high are
            absolute offsets from the baseline), or "catalog" (discrete
            pump identity, neighbours only).
        low (float): Low end, read per ``kind``.
        high (float): High end, read per ``kind``.
        decimals (int): Decimals used for the display labels.
        basis (str): One line explaining WHY the range is this wide.
        clamp (tuple | None): Explicit (lo, hi) override, unit of the field.
            None means use the SimParams field bounds the sidebar enforces.
    """

    id: str
    label: str
    field: str
    unit: str
    kind: str
    low: float
    high: float
    decimals: int
    basis: str
    clamp: Optional[tuple[float, float]] = None


# The frozen knob table, in the order the page renders it. Columns are
# id, label, field, unit, kind, low, high, decimals, basis [, clamp].
KNOBS: tuple[_Knob, ...] = (
    _Knob(
        "ppf_surf", "PF surface pressure", "ppf_surf", "psi", "mult", 0.85, 1.15, 0,
        "Plus or minus 15 percent covers PF gauge and pad allocation uncertainty.",
    ),
    _Knob(
        "surf_pres", "Wellhead pressure", "surf_pres", "psi", "mult", 0.80, 1.20, 0,
        "Plus or minus 20 percent covers the normal wellhead operating swing.",
    ),
    _Knob(
        "ken", "Entrance loss (ken)", "ken", "", "abs", 0.005, 0.40, 3,
        "The full fric_calibration.KEN_BOUNDS search range, 0.005 to 0.40.",
    ),
    _Knob(
        "kth", "Throat loss (kth)", "kth", "", "abs", 0.05, 1.0, 3,
        "The full fric_calibration.KTH_BOUNDS search range, 0.05 to 1.0.",
    ),
    _Knob(
        "kdi", "Diffuser loss (kdi)", "kdi", "", "abs", 0.05, 1.0, 3,
        "The full fric_calibration.KDI_BOUNDS search range, 0.05 to 1.0.",
    ),
    _Knob(
        "form_gor", "Intake GOR", "form_gor", "scf/bbl", "mult", 0.5, 1.5, 0,
        "Half to 1.5 times: free gas at the intake is the dominant choke term.",
    ),
    _Knob(
        "form_wc", "Water cut", "form_wc", "", "delta", -0.10, 0.10, 3,
        "Plus or minus 0.10 absolute watercut, clamped to 0.0 - 0.99.",
        (0.0, 0.99),
    ),
    _Knob(
        "pres", "Reservoir pressure", "pres", "psi", "mult", 0.80, 1.20, 0,
        "Plus or minus 20 percent covers IPR reservoir-pressure uncertainty.",
    ),
    _Knob(
        "qwf", "IPR anchor rate", "qwf", "BLPD", "mult", 0.85, 1.15, 0,
        "Plus or minus 15 percent covers test-rate uncertainty on the anchor.",
    ),
    _Knob(
        "pwf", "IPR anchor BHP", "pwf", "psi", "mult", 0.85, 1.15, 0,
        "Plus or minus 15 percent covers gauge uncertainty on the anchor BHP.",
    ),
    _Knob(
        "bubble_point", "Bubble point", "bubble_point", "psi", "mult", 0.80, 1.20, 0,
        "Plus or minus 20 percent, clamped to 1001 - 2999 where BlackOil validates.",
        (1001.0, 2999.0),
    ),
    _Knob(
        "form_temp", "Formation temp", "form_temp", "degF", "delta", -20.0, 20.0, 0,
        "Plus or minus 20 degF absolute covers gradient and gauge spread.",
    ),
    _Knob(
        "nozzle_no", "Nozzle size", "nozzle_no", "", "catalog", -1, 1, 0,
        "One catalog step each way - pump identity is discrete, not a dial.",
    ),
    _Knob(
        "area_ratio", "Throat size", "area_ratio", "", "catalog", -1, 1, 0,
        "One catalog step each way - pump identity is discrete, not a dial.",
    ),
)

# Catalogs for the two discrete knobs. Sourced from schemas so the sweep can
# never offer a pump the sidebar does not list.
_CATALOGS = {
    "nozzle_no": schemas.NOZZLE_OPTIONS,
    "area_ratio": schemas.THROAT_OPTIONS,
}

# Knob lookup for the requests that name knobs by id - bounds overrides and
# the combined-permutations study.
_BY_ID: dict[str, _Knob] = {knob.id: knob for knob in KNOBS}


def _field_bounds(field: str) -> tuple[float, float]:
    """The (lo, hi) bounds SimParams - and therefore the sidebar - enforces.

    Args:
        field (str): SimParams field name.

    Returns:
        tuple[float, float]: (lo, hi) in the field's own unit. A missing
        constraint falls back to (0.0, inf) so the caller still gets a
        non-negative range.
    """
    info = schemas.SimParams.model_fields[field]
    lo, hi = 0.0, float("inf")
    for con in info.metadata:
        ge = getattr(con, "ge", None)
        le = getattr(con, "le", None)
        if ge is not None:
            lo = float(ge)
        if le is not None:
            hi = float(le)
    return lo, hi


def _clamp_bounds(knob: _Knob) -> tuple[float, float]:
    """Effective clamp for one knob, in the units the sweep works in.

    Args:
        knob (_Knob): Knob table row.

    Returns:
        tuple[float, float]: (lo, hi) - catalog INDEX bounds for a discrete
        knob, the knob's explicit clamp when it has one, else the SimParams
        field bounds the sidebar enforces.
    """
    if knob.kind == "catalog":
        return 0.0, float(len(_CATALOGS[knob.field]) - 1)
    if knob.clamp is not None:
        return knob.clamp
    return _field_bounds(knob.field)


def _short(text: str, limit: int = 80) -> str:
    """First line of an exception message, truncated for the point label."""
    first = str(text).strip().splitlines()[0] if str(text).strip() else ""
    if len(first) > limit:
        return first[: limit - 3] + "..."
    return first


def _fmt(value: float, decimals: int) -> str:
    """Display label for a swept numeric value."""
    return f"{value:,.{decimals}f}"


def _baseline_value(knob: _Knob, sp: schemas.SimParams) -> float:
    """The knob's current value, in the units the sweep works in.

    Discrete knobs report their catalog index. ``bubble_point`` resolves the
    field-model preset when the sidebar leaves it unset, so the knob always
    has a real number to sweep around.

    Args:
        knob (_Knob): Knob table row.
        sp (schemas.SimParams): Current sidebar parameters.

    Returns:
        float: Baseline value, field units (catalog index when discrete).

    Raises:
        ValueError: The discrete knob's current value is not in the catalog.
    """
    if knob.kind == "catalog":
        options = _CATALOGS[knob.field]
        current = str(getattr(sp, knob.field))
        if current not in options:
            raise ValueError(f"{knob.field} '{current}' is not in the catalog")
        return float(options.index(current))

    if knob.field == "bubble_point" and sp.bubble_point is None:
        # None means "use the field-model preset"; resolve it so the sweep
        # is centered on the pressure the solver actually saw.
        oil, _wat, _gas = factories.create_pvt_components(
            field_model=sp.field_model,
            oil_api=sp.oil_api,
            gas_sg=sp.gas_sg,
            wat_sg=sp.wat_sg,
        )
        return float(oil.pbp)

    return float(getattr(sp, knob.field))


def _finite(value: float) -> Optional[float]:
    """A bound as JSON: None when it is not finite, i.e. no limit at all."""
    return float(value) if isfinite(value) else None


def _bound_text(knob: _Knob, value: float) -> str:
    """Display text for one end of a range, for the notes.

    Args:
        knob (_Knob): Knob table row.
        value (float): Range end, field units (catalog index if discrete).

    Returns:
        str: The catalog entry when the value is a real index, else the
        formatted number.
    """
    if knob.kind == "catalog":
        options = _CATALOGS[knob.field]
        idx = int(round(value))
        if 0 <= idx < len(options):
            return options[idx]
        return f"index {_fmt(value, 0)}"
    return _fmt(value, knob.decimals)


class _Sweep(NamedTuple):
    """One knob's resolved grid plus the range description the client edits.

    Attributes:
        base (float): Baseline value, field units (catalog index if discrete).
        pairs (list): Ascending (value, label) pairs with clamping, baseline
            insertion and dedupe already applied.
        default_low (float): Absolute low the frozen table asks for, before
            clamping and with no override.
        default_high (float): Absolute high the frozen table asks for.
        swept_low (float): Lowest value actually swept, 0.0 when none.
        swept_high (float): Highest value actually swept, 0.0 when none.
        note (str | None): Set when the clamp moved an end the caller asked
            for, so the sweep is not what they typed.
    """

    base: float
    pairs: list[tuple[float, str]]
    default_low: float
    default_high: float
    swept_low: float
    swept_high: float
    note: Optional[str]


def _knob_sweep(
    knob: _Knob,
    sp: schemas.SimParams,
    override: Optional[schemas.KnobBounds] = None,
    steps: int = STEPS,
    with_baseline: bool = True,
) -> _Sweep:
    """Build one knob's grid. The only grid builder in this module.

    Both the per-knob sweep and the combined-permutations study come through
    here, so clamping, snapping, dedupe and labels are defined exactly once.

    Continuous knobs get ``steps`` points spanning the range. Discrete knobs
    with no override get the catalog neighbours only; with an override they
    take the same linear grid over catalog INDICES, which lands on whole
    indices because their label carries no decimals. Every point is clamped
    into the range the sidebar enforces, so a baseline already sitting on a
    bound simply sweeps one-sided - that is not an error.

    Args:
        knob (_Knob): Knob table row.
        sp (schemas.SimParams): Current sidebar parameters.
        override (schemas.KnobBounds | None): Engineer range override in the
            knob's own units - absolute field values, or a catalog index for
            a discrete knob. Backwards ends are swapped, equal ends give a
            single point. None keeps the table range.
        steps (int): Points across the range before dedupe, clamped to
            STEPS_BOUNDS. ``override.steps`` wins when supplied.
        with_baseline (bool): Insert the baseline value into the grid. True
            for the per-knob sweep, whose excursions are measured from it;
            False for a factorial, where N levels must mean N levels.

    Returns:
        _Sweep: The grid and the range description.

    Raises:
        ValueError: A discrete knob's current value is not in the catalog.
    """
    base = _baseline_value(knob, sp)
    lo_clamp, hi_clamp = _clamp_bounds(knob)

    if knob.kind == "mult":
        default_lo, default_hi = base * knob.low, base * knob.high
    elif knob.kind == "delta":
        default_lo, default_hi = base + knob.low, base + knob.high
    elif knob.kind == "abs":
        default_lo, default_hi = knob.low, knob.high
    else:  # "catalog" - the table range is in catalog STEPS around baseline
        options = _CATALOGS[knob.field]
        idx = int(round(base))
        default_lo = float(max(0, idx + int(knob.low)))
        default_hi = float(min(len(options) - 1, idx + int(knob.high)))

    if override is None and knob.kind == "catalog":
        # Neighbours only, dropping anything past either end of the catalog.
        options = _CATALOGS[knob.field]
        idx = int(round(base))
        values = [float(i) for i in (idx - 1, idx, idx + 1) if 0 <= i < len(options)]
        return _Sweep(
            base,
            [(v, _point_label(knob, v)) for v in values],
            default_lo,
            default_hi,
            min(values) if values else 0.0,
            max(values) if values else 0.0,
            None,
        )

    if override is None:
        lo_raw, hi_raw = default_lo, default_hi
    else:
        lo_raw, hi_raw = float(override.low), float(override.high)
        if lo_raw > hi_raw:  # typed backwards; swap rather than error
            lo_raw, hi_raw = hi_raw, lo_raw
        if override.steps is not None:
            steps = int(override.steps)
    steps = min(max(steps, STEPS_BOUNDS[0]), STEPS_BOUNDS[1])

    span = hi_raw - lo_raw
    # Baseline first, and flagged so it is never snapped: the sweep must
    # contain the exact operating point the excursions are measured from, and
    # the dedupe below keeps the first value that lands on a label.
    grid: list[tuple[float, bool]] = [(base, False)] if with_baseline else []
    grid += [(lo_raw + span * i / (steps - 1), True) for i in range(steps)]

    kept: dict[float, float] = {}
    for raw, snap in grid:
        val = min(max(raw, lo_clamp), hi_clamp)
        # Snap grid points to the precision their label shows, so a narrow
        # range cannot produce two rows reading the same number (and two
        # identical solves).
        if snap:
            val = min(max(round(val, knob.decimals), lo_clamp), hi_clamp)
        # Nothing non-positive ever reaches the solver. A legitimate zero
        # bound (watercut) is allowed through; a clamped-to-zero value on a
        # strictly positive field is not.
        if val < 0.0 or (val <= 0.0 and lo_clamp > 0.0):
            continue
        kept.setdefault(round(val, knob.decimals), val)
    values = [kept[k] for k in sorted(kept)]

    note = None
    lo_use = min(max(lo_raw, lo_clamp), hi_clamp)
    hi_use = min(max(hi_raw, lo_clamp), hi_clamp)
    if override is not None and (lo_use != lo_raw or hi_use != hi_raw):
        note = (
            f"{knob.label} range {_bound_text(knob, lo_raw)} to "
            f"{_bound_text(knob, hi_raw)} was clamped to "
            f"{_bound_text(knob, lo_use)} to {_bound_text(knob, hi_use)}, "
            "the range the model accepts."
        )

    return _Sweep(
        base,
        [(v, _point_label(knob, v)) for v in values],
        default_lo,
        default_hi,
        min(values) if values else 0.0,
        max(values) if values else 0.0,
        note,
    )


def _point_label(knob: _Knob, value: float) -> str:
    """Display label for one swept value."""
    if knob.kind == "catalog":
        return _CATALOGS[knob.field][int(round(value))]
    return _fmt(value, knob.decimals)


def _field_value(knob: _Knob, value: float) -> Any:
    """The value to write into SimParams for one swept point.

    Args:
        knob (_Knob): Knob table row.
        value (float): Swept value, field units (catalog index if discrete).

    Returns:
        Any: The catalog entry for a discrete knob, else the value itself.
    """
    if knob.kind == "catalog":
        return _CATALOGS[knob.field][int(round(value))]
    return value


def _params_for(knob: _Knob, sp: schemas.SimParams, value: float) -> schemas.SimParams:
    """Copy of the sidebar params with this knob moved to ``value``.

    ``model_copy`` skips validation on purpose - every value handed here has
    already been clamped into the field's own bounds by ``_knob_sweep``.
    """
    return sp.model_copy(update={knob.field: _field_value(knob, value)})


def _metrics(res: dict[str, Any]) -> dict[str, Any]:
    """The four match quantities plus the choke diagnostics from a solve.

    Args:
        res (dict): SolveResult dict from ``solve.solve_single``.

    Returns:
        dict: psu (psig), qoil (STBOPD), qliq (BLPD), qpf (BWPD),
        mach (dimensionless), sonic (bool).
    """
    return {
        "psu": float(res["psu"]),
        "qoil": float(res["qoil_std"]),
        "qliq": float(res["qoil_std"] + res["fwat_bwpd"]),
        "qpf": float(res["qnz_bwpd"]),
        "mach": float(res["mach_te"]),
        "sonic": bool(res["sonic_status"]),
    }


def _solve_point(well: str, sp: schemas.SimParams, value: float, label: str) -> dict[str, Any]:
    """One swept solve as a SensitivityPoint dict; failures become ``error``.

    A per-point failure must never kill the sweep - a knob whose far end
    stops converging is itself a finding.

    Args:
        well (str): Selected well name ("Custom" allowed).
        sp (schemas.SimParams): Params with this knob already moved.
        value (float): Swept value, field units (catalog index if discrete).
        label (str): Display label for the value.

    Returns:
        dict: SensitivityPoint shape.
    """
    point: dict[str, Any] = {"value": float(value), "label": label}
    try:
        point.update(_metrics(solve.solve_single(well, sp)))
    except solve.SolveFailure as exc:
        point["error"] = exc.error
    except ValueError as exc:
        point["error"] = _short(str(exc)) or "invalid"
    return point


def _excursions(
    points: list[dict[str, Any]], baseline: dict[str, Any], base_value: float
) -> tuple[dict[str, Optional[float]], dict[str, Optional[float]], bool]:
    """Signed low/high excursions per metric, plus the inert verdict.

    ``low`` is the most negative point-minus-baseline the sweep produced and
    ``high`` the most positive, per metric, over every point that solved.
    Because the baseline value is always in the sweep, a metric that solved
    anywhere always brackets zero, which is what the tornado draws from.

    Args:
        points (list[dict]): SensitivityPoint dicts for one knob.
        baseline (dict): The baseline SensitivityPoint dict.
        base_value (float): The knob's baseline value, field units, so a
            sweep that collapsed onto the baseline is not called inert.

    Returns:
        tuple: (low, high, inert). low/high are keyed psu|qoil|qliq|qpf with
        None where every solve failed for that metric.
    """
    low: dict[str, Optional[float]] = {}
    high: dict[str, Optional[float]] = {}
    within_tol = True

    for metric in METRICS:
        base = baseline.get(metric)
        deltas = [
            float(p[metric]) - float(base)
            for p in points
            if base is not None and p.get(metric) is not None
        ]
        if not deltas:
            low[metric], high[metric] = None, None
            continue
        low[metric] = min(deltas)
        high[metric] = max(deltas)
        if max(abs(d) for d in deltas) > INERT_TOL[metric]:
            within_tol = False

    # Only claim inertness when the sweep actually moved somewhere off the
    # baseline value and still changed nothing; an all-failed sweep proves
    # nothing either way.
    moved = any(
        p.get("error") is None and p["value"] != base_value for p in points
    )
    return low, high, bool(within_tol and moved)


def run_sensitivity(
    well: str,
    sp: schemas.SimParams,
    targets: dict[str, Optional[float]],
    bounds: Optional[dict[str, schemas.KnobBounds]] = None,
) -> dict[str, Any]:
    """Sweep every calibration knob around the current operating point.

    Solves the baseline once, then re-solves each knob across its range and
    records the signed excursion it produces in the four match quantities.
    Read-only: nothing is persisted and no physics is touched.

    Args:
        well (str): Selected well name ("Custom" allowed).
        sp (schemas.SimParams): Current sidebar parameters (qwf = TOTAL
            LIQUID BLPD).
        targets (dict): Measured test values echoed back for the reference
            lines - target_psu (psig), target_qoil (STBOPD), target_qliq
            (BLPD), target_qpf (BWPD). Any may be None.
        bounds (dict | None): Engineer range overrides keyed by knob id, in
            each knob's own units. An entry for a knob that is not in the
            table is ignored with a note. None sweeps the table ranges.

    Returns:
        dict: SensitivityResponse shape - baseline point, one knob entry per
        table row in table order (including the range description the client
        edits), the echoed targets, and any notes.

    Raises:
        solve.SolveFailure: The BASELINE solve failed; there is nothing to
            measure excursions against, so it propagates to the usual 422.
        ValueError: Invalid inputs (router maps to 422 "invalid").
    """
    base_res = solve.solve_single(well, sp)
    baseline: dict[str, Any] = {"value": 0.0, "label": "baseline", **_metrics(base_res)}

    bounds = bounds or {}
    notes: list[str] = []
    for unknown in sorted(set(bounds) - set(_BY_ID)):
        notes.append(f"Range override for unknown knob '{unknown}' was ignored.")

    # Pass 1: build every knob's sweep, and with it ONE flat job list.
    # Solving per knob inside the loop would leave the pool idle between
    # knobs and load-balance badly (knob sweeps differ in length); flattening
    # first lets the whole table's solves fan out together.
    sweeps: list[tuple[_Knob, _Sweep, Optional[schemas.KnobBounds]]] = []
    jobs: list[tuple] = []
    for knob in KNOBS:
        override = bounds.get(knob.id)
        try:
            sweep = _knob_sweep(knob, sp, override)
        except (ValueError, KeyError) as exc:
            # A knob we cannot even center is reported empty rather than
            # dropped - the page must show the whole table.
            log.warning("sensitivity: skipping knob %s - %s", knob.id, exc)
            sweep = _Sweep(float("nan"), [], 0.0, 0.0, 0.0, 0.0, None)
        if sweep.note is not None:
            notes.append(sweep.note)
        sweeps.append((knob, sweep, override))
        # _params_for uses model_copy(update=...), which SKIPS validation on
        # purpose - the values were already clamped by _knob_sweep. So the
        # params travel as the model itself, never as JSON: round-tripping
        # through model_validate_json would re-impose the very validation
        # _params_for deliberately bypassed.
        jobs.extend(
            (well, _params_for(knob, sp, value), value, label)
            for value, label in sweep.pairs
        )

    # Pass 2: every swept solve at once. Independent and pure, so they fan
    # out over the shared pool; the request thread waits on futures and
    # releases the GIL. Failures are already per-point (_solve_point returns
    # an "error" entry), so nothing here can lose a knob.
    solved = pool.submit_all(_solve_point, jobs)
    if solved is None:  # no pool, or it broke - identical work, serially
        solved = [_solve_point(*job) for job in jobs]

    # Pass 3: hand each knob its own slice back, in table order.
    knobs: list[dict[str, Any]] = []
    cursor = 0
    for knob, sweep, override in sweeps:
        points = solved[cursor : cursor + len(sweep.pairs)]
        cursor += len(sweep.pairs)
        low, high, inert = _excursions(points, baseline, sweep.base)
        lo_clamp, hi_clamp = _clamp_bounds(knob)
        knobs.append(
            {
                "id": knob.id,
                "label": knob.label,
                "unit": knob.unit,
                "baseline_label": _point_label(knob, sweep.base) if sweep.pairs else "n/a",
                "basis": knob.basis,
                "points": points,
                "low": low,
                "high": high,
                "inert": inert,
                "field": knob.field,
                "kind": knob.kind,
                "default_low": sweep.default_low,
                "default_high": sweep.default_high,
                "swept_low": sweep.swept_low,
                "swept_high": sweep.swept_high,
                "clamp_low": _finite(lo_clamp),
                "clamp_high": _finite(hi_clamp),
                "options": list(_CATALOGS[knob.field]) if knob.kind == "catalog" else None,
                "overridden": override is not None and bool(sweep.pairs),
            }
        )

    if baseline["sonic"]:
        notes.append(SONIC_NOTE)

    return {
        "baseline": baseline,
        "knobs": knobs,
        "target_psu": targets.get("target_psu"),
        "target_qoil": targets.get("target_qoil"),
        "target_qliq": targets.get("target_qliq"),
        "target_qpf": targets.get("target_qpf"),
        "notes": notes,
    }


def _score(got: dict[str, Any], targets: dict[str, Optional[float]]) -> Optional[float]:
    """Root-mean-square fractional error against the supplied targets.

    Fractional so the four quantities are comparable: 50 psi on a 2,000 psi
    BHP is the same miss as 25 BOPD on 1,000 BOPD. Lower is better.

    Args:
        got (dict): One solve's psu (psig), qoil (STBOPD), qliq (BLPD),
            qpf (BWPD).
        targets (dict): target_psu / target_qoil / target_qliq / target_qpf;
            any may be None. A zero target has no fraction and is skipped.

    Returns:
        float | None: RMS fractional error (dimensionless), None when no
        usable target was supplied.
    """
    errs = []
    for metric in METRICS:
        target = targets.get(f"target_{metric}")
        value = got.get(metric)
        if target is None or value is None or float(target) == 0.0:
            continue
        errs.append(((float(value) - float(target)) / float(target)) ** 2)
    if not errs:
        return None
    return sqrt(sum(errs) / len(errs))


def _resolve_combine(knobs: list[schemas.CombineKnob]) -> tuple[list[_Knob], int]:
    """Knob rows plus the requested run count, with the three hard refusals.

    Split out of ``run_combine`` so the router can make these checks
    SYNCHRONOUSLY before starting the job: a bad request is the engineer's
    typo and deserves an immediate 422, not a job that fails a second later.

    Args:
        knobs (list): CombineKnob entries from the request.

    Returns:
        tuple: The resolved knob rows in request order, and the number of
        permutations the requested levels imply.

    Raises:
        ValueError: No knobs selected, an unknown knob id, or more than
            MAX_COMBINE_RUNS permutations requested.
    """
    if not knobs:
        raise ValueError("select at least one knob to combine")

    rows: list[_Knob] = []
    requested = 1
    for entry in knobs:
        knob = _BY_ID.get(entry.id)
        if knob is None:
            raise ValueError(f"unknown knob '{entry.id}'")
        rows.append(knob)
        requested *= int(entry.levels)
    if requested > MAX_COMBINE_RUNS:
        # Hard error, not a sample: the caller asked for a factorial and a
        # subset of one would not be the thing they read the answer off.
        raise ValueError(
            f"{requested} permutations requested, the cap is {MAX_COMBINE_RUNS} - "
            "drop a knob, use fewer levels, or narrow the ranges"
        )
    return rows, requested


def _solve_chunk(
    well: str, sp: schemas.SimParams, updates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """One slice of permutation solves, in order.

    Runs inside a ProcessPool CHILD on multi-core hosts: only picklable
    inputs, and every per-point failure comes back as an ``{"error": ...}``
    row - an exception escaping here would poison the whole pool where the
    serial loop would have kept going.
    """
    out: list[dict[str, Any]] = []
    for update in updates:
        try:
            out.append(
                _metrics(solve.solve_single(well, sp.model_copy(update=update)))
            )
        except solve.SolveFailure as exc:
            out.append({"error": exc.error})
        except ValueError as exc:
            out.append({"error": _short(str(exc)) or "invalid"})
    return out


def _solve_parallel(
    well: str,
    sp: schemas.SimParams,
    updates: list[dict[str, Any]],
    workers: int,
    progress: Optional[Callable[[int, int], None]],
) -> Optional[list[dict[str, Any]]]:
    """ProcessPool fan-out over chunked permutations; None = pool unusable.

    Any pool-level failure (spawn refused, BrokenProcessPool, a child dying)
    returns None and the caller reruns serially - the same fallback contract
    as ``network_optimizer.run_all_batch_simulations``. A deterministic
    solver bug costs one wasted parallel attempt and then raises with a
    clean traceback from the serial rerun.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    pool_cls = _EXECUTOR_CLS or ProcessPoolExecutor
    # ~4 chunks per worker: big enough to amortize per-task IPC, small
    # enough that one straggling chunk cannot idle the rest of the pool.
    chunk = max(1, -(-len(updates) // (workers * 4)))
    slices = [updates[i : i + chunk] for i in range(0, len(updates), chunk)]
    results: list[Optional[list[dict[str, Any]]]] = [None] * len(slices)
    done = 0
    try:
        with pool_cls(max_workers=workers) as pool:
            futures = {
                pool.submit(_solve_chunk, well, sp, part): i
                for i, part in enumerate(slices)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                results[i] = fut.result()
                done += len(slices[i])
                if progress is not None:
                    progress(done, len(updates))
    except Exception as exc:  # noqa: BLE001 - fall back, never fail the study
        log.warning("combine ProcessPool failed (%r); rerunning serially", exc)
        return None
    return [point for part in results for point in part or []]


def _solve_combos(
    well: str,
    sp: schemas.SimParams,
    updates: list[dict[str, Any]],
    progress: Optional[Callable[[int, int], None]],
) -> list[dict[str, Any]]:
    """Every permutation solve, in ``updates`` order.

    Parallel when the host allows it: ``worker_ceiling()`` bounds the pool
    (unset = every core locally; the deployed tier pins WOFFL_MAX_WORKERS).
    Grids under ``_PARALLEL_MIN_RUNS`` stay serial, as does everything when
    the ceiling is 1 - that branch is byte-identical to the old loop.
    """
    from woffl.assembly.parallelism import worker_ceiling

    total = len(updates)
    workers = min(worker_ceiling(), total)
    if workers > 1 and total >= _PARALLEL_MIN_RUNS:
        solved = _solve_parallel(well, sp, updates, workers, progress)
        if solved is not None:
            return solved
    out: list[dict[str, Any]] = []
    for done, update in enumerate(updates, start=1):
        out.extend(_solve_chunk(well, sp, [update]))
        if progress is not None and (done % _PROGRESS_EVERY == 0 or done == total):
            progress(done, total)
    return out


def run_combine(
    well: str,
    sp: schemas.SimParams,
    targets: dict[str, Optional[float]],
    knobs: list[schemas.CombineKnob],
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict[str, Any]:
    """Full factorial over the selected knobs: what can the pair reach?

    Single-knob sensitivity cannot answer the question that matters on a well
    where nothing closes the gap alone - whether any COMBINATION inside the
    engineer's believable ranges reaches the measured test. This solves every
    permutation of the requested levels, reports the envelope the combination
    can reach per match quantity, and says whether each supplied target is
    inside it. Read-only: nothing is persisted and no physics is touched.

    Solves fan out over a ProcessPool bounded by ``worker_ceiling()`` when
    the host allows it (a local workstation gets its full core count; the
    deployed 2-vCPU tier stays effectively serial). The caller still runs it
    off the request thread (see ``start_combine``) and reads ``progress``
    instead of holding a socket open for minutes.

    Args:
        well (str): Selected well name ("Custom" allowed).
        sp (schemas.SimParams): Current sidebar parameters (qwf = TOTAL
            LIQUID BLPD).
        targets (dict): Measured test values - target_psu (psig), target_qoil
            (STBOPD), target_qliq (BLPD), target_qpf (BWPD). Any may be None;
            scoring and reachability use whichever are supplied.
        knobs (list): CombineKnob entries - knob id, range in the knob's own
            units, and how many levels to take across that range.
        progress (callable, optional): Called ``(done, total)`` every 25
            permutations and once at the end, where ``total`` counts the
            grids as actually built (levels can collapse under clamping).

    Returns:
        dict: CombineResponse shape - baseline point, one run per permutation
        (failures retained with ``error``), the reachable envelope per
        metric, per-metric reachability, the best run index, the run and
        failure counts, and any notes.

    Raises:
        ValueError: No knobs selected, an unknown knob id, a knob whose range
            leaves no usable value, or more than MAX_COMBINE_RUNS
            permutations requested (router maps to 422 "invalid").
        solve.SolveFailure: The BASELINE solve failed, so there is nothing to
            compare the permutations against (the usual 422).
    """
    rows, _requested = _resolve_combine(knobs)

    base_res = solve.solve_single(well, sp)
    baseline: dict[str, Any] = {"value": 0.0, "label": "baseline", **_metrics(base_res)}

    notes: list[str] = []
    if baseline["sonic"]:
        notes.append(SONIC_NOTE)

    grids: list[list[tuple[float, str]]] = []
    for knob, entry in zip(rows, knobs):
        sweep = _knob_sweep(
            knob,
            sp,
            schemas.KnobBounds(low=entry.low, high=entry.high, steps=entry.levels),
            with_baseline=False,
        )
        if sweep.note is not None:
            notes.append(sweep.note)
        if not sweep.pairs:
            raise ValueError(f"{knob.label} has no usable value in the range given")
        if len(sweep.pairs) < int(entry.levels):
            # The run count is a promise; say so when clamping or rounding
            # collapsed levels instead of quietly returning fewer runs.
            notes.append(
                f"{knob.label} collapsed {entry.levels} levels to "
                f"{len(sweep.pairs)} distinct values after clamping and rounding."
            )
        grids.append(sweep.pairs)

    total = 1
    for grid in grids:
        total *= len(grid)

    combos: list[tuple[dict[str, float], dict[str, str], dict[str, Any]]] = []
    for combo in itertools.product(*grids):
        values: dict[str, float] = {}
        labels: dict[str, str] = {}
        update: dict[str, Any] = {}
        for knob, (value, label) in zip(rows, combo):
            values[knob.id] = float(value)
            labels[knob.id] = label
            update[knob.field] = _field_value(knob, value)
        combos.append((values, labels, update))

    solved = _solve_combos(well, sp, [c[2] for c in combos], progress)

    runs: list[dict[str, Any]] = []
    n_failed = 0
    for (values, labels, _update), got in zip(combos, solved):
        run: dict[str, Any] = {"values": values, "labels": labels}
        if "error" in got:
            run["error"] = got["error"]
            n_failed += 1
        else:
            for metric in METRICS:
                run[metric] = got[metric]
            run["sonic"] = got["sonic"]
            run["score"] = _score(got, targets)
        runs.append(run)

    envelope: dict[str, list[float]] = {}
    for metric in METRICS:
        solved = [run[metric] for run in runs if run.get(metric) is not None]
        if solved:
            envelope[metric] = [min(solved), max(solved)]

    reachable: dict[str, bool] = {}
    for metric in METRICS:
        target = targets.get(f"target_{metric}")
        if target is None:
            continue
        span = envelope.get(metric)
        reachable[metric] = bool(span is not None and span[0] <= float(target) <= span[1])

    scored = [(run["score"], i) for i, run in enumerate(runs) if run.get("score") is not None]
    best_index = min(scored)[1] if scored else None

    if n_failed < len(runs) and all(
        run["sonic"] for run in runs if run.get("error") is None
    ):
        notes.append(
            "Every permutation that solved is on the choked-flow floor, so "
            "suction is set by the throat across this whole box."
        )

    return {
        "baseline": baseline,
        "runs": runs,
        "envelope": envelope,
        "reachable": reachable,
        "best_index": best_index,
        "n_runs": len(runs),
        "n_failed": n_failed,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Combine as a background job
# ---------------------------------------------------------------------------
#
# A full factorial is minutes of serial solving at the sizes engineers
# actually want (8 inputs at 3 levels is 6,561 solves), which no longer fits
# in one request. Same registry the optimization runs use, so the polling
# contract on the client is the one it already speaks.

_JOB_KIND = "sensitivity"


def get_combine_job(job_id: str) -> Optional[dict[str, Any]]:
    """Poll envelope for one combine study; None when unknown/expired."""
    return jobs.get(job_id, (_JOB_KIND,))


def start_combine(req: schemas.CombineRequest) -> str:
    """Start a combined-permutations study as a background job.

    The three request-level refusals (no inputs selected, an unknown id, past
    MAX_COMBINE_RUNS) are raised HERE, on the caller's thread, so the router
    can answer 422 immediately.

    Args:
        req (schemas.CombineRequest): Well, sidebar params, measured targets
            and the entries to vary together.

    Returns:
        str: The job id to poll.

    Raises:
        ValueError: The request itself is bad (router maps to 422).
    """
    _resolve_combine(req.knobs)

    targets = {
        "target_psu": req.target_psu,
        "target_qoil": req.target_qoil,
        "target_qliq": req.target_qliq,
        "target_qpf": req.target_qpf,
    }

    def runner(job: dict[str, Any]) -> dict[str, Any]:
        def report(done: int, total: int) -> None:
            job["progress"] = f"run {done}/{total}"

        return run_combine(req.well, req.params, targets, req.knobs, progress=report)

    return jobs.start(_JOB_KIND, runner, progress="solving the baseline point...")
