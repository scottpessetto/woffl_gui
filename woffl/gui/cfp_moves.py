"""Today's Moves — anchored delta optimization of the CFP-fed jet pump fleet.

Answers Scott's operating question directly (2026-07-30): *"today, at a given
pressure, I have knobs to turn by changing JP size, and shut in or bring online
wells — should I resize JPs to increase PF, SI a well, BOL a well and drop PF,
or BOL wells offset by downsizing a jet pump?"*

THE FORMULATION (docs/cfp_moves_methodology.md has the full treatment + the
literature grounding — Kanu's equal-slope allocation, the Rashid/Bailey/Couët
gas-lift survey, Gunnerud & Foss piecewise-linear sampling):

* **State is measured, everything is a delta.** Anchor at today's measured
  discharge ``P0`` and today's configuration (online set + current JP sizes).
  ``P(x) = min(P0 - s*(W(x) - W0)/1000, trip - margin)``. Injection water,
  disposal position, other pads' carryover — all cancel in the subtraction.
  THERE IS NO EXOGENOUS WATER NUMBER, deliberately.
* **Per-well physics from WOFFL** as response surfaces: oil and total water
  (PF draw + formation — both pass through the machines) per (well, size)
  over a discharge grid. Built once with the existing NetworkOptimizer
  machinery (Stage A, cached); everything below is pure math on the tables
  (Stage B, tested).
* **Optimization starts with an equal-slope / Lagrangian sweep.** Price water
  at λ; each well picks argmax(oil - λ*water), then settle pressure. Retain
  today's baseline and evaluated single/joint moves. Small discrete problems
  are enumerated; larger ones receive a bounded neighborhood search from the
  best plan found and, when required wells are idle today, also from today's
  configuration with them switched on plus a single offset. Required wells
  stay online in the plan (the single-move and pair boards stay relative to
  today and flag ``meets_required``), and reported search scope
  distinguishes these cases.
"""

from bisect import bisect_left
from copy import copy
from dataclasses import dataclass, field
from itertools import combinations, product
import math
from typing import Iterable, Optional

# Option labels that mean "not pumping": SI for an online well shut in, OFF for
# a bring-online candidate left off. Both contribute zero oil and zero water.
SI = "SI"
OFF = "OFF"

MOVE_RESIZE = "resize"
MOVE_SHUT_IN = "shut_in"
MOVE_BRING_ON = "bring_online"

# Machine-curve slope bracket, psi per 1,000 BPD of machine flow. Measured
# -13.69 (r²=0.54) on the real per-machine tags; fitted curve -17.5; operating
# trend -12.2. See cfp_plant.MEASURED_PSI_PER_KBPD.
PSI_PER_KBPD_DEFAULT = 13.69


# ── data model ──────────────────────────────────────────────────────────────


@dataclass
class WellSurface:
    """One well's WOFFL response over the discharge grid.

    ``options`` maps a size label ("12B") to ``{"nozzle", "throat", "oil",
    "water"}`` where oil/water are lists aligned with the grid (None where the
    solver did not converge). Oil is BOPD; water is TOTAL water BPD — power
    fluid draw plus formation water, both of which pass through the machines.
    """

    well: str
    pad: str
    online: bool
    current: Optional[str]  # size label, None when no reviewed pump
    options: dict = field(default_factory=dict)

    def labels(self) -> list:
        """Pumpable options with at least one converged grid point."""
        return [
            lab
            for lab, o in self.options.items()
            if any(v is not None for v in o["oil"])
        ]

    def choice_labels(self) -> list:
        """Everything this well may pick: sizes plus SI (online) / OFF (BOL)."""
        return self.labels() + [SI if self.online else OFF]

    def idle_label(self) -> str:
        return SI if self.online else OFF


@dataclass
class Surfaces:
    p_grid: list
    p0: float
    wells: dict = field(default_factory=dict)  # well -> WellSurface

    def baseline_choices(self) -> dict:
        """Today's configuration: current size when online, idle otherwise.

        Wells that are online but have no usable current-size surface can't be
        anchored and must be excluded before building (see the page).
        """
        out = {}
        for w, ws in self.wells.items():
            # Missing online hardware remains unknown. anchor() must refuse
            # it; treating it as SI would manufacture a bring-online gain.
            out[w] = ws.current if ws.online else ws.idle_label()
        return out


def option_at(ws: WellSurface, label: str, pressure: float) -> Optional[tuple]:
    """(oil, water) for one option at a discharge, linear-interpolated.

    Idle labels are exactly (0, 0). Returns ``None`` where the option is NOT
    available: outside the span of converged grid points, or inside an
    interior gap between two converged points that brackets a non-converged
    one. Non-converged WOFFL points are honest gaps, not values to hold - a
    large pump that only solves at high delivered PF must not be scored at
    a lower pressure with its high-pressure oil (review 2026-09-01, OPT-A1).
    """
    if label in (SI, OFF):
        return 0.0, 0.0
    opt = ws.options.get(label)
    if opt is None:
        return None
    oil = _interp(pressure, opt["_grid"], opt["oil"])
    water = _interp(pressure, opt["_grid"], opt["water"])
    if oil is None or water is None or not all(math.isfinite(v) and v >= 0 for v in (oil, water)):
        return None
    return oil, water


def is_available(ws: WellSurface, label: str, pressure: float) -> bool:
    """Whether ``option_at`` has a converged answer at this pressure."""
    return option_at(ws, label, pressure) is not None


def _interp(x: float, grid: list, vals: list) -> Optional[float]:
    """Linear interpolation across CONVERGED neighbours only.

    ``None`` outside the converged span, and ``None`` inside a gap whose
    bracketing grid points are not both converged (no interpolation across
    a failed solve). Exactly on a converged grid point returns that value.
    """
    n = min(len(grid), len(vals))
    if n == 0:
        return None
    for i in range(n):
        if vals[i] is not None and abs(float(grid[i]) - x) <= 1e-9:
            return float(vals[i])
    for i in range(n - 1):
        g0, g1 = float(grid[i]), float(grid[i + 1])
        if g0 <= x <= g1:
            v0, v1 = vals[i], vals[i + 1]
            if v0 is None or v1 is None:
                return None
            if g1 == g0:
                return float(v0)
            f = (x - g0) / (g1 - g0)
            return float(v0 + f * (float(v1) - float(v0)))
    return None


# ── the anchored plant ──────────────────────────────────────────────────────


@dataclass
class AnchoredPlant:
    """P = min(P0 - s*(W - W0)/1000, trip - margin), anchored at today.

    ``baseline_water`` is the MODEL's water for today's configuration at P0 —
    set by :func:`anchor`, never by summing well tests (Scott: "the summing of
    the tests to the pumps is irrelevant").
    """

    p0: float
    baseline_water: float
    psi_per_kbpd: float = PSI_PER_KBPD_DEFAULT
    trip_psi: float = 2900.0
    trip_margin_psi: float = 20.0
    p_floor: float = 2300.0  # interpolation floor = bottom of the surface grid

    @property
    def cap(self) -> float:
        return self.trip_psi - self.trip_margin_psi

    def pressure_at(self, total_water: float) -> tuple:
        """(pressure, at_trip). Above the cap the disposal re-trim holds the
        plant at the cap — further shedding buys nothing (the kink)."""
        raw = self.raw_pressure_at(total_water)
        if raw >= self.cap:
            return self.cap, True
        # The lower bound is a limit on model support, not a source of extra
        # pressure. settle() reports/rejects an unsupported operating point.
        return raw, False

    def raw_pressure_at(self, total_water: float) -> float:
        """Untrimmed anchored pressure (psi), before the upper disposal cap."""
        return self.p0 + self.psi_per_kbpd * (self.baseline_water - total_water) / 1000.0


def anchor(
    surfaces: Surfaces,
    *,
    psi_per_kbpd: float = PSI_PER_KBPD_DEFAULT,
    trip_psi: float = 2900.0,
    trip_margin_psi: float = 20.0,
) -> AnchoredPlant:
    """Build the anchored plant from the surfaces' own baseline at P0.

    Raises ``ValueError`` naming any ONLINE well whose current size has no
    converged surface at P0: such a well cannot be anchored, and silently
    treating it as idle would make its own "bring online" read as a gain
    (review 2026-09-01, OPT-A9). The caller excludes it with a note.
    """
    if not surfaces.p_grid or not all(math.isfinite(float(p)) for p in surfaces.p_grid):
        raise ValueError("CFP response grid must contain finite pressures")
    cap = float(trip_psi) - float(trip_margin_psi)
    if not math.isfinite(cap) or not math.isfinite(trip_margin_psi) or trip_margin_psi < 0:
        raise ValueError("CFP trip and margin must be finite, with a nonnegative margin")
    if not math.isfinite(surfaces.p0) or not min(surfaces.p_grid) <= surfaces.p0 <= max(surfaces.p_grid):
        raise ValueError("CFP measured anchor P0 must be inside the response grid")
    if surfaces.p0 > cap:
        raise ValueError(f"CFP measured anchor P0 exceeds the trip-minus-margin limit ({cap:g} psi)")
    if not math.isfinite(psi_per_kbpd) or psi_per_kbpd <= 0:
        raise ValueError("CFP pressure slope must be finite and positive")
    choices = surfaces.baseline_choices()
    unanchorable = sorted(
        w
        for w, ws in surfaces.wells.items()
        if ws.online and (not ws.current or ws.current not in ws.options or not is_available(ws, ws.current, surfaces.p0))
    )
    if unanchorable:
        raise ValueError(
            "online wells with no converged current-pump surface at P0: "
            + ", ".join(unanchorable)
        )
    w0 = 0.0
    for w, lab in choices.items():
        ow = option_at(surfaces.wells[w], lab, surfaces.p0)
        w0 += ow[1] if ow is not None else 0.0
    return AnchoredPlant(
        p0=surfaces.p0,
        baseline_water=w0,
        psi_per_kbpd=psi_per_kbpd,
        trip_psi=trip_psi,
        trip_margin_psi=trip_margin_psi,
        p_floor=min(surfaces.p_grid),
    )


# Tolerance of _interp's exact grid-point match, psi. The table replica below
# finds grid points by bisection, so it needs grid spacing above twice this.
_KNOT_TOL_PSI = 1e-9
# A closed-form root this close outside a converged interval sits on its end
# knot (rounding), psi; far below settle()'s 1e-6 psi study tolerance.
_ROOT_EPS_PSI = 1e-8
_UNSET = object()


def _prepare(opt: dict) -> Optional[tuple]:
    """Float tables that reproduce ``option_at`` exactly for one option.

    Returns ``(grid, oil, water, closed_form_ok)``, or None when the option
    is outside what the replica reproduces bit for bit (unsorted or tightly
    spaced grid, ragged lists, oil/water gaps that differ, non-numeric
    values); settle() then uses option_at for that choice set.
    ``closed_form_ok`` needs finite, nonnegative values (so a converged
    interval is available end to end) and nondecreasing water.
    """
    try:
        grid = tuple(float(g) for g in opt["_grid"])
        oil_raw, water_raw = list(opt["oil"]), list(opt["water"])
        oil = [None if v is None else float(v) for v in oil_raw]
        water = [None if v is None else float(v) for v in water_raw]
    except (KeyError, TypeError, ValueError):
        return None
    n = len(grid)
    if n == 0 or len(oil) != n or len(water) != n:
        return None
    if not all(math.isfinite(g) for g in grid) or any(b - a <= 2 * _KNOT_TOL_PSI for a, b in zip(grid, grid[1:])):
        return None
    if any((o is None) != (w is None) for o, w in zip(oil, water)):
        return None
    present = [(o, w) for o, w in zip(oil, water) if o is not None]
    clean = all(math.isfinite(o) and o >= 0 and math.isfinite(w) and w >= 0 for o, w in present)
    monotone = all(b[1] >= a[1] for a, b in zip(present, present[1:]))
    return grid, tuple(oil), tuple(water), clean and monotone


def _usable(o: float, w: float) -> Optional[tuple]:
    """option_at's final check: finite, nonnegative oil and water."""
    if math.isfinite(o) and o >= 0 and math.isfinite(w) and w >= 0:
        return o, w
    return None


def _table_value(table: tuple, x: float) -> Optional[tuple]:
    """``option_at`` on a prepared table: (oil, water) or None, bit-identical."""
    grid, oil, water, _closed_form_ok = table
    n = len(grid)
    j = bisect_left(grid, x)
    # _interp's exact grid-point match: spacing > 2 tol leaves one candidate.
    if j < n and grid[j] - x <= _KNOT_TOL_PSI and oil[j] is not None:
        return _usable(oil[j], water[j])
    if j > 0 and x - grid[j - 1] <= _KNOT_TOL_PSI and oil[j - 1] is not None:
        return _usable(oil[j - 1], water[j - 1])
    # _interp's first bracket holding x; at or below grid[0] that bracket
    # starts on grid[0], which the exact match found missing.
    if j == 0 or j == n:
        return None
    o0, o1, w0, w1 = oil[j - 1], oil[j], water[j - 1], water[j]
    if o0 is None or o1 is None:
        return None
    f = (x - grid[j - 1]) / (grid[j] - grid[j - 1])
    return _usable(o0 + f * (o1 - o0), w0 + f * (w1 - w0))


class _TableCache:
    """Prepared option tables for one study; the surfaces must not change.

    settle() builds a throwaway cache per call when none is passed, so direct
    callers that edit surfaces between calls stay exact.
    """

    def __init__(self, surfaces: Surfaces):
        self.surfaces = surfaces
        self.p_grid = tuple(sorted(float(p) for p in surfaces.p_grid))
        self._tables = {}

    def choice_tables(self, choices: dict) -> Optional[dict]:
        """well -> table (None when idle), or None if any choice has none."""
        out = {}
        for w, lab in choices.items():
            if lab in (SI, OFF):
                out[w] = None
                continue
            key = (w, lab)
            table = self._tables.get(key, _UNSET)
            if table is _UNSET:
                opt = self.surfaces.wells[w].options.get(lab)
                table = _prepare(opt) if opt is not None else None
                self._tables[key] = table
            if table is None:
                return None
            out[w] = table
        return out


def _closed_form_root(tables: dict, p_grid: tuple, plant: AnchoredPlant,
                      floor: float, ceiling: float) -> tuple:
    """The pressure balance root on piecewise-linear tables, in closed form.

    Between grid points each chosen option's water is linear in pressure, so
    on a converged interval ``min(raw(W(p)), cap) = p`` is one linear
    equation (or the cap itself). Eligible only when every pumping option
    sits on the response grid with finite, nonnegative rates and draws
    nondecreasing water: the residual then strictly decreases across every
    converged interval and gap, so the root is unique and is the one any
    bracket search would find.

    Returns:
        tuple: (eligible (bool), root pressure psi (float) or None when no
        converged interval inside [floor, ceiling], nor a converged grid
        point on the trip cap, holds a root).
    """
    live = [t for t in tables.values() if t is not None]
    if not live or not all(t[3] for t in live) or any(t[0] != p_grid for t in live):
        return False, None
    n = len(p_grid)
    water = [0.0] * n
    ok = [True] * n
    for table in live:
        column = table[2]
        for i in range(n):
            value = column[i]
            if value is None:
                ok[i] = False
            else:
                water[i] += value
    kbpd = plant.psi_per_kbpd / 1000.0
    for i in range(n - 1):
        if not (ok[i] and ok[i + 1]):
            continue
        a, b = p_grid[i], p_grid[i + 1]
        lo, hi = max(a, floor), min(b, ceiling)
        if lo > hi:
            continue
        slope = (water[i + 1] - water[i]) / (b - a)  # BPD per psi, >= 0
        raw_a = plant.raw_pressure_at(water[i])
        # raw(W(p)) = p on the line; flat water reuses raw() exactly.
        p_lin = raw_a if slope == 0 else a + (raw_a - a) / (1.0 + kbpd * slope)
        if lo - _ROOT_EPS_PSI <= p_lin <= hi + _ROOT_EPS_PSI:
            return True, min(max(p_lin, lo), hi)
        if p_lin > hi and hi >= plant.cap:
            return True, plant.cap  # the disposal re-trim holds the cap
    # A grid point on the cap where every pump converged, with a failed
    # solve just below it: the re-trim still holds the plant there.
    cap = plant.cap
    if floor <= cap <= ceiling and cap in p_grid:
        k = p_grid.index(cap)
        if ok[k] and plant.raw_pressure_at(water[k]) >= cap:
            return True, cap
    return True, None


def settle(choices: dict, surfaces: Surfaces, plant: AnchoredPlant,
           max_iter: int = 8, tol_psi: float = 0.5, *, _cache: Optional[_TableCache] = None) -> dict:
    """Fixed point of the pressure/water coupling for one configuration.

    Water is piecewise linear in pressure on the response tables. When every
    pumping option shares the response grid and draws nondecreasing water the
    balance has at most one root, solved interval by interval in closed form
    (_closed_form_root). Otherwise, and when that finds no root, the original
    path runs: fixed-point iteration (loop gain ≈ dW/dP * s/1000 ≈ 0.1 here,
    so plain iteration converges in a few passes), then a bracket search per
    continuous interval. Returns pressure, fleet oil, machine water, at_trip,
    and ``feasible`` - False (with ``oil = -inf`` and the offending wells in
    ``infeasible``) when any chosen option has no converged surface at the
    settled pressure. An infeasible state is never a candidate plan.

    Args:
        choices (dict): well -> option label (size, SI or OFF), every well.
        surfaces (Surfaces): the response tables.
        plant (AnchoredPlant): the anchored pressure law.
        max_iter (int): fixed-point iterations before the bracket search.
        tol_psi (float): pressure balance tolerance, psi.
        _cache (_TableCache): prepared tables shared by one study's calls.

    Returns:
        dict: pressure (psig), oil (BOPD), water (BPD), at_trip, choices,
        feasible, converged, pressure_residual_psi, raw_pressure_psi,
        domain_reason and infeasible (well names).
    """
    unknown = set(choices) - set(surfaces.wells)
    omitted = set(surfaces.wells) - set(choices)
    if unknown or omitted:
        raise ValueError("CFP choices must name every modeled well exactly once")
    floor, ceiling = max(plant.p_floor, min(surfaces.p_grid)), min(plant.cap, max(surfaces.p_grid))
    cache = _cache if _cache is not None else _TableCache(surfaces)
    tables = cache.choice_tables(choices)

    if tables is None:
        def _totals(pressure: float):
            oil = water = 0.0
            missing = []
            for w, lab in choices.items():
                ow = option_at(surfaces.wells[w], lab, pressure)
                if ow is None:
                    missing.append(w)
                    continue
                oil += ow[0]
                water += ow[1]
            return oil, water, missing
    else:
        def _totals(pressure: float):
            # Same sums in the same order as option_at would give; an idle
            # well adds exactly zero.
            oil = water = 0.0
            missing = []
            for w, table in tables.items():
                if table is None:
                    continue
                ow = _table_value(table, pressure)
                if ow is None:
                    missing.append(w)
                    continue
                oil += ow[0]
                water += ow[1]
            return oil, water, missing

    def _balance(pressure: float):
        oil, water, missing = _totals(pressure)
        expected, at_trip = plant.pressure_at(water)
        residual = expected - pressure
        converged = not missing and floor <= expected <= ceiling and abs(residual) < tol_psi
        return oil, water, missing, expected, at_trip, residual, converged

    search_brackets = True
    if tables is not None:
        eligible, root = _closed_form_root(tables, cache.p_grid, plant, floor, ceiling)
        if root is not None:
            balance = _balance(root)
            if balance[-1]:
                return _settled(choices, plant, root, balance, floor, ceiling)
        elif eligible:
            # Unique-root balance, no root in any converged interval: the
            # bracket search cannot find one either. The fixed point still
            # runs so rejected states report the same pressure as before.
            search_brackets = False

    pressure = plant.p0
    for _ in range(max_iter):
        _oil, water, missing = _totals(pressure)
        if missing:
            break
        new_pressure, _at_trip = plant.pressure_at(water)
        if not floor <= new_pressure <= ceiling:
            # Probe the boundary for a signed residual, then let the bracket
            # search find an interior root if the demand curve permits one.
            pressure = min(max(new_pressure, floor), ceiling)
            break
        if abs(new_pressure - pressure) < tol_psi:
            pressure = new_pressure
            break
        pressure = new_pressure
    balance = _balance(pressure)
    if not balance[-1] and search_brackets:
        # Fixed-point iteration can oscillate on steep demand. Search each
        # continuous surface interval; never bridge a missing model point.
        from scipy.optimize import brentq

        def residual_at(p):
            _oil, w, absent = _totals(p)
            if absent:
                raise ValueError("missing surface")
            return plant.pressure_at(w)[0] - p

        knots = sorted({floor, ceiling, plant.p0} | {p for p in surfaces.p_grid if floor <= p <= ceiling})
        for lo, hi in zip(knots, knots[1:]):
            try:
                root = brentq(residual_at, lo, hi, xtol=1e-6)
            except ValueError:
                continue
            pressure = root
            balance = _balance(pressure)
            if balance[-1]:
                break
    return _settled(choices, plant, pressure, balance, floor, ceiling)


def _settled(choices: dict, plant: AnchoredPlant, pressure: float, balance: tuple,
             floor: float, ceiling: float) -> dict:
    """settle()'s result for one evaluated pressure balance."""
    oil, water, missing, expected, at_trip, residual, converged = balance
    feasible = not missing and converged
    domain_reason = None
    if not feasible:
        if not missing and expected < floor:
            domain_reason = "required_pressure_below_response_grid"
        elif not missing and expected > ceiling:
            domain_reason = "required_pressure_above_response_grid"
        elif missing:
            domain_reason = "missing_pump_response"
        else:
            domain_reason = "pressure_balance_not_converged"
    return {
        "pressure": pressure,
        "oil": oil if feasible else float("-inf"),
        "water": water if feasible else float("nan"),
        "at_trip": at_trip,
        "choices": dict(choices),
        "feasible": feasible,
        "converged": converged,
        "pressure_residual_psi": residual,
        "raw_pressure_psi": plant.raw_pressure_at(water) if not missing else None,
        "domain_reason": domain_reason,
        "infeasible": sorted(missing),
    }


# ── the equal-slope frontier ────────────────────────────────────────────────

# λ grid in oil-bbl per water-bbl. Marginal oil/water ratios on this fleet run
# ~0.03-0.3, so the grid brackets them with headroom on both sides.
LAMBDA_GRID = (
    0.0, 0.005, 0.01, 0.02, 0.03, 0.045, 0.06, 0.08, 0.10,
    0.13, 0.17, 0.22, 0.30, 0.40, 0.55, 0.75, 1.0,
)


def _required(surfaces: Surfaces, required_wells=None) -> set:
    required = set(required_wells or ())
    unknown = required - set(surfaces.wells)
    if unknown:
        raise ValueError("required CFP wells have no response model: " + ", ".join(sorted(unknown)))
    return required


def _meets_required(state: dict, required: set) -> bool:
    """Whether every required well is pumping in this state."""
    return all(state["choices"].get(w) not in (None, SI, OFF) for w in required)


def _admissible(state: dict, required: set) -> bool:
    return state["feasible"] and _meets_required(state, required)


def _settler(surfaces: Surfaces, plant: AnchoredPlant):
    """settle() at its default tolerance, sharing one table cache."""
    cache = _TableCache(surfaces)
    return lambda choices: settle(choices, surfaces, plant, _cache=cache)


def _nearest_value(ws: WellSurface, label: str, pressure: float) -> Optional[tuple]:
    """(oil, water) at the option's converged grid point nearest ``pressure``.

    Ties go to the higher pressure. None when the option never converged.
    """
    opt = ws.options.get(label)
    if opt is None:
        return None
    for p in sorted((float(g) for g in opt["_grid"]), key=lambda g: (abs(g - pressure), -g)):
        value = option_at(ws, label, p)
        if value is not None:
            return value
    return None


def _best_option(ws: WellSurface, pressure: float, lam: float, required: bool = False):
    """The well's equal-slope pick: argmax oil - λ*water (idle scores 0).

    A required well must pump. When none of its sizes solves at this
    pressure it takes the size scoring best at that size's converged grid
    point nearest the pressure, so settle() can look for a discharge where
    the size runs instead of every λ stalling on a missing option.
    """
    best_lab, best_val = (None, float("-inf")) if required else (ws.idle_label(), 0.0)
    for lab in ws.labels():
        ow = option_at(ws, lab, pressure)
        if ow is None:  # not converged at this pressure - not a choice here
            continue
        oil, water = ow
        val = oil - lam * water
        if val > best_val + 1e-9:
            best_lab, best_val = lab, val
    if best_lab is None and required:
        for lab in ws.labels():
            ow = _nearest_value(ws, lab, pressure)
            if ow is not None and ow[0] - lam * ow[1] > best_val + 1e-9:
                best_lab, best_val = lab, ow[0] - lam * ow[1]
    return best_lab

def sweep_frontier(surfaces: Surfaces, plant: AnchoredPlant,
                   lambdas: Iterable[float] = LAMBDA_GRID, *, required_wells=None,
                   _evaluate=None) -> list:
    """Trace the oil-vs-pressure frontier by sweeping the water price λ.

    At each λ: alternate best-response choices and the pressure fixed point
    until stable (or a cycle — then keep the best-oil state visited). Kanu's
    equal-slope allocation, generalized to discrete options.
    """
    frontier = []
    seen_signatures = set()
    required = _required(surfaces, required_wells)
    evaluate = _evaluate or _settler(surfaces, plant)
    for lam in lambdas:
        pressure = plant.p0
        best_state, visited = None, set()
        for _ in range(10):
            choices = {
                w: _best_option(ws, pressure, lam, w in required) for w, ws in surfaces.wells.items()
            }
            sig = tuple(sorted(choices.items()))
            state = evaluate(choices)
            if _admissible(state, required) and (best_state is None or state["oil"] > best_state["oil"]):
                best_state = state
            if sig in visited:
                break
            visited.add(sig)
            if abs(state["pressure"] - pressure) < 1.0:
                break
            pressure = state["pressure"]
        if best_state is None:  # every visited state was infeasible at its pressure
            continue
        sig = tuple(sorted(best_state["choices"].items()))
        if sig not in seen_signatures:
            seen_signatures.add(sig)
            frontier.append({"lam": lam, **best_state})
    frontier.sort(key=lambda s: s["pressure"])
    return frontier


def best_plan(frontier: list, baseline: dict, surfaces: Surfaces, *, required_wells=None) -> Optional[dict]:
    """Max-oil admissible evaluated state, ties to fewest changes from today; with the
    action diff (what to actually go do)."""
    required = _required(surfaces, required_wells)
    frontier = [s for s in frontier if _admissible(s, required)]
    if not frontier:
        return None

    def _n_changes(state):
        return sum(1 for w, lab in state["choices"].items() if lab != baseline.get(w))

    best = max(frontier, key=lambda s: (round(s["oil"], 3), -_n_changes(s)))
    actions = []
    for w, lab in sorted(best["choices"].items()):
        frm = baseline.get(w)
        if lab == frm:
            continue
        ws = surfaces.wells[w]
        # Compare actual before/after conditions. The old pump need not have
        # a counterfactual solution at the proposed operating pressure.
        oil_a, wat_a = option_at(ws, lab, best["pressure"])
        before = option_at(ws, frm, surfaces.p0)
        oil_b, wat_b = before if before is not None else (float("nan"), float("nan"))
        actions.append(
            {
                "well": w,
                "pad": ws.pad,
                "type": _move_type(ws, frm, lab),
                "from": frm,
                "to": lab,
                "own_oil_delta": oil_a - oil_b,
                "own_water_delta": wat_a - wat_b,
            }
        )
    return {**best, "lam": best.get("lam"), "actions": actions, "n_changes": len(actions)}


# ── single moves and pairs — the knob board ─────────────────────────────────


def _move_type(ws: WellSurface, frm: str, to: str) -> str:
    if to == SI:
        return MOVE_SHUT_IN
    if frm in (SI, OFF):
        return MOVE_BRING_ON
    return MOVE_RESIZE


def rank_single_moves(surfaces: Surfaces, plant: AnchoredPlant,
                      baseline: Optional[dict] = None, *, required_wells=None,
                      _evaluate=None) -> list:
    """Every one-well change from today, exactly settled, best fleet-oil first.

    This is the exhaustive knob board: Resize (either direction), Shut in,
    Bring online — each with the fleet oil delta (the well's own change PLUS
    what the pressure move does to everyone else) and the discharge delta.
    Rows stay relative to today whatever the plan requires; ``meets_required``
    marks the moves that leave every required well pumping. Only moves that
    land where a chosen pump has no solve are left off the board.
    """
    baseline = baseline or surfaces.baseline_choices()
    required = _required(surfaces, required_wells)
    evaluate = _evaluate or _settler(surfaces, plant)
    base = evaluate(baseline)
    moves = []
    for w, ws in surfaces.wells.items():
        for lab in ws.choice_labels():
            if lab == baseline.get(w):
                continue
            state = evaluate({**baseline, w: lab})
            if not state["feasible"]:
                continue  # the move lands at a pressure where a pump has no solve
            own_after, water_after = option_at(ws, lab, state["pressure"])
            own_before, water_before = option_at(ws, baseline[w], base["pressure"])
            moves.append(
                {
                    "well": w,
                    "pad": ws.pad,
                    "type": _move_type(ws, baseline[w], lab),
                    "from": baseline[w],
                    "to": lab,
                    "fleet_oil_delta": state["oil"] - base["oil"],
                    "own_oil_delta": own_after - own_before,
                    "own_water_delta": water_after - water_before,
                    "pressure_delta": state["pressure"] - base["pressure"],
                    "pressure_after": state["pressure"],
                    "at_trip": state["at_trip"],
                    "pressure_residual_psi": state["pressure_residual_psi"],
                    "meets_required": _meets_required(state, required),
                }
            )
    moves.sort(key=lambda m: m["fleet_oil_delta"], reverse=True)
    return moves


# Bounds cover cheap surface evaluations, never additional well simulations.
MAX_PAIR_EVALUATIONS = 2048
MAX_EXACT_COMBINATIONS = 4096
MAX_EXACT_WORK = 250000  # combinations * wells * pressure intervals
MAX_NEIGHBOR_EVALUATIONS = 4096
MAX_NEIGHBOR_PASSES = 3
MAX_SEED_EVALUATIONS = 2048
MAX_SEED_CORES = 64  # required-size combinations enumerated in full
MAX_GREEDY_EVALUATIONS = 512  # multi-offset completion when no seed fits


def _single_changes(surfaces: Surfaces, baseline: dict) -> list:
    """Every one-well change from ``baseline``, scored by the water it sheds.

    The score (BPD) is a same-pressure difference: the baseline option's
    water minus the alternative's, at the pressure nearest P0 where both
    solve. A rate at P0 against a rate at another pressure is no reduction -
    that comparison listed upsizes as offsets. A change with no common
    converged pressure gets no score and is left out.

    Returns:
        list: (well, label, score BPD) tuples, largest reduction first.
    """
    p0 = float(surfaces.p0)
    others = sorted({float(p) for p in surfaces.p_grid} - {p0}, key=lambda p: (abs(p - p0), -p))
    out = []
    for w, ws in surfaces.wells.items():
        frm = baseline[w]
        for lab in ws.choice_labels():
            if lab == frm:
                continue
            for p in [p0, *others]:
                old, new = option_at(ws, frm, p), option_at(ws, lab, p)
                if old is not None and new is not None:
                    out.append((w, lab, old[1] - new[1]))
                    break
    out.sort(key=lambda x: (-x[2], x[0], x[1]))
    return out


def pair_moves(surfaces: Surfaces, plant: AnchoredPlant,
               single_moves: Optional[list] = None, top_n: int = 8, *,
               required_wells=None, _evaluate=None, _diagnostics=None) -> list:
    """Jointly settle raw BOL + water-reducing actions, including halves that
    cannot operate alone. At most MAX_PAIR_EVALUATIONS combinations are tried,
    round-robin: every bring-online (well, size) meets the best offset before
    any meets the second, so the budget reaches every candidate. top_n limits
    displayed rows, not eligibility to enter the final plan. Rows stay
    relative to today and carry ``meets_required``.
    """
    required = _required(surfaces, required_wells)
    baseline = surfaces.baseline_choices()
    evaluate = _evaluate or _settler(surfaces, plant)
    base = evaluate(baseline)
    bols = []
    for w, ws in surfaces.wells.items():
        if baseline[w] in (SI, OFF):
            for lab in ws.labels():
                values = [option_at(ws, lab, p) for p in surfaces.p_grid]
                values = [v for v in values if v is not None]
                if values:
                    bols.append((w, lab, max(v[0] for v in values)))
    offsets = [c for c in _single_changes(surfaces, baseline)
               if baseline[c[0]] not in (SI, OFF) and c[2] > 0]
    bols.sort(key=lambda x: (x[0] not in required, -x[2], x[0], x[1]))
    pairs, evaluated = [], 0

    def describe(action, joint):
        w, lab, _score = action
        ws = surfaces.wells[w]
        own_before, water_before = option_at(ws, baseline[w], base["pressure"])
        own_after, water_after = option_at(ws, lab, joint["pressure"])
        alone = evaluate({**baseline, w: lab})
        return {"well": w, "pad": ws.pad, "type": _move_type(ws, baseline[w], lab),
                "from": baseline[w], "to": lab,
                "own_oil_delta": own_after-own_before, "own_water_delta": water_after-water_before,
                "fleet_oil_delta": alone["oil"]-base["oil"] if alone["feasible"] else None,
                "standalone_feasible": alone["feasible"], "standalone_domain_reason": alone["domain_reason"],
                "pressure_after": joint["pressure"], "pressure_delta": joint["pressure"]-base["pressure"],
                "at_trip": joint["at_trip"], "pressure_residual_psi": joint["pressure_residual_psi"]}

    for r in offsets:
        if evaluated >= MAX_PAIR_EVALUATIONS:
            break
        for b in bols:
            if evaluated >= MAX_PAIR_EVALUATIONS:
                break
            evaluated += 1
            state = evaluate({**baseline, b[0]: b[1], r[0]: r[1]})
            if not state["feasible"]:
                continue
            meets = _meets_required(state, required)
            halves = [evaluate({**baseline, a[0]: a[1]}) for a in (b, r)]
            # A half that breaks a requirement the pair meets cannot outrank it.
            gains = [s["oil"]-base["oil"] for s in halves
                     if s["feasible"] and (_meets_required(s, required) or not meets)]
            gain = state["oil"]-base["oil"]
            if gains and gain <= max(gains) + 1e-6:
                continue
            bring_on, offset = describe(b, state), describe(r, state)
            pairs.append({"bring_on": bring_on, "offset": offset,
                          "fleet_oil_delta": gain,
                          "own_oil_delta": bring_on["own_oil_delta"]+offset["own_oil_delta"],
                          "own_water_delta": bring_on["own_water_delta"]+offset["own_water_delta"],
                          "pressure_after": state["pressure"], "pressure_delta": state["pressure"]-base["pressure"],
                          "at_trip": state["at_trip"], "pressure_residual_psi": state["pressure_residual_psi"],
                          "meets_required": meets})
    if _diagnostics is not None:
        _diagnostics.update(pair_combinations=len(bols)*len(offsets), pair_evaluated=evaluated,
                            pair_search_complete=evaluated == len(bols)*len(offsets), pair_display_limit=top_n)
    pairs.sort(key=lambda p: p["fleet_oil_delta"], reverse=True)
    return pairs[:max(int(top_n), 0)]


def shadow_price_today(surfaces: Surfaces, plant: AnchoredPlant,
                       delta_psi: float = 25.0) -> float:
    """d(fleet oil)/d(discharge) at today's configuration, BOPD per psi.

    The number that prices every knob: a move's pressure delta times this is
    roughly what the rest of the fleet gains or loses.
    """
    choices = surfaces.baseline_choices()

    def fleet_oil(pressure: float) -> Optional[float]:
        total = 0.0
        for w, lab in choices.items():
            ow = option_at(surfaces.wells[w], lab, pressure)
            if ow is None:
                return None
            total += ow[0]
        return total

    hi = min(plant.p0 + delta_psi, plant.cap)
    lo = max(plant.p0 - delta_psi, plant.p_floor)
    if hi <= lo:
        return 0.0
    f_hi, f_lo = fleet_oil(hi), fleet_oil(lo)
    if f_hi is None or f_lo is None:
        # A current pump is not converged on one side: fall back to the
        # one-sided difference through P0 rather than inventing a value.
        f0 = fleet_oil(plant.p0)
        if f0 is None:
            return 0.0
        if f_hi is not None and hi > plant.p0:
            return (f_hi - f0) / (hi - plant.p0)
        if f_lo is not None and plant.p0 > lo:
            return (f0 - f_lo) / (plant.p0 - lo)
        return 0.0
    return (f_hi - f_lo) / (hi - lo)


def _required_cores(surfaces: Surfaces, sizes: dict) -> list:
    """Size assignments for the idle required wells, lightest water first.

    Every combination when there are at most MAX_SEED_CORES; otherwise each
    size of each well with the other wells at their lightest size. Weight is
    the size's water (BPD) at its converged grid point nearest P0.
    """
    def weight(w, lab):
        value = _nearest_value(surfaces.wells[w], lab, surfaces.p0)
        return value[1] if value is not None else math.inf

    ranked = {w: sorted(labs, key=lambda lab: (weight(w, lab), lab)) for w, labs in sizes.items()}
    if math.prod(len(labs) for labs in ranked.values()) <= MAX_SEED_CORES:
        cores = [dict(zip(ranked, combo)) for combo in product(*ranked.values())]
    else:
        light = {w: labs[0] for w, labs in ranked.items()}
        unique = {}
        for w, labs in ranked.items():
            for lab in labs:
                core = {**light, w: lab}
                unique.setdefault(tuple(sorted(core.items())), core)
        cores = list(unique.values())
    return sorted(cores, key=lambda core: sum(weight(w, lab) for w, lab in core.items()))


def _required_seeds(surfaces, required, evaluate, diagnostics):
    """Seed few-change plans that switch the idle required wells on.

    Today's configuration plus each required-size core (_required_cores),
    alone and with each single change on one other well - largest
    same-pressure water reduction first, round-robin over the cores so a
    finite budget reaches all of them. A required well that only solves
    above P0 thereby meets the offset that lifts the discharge to where it
    runs, instead of the search starting from shutting in every optional
    well. When none of those seeds is admissible, each core greedily takes
    further water-reducing changes, one per well, until it is. At most
    MAX_SEED_EVALUATIONS single-offset seeds plus MAX_GREEDY_EVALUATIONS
    completion steps; coverage goes to ``diagnostics``.

    Returns:
        list: the admissible seeded states (the finalists).
    """
    baseline = surfaces.baseline_choices()
    sizes = {w: surfaces.wells[w].labels() for w in sorted(required) if baseline[w] in (SI, OFF)}
    cores = _required_cores(surfaces, sizes) if sizes and all(sizes.values()) else []
    changes = ([c for c in _single_changes(surfaces, baseline)
                if c[0] not in sizes and not (c[0] in required and c[1] in (SI, OFF))] if cores else [])
    count, finalists = 0, []

    def seed(choices):
        nonlocal count
        count += 1
        state = evaluate(choices)
        if _admissible(state, required):
            finalists.append(state)
        return state

    for core in cores:
        if count >= MAX_SEED_EVALUATIONS:
            break
        seed({**baseline, **core})
    for w, lab, _score in changes:
        if count >= MAX_SEED_EVALUATIONS:
            break
        for core in cores:
            if count >= MAX_SEED_EVALUATIONS:
                break
            seed({**baseline, **core, w: lab})
    planned, seeded, greedy = len(cores) * (1 + len(changes)), count, 0
    if cores and not finalists:
        reducers = [c for c in changes if c[2] > 0]
        for core in cores:
            state, changed = {**baseline, **core}, set()
            for w, lab, _score in reducers:
                if greedy >= MAX_GREEDY_EVALUATIONS:
                    break
                if w in changed:
                    continue
                changed.add(w)
                state = {**state, w: lab}
                greedy += 1
                if _admissible(seed(state), required):
                    break
            if greedy >= MAX_GREEDY_EVALUATIONS:
                break
    diagnostics.update(seed_cores=len(cores), seed_combinations=planned, seed_evaluated=seeded,
                       seed_limit=MAX_SEED_EVALUATIONS, seed_search_complete=seeded >= planned,
                       seed_greedy_evaluated=greedy, seed_greedy_limit=MAX_GREEDY_EVALUATIONS,
                       seed_finalists=len(finalists))
    return finalists


def _descend(surfaces, required, evaluate, incumbent, count, limit):
    """Best-improvement walk: one-well changes, then bounded two-well swaps.

    Up to MAX_NEIGHBOR_PASSES passes from ``incumbent`` while ``count`` stays
    below ``limit`` evaluations. The walk keeps its own incumbent, the best
    state it has visited; every evaluated state also stays a plan candidate.

    Returns:
        tuple: (evaluation count (int), passes run (int)).
    """
    baseline = surfaces.baseline_choices()
    passes = 0
    for _ in range(MAX_NEIGHBOR_PASSES):
        if incumbent is None or count >= limit:
            break
        passes += 1
        before = incumbent["oil"]
        origin = incumbent["choices"]
        visited = [incumbent]
        for w, ws in surfaces.wells.items():
            for lab in (ws.labels() if w in required else ws.choice_labels()):
                if lab == origin[w]:
                    continue
                if count >= limit:
                    break
                visited.append(evaluate({**origin, w: lab}))
                count += 1
            if count >= limit:
                break
        incumbent = best_plan(visited, baseline, surfaces, required_wells=required)
        if incumbent["oil"] > before + 1e-6:
            continue

        # A size swap can need a compensating change to work. Keep diverse
        # low-water/high-oil options without constructing an unbounded cross
        # product over every catalog size on every well.
        alternatives = {}
        for w, ws in surfaces.wells.items():
            rows = [(lab, option_at(ws, lab, incumbent["pressure"]))
                    for lab in (ws.labels() if w in required else ws.choice_labels()) if lab != origin[w]]
            rows = [(lab, value) for lab, value in rows if value is not None]
            by_oil = [lab for lab, _v in sorted(rows, key=lambda x: (-x[1][0], x[1][1], x[0]))[:2]]
            by_water = [lab for lab, _v in sorted(rows, key=lambda x: (x[1][1], -x[1][0], x[0]))[:2]]
            # Interleaved (both lists have the same length), so an
            # upsize-plus-offset swap comes early.
            alternatives[w] = list(dict.fromkeys(lab for pick in zip(by_oil, by_water) for lab in pick))
        # Round-robin over well pairs by alternative rank: every pair of wells
        # tries its leading alternatives before any pair tries its later ones,
        # so the budget is not spent on the first wells in name order.
        width = max((len(labs) for labs in alternatives.values()), default=0)
        ranks = sorted(product(range(width), repeat=2), key=lambda ij: (sum(ij), ij))
        for ia, ib in ranks:
            if count >= limit:
                break
            for left, right in combinations(surfaces.wells, 2):
                if ia >= len(alternatives[left]) or ib >= len(alternatives[right]):
                    continue
                if count >= limit:
                    break
                visited.append(evaluate({**origin, left: alternatives[left][ia], right: alternatives[right][ib]}))
                count += 1
        incumbent = best_plan(visited, baseline, surfaces, required_wells=required)
        if incumbent["oil"] <= before + 1e-6:
            break
    return count, passes


def _bounded_neighborhood(surfaces, required, evaluate, candidates, diagnostics):
    """Improve evaluated plans with one-well changes and bounded two-well swaps.

    The main descent starts from the best plan among the baseline, moves,
    pairs, frontier and (last-resort) feasibility seeds that shed all
    optional load. Idle required wells are also seeded into today's
    configuration (_required_seeds), and a second descent on its own budget
    starts from the best admissible seed, a few changes from today: a far-off
    frontier state cannot walk back there in a few passes. Every evaluated
    state stays a plan candidate. The finite budgets are reported; this is
    not a global optimality certificate.
    """
    baseline = surfaces.baseline_choices()
    count = 0
    for pressure in sorted(set(surfaces.p_grid + [surfaces.p0])):
        if count >= MAX_NEIGHBOR_EVALUATIONS:
            break
        choices = {}
        for w, ws in surfaces.wells.items():
            labels = ws.labels() if w in required else [ws.idle_label()]
            available = [(option_at(ws, lab, pressure), lab) for lab in labels]
            available = [(value, lab) for value, lab in available if value is not None]
            choices[w] = min(available, key=lambda x: (x[0][1], -x[0][0], x[1]))[1] if available else None
        evaluate(choices)
        count += 1
    start = best_plan(list(candidates.values()), baseline, surfaces, required_wells=required)
    finalists = _required_seeds(surfaces, required, evaluate, diagnostics)
    if finalists:
        seed_start = best_plan(finalists, baseline, surfaces, required_wells=required)
        seeded, seed_passes = _descend(surfaces, required, evaluate, seed_start, 0, MAX_NEIGHBOR_EVALUATIONS)
        diagnostics.update(seed_descent_evaluated=seeded, seed_descent_passes=seed_passes,
                           seed_descent_limit=MAX_NEIGHBOR_EVALUATIONS)
    count, passes = _descend(surfaces, required, evaluate, start, count, MAX_NEIGHBOR_EVALUATIONS)
    diagnostics.update(neighborhood_evaluated=count, neighborhood_passes=passes,
                       neighborhood_limit=MAX_NEIGHBOR_EVALUATIONS,
                       neighborhood_budget_exhausted=count >= MAX_NEIGHBOR_EVALUATIONS)


def moves_summary(surfaces: Surfaces, plant: AnchoredPlant, *, required_wells=None) -> dict:
    """A measured-baseline comparison with explicit requirements/search scope.

    Every evaluated admissible state may win, including do-nothing, singles
    and pairs. Required wells must be pumping but may change pump size. A
    required future well still starts OFF in the comparison baseline, and
    the single-move and pair boards stay relative to that baseline with a
    ``meets_required`` flag per row.
    """
    required = _required(surfaces, required_wells)
    baseline = surfaces.baseline_choices()
    candidates = {}
    cache = _TableCache(surfaces)

    def evaluate(choices):
        signature = tuple(sorted(choices.items()))
        if signature not in candidates:
            candidates[signature] = settle(choices, surfaces, plant, tol_psi=1e-6, _cache=cache)
        return candidates[signature]

    base = evaluate(baseline)
    if not base["feasible"] or abs(base["pressure"]-surfaces.p0) > 1e-6:
        raise ValueError("CFP baseline must solve at its unchanged measured anchor P0")
    scope = {}
    singles = rank_single_moves(surfaces, plant, baseline, required_wells=required, _evaluate=evaluate)
    pairs = pair_moves(surfaces, plant, singles, required_wells=required, _evaluate=evaluate, _diagnostics=scope)
    frontier = sweep_frontier(surfaces, plant, required_wells=required, _evaluate=evaluate)
    for state in frontier:
        candidates[tuple(sorted(state["choices"].items()))]["lam"] = state["lam"]
    choices_by_well = {w: ws.labels() if w in required else ws.choice_labels() for w, ws in surfaces.wells.items()}
    count = math.prod(len(labels) for labels in choices_by_well.values())
    exact = count <= MAX_EXACT_COMBINATIONS and count * max(len(surfaces.wells), 1) * max(len(surfaces.p_grid)-1, 1) <= MAX_EXACT_WORK
    if exact:
        for labels in product(*choices_by_well.values()):
            evaluate(dict(zip(choices_by_well, labels)))
        scope.update(method="exhaustive_on_response_surfaces", neighborhood_evaluated=0, seed_evaluated=0)
    else:
        _bounded_neighborhood(surfaces, required, evaluate, candidates, scope)
        scope["method"] = "lambda_moves_and_bounded_neighborhood"
    plan = best_plan(list(candidates.values()), baseline, surfaces, required_wells=required)
    # The lambda chart remains that sweep, while plan selection also admits
    # all independently evaluated moves and refinement candidates.
    reasons = {}
    for state in candidates.values():
        if state["domain_reason"]:
            reason = state["domain_reason"]
            reasons[reason] = reasons.get(reason, 0)+1
    # Exhausting discrete choices is only a global statement when demand is
    # nondecreasing with pressure: then each choice has at most one pressure
    # root, including across failed-point gaps. Otherwise branch selection
    # remains part of the unknown search scope.
    monotone = True
    for ws in surfaces.wells.values():
        for option in ws.options.values():
            water = [float(v) for v in option["water"] if v is not None and math.isfinite(float(v))]
            if any(a > b + 1e-9 for a, b in zip(water, water[1:])):
                monotone = False
    scope.update(combinations=count, exact_combination_limit=MAX_EXACT_COMBINATIONS,
                 exact_work_limit=MAX_EXACT_WORK, evaluated_choices=len(candidates),
                 all_choices_evaluated=exact, unique_pressure_response=monotone,
                 global_optimum_on_surfaces=exact and monotone, pressure_tolerance_psi=1e-6,
                 direct_solver_validated=False,
                 rejected_choices_by_reason=reasons)
    # Counts moves that could be taken as they stand: positive and leaving
    # every required well pumping (the board itself lists all of them).
    positive = [m for m in singles if m["fleet_oil_delta"] > 1.0 and m["meets_required"]]
    return {
        "today": {
            "pressure": plant.p0,
            "oil": base["oil"],
            "water": base["water"],
            "n_online": sum(1 for ws in surfaces.wells.values() if ws.online),
            "n_bol_candidates": sum(
                1 for ws in surfaces.wells.values() if not ws.online
            ),
        },
        "lambda_bopd_per_psi": shadow_price_today(surfaces, plant),
        "singles": singles,
        "n_positive_singles": len(positive),
        "pairs": pairs,
        "frontier": [
            {k: s[k] for k in ("lam", "pressure", "oil", "water", "at_trip")}
            for s in frontier
        ],
        "plan": plan,
        "plan_gain": (plan["oil"] - base["oil"]) if plan else None,
        "plan_status": "feasible" if plan else "no_feasible_plan",
        "required_wells": sorted(required),
        "baseline_meets_requirements": _admissible(base, required),
        "search_scope": scope,
        "baseline": baseline,
    }


# ── Stage A: build the surfaces with the real WOFFL machinery ───────────────


def build_response_surfaces(
    pad_configs: dict,
    online: dict,
    current: dict,
    plant_model,
    *,
    p_grid: Iterable[float],
    nozzles: list,
    throats: list,
    p0: float,
    c_pad_pf_psi: float,
    measured_pad_pf: Optional[dict] = None,
    progress=None,
) -> Surfaces:
    """Run WOFFL over (discharge grid) x (all wells) x (candidate sizes).

    Each grid pressure is one pooled ``simulate_jobs`` call covering every
    well, so the whole surface costs ~len(p_grid) batch runs. Every well
    sweeps the requested nozzle x throat catalog; its own CURRENT size joins
    only its own sweep, so the baseline always exists without giving other
    wells sizes nobody requested. A scoped well's installed pump is always
    simulated (``scoped_pumps``) and labelled like ``current``; its clean
    same-size replacement is a candidate only when the catalog requests it.

    The grid changes nothing but each well's delivered PF (``ppf_surf_well``,
    which ``_simulate_single_well`` uses in place of the job pressure), so a
    job is identified by that PF: wells whose pad holds its own PF (C-Pad,
    boosted extra pads) are simulated once, and the host's exact response
    cache sees identical jobs.

    ``pad_configs``: {pad: [WellConfig]} — online wells AND bring-online
    candidates, left unmodified. ``online``/``current`` keyed by well name;
    ``current`` values are (nozzle, throat). Pressures in psig.
    """
    from woffl.assembly.network_optimizer import (
        NetworkOptimizer,
        PowerFluidConstraint,
        simulate_jobs,
    )
    from woffl.gui.cfp_optimize import _assign_well_pressures, delivered_by_pad
    from woffl.assembly.parallelism import worker_ceiling

    pads = sorted(pad_configs)
    wells = [wc for pad in pads for wc in pad_configs[pad]]
    if not wells:
        return Surfaces(p_grid=list(p_grid), p0=p0)

    noz = sorted({str(n) for n in nozzles})
    thr = sorted({str(t) for t in throats})
    catalog = [(n, t) for n in noz for t in thr]
    grid = sorted(float(p) for p in p_grid)

    surfaces = Surfaces(p_grid=grid, p0=float(p0))
    # well -> [(nozzles, throats, [(nozzle, throat, pump_state) read back])]
    sweeps = {}
    for wc in wells:
        cur = current.get(wc.well_name)
        ws = surfaces.wells[wc.well_name] = WellSurface(
            well=wc.well_name,
            pad=wc.pad,
            online=bool(online.get(wc.well_name, False)),
            current=f"{cur[0]}{cur[1]}" if cur else None,
        )
        scoped = getattr(wc, "pump_calibration_scoped", False)
        clean = "replacement" if scoped else None
        installed = (wc.installed_nozzle, wc.installed_throat) if scoped else (None, None)
        installed = installed if all(installed) else None
        jobs = [(noz, thr, [(n, t, clean) for n, t in catalog] + ([(*installed, "installed")] if installed else []))]
        if cur and tuple(cur) not in catalog and tuple(cur) != installed:
            jobs.append(([cur[0]], [cur[1]], [(cur[0], cur[1], clean)]))
        sweeps[wc.well_name] = jobs
        # Label order as before: by size, installed ahead of its clean twin.
        reads = sorted((r for job in jobs for r in job[2]), key=lambda r: (str(r[0]), str(r[1]), r[2] != "installed"))
        for n, t, state in reads:
            ws.options.setdefault(f"{n}{t}" + (" (clean)" if state == "replacement" else ""), {
                "nozzle": n, "throat": t, "pump_state": state,
                "_grid": grid, "oil": [None] * len(grid), "water": [None] * len(grid),
            })

    # Reads batches back through the optimizer's own row lookup; its
    # constraint pressure plays no part in that.
    reader = NetworkOptimizer(
        wells,
        PowerFluidConstraint(total_rate=500000.0, pressure=min(max(float(p0), 1000.0), 5000.0), rho_pf=None),
        noz,
        thr,
        marginal_watercut=1.0,
    )
    # (well, delivered PF psi, job index) -> {(nozzle, throat, state): (oil, water) | None}.
    # Exact within one build: the grid changes no other WellConfig field.
    solved = {}
    for i, pressure in enumerate(grid):
        per_pad, _clamped = delivered_by_pad(
            plant_model, pressure, pads,
            c_pad_pf_psi=c_pad_pf_psi, measured_pad_pf=measured_pad_pf,
            anchor_disch_p=float(p0),  # today's pad readings were taken at p0
        )
        keys, pending, at_point = [], [], []
        for wc in wells:
            clone = copy(wc)
            _assign_well_pressures([clone], per_pad, fallback=c_pad_pf_psi)
            for j, (n_list, t_list, _reads) in enumerate(sweeps[wc.well_name]):
                key = (wc.well_name, clone.ppf_surf_well, j)
                at_point.append(key)
                if key not in solved:
                    keys.append(key)
                    # The job pressure is the PF the well is simulated at, so
                    # the host cache key no longer varies with the grid point.
                    pending.append((clone, clone.ppf_surf_well, n_list, t_list))
        if pending:
            for key, batch in zip(keys, simulate_jobs(pending, max_workers=worker_ceiling())):
                reader.batch_results = {key[0]: batch}
                values = {}
                for n, t, state in sweeps[key[0]][key[2]][2]:
                    perf = reader.get_pump_performance(key[0], n, t, **({"pump_state": state} if state else {}))
                    values[(n, t, state)] = (
                        (float(perf["oil_rate"]), float(perf["total_water"])) if perf is not None else None)
                solved[key] = values
            reader.batch_results = {}
        for key in at_point:
            options = surfaces.wells[key[0]].options
            for (n, t, state), value in solved[key].items():
                if value is not None:
                    entry = options[f"{n}{t}" + (" (clean)" if state == "replacement" else "")]
                    entry["oil"][i], entry["water"][i] = value
        if progress:
            progress(i + 1, len(grid), pressure)

    # Drop options that never converged anywhere — they are not real choices.
    for ws in surfaces.wells.values():
        ws.options = {
            lab: o
            for lab, o in ws.options.items()
            if any(v is not None for v in o["oil"])
        }
    return surfaces
