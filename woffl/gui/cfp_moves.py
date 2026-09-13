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
  are enumerated; larger ones receive a bounded neighborhood search. Required
  wells stay online, and reported search scope distinguishes these cases.
"""

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


def settle(choices: dict, surfaces: Surfaces, plant: AnchoredPlant,
           max_iter: int = 8, tol_psi: float = 0.5) -> dict:
    """Fixed point of the pressure/water coupling for one configuration.

    Loop gain ≈ dW/dP * s/1000 ≈ 0.1 here, so plain iteration converges in a
    few passes. Returns pressure, fleet oil, machine water, at_trip, and
    ``feasible`` - False (with ``oil = -inf`` and the offending wells in
    ``infeasible``) when any chosen option has no converged surface at the
    settled pressure. An infeasible state is never a candidate plan.
    """

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

    unknown = set(choices) - set(surfaces.wells)
    omitted = set(surfaces.wells) - set(choices)
    if unknown or omitted:
        raise ValueError("CFP choices must name every modeled well exactly once")
    floor, ceiling = max(plant.p_floor, min(surfaces.p_grid)), min(plant.cap, max(surfaces.p_grid))
    pressure = plant.p0
    at_trip = False
    missing: list = []
    for _ in range(max_iter):
        _oil, water, missing = _totals(pressure)
        if missing:
            break
        new_pressure, at_trip = plant.pressure_at(water)
        if not floor <= new_pressure <= ceiling:
            # Probe the boundary for a signed residual, then let the bracket
            # search find an interior root if the demand curve permits one.
            pressure = min(max(new_pressure, floor), ceiling)
            break
        if abs(new_pressure - pressure) < tol_psi:
            pressure = new_pressure
            break
        pressure = new_pressure
    oil, water, missing = _totals(pressure)
    expected, at_trip = plant.pressure_at(water)
    residual = expected - pressure
    converged = not missing and floor <= expected <= ceiling and abs(residual) < tol_psi
    if not converged:
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
            oil, water, missing = _totals(pressure)
            expected, at_trip = plant.pressure_at(water)
            residual = expected - pressure
            converged = not missing and floor <= expected <= ceiling and abs(residual) < tol_psi
            if converged:
                break
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


def _admissible(state: dict, required: set) -> bool:
    return state["feasible"] and all(state["choices"].get(w) not in (None, SI, OFF) for w in required)


def _best_option(ws: WellSurface, pressure: float, lam: float, required: bool = False):
    """The well's equal-slope pick: argmax oil - λ*water (idle scores 0)."""
    best_lab, best_val = (None, float("-inf")) if required else (ws.idle_label(), 0.0)
    for lab in ws.labels():
        ow = option_at(ws, lab, pressure)
        if ow is None:  # not converged at this pressure - not a choice here
            continue
        oil, water = ow
        val = oil - lam * water
        if val > best_val + 1e-9:
            best_lab, best_val = lab, val
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
    evaluate = _evaluate or (lambda choices: settle(choices, surfaces, plant))
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
    """
    baseline = baseline or surfaces.baseline_choices()
    required = _required(surfaces, required_wells)
    evaluate = _evaluate or (lambda choices: settle(choices, surfaces, plant))
    base = evaluate(baseline)
    moves = []
    for w, ws in surfaces.wells.items():
        for lab in ws.choice_labels():
            if lab == baseline.get(w):
                continue
            state = evaluate({**baseline, w: lab})
            if not _admissible(state, required):
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


def pair_moves(surfaces: Surfaces, plant: AnchoredPlant,
               single_moves: Optional[list] = None, top_n: int = 8, *,
               required_wells=None, _evaluate=None, _diagnostics=None) -> list:
    """Jointly settle raw BOL + water-reducing actions, including halves that
    cannot operate alone. At most MAX_PAIR_EVALUATIONS combinations are tried;
    top_n limits displayed rows, not eligibility to enter the final plan.
    """
    required = _required(surfaces, required_wells)
    baseline = surfaces.baseline_choices()
    evaluate = _evaluate or (lambda choices: settle(choices, surfaces, plant))
    base = evaluate(baseline)
    bols, offsets = [], []
    for w, ws in surfaces.wells.items():
        if baseline[w] in (SI, OFF):
            for lab in ws.labels():
                values = [option_at(ws, lab, p) for p in surfaces.p_grid]
                values = [v for v in values if v is not None]
                if values:
                    bols.append((w, lab, max(v[0] for v in values)))
        else:
            old_water = option_at(ws, baseline[w], surfaces.p0)[1]
            for lab in ws.choice_labels():
                if lab == baseline[w] or (w in required and lab in (SI, OFF)):
                    continue
                reductions = []
                for p in surfaces.p_grid:
                    new, old = option_at(ws, lab, p), option_at(ws, baseline[w], p)
                    if new is not None:
                        reductions.append(old_water - new[1])
                        if old is not None:
                            reductions.append(old[1] - new[1])
                if reductions and max(reductions) > 0:
                    offsets.append((w, lab, max(reductions)))
    bols.sort(key=lambda x: (x[0] not in required, -x[2], x[0], x[1]))
    offsets.sort(key=lambda x: (-x[2], x[0], x[1]))
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

    for b, r in product(bols, offsets):
        if evaluated >= MAX_PAIR_EVALUATIONS:
            break
        evaluated += 1
        state = evaluate({**baseline, b[0]: b[1], r[0]: r[1]})
        if not _admissible(state, required):
            continue
        halves = [evaluate({**baseline, a[0]: a[1]}) for a in (b, r)]
        gains = [s["oil"]-base["oil"] for s in halves if _admissible(s, required)]
        gain = state["oil"]-base["oil"]
        if gains and gain <= max(gains) + 1e-6:
            continue
        bring_on, offset = describe(b, state), describe(r, state)
        pairs.append({"bring_on": bring_on, "offset": offset,
                      "fleet_oil_delta": gain,
                      "own_oil_delta": bring_on["own_oil_delta"]+offset["own_oil_delta"],
                      "own_water_delta": bring_on["own_water_delta"]+offset["own_water_delta"],
                      "pressure_after": state["pressure"], "pressure_delta": state["pressure"]-base["pressure"],
                      "at_trip": state["at_trip"], "pressure_residual_psi": state["pressure_residual_psi"]})
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


def _bounded_neighborhood(surfaces, required, evaluate, candidates, diagnostics):
    """Improve evaluated plans with one-well changes and bounded two-well swaps.

    Feasibility seeds shed optional load and retain every required well. The
    finite budget is reported; this is not a global optimality certificate.
    """
    baseline = surfaces.baseline_choices()

    def best():
        return best_plan(list(candidates.values()), baseline, surfaces, required_wells=required)

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
    incumbent = best()
    passes = 0
    for _ in range(MAX_NEIGHBOR_PASSES):
        if incumbent is None or count >= MAX_NEIGHBOR_EVALUATIONS:
            break
        passes += 1
        before = incumbent["oil"]
        origin = incumbent["choices"]
        for w, ws in surfaces.wells.items():
            for lab in (ws.labels() if w in required else ws.choice_labels()):
                if lab == origin[w]:
                    continue
                if count >= MAX_NEIGHBOR_EVALUATIONS:
                    break
                evaluate({**origin, w: lab})
                count += 1
            if count >= MAX_NEIGHBOR_EVALUATIONS:
                break
        incumbent = best()
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
            picks = [lab for lab, _v in sorted(rows, key=lambda x: (-x[1][0], x[1][1], x[0]))[:2]]
            picks += [lab for lab, _v in sorted(rows, key=lambda x: (x[1][1], -x[1][0], x[0]))[:2]]
            alternatives[w] = list(dict.fromkeys(picks))
        for left, right in combinations(surfaces.wells, 2):
            for a, b in product(alternatives[left], alternatives[right]):
                if count >= MAX_NEIGHBOR_EVALUATIONS:
                    break
                evaluate({**origin, left: a, right: b})
                count += 1
            if count >= MAX_NEIGHBOR_EVALUATIONS:
                break
        incumbent = best()
        if incumbent["oil"] <= before + 1e-6:
            break
    diagnostics.update(neighborhood_evaluated=count, neighborhood_passes=passes,
                       neighborhood_limit=MAX_NEIGHBOR_EVALUATIONS,
                       neighborhood_budget_exhausted=count >= MAX_NEIGHBOR_EVALUATIONS)


def moves_summary(surfaces: Surfaces, plant: AnchoredPlant, *, required_wells=None) -> dict:
    """A measured-baseline comparison with explicit requirements/search scope.

    Every evaluated admissible state may win, including do-nothing, singles
    and pairs. Required wells must be pumping but may change pump size. A
    required future well still starts OFF in the comparison baseline.
    """
    required = _required(surfaces, required_wells)
    baseline = surfaces.baseline_choices()
    candidates = {}

    def evaluate(choices):
        signature = tuple(sorted(choices.items()))
        if signature not in candidates:
            candidates[signature] = settle(choices, surfaces, plant, tol_psi=1e-6)
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
        scope.update(method="exhaustive_on_response_surfaces", neighborhood_evaluated=0)
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
    positive = [m for m in singles if m["fleet_oil_delta"] > 1.0]
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
    """Run WOFFL over (discharge grid) x (all wells) x (all candidate sizes).

    One NetworkOptimizer batch per grid pressure covers every well and every
    nozzle/throat combo (BatchPump sweeps the cross product), so the whole
    surface costs ~len(p_grid) batch runs, process-pooled. Each well's CURRENT
    size is unioned into the candidate lists so the baseline always exists.

    ``pad_configs``: {pad: [WellConfig]} — online wells AND bring-online
    candidates. ``online``/``current`` keyed by well name.
    """
    from woffl.assembly.network_optimizer import (
        NetworkOptimizer,
        PowerFluidConstraint,
    )
    from woffl.gui.cfp_optimize import _assign_well_pressures, delivered_by_pad
    from woffl.assembly.parallelism import worker_ceiling

    pads = sorted(pad_configs)
    wells = [wc for pad in pads for wc in pad_configs[pad]]
    if not wells:
        return Surfaces(p_grid=list(p_grid), p0=p0)

    noz = sorted({str(n) for n in nozzles} | {c[0] for c in current.values() if c})
    thr = sorted({str(t) for t in throats} | {c[1] for c in current.values() if c})
    grid = sorted(float(p) for p in p_grid)

    surfaces = Surfaces(p_grid=grid, p0=float(p0))
    for wc in wells:
        cur = current.get(wc.well_name)
        surfaces.wells[wc.well_name] = WellSurface(
            well=wc.well_name,
            pad=wc.pad,
            online=bool(online.get(wc.well_name, False)),
            current=f"{cur[0]}{cur[1]}" if cur else None,
        )

    for i, pressure in enumerate(grid):
        per_pad, _clamped = delivered_by_pad(
            plant_model, pressure, pads,
            c_pad_pf_psi=c_pad_pf_psi, measured_pad_pf=measured_pad_pf,
            anchor_disch_p=float(p0),  # today's pad readings were taken at p0
        )
        _assign_well_pressures(wells, per_pad, fallback=c_pad_pf_psi)
        constraint_psi = min(max(pressure, 1000.0), 5000.0)
        opt = NetworkOptimizer(
            wells,
            PowerFluidConstraint(
                total_rate=500000.0, pressure=constraint_psi, rho_pf=None
            ),
            noz,
            thr,
            marginal_watercut=1.0,
        )
        opt.run_all_batch_simulations(max_workers=worker_ceiling())
        for wc in wells:
            ws = surfaces.wells[wc.well_name]
            for n in noz:
                for t in thr:
                    scoped = getattr(wc, "pump_calibration_scoped", False)
                    states = ["replacement"] if scoped else [None]
                    if scoped and (n, t) == (wc.installed_nozzle, wc.installed_throat):
                        states.insert(0, "installed")
                    for state in states:
                        perf = opt.get_pump_performance(wc.well_name, n, t,
                            **({"pump_state": state} if state else {}))
                        label = f"{n}{t}" + (" (clean)" if state == "replacement" else "")
                        entry = ws.options.setdefault(label, {
                            "nozzle": n, "throat": t, "pump_state": state,
                            "_grid": grid, "oil": [None] * len(grid), "water": [None] * len(grid),
                        })
                        if perf is not None:
                            entry["oil"][i] = float(perf["oil_rate"])
                            entry["water"][i] = float(perf["total_water"])
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
