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
* **Optimization = equal-slope / Lagrangian sweep.** Price machine water at
  λ; each well independently picks argmax(oil - λ*water); settle the pressure
  fixed point (loop gain ≈ 0.1 → converges in a few passes); sweep λ to trace
  the oil-vs-pressure frontier. The best frontier point's diff vs today IS
  the action plan. Single moves and BOL+offset pairs are evaluated
  exhaustively on top, because those are the knobs as the operator sees them.
"""

from dataclasses import dataclass, field
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
            out[w] = (
                ws.current
                if (ws.online and ws.current and ws.current in ws.options)
                else ws.idle_label()
            )
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
    opt = ws.options[label]
    oil = _interp(pressure, opt["_grid"], opt["oil"])
    water = _interp(pressure, opt["_grid"], opt["water"])
    if oil is None or water is None:
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
        raw = self.p0 + self.psi_per_kbpd * (self.baseline_water - total_water) / 1000.0
        if raw >= self.cap:
            return self.cap, True
        return max(raw, self.p_floor), False


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
    choices = surfaces.baseline_choices()
    unanchorable = sorted(
        w
        for w, ws in surfaces.wells.items()
        if ws.online and (not ws.current or not is_available(ws, choices[w], surfaces.p0))
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

    pressure = plant.p0
    at_trip = False
    missing: list = []
    for _ in range(max_iter):
        _oil, water, missing = _totals(pressure)
        if missing:
            break
        new_pressure, at_trip = plant.pressure_at(water)
        if abs(new_pressure - pressure) < tol_psi:
            pressure = new_pressure
            break
        pressure = new_pressure
    oil, water, missing = _totals(pressure)
    feasible = not missing
    return {
        "pressure": pressure,
        "oil": oil if feasible else float("-inf"),
        "water": water if feasible else float("nan"),
        "at_trip": at_trip,
        "choices": dict(choices),
        "feasible": feasible,
        "infeasible": sorted(missing),
    }


# ── the equal-slope frontier ────────────────────────────────────────────────

# λ grid in oil-bbl per water-bbl. Marginal oil/water ratios on this fleet run
# ~0.03-0.3, so the grid brackets them with headroom on both sides.
LAMBDA_GRID = (
    0.0, 0.005, 0.01, 0.02, 0.03, 0.045, 0.06, 0.08, 0.10,
    0.13, 0.17, 0.22, 0.30, 0.40, 0.55, 0.75, 1.0,
)


def _best_option(ws: WellSurface, pressure: float, lam: float) -> str:
    """The well's equal-slope pick: argmax oil - λ*water (idle scores 0)."""
    best_lab, best_val = ws.idle_label(), 0.0
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
                   lambdas: Iterable[float] = LAMBDA_GRID) -> list:
    """Trace the oil-vs-pressure frontier by sweeping the water price λ.

    At each λ: alternate best-response choices and the pressure fixed point
    until stable (or a cycle — then keep the best-oil state visited). Kanu's
    equal-slope allocation, generalized to discrete options.
    """
    frontier = []
    seen_signatures = set()
    for lam in lambdas:
        pressure = plant.p0
        best_state, visited = None, set()
        for _ in range(10):
            choices = {
                w: _best_option(ws, pressure, lam) for w, ws in surfaces.wells.items()
            }
            sig = tuple(sorted(choices.items()))
            state = settle(choices, surfaces, plant)
            if state["feasible"] and (best_state is None or state["oil"] > best_state["oil"]):
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


def best_plan(frontier: list, baseline: dict, surfaces: Surfaces) -> Optional[dict]:
    """Max-oil frontier point, ties to fewest changes from today; with the
    action diff (what to actually go do)."""
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
        # The plan state is feasible, so ``lab`` is available at its pressure;
        # the FROM option may not be (that can be why it was changed).
        oil_a, wat_a = option_at(ws, lab, best["pressure"])
        before = option_at(ws, frm, best["pressure"])
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
    return {**best, "actions": actions, "n_changes": len(actions)}


# ── single moves and pairs — the knob board ─────────────────────────────────


def _move_type(ws: WellSurface, frm: str, to: str) -> str:
    if to == SI:
        return MOVE_SHUT_IN
    if frm in (SI, OFF):
        return MOVE_BRING_ON
    return MOVE_RESIZE


def rank_single_moves(surfaces: Surfaces, plant: AnchoredPlant,
                      baseline: Optional[dict] = None) -> list:
    """Every one-well change from today, exactly settled, best fleet-oil first.

    This is the exhaustive knob board: Resize (either direction), Shut in,
    Bring online — each with the fleet oil delta (the well's own change PLUS
    what the pressure move does to everyone else) and the discharge delta.
    """
    baseline = baseline or surfaces.baseline_choices()
    base = settle(baseline, surfaces, plant)
    moves = []
    for w, ws in surfaces.wells.items():
        for lab in ws.choice_labels():
            if lab == baseline.get(w):
                continue
            state = settle({**baseline, w: lab}, surfaces, plant)
            if not state["feasible"]:
                continue  # the move lands at a pressure where a pump has no solve
            own_after, _ = option_at(ws, lab, state["pressure"])
            own_before, _ = option_at(ws, baseline[w], base["pressure"])
            moves.append(
                {
                    "well": w,
                    "pad": ws.pad,
                    "type": _move_type(ws, baseline[w], lab),
                    "from": baseline[w],
                    "to": lab,
                    "fleet_oil_delta": state["oil"] - base["oil"],
                    "own_oil_delta": own_after - own_before,
                    "pressure_delta": state["pressure"] - base["pressure"],
                    "pressure_after": state["pressure"],
                    "at_trip": state["at_trip"],
                }
            )
    moves.sort(key=lambda m: m["fleet_oil_delta"], reverse=True)
    return moves


def pair_moves(surfaces: Surfaces, plant: AnchoredPlant,
               single_moves: Optional[list] = None, top_n: int = 8) -> list:
    """BOL + offset pairs: bring a well online AND raise pressure back with a
    shut-in or downsize elsewhere — Scott's fourth knob, made explicit.

    Pairs the top bring-on moves with the top pressure-RAISING moves and keeps
    combinations that beat both of their halves alone.
    """
    baseline = surfaces.baseline_choices()
    base = settle(baseline, surfaces, plant)
    moves = single_moves or rank_single_moves(surfaces, plant, baseline)
    bols = [m for m in moves if m["type"] == MOVE_BRING_ON][:top_n]
    raisers = [
        m
        for m in moves
        if m["type"] in (MOVE_SHUT_IN, MOVE_RESIZE) and m["pressure_delta"] > 0
    ][:top_n]
    pairs = []
    for b in bols:
        for r in raisers:
            if r["well"] == b["well"]:
                continue
            state = settle(
                {**baseline, b["well"]: b["to"], r["well"]: r["to"]}, surfaces, plant
            )
            if not state["feasible"]:
                continue
            gain = state["oil"] - base["oil"]
            if gain > max(b["fleet_oil_delta"], r["fleet_oil_delta"]) + 1e-6:
                pairs.append(
                    {
                        "bring_on": b,
                        "offset": r,
                        "fleet_oil_delta": gain,
                        "pressure_after": state["pressure"],
                        "pressure_delta": state["pressure"] - base["pressure"],
                        "at_trip": state["at_trip"],
                    }
                )
    pairs.sort(key=lambda p: p["fleet_oil_delta"], reverse=True)
    return pairs


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


def moves_summary(surfaces: Surfaces, plant: AnchoredPlant) -> dict:
    """The whole decision in one call — what the Results stage renders."""
    baseline = surfaces.baseline_choices()
    base = settle(baseline, surfaces, plant)
    singles = rank_single_moves(surfaces, plant, baseline)
    pairs = pair_moves(surfaces, plant, singles)
    frontier = sweep_frontier(surfaces, plant)
    plan = best_plan(frontier, baseline, surfaces)
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
        "plan_gain": (plan["oil"] - base["oil"]) if plan else 0.0,
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
                total_rate=500000.0, pressure=constraint_psi, rho_pf=62.4
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
                    perf = opt.get_pump_performance(wc.well_name, n, t)
                    label = f"{n}{t}"
                    entry = ws.options.setdefault(
                        label,
                        {
                            "nozzle": n,
                            "throat": t,
                            "_grid": grid,
                            "oil": [None] * len(grid),
                            "water": [None] * len(grid),
                        },
                    )
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
