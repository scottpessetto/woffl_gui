"""Unified pad-optimization compute core (R-1 Phase B).

The S/I/M pad pages carried three near-identical compute cores (~75-80% of
~2,900 lines triplicated — docs/code_review_2026-07-01.md, R-1): a coupled
optimization run, a fixed-pump scenario evaluator, an existing-baseline
scenario evaluator, and a pre-flight match check. This module is the single
copy, driven by a :class:`woffl.gui.pad_plant_base.PadPlant` instance; the
pages keep only their render/UI code (Phase C unifies the pages themselves).

PURE compute: no Streamlit at module level — ``progress`` is a plain callback
the pages adapt to their progress bars. Heavy imports (NetworkOptimizer,
worker_ceiling) stay inside the functions, page-style, so tests can
monkeypatch the source modules.

Coupling dispatch (``plant.coupling``):

* ``fixed_curve`` (S-Pad) — the delivered header is a CURVE of total flow.
  ``run_optimization`` damp-iterates the optimizer against it to a fixed
  point: warm start on the curve at 60% of capacity, relax 0.6, tol 10 psi,
  max 8 iterations, every trial header clamped into ``plant.clamp_window``.
  Progress callback: ``(iter, max_iter, trial_psi, total_pf, curve_psi)``.
* ``free_pressure`` (I/M-Pad) — the header is a DECISION VARIABLE bounded by
  a capability frontier. ``run_optimization`` sweeps ``n_steps`` candidate
  pressures across ``plant.pressure_window``, hands the optimizer
  ``plant.budget_at_pressure`` at each, and keeps the most-oil pressure
  (capturing that step's optimizer for the reconciliation, P0-5). Progress
  callback: ``(step, n_steps, pressure, total_pf, total_oil)``.

The scenario evaluators and the match check couple the SAME way but with the
pumps fixed by the engineer; they always construct their optimizer with
``marginal_watercut=1.0`` — deliberate: the marginal-WC gate belongs to the
main optimization run only (a fixed scenario must show what the chosen pumps
DO, not silently shut wells in).

``run_choke_optimization`` is the SHORT-TERM variant of the same problem:
every well is HELD on its installed pump (no JPCOs) and the levers are
per-well PF throttling (choke) and shut-in — the plan for a PF pump outage
measured in days, where a changeout costs a day per pump. free_pressure
plants only.

Every meta dict carries the uniform contract ``{header_psi, total_pf_bpd,
total_oil_bopd, n_pumps, converged, in_range, recirc, over_capacity,
history, sweep, nozzles, throats, reconciliation}`` (P0-9: ``converged`` is
tracked in every fixed-point path) plus the pad extras each page's render
code reads (per_pump_bpd / station_cap_bpd for fixed_curve;
frontier_cap_bpd / suction_psi / pumps / amp_limited / min_total_flow for
free_pressure).

``run_optimization``'s ``marginal_wc`` also accepts ``None`` (AUTO-DERIVE the
gate from the plant's own PF/water budget at each trial header, via
``woffl.assembly.optimization_algorithms.derive_pad_marginal_wc``) alongside
the legacy manual-float override, plus a ``parsimony_bopd`` knob (default 20,
0 disables) that swaps a well down to a smaller/less-water config when it
gives up no more than that much oil (``apply_parsimony`` — the field case
this exists for: a well upsized 13C->15B for ~2 BOPD at +1,500 BPD PF). Its
meta dict additionally carries ``marginal_wc_used`` (the gate actually
applied at the final/winning header), ``marginal_wc_source`` ("auto
(plant-derived)" | "manual"), ``pf_slack`` (True when the demand walk never
exhausted the budget there), and ``parsimony_swaps`` (list of ``{well,
from_pump, to_pump, oil_given_up, pf_saved}`` dicts).
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

from woffl.gui.pad_plant_base import PadPlant
from woffl.flow.inflow import InFlow

# The scenario evaluators / match check never apply the marginal-WC economics
# gate — see the module docstring.
_SCENARIO_MARGINAL_WC = 1.0

# I-Pad's historical guard: when the frontier inverse can't produce a flow
# ceiling (flow_window hi == 0), the scenario optimizer still needs a
# non-binding PF budget.
_EVAL_CAP_FALLBACK_BPD = 120000.0

# [P1-13] PowerFluidConstraint.rho_pf default (fresh-water density, lbm/ft3).
# Every pad-optimize call site used to spell this literal out independently
# (~5 places) instead of relying on the dataclass default. NetworkOptimizer
# never reads PowerFluidConstraint.rho_pf downstream (grepped: only the
# __post_init__ range check touches it) so this is a display-only value,
# same as the sidebar's "Power Fluid Density" widget (see
# docs/code_review_2026-07-01.md P1-13) — it is NOT the I/M plant's real PF
# SG (~1.03-1.04). Naming it here removes the duplication without changing
# any numeric result; actually wiring plant-specific PF density into the
# physics is a separate, behavior-changing task (see utils.py's
# run_jetpump_solver/run_batch_pump/run_power_fluid_range_batch docstrings).
_RHO_PF_DEFAULT = 62.4


# ---------------------------------------------------------------------------
# Header settling (the one place the flow -> pressure coupling is evaluated)
# ---------------------------------------------------------------------------


def settled_header(
    plant: PadPlant,
    total_pf: float,
    fallback: float,
    n_pumps: int | None = None,
) -> tuple[float, bool]:
    """Header the plant settles at for a total PF draw.

    Returns ``(header_psi, over_capacity)``. ``over_capacity`` is True when
    the plant can't push that flow at any pressure (the header collapses to
    ``plant.suction_psi()``). With no draw (``total_pf <= 0``) the fallback is
    returned, capped at the plant's operational limit — this is each page's
    ``_frontier_header`` / S-Pad ``discharge_pressure`` guard, unified.
    """
    cap = plant.max_header_psi
    if total_pf <= 0:
        return (min(cap, fallback) if cap is not None else fallback), False
    p = plant.header_at_flow(total_pf, n_pumps)
    if p is None:
        return plant.suction_psi(), True
    return (min(cap, p) if cap is not None else p), False


def _next_header(
    plant: PadPlant, total_pf: float, trial_psi: float, n_pumps: int | None
) -> tuple[float, bool]:
    """Fixed-point loop update. A fixed_curve plant evaluates its curve even
    at zero flow (the S-Pad curve tops out near shut-in head); a free_pressure
    plant holds the trial when there's no draw (the I/M pages' behavior)."""
    if plant.coupling == "fixed_curve":
        return plant.header_at_flow(total_pf, n_pumps), False
    return settled_header(plant, total_pf, trial_psi, n_pumps)


def _history_key(plant: PadPlant) -> str:
    # preserved verbatim from the pages: the iteration table's column name
    return "curve_psi" if plant.coupling == "fixed_curve" else "frontier_psi"


# ---------------------------------------------------------------------------
# Optimization run
# ---------------------------------------------------------------------------


def run_optimization(
    well_configs: list,
    plant: PadPlant,
    n_pumps: int | None,
    nozzles: Iterable[str],
    throats: Iterable[str],
    method: str,
    marginal_wc: Optional[float],
    *,
    n_steps: int = 11,
    max_iter: int = 8,
    tol_psi: float = 10.0,
    relax: float = 0.6,
    parsimony_bopd: float = 20.0,
    progress: Optional[Callable] = None,
):
    """Optimize nozzle/throat across the pad, coupled to the booster plant.

    Dispatches on ``plant.coupling`` (see the module docstring). ``n_steps``
    only applies to the free_pressure sweep (I-Pad passes 11, M-Pad 9);
    ``max_iter``/``tol_psi``/``relax`` only to the fixed_curve fixed point.

    ``marginal_wc``: a float is a MANUAL gate override (today's behavior,
    unchanged). ``None`` means AUTO-DERIVE the gate from the plant's own
    physical limits at each trial header —
    ``woffl.assembly.optimization_algorithms.derive_pad_marginal_wc`` pools
    every well's oil-per-water Pareto frontier and reads the gate off the
    ratio that exhausts the plant's PF/water budget, so the economics cutoff
    can never be looser than the pad can actually deliver.

    ``parsimony_bopd``: after the optimizer picks pumps, a well is swapped
    down to a smaller (less-water) config if that config gives up no more
    than this many BOPD — so PF slack isn't spent upsizing a pump for a
    noise-level oil gain. 0 disables the tie-break.

    Returns ``(results, optimizer, meta)`` — ``optimizer`` is the one behind
    ``results`` (the winning sweep step for free_pressure), and
    ``meta["reconciliation"]`` is computed from it (per-well drop reasons at
    the winning point, P0-5). ``meta`` also carries ``marginal_wc_used``
    (the gate actually applied at the final/winning header),
    ``marginal_wc_source`` ("auto (plant-derived)" | "manual"), ``pf_slack``
    (True when the demand walk never exhausted the budget there), and
    ``parsimony_swaps`` (list of ``{well, from_pump, to_pump, oil_given_up,
    pf_saved}`` dicts, empty when none).
    """
    if plant.coupling == "fixed_curve":
        return _run_fixed_point(
            well_configs,
            plant,
            n_pumps,
            nozzles,
            throats,
            method,
            marginal_wc,
            max_iter=max_iter,
            tol_psi=tol_psi,
            relax=relax,
            parsimony_bopd=parsimony_bopd,
            progress=progress,
        )
    return _run_pressure_sweep(
        well_configs,
        plant,
        n_pumps,
        nozzles,
        throats,
        method,
        marginal_wc,
        n_steps=n_steps,
        parsimony_bopd=parsimony_bopd,
        progress=progress,
    )


def _run_fixed_point(
    well_configs,
    plant,
    n_pumps,
    nozzles,
    throats,
    method,
    marginal_wc,
    *,
    max_iter,
    tol_psi,
    relax,
    parsimony_bopd=20.0,
    progress,
):
    """fixed_curve: solve optimizer <-> pump-curve to a damped fixed point."""
    from woffl.assembly.network_optimizer import (
        NetworkOptimizer,
        PowerFluidConstraint,
        reconcile_wells,
    )
    from woffl.assembly.optimization_algorithms import (
        apply_parsimony,
        derive_pad_marginal_wc,
        optimize,
    )
    from woffl.gui.scotts_tools._common import worker_ceiling

    cap = plant.flow_window(n_pumps)[1]  # hydraulic (thrust) ceiling
    ppf = plant.warm_start_psi(n_pumps)  # on the curve at 0.6 x capacity
    lo_p, hi_p = plant.clamp_window(n_pumps)
    history = []
    results, optimizer, converged = [], None, False
    mwc_used = mwc_source = pf_slack = None
    parsimony_swaps: list = []

    for it in range(max_iter):
        ppf_c = max(lo_p, min(hi_p, ppf))
        for wc in well_configs:
            wc.ppf_surf_well = ppf_c  # common header pressure for every well
        pf = PowerFluidConstraint(
            total_rate=cap, pressure=ppf_c, rho_pf=_RHO_PF_DEFAULT
        )
        optimizer = NetworkOptimizer(
            well_configs,
            pf,
            nozzles,
            throats,
            marginal_wc if marginal_wc is not None else 1.0,
        )
        optimizer.run_all_batch_simulations(max_workers=worker_ceiling())

        # Marginal-WC gate: manual value stays manual; None auto-derives it
        # from the plant's OWN budget at this trial header (cheap — pools
        # frontiers already in memory). Both branches report ``pf_slack`` —
        # informative even when the gate is manual.
        gate, slack = derive_pad_marginal_wc(optimizer.batch_results, cap, "lift_wat")
        if marginal_wc is None:
            optimizer.marginal_watercut = gate
            mwc_used, mwc_source = gate, "auto (plant-derived)"
        else:
            mwc_used, mwc_source = marginal_wc, "manual"
        pf_slack = slack

        results = optimize(optimizer, method=method, water_key="lift_wat")
        # Parsimony tie-break BEFORE total_pf is computed, so the header
        # fixed point settles on the parsimonious demand, not the raw pick.
        results, parsimony_swaps = apply_parsimony(
            results, optimizer, "lift_wat", parsimony_bopd
        )

        total_pf = sum(r.predicted_lift_water for r in results)
        new_ppf, _ = _next_header(plant, total_pf, ppf_c, n_pumps)
        history.append(
            {
                "iter": it + 1,
                "trial_psi": round(ppf_c, 1),
                "total_pf_bpd": round(total_pf, 0),
                _history_key(plant): round(new_ppf, 1),
            }
        )
        if progress:
            progress(it + 1, max_iter, ppf_c, total_pf, new_ppf)
        if abs(new_ppf - ppf_c) <= tol_psi:
            ppf = new_ppf
            converged = True
            break
        ppf = relax * new_ppf + (1 - relax) * ppf_c

    total_pf = sum(r.predicted_lift_water for r in results)
    meta = {
        "n_pumps": n_pumps,
        "header_psi": max(lo_p, min(hi_p, ppf)),
        "total_pf_bpd": total_pf,
        "total_oil_bopd": sum(r.predicted_oil_rate for r in results),
        "converged": converged,
        "history": history,
        "sweep": [],
        "nozzles": list(nozzles),
        "throats": list(throats),
        **plant.flags(total_pf, n_pumps),
        "per_pump_bpd": (total_pf / n_pumps) if n_pumps else None,
        "station_cap_bpd": cap,
        "marginal_wc_used": mwc_used,
        "marginal_wc_source": mwc_source,
        "pf_slack": pf_slack,
        "parsimony_swaps": parsimony_swaps,
    }
    # Per-well drop accounting (failed sim vs solver shut-in vs marginal-WC
    # exclusion) — Results renders the real reasons instead of a blanket SI.
    meta["reconciliation"] = reconcile_wells(optimizer, results)
    return results, optimizer, meta


def _run_pressure_sweep(
    well_configs,
    plant,
    n_pumps,
    nozzles,
    throats,
    method,
    marginal_wc,
    *,
    n_steps,
    parsimony_bopd=20.0,
    progress,
):
    """free_pressure: sweep candidate headers, keep the most-oil pressure."""
    from woffl.assembly.network_optimizer import (
        NetworkOptimizer,
        PowerFluidConstraint,
        reconcile_wells,
    )
    from woffl.assembly.optimization_algorithms import (
        apply_parsimony,
        derive_pad_marginal_wc,
        optimize,
    )
    from woffl.gui.scotts_tools._common import worker_ceiling

    p_floor, p_ceiling = plant.pressure_window(n_pumps)
    pressures = [
        p_floor + (p_ceiling - p_floor) * i / (n_steps - 1) for i in range(n_steps)
    ]

    sweep, best = [], None
    for idx, P in enumerate(pressures):
        cap = plant.budget_at_pressure(P, n_pumps)  # PF budget at this pressure
        if not cap or cap <= 0:
            if progress:
                progress(idx + 1, n_steps, P, 0.0, 0.0)
            continue
        for wc in well_configs:
            wc.ppf_surf_well = P
        pf = PowerFluidConstraint(total_rate=cap, pressure=P, rho_pf=_RHO_PF_DEFAULT)
        opt = NetworkOptimizer(
            well_configs,
            pf,
            nozzles,
            throats,
            marginal_wc if marginal_wc is not None else 1.0,
        )
        opt.run_all_batch_simulations(max_workers=worker_ceiling())

        # Marginal-WC gate at THIS trial header — manual stays manual; None
        # auto-derives from the plant's own budget at this pressure.
        gate, slack = derive_pad_marginal_wc(opt.batch_results, cap, "lift_wat")
        if marginal_wc is None:
            opt.marginal_watercut = gate
            trial_mwc_used, trial_mwc_source = gate, "auto (plant-derived)"
        else:
            trial_mwc_used, trial_mwc_source = marginal_wc, "manual"

        results = optimize(opt, method=method, water_key="lift_wat")
        results, trial_swaps = apply_parsimony(results, opt, "lift_wat", parsimony_bopd)
        total_pf = sum(r.predicted_lift_water for r in results)
        total_oil = sum(r.predicted_oil_rate for r in results)
        rec = {
            "P": P,
            "cap": cap,
            "total_pf": total_pf,
            "total_oil": total_oil,
            "results": results,
            "opt": opt,
            "mwc_used": trial_mwc_used,
            "mwc_source": trial_mwc_source,
            "pf_slack": slack,
            "parsimony_swaps": trial_swaps,
        }
        sweep.append(rec)
        if best is None or total_oil > best["total_oil"]:
            best = rec
        if progress:
            progress(idx + 1, n_steps, P, total_pf, total_oil)

    if best is None:
        raise RuntimeError(plant.infeasible_sweep_msg)

    env = plant.envelope([best["total_pf"]], n_pumps)[0]
    meta = {
        "n_pumps": n_pumps,
        "header_psi": best["P"],
        "total_pf_bpd": best["total_pf"],
        "total_oil_bopd": best["total_oil"],
        "frontier_cap_bpd": best["cap"],
        "suction_psi": plant.suction_psi(),
        "min_total_flow": plant.flow_window(n_pumps)[0],
        "pumps": env.get("pumps", []),
        "converged": True,  # a sweep has no fixed point to miss
        "history": [],
        "sweep": [
            {
                "header_psi": s["P"],
                "total_pf_bpd": s["total_pf"],
                "total_oil_bopd": s["total_oil"],
            }
            for s in sweep
        ],
        "nozzles": list(nozzles),
        "throats": list(throats),
        **plant.flags(best["total_pf"], n_pumps),
        "marginal_wc_used": best["mwc_used"],
        "marginal_wc_source": best["mwc_source"],
        "pf_slack": best["pf_slack"],
        "parsimony_swaps": best["parsimony_swaps"],
    }
    if "amp_limited" in env:
        meta["amp_limited"] = env["amp_limited"]
    if "feasible" in env:
        meta["feasible"] = env["feasible"]
    # Per-well drop accounting at the WINNING pressure (P0-5).
    meta["reconciliation"] = reconcile_wells(best["opt"], best["results"])
    return best["results"], best["opt"], meta


# ---------------------------------------------------------------------------
# Scenario evaluators (fixed per-well pumps, coupled the same way)
# ---------------------------------------------------------------------------


def _best_feasible_pump(opt, well: str) -> Optional[tuple[str, str]]:
    """(nozzle, throat) of the highest-oil feasible row in the well's batch
    sweep, or None when nothing solved at this header."""
    bp = opt.batch_results.get(well)
    df = getattr(bp, "df", None) if bp is not None else None
    if df is None:
        return None
    feas = df[~df["qoil_std"].isna()]
    if feas.empty:
        return None
    r = feas.loc[feas["qoil_std"].idxmax()]
    return str(r["nozzle"]), str(r["throat"])


def _score_fixed_choices(opt, well_configs, choices, fallback_choices, test_rates):
    """One iteration's per-well scoring for the fixed-pump scenario.

    Fallback chain for a chosen pump with no solution at this header (kept
    verbatim from the pages): measured test rate ★ (Existing baseline only) →
    the optimized pick → the best feasible pump in the batch (both flagged as
    substitutions) → "✗ no feasible pump" counted as zero.
    """
    per_well, total_pf, total_oil = [], 0.0, 0.0
    for wc in well_configs:
        ch = choices.get(wc.well_name)
        if not ch:
            per_well.append(
                {
                    "well": wc.well_name,
                    "pump": "SHUT IN",
                    "oil": 0.0,
                    "pf": 0.0,
                    "note": "",
                }
            )
            continue

        perf = opt.get_pump_performance(wc.well_name, ch[0], ch[1])
        note = ""
        if perf is None and test_rates and wc.well_name in test_rates:
            # "Existing" comparison: a well the model can't solve falls back to
            # its measured latest-test rate (not a substituted pump), starred.
            to, tp = test_rates[wc.well_name]
            per_well.append(
                {
                    "well": wc.well_name,
                    "pump": f"{ch[0]}{ch[1]} ★",
                    "oil": float(to or 0.0),
                    "pf": float(tp or 0.0),
                    "note": "star",
                }
            )
            total_oil += float(to or 0.0)
            total_pf += float(tp or 0.0)
            continue
        if perf is None:
            # Chosen pump can't operate at this header pressure. Fall back so
            # the well isn't a misleading zero: prefer the optimized pick,
            # else the best feasible pump in the batch. Flag the swap.
            fb = (fallback_choices or {}).get(wc.well_name)
            if fb:
                perf = opt.get_pump_performance(wc.well_name, fb[0], fb[1])
                if perf is not None:
                    note, ch = f"{ch[0]}{ch[1]}✗→{fb[0]}{fb[1]}", fb
            if perf is None:
                best = _best_feasible_pump(opt, wc.well_name)
                if best is not None:
                    fbn, fbt = best
                    perf = opt.get_pump_performance(wc.well_name, fbn, fbt)
                    if perf is not None:
                        note, ch = (
                            f"{choices[wc.well_name][0]}{choices[wc.well_name][1]}✗→{fbn}{fbt}",
                            (fbn, fbt),
                        )

        if perf is None:
            orig = choices[wc.well_name]
            per_well.append(
                {
                    "well": wc.well_name,
                    "pump": f"{orig[0]}{orig[1]} ✗ no feasible pump",
                    "oil": 0.0,
                    "pf": 0.0,
                    "note": "infeasible",
                }
            )
            continue

        per_well.append(
            {
                "well": wc.well_name,
                "pump": note or f"{ch[0]}{ch[1]}",
                "oil": perf["oil_rate"],
                "pf": perf["lift_water"],
                "note": note,
            }
        )
        total_pf += perf["lift_water"]
        total_oil += perf["oil_rate"]
    return per_well, total_pf, total_oil


def _ripple_rescale_stars(per_well, choices, current_choices, test_rates) -> bool:
    """★ rows: measured test rate × the average ripple of the UNCHANGED wells
    that DID solve (these moved only because the header pressure moved), so
    the non-solvers reflect the same ripple instead of staying flat. Ratio
    clamped to [0.3, 1.2]. Returns True when any ★ row was rescaled."""
    oil_ratios, pf_ratios = [], []
    for r in per_well:
        if r.get("note") == "star":
            continue
        ch, cur = choices.get(r["well"]), (current_choices or {}).get(r["well"])
        unchanged = ch is not None and cur is not None and tuple(ch) == tuple(cur)
        tr = test_rates.get(r["well"])
        if unchanged and tr and tr[0] and r["oil"] > 0:
            oil_ratios.append(r["oil"] / tr[0])
            if tr[1]:
                pf_ratios.append(r["pf"] / tr[1])
    avg_oil = (
        min(1.2, max(0.3, sum(oil_ratios) / len(oil_ratios))) if oil_ratios else 1.0
    )
    avg_pf = min(1.2, max(0.3, sum(pf_ratios) / len(pf_ratios))) if pf_ratios else 1.0
    rescaled = False
    for r in per_well:
        if r.get("note") != "star":
            continue
        tr = test_rates.get(r["well"])
        if tr and tr[0] is not None:
            r["oil"] = float(tr[0]) * avg_oil
        if tr and tr[1] is not None:
            r["pf"] = float(tr[1]) * avg_pf
        rescaled = True
    return rescaled


def _scenario_meta(plant, n_pumps, ppf, total_pf, total_oil, converged, history):
    """Uniform scenario meta: header clamped into the plant's band, the
    coupling flags at the final totals, and the fixed_curve station extras."""
    lo_p, hi_p = plant.clamp_window(n_pumps)
    meta = {
        "n_pumps": n_pumps,
        "header_psi": max(lo_p, min(hi_p, ppf)),
        "total_pf_bpd": total_pf,
        "total_oil_bopd": total_oil,
        # After max_iter oscillating iterations the rows come from the last
        # trial header while header_psi is the damped extrapolation — flag it.
        "converged": converged,
        "history": history,
        **plant.flags(total_pf, n_pumps),
    }
    if plant.coupling == "fixed_curve":
        meta["per_pump_bpd"] = (total_pf / n_pumps) if n_pumps else None
        meta["station_cap_bpd"] = plant.flow_window(n_pumps)[1]
    return meta


def evaluate_fixed_scenario(
    well_configs,
    plant: PadPlant,
    n_pumps: int | None,
    choices: dict,
    *,
    fallback_choices: dict | None = None,
    test_rates: dict | None = None,
    current_choices: dict | None = None,
    max_iter: int = 8,
    tol_psi: float = 10.0,
    relax: float = 0.6,
    progress: Optional[Callable] = None,
):
    """Evaluate a FIXED per-well pump scenario against the booster coupling.

    Like ``run_optimization`` but instead of letting the optimizer pick pumps,
    each well's pump is fixed by ``choices`` (well_name -> (nozzle, throat), or
    None to shut the well in). Still couples the delivered header to the total
    PF (fixed point for every coupling — a fixed pump set has no pressure to
    sweep), so the engineer sees the real oil + header for THEIR selection.
    See ``_score_fixed_choices`` for the infeasible-pump fallback chain.
    Returns ``(per_well rows, meta)``.
    """
    from woffl.assembly.network_optimizer import NetworkOptimizer, PowerFluidConstraint
    from woffl.gui.scotts_tools._common import worker_ceiling

    # Batch must compute every chosen pump AND every fallback pump (used when
    # a chosen pump is infeasible), so union both.
    all_ch = list(choices.values()) + list((fallback_choices or {}).values())
    nozzles = sorted({c[0] for c in all_ch if c}) or ["12"]
    throats = sorted({c[1] for c in all_ch if c}) or ["B"]
    cap = plant.flow_window(n_pumps)[1] or _EVAL_CAP_FALLBACK_BPD
    lo_p, hi_p = plant.clamp_window(n_pumps)
    ppf = plant.warm_start_psi(n_pumps)
    history, per_well, total_pf, total_oil = [], [], 0.0, 0.0
    converged = False

    for it in range(max_iter):
        ppf_c = max(lo_p, min(hi_p, ppf))
        for wc in well_configs:
            wc.ppf_surf_well = ppf_c
        pf = PowerFluidConstraint(
            total_rate=cap, pressure=ppf_c, rho_pf=_RHO_PF_DEFAULT
        )
        opt = NetworkOptimizer(
            well_configs, pf, nozzles, throats, marginal_watercut=_SCENARIO_MARGINAL_WC
        )
        opt.run_all_batch_simulations(max_workers=worker_ceiling())

        per_well, total_pf, total_oil = _score_fixed_choices(
            opt, well_configs, choices, fallback_choices, test_rates
        )
        new_ppf, _ = _next_header(plant, total_pf, ppf_c, n_pumps)
        history.append(
            {
                "iter": it + 1,
                "trial_psi": round(ppf_c, 1),
                "total_pf_bpd": round(total_pf, 0),
                _history_key(plant): round(new_ppf, 1),
            }
        )
        if progress:
            progress(it + 1, max_iter, ppf_c, total_pf, new_ppf)
        if abs(new_ppf - ppf_c) <= tol_psi:
            ppf = new_ppf
            converged = True
            break
        ppf = relax * new_ppf + (1 - relax) * ppf_c

    # ★ wells (Existing baseline): ripple their measured rate by the average
    # change of the UNCHANGED solving wells, then recompute totals + header.
    if test_rates:
        if _ripple_rescale_stars(per_well, choices, current_choices, test_rates):
            total_oil = sum(r["oil"] for r in per_well)
            total_pf = sum(r["pf"] for r in per_well)
            ppf, _ = settled_header(plant, total_pf, ppf, n_pumps)

    meta = _scenario_meta(plant, n_pumps, ppf, total_pf, total_oil, converged, history)
    return per_well, meta


def _score_existing_choices(opt, names, scenario_choices, mc, cur_oil, cur_pf):
    """One iteration's per-well scoring for the existing-baseline scenario:
    measured current rate × the model's RELATIVE change (bias cancels in the
    ratio); model absolute when there's no current-pump reference; measured
    rate ★ when the scenario pump doesn't solve."""
    per_well, total_pf = [], 0.0
    for w in names:
        ch = scenario_choices.get(w)
        if not ch:
            per_well.append(
                {"well": w, "pump": "SHUT IN", "oil": 0.0, "pf": 0.0, "note": ""}
            )
            continue
        ms = opt.get_pump_performance(w, ch[0], ch[1])
        mcw = mc.get(w)
        if ms and mcw and mcw[0] > 0 and mcw[1] > 0:
            so = cur_oil[w] * (ms["oil_rate"] / mcw[0])
            sp = cur_pf[w] * (ms["lift_water"] / mcw[1])
            per_well.append(
                {"well": w, "pump": f"{ch[0]}{ch[1]}", "oil": so, "pf": sp, "note": ""}
            )
            total_pf += sp
        elif ms:
            # Scenario pump solves but there's no current-pump reference to
            # bias-correct against — use the model absolute.
            per_well.append(
                {
                    "well": w,
                    "pump": f"{ch[0]}{ch[1]}",
                    "oil": float(ms["oil_rate"]),
                    "pf": float(ms["lift_water"]),
                    "note": "",
                }
            )
            total_pf += float(ms["lift_water"])
        else:
            per_well.append(
                {
                    "well": w,
                    "pump": f"{ch[0]}{ch[1]} ★",
                    "oil": cur_oil[w],
                    "pf": cur_pf[w],
                    "note": "star",
                }
            )
            total_pf += cur_pf[w]
    return per_well, total_pf


def _ripple_rescale_existing(
    per_well, opt, scenario_choices, current_choices, cur_oil, cur_pf
) -> None:
    """Existing-baseline ★ rows: measured rate × the unchanged solving wells'
    average ripple (clamped [0.3, 1.2]); a ★ well with NO measured rate is
    estimated from the best feasible pump the model CAN solve (labeled
    "(est NT)") so it isn't a misleading zero."""
    oil_ratios, pf_ratios = [], []
    for r in per_well:
        if r["note"] == "star":
            continue
        ch, cc = scenario_choices.get(r["well"]), current_choices.get(r["well"])
        if ch and cc and tuple(ch) == tuple(cc) and cur_oil[r["well"]] > 0:
            oil_ratios.append(r["oil"] / cur_oil[r["well"]])
            if cur_pf[r["well"]] > 0:
                pf_ratios.append(r["pf"] / cur_pf[r["well"]])
    avg_oil = (
        min(1.2, max(0.3, sum(oil_ratios) / len(oil_ratios))) if oil_ratios else 1.0
    )
    avg_pf = min(1.2, max(0.3, sum(pf_ratios) / len(pf_ratios))) if pf_ratios else 1.0
    for r in per_well:
        if r["note"] != "star":
            continue
        w = r["well"]
        if cur_oil[w] > 0 or cur_pf[w] > 0:
            r["oil"] = cur_oil[w] * avg_oil
            r["pf"] = cur_pf[w] * avg_pf
        else:
            best = _best_feasible_pump(opt, w)
            if best is not None:
                perf = opt.get_pump_performance(w, best[0], best[1])
                if perf:
                    r["oil"], r["pf"] = float(perf["oil_rate"]), float(
                        perf["lift_water"]
                    )
                    r["pump"] = f"{r['pump']} (est {best[0]}{best[1]})"


def evaluate_existing_scenario(
    well_configs,
    plant: PadPlant,
    n_pumps: int | None,
    scenario_choices: dict,
    current_choices: dict,
    *,
    test_rates: dict,
    max_iter: int = 8,
    tol_psi: float = 10.0,
    relax: float = 0.6,
    progress: Optional[Callable] = None,
):
    """Existing-baseline scenario, anchored to MEASURED latest-test rates.

    Each well's displayed scenario value = its measured current oil/PF × the
    MODEL's RELATIVE change (scenario pump @ scenario header ÷ current pump @
    current header). This keeps every well on the same footing as the measured
    'Current' column — the model bias cancels in the ratio — and each row also
    carries that bias (model-at-current ÷ measured, ``bias``) so the engineer
    can target the loose IPR matches. Non-solving wells use the unchanged
    solving wells' average ripple. Returns ``(per_well, scn_meta)``.
    """
    from woffl.assembly.network_optimizer import NetworkOptimizer, PowerFluidConstraint
    from woffl.gui.scotts_tools._common import worker_ceiling

    names = [wc.well_name for wc in well_configs]
    cur_oil = {w: float((test_rates.get(w) or (0, 0))[0] or 0.0) for w in names}
    cur_pf = {w: float((test_rates.get(w) or (0, 0))[1] or 0.0) for w in names}

    all_ch = list(scenario_choices.values()) + list(current_choices.values())
    nozzles = sorted({c[0] for c in all_ch if c}) or ["12"]
    throats = sorted({c[1] for c in all_ch if c}) or ["B"]
    cap = plant.flow_window(n_pumps)[1] or _EVAL_CAP_FALLBACK_BPD
    lo_p, hi_p = plant.clamp_window(n_pumps)
    total_pf_base = sum(cur_pf.values())
    header_base, _ = settled_header(
        plant, total_pf_base, plant.warm_start_psi(n_pumps), n_pumps
    )

    def _run(ppf):
        ppf_c = max(lo_p, min(hi_p, ppf))
        for wc in well_configs:
            wc.ppf_surf_well = ppf_c
        pf = PowerFluidConstraint(
            total_rate=cap, pressure=ppf_c, rho_pf=_RHO_PF_DEFAULT
        )
        opt = NetworkOptimizer(
            well_configs, pf, nozzles, throats, marginal_watercut=_SCENARIO_MARGINAL_WC
        )
        opt.run_all_batch_simulations(max_workers=worker_ceiling())
        return opt

    # Reference: model at the CURRENT pump @ the current (baseline) header.
    opt_base = _run(header_base)
    mc = {}
    for w in names:
        cc = current_choices.get(w)
        perf = opt_base.get_pump_performance(w, cc[0], cc[1]) if cc else None
        mc[w] = (perf["oil_rate"], perf["lift_water"]) if perf else None

    ppf, per_well = header_base, []
    opt = opt_base
    converged = False
    for it in range(max_iter):
        ppf_c = max(lo_p, min(hi_p, ppf))
        opt = _run(ppf_c)
        per_well, total_pf = _score_existing_choices(
            opt, names, scenario_choices, mc, cur_oil, cur_pf
        )
        new_ppf, _ = _next_header(plant, total_pf, ppf_c, n_pumps)
        if progress:
            progress(it + 1, max_iter, ppf_c, total_pf, new_ppf)
        if abs(new_ppf - ppf_c) <= tol_psi:
            ppf = new_ppf
            converged = True
            break
        ppf = relax * new_ppf + (1 - relax) * ppf_c

    # Non-solvers: average ripple of the UNCHANGED solving wells (scenario ÷
    # current), or a best-feasible estimate when there's no measured anchor.
    _ripple_rescale_existing(
        per_well, opt, scenario_choices, current_choices, cur_oil, cur_pf
    )

    # Bias factor per well: model-at-current ÷ measured. >1 = model
    # over-predicts this well's current rate (its IPR/calibration match is
    # loose) — surfaced in the table so the engineer can target those matches.
    for r in per_well:
        mcw, co = mc.get(r["well"]), cur_oil.get(r["well"], 0.0)
        r["bias"] = (mcw[0] / co) if (mcw and mcw[0] and co > 0) else None

    total_oil = sum(r["oil"] for r in per_well)
    total_pf = sum(r["pf"] for r in per_well)
    ppf, _ = settled_header(plant, total_pf, ppf, n_pumps)

    meta = _scenario_meta(plant, n_pumps, ppf, total_pf, total_oil, converged, [])
    return per_well, meta


# ---------------------------------------------------------------------------
# Pre-flight match check
# ---------------------------------------------------------------------------


def match_flag(ratio) -> str:
    """✓/⚠/✗ verdict bands for a model ÷ test ratio."""
    if ratio is None:
        return "— no data"
    if 0.80 <= ratio <= 1.25:
        return "✓ match"
    if 0.50 <= ratio <= 2.0:
        return "⚠ off"
    return "✗ BUST"


def match_check(
    well_configs,
    plant: PadPlant,
    n_pumps: int | None,
    current_choices: dict,
    test_rates: dict,
):
    """Pre-flight diagnostic: model each well at its CURRENT pump + chosen IPR
    and compare to its measured recent tests (median). Flags wells where the
    model is a total mismatch on oil (loose IPR) or PF (a PF bust) — the wells
    to fix before trusting the optimizer. The header comes from
    ``plant.match_check_header`` (each pad's historical derivation + fallback,
    including I-Pad's operational cap, P0-7). Returns ``(rows, header_psi)``.
    """
    from woffl.assembly.network_optimizer import NetworkOptimizer, PowerFluidConstraint
    from woffl.gui.scotts_tools._common import worker_ceiling

    names = [wc.well_name for wc in well_configs]
    cur_oil = {w: float((test_rates.get(w) or (0, 0))[0] or 0.0) for w in names}
    cur_pf = {w: float((test_rates.get(w) or (0, 0))[1] or 0.0) for w in names}
    pumps = [c for c in current_choices.values() if c]
    nozzles = sorted({c[0] for c in pumps}) or ["12"]
    throats = sorted({c[1] for c in pumps}) or ["B"]
    total_pf = sum(cur_pf.values())
    header = plant.match_check_header(total_pf, n_pumps)
    for wc in well_configs:
        wc.ppf_surf_well = header
    pf = PowerFluidConstraint(
        total_rate=plant.match_check_budget_bpd(total_pf, n_pumps),
        pressure=header,
        rho_pf=_RHO_PF_DEFAULT,
    )
    opt = NetworkOptimizer(
        well_configs, pf, nozzles, throats, marginal_watercut=_SCENARIO_MARGINAL_WC
    )
    opt.run_all_batch_simulations(max_workers=worker_ceiling())

    rows = []
    for w in names:
        cc = current_choices.get(w)
        perf = opt.get_pump_performance(w, cc[0], cc[1]) if cc else None
        mo = float(perf["oil_rate"]) if perf else None
        mp = float(perf["lift_water"]) if perf else None
        to, tp = (cur_oil[w] or None), (cur_pf[w] or None)
        oil_ratio = (mo / to) if (mo is not None and to) else None
        pf_ratio = (mp / tp) if (mp is not None and tp) else None
        rows.append(
            {
                "well": w,
                "pump": (f"{cc[0]}{cc[1]}" if cc else "—"),
                "test_oil": to,
                "model_oil": mo,
                "oil_ratio": oil_ratio,
                "oil_flag": match_flag(oil_ratio),
                "test_pf": tp,
                "model_pf": mp,
                "pf_ratio": pf_ratio,
                "pf_flag": match_flag(pf_ratio),
                # Suction pressure + sonic status ride along for the
                # match-health scorecard; .get() so a test fake without the
                # keys degrades to None instead of raising.
                "model_psu": (
                    float(perf["suction_pressure"])
                    if perf and perf.get("suction_pressure") is not None
                    else None
                ),
                "sonic": (
                    bool(perf["sonic_status"])
                    if perf and perf.get("sonic_status") is not None
                    else None
                ),
            }
        )
    return rows, header


# ---------------------------------------------------------------------------
# PF-pressure what-if (current pumps, two forced headers)
# ---------------------------------------------------------------------------


def _model_at_forced_header(well_configs, header_psi: float, current_choices: dict):
    """Model every well at its CURRENT pump with the delivered header FORCED.

    Same plumbing as ``match_check`` but the header is the caller's number,
    not the plant's derivation — this exists so the PF what-if can ask "what
    do these wells do at pressure X" directly. Returns
    ``{well: (oil_bopd, pf_bpd, psu_psig, sonic) | None}`` (None = pump
    missing or unsolvable at this header; psu/sonic are None when the batch
    row lacks them). ``sonic`` True means the solver returned the cavitation
    floor: throat entry at sonic velocity, so psu and oil are pinned there
    and only PF responds to the delivered pressure.
    """
    from woffl.assembly.network_optimizer import NetworkOptimizer, PowerFluidConstraint
    from woffl.gui.scotts_tools._common import worker_ceiling

    pumps = [c for c in current_choices.values() if c]
    nozzles = sorted({c[0] for c in pumps}) or ["12"]
    throats = sorted({c[1] for c in pumps}) or ["B"]
    for wc in well_configs:
        wc.ppf_surf_well = header_psi
    pf = PowerFluidConstraint(
        total_rate=_EVAL_CAP_FALLBACK_BPD, pressure=header_psi, rho_pf=_RHO_PF_DEFAULT
    )
    opt = NetworkOptimizer(
        well_configs, pf, nozzles, throats, marginal_watercut=_SCENARIO_MARGINAL_WC
    )
    opt.run_all_batch_simulations(max_workers=worker_ceiling())

    out = {}
    for wc in well_configs:
        w = wc.well_name
        cc = current_choices.get(w)
        perf = opt.get_pump_performance(w, cc[0], cc[1]) if cc else None
        out[w] = (
            (
                float(perf["oil_rate"]),
                float(perf["lift_water"]),
                # suction pressure rides along for IPR-landing reporting;
                # .get() so a test fake without the key degrades to None
                (
                    float(perf["suction_pressure"])
                    if perf.get("suction_pressure") is not None
                    else None
                ),
                (
                    bool(perf["sonic_status"])
                    if perf.get("sonic_status") is not None
                    else None
                ),
            )
            if perf
            else None
        )
    return out


def pf_what_if_rows(
    names: list, current_choices: dict, base: dict, scen: dict, test_rates: dict
) -> list[dict]:
    """Per-well comparison rows for the PF what-if — pure, for testability.

    ``base`` / ``scen`` are ``_model_at_forced_header`` outputs at the two
    pressures. ``projected_oil`` anchors the scenario on the well's MEASURED
    test oil × the model's ratio between the two pressures — model bias
    cancels in the ratio, so it's the most trustworthy per-well number
    (same trick as ``_score_existing_choices``). Model absolutes are still
    reported for wells with no test.
    """
    rows = []
    for w in names:
        cc = current_choices.get(w)
        b, s = base.get(w), scen.get(w)
        to = float((test_rates.get(w) or (0, 0))[0] or 0.0)
        projected = None
        if b and s and b[0] > 0 and to > 0:
            projected = to * (s[0] / b[0])
        rows.append(
            {
                "well": w,
                "pump": f"{cc[0]}{cc[1]}" if cc else "—",
                "oil_base": b[0] if b else None,
                "oil_scen": s[0] if s else None,
                "d_oil": (s[0] - b[0]) if (b and s) else None,
                "pf_base": b[1] if b else None,
                "pf_scen": s[1] if s else None,
                "d_pf": (s[1] - b[1]) if (b and s) else None,
                "test_oil": to or None,
                "projected_oil": projected,
            }
        )
    return rows


def pf_what_if_totals(rows: list[dict]) -> dict:
    """Pad totals over the rows that solved at BOTH pressures (a well that
    solves at only one would skew the delta with an apples-to-oranges sum).
    ``projected_d_oil`` sums the test-anchored deltas of the wells that have
    one."""
    solved = [r for r in rows if r["d_oil"] is not None]
    return {
        "n_solved": len(solved),
        "n_unsolved": len(rows) - len(solved),
        "oil_base": sum(r["oil_base"] for r in solved),
        "oil_scen": sum(r["oil_scen"] for r in solved),
        "d_oil": sum(r["d_oil"] for r in solved),
        "pf_base": sum(r["pf_base"] for r in solved),
        "pf_scen": sum(r["pf_scen"] for r in solved),
        "projected_d_oil": sum(
            r["projected_oil"] - r["test_oil"]
            for r in solved
            if r["projected_oil"] is not None
        ),
    }


def base_vs_future_rows(
    per_base: list[dict], per_fut: list[dict], future_wells: set
) -> list[dict]:
    """Merge two ``evaluate_fixed_scenario`` per-well lists into comparison
    rows — pure, for testability. Future wells have no base column; existing
    wells carry the Δ the added PF demand cost them (header droop)."""
    base_by = {r["well"]: r for r in per_base}
    fut_by = {r["well"]: r for r in per_fut}
    rows = []
    for w in sorted(set(base_by) | set(fut_by)):
        b, f = base_by.get(w), fut_by.get(w)
        rows.append(
            {
                "well": w,
                "future": w in future_wells,
                "pump": (f or b).get("pump"),
                "oil_base": b["oil"] if b else None,
                "oil_future": f["oil"] if f else None,
                "d_oil": (f["oil"] - b["oil"]) if (b and f) else None,
                "pf_base": b["pf"] if b else None,
                "pf_future": f["pf"] if f else None,
            }
        )
    return rows


def base_vs_future_totals(rows: list[dict], meta_base: dict, meta_fut: dict) -> dict:
    """Pad-level summary of a base-vs-future comparison: the future wells'
    combined oil, the existing wells' combined Δ (what the extra PF demand
    cost them), and the coupled header at each state."""
    existing = [r for r in rows if not r["future"] and r["d_oil"] is not None]
    fut = [r for r in rows if r["future"] and r["oil_future"] is not None]
    return {
        "oil_base": meta_base["total_oil_bopd"],
        "oil_future": meta_fut["total_oil_bopd"],
        "d_oil": meta_fut["total_oil_bopd"] - meta_base["total_oil_bopd"],
        "header_base": meta_base["header_psi"],
        "header_future": meta_fut["header_psi"],
        "pf_base": meta_base["total_pf_bpd"],
        "pf_future": meta_fut["total_pf_bpd"],
        "future_oil": sum(r["oil_future"] for r in fut),
        "existing_d_oil": sum(r["d_oil"] for r in existing),
        "n_future": len(fut),
    }


def pf_pressure_what_if(
    well_configs,
    current_choices: dict,
    test_rates: dict,
    pf_base_psi: float,
    pf_scenario_psi: float,
):
    """Model all wells at their CURRENT (reviewed) pumps at two FORCED
    delivered PF pressures and diff them.

    Deliberately bypasses the booster coupling — the question is "what does
    delivered PF pressure X buy across the pad", assuming the plant can
    supply that pressure at the resulting flow (the capability plot on the
    Configure page is the place to sanity-check that assumption). Two full
    batch passes. Returns ``(rows, totals)``.
    """
    names = [wc.well_name for wc in well_configs]
    base = _model_at_forced_header(well_configs, float(pf_base_psi), current_choices)
    scen = _model_at_forced_header(
        well_configs, float(pf_scenario_psi), current_choices
    )
    rows = pf_what_if_rows(names, current_choices, base, scen, test_rates)
    return rows, pf_what_if_totals(rows)


# ---------------------------------------------------------------------------
# Choke / shut-in plan (installed pumps held — no JPCO)
# ---------------------------------------------------------------------------


def _choke_frontier(
    points: list[tuple[Optional[float], float, float, Optional[float]]],
) -> list[tuple[Optional[float], float, float, Optional[float]]]:
    """Efficient staircase of ``(delivered_psi, oil, pf, psu)`` options.

    Drops every dominated option (another option with <= PF and >= oil,
    solver noise included) and returns the survivors sorted by PF
    DESCENDING — full open first, shut-in ``(None, 0, 0)`` last. Down the
    list both oil and PF strictly decrease, so every adjacent pair is a
    valid trim step with a positive, finite slope (BOPD given up per BPD
    of PF freed).
    """
    kept: list[tuple[Optional[float], float, float, Optional[float]]] = []
    for psi, oil, pf, psu in sorted(points, key=lambda t: (t[2], t[1])):
        if kept and pf <= kept[-1][2] + 1e-9:
            if oil > kept[-1][1] + 1e-9:
                kept[-1] = (psi, oil, pf, psu)  # same PF, more oil: replaces
            continue
        if kept and oil <= kept[-1][1] + 1e-9:
            continue  # more PF for no more oil: dominated
        kept.append((psi, oil, pf, psu))
    kept.reverse()
    return kept


def _trim_to_budget(wells: list[dict], budget: float) -> tuple[float, float, Optional[float]]:
    """Equal-slope greedy walk: step the cheapest well down one option at a
    time until total PF fits ``budget``.

    ``wells`` entries are ``{"opts": staircase, "idx": current option}``;
    ``idx`` is advanced in place. Returns ``(total_pf, total_oil, lam)`` —
    ``lam`` is the slope of the LAST trim taken (the pad's marginal bbl oil
    per bbl PF at the solution), None when no trim was needed. With concave
    well curves the walk equalizes marginal oil per bbl PF across the pad
    (the Kanu/Mach/Brown equal-slope optimum, at ladder resolution).
    """
    total_pf = sum(st["opts"][st["idx"]][2] for st in wells)
    total_oil = sum(st["opts"][st["idx"]][1] for st in wells)
    lam: Optional[float] = None
    while total_pf > budget:
        best: Optional[tuple[float, dict]] = None
        for st in wells:
            i = st["idx"]
            if i + 1 >= len(st["opts"]):
                continue  # already shut in
            _, oil_0, pf_0, _ = st["opts"][i]
            _, oil_1, pf_1, _ = st["opts"][i + 1]
            slope = (oil_0 - oil_1) / (pf_0 - pf_1)  # staircase: pf_0 > pf_1
            if best is None or slope < best[0]:
                best = (slope, st)
        if best is None:
            break  # everything already shut in
        slope, st = best
        _, oil_0, pf_0, _ = st["opts"][st["idx"]]
        st["idx"] += 1
        _, oil_1, pf_1, _ = st["opts"][st["idx"]]
        total_pf -= pf_0 - pf_1
        total_oil -= oil_0 - oil_1
        lam = slope
    return total_pf, total_oil, lam


def _oil_vogel(wc) -> Optional[InFlow]:
    """Oil-basis Vogel inflow anchored exactly like the solver's inflow
    (oil rate = total fluid x (1 - water cut), see
    NetworkOptimizer._create_well_objects). None when the config lacks a
    usable qwf/pwf/res_pres or the anchor is unphysical (pwf >= res_pres).
    """
    qwf = getattr(wc, "qwf", None)
    pwf = getattr(wc, "pwf", None)
    pres = getattr(wc, "res_pres", None)
    if qwf is None or pwf is None or pres is None:
        return None
    form_wc = getattr(wc, "form_wc", None) or 0.0
    try:
        return InFlow(
            qwf=float(qwf) * (1.0 - float(form_wc)),
            pwf=float(pwf),
            pres=float(pres),
        )
    except (ValueError, TypeError):
        return None


def _vogel_ipr_curve(wc) -> Optional[list[list[float]]]:
    """25-point Vogel IPR curve for the landing table's per-well chart.

    Anchored via ``_oil_vogel`` (shared with the evidence suction
    correction). Points are ``[oil_bopd, pwf_psi]`` from pwf = res_pres
    (oil 0) down to pwf = 0 (oil = vogel qmax), rounded to 1 decimal. None
    when the config lacks a usable anchor.
    """
    inflow = _oil_vogel(wc)
    if inflow is None:
        return None
    pres = inflow.pres
    n = 25
    return [
        [
            round(inflow.oil_flow(pres * (n - 1 - i) / (n - 1), method="vogel"), 1),
            round(pres * (n - 1 - i) / (n - 1), 1),
        ]
        for i in range(n)
    ]


# Evidence gate: a model cavitation floor is only "contradicted" when it sits
# more than this far ABOVE the measured flowing-BHP floor (below that the
# field data CONFIRMS the model and the suction response is left alone).
_EVIDENCE_VIOLATION_MIN_PSI = 25.0

# A well-measured response slope (beta = -dBHP/dPpf) this steep falsifies a
# cavitation-pinned (zero-response) model even when the floor itself is
# confirmed. Field separation on M-Pad: insensitive wells measure
# beta <= 0.022, responsive wells >= 0.04, so 0.03 splits the groups cleanly.
_EVIDENCE_BETA_MIN = 0.03


def _apply_suction_evidence(
    grid: list[dict],
    levels: list[float],
    names: list[str],
    evidence: dict[str, dict],
    configs_by_name: dict,
) -> dict[str, dict]:
    """Overwrite the priced grid's suction response with field evidence on
    wells where measurement contradicts the model's cavitation floor.

    Per well with an evidence row (plain dict: floor/psu_ref/beta/
    beta_source/...): find the top solvable ladder level k*; if the model is
    cavitation-pinned there (sonic True), the evidence falsifies it - the
    model's suction floor sits more than ``_EVIDENCE_VIOLATION_MIN_PSI``
    above the measured floor, OR a well-measured beta of at least
    ``_EVIDENCE_BETA_MIN`` demonstrates a suction response the pinned model
    denies - and the evidence + Vogel anchor are usable, replace every
    solvable level k <= k* with the field response::

        psu_e = psu_ref + beta * (levels[k*] - levels[k])
        oil_e = 0 if psu_e >= res_pres else oil_full * q(psu_e) / q(psu_ref)

    PF stays the model's (validated hydraulics); the sonic flag is cleared
    (corrected points are not cavitation-pinned). At k* the point anchors to
    (oil_full, psu_ref) exactly. A psu_ref at or above the fit's res_pres is
    unusable (InFlow rejects it) -> skip, no correction.

    Mutates ``grid`` in place and returns bookkeeping for row assembly:
    ``{well: {"floor", "violation", "beta", "beta_source", "gate"}}`` where
    ``gate`` is "floor" (violation, the stronger claim, wins when both
    trigger) or "response", and ``violation`` is measured against the
    ORIGINAL model floor at k* (the grid's psu there is psu_ref after the
    overwrite).
    """
    corrected: dict[str, dict] = {}
    for w in names:
        ev = evidence.get(w)
        if not ev:
            continue
        k_star = None
        for k in range(len(levels) - 1, -1, -1):
            if grid[k].get(w) is not None:
                k_star = k
                break
        if k_star is None:
            continue  # never solvable: nothing to correct
        v = grid[k_star][w]
        oil_full = float(v[0])
        psu_model = float(v[2]) if len(v) > 2 and v[2] is not None else None
        sonic = v[3] if len(v) > 3 else None
        if sonic is not True:
            continue  # model suction already responsive
        floor = ev.get("floor")
        floor_violated = (
            floor is not None
            and psu_model is not None
            and psu_model - float(floor) > _EVIDENCE_VIOLATION_MIN_PSI
        )
        beta = ev.get("beta")
        # only a MEASURED response can falsify the pinned model: pad- and
        # default-sourced betas are fleet priors, never grounds to correct
        response_shown = (
            ev.get("beta_source") == "well"
            and beta is not None
            and float(beta) >= _EVIDENCE_BETA_MIN
        )
        if not (floor_violated or response_shown):
            continue  # evidence CONFIRMS the model (floor and response)
        psu_ref = ev.get("psu_ref")
        if psu_ref is None or beta is None:
            continue
        inflow = _oil_vogel(configs_by_name.get(w))
        if inflow is None:
            continue
        psu_ref = float(psu_ref)
        beta = float(beta)
        if psu_ref >= inflow.pres:
            continue  # measured suction above the fit's res_pres: unusable
        q_ref = inflow.oil_flow(psu_ref)
        if q_ref <= 0.0:
            continue
        for k in range(k_star + 1):
            vk = grid[k].get(w)
            if vk is None:
                continue
            pf_k = float(vk[1])
            psu_e = psu_ref + beta * (levels[k_star] - levels[k])
            oil_e = (
                0.0
                if psu_e >= inflow.pres
                else oil_full * inflow.oil_flow(psu_e) / q_ref
            )
            grid[k][w] = (oil_e, pf_k, psu_e, None)
        corrected[w] = {
            "floor": float(floor) if floor is not None else None,
            "violation": (
                max(0.0, psu_model - float(floor))
                if floor is not None and psu_model is not None
                else None
            ),
            "beta": beta,
            "beta_source": ev.get("beta_source"),
            "gate": "floor" if floor_violated else "response",
        }
    return corrected


def _classify_action(basis: str, psi, oil: float, pf: float, header_psi: float) -> str:
    """Action label for one well's chosen option at a candidate header.

    Anything delivered below the header is an operator action - including a
    dominant lower-pressure point picked at idx 0 (the free choke to the
    sonic knee).
    """
    if basis == "none":
        return "excluded"
    if pf <= 0.0 and oil <= 0.0:
        return "shut"
    if basis == "test":
        return "hold"
    if psi is not None and psi < header_psi - 1e-6:
        return "choke"
    return "full"


_ACTION_RANK = {"shut": 0, "choke": 1, "hold": 2, "full": 3, "excluded": 4}


def run_choke_optimization(
    well_configs: list,
    plant: PadPlant,
    n_pumps: int | None,
    current_choices: dict,
    test_rates: dict,
    *,
    n_levels: int = 10,
    progress: Optional[Callable] = None,
    evidence: dict[str, dict] | None = None,
):
    """Short-term PF plan with every well HELD on its installed pump: no
    jet-pump changeouts, only per-well PF throttling (choke back) or shut-in.

    The decision problem: sweep candidate delivered headers across
    ``plant.pressure_window(n_pumps)``; at each header the bank's PF budget
    is ``plant.budget_at_pressure`` (the capability frontier — with a pump
    down, pass the reduced ``n_pumps``); every well may run FULL OPEN
    (delivered = header), CHOKED to any lower ladder pressure (its wellhead
    PF throttle burns the difference, so a choked well is exactly a well at
    a lower delivered pressure), or SHUT IN. Wells start full open and
    ``_trim_to_budget`` walks the cheapest trims until the budget fits; the
    header with the most total oil wins.

    One ``_model_at_forced_header`` batch pass per ladder level prices every
    well at every level; a well with no model solution at ANY level is HELD
    at its measured test rates (only shut-in offered — you cannot price a
    choke you cannot model), and a well with neither model nor test is
    excluded with a zero contribution.

    Args:
        well_configs (list): WellConfig list (active wells with saved fits).
        plant (PadPlant): booster plant; must be ``coupling="free_pressure"``
            (I/M). fixed_curve raises ValueError — its header is not a
            decision variable.
        n_pumps (int | None): booster pumps online (the reduced count during
            the outage).
        current_choices (dict): well_name -> (nozzle, throat) installed pump.
        test_rates (dict): well_name -> (oil_bopd, pf_bpd) measured median.
        n_levels (int): ladder size across the pressure window (default 10).
        progress (Callable): ``(step, total, pressure, pf, oil)`` per pass.
        evidence (dict | None): well_name -> evidence row (plain dict with
            floor/psu_ref/beta/beta_source/...) from field pressure history;
            None (default) leaves the run byte-identical to today.

    Returns:
        tuple: ``(rows, meta)``. Rows are sorted action-first (shut, choke,
        hold, full, excluded; biggest PF freed first) and carry per-well
        deltas vs full-open plus ``projected_oil`` (measured test oil x the
        model ratio chosen/today - model bias cancels, same anchoring as
        ``pf_what_if_rows``). When ``evidence`` falsifies a well's
        cavitation-pinned model suction - the model floor is violated or a
        well-measured response beta shows sensitivity the model denies -
        its suction response is replaced by the field data
        (``_apply_suction_evidence``) and the row says so via
        ``suction_basis``/``evidence_gate`` ("floor" | "response" | None;
        "floor" wins when both trigger)/``evidence_floor_psi``/
        ``floor_violation_psi``/``response_beta``/``beta_source``. ``meta``
        keeps the uniform
        contract keys the charts read (header_psi, total_pf_bpd,
        total_oil_bopd, sweep, ...) plus ``lambda_bopd_per_bpd``,
        ``frontier_cap_bpd``, ``header_today_psi``,
        ``projected_d_oil_bopd``, ``n_evidence_corrected``, the action
        counts, and ``ladder`` - the header-drop decision ladder: one rung
        per ladder level below the winning header, answering "if the bank
        degrades until the all-run header settles here, what is the best
        response and what does it gain over doing nothing".
    """
    if plant.coupling != "free_pressure":
        raise ValueError(
            "choke/shut-in plan needs a free-pressure plant (I/M-Pad): "
            "a fixed-curve station's header follows flow and is not a "
            "decision variable"
        )

    names = [wc.well_name for wc in well_configs]
    # IPR context for the landing table: reservoir pressure per well
    res_pres = {
        wc.well_name: getattr(wc, "res_pres", None) for wc in well_configs
    }
    ipr_curves = {wc.well_name: _vogel_ipr_curve(wc) for wc in well_configs}
    p_lo, p_hi = plant.pressure_window(n_pumps)
    levels = [
        p_lo + (p_hi - p_lo) * i / (n_levels - 1) for i in range(n_levels)
    ]

    # -- price the grid: every well at its installed pump at every level ----
    # sonic flag per (well, ladder level): the row assembly reports whether
    # the chosen and full-open points sit at the cavitation floor
    sonic_at: dict[tuple, bool] = {}
    grid: list[dict] = []
    for k, level in enumerate(levels):
        grid.append(_model_at_forced_header(well_configs, level, current_choices))
        for w, v in grid[k].items():
            if v is not None and len(v) > 3 and v[3] is not None:
                sonic_at[(w, levels[k])] = bool(v[3])
        if progress:
            progress(k + 1, n_levels + 1, level, 0.0, 0.0)

    # -- evidence-corrected suction response: overwrite the grid where field
    #    data contradicts the model's cavitation floor (PF stays model). The
    #    frontier/trim/ladder/charts all inherit the corrected grid.
    corrected: dict[str, dict] = {}
    if evidence:
        configs_by_name = {wc.well_name: wc for wc in well_configs}
        corrected = _apply_suction_evidence(
            grid, levels, names, evidence, configs_by_name
        )
        for w in corrected:
            # corrected points are not cavitation-pinned
            for level in levels:
                sonic_at.pop((w, level), None)

    # -- today anchor: the header the plant settles to for the measured PF
    #    draw; per-well model-at-today is the bias reference for projections
    pf_today = sum(float((test_rates.get(w) or (0, 0))[1] or 0.0) for w in names)
    header_today, _ = settled_header(
        plant, pf_today, plant.warm_start_psi(n_pumps), n_pumps
    )
    today = _model_at_forced_header(well_configs, header_today, current_choices)
    if progress:
        progress(n_levels + 1, n_levels + 1, header_today, pf_today, 0.0)

    # -- per-well option sets at a candidate header index h ------------------
    def _well_states(h: int) -> list[dict]:
        states = []
        for w in names:
            pts = []
            for k in range(h + 1):
                v = grid[k].get(w)
                if v is not None:
                    pts.append(
                        (
                            levels[k],
                            float(v[0]),
                            float(v[1]),
                            float(v[2]) if len(v) > 2 and v[2] is not None else None,
                        )
                    )
            oil_t = float((test_rates.get(w) or (0, 0))[0] or 0.0)
            pf_t = float((test_rates.get(w) or (0, 0))[1] or 0.0)
            if pts:
                opts = _choke_frontier(pts + [(None, 0.0, 0.0, None)])
                # RAW full-open reference: the highest ladder level that
                # solved. The staircase may start BELOW it when a lower
                # pressure dominates (more oil, less PF - a pump past its
                # sonic knee); row deltas are vs this raw point so that
                # "free oil" chokes read as gains, not zeros.
                full_raw = pts[-1]
                basis = "model"
            elif oil_t > 0.0 or pf_t > 0.0:
                # unmodelable: hold measured rates, or shut in
                opts = [(None, oil_t, pf_t, None), (None, 0.0, 0.0, None)]
                full_raw = opts[0]
                basis = "test"
            else:
                opts = [(None, 0.0, 0.0, None)]
                full_raw = opts[0]
                basis = "none"
            states.append(
                {"well": w, "opts": opts, "idx": 0, "basis": basis, "full_raw": full_raw}
            )
        return states

    # -- sweep candidate headers, keep the most-oil one ----------------------
    best = None
    sweep = []
    for h, level in enumerate(levels):
        budget = plant.budget_at_pressure(level, n_pumps)
        if not budget or budget <= 0:
            continue
        wells = _well_states(h)
        total_pf, total_oil, lam = _trim_to_budget(wells, budget)
        sweep.append(
            {
                "header_psi": level,
                "total_pf_bpd": total_pf,
                "total_oil_bopd": total_oil,
            }
        )
        if best is None or total_oil > best["total_oil"]:
            best = {
                "P": level,
                "budget": budget,
                "total_pf": total_pf,
                "total_oil": total_oil,
                "lam": lam,
                "wells": wells,
            }

    if best is None:
        raise RuntimeError(plant.infeasible_sweep_msg)

    # -- rows at the winning header ------------------------------------------
    rows = []
    counts = {"full": 0, "choke": 0, "shut": 0, "hold": 0, "excluded": 0}
    for st in best["wells"]:
        w = st["well"]
        psi, oil, pf, psu = st["opts"][st["idx"]]
        psi_f, oil_f, pf_f, psu_f = st["full_raw"]
        action = _classify_action(st["basis"], psi, oil, pf, best["P"])
        counts[action] += 1

        # test-anchored projection: measured oil x model(chosen)/model(today)
        oil_t = float((test_rates.get(w) or (0, 0))[0] or 0.0)
        t = today.get(w)
        if action == "shut":
            projected = 0.0 if (oil_t > 0.0 or st["basis"] == "model") else None
        elif st["basis"] == "test":
            projected = oil_t or None
        elif oil_t > 0.0 and t is not None and float(t[0]) > 0.0:
            projected = oil_t * oil / float(t[0])
        else:
            projected = None

        # the cost of trimming this well one MORE step (who is next in line)
        nxt = None
        if st["idx"] + 1 < len(st["opts"]):
            _, oil_1, pf_1, _ = st["opts"][st["idx"] + 1]
            nxt = (oil - oil_1) / (pf - pf_1) if pf > pf_1 else None

        # evidence provenance: floor/violation for ANY model-basis well with
        # an evidence row; beta/beta_source only when the suction response
        # was actually corrected. For corrected wells full_raw's psu is
        # psu_ref (post-overwrite), so the violation comes from the
        # bookkeeping captured against the ORIGINAL model floor.
        ev = (evidence or {}).get(w)
        corr = corrected.get(w)
        ev_floor = None
        ev_violation = None
        if st["basis"] == "model" and ev is not None:
            ev_floor = ev.get("floor")
            if corr is not None:
                ev_violation = corr["violation"]
            elif ev_floor is not None and psu_f is not None:
                ev_violation = max(0.0, float(psu_f) - float(ev_floor))

        cc = current_choices.get(w)
        rows.append(
            {
                "well": w,
                "pump": f"{cc[0]}{cc[1]}" if cc else None,
                "basis": st["basis"],
                "action": action,
                "delivered_psi": psi,
                "choke_dp_psi": (best["P"] - psi) if psi is not None else None,
                # full-open reference point (raw, highest solvable level)
                "delivered_full_psi": psi_f,
                "oil_full": oil_f,
                "pf_full": pf_f,
                # IPR landing: suction pressure at the chosen and full-open
                # settings, plus reservoir pressure for drawdown
                "psu": psu,
                "psu_full": psu_f,
                # cavitation-floor flags at the chosen / full-open points
                # (None for held/shut/test-basis points off the ladder)
                "sonic": sonic_at.get((w, psi)),
                "sonic_full": sonic_at.get((w, psi_f)),
                "res_pres": res_pres.get(w),
                "ipr_curve": ipr_curves.get(w) if st["basis"] == "model" else None,
                "pf": pf,
                "oil": oil,
                "d_oil_vs_full": oil - oil_f,
                "d_pf_vs_full": pf - pf_f,
                "test_oil": oil_t or None,
                "test_pf": float((test_rates.get(w) or (0, 0))[1] or 0.0) or None,
                "projected_oil": projected,
                "next_trim_bopd_per_bpd": nxt,
                # evidence-corrected suction response provenance
                "evidence_floor_psi": ev_floor,
                "floor_violation_psi": ev_violation,
                "response_beta": corr["beta"] if corr else None,
                "beta_source": corr["beta_source"] if corr else None,
                "suction_basis": "evidence" if corr else "model",
                "evidence_gate": corr["gate"] if corr else None,
            }
        )
    rows.sort(key=lambda r: (_ACTION_RANK[r["action"]], r["d_pf_vs_full"]))

    projected_d = sum(
        r["projected_oil"] - r["test_oil"]
        for r in rows
        if r["projected_oil"] is not None and r["test_oil"] is not None
    )

    # -- header-drop decision ladder ------------------------------------------
    # Operator question: "the header is settling X psi below the plan - what
    # do I do?" Each rung scales the bank's flow frontier by the factor s
    # that makes the ALL-RUN header settle at that rung's level (demand at P
    # over frontier at P: demand rises and the frontier falls with pressure,
    # so the anchor is unique), then re-runs the same header sweep and
    # equal-slope trim against the degraded frontier. Pure allocation over
    # the already-priced grid - no extra solves.
    modelable = {w: any(g.get(w) is not None for g in grid) for w in names}
    ladder = []
    for j, P in enumerate(levels):
        if P >= best["P"] - 1e-6:
            continue
        run_all_oil = 0.0
        demand = 0.0
        for w in names:
            v = grid[j].get(w)
            if v is not None:
                run_all_oil += float(v[0])
                demand += float(v[1])
            elif not modelable[w]:
                # never modelable: held at measured rates (header-independent)
                tr = test_rates.get(w) or (0, 0)
                run_all_oil += float(tr[0] or 0.0)
                demand += float(tr[1] or 0.0)
            # modelable but no solution AT this level: cannot lift here -> 0
        cap = plant.budget_at_pressure(P, n_pumps)
        if not cap or cap <= 0 or demand <= 0:
            continue
        s = demand / cap
        best_r = None
        for h, level in enumerate(levels):
            # + epsilon: s*cap reconstructs the rung's own budget as EXACTLY
            # its all-run demand, and float dust in s*cap must never force a
            # spurious trim at the rung's own level (1e-6 bpd is nothing)
            budget_h = s * (plant.budget_at_pressure(level, n_pumps) or 0.0) + 1e-6
            if budget_h <= 0:
                continue
            wells_h = _well_states(h)
            _pf_h, oil_h, _lam_h = _trim_to_budget(wells_h, budget_h)
            if best_r is None or oil_h > best_r[0]:
                best_r = (oil_h, level, wells_h)
        if best_r is None:
            continue
        oil_r, header_r, wells_r = best_r
        actions = []
        for st in wells_r:
            psi_r, oil_w, pf_w, _psu = st["opts"][st["idx"]]
            act = _classify_action(st["basis"], psi_r, oil_w, pf_w, header_r)
            if act in ("choke", "shut"):
                actions.append({"well": st["well"], "action": act, "set_psi": psi_r})
        ladder.append(
            {
                "drop_psi": best["P"] - P,
                "settles_psi": P,
                "run_all_oil_bopd": run_all_oil,
                "best_header_psi": header_r,
                "plan_oil_bopd": oil_r,
                "gain_bopd": oil_r - run_all_oil,
                "actions": actions,
            }
        )
    ladder.sort(key=lambda r: r["drop_psi"])
    meta = {
        "mode": "choke",
        "n_pumps": n_pumps,
        "header_psi": best["P"],
        "total_pf_bpd": best["total_pf"],
        "total_oil_bopd": best["total_oil"],
        "frontier_cap_bpd": best["budget"],
        "pf_slack": best["total_pf"] < best["budget"] - 1e-6,
        "lambda_bopd_per_bpd": best["lam"],
        "header_today_psi": header_today,
        "projected_d_oil_bopd": projected_d,
        "suction_psi": plant.suction_psi(),
        "min_total_flow": plant.flow_window(n_pumps)[0],
        "n_full": counts["full"],
        "n_choked": counts["choke"],
        "n_shut": counts["shut"],
        "n_held": counts["hold"],
        "n_excluded": counts["excluded"],
        "n_evidence_corrected": len(corrected),
        "converged": True,  # a sweep has no fixed point to miss
        "history": [],
        "sweep": sweep,
        "ladder": ladder,
        **plant.flags(best["total_pf"], n_pumps),
    }
    return rows, meta
