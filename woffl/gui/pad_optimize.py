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
            }
        )
    return rows, header
