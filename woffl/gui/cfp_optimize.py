"""Joint optimization of the four CFP-side pads (J/G/C/B) against one plant.

Separate from ``pad_optimize`` because the single-pad assumptions there don't
hold: four pads share one plant, each pad receives a DIFFERENT delivered
power-fluid pressure, and the binding quantity is TOTAL WATER rather than power
fluid. The scoring/parsimony/reconciliation helpers are reused, not copied.

THE MODEL
---------
Plant discharge is a **decision variable**, swept like the I-Pad / M-Pad
free-pressure plants — not a fixed-point outcome. Operators set it by throttling
the disposal well (Scott, 2026-07-29), which is why 120 days of metered
throughput vs measured discharge shows almost no relationship (-1.8 psi per
1,000 BWPD, r²=0.03) against the -17.5 the pump curve implies.

At each candidate discharge ``P``:

1. ``capacity = plant.budget_at_pressure(P)`` — TOTAL water the machines pass
   there. Rising P means falling capacity: +500 psi costs ~28,800 BWPD. That
   trade is the whole optimization.
2. ``room = capacity - exogenous_bwpd`` — what's left for the four pads after
   the water we don't control (L/K, the S/H/I formation water that passes
   through to CFP, R/D, and imperfect-disposal carryover from any pad including
   F and M). Non-positive room means P is unreachable.
3. Each well is given its pad's delivered PF: B/G/J ride the discharge (measured
   anchor, else the line-dP table); C-Pad holds its own booster pressure.
4. Optimize on ``water_key="totl_wat"`` against ``room``.
5. Re-check that the resulting load actually fits: ``load = pad water +
   exogenous <= capacity``. If not, P cannot be held and the trial is dropped.

Highest total oil among feasible trials wins.

WHAT IS AND ISN'T VALIDATED
---------------------------
The STRUCTURE is validated (pressure is a decision; capacity falls with
pressure; delivery is per-pad). The MAGNITUDES are not: the provisional curve
passes only ~95,000 BWPD at the plant's measured 2,792 psi against ~112,300
metered, so it will cap the feasible pressure roughly 300 psi below where the
plant really runs. Every run therefore carries ``provisional_curve=True`` in its
meta and the page badges it. The acceptance test for replacement coefficients is
in ``woffl.assembly.cfp_plant``: pass ~112,000 BWPD at ~2,790 psi.
"""

from typing import Optional

from woffl.assembly import cfp_plant as _cfp

# CFPMachineSubsetUnvalidated is re-exported deliberately: `run_joint_optimization`
# raises it (via `plant.machines_for`) and the page needs to catch it, so callers
# get it from here rather than reaching into the plant module.
from woffl.gui.cfp_pad_plant import CFPMachineSubsetUnvalidated, CFPPlant  # noqa: F401

_RHO_PF_DEFAULT = 62.4
# PowerFluidConstraint validates 1000 <= pressure <= 5000. A pad's delivered PF
# can fall below that at a low discharge, so clamp and REPORT rather than raise.
_PF_MIN, _PF_MAX = 1000.0, 5000.0


def _clamp_pf(psi: float) -> tuple[float, bool]:
    """Clamp a delivered PF into the constraint band, flagging that it happened."""
    lo, hi = _PF_MIN, _PF_MAX
    if psi < lo:
        return lo, True
    if psi > hi:
        return hi, True
    return float(psi), False


def delivered_by_pad(
    plant: CFPPlant,
    disch_p: float,
    pads: list,
    *,
    c_pad_pf_psi: float,
    measured_pad_pf: Optional[dict] = None,
) -> tuple[dict, list]:
    """Delivered PF per pad at a plant discharge.

    Returns ``(per_pad_psi, clamped_pads)``. Pads the plant supplies (B/G/J) get
    the measured-anchor delivery when a measurement is available and the line-dP
    table otherwise; every other pad — C-Pad, and any pad added later that turns
    out to be boosted on-pad — gets ``c_pad_pf_psi``.
    """
    measured = measured_pad_pf or {}
    out, clamped = {}, []
    for pad in pads:
        psi = plant.delivered_pf_for_pad(pad, disch_p, measured.get(pad))
        if psi is None:  # not plant-supplied — its own booster holds the pressure
            psi = float(c_pad_pf_psi)
        psi, was_clamped = _clamp_pf(psi)
        out[pad] = psi
        if was_clamped:
            clamped.append(pad)
    return out, clamped


def _assign_well_pressures(well_configs, per_pad: dict, fallback: float) -> None:
    """Stamp each well with its pad's delivered PF (mutates ``ppf_surf_well``)."""
    for wc in well_configs:
        wc.ppf_surf_well = float(per_pad.get(wc.pad, fallback))


def run_joint_optimization(
    pad_configs: dict,
    plant: CFPPlant,
    n_machines,
    nozzles,
    throats,
    method: str,
    marginal_wc,
    *,
    exogenous_bwpd: float,
    c_pad_pf_psi: float,
    measured_pad_pf: Optional[dict] = None,
    n_steps: int = 9,
    parsimony_bopd: float = 20.0,
    progress=None,
):
    """Optimize all four CFP pads together over a plant-discharge sweep.

    Args:
        pad_configs: ``{pad_letter: [WellConfig, ...]}``. Wells keep their own
            ``pad`` field; this mapping is just how the caller groups them.
        plant: a :class:`~woffl.gui.cfp_pad_plant.CFPPlant`.
        n_machines: machines online (3, or 2/1 once the per-machine curve is
            validated — otherwise :class:`CFPMachineSubsetUnvalidated` is
            raised, deliberately, rather than emitting extrapolated numbers).
        exogenous_bwpd: water reaching the plant that this optimization does not
            control. NOT derivable bottom-up from well tests — carryover from
            any pad makes that wrong — so it comes from the metered total.
        c_pad_pf_psi: C-Pad's own booster PF pressure (~3,400 psi measured).
        measured_pad_pf: ``{pad: psi}`` live pad PF for the measured anchor. Use
            the header CLUSTER, not a pad median: B/G/J/C are ESP-mixed, so only
            a minority of wells sit on the JP PF header.

    Returns ``(results, optimizer, meta)``. ``results``/``optimizer`` are the
    winning trial's; ``meta`` always records whether anything was feasible, and
    never presents a clamp or an extrapolation as a physics result.
    """
    from woffl.assembly.network_optimizer import (
        NetworkOptimizer,
        PowerFluidConstraint,
    )
    from woffl.assembly.optimization_algorithms import (
        apply_parsimony,
        derive_pad_marginal_wc,
        optimize,
    )
    from woffl.assembly.parallelism import worker_ceiling

    # Raises for an unvalidated subset BEFORE any compute — fail loudly, early.
    machines = plant.machines_for(n_machines)

    pads = sorted(pad_configs)
    all_wells = [wc for pad in pads for wc in pad_configs[pad]]
    if not all_wells:
        raise ValueError("no wells to optimize")

    p_floor, p_ceiling = plant.pressure_window(n_machines)
    steps = max(int(n_steps), 2)
    pressures = [
        p_floor + (p_ceiling - p_floor) * i / (steps - 1) for i in range(steps)
    ]

    sweep, best, skipped = [], None, []
    for idx, P in enumerate(pressures):
        capacity = plant.budget_at_pressure(P, n_machines)
        room = capacity - float(exogenous_bwpd)
        if room <= 0:
            # The machines can't even move the water we don't control at this
            # pressure, let alone the pads'. Record why — a silently missing
            # trial is indistinguishable from one that simply made less oil.
            skipped.append(
                {
                    "P": P,
                    "capacity": capacity,
                    "room": room,
                    "reason": "exogenous water alone exceeds plant capacity here",
                }
            )
            if progress:
                progress(idx + 1, steps, P, 0.0, 0.0)
            continue

        per_pad, clamped = delivered_by_pad(
            plant,
            P,
            pads,
            c_pad_pf_psi=c_pad_pf_psi,
            measured_pad_pf=measured_pad_pf,
        )
        _assign_well_pressures(all_wells, per_pad, fallback=c_pad_pf_psi)

        constraint_psi, _ = _clamp_pf(P)
        pf = PowerFluidConstraint(
            total_rate=room, pressure=constraint_psi, rho_pf=_RHO_PF_DEFAULT
        )
        opt = NetworkOptimizer(
            all_wells,
            pf,
            nozzles,
            throats,
            marginal_wc if marginal_wc is not None else 1.0,
        )
        opt.run_all_batch_simulations(max_workers=worker_ceiling())

        # Marginal-WC gate on the TOTAL-water basis (Scott's decision: the
        # plant's capacity basis is total water, formation + lift).
        gate, slack = derive_pad_marginal_wc(opt.batch_results, room, "totl_wat")
        if marginal_wc is None:
            opt.marginal_watercut = gate
            mwc_used, mwc_source = gate, "auto (plant-derived)"
        else:
            mwc_used, mwc_source = marginal_wc, "manual"

        results = optimize(opt, method=method, water_key="totl_wat")
        results, swaps = apply_parsimony(results, opt, "totl_wat", parsimony_bopd)

        pad_water = sum(r.predicted_total_water for r in results)
        total_oil = sum(r.predicted_oil_rate for r in results)
        load = pad_water + float(exogenous_bwpd)

        # Re-check feasibility against the load the solution actually produces.
        # The constraint above bounds pad water by `room`, so this should hold —
        # it's here because "should hold" is not "does hold", and a pressure the
        # plant can't sustain must never win.
        if load > capacity + 1e-6:
            skipped.append(
                {
                    "P": P,
                    "capacity": capacity,
                    "load": load,
                    "reason": "solution's water load exceeds plant capacity here",
                }
            )
            if progress:
                progress(idx + 1, steps, P, pad_water, total_oil)
            continue

        flags = plant.flags(load, n_machines)
        rec = {
            "P": P,
            "capacity": capacity,
            "room": room,
            "pad_water": pad_water,
            "load": load,
            "total_oil": total_oil,
            "per_pad_pf": dict(per_pad),
            "clamped_pads": list(clamped),
            "results": results,
            "opt": opt,
            "mwc_used": mwc_used,
            "mwc_source": mwc_source,
            "water_slack": slack,
            "parsimony_swaps": swaps,
            "trusted_band": bool(flags["trusted_band"]),
            "pinned": flags["pinned"],
        }
        sweep.append(rec)
        if best is None or total_oil > best["total_oil"]:
            best = rec
        if progress:
            progress(idx + 1, steps, P, pad_water, total_oil)

    meta = {
        "coupling": plant.coupling,
        "machines": ",".join(machines),
        "n_machines": len(machines),
        "exogenous_bwpd": float(exogenous_bwpd),
        "c_pad_pf_psi": float(c_pad_pf_psi),
        "pads": pads,
        "sweep": [
            {k: v for k, v in rec.items() if k not in ("results", "opt")}
            for rec in sweep
        ],
        "skipped": skipped,
        "n_feasible": len(sweep),
        # Loudly true until the SCADA coefficients land — see the module
        # docstring and cfp_plant's acceptance test.
        "provisional_curve": True,
        "measured_discharge_psi": _cfp.MEASURED_DISCHARGE_PSI,
        "measured_produced_water_bwpd": _cfp.MEASURED_PRODUCED_WATER_BWPD,
        "machine_subset_available": plant.machine_subset_available(),
    }

    if best is None:
        meta.update(
            {
                "feasible": False,
                "message": plant.infeasible_sweep_msg,
                "header_psi": None,
                "total_pf_bpd": None,
            }
        )
        return [], None, meta

    meta.update(
        {
            "feasible": True,
            "header_psi": best["P"],
            "plant_load_bwpd": best["load"],
            "plant_capacity_bwpd": best["capacity"],
            "pad_water_bwpd": best["pad_water"],
            "total_oil_bopd": best["total_oil"],
            "per_pad_pf": best["per_pad_pf"],
            "clamped_pads": best["clamped_pads"],
            "marginal_wc_used": best["mwc_used"],
            "marginal_wc_source": best["mwc_source"],
            "water_slack": best["water_slack"],
            "parsimony_swaps": best["parsimony_swaps"],
            "trusted_band": best["trusted_band"],
            "pinned": best["pinned"],
            # total_pf_bpd keeps the key the pad Results view already reads.
            "total_pf_bpd": best["pad_water"],
        }
    )
    return best["results"], best["opt"], meta


# ── model accuracy: is each well's JP model actually right? ─────────────────


def match_summary(rows: list) -> dict:
    """Roll a per-well match check into a trust verdict. PURE.

    The dashboard's whole answer — break-even WC, BOPD per 1,000 BWPD — is
    computed FROM these well models. If they don't reproduce the wells' own
    measured tests, the answer is arithmetic on noise, and the dashboard should
    say so rather than presenting a confident number.

    Bands follow ``pad_optimize.match_flag``: ✓ within 0.80-1.25x, ⚠ within
    0.50-2.0x, ✗ outside. Oil and PF are counted separately because they fail
    for different reasons — a loose IPR busts oil, a wrong nozzle/wear state
    busts PF.
    """
    n = len(rows)
    if not n:
        return {"n": 0, "trust": "none", "reason": "no wells checked"}

    def _count(key, mark):
        return sum(1 for r in rows if str(r.get(key, "")).startswith(mark))

    oil_ok, oil_bust = _count("oil_flag", "✓"), _count("oil_flag", "✗")
    pf_ok, pf_bust = _count("pf_flag", "✓"), _count("pf_flag", "✗")
    both_ok = sum(
        1
        for r in rows
        if str(r.get("oil_flag", "")).startswith("✓")
        and str(r.get("pf_flag", "")).startswith("✓")
    )
    frac = both_ok / n
    if frac >= 0.7 and (oil_bust + pf_bust) == 0:
        trust, reason = "good", f"{both_ok}/{n} wells match on both oil and PF"
    elif frac >= 0.5:
        trust, reason = (
            "fair",
            f"{both_ok}/{n} match on both; {oil_bust + pf_bust} bust — "
            "fix those wells before trusting the size of the answer",
        )
    else:
        trust, reason = (
            "poor",
            f"only {both_ok}/{n} match on both ({oil_bust} oil busts, "
            f"{pf_bust} PF busts) — the direction may hold but the BOPD figures "
            "are not reliable",
        )
    return {
        "n": n,
        "both_ok": both_ok,
        "frac_ok": frac,
        "oil_ok": oil_ok,
        "oil_bust": oil_bust,
        "pf_ok": pf_ok,
        "pf_bust": pf_bust,
        "trust": trust,
        "reason": reason,
        "worst": sorted(
            rows,
            key=lambda r: max(
                abs((r.get("oil_ratio") or 1.0) - 1.0),
                abs((r.get("pf_ratio") or 1.0) - 1.0),
            ),
            reverse=True,
        )[:5],
    }


def cfp_match_check(
    pad_configs: dict,
    plant: CFPPlant,
    discharge_psi: float,
    current_choices: dict,
    test_rates: dict,
    *,
    c_pad_pf_psi: float = 3400.0,
    measured_pad_pf: Optional[dict] = None,
    n_machines=None,
):
    """Model every CFP well at its CURRENT pump and its pad's CURRENT delivered
    PF, and compare against its own measured test.

    The CFP twin of ``pad_optimize.match_check``. That one forces a SINGLE
    header on every well, which would be wrong here — B/G/J each sit at a
    different line loss off the plant discharge and C-Pad is on its own booster.

    ``current_choices``/``test_rates`` are keyed by well name:
    ``{well: (nozzle, throat)}`` and ``{well: (oil_bopd, lift_water_bwpd)}``.

    Returns ``(rows, per_pad_pf)``.
    """
    from woffl.assembly.network_optimizer import (
        NetworkOptimizer,
        PowerFluidConstraint,
    )
    from woffl.gui.pad_optimize import match_flag
    from woffl.assembly.parallelism import worker_ceiling

    pads = sorted(pad_configs)
    wells = [wc for pad in pads for wc in pad_configs[pad]]
    if not wells:
        return [], {}

    per_pad, _clamped = delivered_by_pad(
        plant, discharge_psi, pads,
        c_pad_pf_psi=c_pad_pf_psi, measured_pad_pf=measured_pad_pf,
    )
    _assign_well_pressures(wells, per_pad, fallback=c_pad_pf_psi)

    pumps = [c for c in current_choices.values() if c]
    nozzles = sorted({c[0] for c in pumps}) or ["12"]
    throats = sorted({c[1] for c in pumps}) or ["B"]

    constraint_psi, _ = _clamp_pf(discharge_psi)
    opt = NetworkOptimizer(
        wells,
        PowerFluidConstraint(
            total_rate=max(plant.budget_at_pressure(discharge_psi, n_machines), 1.0),
            pressure=constraint_psi,
            rho_pf=_RHO_PF_DEFAULT,
        ),
        nozzles,
        throats,
        marginal_watercut=1.0,
    )
    opt.run_all_batch_simulations(max_workers=worker_ceiling())

    rows = []
    for wc in wells:
        w = wc.well_name
        cc = current_choices.get(w)
        perf = opt.get_pump_performance(w, cc[0], cc[1]) if cc else None
        mo = float(perf["oil_rate"]) if perf else None
        mp = float(perf["lift_water"]) if perf else None
        to, tp = test_rates.get(w) or (None, None)
        to = float(to) if to else None
        tp = float(tp) if tp else None
        oil_ratio = (mo / to) if (mo is not None and to) else None
        pf_ratio = (mp / tp) if (mp is not None and tp) else None
        rows.append(
            {
                "well": w,
                "pad": wc.pad,
                "pump": (f"{cc[0]}{cc[1]}" if cc else "—"),
                "delivered_pf": per_pad.get(wc.pad),
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
    return rows, per_pad


def summarize_by_pad(results, pad_configs: dict) -> list[dict]:
    """Per-pad roll-up of a joint run: wells, oil, water, delivered PF.

    Pure — the page renders it. Keyed off each well's own ``pad`` so a well that
    was re-padded between review and run lands in the right row.
    """
    by_name = {}
    for pad, wells in pad_configs.items():
        for wc in wells:
            by_name[wc.well_name] = pad

    rows: dict[str, dict] = {}
    for r in results:
        pad = by_name.get(r.well_name, "?")
        row = rows.setdefault(
            pad, {"pad": pad, "wells": 0, "oil_bopd": 0.0, "total_water_bwpd": 0.0}
        )
        row["wells"] += 1
        row["oil_bopd"] += float(r.predicted_oil_rate)
        row["total_water_bwpd"] += float(r.predicted_total_water)
    return [rows[p] for p in sorted(rows)]
