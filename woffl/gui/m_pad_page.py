"""Dedicated M-Pad (Moose Pad) optimization page.

Same 3-stage flow as S-Pad/I-Pad (Review → Configure & Run → Results), reusing the
shared per-well review stage + store + the both-baseline comparator, but with
M-Pad's HYBRID booster physics (see :mod:`woffl.gui.m_pad_plant`):

- Power fluid is delivered by the **HP bank** — up to 3 REDA M675 pumps in
  PARALLEL, fed at the LP-held ~1,400 psig, holding the ~3,500 psig PF header.
- Amps have headroom, so unlike I-Pad the binding limit is **min-flow** (the
  recirculation / high-diff shutdown), not amps. The HP bank holds 3,500 across a
  wide flow band, so the real lever is the **pumps-online count** (fewer pumps →
  lower recirc floor), not pressure. Pressure still floats (capped at 3,500) and
  the amp-limited frontier still bounds the high-flow end.

n_pumps (HP pumps online) threads through the optimizer, comparator, and frontier
plot. Head is field-derated ~0.91 (wear). S-Pad/I-Pad pages are untouched.
"""

import streamlit as st

from woffl.gui import m_pad_plant as plant
from woffl.gui.pad_helpers import parse_pump as _parse_pump
from woffl.gui.pad_helpers import recent_test_rates as _recent_test_rates
from woffl.gui.pad_helpers import (
    render_results_accounting as _render_results_accounting,
)
from woffl.gui.workflow_steps import well_review_store as wrs
from woffl.gui.workflow_steps.step_review_wells import render_review_stage, store_for

PAD = "M"
_STAGES = ["1 · Review Wells", "2 · Configure & Run", "3 · Results"]
_STAGE_KEY = "mp_page_stage"
_MAX_HEADER_PSI = 3500.0  # PF-header discharge cap (PIC-4231 setpoint)


# ---------------------------------------------------------------------------
# Frontier coupling (header pressure as a swept decision var, n-pump aware)
# ---------------------------------------------------------------------------


def _frontier_header(total_pf: float, fallback: float, n_pumps: int):
    """Deliverable header for a total PF at n online HP pumps: the amp/speed
    frontier, capped at the operational discharge limit. (header_psi, over_cap)."""
    if total_pf <= 0:
        return min(_MAX_HEADER_PSI, fallback), False
    fp = plant.max_discharge_pressure(total_pf, n_pumps)
    if fp is None:
        return plant.hp_suction_psi(), True
    return min(_MAX_HEADER_PSI, fp), False


def _eval_cap(n_pumps: int) -> float:
    return plant.max_total_flow(n_pumps)


def _run_frontier_optimization(
    well_configs,
    nozzles,
    throats,
    method,
    marginal_wc,
    n_pumps,
    *,
    n_steps=9,
    progress=None,
):
    """Sweep header pressure; at each, hand the optimizer the PF budget the HP
    bank can deliver there (frontier inverse at n pumps) and let it pick pumps to
    maximize oil. Keep the most-oil pressure. Flags recirc when total PF falls
    below the min-flow floor for the chosen pump count."""
    from woffl.assembly.network_optimizer import NetworkOptimizer, PowerFluidConstraint
    from woffl.assembly.optimization_algorithms import optimize
    from woffl.gui.scotts_tools._common import worker_ceiling

    suction = plant.hp_suction_psi()
    top = plant.max_discharge_pressure(plant.min_total_flow(n_pumps) * 1.2, n_pumps)
    p_ceiling = min(_MAX_HEADER_PSI, top or _MAX_HEADER_PSI)
    p_floor = max(suction + 300.0, 1600.0)
    if p_ceiling <= p_floor:
        p_ceiling = p_floor + 500.0
    pressures = [
        p_floor + (p_ceiling - p_floor) * i / (n_steps - 1) for i in range(n_steps)
    ]

    sweep, best = [], None
    for idx, P in enumerate(pressures):
        cap = plant.max_flow_at_pressure(P, n_pumps)
        if not cap or cap <= 0:
            if progress:
                progress(idx + 1, n_steps, P, 0.0, 0.0)
            continue
        for wc in well_configs:
            wc.ppf_surf_well = P
        pf = PowerFluidConstraint(total_rate=cap, pressure=P, rho_pf=62.4)
        opt = NetworkOptimizer(well_configs, pf, nozzles, throats, marginal_wc)
        opt.run_all_batch_simulations(max_workers=worker_ceiling())
        results = optimize(opt, method=method, water_key="lift_wat")
        total_pf = sum(r.predicted_lift_water for r in results)
        total_oil = sum(r.predicted_oil_rate for r in results)
        rec = {
            "P": P,
            "cap": cap,
            "total_pf": total_pf,
            "total_oil": total_oil,
            "results": results,
            "opt": opt,
        }
        sweep.append(rec)
        if best is None or total_oil > best["total_oil"]:
            best = rec
        if progress:
            progress(idx + 1, n_steps, P, total_pf, total_oil)

    if best is None:
        raise RuntimeError(
            "No feasible header pressure found — the HP bank couldn't deliver any "
            "well's power-fluid demand. Check the IPRs and pump count."
        )

    env = plant.operating_envelope(
        [best["total_pf"]], n_pumps, header_cap=_MAX_HEADER_PSI
    )[0]
    min_flow = plant.min_total_flow(n_pumps)
    meta = {
        "header_psi": best["P"],
        "total_pf_bpd": best["total_pf"],
        "total_oil_bopd": best["total_oil"],
        "frontier_cap_bpd": best["cap"],
        "n_pumps": n_pumps,
        "suction_psi": suction,
        "min_total_flow": min_flow,
        "recirc": best["total_pf"] < min_flow,
        "pumps": env["pumps"],
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
    }
    # Per-well drop accounting at the winning pressure (failed sim vs solver
    # shut-in vs marginal-WC exclusion) — Results renders the real reasons.
    from woffl.assembly.network_optimizer import reconcile_wells

    meta["reconciliation"] = reconcile_wells(best["opt"], best["results"])
    return best["results"], meta


def _match_check_mpad(well_configs, current_choices, test_rates, n_pumps):
    """Pre-flight: model each well at its current pump + chosen IPR vs recent
    tests. Header from the M-Pad frontier at current total PF. (rows, header)."""
    from woffl.assembly.network_optimizer import NetworkOptimizer, PowerFluidConstraint
    from woffl.gui.scotts_tools._common import worker_ceiling

    names = [wc.well_name for wc in well_configs]
    cur_oil = {w: float((test_rates.get(w) or (0, 0))[0] or 0.0) for w in names}
    cur_pf = {w: float((test_rates.get(w) or (0, 0))[1] or 0.0) for w in names}
    total_pf = sum(cur_pf.values())
    header, _ = _frontier_header(total_pf, _MAX_HEADER_PSI, n_pumps)
    pumps = [c for c in current_choices.values() if c]
    nozzles = sorted({c[0] for c in pumps}) or ["12"]
    throats = sorted({c[1] for c in pumps}) or ["B"]
    for wc in well_configs:
        wc.ppf_surf_well = header
    pf = PowerFluidConstraint(
        total_rate=max(total_pf * 1.5, 80000.0), pressure=header, rho_pf=62.4
    )
    opt = NetworkOptimizer(well_configs, pf, nozzles, throats, marginal_watercut=1.0)
    opt.run_all_batch_simulations(max_workers=worker_ceiling())

    def flag(ratio):
        if ratio is None:
            return "— no data"
        if 0.80 <= ratio <= 1.25:
            return "✓ match"
        if 0.50 <= ratio <= 2.0:
            return "⚠ off"
        return "✗ BUST"

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
                "oil_flag": flag(oil_ratio),
                "test_pf": tp,
                "model_pf": mp,
                "pf_ratio": pf_ratio,
                "pf_flag": flag(pf_ratio),
            }
        )
    return rows, header


def _evaluate_fixed_scenario(
    well_configs,
    choices,
    n_pumps,
    *,
    fallback_choices=None,
    test_rates=None,
    current_choices=None,
    max_iter=8,
    tol_psi=10.0,
    relax=0.6,
    progress=None,
):
    """Evaluate FIXED per-well pumps against the M-Pad frontier (fixed point)."""
    from woffl.assembly.network_optimizer import NetworkOptimizer, PowerFluidConstraint
    from woffl.gui.scotts_tools._common import worker_ceiling

    all_ch = list(choices.values()) + list((fallback_choices or {}).values())
    nozzles = sorted({c[0] for c in all_ch if c}) or ["12"]
    throats = sorted({c[1] for c in all_ch if c}) or ["B"]
    cap = _eval_cap(n_pumps)
    ppf = _MAX_HEADER_PSI
    history, per_well, total_pf, total_oil, over_cap = [], [], 0.0, 0.0, False
    converged = False
    lo_p = max(plant.hp_suction_psi(), 1000.0)  # PowerFluidConstraint floor

    for it in range(max_iter):
        ppf_c = max(lo_p, min(_MAX_HEADER_PSI, ppf))
        for wc in well_configs:
            wc.ppf_surf_well = ppf_c
        pf = PowerFluidConstraint(total_rate=cap, pressure=ppf_c, rho_pf=62.4)
        opt = NetworkOptimizer(
            well_configs, pf, nozzles, throats, marginal_watercut=1.0
        )
        opt.run_all_batch_simulations(max_workers=worker_ceiling())

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
                fb = (fallback_choices or {}).get(wc.well_name)
                if fb:
                    perf = opt.get_pump_performance(wc.well_name, fb[0], fb[1])
                    if perf is not None:
                        note, ch = f"{ch[0]}{ch[1]}✗→{fb[0]}{fb[1]}", fb
                if perf is None:
                    bp = opt.batch_results.get(wc.well_name)
                    df = getattr(bp, "df", None) if bp is not None else None
                    if df is not None:
                        feas = df[~df["qoil_std"].isna()]
                        if not feas.empty:
                            r = feas.loc[feas["qoil_std"].idxmax()]
                            fbn, fbt = str(r["nozzle"]), str(r["throat"])
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

        new_ppf, over_cap = _frontier_header(total_pf, ppf_c, n_pumps)
        history.append(
            {
                "iter": it + 1,
                "trial_psi": round(ppf_c, 1),
                "total_pf_bpd": round(total_pf, 0),
                "frontier_psi": round(new_ppf, 1),
            }
        )
        if progress:
            progress(it + 1, max_iter, ppf_c, total_pf, new_ppf)
        if abs(new_ppf - ppf_c) <= tol_psi:
            ppf = new_ppf
            converged = True
            break
        ppf = relax * new_ppf + (1 - relax) * ppf_c

    if test_rates:
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
        avg_pf = (
            min(1.2, max(0.3, sum(pf_ratios) / len(pf_ratios))) if pf_ratios else 1.0
        )
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
        if rescaled:
            total_oil = sum(r["oil"] for r in per_well)
            total_pf = sum(r["pf"] for r in per_well)
            ppf, over_cap = _frontier_header(total_pf, ppf, n_pumps)

    meta = {
        "header_psi": max(lo_p, min(_MAX_HEADER_PSI, ppf)),
        "total_pf_bpd": total_pf,
        "total_oil_bopd": total_oil,
        "over_capacity": over_cap,
        "n_pumps": n_pumps,
        "recirc": total_pf < plant.min_total_flow(n_pumps),
        "converged": converged,
        "history": history,
    }
    return per_well, meta


def _evaluate_existing_scenario(
    well_configs,
    scenario_choices,
    current_choices,
    n_pumps,
    *,
    test_rates,
    max_iter=8,
    tol_psi=10.0,
    relax=0.6,
    progress=None,
):
    """Existing-baseline scenario, anchored to measured rates × the model's
    relative change (bias cancels). Non-solvers use the unchanged-solvers' ripple."""
    from woffl.assembly.network_optimizer import NetworkOptimizer, PowerFluidConstraint
    from woffl.gui.scotts_tools._common import worker_ceiling

    names = [wc.well_name for wc in well_configs]
    cur_oil = {w: float((test_rates.get(w) or (0, 0))[0] or 0.0) for w in names}
    cur_pf = {w: float((test_rates.get(w) or (0, 0))[1] or 0.0) for w in names}
    all_ch = list(scenario_choices.values()) + list(current_choices.values())
    nozzles = sorted({c[0] for c in all_ch if c}) or ["12"]
    throats = sorted({c[1] for c in all_ch if c}) or ["B"]
    cap = _eval_cap(n_pumps)
    lo_p = plant.hp_suction_psi()
    total_pf_base = sum(cur_pf.values())
    header_base, _ = _frontier_header(total_pf_base, _MAX_HEADER_PSI, n_pumps)

    def _run(ppf):
        ppf_c = max(lo_p, min(_MAX_HEADER_PSI, ppf))
        for wc in well_configs:
            wc.ppf_surf_well = ppf_c
        pf = PowerFluidConstraint(total_rate=cap, pressure=ppf_c, rho_pf=62.4)
        opt = NetworkOptimizer(
            well_configs, pf, nozzles, throats, marginal_watercut=1.0
        )
        opt.run_all_batch_simulations(max_workers=worker_ceiling())
        return opt

    opt_base = _run(header_base)
    mc = {}
    for w in names:
        cc = current_choices.get(w)
        perf = opt_base.get_pump_performance(w, cc[0], cc[1]) if cc else None
        mc[w] = (perf["oil_rate"], perf["lift_water"]) if perf else None

    ppf, per_well = header_base, []
    converged = False
    for it in range(max_iter):
        ppf_c = max(lo_p, min(_MAX_HEADER_PSI, ppf))
        opt = _run(ppf_c)
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
                    {
                        "well": w,
                        "pump": f"{ch[0]}{ch[1]}",
                        "oil": so,
                        "pf": sp,
                        "note": "",
                    }
                )
                total_pf += sp
            elif ms:
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
        new_ppf, _ = _frontier_header(total_pf, ppf_c, n_pumps)
        if progress:
            progress(it + 1, max_iter, ppf_c, total_pf, new_ppf)
        if abs(new_ppf - ppf_c) <= tol_psi:
            ppf = new_ppf
            converged = True
            break
        ppf = relax * new_ppf + (1 - relax) * ppf_c

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
            bp = opt.batch_results.get(w)
            df = getattr(bp, "df", None) if bp is not None else None
            if df is not None:
                feas = df[~df["qoil_std"].isna()]
                if not feas.empty:
                    br = feas.loc[feas["qoil_std"].idxmax()]
                    perf = opt.get_pump_performance(
                        w, str(br["nozzle"]), str(br["throat"])
                    )
                    if perf:
                        r["oil"], r["pf"] = float(perf["oil_rate"]), float(
                            perf["lift_water"]
                        )
                        r["pump"] = f"{r['pump']} (est {br['nozzle']}{br['throat']})"

    for r in per_well:
        mcw, co = mc.get(r["well"]), cur_oil.get(r["well"], 0.0)
        r["bias"] = (mcw[0] / co) if (mcw and mcw[0] and co > 0) else None

    total_oil = sum(r["oil"] for r in per_well)
    total_pf = sum(r["pf"] for r in per_well)
    header, over_cap = _frontier_header(total_pf, ppf, n_pumps)
    scn_meta = {
        "header_psi": max(lo_p, min(_MAX_HEADER_PSI, header)),
        "total_pf_bpd": total_pf,
        "total_oil_bopd": total_oil,
        "over_capacity": over_cap,
        "n_pumps": n_pumps,
        "recirc": total_pf < plant.min_total_flow(n_pumps),
        "converged": converged,
        "history": [],
    }
    return per_well, scn_meta


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def _render_review() -> None:
    render_review_stage(PAD)
    store = store_for(PAD)
    active = wrs.active_entries(store)
    st.divider()
    if active:
        n_off = len(store) - len(active)
        st.success(
            f"{len(active)} well(s) ready for optimization"
            + (f"  ·  {n_off} offline (excluded)" if n_off else "")
        )
        if st.button("Next: Configure & Run →", type="primary", key="mp_to_config"):
            st.session_state[_STAGE_KEY] = 1
            st.rerun()
    elif store:
        st.warning(
            "Every reviewed well is marked offline — bring at least one online to continue."
        )
    else:
        st.info("Review at least one well (or add a hypothetical) to continue.")


def _render_configure() -> None:
    from woffl.gui.params import NOZZLE_OPTIONS, THROAT_OPTIONS

    store = store_for(PAD)
    active = wrs.active_entries(store)
    if not active:
        st.warning("No active reviewed wells. Go back to **Review Wells**.")
        return

    offline = [w for w in store if store[w].get("offline")]
    st.markdown(
        f"**{len(active)} active well(s)** to optimize: " + ", ".join(active.keys())
    )
    if offline:
        st.caption("Offline (excluded): " + ", ".join(offline))
    st.caption(
        "M-Pad's HP bank (parallel REDA M675, VFD) holds the ~3,500 psi PF header "
        "with amp headroom, so the real limit is **min-flow** (recirc) — the lever "
        "is how many HP pumps are online. Pressure floats (capped 3,500); the "
        "optimizer keeps the most-oil point above the min-flow floor."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        n_pumps = st.radio(
            "HP pumps online",
            [3, 2, 1],
            horizontal=True,
            key="mp_n_pumps",
            help="3 normally; fewer drops the min-flow floor for low PF demand.",
        )
    with c2:
        method = st.selectbox("Optimizer", ["milp", "mckp"], key="mp_method")
    with c3:
        marginal_wc = st.number_input(
            "Marginal water cut",
            0.0,
            1.0,
            1.0,
            0.01,
            format="%.2f",
            key="mp_marginal_wc",
        )

    c4, c5 = st.columns(2)
    with c4:
        nozzles = st.multiselect(
            "Nozzle sizes to test",
            NOZZLE_OPTIONS,
            default=["9", "10", "11", "12", "13", "14", "15"],
            key="mp_nozzles",
        )
    with c5:
        throats = st.multiselect(
            "Throat ratios to test",
            THROAT_OPTIONS,
            default=["A", "B", "C", "D"],
            key="mp_throats",
        )

    mn, mx = plant.min_total_flow(n_pumps), plant.max_total_flow(n_pumps)
    st.caption(
        f"With **{n_pumps} HP pump(s)**: min-flow (recirc) floor "
        f"**{mn:,.0f} BPD**, off-curve ceiling {mx:,.0f} BPD total PF."
    )

    st.markdown("##### HP-bank capability frontier")
    last_meta = st.session_state.get("mp_run_meta")
    op = (
        [
            {
                "label": "Last run",
                "total_pf_bpd": last_meta["total_pf_bpd"],
                "header_psi": last_meta["header_psi"],
                "color": "#d62728",
            }
        ]
        if last_meta and last_meta.get("n_pumps") == n_pumps
        else None
    )
    if op:
        st.caption("✕ marks the operating point from the most recent run.")
    _render_frontier_plot(n_pumps, op_points=op)

    st.divider()
    st.markdown("##### Pre-flight: model match check")
    st.caption(
        "Model each well at its current pump + chosen IPR vs recent tests. Fix ✗ first."
    )
    current = {
        w: (active[w].get("review_nozzle") or "", active[w].get("review_throat") or "")
        for w in active
    }
    sig = (
        n_pumps,
        tuple(
            sorted(
                (
                    w,
                    current[w][0],
                    current[w][1],
                    int(active[w].get("qwf", 0)),
                    int(active[w].get("pwf", 0)),
                    int(active[w].get("res_pres", 0)),
                )
                for w in active
            )
        ),
    )
    mc = st.session_state.get("mp_matchcheck")
    refresh = st.button("↻ Re-run check", key="mp_matchcheck_run")
    if refresh or not mc or mc.get("sig") != sig:
        cur_ch = {
            w: (current[w] if current[w][0] and current[w][1] else None) for w in active
        }
        tr = {w: _recent_test_rates(w) for w in active}
        try:
            with st.spinner("Modeling each well at its current pump…"):
                rows, hdr = _match_check_mpad(
                    wrs.store_to_well_configs(active), cur_ch, tr, n_pumps
                )
            mc = {"rows": rows, "header": hdr, "sig": sig}
            st.session_state["mp_matchcheck"] = mc
        except Exception as e:
            st.caption(f"Match check unavailable: {e}")
            mc = None

    if mc:
        import pandas as pd

        rows = mc["rows"]
        n_oil = sum(1 for r in rows if r["oil_flag"].startswith("✗"))
        n_pf = sum(1 for r in rows if r["pf_flag"].startswith("✗"))
        st.caption(
            f"Modeled at header ≈ {mc['header']:,.0f} psi.  "
            f"**{n_oil} oil mismatch · {n_pf} PF bust** (✗). Worst first."
        )

        def _dev(r):
            ds = [abs(x - 1) for x in (r["oil_ratio"], r["pf_ratio"]) if x is not None]
            return max(ds) if ds else 0.0

        df = pd.DataFrame(
            [
                {
                    "Well": r["well"],
                    "Pump": r["pump"],
                    "Test oil": round(r["test_oil"]) if r["test_oil"] else None,
                    "Model oil": (
                        round(r["model_oil"]) if r["model_oil"] is not None else None
                    ),
                    "Oil×": round(r["oil_ratio"], 2) if r["oil_ratio"] else None,
                    "Oil match": r["oil_flag"],
                    "Test PF": round(r["test_pf"]) if r["test_pf"] else None,
                    "Model PF": (
                        round(r["model_pf"]) if r["model_pf"] is not None else None
                    ),
                    "PF×": round(r["pf_ratio"], 2) if r["pf_ratio"] else None,
                    "PF match": r["pf_flag"],
                }
                for r in sorted(rows, key=_dev, reverse=True)
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
    st.divider()

    if not nozzles or not throats:
        st.warning("Pick at least one nozzle and one throat.")
        return

    if st.button("▶ Run optimization", type="primary", key="mp_run"):
        well_configs = wrs.store_to_well_configs(active)
        bar = st.progress(0.0, text="Optimizing the HP bank…")

        def _progress(it, max_it, P, total_pf, total_oil):
            bar.progress(
                min(it / max_it, 1.0),
                text=f"Step {it}/{max_it}: header {P:,.0f} psi → "
                f"{total_pf:,.0f} BPD PF → {total_oil:,.0f} BOPD",
            )

        try:
            with st.spinner("Running batch sweeps across the pressure frontier…"):
                results, meta = _run_frontier_optimization(
                    well_configs,
                    nozzles,
                    throats,
                    method,
                    marginal_wc,
                    n_pumps,
                    progress=_progress,
                )
        except Exception as e:
            bar.empty()
            st.error(f"Optimization failed: {e}")
            return
        bar.empty()
        meta["store_sig"] = wrs.store_signature(active)
        st.session_state["mp_opt_results"] = results
        st.session_state["mp_run_meta"] = meta
        # A fresh optimization invalidates any cached scenario (S-Pad already
        # did this; the copy-adaptation dropped it here).
        for k in ("mp_scenario_meta", "mp_scenario_per_well", "mp_scn_base_used"):
            st.session_state.pop(k, None)
        st.session_state[_STAGE_KEY] = 2
        st.rerun()


def _render_results() -> None:
    import pandas as pd

    results = st.session_state.get("mp_opt_results")
    meta = st.session_state.get("mp_run_meta")
    if not results or not meta:
        st.warning("No results yet. Go to **Configure & Run**.")
        return

    store = store_for(PAD)
    active = wrs.active_entries(store)
    n_pumps = meta["n_pumps"]
    opt_choice = {
        r.well_name: (r.recommended_nozzle, r.recommended_throat) for r in results
    }
    si_wells = [w for w in active if w not in opt_choice]
    _render_results_accounting(meta, active, si_wells)

    st.markdown("#### Station operating point")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Header pressure", f"{meta['header_psi']:,.0f} psi")
    m2.metric("Total power fluid", f"{meta['total_pf_bpd']:,.0f} BPD")
    m3.metric("HP pumps online", f"{n_pumps}")
    m4.metric("Total oil", f"{meta['total_oil_bopd']:,.0f} BOPD")

    if meta.get("recirc"):
        st.warning(
            f"Total PF {meta['total_pf_bpd']:,.0f} BPD is BELOW the {n_pumps}-pump "
            f"min-flow floor ({meta['min_total_flow']:,.0f} BPD) — the HP pumps would "
            "be in recirculation / low-flow shutdown. **Drop to fewer HP pumps** to "
            "lower the floor, or this demand can't run as configured."
        )

    st.markdown("##### HP pump speed & amps")
    pump_rows = []
    for p in meta["pumps"]:
        if p.get("hz") is None:
            pump_rows.append(
                {
                    "Pump": p["name"],
                    "Online": p.get("n", n_pumps),
                    "Speed": "—",
                    "Amps/pump": "—",
                    "Limit (A)": round(p["amp_limit"]),
                }
            )
            continue
        head = p["amp_limit"] - p["amps"]
        pump_rows.append(
            {
                "Pump": p["name"],
                "Online": p.get("n", n_pumps),
                "Speed": f"{p['hz']:.1f} Hz",
                "Amps/pump": round(p["amps"], 1),
                "Limit (A)": round(p["amp_limit"]),
                "Headroom": f"{head:.0f} A ({head / p['amp_limit'] * 100:.0f}%)",
            }
        )
    st.dataframe(pd.DataFrame(pump_rows), use_container_width=True, hide_index=True)

    st.markdown("#### Per-well recommendations")
    rows = []
    for r in results:
        rows.append(
            {
                "Well": r.well_name,
                "Nozzle": r.recommended_nozzle,
                "Throat": r.recommended_throat,
                "Oil (BOPD)": round(r.predicted_oil_rate, 0),
                "Power fluid (BPD)": round(r.predicted_lift_water, 0),
                "Form. water (BPD)": round(r.predicted_formation_water, 0),
                "Total WC": round(r.total_watercut, 3),
                "Suction (psi)": round(r.suction_pressure, 0),
                "Status": "⚠ sonic" if r.sonic_status else "run",
            }
        )
    for w in si_wells:
        rows.append(
            {
                "Well": w,
                "Nozzle": "—",
                "Throat": "—",
                "Oil (BOPD)": 0,
                "Power fluid (BPD)": 0,
                "Form. water (BPD)": 0,
                "Total WC": None,
                "Suction (psi)": None,
                "Status": "SHUT IN",
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    if st.button("⬇ Download results CSV", key="mp_res_dl"):
        from woffl.gui.components.download import autodownload

        autodownload(
            df.to_csv(index=False).encode("utf-8"),
            "M_pad_optimization_results.csv",
            "text/csv",
            "mp_res_auto_dl",
        )

    st.markdown("#### HP-bank frontier")
    _render_frontier_plot(
        n_pumps,
        op_points=[
            {
                "label": "Optimized",
                "total_pf_bpd": meta["total_pf_bpd"],
                "header_psi": meta["header_psi"],
                "color": "#d62728",
            }
        ],
    )
    with st.expander("Pressure sweep explored"):
        st.dataframe(
            pd.DataFrame(meta["sweep"]), use_container_width=True, hide_index=True
        )

    _render_scenario_comparator(results, meta, active, n_pumps)


def _render_scenario_comparator(results, meta, active, n_pumps) -> None:
    import pandas as pd

    opt_choice = {
        r.well_name: (r.recommended_nozzle, r.recommended_throat) for r in results
    }
    opt_oil = {r.well_name: r.predicted_oil_rate for r in results}

    st.divider()
    st.markdown("#### Compare a manual scenario")
    st.caption(
        "Pick a pump (or shut-in) per well and see the resulting oil + header "
        "pressure — coupled to the HP-bank frontier — next to a baseline."
    )

    base_choice = st.radio(
        "Compare against",
        ["Existing (recent test rates)", "Optimized"],
        horizontal=True,
        key="mp_cmp_base",
        help="Existing = each well's CURRENT measured oil/PF from recent tests. "
        "Optimized = the optimizer's picks.",
    )
    is_existing = base_choice.startswith("Existing")
    if st.session_state.get("mp_scn_base_used") not in (None, base_choice):
        st.session_state.pop("mp_scenario_meta", None)
        st.session_state.pop("mp_scenario_per_well", None)
    st.session_state["mp_scn_base_used"] = base_choice

    combos = [
        f"{n}{t}" for n in meta.get("nozzles", []) for t in meta.get("throats", [])
    ]
    current = {
        w: (active[w].get("review_nozzle") or "", active[w].get("review_throat") or "")
        for w in active
    }
    cur_labels = [f"{n}{t}" for (n, t) in current.values() if n and t]
    pump_options = sorted(set(combos + cur_labels)) + ["Shut in"]

    preset = st.radio(
        "Scenario",
        ["All current pumps", "All optimized (baseline)", "Custom per-well"],
        horizontal=True,
        key="mp_scn_preset",
    )
    if preset == "Custom per-well":
        basis = st.session_state.get("mp_scn_basis", "optimized")
        nonce = st.session_state.get("mp_scn_nonce", 0)
        bc = st.columns([1.4, 1.4, 3])
        if bc[0].button("⤓ Fill all from latest installed", key="mp_scn_fill_cur"):
            st.session_state["mp_scn_basis"] = "current"
            st.session_state["mp_scn_nonce"] = nonce + 1
            st.rerun()
        if bc[1].button("⤓ Fill all from optimized", key="mp_scn_fill_opt"):
            st.session_state["mp_scn_basis"] = "optimized"
            st.session_state["mp_scn_nonce"] = nonce + 1
            st.rerun()
        bc[2].caption(
            f"Filled from **{'latest installed' if basis == 'current' else 'optimized'}** "
            "— spot-edit any well below."
        )
        with st.expander("Per-well pump selection", expanded=True):
            cols = st.columns(3)
            for i, w in enumerate(sorted(active)):
                src = current.get(w) if basis == "current" else opt_choice.get(w)
                d_lbl = (
                    f"{src[0]}{src[1]}" if (src and src[0] and src[1]) else "Shut in"
                )
                if d_lbl not in pump_options:
                    d_lbl = pump_options[0]
                with cols[i % 3]:
                    st.selectbox(
                        w,
                        pump_options,
                        index=pump_options.index(d_lbl),
                        key=f"mp_scn_{w}_{nonce}",
                    )

    if st.button("▶ Compute scenario & compare", type="primary", key="mp_scn_run"):
        choices = {}
        for w in active:
            if preset == "All current pumps":
                nt = current.get(w)
                choices[w] = nt if (nt and nt[0] and nt[1]) else None
            elif preset == "All optimized (baseline)":
                choices[w] = opt_choice.get(w)
            else:
                _nonce = st.session_state.get("mp_scn_nonce", 0)
                choices[w] = _parse_pump(st.session_state.get(f"mp_scn_{w}_{_nonce}"))
        well_configs = wrs.store_to_well_configs(active)
        bar = st.progress(0.0, text="Solving scenario…")

        def _p(it, mx, ppf, tpf, _npf):
            bar.progress(
                min(it / mx, 1.0),
                text=f"Iter {it}: header {ppf:,.0f} psi → {tpf:,.0f} BPD",
            )

        try:
            with st.spinner("Running scenario (coupled to the frontier)…"):
                if is_existing:
                    cur_ch = {
                        w: (current[w] if current[w][0] and current[w][1] else None)
                        for w in active
                    }
                    tr = {w: _recent_test_rates(w) for w in active}
                    per_well, scn_meta = _evaluate_existing_scenario(
                        well_configs,
                        choices,
                        cur_ch,
                        n_pumps,
                        test_rates=tr,
                        progress=_p,
                    )
                else:
                    per_well, scn_meta = _evaluate_fixed_scenario(
                        well_configs,
                        choices,
                        n_pumps,
                        fallback_choices=dict(opt_choice),
                        progress=_p,
                    )
        except Exception as e:
            bar.empty()
            st.error(f"Scenario failed: {e}")
            return
        bar.empty()
        st.session_state["mp_scenario_per_well"] = per_well
        st.session_state["mp_scenario_meta"] = scn_meta
        st.rerun()

    base_label = "Optimized"
    base_oil = dict(opt_oil)
    base_pf_map = None
    base_pump = {
        w: (f"{opt_choice[w][0]}{opt_choice[w][1]}" if opt_choice.get(w) else "SHUT IN")
        for w in active
    }
    base_meta = {
        "total_oil_bopd": meta["total_oil_bopd"],
        "total_pf_bpd": meta["total_pf_bpd"],
        "header_psi": meta["header_psi"],
    }
    if is_existing:
        base_label = "Current"
        rates = {w: _recent_test_rates(w) for w in active}
        base_oil = {w: (rates[w][0] or 0.0) for w in active}
        base_pf_map = {w: (rates[w][1] or 0.0) for w in active}
        base_pump = {
            w: (
                f"{current[w][0]}{current[w][1]}"
                if current[w][0] and current[w][1]
                else "—"
            )
            for w in active
        }
        tot_pf = sum(base_pf_map.values())
        hdr, _ = _frontier_header(tot_pf, 0.0, n_pumps)
        base_meta = {
            "total_oil_bopd": sum(base_oil.values()),
            "total_pf_bpd": tot_pf,
            "header_psi": hdr,
        }

    scn = st.session_state.get("mp_scenario_meta")
    scn_pw = st.session_state.get("mp_scenario_per_well")
    if scn and scn_pw:
        st.markdown(f"##### {base_label} vs your scenario")
        cmp = pd.DataFrame(
            {
                "Metric": [
                    "Total oil (BOPD)",
                    "Total PF (BPD)",
                    "Header PF pressure (psi)",
                ],
                base_label: [
                    base_meta["total_oil_bopd"],
                    base_meta["total_pf_bpd"],
                    base_meta["header_psi"],
                ],
                "Your scenario": [
                    scn["total_oil_bopd"],
                    scn["total_pf_bpd"],
                    scn["header_psi"],
                ],
            }
        )
        cmp["Δ (scn − base)"] = cmp["Your scenario"] - cmp[base_label]
        for c in (base_label, "Your scenario", "Δ (scn − base)"):
            cmp[c] = cmp[c].round(0)
        st.dataframe(cmp, use_container_width=True, hide_index=True)

        d_oil = scn["total_oil_bopd"] - base_meta["total_oil_bopd"]
        if d_oil > 0:
            st.caption(
                f"Net: your scenario makes **{d_oil:,.0f} BOPD more** than {base_label.lower()}."
            )
        elif d_oil < 0:
            st.caption(
                f"Net: your scenario makes **{abs(d_oil):,.0f} BOPD less** than {base_label.lower()}."
            )
        if scn.get("recirc"):
            st.warning(
                f"Scenario total PF is below the {n_pumps}-pump min-flow floor — "
                "recirc risk. Drop an HP pump or raise demand."
            )
        if scn.get("over_capacity"):
            st.warning(
                "Scenario draws more PF than the HP bank can push at 3,500 — size some pumps down."
            )
        if scn.get("converged") is False:
            st.warning(
                "Scenario frontier coupling did NOT converge in the iteration "
                "cap — the totals and header pressure are approximate; treat "
                "small deltas with caution."
            )

        starred = [r["well"] for r in scn_pw if r.get("note") == "star"]
        infeas = [r["well"] for r in scn_pw if r.get("note") == "infeasible"]
        subs = [
            r
            for r in scn_pw
            if r.get("note") and r["note"] not in ("infeasible", "star")
        ]
        if subs:
            st.caption(
                "⚠ Chosen pump infeasible — substituted: "
                + ", ".join(f"{r['well']} ({r['pump']})" for r in subs)
            )
        if starred:
            st.caption(
                "★ Model couldn't solve — estimated from recent-test rate × ripple: "
                + ", ".join(starred)
            )
        if infeas:
            st.caption("✗ No feasible pump (counted as 0): " + ", ".join(infeas))

        st.markdown("**Where each option sits on the frontier**")
        _render_frontier_plot(
            n_pumps,
            op_points=[
                {
                    "label": base_label,
                    "total_pf_bpd": base_meta["total_pf_bpd"],
                    "header_psi": base_meta["header_psi"],
                    "color": "#2ca02c",
                },
                {
                    "label": "Your scenario",
                    "total_pf_bpd": scn["total_pf_bpd"],
                    "header_psi": scn["header_psi"],
                    "color": "#d62728",
                },
            ],
        )

        bias_note = (
            " **Model÷Test** = model ÷ measured (>1 = loose IPR match)."
            if is_existing
            else ""
        )
        st.caption(
            f"**Δ oil** = per-well change ({base_label.lower()} → scenario). "
            "★ = estimated. Sorted by biggest mover." + bias_note
        )
        scn_map = {r["well"]: r for r in scn_pw}
        mrows = []
        for w in sorted(active):
            srow = scn_map.get(w, {})
            b_oil = round(base_oil.get(w, 0))
            s_oil = round(srow.get("oil", 0))
            row = {
                "Well": w,
                f"{base_label} pump": base_pump.get(w, "—"),
                f"{base_label} oil": b_oil,
            }
            if base_pf_map is not None:
                row[f"{base_label} PF"] = round(base_pf_map.get(w, 0))
            row.update(
                {
                    "Scenario": srow.get("pump", "—"),
                    "Scn oil": s_oil,
                    "Δ oil": s_oil - b_oil,
                    "Scn PF": round(srow.get("pf", 0)),
                }
            )
            if base_pf_map is not None:
                bias = srow.get("bias")
                row["Model÷Test"] = round(bias, 2) if bias else None
            mrows.append(row)
        mdf = pd.DataFrame(mrows).sort_values(
            "Δ oil", key=lambda s: s.abs(), ascending=False
        )
        st.dataframe(mdf, use_container_width=True, hide_index=True)


def _render_frontier_plot(n_pumps, op_points=None) -> None:
    """HP-bank frontier: max deliverable header vs total PF (capped at 3,500),
    with the min-flow recirc region shaded and optional operating-point markers."""
    import plotly.graph_objects as go

    mn = plant.min_total_flow(n_pumps)
    mx = plant.max_total_flow(n_pumps)
    xs, ys = [], []
    q = max(mn * 0.5, 2000.0)
    step = max((mx - q) / 60.0, 1000.0)
    while q <= mx:
        p = plant.max_discharge_pressure(q, n_pumps)
        if p is not None:
            xs.append(q)
            ys.append(min(_MAX_HEADER_PSI, p))
        q += step

    fig = go.Figure()
    if xs:
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name=f"{n_pumps}-pump frontier (capped 3,500)",
                line=dict(color="#1f77b4"),
                hovertemplate="%{x:.0f} BPD<br>%{y:.0f} psi<extra></extra>",
            )
        )
    fig.add_vrect(
        x0=(xs[0] if xs else 0),
        x1=mn,
        fillcolor="red",
        opacity=0.08,
        line_width=0,
        annotation_text="recirc / min-flow",
        annotation_position="top left",
    )
    fig.add_vline(
        x=mn,
        line=dict(color="#d62728", width=1.5, dash="dash"),
        annotation_text=f"min-flow  {mn:,.0f} BPD",
        annotation_position="top right",
    )
    for p in op_points or []:
        fig.add_trace(
            go.Scatter(
                x=[p["total_pf_bpd"]],
                y=[p["header_psi"]],
                mode="markers+text",
                marker=dict(size=14, color=p.get("color", "#d62728"), symbol="x"),
                name=p["label"],
                text=[
                    f"  {p['label']}: {p['header_psi']:.0f} psi @ {p['total_pf_bpd']:,.0f} BPD"
                ],
                textposition="middle right",
                hovertemplate="%{x:.0f} BPD<br>%{y:.0f} psi<extra></extra>",
            )
        )
    fig.update_layout(
        xaxis_title="Total power-fluid flow (BPD)",
        yaxis_title="Header pressure (psi)",
        height=380,
        margin=dict(t=20, b=40),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_m_pad_page() -> None:
    st.title("M-Pad Optimization 🛢")
    st.caption(
        "Review each Moose Pad well, then optimize the pad against the HP "
        "booster bank (parallel, VFD, min-flow-limited)."
    )

    stage = st.session_state.setdefault(_STAGE_KEY, 0)
    have_results = bool(st.session_state.get("mp_opt_results"))
    have_wells = bool(wrs.active_entries(store_for(PAD)))
    cols = st.columns(3)
    for i, label in enumerate(_STAGES):
        unlocked = i == 0 or (i == 1 and have_wells) or (i == 2 and have_results)
        with cols[i]:
            if i == stage:
                st.markdown(f"**:blue[{label}]**")
            elif unlocked:
                if st.button(label, key=f"mp_nav_{i}", use_container_width=True):
                    st.session_state[_STAGE_KEY] = i
                    st.rerun()
            else:
                st.markdown(f":gray[{label}]")
    st.divider()

    if stage == 0:
        _render_review()
    elif stage == 1:
        _render_configure()
    else:
        _render_results()
