"""Dedicated S-Pad optimization page.

A self-contained flow for S-Pad, reached from its own top-level mode:

    Review Wells  →  Configure & Run  →  Results

- **Review Wells** reuses the per-well Solver review stage (``step_review_wells``):
  the engineer opens each S-Pad well, confirms/adjusts the match, and saves it
  into the per-pad store. Future wells can be added as hypotheticals.
- **Configure & Run** optimizes nozzle/throat across the reviewed wells, coupled
  to the 3-pump booster curve (``s_pad_plant``): the delivered header pressure
  every well sees depends on the *total* power-fluid flow — more PF → lower
  pressure. Solved to a fixed point.
- **Results** shows the station operating point (delivered pressure, per-pump
  flow, thrust-range check) and the per-well pump recommendations.

Reuses the ``NetworkOptimizer`` engine; the curve coupling lives here (GUI), so
no upstream library change.
"""

import streamlit as st

from woffl.gui import s_pad_plant
from woffl.gui.pad_helpers import parse_pump as _parse_pump
from woffl.gui.pad_helpers import recent_test_rates as _recent_test_rates
from woffl.gui.pad_helpers import (
    render_results_accounting as _render_results_accounting,
)
from woffl.gui.workflow_steps import well_review_store as wrs
from woffl.gui.workflow_steps.step_review_wells import render_review_stage, store_for

PAD = "S"
_STAGES = ["1 · Review Wells", "2 · Configure & Run", "3 · Results"]
_STAGE_KEY = "sp_page_stage"


# ---------------------------------------------------------------------------
# Coupled optimization (pump-curve fixed point)
# ---------------------------------------------------------------------------


def _run_coupled_optimization(
    well_configs,
    n_pumps,
    nozzles,
    throats,
    method,
    marginal_wc,
    *,
    max_iter=8,
    tol_psi=10.0,
    relax=0.6,
    progress=None,
):
    """Optimize nozzle/throat across the pad, coupling delivered PF pressure to
    total PF flow via the booster curve.

    The optimizer is solved at a trial header pressure; the resulting total PF
    flow sets a new header pressure off the pump curve; we damp-iterate to a
    fixed point. Returns (results, optimizer, meta).
    """
    from woffl.assembly.network_optimizer import NetworkOptimizer, PowerFluidConstraint
    from woffl.assembly.optimization_algorithms import optimize
    from woffl.gui.scotts_tools._common import worker_ceiling

    cap = s_pad_plant.station_capacity(n_pumps)  # hydraulic (thrust) ceiling
    ppf = s_pad_plant.discharge_pressure(0.6 * cap, n_pumps)  # warm start
    history = []
    results, optimizer, converged = [], None, False

    for it in range(max_iter):
        ppf_c = max(1000.0, min(5000.0, ppf))
        for wc in well_configs:
            wc.ppf_surf_well = ppf_c  # common header pressure for every well
        pf = PowerFluidConstraint(total_rate=cap, pressure=ppf_c, rho_pf=62.4)
        optimizer = NetworkOptimizer(well_configs, pf, nozzles, throats, marginal_wc)
        optimizer.run_all_batch_simulations(max_workers=worker_ceiling())
        results = optimize(optimizer, method=method, water_key="lift_wat")

        total_pf = sum(r.predicted_lift_water for r in results)
        new_ppf = s_pad_plant.discharge_pressure(total_pf, n_pumps)
        history.append(
            {
                "iter": it + 1,
                "trial_psi": round(ppf_c, 1),
                "total_pf_bpd": round(total_pf, 0),
                "curve_psi": round(new_ppf, 1),
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
        "header_psi": max(1000.0, min(5000.0, ppf)),
        "total_pf_bpd": total_pf,
        "per_pump_bpd": s_pad_plant.per_pump_flow(total_pf, n_pumps),
        "in_range": s_pad_plant.flow_in_range(total_pf, n_pumps),
        "station_cap_bpd": cap,
        "converged": converged,
        "history": history,
        "total_oil_bopd": sum(r.predicted_oil_rate for r in results),
        "nozzles": list(nozzles),
        "throats": list(throats),
    }
    return results, optimizer, meta


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
    """Evaluate a FIXED per-well pump scenario against the booster curve.

    Like ``_run_coupled_optimization`` but instead of letting the optimizer pick
    pumps, each well's pump is fixed by ``choices`` (well_name -> (nozzle, throat),
    or None to shut the well in). Still couples delivered header pressure to total
    PF via the curve (fixed point), so the engineer sees the real oil + header
    pressure for THEIR selection.

    When a chosen pump has no solution at the resulting header pressure, the well
    is NOT zeroed (which would make the comparison worthless) — it falls back to
    ``fallback_choices`` (the optimized pick) if given, else the best feasible
    pump in the batch, and the substitution is flagged in the row. Returns
    (per_well rows, meta).
    """
    from woffl.assembly.network_optimizer import NetworkOptimizer, PowerFluidConstraint
    from woffl.gui.scotts_tools._common import worker_ceiling

    # Batch must compute every chosen pump AND every fallback pump (used when a
    # chosen pump is infeasible), so union both.
    all_ch = list(choices.values()) + list((fallback_choices or {}).values())
    nozzles = sorted({c[0] for c in all_ch if c}) or ["12"]
    throats = sorted({c[1] for c in all_ch if c}) or ["B"]
    cap = s_pad_plant.station_capacity(n_pumps)
    ppf = s_pad_plant.discharge_pressure(0.6 * cap, n_pumps)
    history, per_well, total_pf, total_oil = [], [], 0.0, 0.0
    converged = False

    for it in range(max_iter):
        ppf_c = max(1000.0, min(5000.0, ppf))
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

        new_ppf = s_pad_plant.discharge_pressure(total_pf, n_pumps)
        history.append(
            {
                "iter": it + 1,
                "trial_psi": round(ppf_c, 1),
                "total_pf_bpd": round(total_pf, 0),
                "curve_psi": round(new_ppf, 1),
            }
        )
        if progress:
            progress(it + 1, max_iter, ppf_c, total_pf, new_ppf)
        if abs(new_ppf - ppf_c) <= tol_psi:
            ppf = new_ppf
            converged = True
            break
        ppf = relax * new_ppf + (1 - relax) * ppf_c

    # Average pressure-ripple from the UNCHANGED wells that DID solve (these
    # dropped only because the header pressure fell) — applied to the wells the
    # model couldn't solve, so the ★ wells reflect the same ripple off their
    # latest test rate instead of staying flat. Keeps the ★ flag, then recomputes
    # the totals + header from the adjusted per-well rows.
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
            if total_pf > 0:
                ppf = s_pad_plant.discharge_pressure(total_pf, n_pumps)

    meta = {
        "n_pumps": n_pumps,
        "header_psi": max(1000.0, min(5000.0, ppf)),
        "total_pf_bpd": total_pf,
        "total_oil_bopd": total_oil,
        "per_pump_bpd": s_pad_plant.per_pump_flow(total_pf, n_pumps),
        "in_range": s_pad_plant.flow_in_range(total_pf, n_pumps),
        "station_cap_bpd": cap,
        "history": history,
        # After 8 oscillating iterations the rows come from the last trial
        # header while header_psi is the damped extrapolation — flag it.
        "converged": converged,
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
    """Existing-baseline scenario, anchored to MEASURED latest-test rates.

    Each well's displayed scenario value = its measured current oil/PF × the
    MODEL's RELATIVE change (scenario pump @ scenario header ÷ current pump @
    current header). This keeps every well on the same footing as the measured
    'Current' column — an unchanged well's PF falls when the header pressure
    falls, instead of jumping because the model's absolute level differs from
    what the test measured. The model bias cancels in the ratio.

    Non-solving wells use the average ripple of the unchanged solving wells.
    Returns (per_well, scn_meta) — per_well oil/pf are the displayed scenario
    values; the caller shows Current = test_rates separately.
    """
    from woffl.assembly.network_optimizer import NetworkOptimizer, PowerFluidConstraint
    from woffl.gui.scotts_tools._common import worker_ceiling

    names = [wc.well_name for wc in well_configs]
    cur_oil = {w: float((test_rates.get(w) or (0, 0))[0] or 0.0) for w in names}
    cur_pf = {w: float((test_rates.get(w) or (0, 0))[1] or 0.0) for w in names}

    all_ch = list(scenario_choices.values()) + list(current_choices.values())
    nozzles = sorted({c[0] for c in all_ch if c}) or ["12"]
    throats = sorted({c[1] for c in all_ch if c}) or ["B"]
    cap = s_pad_plant.station_capacity(n_pumps)
    total_pf_base = sum(cur_pf.values())
    header_base = (
        s_pad_plant.discharge_pressure(total_pf_base, n_pumps)
        if total_pf_base > 0
        else s_pad_plant.discharge_pressure(0.6 * cap, n_pumps)
    )

    def _run(ppf):
        ppf_c = max(1000.0, min(5000.0, ppf))
        for wc in well_configs:
            wc.ppf_surf_well = ppf_c
        pf = PowerFluidConstraint(total_rate=cap, pressure=ppf_c, rho_pf=62.4)
        opt = NetworkOptimizer(
            well_configs, pf, nozzles, throats, marginal_watercut=1.0
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

    ppf = header_base
    per_well = []
    converged = False
    for it in range(max_iter):
        ppf_c = max(1000.0, min(5000.0, ppf))
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
        new_ppf = s_pad_plant.discharge_pressure(total_pf, n_pumps)
        if progress:
            progress(it + 1, max_iter, ppf_c, total_pf, new_ppf)
        if abs(new_ppf - ppf_c) <= tol_psi:
            ppf = new_ppf
            converged = True
            break
        ppf = relax * new_ppf + (1 - relax) * ppf_c

    # Non-solvers: average ripple of the UNCHANGED solving wells (scenario ÷ current).
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
            # No measured rate to anchor to (e.g. no current pump / no test) —
            # estimate from the best feasible pump the model CAN solve for this
            # well, so it isn't a misleading 0.
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

    # Bias factor per well: model-at-current ÷ measured. >1 = model over-predicts
    # this well's current rate (its IPR/calibration match is loose) — surfaced in
    # the table so the engineer can target those matches.
    for r in per_well:
        mcw, co = mc.get(r["well"]), cur_oil.get(r["well"], 0.0)
        r["bias"] = (mcw[0] / co) if (mcw and mcw[0] and co > 0) else None

    total_oil = sum(r["oil"] for r in per_well)
    total_pf = sum(r["pf"] for r in per_well)
    ppf = s_pad_plant.discharge_pressure(total_pf, n_pumps) if total_pf > 0 else ppf

    scn_meta = {
        "n_pumps": n_pumps,
        "header_psi": max(1000.0, min(5000.0, ppf)),
        "total_pf_bpd": total_pf,
        "total_oil_bopd": total_oil,
        "per_pump_bpd": s_pad_plant.per_pump_flow(total_pf, n_pumps),
        "in_range": s_pad_plant.flow_in_range(total_pf, n_pumps),
        "station_cap_bpd": cap,
        "history": [],
        "converged": converged,
    }
    return per_well, scn_meta


def _match_check(well_configs, current_choices, n_pumps, test_rates):
    """Pre-flight diagnostic: model each well at its CURRENT pump + chosen IPR and
    compare to its measured recent tests (median). Flags wells where the model is a total
    mismatch on oil (loose IPR) or PF (a PF bust) — the wells to fix before
    trusting the optimizer. Returns (rows, header_psi)."""
    from woffl.assembly.network_optimizer import NetworkOptimizer, PowerFluidConstraint
    from woffl.gui.scotts_tools._common import worker_ceiling

    names = [wc.well_name for wc in well_configs]
    cur_oil = {w: float((test_rates.get(w) or (0, 0))[0] or 0.0) for w in names}
    cur_pf = {w: float((test_rates.get(w) or (0, 0))[1] or 0.0) for w in names}
    pumps = [c for c in current_choices.values() if c]
    nozzles = sorted({c[0] for c in pumps}) or ["12"]
    throats = sorted({c[1] for c in pumps}) or ["B"]
    cap = s_pad_plant.station_capacity(n_pumps)
    total_pf = sum(cur_pf.values())
    header = (
        s_pad_plant.discharge_pressure(total_pf, n_pumps)
        if total_pf > 0
        else s_pad_plant.discharge_pressure(0.6 * cap, n_pumps)
    )
    hc = max(1000.0, min(5000.0, header))
    for wc in well_configs:
        wc.ppf_surf_well = hc
    pf = PowerFluidConstraint(total_rate=cap, pressure=hc, rho_pf=62.4)
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
    return rows, hc


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
        if st.button("Next: Configure & Run →", type="primary", key="sp_to_config"):
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
        st.warning(
            "No active reviewed wells. Go back to **Review Wells** "
            "(every well may be marked offline)."
        )
        return

    offline = [w for w in store if store[w].get("offline")]
    st.markdown(
        f"**{len(active)} active well(s)** to optimize: " + ", ".join(active.keys())
    )
    if offline:
        st.caption("Offline (excluded): " + ", ".join(offline))
    st.caption(
        "Delivered power-fluid pressure is set by the 3-pump booster curve — "
        "more total PF flow means lower header pressure. The optimizer is solved "
        "to a fixed point against that curve."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        n_pumps = st.radio(
            "Booster pumps online",
            [3, 2],
            horizontal=True,
            key="sp_n_pumps",
            help="All 3 normally; drop to 2 to model one offline.",
        )
    with c2:
        method = st.selectbox("Optimizer", ["milp", "mckp"], key="sp_method")
    with c3:
        marginal_wc = st.number_input(
            "Marginal water cut",
            0.0,
            1.0,
            1.0,
            0.01,
            format="%.2f",
            key="sp_marginal_wc",
            help="S-Pad default 1.0 (no marginal-WC cutoff).",
        )

    c4, c5 = st.columns(2)
    with c4:
        nozzles = st.multiselect(
            "Nozzle sizes to test",
            NOZZLE_OPTIONS,
            default=["9", "10", "11", "12", "13", "14", "15"],
            key="sp_nozzles",
        )
    with c5:
        throats = st.multiselect(
            "Throat ratios to test",
            THROAT_OPTIONS,
            default=["A", "B", "C", "D"],
            key="sp_throats",
        )

    lo, hi = s_pad_plant.recommended_flow_per_pump()
    st.caption(
        f"Per-pump thrust window: {lo:,.0f}–{hi:,.0f} BPD  ·  "
        f"station capacity ({n_pumps} pumps): {s_pad_plant.station_capacity(n_pumps):,.0f} BPD"
    )

    # Where the pad sits on the combined booster curve. Before the first run it
    # shows just the curve + thrust window; once a run for this pump count
    # exists, the last operating point is overlaid.
    st.markdown("##### Combined pump curve")
    last_meta = st.session_state.get("sp_run_meta")
    op = last_meta if (last_meta and last_meta.get("n_pumps") == n_pumps) else None
    if op:
        st.caption("✕ marks the operating point from the most recent run.")
    _render_curve_plot(
        n_pumps,
        op_points=(
            [
                {
                    "label": "Last run",
                    "total_pf_bpd": op["total_pf_bpd"],
                    "header_psi": op["header_psi"],
                    "color": "#d62728",
                }
            ]
            if op
            else None
        ),
    )

    # ── Pre-flight: model-vs-test match check ────────────────────────────
    st.divider()
    st.markdown("##### Pre-flight: model match check")
    st.caption(
        "Before optimizing, model each well at its current pump + chosen IPR and "
        "compare to its recent tests (median). Fix the ✗ wells first — a model that can't "
        "reproduce a well's current oil/PF won't optimize it reliably."
    )
    current = {
        w: (active[w].get("review_nozzle") or "", active[w].get("review_throat") or "")
        for w in active
    }
    # Auto-run on landing; re-runs only when the config changes (pump count, well
    # set, current pumps, or IPRs) — keyed on a lightweight signature so it
    # doesn't re-compute on every interaction.
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
    mc = st.session_state.get("sp_matchcheck")
    refresh = st.button("↻ Re-run check", key="sp_matchcheck_run")
    if refresh or not mc or mc.get("sig") != sig:
        cur_ch = {
            w: (current[w] if current[w][0] and current[w][1] else None) for w in active
        }
        tr = {w: _recent_test_rates(w) for w in active}
        try:
            with st.spinner("Modeling each well at its current pump…"):
                rows, hdr = _match_check(
                    wrs.store_to_well_configs(active), cur_ch, n_pumps, tr
                )
            mc = {"rows": rows, "header": hdr, "sig": sig}
            st.session_state["sp_matchcheck"] = mc
        except Exception as e:  # never block configuring/running on a check failure
            st.caption(f"Match check unavailable: {e}")
            mc = None

    if mc:
        import pandas as pd

        rows = mc["rows"]
        n_oil = sum(1 for r in rows if r["oil_flag"].startswith("✗"))
        n_pf = sum(1 for r in rows if r["pf_flag"].startswith("✗"))
        st.caption(
            f"Modeled at header ≈ {mc['header']:,.0f} psi.  "
            f"**{n_oil} oil mismatch · {n_pf} PF bust** (✗). ⚠ = off, ✓ = match. "
            "Oil×/PF× = model ÷ test. Worst matches first."
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

    if st.button("▶ Run optimization", type="primary", key="sp_run"):
        well_configs = wrs.store_to_well_configs(active)
        bar = st.progress(0.0, text="Solving pump-curve coupling…")

        def _progress(it, max_it, ppf, total_pf, new_ppf):
            bar.progress(
                min(it / max_it, 1.0),
                text=f"Iter {it}: header {ppf:,.0f} psi → {total_pf:,.0f} BPD "
                f"→ curve {new_ppf:,.0f} psi",
            )

        try:
            with st.spinner(
                "Running batch sweeps + optimization (this can take a minute)…"
            ):
                results, optimizer, meta = _run_coupled_optimization(
                    well_configs,
                    n_pumps,
                    nozzles,
                    throats,
                    method,
                    marginal_wc,
                    progress=_progress,
                )
        except Exception as e:
            # I/M already guard their run; S let a single bad entry (e.g. a
            # hand-edited CSV field) blow the page with a raw traceback.
            bar.empty()
            st.error(f"Optimization failed: {e}")
            return
        bar.empty()
        # Stamp what this run was computed FROM, so Results can flag staleness
        # when wells are added/edited/toggled afterwards; the reconciliation
        # gives per-well drop reasons instead of a blanket "optimizer shut in".
        from woffl.assembly.network_optimizer import reconcile_wells

        meta["store_sig"] = wrs.store_signature(active)
        meta["reconciliation"] = reconcile_wells(optimizer, results)
        st.session_state["sp_opt_results"] = results
        st.session_state["sp_run_meta"] = meta
        # A fresh optimization invalidates any cached scenario / existing baseline.
        for k in (
            "sp_scenario_meta",
            "sp_scenario_per_well",
            "sp_baseline_meta",
            "sp_baseline_per_well",
        ):
            st.session_state.pop(k, None)
        st.session_state[_STAGE_KEY] = 2
        st.rerun()


def _render_results() -> None:
    import pandas as pd

    results = st.session_state.get("sp_opt_results")
    meta = st.session_state.get("sp_run_meta")
    if not results or not meta:
        st.warning("No results yet. Go to **Configure & Run**.")
        return

    store = store_for(PAD)
    active = wrs.active_entries(store)
    opt_choice = {
        r.well_name: (r.recommended_nozzle, r.recommended_throat) for r in results
    }
    opt_oil = {r.well_name: r.predicted_oil_rate for r in results}
    # Active wells NOT in the results: solver shut-in, failed simulation,
    # marginal-WC exclusion — or added after the run. The accounting box
    # renders the per-well reason; the banner flags store drift.
    si_wells = [w for w in active if w not in opt_choice]
    _render_results_accounting(meta, active, si_wells)

    # Station operating point
    st.markdown("#### Station operating point")
    conv = "converged" if meta["converged"] else "did NOT fully converge"
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Header pressure", f"{meta['header_psi']:,.0f} psi")
    m2.metric("Total power fluid", f"{meta['total_pf_bpd']:,.0f} BPD")
    m3.metric(f"Per pump (×{meta['n_pumps']})", f"{meta['per_pump_bpd']:,.0f} BPD")
    m4.metric("Total oil", f"{meta['total_oil_bopd']:,.0f} BOPD")

    if not meta["in_range"]:
        lo, hi = s_pad_plant.recommended_flow_per_pump()
        st.warning(
            f"Per-pump flow {meta['per_pump_bpd']:,.0f} BPD is OUTSIDE the "
            f"{lo:,.0f}–{hi:,.0f} thrust window — pump damage / off-curve risk."
        )
    if not meta["converged"]:
        st.warning(
            f"Pump-curve coupling {conv} in the iteration cap — review the trend below."
        )

    # Per-well table — includes SHUT IN wells so the SI decision is visible.
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

    c1, c2 = st.columns(2)
    with c1:
        if st.button("⬇ Download results CSV", key="sp_res_dl"):
            from woffl.gui.components.download import autodownload

            autodownload(
                df.to_csv(index=False).encode("utf-8"),
                "S_pad_optimization_results.csv",
                "text/csv",
                "sp_res_auto_dl",
            )
    with c2:
        with st.expander("Pump-curve coupling iterations"):
            st.dataframe(
                pd.DataFrame(meta["history"]), use_container_width=True, hide_index=True
            )

    # Booster curve
    st.markdown("#### Booster curve")
    _render_curve_plot(
        meta["n_pumps"],
        op_points=[
            {
                "label": "Optimized",
                "total_pf_bpd": meta["total_pf_bpd"],
                "header_psi": meta["header_psi"],
                "color": "#d62728",
            }
        ],
    )

    # ── Scenario comparator ──────────────────────────────────────────────
    st.divider()
    st.markdown("#### Compare a manual scenario")
    st.caption(
        "Choose a pump (or shut-in) per well and see the resulting oil + header "
        "pressure — coupled to the booster curve — side by side with a baseline."
    )

    base_choice = st.radio(
        "Compare against",
        ["Existing (recent test rates)", "Optimized"],
        horizontal=True,
        key="sp_cmp_base",
        help="Existing = each well's CURRENT measured oil/PF from its recent tests (median) "
        "(the natural baseline for 'I only change a couple of wells'). "
        "Optimized = the optimizer's picks. Pair Existing with Custom → "
        "Fill all from latest installed → change a few wells.",
    )
    is_existing = base_choice.startswith("Existing")
    # Switching the baseline changes how non-solving wells are handled, so a
    # scenario computed under the other baseline is stale — drop it.
    if st.session_state.get("sp_scn_base_used") not in (None, base_choice):
        st.session_state.pop("sp_scenario_meta", None)
        st.session_state.pop("sp_scenario_per_well", None)
    st.session_state["sp_scn_base_used"] = base_choice

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
        key="sp_scn_preset",
        help="Presets use every well's current or optimized pump; Custom reads the per-well picks below.",
    )
    if preset == "Custom per-well":
        # Autofill basis for the per-well pickers. The selectbox keys carry a
        # nonce so an autofill click re-initializes them to the new defaults
        # (set logical default + bump nonce = fresh widgets that re-read index);
        # individual edits then persist until the next autofill.
        basis = st.session_state.get("sp_scn_basis", "optimized")
        nonce = st.session_state.get("sp_scn_nonce", 0)
        bc = st.columns([1.4, 1.4, 3])
        if bc[0].button("⤓ Fill all from latest installed", key="sp_scn_fill_cur"):
            st.session_state["sp_scn_basis"] = "current"
            st.session_state["sp_scn_nonce"] = nonce + 1
            st.rerun()
        if bc[1].button("⤓ Fill all from optimized", key="sp_scn_fill_opt"):
            st.session_state["sp_scn_basis"] = "optimized"
            st.session_state["sp_scn_nonce"] = nonce + 1
            st.rerun()
        bc[2].caption(
            f"Filled from **{'latest installed pump' if basis == 'current' else 'optimized'}** "
            "— spot-edit any well below."
        )
        with st.expander(
            "Per-well pump selection (edit individual wells)", expanded=True
        ):
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
                        key=f"sp_scn_{w}_{nonce}",
                    )

    if st.button("▶ Compute scenario & compare", type="primary", key="sp_scn_run"):
        choices = {}
        for w in active:
            if preset == "All current pumps":
                nt = current.get(w)
                choices[w] = nt if (nt and nt[0] and nt[1]) else None
            elif preset == "All optimized (baseline)":
                choices[w] = opt_choice.get(w)  # None => SI (matches optimizer)
            else:
                _nonce = st.session_state.get("sp_scn_nonce", 0)
                choices[w] = _parse_pump(st.session_state.get(f"sp_scn_{w}_{_nonce}"))
        well_configs = wrs.store_to_well_configs(active)
        bar = st.progress(0.0, text="Solving scenario…")

        def _p(it, mx, ppf, tpf, _npf):
            bar.progress(
                min(it / mx, 1.0),
                text=f"Iter {it}: header {ppf:,.0f} psi → {tpf:,.0f} BPD",
            )

        with st.spinner("Running scenario (coupled to the booster curve)…"):
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
                    meta["n_pumps"],
                    test_rates=tr,
                    progress=_p,
                )
            else:
                per_well, scn_meta = _evaluate_fixed_scenario(
                    well_configs,
                    choices,
                    meta["n_pumps"],
                    fallback_choices=dict(opt_choice),
                    progress=_p,
                )
        bar.empty()
        st.session_state["sp_scenario_per_well"] = per_well
        st.session_state["sp_scenario_meta"] = scn_meta
        st.rerun()

    # Baseline: "Existing" (each well's measured latest-test oil/PF — the natural
    # current-state reference) or "Optimized" (the optimizer's picks).
    n = meta["n_pumps"]
    base_label = "Optimized"
    base_meta = meta
    base_oil = dict(opt_oil)
    base_pf_map = None
    base_pump = {
        w: (f"{opt_choice[w][0]}{opt_choice[w][1]}" if opt_choice.get(w) else "SHUT IN")
        for w in active
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
        base_meta = {
            "n_pumps": n,
            "total_oil_bopd": sum(base_oil.values()),
            "total_pf_bpd": tot_pf,
            "header_psi": (
                s_pad_plant.discharge_pressure(tot_pf, n) if tot_pf > 0 else 0.0
            ),
            "per_pump_bpd": (tot_pf / n) if n else 0.0,
            "in_range": s_pad_plant.flow_in_range(tot_pf, n) if tot_pf > 0 else False,
        }

    scn = st.session_state.get("sp_scenario_meta")
    scn_pw = st.session_state.get("sp_scenario_per_well")
    if scn and scn_pw:
        st.markdown(f"##### {base_label} vs your scenario")
        cmp = pd.DataFrame(
            {
                "Metric": [
                    "Total oil (BOPD)",
                    "Total PF (BPD)",
                    "Header PF pressure (psi)",
                    "Per-pump flow (BPD)",
                ],
                base_label: [
                    base_meta["total_oil_bopd"],
                    base_meta["total_pf_bpd"],
                    base_meta["header_psi"],
                    base_meta["per_pump_bpd"],
                ],
                "Your scenario": [
                    scn["total_oil_bopd"],
                    scn["total_pf_bpd"],
                    scn["header_psi"],
                    scn["per_pump_bpd"],
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
        if not scn["in_range"]:
            st.warning("Scenario per-pump flow is outside the thrust window.")
        if scn.get("converged") is False:
            st.warning(
                "Scenario pump-curve coupling did NOT converge in the iteration "
                "cap — the totals and header pressure are approximate; treat "
                "small deltas with caution."
            )

        subs = [
            r
            for r in scn_pw
            if r.get("note") and r["note"] not in ("infeasible", "star")
        ]
        starred = [r["well"] for r in scn_pw if r.get("note") == "star"]
        infeas = [r["well"] for r in scn_pw if r.get("note") == "infeasible"]
        if subs:
            st.caption(
                "⚠ Chosen pump infeasible — substituted a feasible one: "
                + ", ".join(f"{r['well']} ({r['pump']})" for r in subs)
            )
        if starred:
            st.caption(
                "★ Model couldn't solve — estimated from recent-test rate × the "
                "average ripple of the wells that did solve: " + ", ".join(starred)
            )
        if infeas:
            st.caption("✗ No feasible pump at all (counted as 0): " + ", ".join(infeas))

        st.markdown("**Where each option operates on the 3-pump booster curve**")
        _render_curve_plot(
            n,
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
            " **Model÷Test** = model's predicted current rate ÷ the measured test "
            "(>1 = model over-predicts; that well's IPR match is loose — tighten it first)."
            if is_existing
            else ""
        )
        st.caption(
            f"**Δ oil** = net per-well change ({base_label.lower()} → scenario), including the "
            f"ripple on unchanged wells (header pressure {base_meta['header_psi']:,.0f} → "
            f"{scn['header_psi']:,.0f} psi). ★ = model couldn't solve; estimated from latest "
            "test × the average ripple of the solving wells. Sorted by biggest mover."
            + bias_note
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
            if base_pf_map is not None:  # Existing mode: show how off the model is
                bias = srow.get("bias")
                row["Model÷Test"] = round(bias, 2) if bias else None
            mrows.append(row)
        mdf = pd.DataFrame(mrows).sort_values(
            "Δ oil", key=lambda s: s.abs(), ascending=False
        )
        st.dataframe(mdf, use_container_width=True, hide_index=True)


def _render_curve_plot(n_pumps: int, op_points=None) -> None:
    """Combined ``n_pumps``-pump booster curve (header pressure vs total flow),
    thrust window shaded, with optional labeled operating-point markers.

    ``op_points``: list of dicts {label, total_pf_bpd, header_psi, color?} — e.g.
    the optimized point and a manual-scenario point on the same curve.
    """
    import plotly.graph_objects as go

    lo, hi = s_pad_plant.recommended_flow_per_pump()
    down_x, up_x = lo * n_pumps, hi * n_pumps
    # Plot ~12% past the up-thrust limit so the up-thrust region is visible
    # (the per-pump polynomial is valid out to ~21,000 BPD/pump).
    x_max = up_x * 1.12
    n_pts = 72
    xs = [x_max * i / n_pts for i in range(1, n_pts + 1)]
    ys = [s_pad_plant.discharge_pressure(x, n_pumps) for x in xs]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name=f"{n_pumps}-pump curve",
            line=dict(color="#1f77b4"),
            hovertemplate="%{x:.0f} BPD<br>%{y:.0f} psi<extra></extra>",
        )
    )
    # Recommended (safe) window in green, between the down-thrust and up-thrust
    # limits; the up-thrust region beyond the high limit shaded red.
    fig.add_vrect(
        x0=down_x,
        x1=up_x,
        fillcolor="green",
        opacity=0.08,
        line_width=0,
        annotation_text="recommended window",
        annotation_position="top left",
    )
    fig.add_vrect(x0=up_x, x1=x_max, fillcolor="red", opacity=0.07, line_width=0)
    fig.add_vline(
        x=down_x,
        line=dict(color="#888", width=1, dash="dot"),
        annotation_text=f"down-thrust  {down_x:,.0f} BPD",
        annotation_position="bottom right",
    )
    fig.add_vline(
        x=up_x,
        line=dict(color="#d62728", width=1.5, dash="dash"),
        annotation_text=f"up-thrust limit  {up_x:,.0f} BPD",
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
        xaxis_title="Total station flow (BPD)",
        yaxis_title="Header pressure (psi)",
        height=380,
        margin=dict(t=20, b=40),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_s_pad_page() -> None:
    st.title("S-Pad Optimization 🛢")
    st.caption(
        "Review each S-Pad well, then optimize the pad against the 3-pump booster curve."
    )

    stage = st.session_state.setdefault(_STAGE_KEY, 0)

    # Stage navigation. Forward stages stay locked until there's data for them.
    have_results = bool(st.session_state.get("sp_opt_results"))
    have_wells = bool(wrs.active_entries(store_for(PAD)))
    cols = st.columns(3)
    for i, label in enumerate(_STAGES):
        unlocked = i == 0 or (i == 1 and have_wells) or (i == 2 and have_results)
        with cols[i]:
            if i == stage:
                st.markdown(f"**:blue[{label}]**")
            elif unlocked:
                if st.button(label, key=f"sp_nav_{i}", use_container_width=True):
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
