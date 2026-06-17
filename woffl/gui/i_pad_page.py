"""Dedicated I-Pad optimization page.

Same 3-stage flow as S-Pad (Review → Configure & Run → Results), reusing the
shared per-well review stage and store, but with I-Pad's very different booster
physics (see :mod:`woffl.gui.i_pad_plant`):

- S-Pad: 3 parallel fixed-speed pumps; delivered pressure is a curve of total
  flow; the optimizer fixed-points on it.
- **I-Pad: 2 pumps in series, VFD-driven, amp-limited.** There's no fixed-speed
  curve — the drives modulate speed and the real ceiling is motor amps. The
  train's capability is a falling frontier ``max_discharge_pressure(flow)``, and
  **delivered header pressure is an optimizer decision variable**: a lower
  pressure lets the pumps push more PF within their amp limits (Scott, 2026-06-16
  — "no hard cap; lowering discharge pressure to get more flow is a valid
  option"). So we SWEEP the header pressure, and at each candidate pressure give
  the optimizer the PF budget the train can actually deliver there
  (``max_flow_at_pressure``), then keep the pressure that yields the most oil.

Pressure floats freely, floored only at what the jet pumps need to lift. S-Pad's
page/plant are untouched.
"""

import streamlit as st

from woffl.gui import i_pad_plant as plant
from woffl.gui.workflow_steps import well_review_store as wrs
from woffl.gui.workflow_steps.step_review_wells import render_review_stage, store_for
# Reuse the genuinely pad-generic helpers from the S-Pad page (no plant coupling).
from woffl.gui.s_pad_page import _parse_pump, _recent_test_rates

PAD = "I"
_STAGES = ["1 · Review Wells", "2 · Configure & Run", "3 · Results"]
_STAGE_KEY = "ip_page_stage"

# Max safe HP-discharge (header) pressure for I-Pad (Scott, 2026-06-16). The pumps
# CAN make more at low flow, but operations cap the discharge here. Pressure floats
# freely BELOW this; it's never recommended above it. Honored by both the optimizer
# sweep and the scenario comparator.
_MAX_HEADER_PSI = 3500.0


# ---------------------------------------------------------------------------
# Frontier-coupled optimization (header pressure as a swept decision variable)
# ---------------------------------------------------------------------------


def _run_frontier_optimization(
    well_configs, nozzles, throats, method, marginal_wc,
    *, n_steps=11, progress=None,
):
    """Sweep the header pressure; at each, hand the optimizer the PF budget the
    series train can deliver there (the amp-limited frontier) and let it pick
    pumps to maximize oil. Keep the pressure with the most oil. Returns
    (results, meta)."""
    from woffl.assembly.network_optimizer import NetworkOptimizer, PowerFluidConstraint
    from woffl.assembly.optimization_algorithms import optimize
    from woffl.gui.scotts_tools._common import worker_ceiling

    suction = plant.suction_psi()
    # Sweep from a lift floor up to the operational discharge cap (the pumps could
    # make more at low flow, but we never recommend above _MAX_HEADER_PSI).
    p_ceiling = min(_MAX_HEADER_PSI, plant.max_discharge_pressure(6000.0) or 4000.0)
    p_floor = max(suction + 300.0, 1600.0)
    if p_ceiling <= p_floor:
        p_ceiling = p_floor + 500.0
    pressures = [p_floor + (p_ceiling - p_floor) * i / (n_steps - 1)
                 for i in range(n_steps)]

    sweep, best = [], None
    for idx, P in enumerate(pressures):
        cap = plant.max_flow_at_pressure(P)  # train's PF budget at this pressure
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
        rec = {"P": P, "cap": cap, "total_pf": total_pf, "total_oil": total_oil,
               "results": results}
        sweep.append(rec)
        if best is None or total_oil > best["total_oil"]:
            best = rec
        if progress:
            progress(idx + 1, n_steps, P, total_pf, total_oil)

    if best is None:
        raise RuntimeError(
            "No feasible header pressure found — the pumps couldn't deliver any "
            "well's power-fluid demand within their amp limits. Check the IPRs."
        )

    env = plant.operating_envelope([best["total_pf"]])[0]
    meta = {
        "header_psi": best["P"],
        "total_pf_bpd": best["total_pf"],
        "total_oil_bopd": best["total_oil"],
        "frontier_cap_bpd": best["cap"],
        "suction_psi": suction,
        "pumps": env["pumps"],          # per-pump {name, hz, dP, amps, amp_limit}
        "amp_limited": env["amp_limited"],
        "feasible": env["feasible"],
        "sweep": [{"header_psi": s["P"], "total_pf_bpd": s["total_pf"],
                   "total_oil_bopd": s["total_oil"]} for s in sweep],
        "nozzles": list(nozzles),
        "throats": list(throats),
    }
    return best["results"], meta


def _match_check_ipad(well_configs, current_choices, test_rates):
    """I-Pad pre-flight: model each well at its current pump + chosen IPR and
    compare to recent tests. Like the S-Pad check but the header pressure comes
    from the I-Pad frontier at the current total PF (not the S-Pad curve).
    Returns (rows, header_psi)."""
    from woffl.assembly.network_optimizer import NetworkOptimizer, PowerFluidConstraint
    from woffl.gui.scotts_tools._common import worker_ceiling

    names = [wc.well_name for wc in well_configs]
    cur_oil = {w: float((test_rates.get(w) or (0, 0))[0] or 0.0) for w in names}
    cur_pf = {w: float((test_rates.get(w) or (0, 0))[1] or 0.0) for w in names}
    total_pf = sum(cur_pf.values())
    header = (plant.max_discharge_pressure(total_pf) if total_pf > 0 else None)
    if header is None:  # no current PF, or beyond frontier — model at the live setpoint
        header = plant.max_discharge_pressure(8000.0) or 3400.0
    pumps = [c for c in current_choices.values() if c]
    nozzles = sorted({c[0] for c in pumps}) or ["12"]
    throats = sorted({c[1] for c in pumps}) or ["B"]
    for wc in well_configs:
        wc.ppf_surf_well = header
    pf = PowerFluidConstraint(total_rate=max(total_pf * 1.5, 60000.0),
                              pressure=header, rho_pf=62.4)
    opt = NetworkOptimizer(well_configs, pf, nozzles, throats, 0.94)
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
        rows.append({
            "well": w, "pump": (f"{cc[0]}{cc[1]}" if cc else "—"),
            "test_oil": to, "model_oil": mo, "oil_ratio": oil_ratio, "oil_flag": flag(oil_ratio),
            "test_pf": tp, "model_pf": mp, "pf_ratio": pf_ratio, "pf_flag": flag(pf_ratio),
        })
    return rows, header


# ---------------------------------------------------------------------------
# Scenario comparator — fixed per-well pumps vs a baseline, coupled to the frontier
# ---------------------------------------------------------------------------


def _frontier_header(total_pf: float, fallback: float):
    """Header pressure the train settles at for a given total PF: the amp-limited
    frontier at that flow. Returns (header_psi, over_capacity). over_capacity is
    True when the flow exceeds what the train can push at any pressure."""
    if total_pf <= 0:
        return min(_MAX_HEADER_PSI, fallback), False
    fp = plant.max_discharge_pressure(total_pf)
    if fp is None:
        return plant.suction_psi(), True
    return min(_MAX_HEADER_PSI, fp), False  # capped at the operational discharge limit


def _eval_cap() -> float:
    """Non-binding PF budget for the scenario optimizer (fixed pumps draw what
    they draw; we read specific pumps via get_pump_performance, not optimize)."""
    return plant.max_flow_at_pressure(plant.suction_psi() + 200.0) or 120000.0


def _evaluate_fixed_scenario(
    well_configs, choices, *, fallback_choices=None, test_rates=None,
    current_choices=None, max_iter=8, tol_psi=10.0, relax=0.6, progress=None,
):
    """Evaluate FIXED per-well pumps against the I-Pad frontier (fixed point):
    at a trial header the chosen pumps draw a total PF; the frontier at that flow
    sets the new header; iterate. A chosen pump with no solution falls back to the
    optimized pick, else the best feasible pump (flagged); for the Existing
    baseline a non-solving well falls back to its measured test rate (★)."""
    from woffl.assembly.network_optimizer import NetworkOptimizer, PowerFluidConstraint
    from woffl.gui.scotts_tools._common import worker_ceiling

    all_ch = list(choices.values()) + list((fallback_choices or {}).values())
    nozzles = sorted({c[0] for c in all_ch if c}) or ["12"]
    throats = sorted({c[1] for c in all_ch if c}) or ["B"]
    cap = _eval_cap()
    ppf = plant.max_discharge_pressure(6000.0) or 3400.0
    history, per_well, total_pf, total_oil, over_cap = [], [], 0.0, 0.0, False
    lo_p = plant.suction_psi()

    for it in range(max_iter):
        ppf_c = max(lo_p, min(_MAX_HEADER_PSI,ppf))
        for wc in well_configs:
            wc.ppf_surf_well = ppf_c
        pf = PowerFluidConstraint(total_rate=cap, pressure=ppf_c, rho_pf=62.4)
        opt = NetworkOptimizer(well_configs, pf, nozzles, throats, 0.94)
        opt.run_all_batch_simulations(max_workers=worker_ceiling())

        per_well, total_pf, total_oil = [], 0.0, 0.0
        for wc in well_configs:
            ch = choices.get(wc.well_name)
            if not ch:
                per_well.append({"well": wc.well_name, "pump": "SHUT IN",
                                 "oil": 0.0, "pf": 0.0, "note": ""})
                continue
            perf = opt.get_pump_performance(wc.well_name, ch[0], ch[1])
            note = ""
            if perf is None and test_rates and wc.well_name in test_rates:
                to, tp = test_rates[wc.well_name]
                per_well.append({"well": wc.well_name, "pump": f"{ch[0]}{ch[1]} ★",
                                 "oil": float(to or 0.0), "pf": float(tp or 0.0), "note": "star"})
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
                                note, ch = f"{choices[wc.well_name][0]}{choices[wc.well_name][1]}✗→{fbn}{fbt}", (fbn, fbt)
            if perf is None:
                orig = choices[wc.well_name]
                per_well.append({"well": wc.well_name,
                                 "pump": f"{orig[0]}{orig[1]} ✗ no feasible pump",
                                 "oil": 0.0, "pf": 0.0, "note": "infeasible"})
                continue
            per_well.append({"well": wc.well_name, "pump": note or f"{ch[0]}{ch[1]}",
                             "oil": perf["oil_rate"], "pf": perf["lift_water"], "note": note})
            total_pf += perf["lift_water"]
            total_oil += perf["oil_rate"]

        new_ppf, over_cap = _frontier_header(total_pf, ppf_c)
        history.append({"iter": it + 1, "trial_psi": round(ppf_c, 1),
                        "total_pf_bpd": round(total_pf, 0), "frontier_psi": round(new_ppf, 1)})
        if progress:
            progress(it + 1, max_iter, ppf_c, total_pf, new_ppf)
        if abs(new_ppf - ppf_c) <= tol_psi:
            ppf = new_ppf
            break
        ppf = relax * new_ppf + (1 - relax) * ppf_c

    # ★ wells (Existing baseline): ripple their measured rate by the average
    # change of the UNCHANGED solving wells, then recompute totals + header.
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
        avg_oil = min(1.2, max(0.3, sum(oil_ratios) / len(oil_ratios))) if oil_ratios else 1.0
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
        if rescaled:
            total_oil = sum(r["oil"] for r in per_well)
            total_pf = sum(r["pf"] for r in per_well)
            ppf, over_cap = _frontier_header(total_pf, ppf)

    meta = {"header_psi": max(lo_p, min(_MAX_HEADER_PSI,ppf)), "total_pf_bpd": total_pf,
            "total_oil_bopd": total_oil, "over_capacity": over_cap, "history": history}
    return per_well, meta


def _evaluate_existing_scenario(
    well_configs, scenario_choices, current_choices, *,
    test_rates, max_iter=8, tol_psi=10.0, relax=0.6, progress=None,
):
    """Existing-baseline scenario, anchored to MEASURED test rates: each well's
    displayed value = measured current oil/PF × the model's RELATIVE change
    (scenario pump @ scenario header ÷ current pump @ current header). Model bias
    cancels in the ratio, so an unchanged well's PF falls with the header instead
    of jumping. Non-solvers use the average ripple of the unchanged solvers."""
    from woffl.assembly.network_optimizer import NetworkOptimizer, PowerFluidConstraint
    from woffl.gui.scotts_tools._common import worker_ceiling

    names = [wc.well_name for wc in well_configs]
    cur_oil = {w: float((test_rates.get(w) or (0, 0))[0] or 0.0) for w in names}
    cur_pf = {w: float((test_rates.get(w) or (0, 0))[1] or 0.0) for w in names}
    all_ch = list(scenario_choices.values()) + list(current_choices.values())
    nozzles = sorted({c[0] for c in all_ch if c}) or ["12"]
    throats = sorted({c[1] for c in all_ch if c}) or ["B"]
    cap = _eval_cap()
    lo_p = plant.suction_psi()
    total_pf_base = sum(cur_pf.values())
    header_base, _ = _frontier_header(total_pf_base, plant.max_discharge_pressure(6000.0) or 3400.0)

    def _run(ppf):
        ppf_c = max(lo_p, min(_MAX_HEADER_PSI,ppf))
        for wc in well_configs:
            wc.ppf_surf_well = ppf_c
        pf = PowerFluidConstraint(total_rate=cap, pressure=ppf_c, rho_pf=62.4)
        opt = NetworkOptimizer(well_configs, pf, nozzles, throats, 0.94)
        opt.run_all_batch_simulations(max_workers=worker_ceiling())
        return opt

    opt_base = _run(header_base)
    mc = {}
    for w in names:
        cc = current_choices.get(w)
        perf = opt_base.get_pump_performance(w, cc[0], cc[1]) if cc else None
        mc[w] = (perf["oil_rate"], perf["lift_water"]) if perf else None

    ppf, per_well = header_base, []
    for it in range(max_iter):
        ppf_c = max(lo_p, min(_MAX_HEADER_PSI,ppf))
        opt = _run(ppf_c)
        per_well, total_pf = [], 0.0
        for w in names:
            ch = scenario_choices.get(w)
            if not ch:
                per_well.append({"well": w, "pump": "SHUT IN", "oil": 0.0, "pf": 0.0, "note": ""})
                continue
            ms = opt.get_pump_performance(w, ch[0], ch[1])
            mcw = mc.get(w)
            if ms and mcw and mcw[0] > 0 and mcw[1] > 0:
                so = cur_oil[w] * (ms["oil_rate"] / mcw[0])
                sp = cur_pf[w] * (ms["lift_water"] / mcw[1])
                per_well.append({"well": w, "pump": f"{ch[0]}{ch[1]}", "oil": so, "pf": sp, "note": ""})
                total_pf += sp
            elif ms:
                per_well.append({"well": w, "pump": f"{ch[0]}{ch[1]}",
                                 "oil": float(ms["oil_rate"]), "pf": float(ms["lift_water"]), "note": ""})
                total_pf += float(ms["lift_water"])
            else:
                per_well.append({"well": w, "pump": f"{ch[0]}{ch[1]} ★",
                                 "oil": cur_oil[w], "pf": cur_pf[w], "note": "star"})
                total_pf += cur_pf[w]
        new_ppf, _ = _frontier_header(total_pf, ppf_c)
        if progress:
            progress(it + 1, max_iter, ppf_c, total_pf, new_ppf)
        if abs(new_ppf - ppf_c) <= tol_psi:
            ppf = new_ppf
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
    avg_oil = min(1.2, max(0.3, sum(oil_ratios) / len(oil_ratios))) if oil_ratios else 1.0
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
                    perf = opt.get_pump_performance(w, str(br["nozzle"]), str(br["throat"]))
                    if perf:
                        r["oil"], r["pf"] = float(perf["oil_rate"]), float(perf["lift_water"])
                        r["pump"] = f"{r['pump']} (est {br['nozzle']}{br['throat']})"

    for r in per_well:
        mcw, co = mc.get(r["well"]), cur_oil.get(r["well"], 0.0)
        r["bias"] = (mcw[0] / co) if (mcw and mcw[0] and co > 0) else None

    total_oil = sum(r["oil"] for r in per_well)
    total_pf = sum(r["pf"] for r in per_well)
    header, over_cap = _frontier_header(total_pf, ppf)
    scn_meta = {"header_psi": max(lo_p, min(_MAX_HEADER_PSI,header)), "total_pf_bpd": total_pf,
                "total_oil_bopd": total_oil, "over_capacity": over_cap, "history": []}
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
        if st.button("Next: Configure & Run →", type="primary", key="ip_to_config"):
            st.session_state[_STAGE_KEY] = 1
            st.rerun()
    elif store:
        st.warning("Every reviewed well is marked offline — bring at least one online to continue.")
    else:
        st.info("Review at least one well (or add a hypothetical) to continue.")


def _render_configure() -> None:
    from woffl.gui.params import NOZZLE_OPTIONS, THROAT_OPTIONS

    store = store_for(PAD)
    active = wrs.active_entries(store)
    if not active:
        st.warning("No active reviewed wells. Go back to **Review Wells** "
                   "(every well may be marked offline).")
        return

    offline = [w for w in store if store[w].get("offline")]
    st.markdown(f"**{len(active)} active well(s)** to optimize: " + ", ".join(active.keys()))
    if offline:
        st.caption("Offline (excluded): " + ", ".join(offline))
    st.caption(
        "I-Pad's two boosters run in **series on VFDs**, so delivered pressure isn't "
        "a fixed curve — the real limit is motor amps. The optimizer treats the "
        "**header pressure as a free variable**: it can lower pressure to push more "
        "power fluid within the amp limits, and keeps whatever yields the most oil."
    )

    c1, c2 = st.columns(2)
    with c1:
        method = st.selectbox("Optimizer", ["milp", "mckp"], key="ip_method")
    with c2:
        marginal_wc = st.number_input("Marginal water cut", 0.0, 1.0, 1.0, 0.01,
                                      format="%.2f", key="ip_marginal_wc",
                                      help="Default 1.0 (no marginal-WC cutoff).")

    c4, c5 = st.columns(2)
    with c4:
        nozzles = st.multiselect("Nozzle sizes to test", NOZZLE_OPTIONS,
                                 default=["9", "10", "11", "12", "13", "14", "15"],
                                 key="ip_nozzles")
    with c5:
        throats = st.multiselect("Throat ratios to test", THROAT_OPTIONS,
                                 default=["A", "B", "C", "D"], key="ip_throats")

    # Frontier: max deliverable header pressure vs total PF flow.
    st.markdown("##### Booster train capability (amp-limited frontier)")
    last_meta = st.session_state.get("ip_run_meta")
    op = ([{"label": "Last run", "total_pf_bpd": last_meta["total_pf_bpd"],
            "header_psi": last_meta["header_psi"], "color": "#d62728"}]
          if last_meta else None)
    if op:
        st.caption("✕ marks the operating point from the most recent run.")
    _render_frontier_plot(op_points=op)

    # ── Pre-flight: model-vs-test match check ────────────────────────────
    st.divider()
    st.markdown("##### Pre-flight: model match check")
    st.caption(
        "Before optimizing, model each well at its current pump + chosen IPR and "
        "compare to its recent tests (median). Fix the ✗ wells first."
    )
    current = {
        w: (active[w].get("review_nozzle") or "", active[w].get("review_throat") or "")
        for w in active
    }
    sig = tuple(sorted(
        (w, current[w][0], current[w][1], int(active[w].get("qwf", 0)),
         int(active[w].get("pwf", 0)), int(active[w].get("res_pres", 0)))
        for w in active
    ))
    mc = st.session_state.get("ip_matchcheck")
    refresh = st.button("↻ Re-run check", key="ip_matchcheck_run")
    if refresh or not mc or mc.get("sig") != sig:
        cur_ch = {w: (current[w] if current[w][0] and current[w][1] else None) for w in active}
        tr = {w: _recent_test_rates(w) for w in active}
        try:
            with st.spinner("Modeling each well at its current pump…"):
                rows, hdr = _match_check_ipad(wrs.store_to_well_configs(active), cur_ch, tr)
            mc = {"rows": rows, "header": hdr, "sig": sig}
            st.session_state["ip_matchcheck"] = mc
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
            f"**{n_oil} oil mismatch · {n_pf} PF bust** (✗). ⚠ = off, ✓ = match. "
            "Worst matches first."
        )

        def _dev(r):
            ds = [abs(x - 1) for x in (r["oil_ratio"], r["pf_ratio"]) if x is not None]
            return max(ds) if ds else 0.0

        df = pd.DataFrame([{
            "Well": r["well"], "Pump": r["pump"],
            "Test oil": round(r["test_oil"]) if r["test_oil"] else None,
            "Model oil": round(r["model_oil"]) if r["model_oil"] is not None else None,
            "Oil×": round(r["oil_ratio"], 2) if r["oil_ratio"] else None,
            "Oil match": r["oil_flag"],
            "Test PF": round(r["test_pf"]) if r["test_pf"] else None,
            "Model PF": round(r["model_pf"]) if r["model_pf"] is not None else None,
            "PF×": round(r["pf_ratio"], 2) if r["pf_ratio"] else None,
            "PF match": r["pf_flag"],
        } for r in sorted(rows, key=_dev, reverse=True)])
        st.dataframe(df, use_container_width=True, hide_index=True)
    st.divider()

    if not nozzles or not throats:
        st.warning("Pick at least one nozzle and one throat.")
        return

    if st.button("▶ Run optimization", type="primary", key="ip_run"):
        well_configs = wrs.store_to_well_configs(active)
        bar = st.progress(0.0, text="Sweeping header pressure on the amp frontier…")

        def _progress(it, max_it, P, total_pf, total_oil):
            bar.progress(
                min(it / max_it, 1.0),
                text=f"Step {it}/{max_it}: header {P:,.0f} psi → "
                     f"{total_pf:,.0f} BPD PF → {total_oil:,.0f} BOPD",
            )

        try:
            with st.spinner("Running batch sweeps across the pressure frontier "
                            "(this can take a minute)…"):
                results, meta = _run_frontier_optimization(
                    well_configs, nozzles, throats, method, marginal_wc,
                    progress=_progress,
                )
        except Exception as e:
            bar.empty()
            st.error(f"Optimization failed: {e}")
            return
        bar.empty()
        st.session_state["ip_opt_results"] = results
        st.session_state["ip_run_meta"] = meta
        st.session_state[_STAGE_KEY] = 2
        st.rerun()


def _render_results() -> None:
    import pandas as pd

    results = st.session_state.get("ip_opt_results")
    meta = st.session_state.get("ip_run_meta")
    if not results or not meta:
        st.warning("No results yet. Go to **Configure & Run**.")
        return

    store = store_for(PAD)
    active = wrs.active_entries(store)
    opt_choice = {r.well_name: (r.recommended_nozzle, r.recommended_throat) for r in results}
    si_wells = [w for w in active if w not in opt_choice]

    # Station operating point
    st.markdown("#### Station operating point")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Header pressure", f"{meta['header_psi']:,.0f} psi")
    m2.metric("Total power fluid", f"{meta['total_pf_bpd']:,.0f} BPD")
    m3.metric("Frontier budget", f"{meta['frontier_cap_bpd']:,.0f} BPD")
    m4.metric("Total oil", f"{meta['total_oil_bopd']:,.0f} BOPD")

    # Per-pump amp headroom — the real I-Pad constraint.
    st.markdown("##### Booster amps (the binding constraint)")
    pump_rows = []
    for p in meta["pumps"]:
        if p["hz"] is None:
            pump_rows.append({"Pump": p["name"], "Speed": "—", "Amps": "—",
                              "Limit (A)": round(p["amp_limit"]), "Headroom": "INFEASIBLE"})
            continue
        head = p["amp_limit"] - p["amps"]
        pump_rows.append({
            "Pump": p["name"], "Speed": f"{p['hz']:.1f} Hz",
            "Amps": round(p["amps"], 1), "Limit (A)": round(p["amp_limit"]),
            "Headroom": f"{head:.1f} A ({head / p['amp_limit'] * 100:.0f}%)",
        })
    st.dataframe(pd.DataFrame(pump_rows), use_container_width=True, hide_index=True)
    if meta["amp_limited"]:
        st.caption("A pump is at its amp limit — the train is pushing as much PF as "
                   "it can at this pressure. Lower pressure trades for more flow.")

    if si_wells:
        st.info(
            f"**Optimizer shut in {len(si_wells)} well(s):** {', '.join(si_wells)} — "
            "their power fluid bought more oil elsewhere (or they're uneconomic at "
            "the marginal water cut)."
        )

    # Per-well table
    st.markdown("#### Per-well recommendations")
    rows = []
    for r in results:
        rows.append({
            "Well": r.well_name, "Nozzle": r.recommended_nozzle, "Throat": r.recommended_throat,
            "Oil (BOPD)": round(r.predicted_oil_rate, 0),
            "Power fluid (BPD)": round(r.predicted_lift_water, 0),
            "Form. water (BPD)": round(r.predicted_formation_water, 0),
            "Total WC": round(r.total_watercut, 3),
            "Suction (psi)": round(r.suction_pressure, 0),
            "Status": "⚠ sonic" if r.sonic_status else "run",
        })
    for w in si_wells:
        rows.append({
            "Well": w, "Nozzle": "—", "Throat": "—", "Oil (BOPD)": 0,
            "Power fluid (BPD)": 0, "Form. water (BPD)": 0, "Total WC": None,
            "Suction (psi)": None, "Status": "SHUT IN",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    if st.button("⬇ Download results CSV", key="ip_res_dl"):
        from woffl.gui.components.download import autodownload

        autodownload(df.to_csv(index=False).encode("utf-8"),
                     "I_pad_optimization_results.csv", "text/csv", "ip_res_auto_dl")

    # Frontier with the operating point + the pressure sweep that was explored.
    st.markdown("#### Amp-limited frontier")
    _render_frontier_plot(op_points=[{
        "label": "Optimized", "total_pf_bpd": meta["total_pf_bpd"],
        "header_psi": meta["header_psi"], "color": "#d62728",
    }])
    with st.expander("Pressure sweep (oil vs header pressure explored)"):
        st.dataframe(pd.DataFrame(meta["sweep"]), use_container_width=True, hide_index=True)

    _render_scenario_comparator(results, meta, active)


def _render_scenario_comparator(results, meta, active) -> None:
    import pandas as pd

    opt_choice = {r.well_name: (r.recommended_nozzle, r.recommended_throat) for r in results}
    opt_oil = {r.well_name: r.predicted_oil_rate for r in results}

    st.divider()
    st.markdown("#### Compare a manual scenario")
    st.caption(
        "Pick a pump (or shut-in) per well and see the resulting oil + header "
        "pressure — coupled to the amp-limited frontier — next to a baseline. "
        "Size pumps up and watch the PF rise and the pressure fall."
    )

    base_choice = st.radio(
        "Compare against", ["Existing (recent test rates)", "Optimized"],
        horizontal=True, key="ip_cmp_base",
        help="Existing = each well's CURRENT measured oil/PF from its recent tests "
             "(the natural 'I only change a couple of wells' baseline). "
             "Optimized = the optimizer's picks.",
    )
    is_existing = base_choice.startswith("Existing")
    if st.session_state.get("ip_scn_base_used") not in (None, base_choice):
        st.session_state.pop("ip_scenario_meta", None)
        st.session_state.pop("ip_scenario_per_well", None)
    st.session_state["ip_scn_base_used"] = base_choice

    combos = [f"{n}{t}" for n in meta.get("nozzles", []) for t in meta.get("throats", [])]
    current = {
        w: (active[w].get("review_nozzle") or "", active[w].get("review_throat") or "")
        for w in active
    }
    cur_labels = [f"{n}{t}" for (n, t) in current.values() if n and t]
    pump_options = sorted(set(combos + cur_labels)) + ["Shut in"]

    preset = st.radio(
        "Scenario", ["All current pumps", "All optimized (baseline)", "Custom per-well"],
        horizontal=True, key="ip_scn_preset",
        help="Presets use every well's current or optimized pump; Custom reads the per-well picks below.",
    )
    if preset == "Custom per-well":
        basis = st.session_state.get("ip_scn_basis", "optimized")
        nonce = st.session_state.get("ip_scn_nonce", 0)
        bc = st.columns([1.4, 1.4, 3])
        if bc[0].button("⤓ Fill all from latest installed", key="ip_scn_fill_cur"):
            st.session_state["ip_scn_basis"] = "current"
            st.session_state["ip_scn_nonce"] = nonce + 1
            st.rerun()
        if bc[1].button("⤓ Fill all from optimized", key="ip_scn_fill_opt"):
            st.session_state["ip_scn_basis"] = "optimized"
            st.session_state["ip_scn_nonce"] = nonce + 1
            st.rerun()
        bc[2].caption(
            f"Filled from **{'latest installed pump' if basis == 'current' else 'optimized'}** "
            "— spot-edit any well below."
        )
        with st.expander("Per-well pump selection (edit individual wells)", expanded=True):
            cols = st.columns(3)
            for i, w in enumerate(sorted(active)):
                src = current.get(w) if basis == "current" else opt_choice.get(w)
                d_lbl = f"{src[0]}{src[1]}" if (src and src[0] and src[1]) else "Shut in"
                if d_lbl not in pump_options:
                    d_lbl = pump_options[0]
                with cols[i % 3]:
                    st.selectbox(w, pump_options, index=pump_options.index(d_lbl),
                                 key=f"ip_scn_{w}_{nonce}")

    if st.button("▶ Compute scenario & compare", type="primary", key="ip_scn_run"):
        choices = {}
        for w in active:
            if preset == "All current pumps":
                nt = current.get(w)
                choices[w] = nt if (nt and nt[0] and nt[1]) else None
            elif preset == "All optimized (baseline)":
                choices[w] = opt_choice.get(w)
            else:
                _nonce = st.session_state.get("ip_scn_nonce", 0)
                choices[w] = _parse_pump(st.session_state.get(f"ip_scn_{w}_{_nonce}"))
        well_configs = wrs.store_to_well_configs(active)
        bar = st.progress(0.0, text="Solving scenario…")

        def _p(it, mx, ppf, tpf, _npf):
            bar.progress(min(it / mx, 1.0),
                         text=f"Iter {it}: header {ppf:,.0f} psi → {tpf:,.0f} BPD")

        try:
            with st.spinner("Running scenario (coupled to the frontier)…"):
                if is_existing:
                    cur_ch = {
                        w: (current[w] if current[w][0] and current[w][1] else None)
                        for w in active
                    }
                    tr = {w: _recent_test_rates(w) for w in active}
                    per_well, scn_meta = _evaluate_existing_scenario(
                        well_configs, choices, cur_ch, test_rates=tr, progress=_p,
                    )
                else:
                    per_well, scn_meta = _evaluate_fixed_scenario(
                        well_configs, choices, fallback_choices=dict(opt_choice), progress=_p,
                    )
        except Exception as e:
            bar.empty()
            st.error(f"Scenario failed: {e}")
            return
        bar.empty()
        st.session_state["ip_scenario_per_well"] = per_well
        st.session_state["ip_scenario_meta"] = scn_meta
        st.rerun()

    # Baseline: Existing (measured recent-test rates) or Optimized (optimizer picks).
    base_label = "Optimized"
    base_oil = dict(opt_oil)
    base_pf_map = None
    base_pump = {
        w: (f"{opt_choice[w][0]}{opt_choice[w][1]}" if opt_choice.get(w) else "SHUT IN")
        for w in active
    }
    base_meta = {"total_oil_bopd": meta["total_oil_bopd"],
                 "total_pf_bpd": meta["total_pf_bpd"], "header_psi": meta["header_psi"]}
    if is_existing:
        base_label = "Current"
        rates = {w: _recent_test_rates(w) for w in active}
        base_oil = {w: (rates[w][0] or 0.0) for w in active}
        base_pf_map = {w: (rates[w][1] or 0.0) for w in active}
        base_pump = {
            w: (f"{current[w][0]}{current[w][1]}" if current[w][0] and current[w][1] else "—")
            for w in active
        }
        tot_pf = sum(base_pf_map.values())
        hdr, _ = _frontier_header(tot_pf, 0.0)
        base_meta = {"total_oil_bopd": sum(base_oil.values()),
                     "total_pf_bpd": tot_pf, "header_psi": hdr}

    scn = st.session_state.get("ip_scenario_meta")
    scn_pw = st.session_state.get("ip_scenario_per_well")
    if scn and scn_pw:
        st.markdown(f"##### {base_label} vs your scenario")
        cmp = pd.DataFrame({
            "Metric": ["Total oil (BOPD)", "Total PF (BPD)", "Header PF pressure (psi)"],
            base_label: [base_meta["total_oil_bopd"], base_meta["total_pf_bpd"],
                         base_meta["header_psi"]],
            "Your scenario": [scn["total_oil_bopd"], scn["total_pf_bpd"], scn["header_psi"]],
        })
        cmp["Δ (scn − base)"] = cmp["Your scenario"] - cmp[base_label]
        for c in (base_label, "Your scenario", "Δ (scn − base)"):
            cmp[c] = cmp[c].round(0)
        st.dataframe(cmp, use_container_width=True, hide_index=True)

        d_oil = scn["total_oil_bopd"] - base_meta["total_oil_bopd"]
        if d_oil > 0:
            st.caption(f"Net: your scenario makes **{d_oil:,.0f} BOPD more** than {base_label.lower()}.")
        elif d_oil < 0:
            st.caption(f"Net: your scenario makes **{abs(d_oil):,.0f} BOPD less** than {base_label.lower()}.")
        if scn.get("over_capacity"):
            st.warning("Scenario draws more PF than the train can push at any pressure — "
                       "it would force the header below what the wells need. Size some pumps down.")

        subs = [r for r in scn_pw if r.get("note") and r["note"] not in ("infeasible", "star")]
        starred = [r["well"] for r in scn_pw if r.get("note") == "star"]
        infeas = [r["well"] for r in scn_pw if r.get("note") == "infeasible"]
        if subs:
            st.caption("⚠ Chosen pump infeasible — substituted a feasible one: "
                       + ", ".join(f"{r['well']} ({r['pump']})" for r in subs))
        if starred:
            st.caption("★ Model couldn't solve — estimated from recent-test rate × the "
                       "average ripple of the wells that did solve: " + ", ".join(starred))
        if infeas:
            st.caption("✗ No feasible pump at all (counted as 0): " + ", ".join(infeas))

        st.markdown("**Where each option sits on the amp-limited frontier**")
        _render_frontier_plot(op_points=[
            {"label": base_label, "total_pf_bpd": base_meta["total_pf_bpd"],
             "header_psi": base_meta["header_psi"], "color": "#2ca02c"},
            {"label": "Your scenario", "total_pf_bpd": scn["total_pf_bpd"],
             "header_psi": scn["header_psi"], "color": "#d62728"},
        ])

        bias_note = (" **Model÷Test** = model's predicted current rate ÷ the measured "
                     "test (>1 = the well's IPR match is loose — tighten it first)."
                     if is_existing else "")
        st.caption(
            f"**Δ oil** = per-well change ({base_label.lower()} → scenario), including the "
            f"ripple on unchanged wells as the header moves "
            f"({base_meta['header_psi']:,.0f} → {scn['header_psi']:,.0f} psi). "
            "★ = model couldn't solve; estimated from recent-test rate × the average "
            "ripple of the solvers. Sorted by biggest mover." + bias_note
        )
        scn_map = {r["well"]: r for r in scn_pw}
        mrows = []
        for w in sorted(active):
            srow = scn_map.get(w, {})
            b_oil = round(base_oil.get(w, 0))
            s_oil = round(srow.get("oil", 0))
            row = {"Well": w, f"{base_label} pump": base_pump.get(w, "—"),
                   f"{base_label} oil": b_oil}
            if base_pf_map is not None:
                row[f"{base_label} PF"] = round(base_pf_map.get(w, 0))
            row.update({"Scenario": srow.get("pump", "—"), "Scn oil": s_oil,
                        "Δ oil": s_oil - b_oil, "Scn PF": round(srow.get("pf", 0))})
            if base_pf_map is not None:
                bias = srow.get("bias")
                row["Model÷Test"] = round(bias, 2) if bias else None
            mrows.append(row)
        mdf = pd.DataFrame(mrows).sort_values("Δ oil", key=lambda s: s.abs(), ascending=False)
        st.dataframe(mdf, use_container_width=True, hide_index=True)


def _render_frontier_plot(op_points=None) -> None:
    """I-Pad amp-limited frontier: max deliverable header (HP-discharge) pressure
    vs total PF flow, with optional operating-point markers. Falls with flow;
    ends where a pump can no longer pass the flow within its amp limit."""
    import plotly.graph_objects as go

    # Sweep flow until the frontier returns None (a pump can't pass it).
    xs, ys = [], []
    q = 4000.0
    while q <= 4.0 * plant._max_valid_flow():
        p = plant.max_discharge_pressure(q)
        if p is None:
            break
        xs.append(q)
        ys.append(p)
        q += 2000.0

    fig = go.Figure()
    if xs:
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name="amp-limited frontier",
            line=dict(color="#1f77b4"),
            hovertemplate="%{x:.0f} BPD<br>max %{y:.0f} psi<extra></extra>"))
    for p in (op_points or []):
        fig.add_trace(go.Scatter(
            x=[p["total_pf_bpd"]], y=[p["header_psi"]], mode="markers+text",
            marker=dict(size=14, color=p.get("color", "#d62728"), symbol="x"),
            name=p["label"],
            text=[f"  {p['label']}: {p['header_psi']:.0f} psi @ {p['total_pf_bpd']:,.0f} BPD"],
            textposition="middle right",
            hovertemplate="%{x:.0f} BPD<br>%{y:.0f} psi<extra></extra>"))
    fig.update_layout(
        xaxis_title="Total power-fluid flow (BPD)",
        yaxis_title="Max deliverable header pressure (psi)",
        height=380, margin=dict(t=20, b=40), legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_i_pad_page() -> None:
    st.title("I-Pad Optimization 🛢")
    st.caption("Review each I-Pad well, then optimize the pad against the "
               "two-pump series booster train (amp-limited, pressure-free).")

    stage = st.session_state.setdefault(_STAGE_KEY, 0)

    have_results = bool(st.session_state.get("ip_opt_results"))
    have_wells = bool(wrs.active_entries(store_for(PAD)))
    cols = st.columns(3)
    for i, label in enumerate(_STAGES):
        unlocked = i == 0 or (i == 1 and have_wells) or (i == 2 and have_results)
        with cols[i]:
            if i == stage:
                st.markdown(f"**:blue[{label}]**")
            elif unlocked:
                if st.button(label, key=f"ip_nav_{i}", use_container_width=True):
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
