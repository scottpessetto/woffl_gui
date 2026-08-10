"""One-off experiment: M-Pad choke-model knee probe vs Prosper sensitivity.

Sections:
  1. Per-well knee ladder (oil/pf/psu/sonic vs forced delivered header).
  2. Model psu vs measured PIP gauges at today's settled header.
  3. Prosper power-fluid drop ladder comparison via a budget-scaled plant proxy.

Run:  venv python -u scripts/mpad_knee_probe.py   (cwd = repo root, PYTHONPATH=.)
Live Databricks READS only. No product code touched.

Everything lives inside main() behind the __main__ guard: NetworkOptimizer
uses a spawn-based ProcessPoolExecutor on Windows, which re-imports this
module in every worker.
"""

from __future__ import annotations

import statistics
import time
import traceback

T0 = time.time()


def log(msg: str = "") -> None:
    print(msg, flush=True)


def stamp(msg: str) -> None:
    log(f"[{time.time() - T0:7.0f}s] {msg}")


def fmt(v, w=7, d=0):
    if v is None:
        return " " * (w - 1) + "-"
    return f"{v:{w}.{d}f}"


# ---------------------------------------------------------------------------
# Coworker (Prosper) reference data
# ---------------------------------------------------------------------------

CW_WELL_NUMS = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 43, 45, 60, 62, 64]

# measured PIP gauges, psig (second number of each pair)
GAUGE_PIP = {
    10: 750, 12: 656, 14: 661, 16: 469, 18: 826, 20: 704,
    28: 317, 30: 422, 32: 420, 43: 310, 45: 293, 60: 465,
    62: 706, 64: 373,
}

# drop_psi -> (run_all_oil, action, oil_after, gain)
CW_SENS = {
    100: (10661, "shut M-24", 10731, 70),
    200: (10366, "shut M-24", 10731, 365),
    300: (10050, "shut M-22", 10706, 656),
    400: (9724, "shut M-43", 10607, 884),
    500: (9377, "shut M-22+M-43", 10397, 1020),
    600: (9013, "shut M-22+M-43", 10397, 1384),
    700: (8634, "shut M-22+M-43", 10383, 1748),
    800: (8237, "shut M-43+M-45", 10170, 1933),
    900: (7823, "shut M-22+M-62", 10052, 2228),
    1000: (7394, "shut M-43+M-62", 9950, 2556),
}
CW_NORMAL_HEADER = 3175.0
CW_NORMAL_OIL = 10946.0


def cw_name(num: int) -> str:
    return f"MPM-{num}"


class ScaledPlant:
    """M plant with its PF budget frontier scaled by s (degradation proxy)."""

    def __init__(self, inner, s: float):
        self._inner = inner
        self._s = float(s)

    def budget_at_pressure(self, pressure, n_pumps=None):
        return self._s * self._inner.budget_at_pressure(pressure, n_pumps)

    # explicit passthroughs for everything run_choke_optimization /
    # settled_header touch (coupling, windows, warm start, flags, msg, cap)
    @property
    def coupling(self):
        return self._inner.coupling

    @property
    def max_header_psi(self):
        return self._inner.max_header_psi

    @property
    def infeasible_sweep_msg(self):
        return self._inner.infeasible_sweep_msg

    def pressure_window(self, n_pumps=None):
        return self._inner.pressure_window(n_pumps)

    def warm_start_psi(self, n_pumps=None):
        return self._inner.warm_start_psi(n_pumps)

    def suction_psi(self):
        return self._inner.suction_psi()

    def flow_window(self, n_pumps=None):
        return self._inner.flow_window(n_pumps)

    def flags(self, q_total, n_pumps=None):
        return self._inner.flags(q_total, n_pumps)

    def header_at_flow(self, q_total, n_pumps=None):
        return self._inner.header_at_flow(q_total, n_pumps)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def compact_actions(rows) -> str:
    acts = []
    for r in rows:
        if r["action"] == "shut":
            acts.append(f"shut {r['well']}")
        elif r["action"] == "choke":
            acts.append(f"choke {r['well']}@{r['delivered_psi']:.0f}")
        elif r["action"] in ("hold", "excluded"):
            acts.append(f"{r['action']} {r['well']}")
    return "; ".join(acts) if acts else "(all full open)"


def main() -> None:
    # -- hydration (mirrors server/services/optimizer_runs._run_pad_job) ----
    stamp("hydrating configs / current pumps / tests / plant ...")

    from server.services.optimizer_runs import (
        _PAD_DEFAULTS,
        _build_configs,
        _current_and_tests,
        _pad_plant,
    )

    notes: list[str] = []
    prov: dict = {}
    configs = _build_configs(["M"], set(), [], notes, prov)
    names = [wc.well_name for wc in configs]
    current, test_rates = _current_and_tests(names)
    plant = _pad_plant("M")
    n_pumps = _PAD_DEFAULTS["M"]["n_pumps"]

    stamp(f"hydrated {len(configs)} wells: {sorted(names)}")
    log(f"hydration notes: {notes if notes else '(none)'}")
    log(f"wells with current pump: {sorted(current)}")
    log(f"wells with test rates:   {sorted(test_rates)}")

    missing = [cw_name(n) for n in CW_WELL_NUMS if cw_name(n) not in names]
    log(f"coworker wells we LACK: {missing if missing else '(none - all 18 present)'}")
    extra = [w for w in sorted(names) if w not in {cw_name(n) for n in CW_WELL_NUMS}]
    log(f"our wells not in his study: {extra if extra else '(none)'}")

    # -- inline replication of _model_at_forced_header + sonic status -------
    from woffl.assembly.network_optimizer import (
        NetworkOptimizer,
        PowerFluidConstraint,
    )
    from woffl.gui.scotts_tools._common import worker_ceiling
    from woffl.gui.pad_optimize import (
        _EVAL_CAP_FALLBACK_BPD,
        _RHO_PF_DEFAULT,
        _SCENARIO_MARGINAL_WC,
        run_choke_optimization,
        settled_header,
    )

    solve_cache: dict[float, dict] = {}

    def solve_at(header_psi: float) -> dict:
        """{well: {oil, pf, psu, sonic, mach} | None} at a forced header.

        Same ~15 lines as pad_optimize._model_at_forced_header, plus
        sonic_status and mach_te from get_pump_performance. Cached.
        """
        key = round(float(header_psi), 2)
        if key in solve_cache:
            return solve_cache[key]
        pumps = [c for c in current.values() if c]
        nozzles = sorted({c[0] for c in pumps}) or ["12"]
        throats = sorted({c[1] for c in pumps}) or ["B"]
        for wc in configs:
            wc.ppf_surf_well = header_psi
        pf = PowerFluidConstraint(
            total_rate=_EVAL_CAP_FALLBACK_BPD,
            pressure=header_psi,
            rho_pf=_RHO_PF_DEFAULT,
        )
        opt = NetworkOptimizer(
            configs, pf, nozzles, throats, marginal_watercut=_SCENARIO_MARGINAL_WC
        )
        opt.run_all_batch_simulations(max_workers=worker_ceiling())
        out: dict = {}
        for wc in configs:
            w = wc.well_name
            cc = current.get(w)
            perf = opt.get_pump_performance(w, cc[0], cc[1]) if cc else None
            if perf:
                out[w] = {
                    "oil": float(perf["oil_rate"]),
                    "pf": float(perf["lift_water"]),
                    "psu": (
                        float(perf["suction_pressure"])
                        if perf.get("suction_pressure") is not None
                        else None
                    ),
                    "sonic": bool(perf.get("sonic_status")),
                    "mach": float(perf.get("mach_te") or 0.0),
                }
            else:
                out[w] = None
        solve_cache[key] = out
        stamp(f"  solved header {header_psi:7.1f} psi "
              f"(oil {sum(v['oil'] for v in out.values() if v):7.0f} bopd, "
              f"pf {sum(v['pf'] for v in out.values() if v):7.0f} bpd, "
              f"{sum(1 for v in out.values() if v is None)} unsolved)")
        return out

    # ------------------------------------------------------------------
    # SECTION 1 - per-well knee ladder
    # ------------------------------------------------------------------
    log()
    log("=" * 78)
    log("SECTION 1 - PER-WELL KNEE LADDER (100 psi steps across the pressure window)")
    log("=" * 78)

    p_lo, p_hi = plant.pressure_window(n_pumps)
    log(f"pressure window ({n_pumps} pumps): floor {p_lo:.0f} psi, ceiling {p_hi:.0f} psi")

    ladder = []
    p = p_lo
    while p < p_hi - 1e-6:
        ladder.append(round(p, 1))
        p += 100.0
    ladder.append(round(p_hi, 1))
    log(f"ladder: {len(ladder)} levels from {ladder[0]} to {ladder[-1]}")

    grid = {lvl: solve_at(lvl) for lvl in ladder}

    well_list = sorted(names)
    knee_summary = []
    for w in well_list:
        log()
        cc = current.get(w)
        if not cc:
            log(f"--- {w}  (NO CURRENT PUMP - unmodelable)")
            continue
        log(f"--- {w}  (pump {cc[0]}{cc[1]})")
        log("  level_psi |    oil |     pf |    psu | sonic (mach)")
        solved = []
        for lvl in ladder:
            v = grid[lvl].get(w)
            if v is None:
                log(f"  {lvl:9.0f} |      - |      - |      - | unsolved")
            else:
                log(f"  {lvl:9.0f} | {v['oil']:6.1f} | {v['pf']:6.0f} | {fmt(v['psu'], 6)} | "
                    f"{'SONIC' if v['sonic'] else 'sub  '} ({v['mach']:.2f})")
                solved.append((lvl, v))
        if not solved:
            log("  -> no solvable level; excluded from knee stats")
            continue
        top_lvl, top_v = solved[-1]
        oil_top = top_v["oil"]
        # knee: walk DOWN from the top; the knee is the lowest level still
        # within 1% of top-level oil before the first >1% drop or a gap
        knee_lvl = top_lvl
        knee_v = top_v
        for lvl, v in reversed(solved):
            if lvl >= knee_lvl:
                continue
            if v["oil"] >= 0.99 * oil_top and (knee_lvl - lvl) <= 100.0 + 1e-6:
                knee_lvl, knee_v = lvl, v
            else:
                break
        free = top_lvl - knee_lvl
        knee_summary.append(
            {
                "well": w,
                "top": top_lvl,
                "oil_top": oil_top,
                "knee": knee_lvl,
                "free": free,
                "sonic_top": top_v["sonic"],
                "sonic_knee": knee_v["sonic"],
            }
        )
        log(f"  KNEE: top {top_lvl:.0f} psi (oil {oil_top:.1f}, "
            f"{'SONIC' if top_v['sonic'] else 'subsonic'}) | "
            f"knee {knee_lvl:.0f} psi ({'SONIC' if knee_v['sonic'] else 'subsonic'}) | "
            f"FREE CHOKE {free:.0f} psi")

    log()
    log("SECTION 1 SUMMARY - knee ladder")
    log("  well    | top_psi | knee_psi | free_choke_psi | sonic@top | sonic@knee")
    for s in knee_summary:
        log(f"  {s['well']:<7} | {s['top']:7.0f} | {s['knee']:8.0f} | {s['free']:14.0f} | "
            f"{'yes' if s['sonic_top'] else 'NO ':<9} | {'yes' if s['sonic_knee'] else 'NO'}")
    n_sonic_top = sum(1 for s in knee_summary if s["sonic_top"])
    frees = [s["free"] for s in knee_summary]
    log(f"  sonic at full open: {n_sonic_top} of {len(knee_summary)} modelable wells")
    if frees:
        log(f"  free-choke margin psi: min {min(frees):.0f} / "
            f"median {statistics.median(frees):.0f} / max {max(frees):.0f}")
    full_window = [s["well"] for s in knee_summary if s["knee"] <= ladder[0] + 1e-6]
    log(f"  wells flat (within 1%) across the ENTIRE window: {len(full_window)} -> {full_window}")

    # ------------------------------------------------------------------
    # SECTION 2 - psu vs measured PIP gauges at today's settled header
    # ------------------------------------------------------------------
    log()
    log("=" * 78)
    log("SECTION 2 - MODEL PSU vs MEASURED PIP GAUGES (today's settled header)")
    log("=" * 78)

    pf_today = sum(float((test_rates.get(w) or (0, 0))[1] or 0.0) for w in names)
    header_today, over_cap = settled_header(
        plant, pf_today, plant.warm_start_psi(n_pumps), n_pumps
    )
    log(f"total test PF draw: {pf_today:.0f} bpd -> settled header today: "
        f"{header_today:.1f} psi (over_capacity={over_cap})")

    today = solve_at(header_today)

    log("  well    | model psu | gauge PIP | delta (model-gauge)")
    deltas = []
    flagged = []
    for num in sorted(GAUGE_PIP):
        w = cw_name(num)
        gauge = GAUGE_PIP[num]
        v = today.get(w)
        psu = v["psu"] if v else None
        if psu is None:
            log(f"  {w:<7} | {'-':>9} | {gauge:9.0f} | (no model solution)")
            continue
        d = psu - gauge
        deltas.append(d)
        if abs(d) > 150:
            flagged.append((w, d))
        log(f"  {w:<7} | {psu:9.0f} | {gauge:9.0f} | {d:+7.0f}")
    if deltas:
        mad = sum(abs(d) for d in deltas) / len(deltas)
        bias = sum(deltas) / len(deltas)
        log(f"  mean ABS delta: {mad:.0f} psi over {len(deltas)} gauged wells "
            f"(mean signed bias {bias:+.0f} psi)")
        log(f"  wells off by >150 psi: "
            f"{[f'{w} ({d:+.0f})' for w, d in flagged] if flagged else '(none)'}")

    # ------------------------------------------------------------------
    # SECTION 3 - Prosper drop-ladder comparison via ScaledPlant proxy
    # ------------------------------------------------------------------
    log()
    log("=" * 78)
    log("SECTION 3 - PROSPER POWER-FLUID DROP LADDER COMPARISON")
    log("=" * 78)

    def demand(P: float) -> float:
        return sum(v["pf"] for v in solve_at(P).values() if v)

    def runall_oil(P: float) -> float:
        return sum(v["oil"] for v in solve_at(P).values() if v)

    # settle the ALL-RUN header: budget_at_pressure(P) == demand(P), bisection
    lo, hi = p_lo, p_hi
    g_hi = plant.budget_at_pressure(hi, n_pumps) - demand(hi)
    g_lo = plant.budget_at_pressure(lo, n_pumps) - demand(lo)
    stamp(f"bisection bracket: g(floor {lo:.0f})={g_lo:+.0f} bpd, "
          f"g(ceil {hi:.0f})={g_hi:+.0f} bpd")
    if g_hi >= 0:
        our_normal = hi
        log(f"plant holds the window ceiling with every well full open -> normal = {hi:.0f} psi")
    elif g_lo <= 0:
        our_normal = lo
        log(f"WARNING: demand exceeds budget even at the floor -> normal = {lo:.0f} psi")
    else:
        a, b = lo, hi  # g(a) > 0 > g(b)
        while b - a > 1.0:
            m = 0.5 * (a + b)
            if plant.budget_at_pressure(m, n_pumps) - demand(m) > 0:
                a = m
            else:
                b = m
        our_normal = 0.5 * (a + b)

    oil_normal = runall_oil(our_normal)
    log(f"OUR normal: header {our_normal:.0f} psi, run-all pad oil {oil_normal:.0f} bopd")
    log(f"HIS normal: header {CW_NORMAL_HEADER:.0f} psi, run-all pad oil {CW_NORMAL_OIL:.0f} bopd")

    results = []
    for drop in range(100, 1001, 100):
        P_X = our_normal - drop
        log()
        stamp(f"--- drop {drop} psi -> forced header {P_X:.0f} psi ---")
        if P_X < p_lo - 1e-6:
            log(f"    SKIPPED: {P_X:.0f} psi is below the window floor ({p_lo:.0f} psi)")
            results.append({"drop": drop, "skipped": True})
            continue
        sol = solve_at(P_X)
        D_X = sum(v["pf"] for v in sol.values() if v)
        oil_X = sum(v["oil"] for v in sol.values() if v)
        budget_X = plant.budget_at_pressure(P_X, n_pumps)
        s_X = D_X / budget_X
        log(f"    run-all demand {D_X:.0f} bpd, run-all oil {oil_X:.0f} bopd, "
            f"true budget {budget_X:.0f} bpd -> degradation s = {s_X:.4f}")
        try:
            rows, meta = run_choke_optimization(
                configs, ScaledPlant(plant, s_X), n_pumps, current, test_rates,
                n_levels=19,
            )
            oil_after = float(meta["total_oil_bopd"])
            action = compact_actions(rows)
            gain = oil_after - oil_X
            results.append(
                {
                    "drop": drop, "skipped": False, "runall_oil": oil_X,
                    "oil_after": oil_after, "gain": gain, "action": action,
                    "best_header": float(meta["header_psi"]),
                    "n_shut": meta["n_shut"], "n_choked": meta["n_choked"],
                    "s": s_X,
                }
            )
            log(f"    optimizer: best header {meta['header_psi']:.0f} psi, "
                f"oil {oil_after:.0f} bopd (gain {gain:+.0f}), action: {action}")
        except Exception as exc:  # noqa: BLE001 - report and continue ladder
            log(f"    run_choke_optimization FAILED: {exc}")
            traceback.print_exc()
            results.append({"drop": drop, "skipped": False, "runall_oil": oil_X,
                            "error": str(exc)})

    log()
    log("SECTION 3 SUMMARY - drop ladder, ours vs Prosper")
    log("  drop | runall ours/his | our best action                | his action        | after ours/his | gain ours/his")
    shed_order: list[str] = []
    our_breakeven = None
    for r in results:
        drop = r["drop"]
        cw = CW_SENS.get(drop)
        if r.get("skipped"):
            log(f"  {drop:4d} | below window floor - skipped | his: {cw[1]}, "
                f"after {cw[2]}, gain +{cw[3]}")
            continue
        if "error" in r:
            log(f"  {drop:4d} | {r['runall_oil']:5.0f}/{cw[0]:5d} | "
                f"OPTIMIZER ERROR: {r['error'][:40]}")
            continue
        log(f"  {drop:4d} | {r['runall_oil']:5.0f}/{cw[0]:5d}     | {r['action']:<30} | "
            f"{cw[1]:<17} | {r['oil_after']:5.0f}/{cw[2]:5d}    | {r['gain']:+5.0f}/+{cw[3]}")
        if (r["n_shut"] + r["n_choked"]) > 0 and r["gain"] > 5:
            if our_breakeven is None:
                our_breakeven = drop
            for w in [t.split("@")[0].split()[-1] for t in r["action"].split("; ")
                      if t.startswith(("shut", "choke"))]:
                if w not in shed_order:
                    shed_order.append(w)

    log()
    log(f"  our break-even drop (first drop where an action beats run-all by >5 bopd): "
        f"{our_breakeven if our_breakeven is not None else 'NONE within 1000 psi'}")
    log("  his break-even: ~100 psi (first action at 100: shut M-24, +70)")
    log(f"  our shed/choke order (first appearance): "
        f"{shed_order if shed_order else '(no actions ever chosen)'}")
    log("  his shed order: M-24, M-22, M-43, M-45, M-62")

    stamp("DONE")


if __name__ == "__main__":
    main()
