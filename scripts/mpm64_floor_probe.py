"""One-off experiment: MPM-64 cavitation-floor audit vs field SCADA evidence.

SECTION 1 - MPM-64 modeled at the four SCADA-event PF headers vs measured
            (BHP, PF rate); does the model reproduce dPF but miss dBHP?
SECTION 2 - what moves the psu_min floor (jetflow.psu_minimize sensitivity:
            ken, ate, form_gor, form_wc, bubble point, IPR qwf)?
SECTION 3 - fleet audit: model floor psu at top ladder level vs each well's
            minimum measured test BHP over the last 12 months.

Run:
  cd C:/dev/woffl_gui/woffl_gui
  PYTHONPATH=. WOFFL_MAX_WORKERS=8 venv/Scripts/python.exe scripts/mpm64_floor_probe.py
"""

from __future__ import annotations

import time
import traceback

T0 = time.time()

TARGET = "MPM-64"

# Field evidence (PI Vision, MPM-64): date -> (BHP psi, PF rate bpd, PF press psi)
SCADA = [
    ("2025-11-18", 342.7, 5500.8, 3429.0),
    ("2025-12-06", 377.4, 5231.2, 3030.0),
    ("2025-12-20", 343.6, 5162.9, 3107.0),
    ("2026-08-10 (today)", 428.2, 4956.4, 3361.0),
]
NOV_BHP = 342.7  # November measured operating BHP; floor target for Section 2
VIOLATION_THRESH = 25.0  # psi


def log(msg: str = "") -> None:
    print(msg, flush=True)


def stamp(msg: str) -> None:
    log(f"[{time.time() - T0:7.0f}s] {msg}")


def fmt(v, w=7, d=1):
    if v is None:
        return " " * (w - 1) + "-"
    return f"{v:{w}.{d}f}"


def main() -> None:
    # -- hydration (mirrors server/services/optimizer_runs._run_pad_job) ----
    stamp("hydrating M-Pad configs / current pumps / tests / plant ...")

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

    stamp(f"hydrated {len(configs)} wells")
    log(f"hydration notes: {notes if notes else '(none)'}")

    wc64 = next((wc for wc in configs if wc.well_name == TARGET), None)
    if wc64 is None:
        raise SystemExit(f"{TARGET} not in hydrated configs: {sorted(names)}")
    cc64 = current.get(TARGET)
    if not cc64:
        raise SystemExit(f"{TARGET} has no current pump; cannot proceed")
    log(f"{TARGET} current pump: {cc64[0]}{cc64[1]} (expected 14B)")
    log(
        f"{TARGET} config: form_temp={wc64.form_temp}, form_wc={wc64.form_wc}, "
        f"form_gor={wc64.form_gor}, qwf={wc64.qwf}, pwf={wc64.pwf}, "
        f"res_pres={wc64.res_pres}, bubble_point={wc64.bubble_point}, "
        f"ken_well={wc64.ken_well}, knz_well={wc64.knz_well}, "
        f"kth_well={wc64.kth_well}, kdi_well={wc64.kdi_well}"
    )

    from woffl.assembly.network_optimizer import (
        NetworkOptimizer,
        PowerFluidConstraint,
    )
    from woffl.gui.scotts_tools._common import worker_ceiling
    from woffl.gui.pad_optimize import (
        _EVAL_CAP_FALLBACK_BPD,
        _RHO_PF_DEFAULT,
        _SCENARIO_MARGINAL_WC,
    )

    def solve_single(header_psi: float) -> dict | None:
        """Model MPM-64 alone at a forced header. Returns perf dict or None."""
        wc64.ppf_surf_well = header_psi
        pf = PowerFluidConstraint(
            total_rate=_EVAL_CAP_FALLBACK_BPD,
            pressure=header_psi,
            rho_pf=_RHO_PF_DEFAULT,
        )
        opt = NetworkOptimizer(
            [wc64], pf, [cc64[0]], [cc64[1]],
            marginal_watercut=_SCENARIO_MARGINAL_WC,
        )
        opt.run_all_batch_simulations(max_workers=1)
        perf = opt.get_pump_performance(TARGET, cc64[0], cc64[1])
        if not perf:
            return None
        return {
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

    # ------------------------------------------------------------------
    # SECTION 1 - MPM-64 model vs the SCADA events
    # ------------------------------------------------------------------
    log()
    log("=" * 78)
    log("SECTION 1 - MPM-64 MODEL AT SCADA-EVENT HEADERS vs MEASURED (BHP, PF rate)")
    log("=" * 78)
    log("CAVEAT: model uses TODAY's saved fit (IPR/GOR/WC/pump as of 2026-08-10);")
    log("        the first three SCADA events are from Nov-Dec 2025.")
    log()

    model_at: dict[float, dict | None] = {}
    for _date, _bhp, _pfrate, hdr in SCADA:
        if hdr not in model_at:
            stamp(f"solving {TARGET} at forced header {hdr:.0f} psi ...")
            try:
                model_at[hdr] = solve_single(hdr)
            except Exception:
                traceback.print_exc()
                model_at[hdr] = None

    log()
    log("  date                | hdr_psi | meas_BHP | meas_PF | mdl_oil | mdl_pf | mdl_psu | sonic (mach)")
    for date, bhp, pfrate, hdr in SCADA:
        v = model_at.get(hdr)
        if v is None:
            log(f"  {date:<19} | {hdr:7.0f} | {bhp:8.1f} | {pfrate:7.1f} |       - |      - |       - | unsolved")
        else:
            log(
                f"  {date:<19} | {hdr:7.0f} | {bhp:8.1f} | {pfrate:7.1f} | "
                f"{v['oil']:7.1f} | {v['pf']:6.0f} | {fmt(v['psu'])} | "
                f"{'SONIC' if v['sonic'] else 'sub  '} ({v['mach']:.2f})"
            )

    v_hi = model_at.get(3429.0)
    v_lo = model_at.get(3030.0)
    log()
    log("  3429 -> 3030 psi cut (-399 psi), measured 2025-11-18 -> 2025-12-06:")
    log(f"    measured: dPF = {5231.2 - 5500.8:+.1f} bpd ({(5231.2 - 5500.8) / 5500.8 * 100:+.1f}%), dBHP = {377.4 - 342.7:+.1f} psi")
    if v_hi and v_lo:
        dpf_m = v_lo["pf"] - v_hi["pf"]
        dpsu_m = (
            (v_lo["psu"] - v_hi["psu"])
            if (v_lo["psu"] is not None and v_hi["psu"] is not None)
            else None
        )
        doil_m = v_lo["oil"] - v_hi["oil"]
        log(
            f"    model:    dPF = {dpf_m:+.1f} bpd ({dpf_m / v_hi['pf'] * 100:+.1f}%), "
            f"dPSU = {fmt(dpsu_m, 6)} psi, dOIL = {doil_m:+.1f} bopd"
        )
        pf_ok = abs(dpf_m / v_hi["pf"] * 100 - (5231.2 - 5500.8) / 5500.8 * 100) < 2.5
        psu_flat = dpsu_m is not None and abs(dpsu_m) < 10.0
        log(
            f"    -> model {'REPRODUCES' if pf_ok else 'MISSES'} the PF-rate response; "
            f"model psu is {'FLAT (misses the +35 psi BHP response)' if psu_flat else 'NOT flat: ' + fmt(dpsu_m, 6) + ' psi'}"
        )
    else:
        log("    model deltas unavailable (unsolved endpoint)")

    # ------------------------------------------------------------------
    # SECTION 2 - what moves MPM-64's psu_min floor
    # ------------------------------------------------------------------
    log()
    log("=" * 78)
    log(f"SECTION 2 - psu_min FLOOR SENSITIVITY ({TARGET}, target <= {NOV_BHP:.0f} psi)")
    log("=" * 78)

    import woffl.flow.jetflow as jf
    from woffl.flow.inflow import InFlow
    from woffl.geometry.jetpump import JetPump
    from woffl.pvt.resmix import ResMix
    from woffl.gui.utils import create_pvt_components

    wellbore, wellprofile, inflow, res_mix, prop_pf = (
        NetworkOptimizer._create_well_objects(wc64)
    )

    jp_kwargs = {}
    if wc64.knz_well is not None:
        jp_kwargs["knz"] = wc64.knz_well
    if wc64.ken_well is not None:
        jp_kwargs["ken"] = wc64.ken_well
    if wc64.kth_well is not None:
        jp_kwargs["kth"] = wc64.kth_well
    if wc64.kdi_well is not None:
        jp_kwargs["kdi"] = wc64.kdi_well

    jp_base = JetPump(cc64[0], cc64[1], **jp_kwargs)
    tsu = wc64.form_temp
    pbp_eff = res_mix.oil.pbp  # effective bubble point actually in the model
    oil_qwf = wc64.qwf * (1 - wc64.form_wc)

    log(
        f"base pump {cc64[0]}{cc64[1]}: dnz={jp_base.dnz:.4f} in, dth={jp_base.dth:.4f} in, "
        f"ate={jp_base.ate * 144:.5f} in2, ken={jp_base.ken}"
    )
    log(
        f"base inputs: tsu={tsu} F, fgor={res_mix.fgor}, wc={res_mix.wc}, "
        f"pbp_eff={pbp_eff}, oil_qwf={oil_qwf:.1f} bopd, pwf={wc64.pwf}, pres={wc64.res_pres}"
    )
    log()

    def floor(ken=None, ate=None, ipr=None, mix=None, label="") -> float | None:
        try:
            psu_min, _q, _book = jf.psu_minimize(
                tsu=tsu,
                ken=jp_base.ken if ken is None else ken,
                ate=jp_base.ate if ate is None else ate,
                ipr_su=inflow if ipr is None else ipr,
                prop_su=res_mix if mix is None else mix,
            )
            return float(psu_min)
        except Exception as e:
            log(f"    ({label}: psu_minimize FAILED: {e})")
            return None

    def remix(fgor=None, wc=None, pbp=None) -> ResMix:
        oil, water, gas = create_pvt_components(
            field_model=wc64.field_model,
            oil_api=wc64.oil_api,
            gas_sg=wc64.gas_sg,
            wat_sg=wc64.wat_sg,
            bubble_point=pbp_eff if pbp is None else pbp,
        )
        return ResMix(
            wc=res_mix.wc if wc is None else wc,
            fgor=res_mix.fgor if fgor is None else fgor,
            oil=oil,
            wat=water,
            gas=gas,
        )

    stamp("computing baseline psu_min ...")
    base_floor = floor(label="baseline")
    log(f"BASELINE psu_min = {fmt(base_floor)} psi (expect ~430)")
    log()

    rows: list[tuple[str, float | None]] = []

    # ken
    rows.append((f"ken x2   ({jp_base.ken * 2:.3f})", floor(ken=jp_base.ken * 2, label="ken x2")))
    rows.append((f"ken x0.5 ({jp_base.ken * 0.5:.4f})", floor(ken=jp_base.ken * 0.5, label="ken x0.5")))

    # ate: one throat size larger (same nozzle, next ratio letter), and
    # nozzle one size larger with the SAME throat (washout: 15A has the same
    # throat index as 14B since throat_idx = nozzle_idx + area_code)
    ratio_letters = ["X", "A", "B", "C", "D", "E"]
    r_idx = ratio_letters.index(cc64[1])
    if r_idx + 1 < len(ratio_letters):
        jp_th = JetPump(cc64[0], ratio_letters[r_idx + 1], **jp_kwargs)
        rows.append(
            (
                f"throat +1 ({cc64[0]}{ratio_letters[r_idx + 1]}, ate={jp_th.ate * 144:.5f} in2)",
                floor(ate=jp_th.ate, label="throat+1"),
            )
        )
    noz_up = str(int(cc64[0]) + 1)
    if r_idx - 1 >= 0:
        jp_wash = JetPump(noz_up, ratio_letters[r_idx - 1], **jp_kwargs)
        rows.append(
            (
                f"nozzle +1 washout ({noz_up}{ratio_letters[r_idx - 1]}, same throat, "
                f"ate={jp_wash.ate * 144:.5f} in2)",
                floor(ate=jp_wash.ate, label="washout"),
            )
        )

    # form_gor
    for f in (0.5, 0.75, 1.25, 1.5):
        g = res_mix.fgor * f
        rows.append((f"fgor x{f} ({g:.0f})", floor(mix=remix(fgor=g), label=f"fgor x{f}")))

    # form_wc
    for dw in (-0.05, +0.05):
        w = min(max(res_mix.wc + dw, 0.0), 0.999)
        rows.append((f"wc {dw:+.2f} ({w:.3f})", floor(mix=remix(wc=w), label=f"wc {dw:+.2f}")))

    # bubble point
    for f in (0.8, 1.2):
        p = pbp_eff * f
        rows.append((f"pbp x{f} ({p:.0f})", floor(mix=remix(pbp=p), label=f"pbp x{f}")))

    # IPR qwf
    for f in (0.8, 1.2):
        ipr = InFlow(qwf=oil_qwf * f, pwf=wc64.pwf, pres=wc64.res_pres)
        rows.append((f"qwf x{f} ({oil_qwf * f:.0f} bopd)", floor(ipr=ipr, label=f"qwf x{f}")))

    log("  lever                                              | psu_min | d_vs_base | <=343?")
    for label, v in rows:
        if v is None:
            log(f"  {label:<50} |       - |         - | -")
        else:
            d = v - base_floor if base_floor is not None else float("nan")
            log(f"  {label:<50} | {v:7.1f} | {d:+9.1f} | {'YES' if v <= NOV_BHP + 0.5 else 'no'}")

    closers = [lbl for lbl, v in rows if v is not None and v <= NOV_BHP + 0.5]
    log()
    if closers:
        log(f"  levers reaching <= {NOV_BHP:.0f} psi single-handed: {closers}")
    else:
        log(f"  NO single lever reaches <= {NOV_BHP:.0f} psi.")

    # plausible 2-3 lever combinations of the strongest movers
    log()
    log("  plausible small combinations:")
    ipr08 = InFlow(qwf=oil_qwf * 0.8, pwf=wc64.pwf, pres=wc64.res_pres)
    combo_rows: list[tuple[str, float | None]] = []
    try:
        if r_idx + 1 < len(ratio_letters):
            jp_th = JetPump(cc64[0], ratio_letters[r_idx + 1], **jp_kwargs)
            combo_rows.append((
                "throat+1 + fgor x0.75",
                floor(ate=jp_th.ate, mix=remix(fgor=res_mix.fgor * 0.75), label="th1+g075"),
            ))
            combo_rows.append((
                "throat+1 + fgor x0.5",
                floor(ate=jp_th.ate, mix=remix(fgor=res_mix.fgor * 0.5), label="th1+g05"),
            ))
            combo_rows.append((
                "throat+1 + qwf x0.8",
                floor(ate=jp_th.ate, ipr=ipr08, label="th1+q08"),
            ))
            combo_rows.append((
                "throat+1 + fgor x0.75 + qwf x0.8",
                floor(ate=jp_th.ate, mix=remix(fgor=res_mix.fgor * 0.75), ipr=ipr08,
                      label="th1+g075+q08"),
            ))
        combo_rows.append((
            "fgor x0.75 + qwf x0.8",
            floor(mix=remix(fgor=res_mix.fgor * 0.75), ipr=ipr08, label="g075+q08"),
        ))
        if r_idx - 1 >= 0:
            jp_wash = JetPump(noz_up, ratio_letters[r_idx - 1], **jp_kwargs)
            combo_rows.append((
                "washout nozzle+1 + fgor x0.75",
                floor(ate=jp_wash.ate, mix=remix(fgor=res_mix.fgor * 0.75), label="wash+g075"),
            ))
            combo_rows.append((
                "washout nozzle+1 + fgor x0.5",
                floor(ate=jp_wash.ate, mix=remix(fgor=res_mix.fgor * 0.5), label="wash+g05"),
            ))
    except Exception:
        traceback.print_exc()
    for label, v in combo_rows:
        if v is None:
            log(f"    {label:<40} psu_min =      -")
        else:
            log(f"    {label:<40} psu_min = {v:6.1f}  {'<=343 YES' if v <= NOV_BHP + 0.5 else 'no'}")

    # ------------------------------------------------------------------
    # SECTION 3 - fleet floor-violation audit
    # ------------------------------------------------------------------
    log()
    log("=" * 78)
    log("SECTION 3 - M-PAD FLOOR vs MINIMUM MEASURED TEST BHP (last 12 months)")
    log("=" * 78)

    p_lo, p_hi = plant.pressure_window(n_pumps)
    top = round(p_hi, 1)
    log(f"pressure window ({n_pumps} pumps): {p_lo:.0f} - {p_hi:.0f} psi; auditing at top level {top:.0f}")

    stamp(f"solving full pad at top level {top:.0f} psi ...")
    pumps = [c for c in current.values() if c]
    nozzles = sorted({c[0] for c in pumps}) or ["12"]
    throats = sorted({c[1] for c in pumps}) or ["B"]
    for wc in configs:
        wc.ppf_surf_well = top
    pf = PowerFluidConstraint(
        total_rate=_EVAL_CAP_FALLBACK_BPD, pressure=top, rho_pf=_RHO_PF_DEFAULT
    )
    opt = NetworkOptimizer(
        configs, pf, nozzles, throats, marginal_watercut=_SCENARIO_MARGINAL_WC
    )
    opt.run_all_batch_simulations(max_workers=worker_ceiling())
    stamp("pad solve done; pulling test BHPs ...")

    from server.services.tests import tests_for_well

    log()
    log("  well    | pump | floor_psu | sonic@top | min_test_BHP | n_tests(BHP) | violation_psi")
    audit = []
    for w in sorted(names):
        cc = current.get(w)
        perf = opt.get_pump_performance(w, cc[0], cc[1]) if cc else None
        psu = (
            float(perf["suction_pressure"])
            if perf and perf.get("suction_pressure") is not None
            else None
        )
        sonic = bool(perf.get("sonic_status")) if perf else None

        min_bhp = None
        n_bhp = 0
        try:
            df = tests_for_well(w, 12)
            if df is not None and not df.empty and "BHP" in df.columns:
                bhps = df["BHP"].dropna()
                bhps = bhps[bhps > 0]
                n_bhp = int(len(bhps))
                if n_bhp:
                    min_bhp = float(bhps.min())
        except Exception as e:
            log(f"  ({w}: test fetch failed: {e})")

        viol = None
        if psu is not None and min_bhp is not None:
            viol = psu - min_bhp
        audit.append((w, cc, psu, sonic, min_bhp, n_bhp, viol))
        log(
            f"  {w:<7} | {cc[0] + cc[1] if cc else ' - ':>4} | {fmt(psu, 9)} | "
            f"{('yes' if sonic else 'NO ') if sonic is not None else ' - ':<9} | "
            f"{fmt(min_bhp, 12)} | {n_bhp:12d} | "
            f"{fmt(viol, 13) if viol is not None else '            -'}"
        )

    log()
    violators = [
        (w, psu, mb, v, s)
        for (w, cc, psu, s, mb, nb, v) in audit
        if v is not None and v > VIOLATION_THRESH
    ]
    log(f"SECTION 3 SUMMARY - wells whose modeled floor EXCEEDS min measured BHP by > {VIOLATION_THRESH:.0f} psi:")
    if violators:
        for w, psu, mb, v, s in sorted(violators, key=lambda r: -r[3]):
            log(
                f"  {w:<7} floor {psu:6.1f} vs min BHP {mb:6.1f} -> violation {v:+7.1f} psi "
                f"({'SONIC at top' if s else 'subsonic at top'})"
            )
        log(f"  -> {len(violators)} of {sum(1 for a in audit if a[6] is not None)} comparable wells contradict the modeled floor.")
    else:
        log("  (none - no floors contradicted by measurement)")

    log()
    stamp("DONE")


if __name__ == "__main__":
    main()
