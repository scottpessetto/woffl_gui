"""One-off Phase 5 validation: mined evidence table + choke plan before/after.

Run:
  cd C:/dev/woffl_gui/woffl_gui
  PYTHONPATH=. WOFFL_MAX_WORKERS=8 venv/Scripts/python.exe scripts/evidence_validation.py
"""

from __future__ import annotations

import time

T0 = time.time()


def log(msg: str = "") -> None:
    print(msg, flush=True)


def main() -> None:
    log(f"[{time.time() - T0:4.0f}s] hydrating ...")
    from server.services.evidence import pad_evidence
    from server.services.optimizer_runs import (
        _PAD_DEFAULTS,
        _build_configs,
        _current_and_tests,
        _pad_plant,
    )
    from woffl.gui.pad_optimize import run_choke_optimization

    notes: list = []
    prov: dict = {}
    configs = _build_configs(["M"], set(), [], notes, prov)
    names = [c.well_name for c in configs]
    current, test_rates = _current_and_tests(names)
    plant = _pad_plant("M")
    n_pumps = _PAD_DEFAULTS["M"]["n_pumps"]
    res_map = {c.well_name: getattr(c, "res_pres", None) for c in configs}

    log(f"[{time.time() - T0:4.0f}s] mining evidence ...")
    ev = pad_evidence(names, res_pres={w: r for w, r in res_map.items() if r})

    log()
    log("MINED EVIDENCE (12-month daily PF/BHP history)")
    log("  well    | floor | psu_ref | beta  | source  | days | pairs")
    for w in sorted(names):
        e = ev.get(w)
        if e is None:
            log(f"  {w:<7} |     - |       - |     - | no data |    - |    -")
        else:
            log(
                f"  {w:<7} | {e['floor'] or 0:5.0f} | {e['psu_ref'] or 0:7.0f} | "
                f"{e['beta'] if e['beta'] is not None else -1:5.3f} | {e['beta_source']:<7} | "
                f"{e['n_days']:4d} | {e['n_pairs']:5d}"
            )

    log()
    log(f"[{time.time() - T0:4.0f}s] running plan WITHOUT evidence ...")
    rows0, meta0 = run_choke_optimization(configs, plant, n_pumps, current, test_rates)
    log(f"[{time.time() - T0:4.0f}s] running plan WITH evidence ...")
    rows1, meta1 = run_choke_optimization(
        configs, plant, n_pumps, current, test_rates, evidence=ev
    )

    by0 = {r["well"]: r for r in rows0}
    by1 = {r["well"]: r for r in rows1}

    log()
    log(f"header {meta0['header_psi']:.0f} -> {meta1['header_psi']:.0f} psi | "
        f"total oil {meta0['total_oil_bopd']:.0f} -> {meta1['total_oil_bopd']:.0f} bopd | "
        f"corrected wells: {meta1['n_evidence_corrected']}")
    log()
    log("PER-WELL before -> after (action, set psi, oil, oil cost vs full, psu)")
    log("  well    | basis    | action        | set psi     | oil bopd    | dOil vs full | psu")
    for w in sorted(names):
        a, b = by0.get(w), by1.get(w)
        if a is None or b is None:
            continue
        log(
            f"  {w:<7} | {b['suction_basis']:<8} | "
            f"{a['action']:>5} -> {b['action']:<5} | "
            f"{(a['delivered_psi'] or 0):5.0f}->{(b['delivered_psi'] or 0):5.0f} | "
            f"{a['oil']:5.0f}->{b['oil']:5.0f} | "
            f"{a['d_oil_vs_full']:+6.1f}->{b['d_oil_vs_full']:+6.1f} | "
            f"{(a['psu'] or 0):4.0f}->{(b['psu'] or 0):4.0f}"
        )

    log()
    log("LADDER first 5 rungs, gain before -> after (bopd):")
    l0 = {r["drop_psi"]: r for r in meta0["ladder"]}
    l1 = {r["drop_psi"]: r for r in meta1["ladder"]}
    for d in sorted(l0)[:5]:
        r0, r1 = l0[d], l1.get(d)
        if r1:
            log(
                f"  drop {d:5.0f}: gain {r0['gain_bopd']:+7.0f} -> {r1['gain_bopd']:+7.0f} | "
                f"run-all {r0['run_all_oil_bopd']:7.0f} -> {r1['run_all_oil_bopd']:7.0f}"
            )

    # sanity checks
    log()
    same = [w for w in names if by0.get(w) and by1.get(w) and by1[w]["suction_basis"] == "model"
            and by0[w]["oil"] == by1[w]["oil"] and by0[w]["psu"] == by1[w]["psu"]]
    log(f"uncorrected wells with identical oil+psu: {len(same)} of "
        f"{sum(1 for w in names if by1.get(w) and by1[w]['suction_basis'] == 'model')}")
    m64 = by1.get("MPM-64")
    if m64:
        log(
            f"MPM-64: basis {m64['suction_basis']}, beta {m64['response_beta']}, "
            f"floor {m64['evidence_floor_psi']}, violation {m64['floor_violation_psi']}, "
            f"choke to {m64['delivered_psi']}, oil cost {m64['d_oil_vs_full']:.0f} bopd, "
            f"psu {m64['psu']:.0f} (full {m64['psu_full']:.0f})"
        )
    log(f"[{time.time() - T0:4.0f}s] DONE")


if __name__ == "__main__":
    main()
