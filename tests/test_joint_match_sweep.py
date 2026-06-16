"""Broad parameter sweep for joint_match — hunt for cases that SHOULD match but
don't, and measure per-well timing (the GUI button is interactive).

For each realistic (nozzle, throat, wc, qmax, ppf) combo we forward-solve a truth
well to get (oil, pf) targets, then ask joint_match to recover a match. Because a
solution provably exists (the truth params), anything not 'matched' is a real
optimizer weakness to investigate. Run:

    WOFFL_MAX_WORKERS=1 PYTHONPATH=. ./venv/Scripts/python.exe -m tests.test_joint_match_sweep
"""

import itertools
import os
import time

os.environ.setdefault("WOFFL_MAX_WORKERS", "1")

from woffl.gui.joint_match import _inflow_from_qmax, joint_match  # noqa: E402
from woffl.gui.utils import (  # noqa: E402
    create_jetpump,
    create_pipes,
    create_reservoir_mix,
    create_well_profile,
    run_jetpump_solver,
)

FIELD = "schrader"
SURF, TEMP, RHO_PF = 250.0, 120.0, 62.4
_, _, WELLBORE = create_pipes()
WELLPROF = create_well_profile(field_model=FIELD, jpump_tvd=5200.0)
PRES, GOR = 2000.0, 800.0

NOZZLES = ["11", "12", "13"]
THROATS = ["A", "B", "C"]
WCS = [0.70, 0.90]
QMAXES = [1000.0, 2000.0]
PPFS = [3000.0, 3600.0]


def _truth(qmax, ppf, nozzle, throat, wc):
    rm = create_reservoir_mix(wc, GOR, TEMP, FIELD)
    jp = create_jetpump(nozzle, throat, 0.03, 0.3, 0.4)
    out = run_jetpump_solver(SURF, TEMP, RHO_PF, ppf, jp, WELLBORE, WELLPROF,
                             _inflow_from_qmax(qmax, PRES), rm, field_model=FIELD,
                             quiet=True)
    if out is None:
        return None
    psu, sonic, qoil, _f, qnz, _m = out
    return {"oil": qoil, "pf": qnz, "bhp": psu, "sonic": sonic}


def main():
    combos = list(itertools.product(NOZZLES, THROATS, WCS, QMAXES, PPFS))
    n_match = n_partial = n_fail = n_skip = 0
    slow = []
    fails = []
    t0 = time.time()
    for nozzle, throat, wc, qmax, ppf in combos:
        t = _truth(qmax, ppf, nozzle, throat, wc)
        if t is None or t["oil"] <= 1 or t["pf"] <= 1:
            n_skip += 1
            continue
        ts = time.time()
        r = joint_match(
            oil_target=t["oil"], pf_target=t["pf"], pres=PRES, nozzle=nozzle,
            throat=throat, surf_pres=SURF, form_temp=TEMP, rho_pf=RHO_PF,
            ppf_surf0=3200.0, wellbore=WELLBORE, well_profile=WELLPROF,
            form_wc=wc, form_gor=GOR, field_model=FIELD,
        )
        dt = time.time() - ts
        if dt > 6:
            slow.append((f"{nozzle}{throat} wc{wc} q{qmax:.0f} pf{ppf:.0f}", dt,
                         r.iterations))
        if r.status == "matched":
            n_match += 1
        elif r.status == "partial":
            n_partial += 1
            fails.append((f"{nozzle}{throat} wc{wc:.2f} q{qmax:.0f} pf{ppf:.0f}",
                          t, r))
        else:
            n_fail += 1
            fails.append((f"{nozzle}{throat} wc{wc:.2f} q{qmax:.0f} pf{ppf:.0f}",
                          t, r))

    tested = n_match + n_partial + n_fail
    print(f"\n=== sweep: {len(combos)} combos, {n_skip} truth-unsolvable skipped, "
          f"{tested} tested in {time.time()-t0:.0f}s ===")
    print(f"  matched: {n_match}/{tested}   partial: {n_partial}   failed: {n_fail}")
    if slow:
        print(f"\n  SLOW (>6s) — {len(slow)}:")
        for name, dt, it in slow[:12]:
            print(f"    {name}: {dt:.1f}s ({it} iters)")
    if fails:
        print(f"\n  NON-MATCHES to investigate ({len(fails)}):")
        for name, t, r in fails[:25]:
            print(f"    {name}: truth oil={t['oil']:.0f} pf={t['pf']:.0f} "
                  f"sonic={t['sonic']} -> {r.status} "
                  f"oil_err={r.oil_err_pct:+.0f}% pf_err={r.pf_err_pct:+.0f}%")
            print(f"        {r.diagnostic}")
    else:
        print("\n  ALL solvable combos matched. ✔")


if __name__ == "__main__":
    main()
