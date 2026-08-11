"""P3 validation harness: multi-point event calibration on live M-Pad data.

Per well: builder result (points/spread/refusal), fit old-vs-new parameters,
RMS residuals, implied beta vs the mined evidence beta, washout factor.

Run:
  cd C:/dev/woffl_gui/woffl_gui
  PYTHONPATH=. venv/Scripts/python.exe scripts/multipoint_validation.py
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed

T0 = time.time()


def log(msg: str = "") -> None:
    print(msg, flush=True)


def _fit_one(args):
    """Worker: fit one well. Module-level for spawn pickling."""
    wc, nozzle, throat, built = args
    from woffl.gui.fric_calibration import calibrate_multipoint

    t0 = time.time()
    res = calibrate_multipoint(wc, nozzle, throat, built)
    return wc.well_name, res, time.time() - t0


def main() -> None:
    log(f"[{time.time() - T0:5.0f}s] hydrating ...")
    from server.services.calibration_points import pad_points
    from server.services.evidence import pad_evidence
    from server.services.optimizer_runs import _build_configs, _current_and_tests

    notes: list = []
    prov: dict = {}
    configs = _build_configs(["M"], set(), [], notes, prov)
    by_name = {c.well_name: c for c in configs}
    names = list(by_name)
    current, _tests = _current_and_tests(names)

    res_map = {w: getattr(by_name[w], "res_pres", None) for w in names}
    surf_map = {w: getattr(by_name[w], "surf_pres", None) for w in names}

    log(f"[{time.time() - T0:5.0f}s] building points + evidence ...")
    built = pad_points(
        names,
        res_pres={w: v for w, v in res_map.items() if v},
        surf_pres={w: v for w, v in surf_map.items() if v},
    )
    ev = pad_evidence(names, res_pres={w: v for w, v in res_map.items() if v})

    jobs = []
    for w in names:
        b = built.get(w)
        if b is None:
            log(f"  {w}: no builder result (no data)")
            continue
        if b.get("refusal"):
            log(f"  {w}: REFUSED - {b['refusal']} (n_daily {b.get('n_daily')}, "
                f"n_test {b.get('n_test')}, spread {b.get('ppf_spread', 0):.0f})")
            continue
        cc = current.get(w)
        if not cc:
            log(f"  {w}: no current pump")
            continue
        jobs.append((by_name[w], cc[0], cc[1], b))

    log(f"[{time.time() - T0:5.0f}s] fitting {len(jobs)} wells (4 workers) ...")
    results = {}
    with ProcessPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(_fit_one, j): j[0].well_name for j in jobs}
        for fut in as_completed(futs):
            w = futs[fut]
            try:
                well, res, dt = fut.result()
                results[well] = res
                log(f"  [{time.time() - T0:5.0f}s] {well} fitted in {dt:.0f}s")
            except Exception as e:
                log(f"  [{time.time() - T0:5.0f}s] {w} FAILED: {type(e).__name__}: {e}")

    log()
    log("MULTI-POINT CALIBRATION - old (single-point) vs new (event) parameters")
    log("  well    | old ken/kth/kdi      | new ken/kth/kdi/fnz        | RMS bhp | PF%  | beta fit | beta mined | railed")
    for w in sorted(results):
        r = results[w]
        wc = by_name[w]
        old = f"{wc.ken_well or 0.03:.2f}/{wc.kth_well or 0.30:.2f}/{wc.kdi_well or 0.40:.2f}"
        if r.refusal:
            log(f"  {w:<7} | {old:<20} | REFUSED: {r.refusal}")
            continue
        new = (
            f"{r.best_ken:.3f}/{r.best_kth:.2f}/{r.best_kdi:.2f}/"
            f"{r.best_fnz:.2f}/mc{r.best_mach_crit:.2f}"
        )
        e = ev.get(w) or {}
        bm = e.get("beta")
        bf = r.implied_beta
        log(
            f"  {w:<7} | {old:<20} | {new:<30} | {r.rms_bhp_psi:7.1f} | {r.rms_pf_pct:4.1f} | "
            f"{bf if bf is not None else float('nan'):8.3f} | {bm if bm is not None else float('nan'):10.3f} | "
            f"{','.join(r.railed) or '-'}"
        )
    log()
    for w in sorted(results):
        r = results[w]
        if not r.refusal:
            log(f"  {w}: {r.message} (used {r.n_used}, dropped {r.n_dropped}, iters {r.iterations})")
    log(f"[{time.time() - T0:5.0f}s] DONE")


if __name__ == "__main__":
    main()
