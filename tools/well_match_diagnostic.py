"""Offline fixed-IPR replay: isolate solver recovery from measured WC/GOR.

Uses the trusted local September 8 snapshot, never live queries or saves. The
legacy variant disables only the new interior-root fallback. All variants use
the same eligible tests, saved oil IPR, catalog pumps and reference losses.
"""
import argparse
from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import pickle
import time
from unittest.mock import patch

import numpy as np
import pandas as pd

from server import schemas
from server.services import pump_match as pm
from server.services.optimizer_runs import _plain
from woffl.assembly import solopump
from woffl.assembly.network_optimizer import NetworkOptimizer, WellConfig
from woffl.flow import jetflow
from woffl.geometry import JetPump

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("legacy_saved_composition", "interior_search_saved_composition", "interior_search_test_composition")
WELLS = ("MPB-28", "MPB-30", "MPB-37", "MPB-39", "MPF-107", "MPE-42", "MPF-73")


def compare(left, right):
    common = [(a, b) for a, b in zip(left, right) if a["status"] == b["status"] == "replay"]
    return dict(count=len(common), before=pm.scores([a for a, _ in common]),
                after=pm.scores([b for _, b in common]),
                max_prediction_delta=max((abs(a[f"predicted_{q}"] - b[f"predicted_{q}"])
                    for a, b in common for q in ("bhp", "oil", "pf")), default=None))


def root_trace(cfg, controls):
    bore, profile, ipr, mix, pf = NetworkOptimizer._create_well_objects(cfg)
    pump = JetPump(cfg.installed_nozzle, cfg.installed_throat, ken=.03, kth=.3, kdi=.4)
    args = (controls["pwh"], cfg.form_temp, controls["ppf"], pump, bore, profile, ipr, mix, pf, cfg.jpump_direction)
    lower, *_ = jetflow.psu_minimize(cfg.form_temp, pump.ken, pump.ate, ipr, mix)
    upper = cfg.res_pres - 10
    curve = []
    for suction in np.linspace(lower, upper, 129):
        try:
            residual = solopump.discharge_residual(suction, *args)[0]
        except (ValueError, jetflow.JetPumpError):
            residual = None
        curve.append(dict(suction=float(suction), residual=residual))
    root = solopump.jetpump_solver(*args)
    closed = solopump.discharge_residual(root[0], *args)[0]
    return dict(well=cfg.well_name, date=controls["date"], config=vars(cfg), controls=controls,
                curve=curve, root=root, residual_at_root=closed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "build/fleet-actuality-snapshot.pkl")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/well_match_diagnostic_2026-09-12.json")
    parser.add_argument("--wells", nargs="+", default=WELLS)
    args = parser.parse_args()
    snapshot_bytes = args.snapshot.read_bytes()
    snapshot = pickle.loads(snapshot_bytes)
    configs = snapshot["configs"]
    if isinstance(configs, list):
        configs = {c.well_name: c for c in configs}
    started, reports, trace = time.perf_counter(), [], None
    for well in args.wells:
        base = deepcopy(configs[well])
        if isinstance(base, dict):
            base = WellConfig(**base)
        base.hydraulics_model = "beggs"
        tracker = snapshot["frames"]["pump_history"]
        tracker = tracker[tracker["Well Name"] == well].copy()
        frame = snapshot["frames"]["tests"]
        frame = frame[frame["well"] == well].copy()
        request = schemas.PumpMatchRequest(months=12 if well == "MPB-30" else 24)
        eras, rows, tasks, notes = pm.assemble(base, tracker, frame, request, snapshot["captured_at"])
        variants = {}
        for variant in VARIANTS:
            work, results = deepcopy(tasks), deepcopy(rows)
            if variant != "interior_search_test_composition":
                for cfg, _, _ in work:
                    cfg.qwf, cfg.form_wc, cfg.form_gor = base.qwf, base.form_wc, base.form_gor
            legacy = patch.object(solopump, "_interior_lift_solution", return_value=None)
            with legacy if variant.startswith("legacy") else nullcontext():
                for start in range(0, len(work), pm.CHUNK_SIZE):
                    for index, values in pm.predict_chunk(work[start:start + pm.CHUNK_SIZE]):
                        results[index].update(values)
                        results[index]["status"] = "failed" if values.get("message") else "replay"
                for cfg, _, index in work:
                    results[index].update(input_wc=cfg.form_wc, input_gor=cfg.form_gor)
            variants[variant] = dict(scores=pm.scores(results), rows=results,
                failure_reasons=dict(Counter(r["message"] for r in results if r["status"] == "failed")))
            if well == "MPB-30" and variant == "interior_search_saved_composition":
                cfg, controls, _ = next(t for t in work if t[1]["date"].startswith("2026-09-02"))
                trace = root_trace(cfg, controls)
        prior, fixed, composition = [variants[v]["rows"] for v in VARIANTS]
        report = dict(well=well, request=request.model_dump(), saved_inputs=vars(base),
                      total_tests=len(rows), eligible_tests=len(tasks), notes=notes, variants=variants,
                      solver_comparison=compare(prior, fixed), composition_comparison=compare(fixed, composition),
                      tests_at_or_above_saved_reservoir_pressure=[r["wt_uid"] for r in rows
                          if r["bhp"] is not None and r["bhp"] >= base.res_pres])
        reports.append(report)
        print(json.dumps(dict(well=well, total=len(rows), scores={v: variants[v]["scores"] for v in VARIANTS},
            max_existing_delta=report["solver_comparison"]["max_prediction_delta"])), flush=True)
    out = dict(scope="Retrospective fixed oil IPR comparison, not independent prediction validation or a sizing qualification",
               live_queries=0, snapshot_sha256=sha256(snapshot_bytes).hexdigest(), captured_at=snapshot["captured_at"],
               reports=reports, root_trace=trace, elapsed_seconds=time.perf_counter() - started,
               source_sha256={name: sha256((ROOT / name).read_bytes()).hexdigest() for name in
                   ("tools/well_match_diagnostic.py", "server/services/pump_match.py", "woffl/assembly/solopump.py")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_plain(out), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if trace:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
        ax.plot([r["suction"] for r in trace["curve"]], [r["residual"] for r in trace["curve"]], color="#1565c0")
        ax.axhline(0, color="#607080", linewidth=1)
        ax.scatter([trace["root"][0]], [trace["residual_at_root"]], color="#e65100", zorder=3, label="Recovered pressure balance")
        ax.set(xlabel="Suction pressure (psig)", ylabel="Pump discharge minus required discharge (psi)",
               title="B-30, September 2: negative endpoints hide interior solutions\nSame saved IPR, fluids, catalog pump and equations")
        ax.legend(frameon=False)
        ax.grid(alpha=.2)
        fig.savefig(args.output.with_suffix(".png"), dpi=170)
        plt.close(fig)
    print(f"Saved {args.output}; {out['elapsed_seconds']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
