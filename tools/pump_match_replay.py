"""Exercise the app replay using the trusted, locally captured September 8 snapshot.

No data access or saves. Do not load an untrusted pickle. The new output is
separate from the preserved September 11 benchmark.
"""
import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import pickle
import time

import pandas as pd

from server import schemas
from server.services import pump_match as pm
from server.services.optimizer_runs import _plain
from woffl.assembly.network_optimizer import WellConfig

ROOT = Path(__file__).resolve().parents[1]
WELLS = ("MPB-30", "MPB-37", "MPB-39", "MPF-107", "MPE-42", "MPF-73")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "build/fleet-actuality-snapshot.pkl")
    parser.add_argument("--output", type=Path, default=ROOT / "build/pump-match-replay-2026-09-12.json")
    parser.add_argument("--wells", nargs="+", default=WELLS)
    parser.add_argument("--mode", choices=("all_tests", "same_pump", "previous_pump"), default="previous_pump")
    parser.add_argument("--months", type=int, choices=(6, 12, 24, 60), default=24)
    parser.add_argument("--training-tests", type=int, default=10)
    args = parser.parse_args()
    snap = pickle.loads(args.snapshot.read_bytes())
    configs = snap["configs"]
    if isinstance(configs, list): configs = {c.well_name: c for c in configs}
    before = json.loads((ROOT / "docs/pump_history_benchmark_2026-09-11.json").read_text(encoding="utf-8"))
    reports = []
    started = time.perf_counter()
    for well in args.wells:
        cfg = deepcopy(configs[well])
        if isinstance(cfg, dict): cfg = WellConfig(**cfg)
        cfg.hydraulics_model = "beggs"
        tracker = snap["frames"]["pump_history"]
        tracker = tracker[tracker["Well Name"] == well].copy()
        tests = snap["frames"]["tests"]
        tests = tests[tests["well"] == well].copy()
        as_of = pd.to_datetime(snap["captured_at"], utc=True).isoformat()
        req = schemas.PumpMatchRequest(mode=args.mode, months=args.months, training_tests=args.training_tests)
        eras, rows, work, notes = pm.assemble(cfg, tracker, tests, req, as_of)
        for start in range(0, len(work), pm.CHUNK_SIZE):
            for index, values in pm.predict_chunk(work[start:start+pm.CHUNK_SIZE]):
                rows[index].update(values)
                rows[index]["status"] = "failed" if values.get("message") else rows[index]["phase"]
        for era in eras:
            own = [r for r in rows if r["installation_id"] == era["installation_id"]]
            era["prediction_scores"] = pm.scores([r for r in own if r["phase"] == "prediction"])
            era["fit_scores"] = pm.scores([r for r in own if r["phase"] == "fit"])
            era["replay_scores"] = pm.scores([r for r in own if r["phase"] == "replay"])
        common = []
        for old in before["challenges"]:
            if old["well"] != well: continue
            era = next((e for e in eras if e["date_set"][:10] == old["prediction_start"]), None)
            if not era or set(era.get("training_test_ids", [])) != {r["wt_uid"] for r in old["train"]}: continue
            for old_row in old["rows"]:
                new = next((r for r in rows if r["wt_uid"] == old_row["wt_uid"] and r["installation_id"] == era["installation_id"]), None)
                prediction = old_row["predictions"]["frozen_composition"]
                if new and new["status"] == "prediction" and "error" not in prediction:
                    common.append({q: abs(new[f"predicted_{q}"]-prediction[f"predicted_{q}"]) for q in ("bhp", "oil", "pf")})
        scored = pm.scores([r for r in rows if r["phase"] == ("replay" if args.mode == "all_tests" else "prediction")])
        report = dict(well=well, request=req.model_dump(), eras=eras, rows=rows, notes=notes, scores=scored,
                      common_identical_training=len(common), max_common_prediction_delta=max((v for d in common for v in d.values()), default=None))
        reports.append(report)
        print(json.dumps({k: report[k] for k in ("well", "scores", "common_identical_training", "max_common_prediction_delta")}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = dict(scope="App adapter replay of frozen historical data; not independent field validation", live_queries=0,
               snapshot_sha256=sha256(args.snapshot.read_bytes()).hexdigest(), reports=reports,
               elapsed_seconds=time.perf_counter()-started,
               source_sha256={name: sha256((ROOT/name).read_bytes()).hexdigest() for name in
                 ("server/services/pump_match.py", "server/services/field_validation.py", "woffl/assembly/solopump.py")})
    args.output.write_text(json.dumps(_plain(out), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"Saved {args.output}; {out['elapsed_seconds']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
