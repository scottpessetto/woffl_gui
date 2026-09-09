"""Read-only field-event validation, with reusable local snapshots.

Example: python tools/field_holdout_validation.py --live --wells MPM-64 MPM-28 MPM-45
Then rerun without --live to reuse the snapshot and avoid warehouse reads.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Fetch a read-only snapshot")
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--wells", nargs="+", default=["MPM-64", "MPM-28", "MPM-45"])
    parser.add_argument("--snapshot", type=Path, default=Path("build/field-validation-snapshot.json"))
    parser.add_argument("--output", type=Path, default=Path("build/field-holdout-report.json"))
    args = parser.parse_args()
    from server.services.field_validation import split_events, evaluate_events
    from woffl.assembly.network_optimizer import WellConfig
    from woffl.flow.entry_energy import MODEL_VERSION
    if args.live:
        from server.services import wells, calibration_points, optimizer_runs
        configs, errors = {}, {}
        for name in args.wells:
            print(f"Reading {name} configuration...", flush=True)
            try:
                context = wells.well_context(name)
                configs[name] = optimizer_runs._config_from_seeds(name, "", context["seeds"])
            except Exception as exc:
                errors[name] = f"{type(exc).__name__}: {exc}"
        print("Reading pressure events and tests...", flush=True)
        built = calibration_points.pad_points(list(configs),
            res_pres={n: c.res_pres for n, c in configs.items()},
            surf_pres={n: c.surf_pres for n, c in configs.items()}, include_full=True) if configs else {}
        snapshot = dict(captured_at=datetime.now(timezone.utc).isoformat(),
                        configs={n: asdict(c) for n, c in configs.items()}, data=built, errors=errors)
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        args.snapshot.write_text(json.dumps(optimizer_runs._plain(snapshot), indent=2, allow_nan=False), encoding="utf-8")
    else:
        snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    if args.fetch_only:
        print(f"Snapshot saved to {args.snapshot}", flush=True)
        return
    report = dict(physics_model=MODEL_VERSION, snapshot_time=snapshot["captured_at"],
                  field_validated=False, read_only=True, wells=[])
    for name in args.wells:
        print(f"Validating {name}...", flush=True)
        try:
            if name in snapshot.get("errors", {}):
                raise ValueError(snapshot["errors"][name])
            cfg = WellConfig(**snapshot["configs"][name])
            train, held, refusal = split_events(snapshot["data"].get(name, {}).get("validation_points", []))
            if refusal:
                result = dict(well=name, refusal=refusal, train_points=len(train), held_points=len(held))
            else:
                result = evaluate_events(cfg, train, held, progress=lambda m: print(f"{name}: {m}", flush=True))
            report["wells"].append(result)
        except Exception as exc:
            report["wells"].append(dict(well=name, error=f"{type(exc).__name__}: {exc}"))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print({k: v for k, v in report["wells"][-1].items()
               if k not in {"rows", "prediction_config", "fit_observations"}}, flush=True)


if __name__ == "__main__":
    main()
