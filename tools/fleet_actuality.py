"""Read-only fleet snapshot and reproducible model-versus-observation audit.

python tools/fleet_actuality.py --live --fetch-only
python tools/fleet_actuality.py
Raw snapshots are local, trusted pickle files; do not load third-party snapshots.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import os
from pathlib import Path
import pickle
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]


def capture(path):
    from woffl.assembly import databricks_client as db
    query_stats = []
    original = db.execute_query
    def counted(sql):
        started = perf_counter()
        result = original(sql)
        query_stats.append(dict(seconds=perf_counter()-started, rows=len(result)))
        return result
    db.execute_query = counted
    from server.services import calibration_points, datasources, evidence, ipr, tests, wells, optimizer_runs
    from woffl.assembly.well_test_client import _normalize_well_name
    from woffl.gui.ipr_anchor import load_saved_ipr

    snapshot = dict(captured_at=datetime.now(timezone.utc).isoformat(), frames={}, contexts={}, configs={}, errors={}, saved={})
    def take(name, function):
        print(f"Reading {name}...", flush=True)
        result = function()
        snapshot["frames"][name] = result
        print(f"  {len(result)} rows", flush=True)
        return result

    take("gauges", lambda: db.execute_query("""SELECT g.* FROM mpu.wells.vw_bhp_tags g
        WHERE EXISTS (SELECT 1 FROM mpu.wells.vw_well_header h WHERE h.enthid=g.enthid AND h.field='MPU')"""))
    take("bhp", lambda: db.execute_query("""SELECT b.tag_date, b.enthid, b.well_name, b.bhp_source,
        b.bhp_cln_value, b.bhp_esp_value, b.bhp_other_value FROM mpu.wells.vw_bhp_daily_clean b
        WHERE b.tag_date >= date_sub(current_date(),365) AND b.tag_date < current_date()
        AND EXISTS (SELECT 1 FROM mpu.wells.vw_well_header h WHERE h.enthid=b.enthid AND h.field='MPU')"""))
    take("tests", lambda: tests.fetch_all_well_tests(24))
    take("pressure", evidence._fleet_pressure_daily)
    take("pf_volume", calibration_points._fleet_pf_volume)
    take("chars", lambda: datasources.well_chars()[0])
    take("pump_history", datasources._jp_history_databricks)
    take("pf_latest", datasources.pf_latest)
    ipr._saved_ipr_snapshot()
    universe = wells.list_wells()
    if universe["source"] != "databricks":
        raise RuntimeError("current Databricks well universe unavailable")
    snapshot["universe"] = universe
    registered = {_normalize_well_name(w) for w in snapshot["frames"]["gauges"]["well_name"]}
    names = sorted(w["name"] for w in universe["wells"] if w["name"] in registered)
    for i, name in enumerate(names, 1):
        print(f"Hydrating {i}/{len(names)}: {name}", flush=True)
        try:
            context = wells.well_context(name, 6, 0)
            snapshot["contexts"][name] = context
            seeds = dict(context["seeds"])
            if context.get("jpump_md") is not None:
                seeds["jpump_md"] = context["jpump_md"]
            cfg = optimizer_runs._config_from_seeds(name, name.split('-')[0].removeprefix('MP'), seeds)
            snapshot["configs"][name] = asdict(cfg)
            snapshot["saved"][name] = load_saved_ipr(name)
        except Exception as exc:
            snapshot["errors"][name] = f"{type(exc).__name__}: {exc}"
    snapshot["query_stats"] = query_stats
    snapshot["captured_until"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(snapshot, protocol=5))
    print(f"Snapshot complete: {len(names)} registered-gauge app wells, {len(snapshot['configs'])} models, "
          f"{len(query_stats)} SELECT queries. Saved {path}", flush=True)
    return snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--snapshot", type=Path, default=ROOT/"build/fleet-actuality-snapshot.pkl")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("WOFFL_MAX_WORKERS", "2")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    snapshot = capture(args.snapshot) if args.live else pickle.loads(args.snapshot.read_bytes())
    if args.fetch_only:
        return
    from server.services.fleet_validation import run_audit, write_report
    from hashlib import sha256
    report = run_audit(snapshot, progress=lambda m: print(m, flush=True))
    report["snapshot_sha256"] = sha256(args.snapshot.read_bytes()).hexdigest()
    write_report(report, args.output or ROOT/"docs"/f"fleet_actuality_{snapshot['captured_at'][:10]}")


if __name__ == "__main__":
    main()
