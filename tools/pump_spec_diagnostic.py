"""Audit nominal tracker specs against model catalogs and optionally probe physics.

Offline only. Tracker dimensions are specifications, never measured wear
(user clarification, 2026-09-11). Conflicting catalogs are investigated,
not silently installed as new app defaults. The pickle must be trusted.
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import pickle

import pandas as pd

from tools.bhp_model_diagnostic import ROOT, config_for, objects, scored
from woffl.assembly import solopump
from woffl.geometry import JetPump


def finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def matches(catalog, part, diameter):
    if diameter is None:
        return []
    return [f"{family} {size}" for family, parts in catalog.items()
            for size, value in parts.get(part, {}).items() if abs(value-diameter) <= .00011]


def replay(well, variant, dnz, dth):
    try:
        cfg = config_for(well, variant)
        pump, bore, profile, ipr, mix, pf = objects(cfg)
        pump.dnz = dnz * math.sqrt(cfg.fnz_well or 1.)
        pump.dth = dth
        if pump.ate <= 0:
            return {"error": "nominal specs plus frozen area factor leave no entry area"}
        obs = well["latest_test"]
        return scored(solopump.jetpump_solver(obs["pwh"], cfg.form_temp, obs["ppf"],
            pump, bore, profile, ipr, mix, pf, cfg.jpump_direction), obs)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT/"docs/pump_spec_diagnostic_2026-09-11.json")
    args = parser.parse_args()
    snapshot_path = ROOT/"build/fleet-actuality-snapshot.pkl"
    snapshot = pickle.loads(snapshot_path.read_bytes())
    history = snapshot["frames"]["pump_history"].copy()
    history["Date Set"] = pd.to_datetime(history["Date Set"], utc=True, errors="coerce")
    catalog = json.loads((ROOT/"data/jetpump_dimensions.json").read_text(encoding="utf-8"))
    frozen = json.loads((ROOT/"docs/fleet_actuality_2026-09-08.json").read_text(encoding="utf-8"))
    previous = json.loads((ROOT/"docs/bhp_model_diagnostic_2026-09-11.json").read_text(encoding="utf-8"))
    previous = {w["well"]: w["results"] for w in previous["wells"]}
    wells = []
    for w in frozen["wells"]:
        if not w.get("latest_test"):
            continue
        rows = history[history["Well Name"] == w["well"]].dropna(subset=["Date Set"])
        rows = rows[rows["Date Set"] == rows["Date Set"].max()]
        if len(rows) != 1:
            wells.append({"well": w["well"], "error": "ambiguous latest installation"})
            continue
        row = rows.iloc[0]
        cfg = w["config"]
        pump = JetPump(cfg["installed_nozzle"], cfg["installed_throat"])
        dnz, dth = finite(row.get("Nozzle Diameter")), finite(row.get("Throat Diameter"))
        conflict = dnz is not None and dth is not None and (
            abs(dnz-pump.dnz) > .00011 or abs(dth-pump.dth) > .00011)
        record = dict(well=w["well"], manufacturer=str(row.get("Manufacturer")),
            installed_at=row["Date Set"].isoformat(), pump=f"{pump.noz_no}{pump.rat_ar}",
            converted=bool(row.get("Pump Converted", False)),
            modeled_nominal_nozzle=pump.dnz, modeled_nominal_throat=pump.dth,
            tracker_nominal_nozzle=dnz, tracker_nominal_throat=dth,
            nozzle_catalog_matches=matches(catalog, "nozzle", dnz),
            throat_catalog_matches=matches(catalog, "throat", dth), spec_conflict=bool(conflict),
            nozzle_area_difference_pct=100*((dnz/pump.dnz)**2-1) if dnz else None,
            throat_area_difference_pct=100*((dth/pump.dth)**2-1) if dth else None,
            results={})
        if args.simulate and conflict and dnz > .04 and dth > dnz:
            for variant in ("baseline", "test_state"):
                record["results"][f"model_catalog_{variant}"] = previous[w["well"]][variant]
                record["results"][f"tracker_specs_{variant}"] = replay(w, variant, dnz, dth)
        wells.append(record)
        print(f"{w['well']}: {record['pump']}, spec conflict={conflict}", flush=True)
    report = dict(scope="Nominal-spec conflict audit and hypothetical geometry probe; no refit or app changes; not independent validation",
        user_clarifications={"tracker_dimensions": "Nominal specifications based on pump size; never physically measured",
                             "gauge_position": "Typically within 40 ft of the jet pump"},
        snapshot_time=snapshot["captured_at"], snapshot_sha256=sha256(snapshot_path.read_bytes()).hexdigest(),
        live_queries=0, simulated=args.simulate,
        summary=dict(wells=len(wells), spec_conflicts=sum(w.get("spec_conflict", False) for w in wells),
                     conflicts_by_manufacturer=dict(Counter(w["manufacturer"] for w in wells if w.get("spec_conflict")))),
        wells=wells)
    report["code_sha256"] = {rel: sha256((ROOT/rel).read_bytes()).hexdigest() for rel in
        ("tools/pump_spec_diagnostic.py", "tools/bhp_model_diagnostic.py", "data/jetpump_dimensions.json",
         "woffl/geometry/jetpump.py", "woffl/assembly/solopump.py")}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
