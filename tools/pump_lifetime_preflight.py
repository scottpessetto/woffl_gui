"""Inventory historical pump/test overlap from a trusted local fleet snapshot.

This is data-readiness screening, not a lifetime model fit. A pickle must be a
locally captured, trusted artifact; do not supply an untrusted downloaded file.
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import pickle

import pandas as pd

from woffl.geometry import JetPump


ROOT = Path(__file__).resolve().parents[1]


def finite(value):
    try:
        value = float(value)
        return value if pd.notna(value) and abs(value) != float("inf") else None
    except (ValueError, TypeError):
        return None


def stamp(value):
    value = pd.to_datetime(value, errors="coerce", utc=True)
    return value.normalize().tz_localize(None) if pd.notna(value) else None


def text_value(value):
    return str(value) if pd.notna(value) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT/"build/fleet-actuality-snapshot.pkl")
    parser.add_argument("--output", type=Path, default=ROOT/"docs/pump_lifetime_preflight_v2_2026-09-11.json")
    args = parser.parse_args()
    snapshot = pickle.loads(args.snapshot.read_bytes())
    history = snapshot["frames"]["pump_history"].copy()
    history["start"] = history["Date Set"].map(stamp)
    tests = snapshot["frames"]["tests"].copy()
    tests["date"] = tests["WtDate"].map(stamp)
    asof = stamp(snapshot["captured_at"])
    report = dict(snapshot_time=snapshot["captured_at"], snapshot_sha256=sha256(args.snapshot.read_bytes()).hexdigest(),
        scope="Available-history readiness; no model fit or independent validation", live_queries=0,
        tests_from=str(tests.date.min().date()), tests_through=str(tests.date.max().date()),
        notes=["Tenure is Date Set to next Date Set; Date Pulled is never used. Installation days excluded for day-level observations.",
               "Historical tubing wall, pump depth, gauge datum, PVT and reservoir evolution still require reconciliation.",
               "User confirmed tracker diameters are nominal size specifications, never measured wear; catalog disagreements require reconciliation.",
               "User reports gauges typically within 40 ft of the JP; per-well signed offset remains unverified.",
               "Basic test screen requires positive oil/BHP, valid WC/GOR and measured operating pressures; PF rates above 20000 BPD flagged.",
               "Minimum three tests per era is a screening convenience, not sufficient qualification."], wells=[])
    for well in sorted(w["name"] for w in snapshot["universe"]["wells"]):
        hh = history[history["Well Name"] == well].dropna(subset=["start"]).sort_values("start")
        wt = tests[tests.well == well]
        starts = sorted(set(hh.start))
        duplicate_starts = set(hh.loc[hh.duplicated("start", keep=False), "start"])
        eras, assigned = [], set()
        for row in hh.to_dict("records"):
            start = row["start"]
            if start in duplicate_starts:
                continue
            later = [s for s in starts if s > start]
            end = min([asof, *(later[:1])])
            if end <= start:
                continue
            nozzle, throat = text_value(row["Nozzle Number"]), text_value(row["Throat Ratio"])
            direction = text_value(row["Circ Direction"])
            try:
                nominal = JetPump(nozzle, throat)
            except (ValueError, TypeError):
                continue
            if direction not in {"reverse", "forward"}:
                continue
            subset = wt[(wt.date > start) & (wt.date < end)]
            accepted, rejected = [], Counter()
            for test in subset.to_dict("records"):
                nums = {k: finite(test[k]) for k in ["BHP", "WtOilVol", "form_wc", "fgor", "pf_press", "whp", "lift_wat"]}
                if not (nums["BHP"] is not None and nums["BHP"] > 50 and nums["WtOilVol"] is not None and nums["WtOilVol"] > 0):
                    rejected["no positive measured BHP/oil"] += 1
                    continue
                if not (nums["form_wc"] is not None and 0 <= nums["form_wc"] < .99 and nums["fgor"] is not None and nums["fgor"] >= 0):
                    rejected["invalid composition"] += 1
                    continue
                if not (nums["pf_press"] is not None and 800 <= nums["pf_press"] <= 5500 and nums["whp"] is not None and 10 <= nums["whp"] <= 600):
                    rejected["no plausible operating pressure"] += 1
                    continue
                observed_direction = {"tubing": "forward", "annulus": "reverse"}.get(test["pf_source"])
                if observed_direction != direction:
                    rejected["test and tracker circulation disagree"] += 1
                    continue
                uid = str(test["wt_uid"])
                if uid in assigned:
                    raise AssertionError(f"test assigned twice: {well} {uid}")
                assigned.add(uid)
                accepted.append(dict(date=str(test["date"].date()), wt_uid=uid, bhp=nums["BHP"],
                    ppf=nums["pf_press"], pwh=nums["whp"], wc=nums["form_wc"], gor=nums["fgor"],
                    oil=nums["WtOilVol"], pf=nums["lift_wat"]))
            if not len(subset):
                continue
            raw_nozzle, raw_throat = finite(row["Nozzle Diameter"]), finite(row["Throat Diameter"])
            conflicts = []
            if raw_nozzle and abs(raw_nozzle-nominal.dnz) > .00011:
                conflicts.append("nozzle diameter differs from modeled label")
            if raw_throat and abs(raw_throat-nominal.dth) > .00011:
                conflicts.append("throat diameter differs from modeled label")
            eras.append(dict(start=str(start.date()), end_exclusive=str(end.date()), nozzle=nozzle,
                throat=throat, direction=direction, tubing_od=finite(row["Tubing Diameter"]),
                manufacturer=text_value(row["Manufacturer"]), converted=bool(row["Pump Converted"]),
                nominal_nozzle_diameter=nominal.dnz, nominal_throat_diameter=nominal.dth,
                tracker_nozzle_diameter=raw_nozzle, tracker_throat_diameter=raw_throat,
                geometry_flags=conflicts, raw_tests=len(subset), eligible_tests=len(accepted),
                pressure_span=max((r["ppf"] for r in accepted), default=0)-min((r["ppf"] for r in accepted), default=0),
                missing_pf=sum(r["pf"] is None or r["pf"] <= 0 for r in accepted),
                high_pf=sum(r["pf"] is not None and r["pf"] > 20000 for r in accepted),
                rejected=dict(rejected), tests=accepted))
        usable = [e for e in eras if e["eligible_tests"] >= 3]
        hardware = {(e["nozzle"], e["throat"], e["direction"]) for e in usable}
        report["wells"].append(dict(well=well, history_rows=len(hh), ambiguous_installation_dates=len(duplicate_starts),
            available_tests=len(wt), eligible_tests=sum(e["eligible_tests"] for e in eras),
            eras_with_three_tests=len(usable), distinct_hardware_with_three_tests=len(hardware),
            multi_pump_candidate=len(usable) >= 2 and len(hardware) >= 2, eras=eras))
    candidates = sorted([w for w in report["wells"] if w["multi_pump_candidate"]],
                         key=lambda w: (-w["distinct_hardware_with_three_tests"], -w["eligible_tests"]))
    report["candidate_wells"] = [w["well"] for w in candidates]
    report["summary"] = dict(app_wells=len(report["wells"]), historical_tests=len(tests),
        basic_eligible_tests=sum(w["eligible_tests"] for w in report["wells"]),
        multi_pump_candidates=len(candidates),
        supported_eras=sum(w["eras_with_three_tests"] for w in report["wells"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(report["summary"]))
    for w in candidates[:20]:
        print(w["well"], "eligible", w["eligible_tests"], "eras", w["eras_with_three_tests"],
              "hardware", w["distinct_hardware_with_three_tests"],
              [(e["start"], e["nozzle"]+e["throat"], e["eligible_tests"], round(e["pressure_span"])) for e in w["eras"] if e["eligible_tests"] >= 3])


if __name__ == "__main__":
    main()
