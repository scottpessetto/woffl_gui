"""Replay the pre-change fixture matrix against the shared production balance."""
import argparse
import json
from pathlib import Path
import warnings

from tools.critical_mach_study import CASES, solve_probe
from woffl.flow.entry_energy import MODEL_VERSION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("entry-energy-cases.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    previous = json.loads((root / "docs/critical_mach_study_2026-09-08.json").read_text())
    rows = []
    for before in previous["solves"]:
        if before["mode"] != "legacy":
            continue
        case, mach, ppf = before["case"], before["mach_crit"], before["ppf"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            try:
                after = solve_probe(CASES[case], "legacy", mach, ppf)
            except Exception as exc:
                after = {"error": f"{type(exc).__name__}: {exc}"}
        rows.append(dict(case=case, legacy_mach_input=mach, ppf=ppf, before=before, after=after))
    summary = dict(model_version=MODEL_VERSION, scope="offline fixtures, not field validation",
                   cases=len(rows), previous_errors=sum("error" in r["before"] for r in rows),
                   current_errors=sum("error" in r["after"] for r in rows))
    report = dict(summary=summary, rows=rows)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["current_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
