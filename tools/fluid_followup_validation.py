"""Compare fluid v2 against the saved v1 fixture matrix, without field reads."""
import argparse
import json
from pathlib import Path

from tools.critical_mach_study import CASES, solve_probe
from woffl.flow.entry_energy import MODEL_VERSION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("fluid-followup-cases.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / "docs/entry_energy_cases_2026-09-08.json"
    previous = json.loads(source.read_text(encoding="utf-8"))
    rows = []
    for record in previous["rows"]:
        if record["legacy_mach_input"] != 1.:
            continue
        case, ppf = record["case"], record["ppf"]
        after = solve_probe(CASES[case], "legacy", 1., ppf)
        rows.append(dict(case=case, ppf=ppf, before=record["after"], after=after))
    report = dict(physics_model=MODEL_VERSION, baseline=source.name,
                  scope="offline fixtures; no field qualification", rows=rows)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"{len(rows)} solves; max BHP change {max(abs(r['after']['psu']-r['before']['psu']) for r in rows):.6f} psi")


if __name__ == "__main__":
    main()
