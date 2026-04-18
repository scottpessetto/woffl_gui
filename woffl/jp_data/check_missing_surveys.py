"""Check for wells in Databricks vw_prop_mech that lack a deviation survey CSV.

Run this before deploying or after the data team adds new wells to flag any
that need surveys pulled from Oracle PDB via pull_surveys.py.

Usage:
    python -m woffl.jp_data.check_missing_surveys
"""

import os
import sys

from woffl.assembly.databricks_client import fetch_well_props


def main() -> int:
    survey_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "well_surveys")
    if not os.path.isdir(survey_dir):
        print(f"ERROR: survey directory not found: {survey_dir}", file=sys.stderr)
        return 2

    existing = {
        f.replace(" Deviation Survey.csv", "")
        for f in os.listdir(survey_dir)
        if f.endswith(" Deviation Survey.csv")
    }

    df = fetch_well_props()
    if df.empty:
        print("ERROR: vw_prop_mech returned no rows", file=sys.stderr)
        return 2

    db_wells = set(df["Well"].dropna().tolist())
    missing = sorted(db_wells - existing)
    extra = sorted(existing - db_wells)

    print(f"Databricks wells:  {len(db_wells)}")
    print(f"Local surveys:     {len(existing)}")
    print(f"Missing surveys:   {len(missing)}")
    if missing:
        for w in missing:
            print(f"  - {w}")
        print()
        print("To pull missing surveys, edit pull_surveys.py to query these wells")
        print("and run it from a machine with Oracle PDB access.")
    if extra:
        print(f"\nLocal-only (not in Databricks): {len(extra)}")
        for w in extra:
            print(f"  - {w}")

    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
