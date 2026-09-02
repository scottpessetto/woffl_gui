"""Pull deviation surveys from Oracle PDB for any wells in Databricks
vw_prop_mech that don't yet have a local CSV in well_surveys/.

Run from a machine with Oracle PDB network access (i.e. on Hilcorp VPN/LAN).
Requires .env with pdb_user, pdb_pw, plus Databricks creds for the diff.

Usage:
    python -m woffl.jp_data.pull_missing_surveys            # only wells with no CSV
    python -m woffl.jp_data.pull_missing_surveys --refresh  # re-pull every well

The query (deviation_survey_pdb.sql) takes the PREFERRED survey only; the
PDB view is a history view with every survey version per well.
"""

import os
import sys
from pathlib import Path

import oracledb
import pandas as pd
from dotenv import load_dotenv

from woffl.assembly.databricks_client import fetch_well_props


def main() -> int:
    load_dotenv()

    here = Path(__file__).resolve().parent
    survey_dir = here / "well_surveys"
    survey_dir.mkdir(exist_ok=True)
    sql_path = here / "deviation_survey_pdb.sql"
    sql_text = sql_path.read_text()

    df = fetch_well_props()
    if df.empty:
        print("ERROR: vw_prop_mech returned no rows", file=sys.stderr)
        return 2

    existing = {
        f.name.replace(" Deviation Survey.csv", "")
        for f in survey_dir.glob("*Deviation Survey.csv")
    }
    # --refresh re-pulls EVERY well (existing files included). Needed once on
    # 2026-09-02 after the query gained its PREFERRED_FLAG filter: the history
    # view stacks every survey version, and 5 of 91 local CSVs were unusable.
    refresh = "--refresh" in sys.argv
    missing = sorted(set(df["Well"].dropna()) - (set() if refresh else existing))

    if not missing:
        print("All wells already have surveys.")
        return 0

    print(f"Pulling {len(missing)} {'(refresh)' if refresh else 'missing'} surveys from Oracle PDB...")

    conn = oracledb.connect(
        user=os.getenv("pdb_user"),
        password=os.getenv("pdb_pw"),
        dsn="pdbfprd.world",
    )

    pulled, empty = [], []
    try:
        for well in missing:
            cur = conn.cursor()
            cur.execute(sql_text, {"param": well})
            cols = [d[0].lower() for d in cur.description]
            rows = cur.fetchall()
            cur.close()

            if not rows:
                print(f"  {well}: no survey rows returned")
                empty.append(well)
                continue

            survey = pd.DataFrame(rows, columns=cols)
            survey = survey.dropna(subset=["meas_depth", "tvd_depth"])
            survey = survey.drop_duplicates()
            survey = survey.sort_values("meas_depth")

            out = survey_dir / f"{well} Deviation Survey.csv"
            survey.to_csv(out, index=False)
            print(f"  {well}: {len(survey)} rows -> {out.name}")
            pulled.append(well)
    finally:
        conn.close()

    print(f"\nPulled: {len(pulled)} | Empty: {len(empty)}")
    if empty:
        print("Empty (no Oracle data):", ", ".join(empty))
    return 0


if __name__ == "__main__":
    sys.exit(main())
