import os
from typing import Any, List, Tuple

import numpy as np
import oracledb
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import URL, create_engine, text

# Don't initialize Oracle Client - use thin mode instead
# oracledb.init_oracle_client()  # Commented out to use thin mode
load_dotenv()

well_list = [
    "MPB-28",
    "MPB-30",
    "MPB-32",
    "MPB-35",
    "MPB-37",
    "MPB-39",
    "MPC-23",
    "MPE-19",
    "MPE-24",
    "MPE-31",
    "MPE-35",
    "MPE-37",
    "MPE-41",
    "MPE-42",
    "MPS-05",
    "MPS-17",
    "MPS-45",
    "MPS-57",
    "MPG-16",
    "MPG-18",
    "MPH-19",
    "MPI-15",
    "MPI-17",
    "MPI-27",
    "MPI-29",
    "MPI-31",
    "MPI-33",
    "MPI-36",
    "MPI-40",
    "MPJ-27",
    "MPF-73",
    "MPF-107",
    "MPM-10",
    "MPM-12",
    "MPM-14",
    "MPM-16",
    "MPM-18",
    "MPM-20",
    "MPM-22",
    "MPM-24",
    "MPM-26",
    "MPM-28",
    "MPM-30",
    "MPM-32",
    "MPM-34",
    "MPM-43",
    "MPM-45",
    "MPM-60",
    "MPM-62",
    "MPM-64",
]

well_chars = pd.read_csv("jp_chars.csv")


engine = create_engine(
    "oracle+oracledb://:@",
    connect_args={"user": os.getenv("pdb_user"), "password": os.getenv("pdb_pw"), "dsn": "pdbfprd.world"},
)

pdb_conn = engine.connect()

query = "deviation_survey_pdb.sql"
with open(query, "r") as f:
    sql_text = f.read()


def tvd_interp(md: List[float], tvd: List[float], md_depth: float) -> float:
    """
    Return TVD interpolation given MD and TVD data and an MD depth using 1d interpolation

    Args:
        md (List[float]): List of measured depths (MD).
        tvd (List[float]): List of true vertical depths (TVD).
        md_depth (float): Single measured depth to return the equivalent TVD.

    Returns:
        float: TVD depth at the equivalent MD depth.
    """
    tvd_interp = np.interp(md_depth, md, tvd)

    return float(tvd_interp)


# Check which wells are missing survey files
wells_to_pull = []
for index, row in well_chars.iterrows():
    well_name = row["Well"]
    survey_file = f"well_surveys/{well_name} Deviation Survey.csv"

    if not os.path.exists(survey_file):
        wells_to_pull.append((index, well_name, row["JP_MD"]))
        print(f"Missing survey for: {well_name}")
    else:
        print(f"Survey exists for: {well_name} - skipping")

# Only pull surveys for wells that don't have files
if wells_to_pull:
    print(f"\nPulling surveys for {len(wells_to_pull)} wells...")

    for index, well_name, jp_md in wells_to_pull:
        print(f"Querying database for {well_name}...")
        survey = pd.read_sql_query(sql=text(sql_text), con=pdb_conn, params={"param": well_name})

        survey = survey.dropna()
        survey = survey.drop_duplicates()
        survey = survey.sort_values(by=["meas_depth"], ascending=True)

        survey.to_csv(f"well_surveys/{well_name} Deviation Survey.csv")
        print(f"  Saved survey for {well_name}")

        jp_tvd = tvd_interp(survey["meas_depth"].tolist(), survey["tvd_depth"].tolist(), jp_md)
        well_chars.at[index, "JP_TVD"] = jp_tvd
        print(f"  Calculated JP_TVD: {jp_tvd:.2f} ft")

    # Only save jp_chars.csv if we updated any TVD values
    well_chars.to_csv("jp_chars.csv")
    print(f"\nUpdated jp_chars.csv with {len(wells_to_pull)} new JP_TVD values")
else:
    print("\nAll wells already have survey files - nothing to pull")
