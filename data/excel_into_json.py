"""
Take Hysys PVT data that is in excel and format it properly into a json that can
then be easily read by PVT tests for blackoil, formation gas and reservoir mixture
"""

import json
from pathlib import Path

import pandas as pd

filename = (
    Path(__file__).parents[3]
    / "Facility_Engineer"
    / "Jet Pump"
    / "subsurface_dimensions"
    / "dims_nozzle_throat_rev0.xlsx"
)

excel_file = pd.ExcelFile(filename)

sheet_names = excel_file.sheet_names

print(sheet_names)


# jetpump_path = Path(__file__).parents[1] / "data" / "jetpump_dimensions.json"
# with open(jetpump_path, "w") as json_file:
# json.dump(fin_dict, json_file, indent=4)
