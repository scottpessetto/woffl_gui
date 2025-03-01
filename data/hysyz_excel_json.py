"""
Take Hysys PVT data that is in excel and format it properly into a json that can
then be easily read by PVT tests for blackoil, formation gas and reservoir mixture
"""

import json
from pathlib import Path

import pandas as pd

# CODE FROM FORMGAS, USE TO MAKE JSON FILE

filename = Path(__file__).parents[0] / "oil_22_api_hysys_peng_rob.xlsx"

hys_df = pd.read_excel(filename, header=1)

print(hys_df)

hy_props = {"oil_api": 22, "bubblepoint": 1750, "gas_sg": 0.55, "temp_degf": 80}

rename = {"pressure": "pres_psig", "density": "rho_oil", "viscosity": "visc_oil"}

hys_df = hys_df.rename(columns=rename)
hy_dict = hys_df[["pres_psig", "rho_oil", "visc_oil"]].to_dict(orient="list")

print(hy_dict)

fin_dict = hy_props | hy_dict
print(fin_dict)

hysys_path = Path(__file__).parents[1] / "data" / "hysys_blackoil_peng_rob.json"
with open(hysys_path, "w") as json_file:
    json.dump(fin_dict, json_file, indent=4)
