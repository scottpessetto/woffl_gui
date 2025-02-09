import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from woffl.pvt import BlackOil, FormGas, FormWater, ResMix

# test python reservoir mixture equations vs hysys
# only works if the command python -m tests.rmix_test is used
# q_oil_std = 100  # bopd

temp = 80
oil_api = 22
pbub = 1750  # psig
gas_sg = 0.55  # methane
wc = 0.3  # watercut
fgor = 800  # scf/stb

# mass fraction, volm fraction, mixture density, speed of sound (no hysys...)

filepath = Path(__file__).parents[1] / "data" / "resmix_hysys_peng_rob.xlsx"
hy_df = pd.read_excel(filepath, header=1)

hysys_data = {
    "temp_degf": 80,
    "pres_psig": list(hy_df["press"]),
    "oil_api": 22,
    "pbub": 1750,
    "gas_sg": 0.55,
    "watercut": 0.3,
    "fgor": 800,
    "mass_fracs": {
        "oil": list(hy_df["oil_mfac"]),
        "wat": list(hy_df["wat_mfac"]),
        "gas": list(hy_df["gas_mfac"]),
    },
    "volm_fracs": {
        "oil": list(hy_df["oil_vfac"]),
        "wat": list(hy_df["wat_vfac"]),
        "gas": list(hy_df["gas_vfac"]),
    },
    "rho_mix": list(hy_df["rho_mix"]),
}

big_sosa = Path(__file__).parents[1] / "data" / "resmix_hysys_peng_rob.json"

with open(big_sosa, "w") as outfile:
    json.dump(hysys_data, outfile, indent=4)

prs_ray = hy_df["press"]
hy_mfac = [hy_df["oil_mfac"], hy_df["wat_mfac"], hy_df["gas_mfac"]]
hy_vfac = [hy_df["oil_vfac"], hy_df["wat_vfac"], hy_df["gas_vfac"]]
hy_rho_mix = hy_df["rho_mix"]


def compute_resmix_python(
    prs_ray: np.ndarray | pd.Series, temp: float, wc: float, fgor: float, oil_api: float, pbub: float, gas_sg: float
) -> tuple[list, list, list]:
    """Compute Reservoir Mixture

    Create a list of mass and volume fractions for Oil, Water and Gas
    in a mixture. Can be used to compare to results obtained with hysys
    """

    py_oil = BlackOil(oil_api=oil_api, bubblepoint=pbub, gas_sg=gas_sg)
    py_wat = FormWater(wat_sg=1)
    py_gas = FormGas(gas_sg=gas_sg)
    py_mix = ResMix(wc=wc, fgor=fgor, oil=py_oil, wat=py_wat, gas=py_gas)

    py_mfac = []
    py_vfac = []
    py_rho_mix = []

    for prs in prs_ray:
        py_mix = py_mix.condition(prs, temp)
        py_mfac.append(py_mix.mass_fract())
        py_vfac.append(py_mix.volm_fract())
        py_rho_mix.append(py_mix.rho_mix())

    return list(zip(*py_mfac)), list(zip(*py_vfac)), py_rho_mix


def plot_resmix_compare(prs_ray: np.ndarray | pd.Series):
    return None


plot_names = ["Oil", "Wat", "Gas"]

fig, axs = plt.subplots(4, sharex=True)

for i, hy in enumerate(hy_mfac):
    axs[0].scatter(prs_ray, hy, label=f"Hysys {plot_names[i]}")
    axs[0].scatter(prs_ray, py_mfac[i], marker="*", label=f"Python {plot_names[i]}")
axs[0].set_ylabel("Mass Fraction")
axs[0].legend()

for i, hy in enumerate(hy_vfac):
    axs[1].scatter(prs_ray, hy, label=f"Hysys {plot_names[i]}")
    axs[1].scatter(prs_ray, py_vfac[i], marker="*", label=f"Python {plot_names[i]}")
axs[1].set_ylabel("Volume Fraction")
axs[1].legend()

axs[2].scatter(prs_ray, hy_rho_mix, label="hysys")
axs[2].scatter(prs_ray, py_rho_mix, label="python")
axs[2].set_ylabel("Mixture Density, lbm/ft3")
axs[2].legend()

axs[3].scatter(prs_ray, py_cmix, label="python")
axs[3].set_ylabel("Speed of Sound, ft/s")
axs[3].set_xlabel("Pressure, psig")
axs[3].legend()

fig.suptitle(f"{py_mix}")

plt.show()
