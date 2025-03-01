import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from woffl.pvt.blackoil import BlackOil

# only works if the command python -m tests.boil_test is used


def compute_blackoil_data(
    prs_ray: np.ndarray | list, temp: float, oil_api: float, bubblepoint: float, gas_sg: float
) -> dict:
    """Compute BlackOil Data

    Create a list of properties of a formgas. Can be used to compare to results obtained with hysys.
    Density and oil viscosity.
    """
    py_oil = BlackOil(oil_api=oil_api, bubblepoint=bubblepoint, gas_sg=gas_sg)
    rho_oil, visc_oil = [], []

    for prs in prs_ray:
        py_gas = py_oil.condition(prs, temp)

        rho_oil.append(py_gas.density)
        visc_oil.append(py_gas.viscosity())

    pyoil = {"rho_oil": rho_oil, "visc_oil": visc_oil}
    return pyoil


def plot_blackoil_compare(hydict: dict, pydict: dict):
    """Plot Black Oil

    Compare hysys generated properties with the python created properties.
    Used for if the tests failed and you are trying to understand why they failed
    """

    fig, axs = plt.subplots(2, sharex=True)
    axs = np.array(axs).flatten()

    axs[0].scatter(hydict["pres_psig"], hydict["rho_oil"], label="hysys")
    axs[0].scatter(hydict["pres_psig"], pydict["rho_oil"], label="python")
    axs[0].set_ylabel("Density, lbm/ft3")
    axs[0].legend()

    axs[1].scatter(hydict["pres_psig"], hydict["visc_oil"], label="hysys")
    axs[1].scatter(hydict["pres_psig"], pydict["visc_oil"], label="python")
    axs[1].set_ylabel("Viscosity, cP")
    axs[1].legend()

    fig.suptitle(f"{hydict['oil_api']}\u00b0 API Oil Properties at {hydict['temp_degf']}\u00b0 F")
    plt.show()
    return None


# read in hysys data from json
hysys_path = Path(__file__).parents[1] / "data" / "hysys_blackoil_peng_rob.json"
with open(hysys_path) as json_file:
    hyprop = json.load(json_file)

# generate python comparison data
pyprop = compute_blackoil_data(
    hyprop["pres_psig"], hyprop["temp_degf"], hyprop["oil_api"], hyprop["bubblepoint"], hyprop["gas_sg"]
)


py_boil = BlackOil.test_oil()
temp_degf = 80
pres_psig = 2500
py_boil.condition(pres_psig, temp_degf)
print(f"Oil Surface Tension: {round(py_boil.tension() / 0.0000685, 2)} dyne/cm")


if __name__ == "__main__":
    plot_blackoil_compare(hyprop, pyprop)
