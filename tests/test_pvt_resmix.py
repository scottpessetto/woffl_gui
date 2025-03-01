import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from woffl.pvt import BlackOil, FormGas, FormWater, ResMix

# run command python -m tests.rmix_test is used


def compute_resmix_data(
    prs_ray: np.ndarray | pd.Series, temp: float, wc: float, fgor: float, oil_api: float, pbub: float, gas_sg: float
) -> dict:
    """Compute Reservoir Mixture

    Create a list of mass and volume fractions for Oil, Water and Gas
    in a mixture. Can be used to compare to results obtained with hysys
    """

    py_oil = BlackOil(oil_api=oil_api, bubblepoint=pbub, gas_sg=gas_sg)
    py_wat = FormWater(wat_sg=1)
    py_gas = FormGas(gas_sg=gas_sg)
    py_mix = ResMix(wc=wc, fgor=fgor, oil=py_oil, wat=py_wat, gas=py_gas)

    mfac_oil, mfac_wat, mfac_gas = [], [], []
    vfac_oil, vfac_wat, vfac_gas = [], [], []
    rho_mix = []

    for prs in prs_ray:
        py_mix = py_mix.condition(prs, temp)

        mfac = py_mix.mass_fract()
        vfac = py_mix.volm_fract()

        mfac_oil.append(mfac[0])
        mfac_wat.append(mfac[1])
        mfac_gas.append(mfac[2])

        vfac_oil.append(vfac[0])
        vfac_wat.append(vfac[1])
        vfac_gas.append(vfac[2])

        rho_mix.append(py_mix.rho_mix())

    pymix = {
        "mass_fracs": {"oil": mfac_oil, "wat": mfac_wat, "gas": mfac_gas},
        "volm_fracs": {"oil": vfac_oil, "wat": vfac_wat, "gas": vfac_gas},
        "rho_mix": rho_mix,
    }
    return pymix


def plot_resmix_compare(hydict: dict, pydict: dict):
    """Plot Reservoir Mixture

    Compare the hysys generated with the python created mass and volm fractions
    Used for if the tests failed and you are trying to understand why they failed
    """
    cats = ["oil", "wat", "gas"]
    fig, axs = plt.subplots(3, sharex=True)
    axs = np.array(axs).flatten()
    for cat in cats:
        axs[0].scatter(hydict["pres_psig"], hydict["mass_fracs"][cat], label=f"Hy {cat.capitalize()}")
        axs[0].scatter(hydict["pres_psig"], pydict["mass_fracs"][cat], marker="*", label=f"Py {cat.capitalize()}")

        axs[1].scatter(hydict["pres_psig"], hydict["volm_fracs"][cat], label=f"Hy {cat.capitalize()}")
        axs[1].scatter(hydict["pres_psig"], pydict["volm_fracs"][cat], marker="*", label=f"Py {cat.capitalize()}")

    axs[0].set_ylabel("Mass Fraction")
    axs[0].legend()

    axs[1].set_ylabel("Volume Fraction")
    axs[1].legend()

    axs[2].scatter(hydict["pres_psig"], hydict["rho_mix"], label="hysys")
    axs[2].scatter(hydict["pres_psig"], pydict["rho_mix"], label="python")
    axs[2].set_ylabel("Mixture Density, lbm/ft3")
    axs[2].legend()
    plt.show()

    return None


hysys_path = Path(__file__).parents[1] / "data" / "hysys_resmix_peng_rob.json"
with open(hysys_path) as json_file:
    hymix = json.load(json_file)

pymix = compute_resmix_data(
    hymix["pres_psig"],
    hymix["temp_degf"],
    hymix["watercut"],
    hymix["fgor"],
    hymix["oil_api"],
    hymix["pbub"],
    hymix["gas_sg"],
)


def test_mass_fractions() -> None:
    name_frac = "mass_fracs"
    np.testing.assert_allclose(hymix[name_frac]["oil"], pymix[name_frac]["oil"], rtol=0.01)
    np.testing.assert_allclose(hymix[name_frac]["wat"], pymix[name_frac]["wat"], rtol=0.01)
    np.testing.assert_allclose(hymix[name_frac]["gas"], pymix[name_frac]["gas"], rtol=0.06)


def test_volm_fractions() -> None:
    name_frac = "volm_fracs"
    np.testing.assert_allclose(hymix[name_frac]["oil"], pymix[name_frac]["oil"], rtol=0.03)
    np.testing.assert_allclose(hymix[name_frac]["wat"], pymix[name_frac]["wat"], rtol=0.04)
    np.testing.assert_allclose(hymix[name_frac]["gas"], pymix[name_frac]["gas"], rtol=0.06)


def test_mixture_density() -> None:
    np.testing.assert_allclose(hymix["rho_mix"], pymix["rho_mix"], rtol=0.04)


if __name__ == "__main__":
    plot_resmix_compare(hymix, pymix)
