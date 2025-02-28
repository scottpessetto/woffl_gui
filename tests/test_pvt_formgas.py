import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from woffl.pvt.formgas import FormGas

# only works if the command python -m tests.test_pvt_formgas


def compute_formgas_data(prs_ray: np.ndarray | list, temp: float, gas_sg: float) -> dict:
    """Compute FormGas Data

    Create a list of properties of a formgas. Can be used to compare to results obtained with hysys.
    Density, viscosity and zfactor.
    """
    py_gas = FormGas(gas_sg=gas_sg)
    rho_gas, visc_gas, zfactor = [], [], []

    for prs in prs_ray:
        py_gas = py_gas.condition(prs, temp)

        rho_gas.append(py_gas.density)
        visc_gas.append(py_gas.viscosity())
        zfactor.append(py_gas.zfactor())

    pygas = {"rho_gas": rho_gas, "visc_gas": visc_gas, "zfactor": zfactor}
    return pygas


def plot_formgas_compare(hydict: dict, pydict: dict):
    """Plot Formation Gas

    Compare hysys generated properties with the python created properties.
    Used for if the tests failed and you are trying to understand why they failed
    """

    fig, axs = plt.subplots(3, sharex=True)
    axs = np.array(axs).flatten()

    axs[0].scatter(hydict["pres_psig"], hydict["rho_gas"], label="hysys")
    axs[0].scatter(hydict["pres_psig"], pydict["rho_gas"], label="python")
    axs[0].set_ylabel("Density, lbm/ft3")
    axs[0].legend()

    axs[1].scatter(hydict["pres_psig"], hydict["visc_gas"], label="hysys")
    axs[1].scatter(hydict["pres_psig"], pydict["visc_gas"], label="python")
    axs[1].set_ylabel("Viscosity, cP")
    axs[1].legend()

    axs[2].scatter(hydict["pres_psig"], hydict["zfactor"], label="hysys")
    axs[2].scatter(hydict["pres_psig"], pydict["zfactor"], label="python")
    axs[2].set_ylabel("Z-Factor, Unitless")
    axs[2].legend()

    fig.suptitle(f"{hydict['gas_sg']} Gas Properties at {hydict['temp_degf']}\u00b0 F")
    plt.show()
    return None


# blasingame 1988 PDF
blasing_ppr = 2.958
blasing_tpr = 1.867
blasing_zf = 0.9117

# read in hysys data from json
hysys_path = Path(__file__).parents[1] / "data" / "hysys_formgas_peng_rob.json"
with open(hysys_path) as json_file:
    hygas = json.load(json_file)

# generate python comparison data
pygas = compute_formgas_data(hygas["pres_psig"], hygas["temp_degf"], hygas["gas_sg"])


def test_zfactor_gradschool() -> None:
    grad_zf = FormGas._zfactor_grad_school(blasing_ppr, blasing_tpr)
    assert grad_zf == pytest.approx(blasing_zf, rel=0.01)


def test_zfactor_dranchuk() -> None:
    dak_zf = FormGas._zfactor_dak(blasing_ppr, blasing_tpr)
    assert dak_zf == pytest.approx(blasing_zf, rel=0.01)


def test_gas_compressibility() -> None:
    # properties of petroleum fluids, 2nd Edition, McCain, Pag 174
    pres_mccain = 1000 + 14.7  # psia + atm
    temp_mccain = 68
    mccain_compress = 0.001120
    methane = FormGas(gas_sg=0.55)
    calc_compress = methane.condition(pres_mccain, temp_mccain).compress()
    assert calc_compress == pytest.approx(mccain_compress, rel=0.03)


def test_gas_density() -> None:
    np.testing.assert_allclose(hygas["rho_gas"], pygas["rho_gas"], rtol=0.05)


def test_gas_viscosity() -> None:
    np.testing.assert_allclose(hygas["visc_gas"], pygas["visc_gas"], rtol=0.03)


def test_gas_zfactor() -> None:
    np.testing.assert_allclose(hygas["zfactor"], pygas["zfactor"], rtol=0.04)


if __name__ == "__main__":
    plot_formgas_compare(hygas, pygas)
