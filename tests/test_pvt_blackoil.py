import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from woffl.pvt.blackoil import BlackOil

# only works if the command python -m tests.boil_test is used


def compute_blackoil_data(
    prs_ray: np.ndarray | list,
    temp: float,
    oil_api: float,
    bubblepoint: float,
    gas_sg: float,
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
        visc_oil.append(py_gas.viscosity)

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

    fig.suptitle(
        f"{hydict['oil_api']}\u00b0 API Oil Properties at {hydict['temp_degf']}\u00b0 F"
    )
    plt.show()
    return None


# read in hysys data from json
hysys_path = Path(__file__).parents[1] / "data" / "hysys_blackoil_peng_rob.json"
with open(hysys_path) as json_file:
    hyprop = json.load(json_file)

# generate python comparison data
pyprop = compute_blackoil_data(
    hyprop["pres_psig"],
    hyprop["temp_degf"],
    hyprop["oil_api"],
    hyprop["bubblepoint"],
    hyprop["gas_sg"],
)


def test_oil_density() -> None:
    np.testing.assert_allclose(hyprop["rho_oil"], pyprop["rho_oil"], rtol=0.05)


def test_oil_viscosity() -> None:
    # 75% error, why are we even testing...haha
    np.testing.assert_allclose(hyprop["visc_oil"], pyprop["visc_oil"], rtol=0.75)


# singular propertiest, need to find something to test these...book example....
temp_degf = 80
pres_psig = 2500
py_boil = BlackOil.test_oil()
py_boil.condition(pres_psig, temp_degf)


def test_oil_tension() -> None:
    # try to find where this example is
    assert py_boil.tension / 0.0000685 == pytest.approx(16.04, rel=0.01)  # dyne/cm


def test_oil_compressibility_above() -> None:
    oil = BlackOil.test_oil()
    oil.condition(2500, 80)
    assert oil.compress == pytest.approx(2.7953e-06, rel=0.01)


def test_oil_compressibility_below() -> None:
    # McCain SPE-15664 Eq. 5 takes temperature in deg R; an earlier expected
    # value (2.5762e-05) locked in a deg F implementation bug (fixed 2026-06).
    # The correlation is defined with Rsb — solution GOR AT THE BUBBLE POINT —
    # not Rs at the current pressure; passing Rsb (fixed 2026-06, library patch)
    # raised the previous 2.16e-4 (Rs(p)) to 2.40e-4 psi^-1, both within the
    # range McCain's paper reports. Below-bubblepoint co is 1-2 orders of
    # magnitude above the above-pbp value because of gas coming out of solution.
    # Tripwire for the Rsb library patch — reverts to ~2.16e-4 if it's lost.
    oil = BlackOil.test_oil()
    oil.condition(1000, 80)
    assert oil.compress == pytest.approx(2.3968e-04, rel=0.01)


def test_validation_bounds_inclusive() -> None:
    # Boundary inputs must be ACCEPTED (docstrings say inclusive "10 to 40",
    # etc). Tripwire for the inclusive-bounds library patch — reverts to raising
    # if the strict < comes back.
    BlackOil(oil_api=10, bubblepoint=1000, gas_sg=0.5)
    BlackOil(oil_api=40, bubblepoint=3000, gas_sg=1.2)


def test_validation_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        BlackOil(oil_api=9, bubblepoint=1750, gas_sg=0.65)
    with pytest.raises(ValueError):
        BlackOil(oil_api=22, bubblepoint=1750, gas_sg=1.3)


def test_solubility_negative_abs_pressure_raises() -> None:
    # Tripwire for the real pabs<=0 guard that replaced the dead np.errstate
    # (which is a no-op on Python-float math): a negative absolute pressure must
    # raise, not silently return a complex.
    with pytest.raises(ValueError):
        BlackOil.solubility_kartoatmodjo(
            press=-20.0, temp=120.0, oil_api=22.0, gas_sg=0.65
        )


if __name__ == "__main__":
    plot_blackoil_compare(hyprop, pyprop)
