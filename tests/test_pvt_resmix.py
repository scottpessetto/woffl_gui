import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from woffl.pvt import BlackOil, FormGas, FormWater, ResMix

# run command python -m tests.rmix_test is used


def compute_resmix_data(
    prs_ray: np.ndarray | pd.Series,
    temp: float,
    wc: float,
    fgor: float,
    oil_api: float,
    pbub: float,
    gas_sg: float,
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
        axs[0].scatter(
            hydict["pres_psig"],
            hydict["mass_fracs"][cat],
            label=f"Hy {cat.capitalize()}",
        )
        axs[0].scatter(
            hydict["pres_psig"],
            pydict["mass_fracs"][cat],
            marker="*",
            label=f"Py {cat.capitalize()}",
        )

        axs[1].scatter(
            hydict["pres_psig"],
            hydict["volm_fracs"][cat],
            label=f"Hy {cat.capitalize()}",
        )
        axs[1].scatter(
            hydict["pres_psig"],
            pydict["volm_fracs"][cat],
            marker="*",
            label=f"Py {cat.capitalize()}",
        )

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
    np.testing.assert_allclose(
        hymix[name_frac]["oil"], pymix[name_frac]["oil"], rtol=0.01
    )
    np.testing.assert_allclose(
        hymix[name_frac]["wat"], pymix[name_frac]["wat"], rtol=0.01
    )
    np.testing.assert_allclose(
        hymix[name_frac]["gas"], pymix[name_frac]["gas"], rtol=0.06
    )


def test_volm_fractions() -> None:
    name_frac = "volm_fracs"
    np.testing.assert_allclose(
        hymix[name_frac]["oil"], pymix[name_frac]["oil"], rtol=0.03
    )
    np.testing.assert_allclose(
        hymix[name_frac]["wat"], pymix[name_frac]["wat"], rtol=0.04
    )
    np.testing.assert_allclose(
        hymix[name_frac]["gas"], pymix[name_frac]["gas"], rtol=0.06
    )


def test_mixture_density() -> None:
    np.testing.assert_allclose(hymix["rho_mix"], pymix["rho_mix"], rtol=0.04)


def _dry_mix(wc: float) -> ResMix:
    return ResMix(
        wc=wc,
        fgor=500,
        oil=BlackOil(oil_api=22, bubblepoint=2000, gas_sg=0.55),
        wat=FormWater(wat_sg=1.0),
        gas=FormGas(gas_sg=0.55),
    ).condition(1500, 100)


def test_full_watercut_raises_valueerror() -> None:
    """100% water cut -> zero oil volume fraction.

    insitu_volm_flow must raise a typed ValueError (caught by the GUI's
    run_jetpump_solver and the batch solvers), NOT a bare ZeroDivisionError
    that escapes every `except ValueError` and crashes the Streamlit page.

    Tripwire for the local library guard in
    ``ResMix._static_insitu_volm_flow`` (see docs/upstream_sync.md). If an
    upstream sync drops the guard, ``qtot = qoil / yoil`` raises
    ZeroDivisionError (not a ValueError) and this test goes red.
    """
    mix = _dry_mix(1.0)
    yoil, _, _ = mix.volm_fract()
    assert yoil == 0  # 100% water cut => no oil by volume
    with pytest.raises(ValueError):
        mix.insitu_volm_flow(qoil_std=100)


def test_near_full_watercut_still_solves() -> None:
    """Just below 100% WC keeps a tiny nonzero oil fraction and must NOT raise
    — the guard is specific to the degenerate yoil == 0 case."""
    qoil, qwat, qgas = _dry_mix(0.99).insitu_volm_flow(qoil_std=100)
    assert qoil > 0 and qwat > 0


def test_water_mode_anchors_on_water() -> None:
    """Opt-in water-pump mode: with model_as_water=True a 100% WC mixture
    anchors insitu flow on WATER (the input rate is water bwpd) instead of
    raising — oil=0, water>0, gas=0, and the flow scales linearly with rate."""
    mix = ResMix(
        wc=1.0,
        fgor=500,
        oil=BlackOil(oil_api=22, bubblepoint=2000, gas_sg=0.55),
        wat=FormWater(wat_sg=1.0),
        gas=FormGas(gas_sg=0.55),
        model_as_water=True,
    ).condition(1500, 100)

    qoil, qwat, qgas = mix.insitu_volm_flow(qoil_std=500)  # input is WATER bwpd
    assert qoil == 0.0
    assert qwat > 0
    assert qgas == 0.0
    # doubling the water rate doubles the insitu water flow (linear anchor)
    _qoil2, qwat2, _qgas2 = mix.insitu_volm_flow(qoil_std=1000)
    assert abs(qwat2 - 2 * qwat) < 1e-9


if __name__ == "__main__":
    plot_resmix_compare(hymix, pymix)
