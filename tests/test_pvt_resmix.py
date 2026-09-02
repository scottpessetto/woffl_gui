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
    # rtol 0.04 -> 0.05 on 2026-09-02 (PVT-F3, z-factor cubic -> DAK): only
    # the 2,500 psig end point moved, -3.68 % -> -3.92 % vs Peng-Robinson
    # (DAK z 0.818 vs cubic 0.811 vs PR 0.798 for the 0.55 SG gas at 80 degF).
    # DAK is closer to PR over the sweep as a whole; see test_pvt_formgas.
    np.testing.assert_allclose(hymix["rho_mix"], pymix["rho_mix"], rtol=0.05)


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


def _schrader_mix(wc: float, fgor: float) -> ResMix:
    return ResMix(
        wc=wc,
        fgor=fgor,
        oil=BlackOil.schrader(),  # Pb = 1750 psig
        wat=FormWater.schrader(),
        gas=FormGas.schrader(),
    )


@pytest.mark.parametrize("wc, fgor", [(0.8, 400), (0.3, 300)])
def test_cmix_continuous_across_bubble_point(wc: float, fgor: float) -> None:
    """PVT-F1 tripwire (docs/upstream_sync.md #17, upstream PR to kwellis/woffl).

    cmix (Wood's equation) must use the ACOUSTIC oil compressibility. With the
    material-balance ``compress`` the mixture sound speed was 1,244 ft/s at
    1,749 psig vs 1,657 at 1,751 (WC 0.8 / GOR 400, +33 % across one psi of
    the bubble point) and 895 vs 1,597 (WC 0.3 / GOR 300, +78 %). Goes red if
    cmix reverts to ``comp_comp`` / ``oil.compress``.
    """
    mix = _schrader_mix(wc, fgor)
    below = mix.condition(1749, 100).cmix()
    above = mix.condition(1751, 100).cmix()
    assert below == pytest.approx(above, rel=0.01), (below, above)
    assert below > 1400  # the material-balance path sat at 895-1,244 here


def test_cmix_above_bubble_point_unchanged_by_acoustic_patch() -> None:
    """Above Pb the acoustic co equals the legacy Vasquez-Beggs ``compress``,
    so cmix there is bit-identical to Wood's equation evaluated with the
    material-balance ``comp_comp`` tuple (the pre-patch formula)."""
    import math

    mix = _schrader_mix(0.8, 400)
    for press in (2000, 2500):
        mix.condition(press, 100)
        co, cw, cg = mix.comp_comp()  # material-balance tuple, unchanged API
        yoil, ywat, ygas = mix.volm_fract()
        cs = ResMix._homogenous_mixture(yoil, ywat, ygas, co, cw, cg)
        legacy = math.sqrt(32.174 * 144 * (1 / cs) / mix.rho_mix())
        assert mix.cmix() == legacy


def test_cmix_does_not_raise_for_heavy_oil_at_low_temp() -> None:
    """PVT-F2 tripwire: an oil whose Vasquez-Beggs co is negative (14 API,
    60 degF) with no free gas used to put a negative bulk modulus into
    cmix's sqrt (untyped ValueError). With the floor it returns a finite,
    positive sound speed."""
    mix = ResMix(
        wc=0.0,
        fgor=10,  # < Rs(p): no free gas, so the oil term is the whole modulus
        oil=BlackOil(oil_api=14, bubblepoint=1500, gas_sg=0.65),
        wat=FormWater.schrader(),
        gas=FormGas.schrader(),
    )
    import math

    c = mix.condition(2000, 60).cmix()
    assert math.isfinite(c) and c > 0


@pytest.mark.parametrize("wc", [-0.01, 1.05, 2.0])
def test_watercut_out_of_range_raises(wc: float) -> None:
    """PVT-F4 tripwire (docs/upstream_sync.md #20, upstream PR to kwellis/woffl).

    ResMix silently accepted wc outside [0, 1] (wc 1.05 -> oil volume fraction
    -0.055) while every child class validates its own inputs. WellConfig, CSV
    stores and prop_hist.form_wc reach this constructor unguarded."""
    with pytest.raises(ValueError, match="[Ww]atercut"):
        _schrader_mix(wc, 400)


@pytest.mark.parametrize("fgor", [-1, -500.0])
def test_negative_fgor_raises(fgor: float) -> None:
    with pytest.raises(ValueError, match="GOR"):
        _schrader_mix(0.5, fgor)


def test_watercut_and_fgor_boundaries_accepted() -> None:
    """The guard is inclusive: 0 and 1 water cut and 0 GOR are legitimate
    (dewatering / dead oil) and must still construct."""
    _schrader_mix(0.0, 0)
    _schrader_mix(1.0, 0)
    _schrader_mix(0.5, 0.0)


def test_undersaturated_stream_oil_carries_only_fgor() -> None:
    """PVT-F5 tripwire (docs/upstream_sync.md #21, upstream PR to kwellis/woffl).

    Schrader at fgor 150 / 1,400 psig / 100 degF: Rs(p) is ~198 scf/stb, so
    the mass balance clamps free gas to zero — but the oil used to be
    evaluated at Rs(p), carrying 48 scf/stb of gas the stream does not have
    (-0.4 lbm/ft3 density, -18 % viscosity). The oil in the mixture must now
    match the same oil evaluated at Rs = fgor exactly, and the mass fractions
    must still close.
    """
    mix = _schrader_mix(0.5, 150).condition(1400, 100)
    oil = mix.oil
    assert BlackOil.solubility_kartoatmodjo(1400, 100, 22, 0.65) > 150
    assert oil.gas_solubility() == 150

    bo_ref = BlackOil.fvf_kartoatmodjo_below(100, 22, 0.65, 150)
    rho_ref = BlackOil.live_oil_density(22, 0.65, 150, bo_ref)
    uod = BlackOil.viscosity_dead_kartoatmodjo(100, 22)
    visc_ref = BlackOil.viscosity_live_kartoatmodjo_below(uod, 150)
    assert oil.density == rho_ref
    assert oil.viscosity == visc_ref

    xoil, xwat, xgas = mix.mass_fract()
    assert xgas == 0.0
    assert xoil + xwat + xgas == pytest.approx(1.0, abs=1e-12)
    assert all(0 <= x <= 1 for x in (xoil, xwat, xgas))


def test_saturated_stream_oil_is_bit_identical() -> None:
    """fgor >= Rs(p) (the usual case) must not move: the mixture's oil equals a
    standalone BlackOil at the same condition."""
    mix = _schrader_mix(0.5, 600).condition(1400, 100)
    solo = BlackOil.schrader().condition(1400, 100)
    assert mix.oil.gas_solubility() == solo.gas_solubility()
    assert mix.oil.density == solo.density
    assert mix.oil.viscosity == solo.viscosity


def test_standalone_blackoil_condition_unchanged() -> None:
    """The child API is intact: condition() without rs_max caps nothing, and a
    later plain condition() call drops a cap set earlier."""
    oil = BlackOil.schrader()
    rs_free = oil.condition(1400, 100).gas_solubility()
    assert oil.condition(1400, 100, rs_max=150).gas_solubility() == 150
    assert oil.condition(1400, 100).gas_solubility() == rs_free


if __name__ == "__main__":
    plot_resmix_compare(hymix, pymix)
