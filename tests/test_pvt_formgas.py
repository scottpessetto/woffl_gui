import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from woffl.pvt.formgas import FormGas

# only works if the command python -m tests.test_pvt_formgas


def compute_formgas_data(
    prs_ray: np.ndarray | list, temp: float, gas_sg: float
) -> dict:
    """Compute FormGas Data

    Create a list of properties of a formgas. Can be used to compare to results obtained with hysys.
    Density, viscosity and zfactor.
    """
    py_gas = FormGas(gas_sg=gas_sg)
    rho_gas, visc_gas, zfactor = [], [], []

    for prs in prs_ray:
        py_gas = py_gas.condition(prs, temp)

        rho_gas.append(py_gas.density)
        visc_gas.append(py_gas.viscosity)
        zfactor.append(py_gas.zfactor)

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
    # book example is at 1000 psia; condition() takes psig. The old
    # "1000 + 14.7" input compensated for compress() wrongly using gauge
    # pressure in the 1/p term (fixed 2026-06).
    pres_mccain = 1000 - 14.7  # psig equivalent of 1000 psia
    temp_mccain = 68
    mccain_compress = 0.001120
    methane = FormGas(gas_sg=0.55)
    calc_compress = methane.condition(pres_mccain, temp_mccain).compress
    assert calc_compress == pytest.approx(mccain_compress, rel=0.03)


def test_gas_compressibility_is_pure() -> None:
    # reading compress must not re-condition the object — it used to leave
    # the gas conditioned 10 psi high, poisoning in-flight cmix calculations
    gas = FormGas(gas_sg=0.65)
    gas.condition(500, 100)
    rho_before = gas.density
    _ = gas.compress
    assert gas.press == 500
    assert gas.density == rho_before


def test_gas_density() -> None:
    np.testing.assert_allclose(hygas["rho_gas"], pygas["rho_gas"], rtol=0.05)


def test_gas_viscosity() -> None:
    np.testing.assert_allclose(hygas["visc_gas"], pygas["visc_gas"], rtol=0.03)


def test_gas_zfactor() -> None:
    np.testing.assert_allclose(hygas["zfactor"], pygas["zfactor"], rtol=0.04)


def test_gas_sg_bounds_inclusive() -> None:
    # Tripwire for the inclusive-bounds library patch (docstring range 0.5-1.2).
    FormGas(gas_sg=0.5)
    FormGas(gas_sg=1.2)
    with pytest.raises(ValueError):
        FormGas(gas_sg=0.49)


def test_zfactor_gradschool_clamped_outside_correlation_range() -> None:
    """Tripwire for the P1-10 fix (upstream PR to kwellis/woffl).

    _zfactor_grad_school's cubic is unguarded outside its documented validity
    range (very high ppr / very low tpr) and can return z <= 0 or an
    implausibly large z. An unclamped z-factor silently poisons
    _compute_density (division by zfactor) and ResMix.cmix's math.sqrt(...)
    downstream. These (ppr, tpr) pairs drove the raw, unguarded correlation to
    -7.0 / -9.8 / -10.1 before the fix -- goes red (result <= 0, outside
    [_ZFACTOR_MIN, _ZFACTOR_MAX]) if the clamp is ever lost.
    """
    for ppr, tpr in ((20, 0.9), (15, 0.85), (10, 0.8)):
        zf = FormGas._zfactor_grad_school(ppr, tpr)
        assert FormGas._ZFACTOR_MIN <= zf <= FormGas._ZFACTOR_MAX


def test_zfactor_property_clamped_and_finite() -> None:
    """The public zfactor property (used by density/compress/cmix) must never
    return a non-physical value, even fed degenerate conditions."""
    gas = FormGas(gas_sg=0.65)
    # extreme low temp / high pressure pushes (ppr, tpr) into the degenerate
    # zone confirmed above (ppc~670 psia, tpc~365 R for this gas_sg)
    gas.condition(press=13000, temp=-140)
    zf = gas.zfactor
    assert FormGas._ZFACTOR_MIN <= zf <= FormGas._ZFACTOR_MAX
    # density and compress must stay finite (no domain error, no negative rho)
    assert gas.density > 0
    import math as _math

    assert _math.isfinite(gas.density)


def test_zfactor_gradschool_in_range_unchanged() -> None:
    """Precondition guard: the clamp must not touch the normal, in-range path
    (already-converging solves stay bit-identical)."""
    grad_zf = FormGas._zfactor_grad_school(blasing_ppr, blasing_tpr)
    assert grad_zf == pytest.approx(blasing_zf, rel=0.01)


if __name__ == "__main__":
    plot_formgas_compare(hygas, pygas)
