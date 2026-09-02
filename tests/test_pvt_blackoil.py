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


def _pure_oil_sound_speed(oil: BlackOil, co: float) -> float:
    """Speed of sound in the single-phase oil, ft/s, from a compressibility."""
    import math

    return math.sqrt(32.174 * 144 / (co * oil.density))


def test_compress_acoustic_continuous_across_bubble_point() -> None:
    """PVT-F1 tripwire (docs/upstream_sync.md #17, upstream PR to kwellis/woffl).

    ``compress`` below Pb is McCain material-balance co (includes liberated
    gas) and is 1-2 orders of magnitude above the liquid-phase value; fed to
    Wood's equation it gave pure-oil "sound speeds" of 111-877 ft/s below Pb
    and a 4.4x jump across Pb. ``compress_acoustic`` is Vasquez-Beggs at Rs(p):
    continuous across Pb, identical to ``compress`` above it. Goes red if the
    property is lost or aliased back onto ``compress``.
    """
    oil = BlackOil.schrader()  # Pb = 1750 psig
    below = oil.condition(1749, 100).compress_acoustic
    above = oil.condition(1751, 100).compress_acoustic
    assert below == pytest.approx(above, rel=0.005)
    # above Pb the acoustic value IS the existing Vasquez-Beggs compress
    oil.condition(2000, 100)
    assert oil.compress_acoustic == oil.compress
    oil.condition(2500, 100)
    assert oil.compress_acoustic == oil.compress
    # below Pb it must NOT be the McCain material-balance value
    oil.condition(1000, 100)
    assert oil.compress_acoustic < oil.compress / 10


def test_pure_oil_acoustic_sound_speed_is_physical_below_pb() -> None:
    """Below Pb the single-phase oil sound speed from ``compress_acoustic`` must
    sit in the physical band (crude oils: ~3,500-5,000 ft/s). The material-
    balance ``compress`` gives 344-877 ft/s on the same points."""
    oil = BlackOil.schrader()
    for press in (500, 1000, 1400, 1749):
        oil.condition(press, 100)
        c_ac = _pure_oil_sound_speed(oil, oil.compress_acoustic)
        assert 2000 <= c_ac <= 5500, (press, c_ac)
        c_mb = _pure_oil_sound_speed(oil, oil.compress)
        assert c_mb < 1000, (press, c_mb)  # documents WHY compress is unusable here


def test_compressibility_floor_heavy_oil_low_temp() -> None:
    """PVT-F2 tripwire (docs/upstream_sync.md #18, upstream PR to kwellis/woffl).

    The Vasquez-Beggs numerator goes negative for ~10-14 API at 60-80 degF
    (Ugnu range): 14 API / 0.65 SG / Pb 1500 / 60 degF returned -1.6e-6 psi^-1
    above Pb, so Bo ROSE with pressure and ResMix.cmix hit a sqrt domain
    error. Both the material-balance and acoustic paths must floor at
    BlackOil._CO_FLOOR (1e-6) and warn exactly once per process.
    """
    import warnings

    oil = BlackOil(oil_api=14, bubblepoint=1500, gas_sg=0.65)
    raw = BlackOil.compressibility_vasquez_above(2000, 60, 14, 0.65, 100.0)
    assert raw < 0  # the precondition the floor exists for

    BlackOil._co_floor_warned = False  # fresh process semantics for this test
    with pytest.warns(RuntimeWarning, match="floored"):
        co_mb = oil.condition(2000, 60).compress
    assert co_mb >= BlackOil._CO_FLOOR == 1e-6
    assert oil.compress_acoustic >= BlackOil._CO_FLOOR

    # one-time: a second engagement is silent
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert oil.condition(3000, 60).compress == BlackOil._CO_FLOOR
        # The acoustic path carries its own, higher, physical floor (a
        # dead-oil sound speed of ~4,600 ft/s), applied silently.
        assert oil.compress_acoustic == max(BlackOil._CO_FLOOR, BlackOil._CO_ACOUSTIC_FLOOR)

    # Bo must not rise with pressure above Pb once co is floored
    bo_lo = oil.condition(1600, 60).oil_fvf()
    bo_hi = oil.condition(3000, 60).oil_fvf()
    assert bo_hi < bo_lo


def test_compressibility_floor_is_noop_in_range() -> None:
    """The floor must not touch an in-range oil (bit-identical solves)."""
    oil = BlackOil.test_oil().condition(2500, 80)
    raw = BlackOil.compressibility_vasquez_above(
        2500, 80, oil.oil_api, oil.gas_sg, oil.gas_solubility()
    )
    assert raw > BlackOil._CO_FLOOR
    assert oil.compress == raw
    # Acoustic path: the raw value unless it is below the dead-oil floor
    # (c ~ 4,600 ft/s), which caps pure-oil sound speed at a physical value.
    assert oil.compress_acoustic == max(raw, BlackOil._CO_ACOUSTIC_FLOOR)


def test_acoustic_floor_caps_pure_oil_sound_speed() -> None:
    """Vasquez-Beggs extrapolates LOW on co at low Rs / low pressure, which
    with only the 1e-6 material-balance floor gave pure-oil sound speeds up
    to ~9,000 ft/s below ~900 psig. The acoustic floor (4e-6 psi^-1) caps
    the liquid at ~4,600 ft/s - the physical ceiling for a hydrocarbon
    liquid - without touching the material-balance path."""
    import math

    oil = BlackOil.schrader()
    for press in (100, 300, 500, 800):
        oil.condition(press, 80)
        co = oil.compress_acoustic
        assert co >= BlackOil._CO_ACOUSTIC_FLOOR
        c = math.sqrt(144.0 * 32.174 / (co * oil.density))
        assert c <= 4800.0, f"{press} psig: {c:.0f} ft/s"


if __name__ == "__main__":
    plot_blackoil_compare(hyprop, pyprop)
