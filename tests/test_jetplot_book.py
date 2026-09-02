"""JetBook regression guards (docs/upstream_sync.md #31 FLOW-5, #34 FLOW-7).

The sweep book keeps Python lists while it is built and materializes the
``*_ray`` numpy views on demand; every stored number must be bit-identical to
what the old ``np.append`` + ``scipy.integrate.trapezoid`` path produced, and
the positive-tde clamp in ``_dete_zero`` must be bounded.
"""

import numpy as np
import pytest

from woffl.flow import jetflow as jf
from woffl.flow.errors import ThroatEntryNoSolution
from woffl.flow.jetplot import _CLAMP_TOL_FRAC, JetBook, ThroatEntryChoked

# ------------------------------------------------------------------ FLOW-7


def test_append_expansion_energy_bit_identical_to_incremental_ee():
    """The inline two-point trapezoid reproduces jf.incremental_ee to the bit."""
    rng = np.random.default_rng(20260902)
    for _ in range(2000):
        p1 = rng.uniform(50.0, 3000.0)
        p2 = p1 - 25.0
        r1, r2 = rng.uniform(5.0, 70.0, 2)
        ref = jf.incremental_ee(np.array([p1, p2]), np.array([r1, r2]))
        book = JetBook(p1, 10.0, r1, 1000.0, 100.0)
        book.append(p2, 12.0, r2, 1000.0, 120.0)
        assert book.ede_ray[-1] == ref  # bitwise, not approx
        assert book.tde_ray[-1] == 120.0 + ref
        assert book.grad_ray[-1] == (100.0 - (120.0 + ref)) / (p1 - p2)


def test_ray_views_track_appends_setter_and_copy():
    book = JetBook(1000.0, 10.0, 50.0, 1000.0, 100.0)
    first = book.prs_ray
    assert isinstance(first, np.ndarray) and first.tolist() == [1000.0]
    assert book.ede_ray.tolist() == [0] and book.tde_ray.tolist() == [100.0]
    book.append(975.0, 11.0, 49.0, 1000.0, 121.0)
    assert book.prs_ray.tolist() == [1000.0, 975.0]
    assert book.mach_ray.tolist() == [0.01, 0.011]
    assert np.isnan(book.grad_ray[0]) and len(book.grad_ray) == 2
    # whole-array assignment (tests/test_flow_geometry_guards.py relies on it)
    book.tde_ray = np.array([1.0, -1.0])
    assert book.tde == [1.0, -1.0] and book.tde_ray.tolist() == [1.0, -1.0]
    # copy is independent
    twin = book.copy()
    twin.append(950.0, 12.0, 48.0, 1000.0, 144.0)
    assert len(book.prs_ray) == 2 and len(twin.prs_ray) == 3


def test_integer_psu_keeps_the_integer_element_type():
    """The sweep loops read ``prs_ray[-1]`` and feed it back into the PVT; the
    scalar TYPE must be what np.append produced (int64 stays int64 until a
    float is appended) or the PVT rounds differently and bit-identity breaks."""
    book = JetBook(1400, 10.0, 50.0, 1000.0, 100.0)
    assert book.prs_ray.dtype.kind == "i"
    book.append(book.prs_ray[-1] - 25, 11.0, 49.0, 1000.0, 121.0)
    assert book.prs_ray.dtype.kind == "i"
    assert isinstance(book.prs_ray[-1], np.integer)
    book.append(1350.5, 11.0, 49.0, 1000.0, 121.0)
    assert book.prs_ray.dtype.kind == "f"


def test_zero_tde_walk_bit_identical_to_manual_sweep():
    """A real throat-entry sweep built through the book equals one built with
    freshly-materialized arrays at every step (the pre-patch data flow)."""
    from woffl.flow.inflow import InFlow
    from woffl.geometry.jetpump import JetPump
    from woffl.pvt.blackoil import BlackOil
    from woffl.pvt.formgas import FormGas
    from woffl.pvt.formwat import FormWater
    from woffl.pvt.resmix import ResMix

    ipr = InFlow(qwf=246, pwf=1049, pres=1400)
    res = ResMix(wc=0.894, fgor=600, oil=BlackOil.schrader(), wat=FormWater.schrader(), gas=FormGas.schrader())
    jp = JetPump("12", "B")
    _q, book = jf.throat_entry_zero_tde(psu=1250.0, tsu=80, ken=jp.ken, ate=jp.ate, ipr_su=ipr, prop_su=res)
    arrays = {n: getattr(book, n + "_ray").copy() for n in ("prs", "vel", "rho", "snd", "kde", "ede", "tde", "mach")}
    # rebuild from the same inputs through np.append + scipy trapezoid
    ede = np.array([0])
    for i in range(1, len(arrays["prs"])):
        inc = jf.incremental_ee(arrays["prs"][i - 1 : i + 1], arrays["rho"][i - 1 : i + 1])
        ede = np.append(ede, ede[-1] + inc)
    assert np.array_equal(ede, arrays["ede"])
    assert np.array_equal(arrays["kde"] + ede, arrays["tde"])
    assert np.array_equal(arrays["vel"] / arrays["snd"], arrays["mach"])


# ------------------------------------------------------------------ FLOW-5


def _v_shaped(tmin: float, kde0: float = 10000.0):
    """Descending pressures with a convex tde that bottoms out at ``tmin``
    six 25-psi steps in - the shape of a choked throat-entry sweep."""
    prs = 1000.0 - 25.0 * np.arange(13)
    k = (kde0 - tmin) / 150.0**2
    tde = tmin + k * (prs - 850.0) ** 2
    vel = 100.0 + np.arange(13) * 10.0
    rho = 50.0 - np.arange(13) * 0.5
    mach = vel / 1000.0
    return prs, vel, rho, tde, mach


def test_clamp_accepted_when_minimum_is_within_tolerance():
    prs, vel, rho, tde, mach = _v_shaped(tmin=0.005 * 10000.0)
    pte, vte, rho_te, mach_te = JetBook._dete_zero(prs, vel, rho, tde, mach)
    assert pte == 850.0  # clamps onto the minimum-tde (choke) point
    assert (vte, rho_te, mach_te) == (160.0, 47.0, 0.16)


def test_clamp_accepted_exactly_at_tolerance():
    prs, vel, rho, tde, mach = _v_shaped(tmin=_CLAMP_TOL_FRAC * 10000.0)
    assert JetBook._dete_zero(prs, vel, rho, tde, mach)[0] == 850.0


def test_clamp_rejected_when_the_branch_never_closes():
    """25 % of the entry kinetic energy unaccounted: the pump cannot be fed at
    this suction. The old code fabricated a choked state here."""
    prs, vel, rho, tde, mach = _v_shaped(tmin=0.25 * 10000.0)
    with pytest.raises(ThroatEntryChoked) as info:
        JetBook._dete_zero(prs, vel, rho, tde, mach)
    err = info.value
    assert isinstance(err, ThroatEntryNoSolution)
    assert isinstance(err, ValueError) and isinstance(err, IndexError)
    assert "25%" in str(err)


def test_zero_crossing_path_unchanged():
    prs, vel, rho, tde, mach = _v_shaped(tmin=-500.0)
    pte, *_ = JetBook._dete_zero(prs, vel, rho, tde, mach)
    assert 850.0 < pte < 1000.0
    assert pte == np.interp(0, np.flip(tde[:7]), np.flip(prs[:7]))
