"""jetflow regression guards (docs/upstream_sync.md #32 FLOW-9, #33 FLOW-6).

``_throat_discharge_bracketed`` must find a momentum-balance hump narrower
than its scan step, and ``_tde_at_mach`` must interpolate between the two
sweep points that bracket the critical Mach number.
"""

import numpy as np
import pytest

from woffl.flow import jetflow as jf
from woffl.flow.errors import ConvergenceError
from woffl.flow.jetplot import JetBook

# ------------------------------------------------------------------ FLOW-6


def test_narrow_hump_missed_by_the_scan_is_still_found():
    pte = 400.0
    pc, half = 843.6, 7.5  # a 15-psi-wide positive hump
    scan_ranges = (max(6.0 * pte, 300.0), max(15.0 * pte, 1500.0))
    for hi in scan_ranges:
        grid = np.linspace(15.0, hi, 60)
        assert (np.abs(grid - pc) > half).all()  # every scan point is negative

    def bal(p):
        return 1.0 - ((p - pc) / half) ** 2  # -inf at both ends, two roots

    ptm = jf._throat_discharge_bracketed(bal, pte)
    assert ptm == pytest.approx(pc + half, abs=0.05)  # the physical HIGH root
    assert abs(bal(ptm)) < 1e-2


def test_hump_extending_past_the_scan_range_is_handled():
    pte = 400.0
    pc, half = 6003.0, 7.5  # hump straddles the top of the scan range (15 * pte)

    def bal(p):
        return 1.0 - ((p - pc) / half) ** 2

    ptm = jf._throat_discharge_bracketed(bal, pte)
    assert ptm == pytest.approx(pc + half, abs=0.05)


def test_no_positive_region_still_raises():
    with pytest.raises(ConvergenceError):
        jf._throat_discharge_bracketed(lambda p: -1.0 - 1e-6 * p, 400.0)


# ------------------------------------------------------------------ FLOW-9


def test_tde_at_mach_interpolates_between_bracketing_points():
    book = JetBook(1000.0, 900.0, 50.0, 1000.0, 100.0)  # Mach 0.9
    book.append(975.0, 1100.0, 50.0, 1000.0, 121.0)  # Mach 1.1
    t1, t2 = book.tde_ray
    assert jf._tde_at_mach(book, 1.0) == pytest.approx(t1 + (t2 - t1) * 0.5)
    assert jf._tde_at_mach(book, 1.05) == pytest.approx(t1 + (t2 - t1) * 0.75)


def test_tde_at_mach_degenerate_pair_falls_back_to_sub_threshold_value():
    book = JetBook(1000.0, 900.0, 50.0, 1000.0, 100.0)
    book.append(975.0, 900.0, 50.0, 1000.0, 121.0)  # same Mach twice
    assert jf._tde_at_mach(book, 1.0) == book.tde_ray[-2]


def test_throat_entry_mach_one_returns_the_interpolated_value():
    from woffl.flow.inflow import InFlow
    from woffl.geometry.jetpump import JetPump
    from woffl.pvt.blackoil import BlackOil
    from woffl.pvt.formgas import FormGas
    from woffl.pvt.formwat import FormWater
    from woffl.pvt.resmix import ResMix

    ipr = InFlow(qwf=246, pwf=1049, pres=1400)
    res = ResMix(wc=0.894, fgor=600, oil=BlackOil.schrader(), wat=FormWater.schrader(), gas=FormGas.schrader())
    jp = JetPump("9", "X")
    tee, _q, book = jf.throat_entry_mach_one(1300.0, 80, jp.ken, jp.ate, ipr, res)
    assert book.mach_ray[-1] >= 1.0 > book.mach_ray[-2]
    assert tee == jf._tde_at_mach(book, 1.0)
    assert tee != book.tde_ray[-2]
