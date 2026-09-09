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


# FLOW-9's Mach interpolation was superseded by entry-energy-v1. The
# independent turning-point, conservation and analytic-limit checks live in
# test_entry_energy.py; the momentum-bracketing checks above remain active.
