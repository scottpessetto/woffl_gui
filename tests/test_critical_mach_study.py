"""Analytic checks for the offline closure investigation, not a new app mode."""
import math

import numpy as np
import pytest

from tools import critical_mach_study as study


class ConstantInflow:
    pres = 2000.

    def oil_flow(self, pressure, method):
        return 1.


class IsothermalGas:
    """rho = b Pabs, with constant mass flow; analytic pressure-work integral."""
    b = .01
    mass_flow = 3.

    def condition(self, pressure, temperature):
        self.pressure = pressure
        return self

    def rho_mix(self):
        return self.b * (self.pressure+14.7)

    def cmix(self):
        return math.sqrt(study.GC_PSI/self.b)

    def insitu_volm_flow(self, oil_rate):
        return (self.mass_flow/self.rho_mix(), 0., 0.)


@pytest.mark.parametrize("ken", [0., .03, .4])
def test_energy_minimum_matches_isothermal_gas_analytic_limit(ken):
    area = .001
    gas = IsothermalGas()
    # At the maximum mass flux, (1+k)*v^2=c^2 and P*/Psu=exp(-1/2).
    critical_abs = gas.mass_flow/area * math.sqrt((1+ken)/(study.GC_PSI*gas.b))
    suction_abs = math.sqrt(math.e)*critical_abs
    psu, _, book = study.energy_minimum_floor(100., ken, area, ConstantInflow(), gas, step=1.25)
    assert psu+14.7 == pytest.approx(suction_abs, abs=.01)
    i = int(np.argmin(book.tde))
    assert 0 < i < len(book.tde)-1
    assert book.prs[i]+14.7 == pytest.approx(critical_abs, abs=1.25)
    assert book.mach[i] == pytest.approx(1/math.sqrt(1+ken), abs=.003)


def test_trial_closures_restore_production_functions_after_exception():
    original_zero = study.jf.throat_entry_zero_tde
    original_choke = study.jf.throat_entry_mach_one
    original_floor = study.jf.psu_minimize
    for mode in ("threshold_only", "scale_both", "energy_minimum"):
        with pytest.raises(RuntimeError):
            with study.variant(mode, 2.):
                raise RuntimeError("stop trial")
        assert study.jf.throat_entry_zero_tde is original_zero
        assert study.jf.throat_entry_mach_one is original_choke
        assert study.jf.psu_minimize is original_floor
