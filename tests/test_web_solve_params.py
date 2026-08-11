"""SimParams passthrough of the multi-point event calibration knobs.

solve_single must hand SimParams.mach_crit to the assembly jetpump_solver
and apply SimParams.nozzle_area_factor to the built JetPump as
dnz_eff = dnz_catalog * sqrt(factor) (the pf_calibration wear mechanics).
Defaults (1.0 / 1.0) must leave both untouched so existing web solves stay
byte-identical. The assembly solver is monkeypatched at its module - solve's
_run_solver imports it at call time - and well "Custom" keeps the whole path
offline (preset well profile, no Databricks).
"""

import math

import pytest
from pydantic import ValidationError

import server.services.solve as solve_svc
import woffl.assembly.solopump as solopump
from server import schemas
from woffl.geometry.jetpump import JetPump


@pytest.fixture()
def captured_solver(monkeypatch):
    """Capture the kwargs solve._run_solver hands the assembly solver."""
    seen = {}

    def fake_solver(**kwargs):
        seen.update(kwargs)
        # (psu, sonic_status, qoil_std, fwat_bwpd, qnz_bwpd, mach_te)
        return (900.0, False, 100.0, 200.0, 300.0, 0.4)

    monkeypatch.setattr(solopump, "jetpump_solver", fake_solver)
    return seen


def test_mach_crit_and_fnz_passthrough(captured_solver):
    sp = schemas.SimParams(mach_crit=1.8, nozzle_area_factor=1.2)
    solve_svc.solve_single("Custom", sp)
    assert captured_solver["mach_crit"] == 1.8
    catalog_dnz = JetPump(sp.nozzle_no, sp.area_ratio).dnz
    assert captured_solver["jpump"].dnz == pytest.approx(
        catalog_dnz * math.sqrt(1.2)
    )


def test_defaults_leave_solver_untouched(captured_solver):
    sp = schemas.SimParams()
    solve_svc.solve_single("Custom", sp)
    assert captured_solver["mach_crit"] == 1.0
    catalog_dnz = JetPump(sp.nozzle_no, sp.area_ratio).dnz
    assert captured_solver["jpump"].dnz == catalog_dnz  # exact, not approx


@pytest.mark.parametrize(
    "field, value",
    [
        ("mach_crit", 0.9),
        ("mach_crit", 2.6),
        ("nozzle_area_factor", 0.7),
        ("nozzle_area_factor", 1.4),
    ],
)
def test_bounds_rejected(field, value):
    with pytest.raises(ValidationError):
        schemas.SimParams(**{field: value})
