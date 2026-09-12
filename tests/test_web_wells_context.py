"""well_context saved-fit hydration: the friction channel of the saved-IPR
overlay (step d) must restore BHP-calibrated ken/kth/kdi AND the pillar-1b
event-calibration knobs (nozzle_area_factor / mach_crit) into the solver
sidebar seeds, clamped to the SimParams widget bounds and independent of the
pin-vs-values precedence. Everything upstream (chars, JP history, well tests,
live PF) is monkeypatched - no Databricks.
"""

from __future__ import annotations

import pandas as pd
import pytest

import server.services.wells as wells_svc
from woffl.gui import ipr_anchor


def _saved(friction: dict) -> dict:
    """Minimal _assemble_saved_ipr record: friction only, no curve."""
    return {
        "values": {},
        "friction": friction,
        "locks": {},
        "lock_values": {},
        "wc_locked": False,
        "wc_value": None,
        "saved_at": None,
        "saved_by": None,
        "pin_at": None,
        "pin_value": None,
        "pin_user": None,
    }


@pytest.fixture()
def context(monkeypatch):
    """Run well_context for MPB-28 against a stubbed pipeline; the test picks
    the saved-fit record by mutating `saved` before calling."""
    from server.services import pump_calibration
    monkeypatch.setattr(pump_calibration, "snapshot", lambda: {})
    saved: dict[str, dict] = {}
    chars = pd.DataFrame([{"Well": "MPB-28", "res_pres": 1700.0}])
    monkeypatch.setattr(
        wells_svc.datasources, "well_chars_safe", lambda: (chars, "test")
    )
    monkeypatch.setattr(
        wells_svc.datasources, "jp_history_safe", lambda: (None, "none")
    )
    monkeypatch.setattr(
        wells_svc.tests_svc, "tests_for_well", lambda well, months, cap: None
    )
    monkeypatch.setattr(wells_svc, "_live_pf_seed", lambda well, tests_df: None)
    monkeypatch.setattr(
        ipr_anchor, "load_saved_ipr", lambda well: saved.get(well)
    )

    def run() -> dict:
        return wells_svc.well_context("MPB-28")

    return saved, run


def test_legacy_friction_is_reported_but_not_inherited(context):
    saved, run = context
    saved["MPB-28"] = _saved({"ken": .005, "nozzle_area_factor": 1.12, "mach_crit": 1.6})
    ctx = run()
    assert ctx["pump_calibration"]["status"] == "legacy"
    assert ctx["seeds"]["ken"] == .03
    assert ctx["seeds"]["nozzle_area_factor"] == 1.
    assert ctx["seeds"]["mach_crit"] == 1.


def test_saved_hydraulics_hydrates_sidebar_and_optimizer_from_same_context(context, monkeypatch):
    from server.services import pump_calibration, optimizer_runs
    _, run = context
    monkeypatch.setattr(pump_calibration, "resolve", lambda *a, **kw: {
        "status": "active", "hydraulics_model": "drift_flux", "quality": {},
        "coefficients": {"ken": .06, "kth": .35, "kdi": .3, "nozzle_area_factor": 1.02},
    })
    ctx = run()
    assert ctx["seeds"]["hydraulics_model"] == "drift_flux"
    cfg = optimizer_runs._config_from_seeds("MPB-28", "B", ctx["seeds"])
    assert cfg.hydraulics_model == "drift_flux" and cfg.ken_well == .06


def test_invalid_legacy_coefficients_cannot_poison_hydration(context):
    saved, run = context
    saved["MPB-28"] = _saved({"nozzle_area_factor": 2.5, "mach_crit": 9.0})
    assert run()["seeds"]["nozzle_area_factor"] == 1.


def test_unfitted_well_uses_explicit_reference_coefficients(context):
    from woffl.assembly.pump_candidates import CLEAN_PUMP
    _, run = context
    ctx = run()
    assert ctx["pump_calibration"]["status"] == "none"
    for key, value in CLEAN_PUMP.items():
        assert ctx["seeds"][key] == value


@pytest.mark.parametrize("stamp", ["2026-08-10T00:00:00", "2026-08-10T15:30:00"])
def test_context_keeps_exact_installation_timestamp(context, monkeypatch, stamp):
    _, run = context
    frame = pd.DataFrame([{"Well Name": "MPB-28", "Date Set": pd.Timestamp(stamp),
                           "Nozzle Number": 12, "Throat Ratio": "B", "Tubing Diameter": 4.5}])
    monkeypatch.setattr(wells_svc.datasources, "jp_history_safe", lambda: (frame, "databricks"))
    assert run()["pump"]["date_set"] == stamp + "+00:00"
