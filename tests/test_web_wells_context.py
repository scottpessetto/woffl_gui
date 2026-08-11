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


def test_event_cal_knobs_restore_into_seeds(context):
    saved, run = context
    saved["MPB-28"] = _saved(
        {"ken": 0.05, "nozzle_area_factor": 1.12, "mach_crit": 1.6}
    )
    seeds = run()["seeds"]
    assert seeds["ken"] == pytest.approx(0.05)
    assert seeds["nozzle_area_factor"] == pytest.approx(1.12)  # full precision
    assert seeds["mach_crit"] == pytest.approx(1.6)


def test_event_cal_seeds_are_clamped_to_widget_bounds(context):
    """A stored row outside the SimParams bounds must not poison the client
    store (seeds are applied wholesale over defaults)."""
    saved, run = context
    saved["MPB-28"] = _saved({"nozzle_area_factor": 2.5, "mach_crit": 9.0})
    seeds = run()["seeds"]
    assert seeds["nozzle_area_factor"] == pytest.approx(1.3)
    assert seeds["mach_crit"] == pytest.approx(2.5)


def test_no_saved_fit_leaves_knobs_unseeded(context):
    """Never-calibrated wells carry NO explicit knob seeds - the client's
    SimParams defaults (1.0 / 1.0) stand."""
    _saved_map, run = context
    seeds = run()["seeds"]
    assert "nozzle_area_factor" not in seeds
    assert "mach_crit" not in seeds
    assert "ken" not in seeds
