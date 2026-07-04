"""Tests for woffl.gui.tab_helpers.

Covers the two P1 fixes from docs/code_review_2026-07-01.md:

- P1-4: ``physical_sweep_signature`` — the shared sweep-cache signature must
  include every physical input (``model_as_water`` was missing from both
  tabs' local tuples, so at form_wc = 1.0 the cache served the other mode's
  sweep).
- P1-11: ``pump_at_test_matches`` — set-to-set pump-at-test-date resolution
  so calibration never pairs the current pump's model with a test taken
  under the previous pump (JPCO→next-test window).
"""

import dataclasses

import pandas as pd
import pytest

from woffl.gui.params import SimulationParams
from woffl.gui.tab_helpers import physical_sweep_signature, pump_at_test_matches


class TestPhysicalSweepSignature:
    def test_includes_model_as_water(self):
        # The exact P1-4 dewatering case: identical params except the mode.
        base = SimulationParams(form_wc=1.0, model_as_water=False)
        water = SimulationParams(form_wc=1.0, model_as_water=True)
        assert physical_sweep_signature(base) != physical_sweep_signature(water)

    def test_identical_params_identical_signature(self):
        assert physical_sweep_signature(SimulationParams()) == (
            physical_sweep_signature(SimulationParams())
        )

    @pytest.mark.parametrize(
        "field,value",
        [
            ("selected_well", "MPB-28"),
            ("field_model", "Kuparuk"),
            ("jpump_direction", "forward"),
            ("nozzle_no", "13"),
            ("area_ratio", "C"),
            ("ken", 0.05),
            ("kth", 0.35),
            ("kdi", 0.5),
            ("tubing_od", 3.5),
            ("tubing_thickness", 0.4),
            ("casing_od", 7.0),
            ("casing_thickness", 0.6),
            ("jpump_tvd", 5000),
            ("form_wc", 0.9),
            ("form_gor", 400),
            ("form_temp", 90),
            ("model_as_water", True),
            ("oil_api", 30.0),
            ("gas_sg", 0.7),
            ("wat_sg", 1.05),
            ("bubble_point", 1200.0),
            ("surf_pres", 300),
            ("rho_pf", 60.0),
            ("ppf_surf", 3000),
            ("qwf", 900),
            ("pwf", 600),
            ("pres", 1800),
        ],
    )
    def test_changing_physical_field_changes_signature(self, field, value):
        base = SimulationParams()
        assert getattr(base, field) != value, f"pick a non-default value for {field}"
        changed = dataclasses.replace(base, **{field: value})
        assert physical_sweep_signature(base) != physical_sweep_signature(changed)

    def test_signature_is_hashable(self):
        sig = physical_sweep_signature(SimulationParams())
        # Usable as a dict key / comparable across reruns
        assert {sig: "cached"}[sig] == "cached"

    def test_tab_extras_append_cleanly(self):
        # Both tabs build: shared physical tuple + tab-specific extras.
        params = SimulationParams()
        sig = physical_sweep_signature(params) + (
            tuple(params.nozzle_batch_options),
            tuple(params.throat_batch_options),
        )
        assert {sig: "cached"}[sig] == "cached"


def _make_jp_hist() -> pd.DataFrame:
    """Two installs on MPB-28: 12B (2024-01-10) then a JPCO to 14C (2025-06-01).

    Date Pulled is deliberately all-NaT — pump tenure is set-to-set and the
    helper must never consult Date Pulled (CLAUDE.md hard rule).
    """
    return pd.DataFrame(
        {
            "Well Name": ["MPB-28", "MPB-28"],
            "Date Set": pd.to_datetime(["2024-01-10", "2025-06-01"]),
            "Date Pulled": pd.to_datetime([None, None]),
            "Nozzle Number": [12, 14],
            "Throat Ratio": ["B", "C"],
            "Tubing Diameter": [4.5, 4.5],
        }
    )


class TestPumpAtTestMatches:
    def test_pre_jpco_test_does_not_match_current_pump(self):
        # Test taken under the old 12B — current pump 14C is NOT a match.
        hist = _make_jp_hist()
        assert (
            pump_at_test_matches(hist, "MPB-28", pd.Timestamp("2025-05-15"), "14", "C")
            is False
        )

    def test_post_jpco_test_matches_current_pump(self):
        hist = _make_jp_hist()
        assert (
            pump_at_test_matches(hist, "MPB-28", pd.Timestamp("2025-06-20"), "14", "C")
            is True
        )

    def test_pre_jpco_test_matches_old_pump(self):
        hist = _make_jp_hist()
        assert (
            pump_at_test_matches(hist, "MPB-28", pd.Timestamp("2024-05-01"), "12", "B")
            is True
        )

    def test_empty_history_returns_none(self):
        empty = _make_jp_hist().iloc[0:0]
        assert (
            pump_at_test_matches(empty, "MPB-28", pd.Timestamp("2025-06-20"), "14", "C")
            is None
        )

    def test_none_history_returns_none(self):
        assert (
            pump_at_test_matches(None, "MPB-28", pd.Timestamp("2025-06-20"), "14", "C")
            is None
        )

    def test_missing_test_date_returns_none(self):
        hist = _make_jp_hist()
        assert pump_at_test_matches(hist, "MPB-28", None, "14", "C") is None
        assert pump_at_test_matches(hist, "MPB-28", pd.NaT, "14", "C") is None

    def test_date_before_first_install_returns_none(self):
        hist = _make_jp_hist()
        assert (
            pump_at_test_matches(hist, "MPB-28", pd.Timestamp("2023-01-01"), "14", "C")
            is None
        )

    def test_unknown_well_returns_none(self):
        hist = _make_jp_hist()
        assert (
            pump_at_test_matches(hist, "FAKE-99", pd.Timestamp("2025-06-20"), "14", "C")
            is None
        )

    def test_corrupt_record_fails_open(self):
        # S-17-style row: throat letter in the Nozzle Number column. Must NOT
        # be treated as a real test pump (mirrors _is_valid_pump_code).
        hist = pd.DataFrame(
            {
                "Well Name": ["S-17"],
                "Date Set": pd.to_datetime(["2024-01-10"]),
                "Date Pulled": pd.to_datetime([None]),
                "Nozzle Number": ["D"],
                "Throat Ratio": ["B"],
                "Tubing Diameter": [4.5],
            }
        )
        assert (
            pump_at_test_matches(hist, "S-17", pd.Timestamp("2024-05-01"), "14", "C")
            is None
        )

    def test_missing_pump_codes_return_none(self):
        hist = _make_jp_hist()
        when = pd.Timestamp("2025-06-20")
        assert pump_at_test_matches(hist, "MPB-28", when, None, "C") is None
        assert pump_at_test_matches(hist, "MPB-28", when, "14", None) is None

    def test_date_pulled_is_ignored(self):
        # Even with a (bogus) Date Pulled between installs, tenure is
        # set-to-set: a test in the phantom "out of hole" window still
        # resolves to the old pump.
        hist = _make_jp_hist()
        hist.loc[0, "Date Pulled"] = pd.Timestamp("2025-04-01")
        assert (
            pump_at_test_matches(hist, "MPB-28", pd.Timestamp("2025-05-15"), "12", "B")
            is True
        )
