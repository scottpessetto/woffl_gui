"""Tests for JP history parsing and current pump lookup."""

import pandas as pd
import pytest

from woffl.assembly.jp_history import get_all_current_pumps, get_current_pump


def _make_jp_history():
    """Create synthetic JP history DataFrame."""
    return pd.DataFrame(
        {
            "Well Name": ["MPB-28", "MPB-28", "MPB-28", "MPE-41", "MPE-41"],
            "Date Set": pd.to_datetime(
                [
                    "2023-01-15",
                    "2023-06-20",
                    "2024-01-10",
                    "2023-03-01",
                    "2024-02-15",
                ]
            ),
            "Date Pulled": pd.to_datetime(
                [
                    "2023-06-19",
                    "2024-01-09",
                    None,
                    "2024-02-14",
                    None,
                ]
            ),
            "Nozzle Number": [12, 13, 14, 11, 12],
            "Throat Ratio": ["A", "B", "C", "A", "B"],
            "Tubing Diameter": [4.5, 4.5, 4.5, 4.5, 4.5],
        }
    )


class TestGetCurrentPump:
    def test_returns_latest(self):
        hist = _make_jp_history()
        result = get_current_pump(hist, "MPB-28")
        assert result is not None
        assert result["nozzle_no"] == "14"
        assert result["throat_ratio"] == "C"
        assert result["date_set"] == pd.Timestamp("2024-01-10")

    def test_different_well(self):
        hist = _make_jp_history()
        result = get_current_pump(hist, "MPE-41")
        assert result is not None
        assert result["nozzle_no"] == "12"
        assert result["throat_ratio"] == "B"

    def test_unknown_well_returns_none(self):
        hist = _make_jp_history()
        assert get_current_pump(hist, "FAKE-99") is None

    def test_all_nat_dates_returns_none(self):
        hist = pd.DataFrame(
            {
                "Well Name": ["MPB-28"],
                "Date Set": [pd.NaT],
                "Nozzle Number": [12],
                "Throat Ratio": ["A"],
                "Tubing Diameter": [4.5],
            }
        )
        assert get_current_pump(hist, "MPB-28") is None

    def test_result_keys(self):
        hist = _make_jp_history()
        result = get_current_pump(hist, "MPB-28")
        assert set(result.keys()) == {
            "nozzle_no",
            "throat_ratio",
            "tubing_od",
            "date_set",
        }

    def test_tubing_od_float(self):
        hist = _make_jp_history()
        result = get_current_pump(hist, "MPB-28")
        assert isinstance(result["tubing_od"], float)


class TestGetAllCurrentPumps:
    def test_one_row_per_well(self):
        hist = _make_jp_history()
        result = get_all_current_pumps(hist)
        assert len(result) == 2  # MPB-28 and MPE-41
        assert set(result["Well Name"]) == {"MPB-28", "MPE-41"}

    def test_latest_per_well(self):
        hist = _make_jp_history()
        result = get_all_current_pumps(hist)
        mpb28 = result[result["Well Name"] == "MPB-28"].iloc[0]
        assert mpb28["Nozzle Number"] == 14  # latest

    def test_empty_dataframe(self):
        hist = pd.DataFrame(
            columns=["Well Name", "Date Set", "Nozzle Number", "Throat Ratio"]
        )
        result = get_all_current_pumps(hist)
        assert result.empty

    def test_all_nat_returns_empty(self):
        hist = pd.DataFrame(
            {
                "Well Name": ["MPB-28"],
                "Date Set": [pd.NaT],
                "Nozzle Number": [12],
                "Throat Ratio": ["A"],
            }
        )
        result = get_all_current_pumps(hist)
        assert result.empty
