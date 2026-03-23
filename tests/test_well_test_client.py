"""Tests for well test client: name normalization, pad filtering, mocked DB queries."""

from unittest.mock import patch

import pandas as pd
import pytest

from woffl.assembly.well_test_client import (
    _denormalize_well_name,
    _normalize_well_name,
    fetch_milne_well_tests,
    filter_wells_by_pad,
    get_mpu_well_names,
    get_pad_names,
)

# ── _normalize_well_name ───────────────────────────────────────────────────


class TestNormalizeWellName:
    def test_b028_to_mpb28(self):
        assert _normalize_well_name("B-028") == "MPB-28"

    def test_e041_to_mpe41(self):
        assert _normalize_well_name("E-041") == "MPE-41"

    def test_b003_to_mpb03(self):
        """Regex strips one leading zero: 003 -> 03."""
        assert _normalize_well_name("B-003") == "MPB-03"

    def test_already_normalized(self):
        result = _normalize_well_name("MPB-28")
        assert result == "MPB-28"

    def test_no_leading_zeros(self):
        assert _normalize_well_name("B-100") == "MPB-100"


# ── _denormalize_well_name ──────────────────────────────────────────────────


class TestDenormalizeWellName:
    def test_mpb28_to_b028(self):
        assert _denormalize_well_name("MPB-28") == "B-028"

    def test_mpe41_to_e041(self):
        assert _denormalize_well_name("MPE-41") == "E-041"

    def test_mpb3_to_b003(self):
        assert _denormalize_well_name("MPB-3") == "B-003"


# ── Roundtrip ──────────────────────────────────────────────────────────────


class TestRoundtrip:
    @pytest.mark.parametrize("name", ["MPB-28", "MPE-41", "MPB-03", "MPB-100"])
    def test_normalize_denormalize_roundtrip(self, name):
        assert _normalize_well_name(_denormalize_well_name(name)) == name


# ── get_pad_names ──────────────────────────────────────────────────────────


class TestGetPadNames:
    def test_extracts_unique_sorted(self):
        wells = ["B-028", "B-030", "E-041", "E-035", "L-001"]
        pads = get_pad_names(wells)
        assert pads == ["B", "E", "L"]

    def test_single_pad(self):
        pads = get_pad_names(["B-028", "B-030", "B-032"])
        assert pads == ["B"]

    def test_empty(self):
        assert get_pad_names([]) == []


# ── filter_wells_by_pad ───────────────────────────────────────────────────


class TestFilterWellsByPad:
    def test_single_pad(self):
        wells = ["B-028", "B-030", "E-041"]
        result = filter_wells_by_pad(wells, ["B"])
        assert result == ["B-028", "B-030"]

    def test_multiple_pads(self):
        wells = ["B-028", "E-041", "L-001"]
        result = filter_wells_by_pad(wells, ["B", "L"])
        assert result == ["B-028", "L-001"]

    def test_no_match(self):
        result = filter_wells_by_pad(["B-028"], ["E"])
        assert result == []

    def test_empty_wells(self):
        result = filter_wells_by_pad([], ["B"])
        assert result == []


# ── get_mpu_well_names (mocked) ───────────────────────────────────────────


class TestGetMPUWellNames:
    @patch("woffl.assembly.well_test_client.execute_query")
    def test_returns_well_names(self, mock_query):
        mock_query.return_value = pd.DataFrame(
            {"well_name": ["B-028", "E-041", "L-001"]}
        )
        result = get_mpu_well_names()
        assert result == ["B-028", "E-041", "L-001"]

    @patch("woffl.assembly.well_test_client.execute_query")
    def test_empty_result(self, mock_query):
        mock_query.return_value = pd.DataFrame()
        result = get_mpu_well_names()
        assert result == []


# ── fetch_milne_well_tests (mocked) ───────────────────────────────────────


class TestFetchMilneWellTests:
    @patch("woffl.assembly.well_test_client.execute_query")
    def test_column_mapping(self, mock_query):
        mock_query.return_value = pd.DataFrame(
            {
                "well_name": ["B-028", "B-028"],
                "wt_date": pd.to_datetime(["2024-01-15", "2024-02-20"]),
                "bhp": [800.0, 850.0],
                "oil_rate": [100.0, 120.0],
                "fwat_rate": [200.0, 250.0],
                "fgas_rate": [50.0, 60.0],
                "whp": [150.0, 160.0],
                "form_wc": [0.67, 0.68],
                "fgor": [500, 500],
                "lift_wat": [300.0, 350.0],
            }
        )
        df, dropped = fetch_milne_well_tests("2024-01-01", "2024-12-31", ["B-028"])
        assert "well" in df.columns  # renamed from well_name
        assert "WtDate" in df.columns  # renamed from wt_date
        assert "BHP" in df.columns  # renamed from bhp
        assert "WtOilVol" in df.columns  # renamed from oil_rate
        assert "WtWaterVol" in df.columns  # renamed from fwat_rate

    @patch("woffl.assembly.well_test_client.execute_query")
    def test_total_fluid_computed(self, mock_query):
        mock_query.return_value = pd.DataFrame(
            {
                "well_name": ["B-028"],
                "wt_date": pd.to_datetime(["2024-01-15"]),
                "bhp": [800.0],
                "oil_rate": [100.0],
                "fwat_rate": [200.0],
                "fgas_rate": [50.0],
                "whp": [150.0],
                "form_wc": [0.67],
                "fgor": [500],
                "lift_wat": [300.0],
            }
        )
        df, _ = fetch_milne_well_tests("2024-01-01", "2024-12-31", ["B-028"])
        assert "WtTotalFluid" in df.columns
        assert df["WtTotalFluid"].iloc[0] == pytest.approx(300.0)  # 100 + 200

    @patch("woffl.assembly.well_test_client.execute_query")
    def test_well_names_normalized(self, mock_query):
        mock_query.return_value = pd.DataFrame(
            {
                "well_name": ["B-028"],
                "wt_date": pd.to_datetime(["2024-01-15"]),
                "bhp": [800.0],
                "oil_rate": [100.0],
                "fwat_rate": [200.0],
                "fgas_rate": [50.0],
                "whp": [150.0],
                "form_wc": [0.67],
                "fgor": [500],
                "lift_wat": [300.0],
            }
        )
        df, _ = fetch_milne_well_tests("2024-01-01", "2024-12-31", ["B-028"])
        assert df["well"].iloc[0] == "MPB-28"

    @patch("woffl.assembly.well_test_client.execute_query")
    def test_dropped_wells_tracked(self, mock_query):
        mock_query.return_value = pd.DataFrame(
            {
                "well_name": ["B-028", "E-041"],
                "wt_date": pd.to_datetime(["2024-01-15", "2024-02-20"]),
                "bhp": [800.0, None],  # E-041 has no BHP
                "oil_rate": [100.0, 50.0],
                "fwat_rate": [200.0, 100.0],
                "fgas_rate": [50.0, 30.0],
                "whp": [150.0, 140.0],
                "form_wc": [0.67, 0.67],
                "fgor": [500, 500],
                "lift_wat": [300.0, 200.0],
            }
        )
        df, dropped = fetch_milne_well_tests(
            "2024-01-01", "2024-12-31", ["B-028", "E-041"]
        )
        assert "MPE-41" in dropped

    @patch("woffl.assembly.well_test_client.execute_query")
    def test_empty_result(self, mock_query):
        mock_query.return_value = pd.DataFrame()
        df, dropped = fetch_milne_well_tests("2024-01-01", "2024-12-31", ["B-028"])
        assert df.empty
        assert dropped == []
