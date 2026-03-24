"""Tests for Databricks client with mocked database calls."""

import io
from unittest.mock import patch

import pandas as pd
import pytest

from woffl.assembly.databricks_client import (
    fetch_jp_history,
    get_tags_for_wells,
    load_tag_dict,
    query_bhp_for_well_tests,
)

# ── get_tags_for_wells ──────────────────────────────────────────────────────


class TestGetTagsForWells:
    @staticmethod
    def _tag_dict():
        return {
            "MPB-28": ("bhp_b28", "hp_b28", "whp_b28"),
            "MPE-41": ("bhp_e41", "hp_e41", "whp_e41"),
        }

    def test_all_found(self):
        found, missing = get_tags_for_wells(["MPB-28", "MPE-41"], self._tag_dict())
        assert len(found) == 2
        assert missing == []

    def test_some_missing(self):
        found, missing = get_tags_for_wells(["MPB-28", "FAKE-99"], self._tag_dict())
        assert "MPB-28" in found
        assert "FAKE-99" in missing

    def test_all_missing(self):
        found, missing = get_tags_for_wells(["FAKE-1", "FAKE-2"], self._tag_dict())
        assert found == {}
        assert len(missing) == 2

    def test_empty_wells(self):
        found, missing = get_tags_for_wells([], self._tag_dict())
        assert found == {}
        assert missing == []

    def test_tag_tuple_structure(self):
        found, _ = get_tags_for_wells(["MPB-28"], self._tag_dict())
        bhp, hp, whp = found["MPB-28"]
        assert bhp == "bhp_b28"
        assert hp == "hp_b28"
        assert whp == "whp_b28"


# ── load_tag_dict ───────────────────────────────────────────────────────────


class TestLoadTagDict:
    def test_custom_source_stringio(self):
        csv = "wellname,bhp_tag,headerP_tag,whp_tag\nMPB-28,bhp_b28,hp_b28,whp_b28\nMPE-41,bhp_e41,hp_e41,whp_e41\n"
        result = load_tag_dict(custom_source=io.StringIO(csv))
        assert "MPB-28" in result
        assert result["MPB-28"] == ("bhp_b28", "hp_b28", "whp_b28")
        assert "MPE-41" in result

    def test_custom_source_has_all_wells(self):
        csv = "wellname,bhp_tag,headerP_tag,whp_tag\nW1,a,b,c\nW2,d,e,f\n"
        result = load_tag_dict(custom_source=io.StringIO(csv))
        assert len(result) == 2


# ── fetch_jp_history (mocked) ──────────────────────────────────────────────


class TestFetchJPHistory:
    @patch("woffl.assembly.databricks_client.execute_query")
    def test_date_columns_are_datetime(self, mock_query):
        mock_query.return_value = pd.DataFrame(
            {
                "Well Name": ["MPB-28 ", "MPE-41"],
                "Date Set": ["2024-01-15", "2024-02-20"],
                "Date Pulled": ["2024-06-01", None],
                "Nozzle Number": [12, 13],
                "Throat Ratio": ["A", "B"],
                "Tubing Diameter": [4.5, 4.5],
            }
        )
        result = fetch_jp_history()
        assert pd.api.types.is_datetime64_any_dtype(result["Date Set"])

    @patch("woffl.assembly.databricks_client.execute_query")
    def test_well_name_stripped(self, mock_query):
        mock_query.return_value = pd.DataFrame(
            {
                "Well Name": ["  MPB-28  ", "MPE-41 "],
                "Date Set": ["2024-01-15", "2024-02-20"],
                "Nozzle Number": [12, 13],
                "Throat Ratio": ["A", "B"],
            }
        )
        result = fetch_jp_history()
        assert result["Well Name"].iloc[0] == "MPB-28"
        assert result["Well Name"].iloc[1] == "MPE-41"

    @patch("woffl.assembly.databricks_client.execute_query")
    def test_returns_dataframe(self, mock_query):
        mock_query.return_value = pd.DataFrame(
            {
                "Well Name": ["MPB-28"],
                "Date Set": ["2024-01-15"],
                "Nozzle Number": [12],
                "Throat Ratio": ["A"],
            }
        )
        result = fetch_jp_history()
        assert isinstance(result, pd.DataFrame)


# ── query_bhp_for_well_tests (mocked) ──────────────────────────────────────


class TestQueryBhpForWellTests:
    @patch("woffl.assembly.databricks_client.execute_query")
    def test_returns_pivoted_data(self, mock_query):
        tag_dict = {
            "MPB-28": ("bhp_b28", "hp_b28", "whp_b28"),
        }
        mock_query.return_value = pd.DataFrame(
            {
                "date": ["2024-01-15", "2024-01-15", "2024-01-15"],
                "tag": ["bhp_b28", "hp_b28", "whp_b28"],
                "max_average_value": [800.0, 200.0, 150.0],
            }
        )
        result = query_bhp_for_well_tests(tag_dict, ["MPB-28"])
        assert "MPB-28" in result
        df = result["MPB-28"]
        assert "BHP" in df.columns

    @patch("woffl.assembly.databricks_client.execute_query")
    def test_empty_tag_dict_returns_empty(self, mock_query):
        result = query_bhp_for_well_tests({}, ["MPB-28"])
        assert result == {}
        mock_query.assert_not_called()

    @patch("woffl.assembly.databricks_client.execute_query")
    def test_no_matching_wells_returns_empty(self, mock_query):
        tag_dict = {"MPB-28": ("bhp_b28", "hp_b28", "whp_b28")}
        # query_bhp_for_well_tests filters by well_list against tag_dict
        result = query_bhp_for_well_tests(tag_dict, ["FAKE-99"])
        assert result == {}
