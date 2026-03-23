"""Tests for WellTestProcessor and merge_tests_with_bhp."""

import io

import pandas as pd
import pytest

from woffl.assembly.well_test_processor import WellTestProcessor, merge_tests_with_bhp

# --- Shared test CSV data ---
SAMPLE_CSV = """\
EntName1,WtDate,WtOilVol,WtWaterVol,WtTotalFluid,WtGasVol,WtGasRate,WtGasLiftVol,IA,WtrLift,BHP,RouteGroupName,WtHours,Choke
B-028A,2024-01-15,100,200,300,50,10,0,5,0,800,GroupA,24,32
B-028A,2024-02-20,"1,234",500,"1,734",80,20,0,10,0,850,GroupA,24,32
E-041,2024-01-10,150,300,450,60,15,0,7,0,900,GroupB,24,40
B-003X,2024-03-01,80,120,200,30,8,0,3,0,750,GroupC,24,28
"""


def _make_processor():
    return WellTestProcessor(io.StringIO(SAMPLE_CSV))


# ── Parse ───────────────────────────────────────────────────────────────────


class TestParse:
    def test_well_name_b028(self):
        df = _make_processor().parse()
        wells = df["well"].unique()
        assert "MPB-28" in wells

    def test_well_name_e041(self):
        df = _make_processor().parse()
        assert "MPE-41" in df["well"].values

    def test_well_name_b003_leading_zero_removed(self):
        """Regex strips one leading zero: 003 -> 03 (not 3)."""
        df = _make_processor().parse()
        assert "MPB-03" in df["well"].values
        assert "MPB-003" not in df["well"].values

    def test_numeric_comma_cleaning(self):
        df = _make_processor().parse()
        b28 = df[df["well"] == "MPB-28"]
        assert 1234.0 in b28["WtOilVol"].values

    def test_nan_filled_to_zero(self):
        csv = "EntName1,WtDate,WtOilVol,WtWaterVol,WtTotalFluid,IA\nB-028,2024-01-01,,,100,\n"
        df = WellTestProcessor(io.StringIO(csv)).parse()
        assert df["WtOilVol"].iloc[0] == 0
        assert df["IA"].iloc[0] == 0

    def test_dropped_columns(self):
        df = _make_processor().parse()
        for col in ["BHP", "RouteGroupName", "EntName1", "WtHours", "Choke"]:
            assert col not in df.columns

    def test_wtdate_is_datetime(self):
        df = _make_processor().parse()
        assert pd.api.types.is_datetime64_any_dtype(df["WtDate"])

    def test_row_count_preserved(self):
        df = _make_processor().parse()
        assert len(df) == 4

    def test_well_column_exists(self):
        df = _make_processor().parse()
        assert "well" in df.columns


# ── Unique wells ────────────────────────────────────────────────────────────


class TestGetUniqueWells:
    def test_returns_sorted_list(self):
        proc = _make_processor()
        proc.parse()
        wells = proc.get_unique_wells()
        assert wells == sorted(wells)

    def test_correct_wells(self):
        proc = _make_processor()
        proc.parse()
        wells = proc.get_unique_wells()
        assert set(wells) == {"MPB-28", "MPE-41", "MPB-03"}

    def test_auto_parses(self):
        proc = _make_processor()
        wells = proc.get_unique_wells()  # no explicit parse()
        assert len(wells) > 0


# ── Well test filtering ────────────────────────────────────────────────────


class TestGetWellTests:
    def test_filter_single_well(self):
        proc = _make_processor()
        proc.parse()
        df = proc.get_well_tests(wells=["MPB-28"])
        assert all(df["well"] == "MPB-28")

    def test_filter_multiple_wells(self):
        proc = _make_processor()
        proc.parse()
        df = proc.get_well_tests(wells=["MPB-28", "MPE-41"])
        assert set(df["well"].unique()) == {"MPB-28", "MPE-41"}

    def test_none_returns_all(self):
        proc = _make_processor()
        proc.parse()
        df = proc.get_well_tests(wells=None)
        assert len(df) == 4

    def test_nonexistent_well_returns_empty(self):
        proc = _make_processor()
        proc.parse()
        df = proc.get_well_tests(wells=["FAKE-99"])
        assert df.empty

    def test_returns_copy(self):
        proc = _make_processor()
        proc.parse()
        df1 = proc.get_well_tests()
        df2 = proc.get_well_tests()
        assert df1 is not df2


# ── Date range ──────────────────────────────────────────────────────────────


class TestGetTestDateRange:
    def test_returns_timestamps(self):
        proc = _make_processor()
        proc.parse()
        lo, hi = proc.get_test_date_range()
        assert isinstance(lo, pd.Timestamp)
        assert isinstance(hi, pd.Timestamp)

    def test_correct_range(self):
        proc = _make_processor()
        proc.parse()
        lo, hi = proc.get_test_date_range()
        assert lo == pd.Timestamp("2024-01-10")
        assert hi == pd.Timestamp("2024-03-01")


# ── Test count ──────────────────────────────────────────────────────────────


class TestGetTestCountPerWell:
    def test_correct_counts(self):
        proc = _make_processor()
        proc.parse()
        counts = proc.get_test_count_per_well()
        assert counts["MPB-28"] == 2
        assert counts["MPE-41"] == 1
        assert counts["MPB-03"] == 1

    def test_returns_series(self):
        proc = _make_processor()
        proc.parse()
        counts = proc.get_test_count_per_well()
        assert isinstance(counts, pd.Series)


# ── Merge with BHP ─────────────────────────────────────────────────────────


class TestMergeTestsWithBhp:
    @staticmethod
    def _make_bhp_data():
        """Create sample BHP data dict."""
        dates = pd.to_datetime(["2024-01-15", "2024-02-20", "2024-03-01"])
        bhp_df = pd.DataFrame(
            {
                "BHP": [800, 850, 900],
                "HeaderP": [200, 210, 220],
                "WHP": [150, 160, 170],
            },
            index=dates,
        )
        bhp_df.index.name = "date"
        return {"MPB-28": bhp_df}

    @staticmethod
    def _make_well_tests():
        proc = _make_processor()
        return proc.parse()

    def test_successful_merge(self):
        bhp = self._make_bhp_data()
        wt = self._make_well_tests()
        merged = merge_tests_with_bhp(["MPB-28"], bhp, wt)
        assert not merged.empty
        assert "BHP" in merged.columns

    def test_well_not_in_bhp_skipped(self):
        bhp = self._make_bhp_data()
        wt = self._make_well_tests()
        merged = merge_tests_with_bhp(["MPE-41"], bhp, wt)
        assert merged.empty

    def test_empty_bhp_data(self):
        wt = self._make_well_tests()
        merged = merge_tests_with_bhp(["MPB-28"], {}, wt)
        assert merged.empty

    def test_no_overlapping_dates(self):
        dates = pd.to_datetime(["2025-06-01", "2025-07-01"])
        bhp_df = pd.DataFrame({"BHP": [100, 200]}, index=dates)
        bhp_df.index.name = "date"
        wt = self._make_well_tests()
        merged = merge_tests_with_bhp(["MPB-28"], {"MPB-28": bhp_df}, wt)
        assert merged.empty

    def test_merge_date_column_dropped(self):
        bhp = self._make_bhp_data()
        wt = self._make_well_tests()
        merged = merge_tests_with_bhp(["MPB-28"], bhp, wt)
        assert "merge_date" not in merged.columns

    def test_tz_aware_wtdate(self):
        """Timezone-aware WtDate should still merge correctly."""
        bhp = self._make_bhp_data()
        wt = self._make_well_tests()
        wt["WtDate"] = wt["WtDate"].dt.tz_localize("UTC")
        merged = merge_tests_with_bhp(["MPB-28"], bhp, wt)
        assert not merged.empty
