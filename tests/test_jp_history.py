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
            # Enrichment passthroughs (pump_identity) — None on plain frames.
            "circ_direction",
            "manufacturer",
            "raw_pump",
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


# ── pump_ages: the Well Database "aging jet pumps" list (2026-07-31) ────────
# Tenure is set-to-set (the JPCO rule) — age of the CURRENT pump is
# today − its Date Set; Date Pulled is never consulted.


class TestPumpAges:
    def test_ages_current_pump_from_latest_date_set(self):
        from woffl.assembly.jp_history import pump_ages

        ages = pump_ages(_make_jp_history(), today="2026-07-31")
        row = ages[ages["Well Name"] == "MPB-28"].iloc[0]
        # latest install 2024-01-10 → 933 days to 2026-07-31
        assert row["Date Set"] == pd.Timestamp("2024-01-10")
        assert row["Days In Hole"] == (
            pd.Timestamp("2026-07-31") - pd.Timestamp("2024-01-10")
        ).days
        assert row["Installs"] == 3

    def test_sorted_oldest_first(self):
        from woffl.assembly.jp_history import pump_ages

        ages = pump_ages(_make_jp_history(), today="2026-07-31")
        days = list(ages["Days In Hole"])
        assert days == sorted(days, reverse=True)

    def test_date_pulled_never_consulted(self):
        """Blanking every Date Pulled must change nothing — the JPCO rule."""
        from woffl.assembly.jp_history import pump_ages

        hist = _make_jp_history()
        blanked = hist.copy()
        blanked["Date Pulled"] = pd.NaT
        a = pump_ages(hist, today="2026-07-31")
        b = pump_ages(blanked, today="2026-07-31")
        pd.testing.assert_frame_equal(a, b)

    def test_rows_without_date_set_ignored(self):
        from woffl.assembly.jp_history import pump_ages

        hist = _make_jp_history()
        extra = hist.iloc[[0]].copy()
        extra["Date Set"] = pd.NaT
        ages = pump_ages(pd.concat([hist, extra], ignore_index=True),
                         today="2026-07-31")
        assert ages[ages["Well Name"] == "MPB-28"].iloc[0]["Installs"] == 3

    def test_empty_in_empty_out(self):
        from woffl.assembly.jp_history import pump_ages

        assert pump_ages(None).empty
        assert pump_ages(pd.DataFrame()).empty


class TestFilterRecentlyOnline:
    """The aging list's online filter — latest allocated test as the proxy."""

    def _ages(self):
        from woffl.assembly.jp_history import pump_ages

        return pump_ages(_make_jp_history(), today="2026-07-31")

    def test_keeps_only_wells_with_a_recent_test(self):
        from woffl.assembly.jp_history import filter_recently_online

        last = {
            "MPB-28": pd.Timestamp("2026-07-20"),   # 11 days ago — online
            "MPE-41": pd.Timestamp("2026-03-01"),   # months stale
        }
        out = filter_recently_online(self._ages(), last, days=60,
                                     today="2026-07-31")
        assert list(out["Well Name"]) == ["MPB-28"]
        assert out.iloc[0]["Last Test"] == pd.Timestamp("2026-07-20")

    def test_wells_with_no_test_are_dropped(self):
        from woffl.assembly.jp_history import filter_recently_online

        last = {"MPB-28": pd.Timestamp("2026-07-20")}  # MPE-41 unknown
        out = filter_recently_online(self._ages(), last, days=60,
                                     today="2026-07-31")
        assert "MPE-41" not in set(out["Well Name"])

    def test_empty_map_returns_unfiltered_not_empty(self):
        """Source unavailable must not silently drop every well."""
        from woffl.assembly.jp_history import filter_recently_online

        ages = self._ages()
        out = filter_recently_online(ages, {}, days=60, today="2026-07-31")
        assert len(out) == len(ages)

    def test_window_boundary_is_inclusive(self):
        from woffl.assembly.jp_history import filter_recently_online

        last = {"MPB-28": pd.Timestamp("2026-06-01")}  # exactly 60 days
        out = filter_recently_online(self._ages(), last, days=60,
                                     today="2026-07-31")
        assert list(out["Well Name"]) == ["MPB-28"]

    def test_tz_aware_test_dates_from_databricks(self):
        """Regression (live crash 2026-07-31): vw_well_test dates arrive
        tz-AWARE (Etc/UTC) while the cutoff is naive — pandas raises
        'Invalid comparison between dtype=datetime64[ns, Etc/UTC] and
        Timestamp'. The filter must normalize both sides."""
        from woffl.assembly.jp_history import filter_recently_online

        last = {
            "MPB-28": pd.Timestamp("2026-07-20 06:30", tz="Etc/UTC"),
            "MPE-41": pd.Timestamp("2026-01-05 12:00", tz="Etc/UTC"),
        }
        out = filter_recently_online(self._ages(), last, days=60,
                                     today="2026-07-31")
        assert list(out["Well Name"]) == ["MPB-28"]
        # column comes back tz-naive, ready for display + Excel export
        assert out["Last Test"].dt.tz is None
