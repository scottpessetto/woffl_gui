"""Tests for woffl.assembly.pf_pressure — live PF from vw_pressure_daily.

Covers the forward/reverse-circ resolution rule, the dead-gauge/zero guards,
the latest-per-well fetch (duplicate days, row integrity, name
normalization), and the annulus-only pad medians.
"""

from decimal import Decimal
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from woffl.assembly.pf_pressure import (
    PF_MIN_VALID,
    add_pf_columns,
    fetch_pf_latest,
    pad_pf_medians,
    resolve_pf_pressure,
)

# ── resolve_pf_pressure ────────────────────────────────────────────────────


class TestResolvePFPressure:
    def test_reverse_circ_standard_jp(self):
        # B-028 shape: production tubing ~165, PF annulus ~2522
        val, src = resolve_pf_pressure(165.6, 2522.7)
        assert val == pytest.approx(2522.7)
        assert src == "annulus"

    def test_forward_circ_tubing_is_pf(self):
        # F-002 shape: PF tubing ~2748, annulus ~140
        val, src = resolve_pf_pressure(2748.0, 140.0)
        assert val == pytest.approx(2748.0)
        assert src == "tubing"

    def test_forward_circ_with_null_annulus(self):
        # F-058 shape: annulus gauge absent entirely
        val, src = resolve_pf_pressure(2758.0, None)
        assert val == pytest.approx(2758.0)
        assert src == "tubing"

    def test_both_above_floor_tubing_higher(self):
        # S-17 shape: forward circ with a live annulus reading
        val, src = resolve_pf_pressure(3460.0, 830.0)
        assert val == pytest.approx(3460.0)
        assert src == "tubing"

    def test_both_above_floor_annulus_higher(self):
        val, src = resolve_pf_pressure(900.0, 3239.0)
        assert val == pytest.approx(3239.0)
        assert src == "annulus"

    def test_tie_resolves_to_annulus(self):
        # Not strictly greater -> reverse circ (the standard configuration)
        val, src = resolve_pf_pressure(2000.0, 2000.0)
        assert val == pytest.approx(2000.0)
        assert src == "annulus"

    def test_production_tubing_alone_is_not_pf(self):
        # Reverse-circ well with a dead annulus gauge: tubing ~170 is
        # production, NOT power fluid — must resolve to nothing.
        val, src = resolve_pf_pressure(170.0, None)
        assert val is None
        assert src is None

    def test_dead_gauges_zero(self):
        assert resolve_pf_pressure(0.0, 0.0) == (None, None)

    def test_both_null(self):
        assert resolve_pf_pressure(None, None) == (None, None)
        assert resolve_pf_pressure(np.nan, np.nan) == (None, None)

    def test_decimal_inputs_cast_to_float(self):
        # Databricks returns decimal(10,3)
        val, src = resolve_pf_pressure(Decimal("163.367"), Decimal("2719.763"))
        assert isinstance(val, float)
        assert val == pytest.approx(2719.763)
        assert src == "annulus"

    def test_floor_is_pf_min_valid(self):
        just_below = PF_MIN_VALID - 0.5
        assert resolve_pf_pressure(None, just_below) == (None, None)
        val, src = resolve_pf_pressure(None, PF_MIN_VALID)
        assert val == pytest.approx(PF_MIN_VALID)
        assert src == "annulus"

    def test_non_numeric_garbage(self):
        assert resolve_pf_pressure("n/a", "None") == (None, None)


# ── add_pf_columns ─────────────────────────────────────────────────────────


class TestAddPFColumns:
    def test_resolves_per_row(self):
        df = pd.DataFrame(
            {
                "pf_tubing_prs": [165.0, 2748.0, 0.0],
                "pf_inn_ann_prs": [2522.0, 140.0, 0.0],
            }
        )
        out = add_pf_columns(df)
        assert out["pf_press"].tolist()[:2] == [2522.0, 2748.0]
        assert pd.isna(out["pf_press"].iloc[2])
        assert out["pf_source"].tolist()[:2] == ["annulus", "tubing"]
        assert out["pf_source"].iloc[2] is None

    def test_missing_columns_tolerated(self):
        # Mocked/manual test frames without the vw_pressure_daily join still
        # get uniform (all-NaN) pf columns.
        df = pd.DataFrame({"well": ["MPB-28"]})
        out = add_pf_columns(df)
        assert "pf_press" in out.columns
        assert "pf_source" in out.columns
        assert out["pf_press"].isna().all()


# ── fetch_pf_latest (mocked) ───────────────────────────────────────────────


def _daily(well, date, tub, ann):
    return {
        "well_name": well,
        "sample_date": date,
        "tubing_prs": tub,
        "inn_ann_prs": ann,
    }


class TestFetchPFLatest:
    @patch("woffl.assembly.pf_pressure.execute_query")
    def test_latest_valid_row_per_well(self, mock_query):
        mock_query.return_value = pd.DataFrame(
            [
                _daily("B-028", "2026-06-30", 165.0, 2500.0),
                _daily("B-028", "2026-07-02", 163.0, 2719.0),
                # newest day is a dead gauge — the older valid day must win
                _daily("B-028", "2026-07-03", 0.0, 0.0),
                _daily("F-058", "2026-07-03", Decimal("2757.999"), None),
            ]
        )
        df = fetch_pf_latest()
        assert set(df["well"]) == {"MPB-28", "MPF-58"}
        b28 = df[df["well"] == "MPB-28"].iloc[0]
        assert b28["pf_press"] == pytest.approx(2719.0)
        assert b28["pf_source"] == "annulus"
        assert b28["pf_date"] == pd.Timestamp("2026-07-02")
        # row integrity: the tubing value must be from the SAME day
        assert b28["tubing_prs"] == pytest.approx(163.0)
        f58 = df[df["well"] == "MPF-58"].iloc[0]
        assert f58["pf_press"] == pytest.approx(2757.999)
        assert f58["pf_source"] == "tubing"

    @patch("woffl.assembly.pf_pressure.execute_query")
    def test_well_with_no_valid_reading_absent(self, mock_query):
        mock_query.return_value = pd.DataFrame(
            [
                _daily("I-003", "2026-07-03", 0.0, 50.0),  # dead / non-JP
                _daily("B-028", "2026-07-03", 165.0, 2719.0),
            ]
        )
        df = fetch_pf_latest()
        assert set(df["well"]) == {"MPB-28"}

    @patch("woffl.assembly.pf_pressure.execute_query")
    def test_empty_result(self, mock_query):
        mock_query.return_value = pd.DataFrame()
        df = fetch_pf_latest()
        assert df.empty
        assert "pf_press" in df.columns


# ── pad_pf_medians ─────────────────────────────────────────────────────────


class TestPadPFMedians:
    def _latest(self):
        return pd.DataFrame(
            {
                "well": ["MPJ-27", "MPJ-29", "MPJ-32", "MPJ-2", "MPB-28"],
                "pf_press": [2483.0, 2609.0, 2558.0, 832.0, 2719.0],
                # MPJ-2 is an ESP whose tubing pressure cleared the floor
                "pf_source": ["annulus", "annulus", "annulus", "tubing", "annulus"],
            }
        )

    def test_annulus_only_excludes_esp_tubing(self):
        medians = pad_pf_medians(self._latest())
        # J median over annulus wells = 2558 -> rounds to 2550
        assert medians["J"] == 2550
        assert medians["B"] == 2700

    def test_all_sources_when_asked(self):
        medians = pad_pf_medians(self._latest(), sources=())
        # median(2483, 2609, 2558, 832) = 2520.5 -> 2500
        assert medians["J"] == 2500

    def test_empty(self):
        assert pad_pf_medians(pd.DataFrame()) == {}
        assert pad_pf_medians(None) == {}
