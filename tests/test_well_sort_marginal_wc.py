"""Tests for the canonical field marginal-WC calculator (R-8 / P1-30).

Pins the semantics of the single canonical implementation living in
``woffl.gui.scotts_tools.well_sort.compute_field_marginal_wc``: a
cumulative-water-threshold walk (with a buffer/threshold knob) over
online non-POPS wells. A second, divergent max-per-well-WC implementation
used to live in ``woffl.assembly.well_sort_client`` — it was deleted, and
``woffl.gui.workflow_steps.step3_configure_optimize`` was repointed at
this one, so Step-3's auto-fill, the Well Sort tab, the Batch-Run
"Import" button, and Triage all now agree on one number (see
docs/code_review_2026-07-01.md R-8).

Bypasses Databricks/session-state entirely by monkeypatching
``_build_online_full`` (the shared cached-data-assembly helper) with a
small synthetic frame, so these tests exercise the real threshold-walk +
POPS-exclusion math end to end without a warehouse connection.
"""

import pandas as pd
import pytest

import woffl.gui.scotts_tools.well_sort as well_sort


def _synthetic_online_full() -> pd.DataFrame:
    """5 "online" wells: 4 non-POPS + 1 POPS well with a dominating WC/water.

    Non-POPS, sorted by TotalWC descending:
      W1  0.99 WC,    50 BWPD  (stripper noise — tiny volume, huge WC)
      W2  0.95 WC,   500 BWPD
      W3  0.90 WC,  1000 BWPD
      W4  0.50 WC,  2000 BWPD
    Non-POPS field water = 3550 BWPD. Cumulative-water % from the top:
      W1  50 / 3550   =  1.408%
      W2  550 / 3550  = 15.493%
      W3  1550 / 3550 = 43.662%
      W4  3550 / 3550 = 100.0%

    W5-POPS carries a 0.999 WC and 100,000 BWPD — if POPS exclusion were
    silently dropped, it would swamp both the field-water total and the
    marginal-well pick. It must never appear in the result.
    """
    return pd.DataFrame(
        {
            "Well": ["W1", "W2", "W3", "W4", "W5-POPS"],
            "Pad": ["A", "B", "C", "D", "E"],
            "TotalWC": [0.99, 0.95, 0.90, 0.50, 0.999],
            "TotalWater": [50.0, 500.0, 1000.0, 2000.0, 100_000.0],
            "PopsPad": [False, False, False, False, True],
        }
    )


@pytest.fixture(autouse=True)
def _patch_online_full(monkeypatch):
    """Swap the Databricks-backed builder for the synthetic frame above."""
    monkeypatch.setattr(
        well_sort,
        "_build_online_full",
        lambda stale_days=60: _synthetic_online_full(),
    )


class TestPopsExclusion:
    def test_pops_well_excluded_from_field_water_and_candidates(self):
        result = well_sort.compute_field_marginal_wc(threshold_pct=2.0)
        assert result is not None
        assert result["well_count"] == 4  # W5-POPS excluded from the count
        assert result["total_field_water"] == pytest.approx(3550.0)
        assert "W5-POPS" not in result["ranked_df"]["Well"].tolist()

    def test_pops_well_never_picked_as_marginal_even_with_huge_wc(self):
        # Even at a threshold that would walk all the way to the bottom of
        # the (non-POPS) list, the POPS well - with by far the highest WC
        # and water volume - must never be selected.
        result = well_sort.compute_field_marginal_wc(threshold_pct=150.0)
        assert result["well"] != "W5-POPS"


class TestThresholdWalk:
    """The marginal well is the first (worst-WC-first) well at which
    cumulative water crosses the threshold % of (non-POPS) field water —
    not simply the single worst-WC well."""

    @pytest.mark.parametrize(
        "threshold_pct,expected_well,expected_wc",
        [
            (1.0, "W1", 0.99),  # crosses 1% within W1 itself
            (2.0, "W2", 0.95),  # crosses 2% at W2 (rejects W1 as noise)
            (20.0, "W3", 0.90),  # crosses 20% at W3
            (50.0, "W4", 0.50),  # crosses 50% at W4
        ],
    )
    def test_threshold_walk_picks_correct_marginal_well(
        self, threshold_pct, expected_well, expected_wc
    ):
        result = well_sort.compute_field_marginal_wc(threshold_pct=threshold_pct)
        assert result["well"] == expected_well
        assert result["marginal_wc"] == pytest.approx(expected_wc)

    def test_threshold_above_100pct_falls_back_to_worst_wc_well(self):
        """Defensive fallback: an (impossible-by-construction) threshold
        above 100% of field water falls back to the bottom of the ranked
        list rather than raising or returning a bogus index."""
        result = well_sort.compute_field_marginal_wc(threshold_pct=150.0)
        assert result["well"] == "W4"
        assert result["marginal_wc"] == pytest.approx(0.50)
        assert result["marg_idx"] == len(result["ranked_df"]) - 1


class TestBufferKnob:
    def test_stricter_threshold_yields_lower_marginal_wc(self):
        """The whole point of the threshold walk: a stricter (higher)
        cumulative-water threshold rejects more single-well noise and
        lands on a LOWER marginal WC than a looser (lower) threshold."""
        loose = well_sort.compute_field_marginal_wc(threshold_pct=1.0)
        strict = well_sort.compute_field_marginal_wc(threshold_pct=50.0)
        assert loose["marginal_wc"] > strict["marginal_wc"]

    def test_threshold_pct_echoed_in_result(self):
        result = well_sort.compute_field_marginal_wc(threshold_pct=7.5)
        assert result["threshold_pct"] == pytest.approx(7.5)


class TestEmptyInput:
    def test_no_online_non_pops_wells_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            well_sort,
            "_build_online_full",
            lambda stale_days=60: pd.DataFrame(
                columns=["Well", "Pad", "TotalWC", "TotalWater", "PopsPad"]
            ),
        )
        assert well_sort.compute_field_marginal_wc(threshold_pct=2.0) is None

    def test_all_pops_wells_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            well_sort,
            "_build_online_full",
            lambda stale_days=60: pd.DataFrame(
                {
                    "Well": ["W5-POPS"],
                    "Pad": ["E"],
                    "TotalWC": [0.999],
                    "TotalWater": [100_000.0],
                    "PopsPad": [True],
                }
            ),
        )
        assert well_sort.compute_field_marginal_wc(threshold_pct=2.0) is None
