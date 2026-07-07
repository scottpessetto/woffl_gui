"""Tests for the pad-level marginal-WC auto-derivation + parsimony tie-break.

Feature: the pad optimizer's marginal-watercut gate is now DERIVED from the
pad's own physical limits (booster-pump curves / PF recycle budget) instead
of a hand-set threshold, plus a parsimony tie-break so PF slack isn't spent
on noise-level oil gains (the field case: a well upsized 13C->15B for ~2
BOPD at +1,500 BPD PF).

``derive_pad_marginal_wc`` pools every well's oil-per-water Pareto frontier
into marginal segments, sorts them best-ratio-first, and spends the plant's
budget on the best ratios first — the gate is the ratio of the segment that
crosses the budget. ``apply_parsimony`` then looks, per well, for a
less-water config that gives up no more than a threshold's worth of oil.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from woffl.assembly.network_optimizer import OptimizationResult
from woffl.assembly.optimization_algorithms import (
    apply_parsimony,
    derive_pad_marginal_wc,
)

# ── derive_pad_marginal_wc helpers ──────────────────────────────────────────


def _df(rows, water_key="lift_wat"):
    """rows: (nozzle, throat, qoil_std, water, sonic_status)."""
    out = []
    for nozzle, throat, qoil, water, sonic in rows:
        out.append(
            {
                "nozzle": nozzle,
                "throat": throat,
                "qoil_std": qoil,
                water_key: water,
                "sonic_status": sonic,
                "error": "na",
            }
        )
    return pd.DataFrame(out)


def _bp(df):
    return SimpleNamespace(df=df)


class TestDeriveFrontierAndCrossing:
    def test_binding_cap_hand_computed_gate(self):
        # WellX frontier: (1000,100) ratio .1; (3000,280) -> segment
        # dw=2000, doil=180, ratio .09
        well_x = _df(
            [
                ("10", "A", 100.0, 1000.0, False),
                ("11", "A", 280.0, 3000.0, False),
            ]
        )
        # WellY frontier: (500,60) ratio .12; (1500,100) -> segment
        # dw=1000, doil=40, ratio .04
        well_y = _df(
            [
                ("10", "A", 60.0, 500.0, False),
                ("11", "A", 100.0, 1500.0, False),
            ]
        )
        batch_results = {"WellX": _bp(well_x), "WellY": _bp(well_y)}
        # pooled ratio order: Y1(.12,500) X1(.1,1000) X2(.09,2000) Y2(.04,1000)
        # cap=3200: 500 + 1000 = 1500 fits; +2000 = 3500 crosses X2 (ratio .09)
        gate, slack = derive_pad_marginal_wc(
            batch_results, cap=3200.0, water_key="lift_wat"
        )
        assert slack is False
        assert gate == pytest.approx(1.0 / 1.09)

    def test_slack_when_every_segment_fits(self):
        well_x = _df([("10", "A", 100.0, 1000.0, False)])
        gate, slack = derive_pad_marginal_wc(
            {"WellX": _bp(well_x)}, cap=10000.0, water_key="lift_wat"
        )
        assert (gate, slack) == (1.0, True)

    def test_degenerate_all_error_dfs_returns_slack(self):
        df = pd.DataFrame(
            [
                {
                    "nozzle": "10",
                    "throat": "A",
                    "qoil_std": float("nan"),
                    "lift_wat": 1000.0,
                    "error": repr(ValueError("solver failed")),
                }
            ]
        )
        gate, slack = derive_pad_marginal_wc(
            {"WellX": _bp(df)}, cap=5000.0, water_key="lift_wat"
        )
        assert (gate, slack) == (1.0, True)

    def test_empty_batch_results(self):
        assert derive_pad_marginal_wc({}, cap=5000.0, water_key="lift_wat") == (
            1.0,
            True,
        )

    def test_nonpositive_cap_returns_slack_by_design(self):
        # Documented choice (see derive_pad_marginal_wc's docstring): cap<=0
        # is a degenerate BUDGET, not an economics decision — the MILP/MCKP
        # water-budget constraint already zeroes everything out, so the gate
        # itself stays 1.0 (no double-penalizing / misreporting "above
        # marginal WC" for a well actually cut by the budget).
        well_x = _df([("10", "A", 100.0, 1000.0, False)])
        assert derive_pad_marginal_wc(
            {"WellX": _bp(well_x)}, cap=0.0, water_key="lift_wat"
        ) == (1.0, True)
        assert derive_pad_marginal_wc(
            {"WellX": _bp(well_x)}, cap=-5.0, water_key="lift_wat"
        ) == (1.0, True)

    def test_non_pareto_configs_excluded_from_segments(self):
        # config (2000, 50) is dominated by (1000, 100) -- more water AND
        # less oil -- must not appear on the frontier / contribute a segment.
        well_x = _df(
            [
                ("10", "A", 100.0, 1000.0, False),
                ("11", "A", 50.0, 2000.0, False),  # dominated -- excluded
                ("12", "A", 150.0, 4000.0, False),
            ]
        )
        # Real frontier segments: (1000,100) ratio .1; (4000,150) -> a
        # second segment dw=3000, doil=50, ratio 1/60. Total water = 4000.
        gate, slack = derive_pad_marginal_wc(
            {"WellX": _bp(well_x)}, cap=4000.0, water_key="lift_wat"
        )
        # cap covers BOTH real segments -> no crossing -> slack
        assert (gate, slack) == (1.0, True)

        gate2, slack2 = derive_pad_marginal_wc(
            {"WellX": _bp(well_x)}, cap=2000.0, water_key="lift_wat"
        )
        # crosses on the SECOND real segment (ratio 1/60), proof the
        # dominated (2000,50) config never contributed a segment of its own
        # (which would have crossed at a different ratio/point).
        assert slack2 is False
        assert gate2 == pytest.approx(1.0 / (1.0 + 50.0 / 3000.0))

    def test_water_key_totl_wat_does_not_need_molwr(self):
        # Confirms the derivation reads df[water_key]/qoil_std directly and
        # has NO dependency on the batch df's molwr/motwr marginal-ratio
        # columns at all (unlike the MILP/MCKP gate above it in the module).
        df = pd.DataFrame(
            [
                {
                    "nozzle": "10",
                    "throat": "A",
                    "qoil_std": 100.0,
                    "totl_wat": 1200.0,
                    "error": "na",
                },
                {
                    "nozzle": "11",
                    "throat": "A",
                    "qoil_std": 200.0,
                    "totl_wat": 3200.0,
                    "error": "na",
                },
            ]
        )
        assert "molwr" not in df.columns and "motwr" not in df.columns
        # frontier segments: (1200,100) ratio 1/12; (3200,200) -> dw=2000,
        # doil=100, ratio .05. Total water = 3200 -> a cap that big -> slack.
        gate, slack = derive_pad_marginal_wc(
            {"WellX": _bp(df)}, cap=3200.0, water_key="totl_wat"
        )
        assert (gate, slack) == (1.0, True)

    def test_bare_dataframe_value_accepted(self):
        # batch_results values may be a raw DataFrame (not just a
        # BatchPump-like object with a .df attribute).
        df = _df([("10", "A", 100.0, 1000.0, False)])
        gate, slack = derive_pad_marginal_wc(
            {"WellX": df}, cap=10000.0, water_key="lift_wat"
        )
        assert (gate, slack) == (1.0, True)


# ── apply_parsimony ─────────────────────────────────────────────────────────


def _batch_df(rows):
    """rows: (nozzle, throat, qoil_std, lift_wat, sonic_status)."""
    out = []
    for nozzle, throat, qoil, lift, sonic in rows:
        out.append(
            {
                "nozzle": nozzle,
                "throat": throat,
                "qoil_std": qoil,
                "lift_wat": lift,
                "form_wat": 0.0,
                "totl_wat": lift,
                "psu_solv": 1000.0,
                "sonic_status": sonic,
                "mach_te": 0.5,
                "error": "na",
            }
        )
    return pd.DataFrame(out)


def _result(well, nozzle, throat, oil, lift_water, sonic=False):
    return OptimizationResult(
        well_name=well,
        recommended_nozzle=nozzle,
        recommended_throat=throat,
        allocated_power_fluid=lift_water,
        predicted_oil_rate=oil,
        predicted_formation_water=0.0,
        predicted_lift_water=lift_water,
        suction_pressure=1000.0,
        marginal_oil_rate=0.0,
        sonic_status=sonic,
        mach_te=0.5,
    )


class _FakeOpt:
    """Duck-typed NetworkOptimizer: .batch_results + .get_pump_performance."""

    def __init__(self, batch_results):
        self.batch_results = batch_results

    def get_pump_performance(self, well_name, nozzle, throat):
        bp = self.batch_results.get(well_name)
        if bp is None:
            return None
        df = bp.df
        mask = (df["nozzle"] == nozzle) & (df["throat"] == throat)
        rows = df[mask]
        if rows.empty:
            return None
        row = rows.iloc[0]
        if pd.isna(row["qoil_std"]):
            return None
        return {
            "oil_rate": float(row["qoil_std"]),
            "formation_water": float(row["form_wat"]),
            "lift_water": float(row["lift_wat"]),
            "total_water": float(row["totl_wat"]),
            "suction_pressure": float(row["psu_solv"]),
            "sonic_status": bool(row["sonic_status"]),
            "mach_te": float(row["mach_te"]),
            "marginal_oil_lift_water": 0.0,
            "marginal_oil_total_water": 0.0,
        }


class TestApplyParsimony:
    def test_swap_within_threshold_picks_least_water(self):
        # chosen 15B: 302 BOPD @ 3200 BPD. 13C gives up only 2 BOPD for 1500
        # less water -- within threshold(5) -- and is the least-water
        # qualifier. 12B gives up 22 BOPD -- excluded (beyond threshold).
        df = _batch_df(
            [
                ("15", "B", 302.0, 3200.0, False),
                ("13", "C", 300.0, 1700.0, False),
                ("12", "B", 280.0, 1500.0, False),
            ]
        )
        opt = _FakeOpt({"W1": _bp(df)})
        results = [_result("W1", "15", "B", 302.0, 3200.0)]
        new_results, swaps = apply_parsimony(
            results, opt, water_key="lift_wat", threshold_bopd=5.0
        )
        assert len(swaps) == 1
        (r,) = new_results
        assert (r.recommended_nozzle, r.recommended_throat) == ("13", "C")
        assert r.predicted_oil_rate == pytest.approx(300.0)
        assert r.predicted_lift_water == pytest.approx(1700.0)
        assert swaps[0] == {
            "well": "W1",
            "from_pump": "15B",
            "to_pump": "13C",
            "oil_given_up": pytest.approx(2.0),
            "pf_saved": pytest.approx(1500.0),
        }

    def test_no_swap_when_chosen_is_already_least_water(self):
        df = _batch_df(
            [
                ("10", "A", 100.0, 1000.0, False),
                ("11", "A", 150.0, 2000.0, False),  # more water -- irrelevant
            ]
        )
        opt = _FakeOpt({"W1": _bp(df)})
        results = [_result("W1", "10", "A", 100.0, 1000.0)]
        new_results, swaps = apply_parsimony(
            results, opt, water_key="lift_wat", threshold_bopd=5.0
        )
        assert swaps == []
        assert new_results[0] is results[0]

    def test_no_swap_beyond_threshold(self):
        df = _batch_df(
            [
                ("15", "B", 302.0, 3200.0, False),
                ("13", "C", 280.0, 1700.0, False),  # gives up 22 > threshold
            ]
        )
        opt = _FakeOpt({"W1": _bp(df)})
        results = [_result("W1", "15", "B", 302.0, 3200.0)]
        new_results, swaps = apply_parsimony(
            results, opt, water_key="lift_wat", threshold_bopd=5.0
        )
        assert swaps == []
        assert new_results[0] is results[0]

    def test_sonic_guard_skips_sonic_candidate(self):
        # chosen is non-sonic. The least-water candidate within threshold is
        # SONIC -- must be skipped in favor of the next-best non-sonic one.
        df = _batch_df(
            [
                ("15", "B", 300.0, 3000.0, False),  # chosen
                ("14", "B", 298.0, 2000.0, True),  # least water but sonic
                ("13", "C", 296.0, 2200.0, False),  # non-sonic -- picked
            ]
        )
        opt = _FakeOpt({"W1": _bp(df)})
        results = [_result("W1", "15", "B", 300.0, 3000.0, sonic=False)]
        new_results, swaps = apply_parsimony(
            results, opt, water_key="lift_wat", threshold_bopd=5.0
        )
        (r,) = new_results
        assert (r.recommended_nozzle, r.recommended_throat) == ("13", "C")
        assert swaps[0]["to_pump"] == "13C"

    def test_threshold_zero_is_noop(self):
        df = _batch_df(
            [
                ("15", "B", 302.0, 3200.0, False),
                ("13", "C", 300.0, 1700.0, False),
            ]
        )
        opt = _FakeOpt({"W1": _bp(df)})
        results = [_result("W1", "15", "B", 302.0, 3200.0)]
        new_results, swaps = apply_parsimony(
            results, opt, water_key="lift_wat", threshold_bopd=0.0
        )
        assert swaps == []
        assert new_results == results
        assert new_results is not results  # new list, per the contract

    def test_well_with_no_batch_df_passes_through_unchanged(self):
        opt = _FakeOpt({})
        results = [_result("W1", "15", "B", 302.0, 3200.0)]
        new_results, swaps = apply_parsimony(
            results, opt, water_key="lift_wat", threshold_bopd=5.0
        )
        assert swaps == []
        assert new_results[0] is results[0]
