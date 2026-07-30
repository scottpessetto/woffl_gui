"""Water vs pressure — the CFP dashboard's verdict engine.

Scott's question: "is the right way to run higher water pressure or more water?"
These pin the arithmetic that answers it, and the two places it can quietly go
wrong: the break-even algebra, and the 2,900 psi kink (past which cutting water
buys no pressure at all, so it becomes pure oil loss).
"""

import pytest

from woffl.gui.cfp_tradeoff import (
    OIL_SENS_FRAC_PER_PSI,
    PSI_PER_KBWPD_HIGH,
    PSI_PER_KBWPD_LOW,
    PSI_PER_KBWPD_MID,
    TradeoffInputs,
    breakeven_wc,
    discharge_at_water,
    marginal_cost_bopd_per_kbwpd,
    oil_from_water,
    sensitivity_table,
    tradeoff_curve,
    verdict,
    water_at_trip,
)

# The live CFP picture, July 2026.
def _inp(**kw):
    base = dict(
        exposed_oil_bopd=7793.0,      # B + G + J (C-Pad is on its own booster)
        current_water_bwpd=111191.0,
        current_discharge_psi=2816.0,
        psi_per_kbwpd=PSI_PER_KBWPD_MID,
        responsive_frac=0.5,
    )
    base.update(kw)
    return TradeoffInputs(**base)


class TestOilFromWater:
    @pytest.mark.parametrize(
        "wc,expected", [(0.90, 111.1), (0.95, 52.6), (0.99, 10.1), (0.50, 1000.0)]
    )
    def test_known_water_cuts(self, wc, expected):
        assert oil_from_water(1000.0, wc) == pytest.approx(expected, rel=0.01)

    def test_guards_the_degenerate_ends(self):
        assert oil_from_water(1000.0, 1.0) >= 0.0        # no divide-by-zero
        assert oil_from_water(1000.0, 0.0) > 1e6         # ~all oil


class TestBreakeven:
    def test_matches_the_hand_calculation(self):
        """7,793 x 0.5 x 0.00025 x 12.2 = 11.9 BOPD -> 1000/1011.9 = 98.8%."""
        inp = _inp()
        assert marginal_cost_bopd_per_kbwpd(inp) == pytest.approx(11.88, rel=0.01)
        assert breakeven_wc(inp) == pytest.approx(0.9883, rel=0.001)

    def test_pessimistic_corner_still_favours_water(self):
        """Steepest slope, whole fleet responsive — break-even is still ~95%."""
        inp = _inp(psi_per_kbwpd=PSI_PER_KBWPD_HIGH, responsive_frac=1.0)
        assert marginal_cost_bopd_per_kbwpd(inp) == pytest.approx(50.65, rel=0.01)
        assert breakeven_wc(inp) == pytest.approx(0.952, rel=0.002)

    def test_breakeven_falls_as_the_cost_rises(self):
        low = breakeven_wc(_inp(psi_per_kbwpd=PSI_PER_KBWPD_LOW))
        high = breakeven_wc(_inp(psi_per_kbwpd=PSI_PER_KBWPD_HIGH))
        assert high < low

    def test_no_response_means_water_always_wins(self):
        """If oil doesn't respond to pressure there is no cost, so every barrel
        of water pays."""
        inp = _inp(responsive_frac=0.0)
        assert marginal_cost_bopd_per_kbwpd(inp) == 0.0
        assert breakeven_wc(inp) == pytest.approx(1.0)
        assert verdict(inp, 0.999)["action"] == "more_water"


class TestVerdict:
    def test_normal_water_says_bring_it_on(self):
        v = verdict(_inp(), marginal_wc=0.90)
        assert v["action"] == "more_water"
        assert v["net_bopd_per_kbwpd"] == pytest.approx(111.1 - 11.9, rel=0.02)

    def test_nearly_dead_water_says_cut(self):
        v = verdict(_inp(), marginal_wc=0.995)
        assert v["action"] == "cut_water"
        assert v["net_bopd_per_kbwpd"] < 0

    def test_at_breakeven_says_hold(self):
        inp = _inp()
        assert verdict(inp, breakeven_wc(inp))["action"] == "hold"

    def test_missing_marginal_wc_is_unknown_not_a_guess(self):
        v = verdict(_inp())
        assert v["action"] == "unknown"
        assert "gain_bopd_per_kbwpd" not in v

    def test_reports_the_headroom_and_the_cut_to_reach_it(self):
        v = verdict(_inp(), marginal_wc=0.90)
        assert v["headroom_psi"] == pytest.approx(84.0)
        # 84 psi / 12.2 psi-per-1000 = 6,885 BWPD
        assert v["water_to_cut_for_trip_bwpd"] == pytest.approx(6885.0, rel=0.01)


class TestCurveAndKink:
    def test_current_point_is_the_origin(self):
        rows = tradeoff_curve(_inp(), marginal_wc=0.90, steps=41)
        here = min(rows, key=lambda r: abs(r["delta_water_bwpd"]))
        assert here["delta_water_bwpd"] == pytest.approx(0.0)
        assert here["delta_oil_bopd"] == pytest.approx(0.0)

    def test_discharge_never_passes_the_trip(self):
        rows = tradeoff_curve(_inp(), marginal_wc=0.90, span_bwpd=40000.0)
        assert all(r["discharge_psi"] <= 2900.0 + 1e-9 for r in rows)

    def test_the_kink_is_where_cutting_stops_paying(self):
        """Past the trip the pressure is capped, so further cuts lose oil with
        nothing bought — the reason 'cut water for PF' stops early."""
        inp = _inp()
        rows = tradeoff_curve(inp, marginal_wc=0.90, span_bwpd=20000.0, steps=41)
        tripped = [r for r in rows if r["at_trip"]]
        assert tripped, "the sweep should reach the trip"
        assert all(r["delta_water_bwpd"] < 0 for r in tripped)
        # every tripped point sits at the cap and gains no further pressure
        assert len({round(r["discharge_psi"], 6) for r in tripped}) == 1

    def test_water_at_trip_matches_the_curve(self):
        inp = _inp()
        assert discharge_at_water(inp, water_at_trip(inp)) == pytest.approx(2900.0)

    def test_adding_water_gains_oil_at_normal_wc(self):
        rows = tradeoff_curve(_inp(), marginal_wc=0.90)
        adds = [r for r in rows if r["delta_water_bwpd"] > 0]
        assert all(r["delta_oil_bopd"] > 0 for r in adds)

    def test_adding_water_loses_oil_at_dead_wc(self):
        rows = tradeoff_curve(_inp(), marginal_wc=0.999)
        adds = [r for r in rows if r["delta_water_bwpd"] > 0]
        assert all(r["delta_oil_bopd"] < 0 for r in adds)

    def test_curve_requires_a_marginal_wc(self):
        with pytest.raises(ValueError):
            tradeoff_curve(_inp())


class TestSensitivityTable:
    def test_covers_the_whole_uncertainty_box(self):
        rows = sensitivity_table(_inp(), marginal_wc=0.90)
        assert len(rows) == 9  # 3 slopes x 3 responsive fractions
        assert {r["slope"] for r in rows} == {"low (9)", "mid (12.2)", "high (26)"}

    def test_water_wins_across_the_entire_box_at_90pct_wc(self):
        """The headline robustness claim: at 90% WC every corner says more water."""
        rows = sensitivity_table(_inp(), marginal_wc=0.90)
        assert {r["action"] for r in rows} == {"more_water"}

    def test_breakeven_stays_in_the_95_to_996_band(self):
        """Across the whole uncertainty box break-even runs 95.2% - 99.6% WC.
        Pins the headline number quoted to Scott — the pessimistic corner
        (steep slope, whole fleet responsive) is 95.2%, the optimistic one
        (shallow slope, quarter responsive) is 99.6%."""
        rows = sensitivity_table(_inp(), marginal_wc=0.90)
        bes = [r["breakeven_wc"] for r in rows]
        assert min(bes) == pytest.approx(0.9518, abs=0.001)
        assert max(bes) == pytest.approx(0.9956, abs=0.001)


# ── dashboard helpers (pure parts of the CFP page) ──────────────────────────


class TestDashboardState:
    def test_current_state_uses_the_trailing_mean(self):
        import pandas as pd

        from woffl.gui.cfp_pad_page import current_state

        h = pd.DataFrame(
            {"disch": [2800.0, 2816.0, 2830.0], "prod_w": [110000.0, 111191.0, 112000.0]},
            index=pd.to_datetime(["2026-07-28", "2026-07-29", "2026-07-30"]),
        )
        s = current_state(h, days=3)
        assert s["discharge_psi"] == pytest.approx(2815.33, rel=0.001)
        assert s["days"] == 3
        assert "historian" in s["source"]

    def test_current_state_falls_back_without_history(self):
        """The hosted app can't read the `reporting` catalog — the dashboard must
        degrade to the recorded measurement, not crash."""
        import pandas as pd

        from woffl.assembly import cfp_plant as cfp
        from woffl.gui.cfp_pad_page import current_state

        s = current_state(pd.DataFrame())
        assert s["discharge_psi"] == cfp.MEASURED_DISCHARGE_PSI
        assert s["prod_w"] == cfp.MEASURED_PRODUCED_WATER_BWPD
        assert "fallback" in s["source"]

    def test_exposed_pads_exclude_c(self):
        """C-Pad is boosted on-pad, so its oil is NOT exposed to a plant sag —
        including it would overstate the cost of water."""
        from woffl.gui.cfp_pad_page import _EXPOSED_PADS

        assert set(_EXPOSED_PADS) == {"B", "G", "J"}


class TestBopdPerPsi:
    def test_the_communication_number(self):
        """7,793 × 0.5 × 0.00025 ≈ 0.97 BOPD per psi — the dashboard's
        '1 psi is worth' tile before a moves run refines it per-well."""
        from woffl.gui.cfp_tradeoff import bopd_per_psi

        assert bopd_per_psi(_inp()) == pytest.approx(0.974, abs=0.01)

    def test_consistent_with_the_marginal_cost(self):
        """cost per 1,000 BWPD = (BOPD per psi) × (psi per 1,000 BWPD)."""
        from woffl.gui.cfp_tradeoff import bopd_per_psi

        inp = _inp()
        assert marginal_cost_bopd_per_kbwpd(inp) == pytest.approx(
            bopd_per_psi(inp) * inp.psi_per_kbwpd
        )


class TestDashboardHistTags:
    def test_machine_flow_tags_are_the_confirmed_ones(self):
        """Tripwire: the dashboard briefly charted MPU_FIC_5488/5489 as
        'machine flow' — a different stream that drove a wrong two-machine
        conclusion. It must read the three confirmed per-machine tags."""
        from woffl.assembly.cfp_plant import MACHINE_FLOW_TAGS
        from woffl.gui.cfp_pad_page import _HIST_TAGS

        for tag in MACHINE_FLOW_TAGS.values():
            assert tag in _HIST_TAGS
        assert "MPU_FIC_5488" not in _HIST_TAGS
        assert "MPU_FIC_5489" not in _HIST_TAGS
        assert sorted(
            v for v in _HIST_TAGS.values() if v.startswith("m_")
        ) == ["m_a", "m_b", "m_c"]
