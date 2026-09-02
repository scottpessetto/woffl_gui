"""E-Pad booster candidate capability - workbook pins, constraints, contract.

Three layers:

1. **Workbook pins.** The model must reproduce
   ``temp/Summit E Pad Booster (1).xlsx`` and
   ``temp/Summit E Pad Booster as SN35000.xlsx`` cell for cell at their own
   scenario settings (26 stg / 55 Hz / SG 1.02 / 2,800 psi and 18 stg / 55 Hz
   / SG 1.00 / 2,600 psi). Those two sheets are the engineer's own arithmetic
   and the reason this screen is trusted; if a pin here moves, the screen and
   the workbook disagree and one of them is wrong.
2. **The constraints the screen exists to enforce** - the recommended
   operating range moving with speed, and the optional motor amp cap.
3. **The API contract** - JSON-safe shapes, installed build first, and the
   request bounds.

Pure static physics: nothing is monkeypatched, and nothing here touches
Databricks. If this path ever grows a query these tests fail on the .env
write-gate leak (AGENTS.md section 3).
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from server import schemas
from server.main import app
from woffl.gui import e_pad_booster as epb

# The efficiency constant the two workbooks round: BPD-ft-to-HP is
# 3960 * 1440 / 42 = 135,771.43, and sheet 1 types 136,000. The model uses the
# exact value (PadPlant.hydraulic_efficiency), so every efficiency reads
# 0.168 pct high against the sheet - always in that one direction.
_EXACT_BPD_FT_HP = 3960.0 * 1440.0 / 42.0
_SHEET1_CONST = 136000.0
# Sheet 2 writes the same thing in dP form: BPD-psi-to-HP is 135,771.43 / 2.31
# = 58,775.73, and it types 58,766.
_SHEET2_CONST = 58766.0

# Sheet 1 "Current": SG 1.02, 26 stages, 55 Hz, Condition 1, suction 2,800.
# Columns A (BPD at 60 Hz), G (Delta PSI), I (HP load), K (Efficiency).
_SHEET1 = [
    (0, 1485.6111111111113, 324.7917291666667, 0.0),
    (5000, 1504.904761904762, 384.0304722222223, 0.30506883604505625),
    (10000, 1466.3174603174605, 441.22650000000004, 0.5174291938997821),
    (15000, 1389.1428571428573, 500.46524305555545, 0.6482593037214888),
    (20000, 1331.2619047619048, 563.7894166666666, 0.7352941176470589),
    (25000, 1234.7936507936508, 620.9854444444444, 0.7739938080495357),
    (30000, 1109.3849206349205, 670.0106111111111, 0.7734038737446198),
    (35000, 906.8015873015871, 702.6940555555557, 0.7032318741450069),
    (40000, 636.6904761904763, 721.0784930555554, 0.5499083486085654),
    (41000, 578.8095238095237, 721.0784930555554, 0.5124145975670722),
]
_SHEET1_HZ, _SHEET1_SG, _SHEET1_SUCTION = 55.0, 1.02, 2800.0

# Sheet 2 "Current": SG 1.00, 18 stages, 55 Hz, Condition 1, suction 2,600.
_SHEET2 = [
    (0, 1538.6904761904761, 582.3125, 0.0),
    (2500, 1538.6904761904761, 582.3125, 0.10304346874083505),
    (5000, 1532.1428571428573, 596.1770833333334, 0.20043764686490387),
    (10000, 1473.2142857142856, 610.0416666666666, 0.37669662653806235),
    (15000, 1407.738095238095, 651.6354166666666, 0.5054680974539106),
    (20000, 1335.7142857142856, 693.2291666666666, 0.601107363521004),
    (25000, 1263.690476190476, 734.8229166666666, 0.6706305560925),
    (30000, 1204.7619047619048, 790.28125, 0.7133885119186323),
    (35000, 1139.2857142857144, 831.875, 0.7477009484973274),
    (40000, 1073.8095238095236, 873.46875, 0.7670526580878047),
    (45000, 988.6904761904764, 915.0625, 0.7584158747632991),
    (50000, 857.7380952380952, 928.9270833333334, 0.7201589901709422),
    (55000, 707.1428571428571, 935.859375, 0.6482530390912787),
    (60000, 504.1666666666667, 931.7000000000002, 0.5064476867900616),
    (62000, 412.5, 930.3135416666668, 0.4288166188000499),
]
_SHEET2_HZ, _SHEET2_SG, _SHEET2_SUCTION = 55.0, 1.0, 2600.0

_SM = "SM25000_26STG"
_SN = "SN35000_18STG"


def build(key: str) -> epb.EPadBooster:
    hits = [b for b in epb.candidates() if b.key == key]
    assert len(hits) == 1, f"no unique candidate {key}"
    return hits[0]


def report(**over) -> dict:
    """capability_report at the 3,400 psig header duty unless overridden."""
    kwargs = {
        "dp_psid": 600.0,
        "suction_psi": 2800.0,
        "sg": 1.02,
        "condition": 1.0,
        "hz_max": 60.0,
        "amps_per_bhp": 0.1435,
        "amp_limit": None,
    }
    kwargs.update(over)
    return epb.capability_report(**kwargs)


def candidate(rep: dict, key: str) -> dict:
    hits = [c for c in rep["candidates"] if c["nameplate"]["key"] == key]
    assert len(hits) == 1
    return hits[0]


def leaves(node, path="$"):
    """Yield (json path, value) for every scalar leaf of a nested payload."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from leaves(v, f"{path}.{k}")
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, node


# ---------------------------------------------------------------------------
# 1. Workbook pins - the model IS the engineer's sheet, run backwards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key, rows, hz, sg",
    [(_SM, _SHEET1, _SHEET1_HZ, _SHEET1_SG), (_SN, _SHEET2, _SHEET2_HZ, _SHEET2_SG)],
)
def test_dp_and_bhp_reproduce_the_workbook(key, rows, hz, sg):
    b = build(key)
    for q60, dp, bhp, _eff in rows:
        # The sheets tabulate 60-Hz-equivalent flow in column A and the ACTUAL
        # flow in "Adj BPD"; the model is indexed on actual flow.
        q = q60 * hz / 60.0
        assert b.dp_psi(q, hz, sg) == pytest.approx(dp, rel=1e-12)
        assert b.bhp(q, hz, sg) == pytest.approx(bhp, rel=1e-12)


def test_discharge_reproduces_the_workbook_disc_psi_column():
    b = build(_SM)
    for q60, dp, _bhp, _eff in _SHEET1:
        q = q60 * _SHEET1_HZ / 60.0
        assert _SHEET1_SUCTION + b.dp_psi(q, _SHEET1_HZ, _SHEET1_SG) == pytest.approx(
            _SHEET1_SUCTION + dp, rel=1e-12
        )


def test_efficiency_matches_sheet1_up_to_its_rounded_constant():
    b = build(_SM)
    for q60, _dp, bhp, eff in _SHEET1:
        if eff == 0.0:
            continue
        q = q60 * _SHEET1_HZ / 60.0
        head = b.head_ft(q, _SHEET1_HZ, 1.0)
        got = epb.PadPlant.hydraulic_efficiency(q, head, bhp, _SHEET1_SG)
        expected = 100.0 * eff * _SHEET1_CONST / _EXACT_BPD_FT_HP
        assert got == pytest.approx(expected, rel=1e-9)


def test_efficiency_matches_sheet2_up_to_its_rounded_constant():
    b = build(_SN)
    for q60, dp, bhp, eff in _SHEET2:
        if eff == 0.0:
            continue
        q = q60 * _SHEET2_HZ / 60.0
        head = b.head_ft(q, _SHEET2_HZ, 1.0)
        got = epb.PadPlant.hydraulic_efficiency(q, head, bhp, _SHEET2_SG)
        expected = 100.0 * eff * _SHEET2_CONST / (_EXACT_BPD_FT_HP / 2.31)
        assert got == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("key", [_SM, _SN])
def test_stage_curve_returns_the_vendor_rows_verbatim(key):
    # Linear interpolation, so every digitized row must come back untouched -
    # this is the guard against someone swapping in a polynomial fit that
    # smooths the catalog page.
    b = build(key)
    for q60, head, bhp in b.table:
        assert b.head_per_stage(q60) == pytest.approx(head, rel=1e-12)
        assert b.bhp_per_stage(q60) == pytest.approx(bhp, rel=1e-12)


def test_sm25000_head_rises_off_shut_off_then_falls():
    # Mixed-flow stage: the catalog page really does climb 154 -> 156 ft over
    # the first 5,000 BPD. A monotone-falling assertion would be false here.
    b = build(_SM)
    assert b.head_per_stage(5000.0) > b.head_per_stage(0.0)
    heads = [b.head_per_stage(q) for q in range(5000, 41001, 1000)]
    assert all(a >= c for a, c in zip(heads, heads[1:]))


# ---------------------------------------------------------------------------
# 2. The inverse solve, the recommended range, and the amp cap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", [_SM, _SN])
@pytest.mark.parametrize("dp", [400.0, 600.0, 900.0])
def test_hz_for_dp_round_trips(key, dp):
    b = build(key)
    for q in (2000.0, 8000.0, 16000.0, 24000.0):
        hz = b.hz_for_dp(q, dp, 1.02, 1.0, 60.0)
        if hz is None:
            continue
        assert 20.0 <= hz <= 60.0
        assert b.dp_psi(q, hz, 1.02, 1.0) == pytest.approx(dp, rel=1e-7)


@pytest.mark.parametrize("key", [_SM, _SN])
def test_hz_for_dp_is_none_past_the_capability_wall(key):
    b = build(key)
    wall = b.max_flow_for_dp(600.0, 1.02, 1.0, 60.0)
    assert b.hz_for_dp(wall * 0.99, 600.0, 1.02, 1.0, 60.0) is not None
    assert b.hz_for_dp(wall * 1.05, 600.0, 1.02, 1.0, 60.0) is None


def test_the_sm25000_wall_at_the_header_duty_is_over_delivery_not_head():
    # Subtle and worth pinning: at 600 psid the SM25000 runs out of TABLE
    # before it runs out of HEAD. Past ~38.3 k BPD the slowest speed that
    # keeps the flow on the digitized curve (60 Hz-equivalent 41,000 BPD)
    # already makes more than 600 psid, so the pump cannot hold exactly that
    # dP there - it would over-deliver. Taking the naive head wall instead
    # would put the locus out to 41,000 BPD on points that have no solution.
    b = build(_SM)
    wall = b.max_flow_for_dp(600.0, 1.02, 1.0, 60.0)
    assert wall == pytest.approx(38266.7, abs=5.0)
    assert wall < b.max_valid_flow
    assert b.dp_psi(b.max_valid_flow, 60.0, 1.02, 1.0) > 600.0  # head to spare
    assert b.wall_kind(wall, 600.0, 1.02, 1.0, 60.0) == epb.LIMIT_OVER_DELIVERS
    hz_at_wall = b.hz_for_dp(wall, 600.0, 1.02, 1.0, 60.0)
    assert hz_at_wall == pytest.approx(wall * 60.0 / b.max_valid_flow, rel=1e-6)


@pytest.mark.parametrize("key", [_SM, _SN])
def test_recommended_range_scales_linearly_with_speed(key):
    b = build(key)
    lo60, hi60 = b.ror_60hz
    lo, hi = b.ror(45.0)
    assert (lo, hi) == pytest.approx((lo60 * 0.75, hi60 * 0.75), rel=1e-12)
    assert b.ror(60.0) == pytest.approx((lo60, hi60), rel=1e-12)


@pytest.mark.parametrize("key", [_SM, _SN])
def test_vendor_range_top_is_exactly_120_percent_of_bep(key):
    # 32,400 / 27,000 and 49,500 / 41,250 both equal 1.20 - the arithmetic
    # check that the two ranges were digitized off the right catalog rows.
    b = build(key)
    assert b.ror_60hz[1] / b.bep == pytest.approx(1.20, rel=1e-9)


@pytest.mark.parametrize("key", [_SM, _SN])
def test_window_edges_are_the_actual_feasibility_boundary(key):
    c = candidate(report(), key)
    q_lo, q_hi = c["window"]
    b = build(key)

    def ok(q: float) -> bool:
        return b.point_at_dp(q, 600.0, 1.02, 1.0, 60.0, 0.1435, None, 2800.0)["ok"]

    assert ok(q_lo) and ok(q_hi)
    assert ok(0.5 * (q_lo + q_hi))
    assert not ok(q_hi * 1.002)
    assert not ok(q_lo * 0.998)


def test_the_header_duty_is_range_limited_on_both_candidates():
    # At 600 psid neither build is anywhere near pressure- or amp-limited;
    # they run in the low 40s Hz and the vendor range is what caps the rate.
    # This is the screen's headline answer, so it is pinned.
    rep = report()
    sm, sn = candidate(rep, _SM), candidate(rep, _SN)
    for c in (sm, sn):
        assert c["limited_by"] == epb.LIMIT_ROR_HIGH
        assert c["duty"]["hz"] < 50.0
        assert c["duty"]["q_bpd"] == pytest.approx(c["duty"]["ror_hi"], rel=1e-6)
    assert sm["duty"]["q_bpd"] == pytest.approx(22867.0, abs=1.0)
    assert sn["duty"]["q_bpd"] == pytest.approx(37293.0, abs=1.0)
    # The alternative is the whole point of the screen: more water, same header.
    assert sn["duty"]["q_bpd"] > sm["duty"]["q_bpd"]


def test_below_the_window_the_block_reason_is_the_range_floor():
    c = candidate(report(), _SM)
    q_lo = c["window"][0]
    low = [r for r in c["locus"] if r["q_bpd"] < q_lo * 0.9 and r["hz"] is not None]
    assert low, "expected locus points below the window"
    assert all(r["blocked_by"] == epb.BLOCK_ROR_LOW for r in low)
    assert all(r["q_bpd"] < r["ror_lo"] for r in low)


def test_amp_limit_binds_and_moves_the_duty_onto_the_limit():
    free = candidate(report(), _SN)
    capped = candidate(report(amp_limit=60.0), _SN)
    assert capped["limited_by"] == epb.LIMIT_AMPS
    assert capped["duty"]["q_bpd"] < free["duty"]["q_bpd"]
    assert capped["duty"]["amps"] == pytest.approx(60.0, abs=1e-6)
    assert capped["duty"]["amp_headroom_a"] == pytest.approx(0.0, abs=1e-6)
    # And the range is still respected inside the smaller window.
    assert capped["duty"]["in_ror"] is True


def test_no_amp_limit_reports_amps_but_enforces_nothing():
    c = candidate(report(), _SM)
    assert c["nameplate"]["amp_limit_a"] is None
    assert c["duty"]["amp_headroom_a"] is None
    assert all(r["amp_ok"] for r in c["locus"] if r["hz"] is not None)
    live = [r for r in c["locus"] if r["hz"] is not None and r["q_bpd"] > 0]
    assert all(r["amps"] > 0 for r in live)


# ---------------------------------------------------------------------------
# The fixed-speed ladder, and the two operating policies
# ---------------------------------------------------------------------------


def speed_row(c: dict, hz: float) -> dict:
    hits = [r for r in c["speed_table"] if abs(r["hz"] - hz) < 0.01]
    assert len(hits) == 1, f"no unique {hz} Hz row"
    return hits[0]


def test_pinning_the_drive_at_55_hz_answers_the_operator_question():
    # The question the deliverable-rate view does NOT answer: "I am going to
    # run it at 55 Hz against 600 psid, what comes out?" 37,248 BPD comes out
    # - and it is 125 pct of the recommended range ceiling at that speed, at
    # 53 pct efficiency for 721 BHP, against 19,336 BPD / 78 pct / 254 BHP at
    # 40 Hz. That whole comparison is why this table exists.
    c = candidate(report(hz_max=55.0), _SM)
    r55 = speed_row(c, 55.0)
    assert r55["q_bpd"] == pytest.approx(37248.0, abs=5.0)
    assert r55["in_ror"] is False
    assert r55["blocked_by"] == epb.BLOCK_ROR_HIGH
    assert r55["pct_of_ror_hi"] == pytest.approx(125.4, abs=0.2)
    assert r55["bhp"] == pytest.approx(721.0, abs=1.0)
    assert r55["eff_pct"] == pytest.approx(52.7, abs=0.2)

    r40 = speed_row(c, 40.0)
    assert r40["in_ror"] is True and r40["blocked_by"] is None
    assert r40["q_bpd"] == pytest.approx(19336.0, abs=5.0)
    assert r40["eff_pct"] > r55["eff_pct"] + 20.0
    assert r40["bhp"] < 0.4 * r55["bhp"]


def test_the_ladder_explains_where_the_deliverable_rate_caps():
    # The duty row is the LAST in-range rung: it sits exactly on the range
    # ceiling at its own speed, and every faster rung is over it. That is the
    # mechanism behind "capped by recommended range (high)".
    c = candidate(report(), _SM)
    duty_rows = [r for r in c["speed_table"] if r["is_duty"]]
    assert len(duty_rows) == 1
    duty = duty_rows[0]
    assert duty["hz"] == pytest.approx(c["duty"]["hz"], rel=1e-12)
    assert duty["q_bpd"] == pytest.approx(c["duty"]["q_bpd"], rel=1e-6)
    assert duty["in_ror"] is True
    assert duty["pct_of_ror_hi"] == pytest.approx(100.0, abs=0.01)
    faster = [r for r in c["speed_table"] if r["hz"] > duty["hz"] + 0.01]
    assert faster, "expected rungs above the duty speed"
    assert all(r["blocked_by"] == epb.BLOCK_ROR_HIGH for r in faster)


def test_ladder_flow_rises_with_speed_and_respects_the_cap():
    c = candidate(report(hz_max=50.0), _SM)
    speeds = [r["hz"] for r in c["speed_table"]]
    assert speeds == sorted(speeds)
    assert max(speeds) == pytest.approx(50.0)
    flows = [r["q_bpd"] for r in c["speed_table"] if r["q_bpd"] is not None]
    assert all(a < b for a, b in zip(flows, flows[1:])), flows
    pump = build(_SM)
    for r in c["speed_table"]:
        if r["q_bpd"] is None:
            continue
        # Each rung really is the crossing of that speed's curve with the dP.
        got = pump.dp_psi(r["q_bpd"], r["hz"], 1.02, 1.0)
        assert got == pytest.approx(600.0, rel=1e-6)


def test_a_speed_that_cannot_make_the_dp_is_reported_not_dropped():
    # 26 stg at 35 Hz tops out around 600 psid, so a bigger dP leaves the
    # bottom rungs unreachable. They must still be listed with the reason.
    c = candidate(report(dp_psid=1100.0), _SM)
    low = [r for r in c["speed_table"] if r["hz"] <= 40.0]
    assert low, "expected low rungs"
    assert all(r["q_bpd"] is None for r in low)
    assert all(r["blocked_by"] == epb.BLOCK_DP_UNREACHABLE for r in low)
    assert all(r["in_ror"] is False for r in low)


def test_throttled_policy_moves_more_water_for_more_power_and_a_choke_loss():
    # The trade behind "do I run the pump slower?": hold the dP exactly by
    # slowing down (in range, no loss, least water) or run flat out at the
    # range ceiling and choke the surplus off.
    c = candidate(report(), _SM)
    duty, t = c["duty"], c["throttled"]
    assert t is not None
    assert t["hz"] == pytest.approx(60.0)
    assert t["q_bpd"] == pytest.approx(32400.0, rel=1e-6)  # ror_hi at 60 Hz
    assert t["in_ror"] is True
    assert t["q_bpd"] > duty["q_bpd"]
    assert t["bhp"] > duty["bhp"]
    assert t["dp_made_psid"] > 600.0
    assert t["throttle_psid"] == pytest.approx(t["dp_made_psid"] - 600.0, rel=1e-12)
    assert t["discharge_psi"] == pytest.approx(2800.0 + t["dp_made_psid"], rel=1e-12)
    # Choke loss priced as hydraulic HP: BPD x psi / 58,776.
    assert t["throttle_hhp"] == pytest.approx(
        t["q_bpd"] * t["throttle_psid"] / (3960.0 * (1440.0 / 42.0) / 2.31), rel=1e-12
    )


def test_throttled_policy_follows_the_speed_cap():
    at55 = candidate(report(hz_max=55.0), _SM)["throttled"]
    assert at55["hz"] == pytest.approx(55.0)
    assert at55["q_bpd"] == pytest.approx(32400.0 * 55.0 / 60.0, rel=1e-6)
    assert at55["q_bpd"] < candidate(report(), _SM)["throttled"]["q_bpd"]


def test_an_amp_cap_pulls_the_throttled_flow_back_onto_the_limit():
    t = candidate(report(amp_limit=90.0), _SM)["throttled"]
    assert t is not None
    assert t["amps"] == pytest.approx(90.0, abs=1e-6)
    assert t["amp_headroom_a"] == pytest.approx(0.0, abs=1e-6)
    assert t["q_bpd"] < 32400.0


def test_no_throttled_policy_when_the_speed_cap_cannot_reach_the_dp():
    # At 35 Hz the range-ceiling flow makes far less than 1,100 psid, so there
    # is no surplus to choke and the policy is simply unavailable.
    assert candidate(report(dp_psid=1100.0, hz_max=35.0), _SM)["throttled"] is None


def test_amps_are_the_calibration_constant_times_shaft_bhp():
    for k in (0.1435, 0.25):
        c = candidate(report(amps_per_bhp=k), _SM)
        d = c["duty"]
        assert d["amps"] == pytest.approx(k * d["bhp"], rel=1e-12)


def test_an_impossible_amp_limit_reports_why_instead_of_a_duty():
    c = candidate(report(amp_limit=5.0), _SN)
    assert c["duty"] is None and c["min_duty"] is None and c["window"] is None
    assert c["limited_by"] in (epb.LIMIT_AMPS, epb.LIMIT_ROR_HIGH)
    assert "5 A" in c["infeasible_reason"]
    # The locus still comes back so the chart can show the wall it hit.
    assert len(c["locus"]) == epb.CURVE_POINTS


def test_a_dp_above_shut_off_head_reports_the_shut_off_ceiling():
    c = candidate(report(dp_psid=2500.0, suction_psi=900.0), _SM)
    assert c["duty"] is None and c["locus"] == []
    assert c["q_ceiling"] == 0.0
    assert c["limited_by"] == epb.LIMIT_CAPABILITY
    assert "shut-off" in c["infeasible_reason"]


def test_speed_cap_binds_the_duty():
    capped = candidate(report(dp_psid=1200.0, suction_psi=2200.0, hz_max=50.0), _SM)
    assert capped["duty"]["hz"] == pytest.approx(50.0, abs=1e-6)
    assert capped["limited_by"] == epb.LIMIT_CAPABILITY
    assert all(r["hz"] <= 50.0 + 1e-9 for r in capped["locus"] if r["hz"] is not None)


def test_wear_derate_costs_head_and_buys_speed():
    asnew = candidate(report(), _SM)["duty"]
    worn = candidate(report(condition=0.85), _SM)["duty"]
    # Same dP demanded, so a derated pump must spin faster and, at the range
    # ceiling that speed unlocks, moves more water for more power.
    assert worn["hz"] > asnew["hz"]
    assert worn["q_bpd"] > asnew["q_bpd"]
    assert worn["bhp"] > asnew["bhp"]
    b = build(_SM)
    assert b.head_ft(20000.0, 45.0, 0.85) == pytest.approx(
        0.85 * b.head_ft(20000.0, 45.0, 1.0), rel=1e-12
    )
    # Wear does not lighten the shaft.
    assert b.bhp(20000.0, 45.0, 1.02) == pytest.approx(b.bhp(20000.0, 45.0, 1.02))


# ---------------------------------------------------------------------------
# 3. Payload contract
# ---------------------------------------------------------------------------


def test_installed_build_comes_first():
    rep = report()
    assert [c["nameplate"]["key"] for c in rep["candidates"]] == [_SM, _SN]
    assert rep["candidates"][0]["nameplate"]["installed"] is True
    assert rep["candidates"][1]["nameplate"]["installed"] is False


def test_report_is_json_safe():
    rep = report()
    for path, value in leaves(rep):
        assert isinstance(value, (bool, int, float, str, type(None))), path
    assert json.loads(json.dumps(rep)) == rep


@pytest.mark.parametrize("key", [_SM, _SN])
def test_curve_and_locus_grids_match_the_contract(key):
    c = candidate(report(), key)
    assert len(c["locus"]) == epb.CURVE_POINTS
    assert c["locus"][0]["q_bpd"] == 0.0
    assert c["locus"][-1]["q_bpd"] == pytest.approx(c["q_ceiling"], rel=1e-12)
    assert [cv["hz"] for cv in c["curves"]] == [45.0, 50.0, 55.0, 60.0]
    for cv in c["curves"]:
        assert len(cv["points"]) == epb.CURVE_POINTS
        assert all(len(p) == 5 for p in cv["points"])
        # Each line stops where the digitized table does at that speed.
        assert cv["points"][-1][0] == pytest.approx(
            c["max_valid_flow_60hz"] * cv["hz"] / 60.0, rel=1e-12
        )
    assert len(c["machine"]["points"]) == epb.CURVE_POINTS
    assert all(len(p) == 4 for p in c["machine"]["points"])


@pytest.mark.parametrize("key", [_SM, _SN])
def test_machine_sheet_is_the_vendor_page_not_the_scenario(key):
    # As-new, on water, at 60 Hz: whole-pump head = stages x the catalog stage
    # curve, untouched by the screen's SG or wear inputs.
    b = build(key)
    c = candidate(report(sg=1.15, condition=1.0), key)
    m = c["machine"]
    assert m["hz"] == 60.0
    assert m["head_derated"] is None and m["derate_note"] is None
    assert m["points"][0][1] == pytest.approx(b.n_stages * b.head_per_stage(0.0))
    assert m["aor"] == list(b.ror_60hz)
    assert m["bep"] == b.bep
    assert m["por"] == pytest.approx([0.70 * b.bep, 1.20 * b.bep], rel=1e-12)
    assert m["min_flow"] is None
    for _q, _head, _bhp, eff in m["points"]:
        assert 0.0 <= eff <= 100.0
    assert m["points"][0][3] == 0.0  # efficiency is undefined at shut-off


def test_machine_sheet_carries_the_wear_derate_when_one_is_modeled():
    m = candidate(report(condition=0.9), _SM)["machine"]
    assert m["derate_note"] is not None and "0.90" in m["derate_note"]
    assert len(m["head_derated"]) == epb.CURVE_POINTS
    for asnew, worn in zip(m["points"], m["head_derated"]):
        assert worn[1] == pytest.approx(0.9 * asnew[1], rel=1e-12)


def test_notes_carry_the_caveats_the_engineer_must_see():
    n = report()["notes"]
    assert "TRANSFERRED ESTIMATE" in n["amps"]
    assert "2,800 psi" in n["housing_pressure"] and "3,408" in n["housing_pressure"]
    assert any("housing pressure" in x for x in n["not_enforced"])


def test_request_defaults_do_not_drift_from_the_data_file():
    # Two sources describe the same seed values: the meta json (which carries
    # the provenance notes) and the request schema (which carries the OpenAPI
    # contract). They must agree.
    d = epb.defaults()
    req = schemas.EPadBoosterRequest()
    assert req.suction_psi == d["suction_psi"]
    assert req.dp_psid == d["target_discharge_psi"] - d["suction_psi"]
    assert req.sg == d["sg"]
    assert req.condition == d["condition"]
    assert req.hz_max == d["hz_max"]
    assert req.amps_per_bhp == d["amps_per_bhp"]
    # No E-Pad motor nameplate exists, so nothing may seed an amp limit.
    assert req.amp_limit_a is None
    for spec in epb.meta()["pumps"].values():
        assert spec["amp_limit_A"] is None


# ---------------------------------------------------------------------------
# POST /api/optimize/e-pad-booster
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def test_endpoint_validates_against_the_schema(client):
    r = client.post("/api/optimize/e-pad-booster", json={"dp_psid": 600})
    assert r.status_code == 200
    body = schemas.EPadBoosterResponse.model_validate(r.json())
    assert body.pad == "E"
    assert body.target.discharge_psi == 3400.0
    assert body.target.header_default_psi == 3400.0
    assert len(body.candidates) == 2
    assert body.candidates[0].nameplate.installed is True


def test_endpoint_accepts_an_empty_body_and_lands_on_the_header_duty(client):
    r = client.post("/api/optimize/e-pad-booster", json={})
    assert r.status_code == 200
    assert r.json()["target"]["discharge_psi"] == 3400.0


def test_endpoint_passes_the_amp_limit_through(client):
    r = client.post(
        "/api/optimize/e-pad-booster", json={"dp_psid": 800, "amp_limit_a": 60}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["target"]["amp_limit_a"] == 60.0
    for c in body["candidates"]:
        assert c["nameplate"]["amp_limit_a"] == 60.0
        if c["duty"] is not None:
            assert c["duty"]["amps"] <= 60.0 + 1e-6


@pytest.mark.parametrize(
    "payload",
    [
        {"hz_max": 61},  # the stage tables are 60 Hz catalog curves
        {"hz_max": 29},
        {"condition": 1.01},  # condition is a DERATE
        {"condition": 0.0},
        {"dp_psid": 0},
        {"sg": 0.5},
        {"amps_per_bhp": 0},
        {"amp_limit_a": 0},
    ],
)
def test_endpoint_rejects_out_of_range_inputs(client, payload):
    r = client.post("/api/optimize/e-pad-booster", json=payload)
    assert r.status_code == 422
