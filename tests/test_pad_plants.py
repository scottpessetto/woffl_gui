"""Pad booster-plant models — pinned-value regression tests.

Covers the three per-pad plant modules (all pure stdlib, no Streamlit):

  * ``woffl.gui.s_pad_plant`` — 3x parallel fixed-speed Borets ESPi675TJ
  * ``woffl.gui.i_pad_plant`` — 2-pump SERIES VFD train, amp-limited frontier
  * ``woffl.gui.m_pad_plant`` — 3x parallel VFD REDA M675 HP bank, wear-derated

The exact pins (rel=1e-6) are today's computed numbers: a future refactor
(e.g. the planned PadPlant base-class unification) must reproduce them
bit-for-bit. Datasheet / live-SCADA anchors use loose abs tolerances and tie
the model back to its provenance:

  * S-Pad meta "validated": 30,694 BPD total -> ~3,460 psi discharge
  * I-Pad meta "validated" (live SCADA 2026-06-16): LP dP model 1874 vs 1887
    psid live; HP dP model 1336 vs 1326 psid live; LP amps 190.9 at the 192 A
    drive limit
  * M-Pad datasheet (Schlumberger set-up sheet, 60 Hz SG 1.05): HP shut-in
    4,470.7 psig at 1,500 suction (boost 2,970.7), design point 32,000 BPD ->
    boost 2,001.5 psi; live SCADA P-4230A 16.5 kBPD / ~2,103 psid / 125 A;
    FIELD_HEAD_FACTOR 0.91 wear derate from the meta JSON
"""

import pytest

import woffl.gui.i_pad_plant as i_pad
import woffl.gui.m_pad_plant as m_pad
import woffl.gui.s_pad_plant as s_pad

# ---------------------------------------------------------------------------
# S-Pad — 3 parallel fixed-speed pumps: discharge = suction + poly(dP(Q/n))
# ---------------------------------------------------------------------------


def test_s_pad_meta_parameters():
    assert s_pad.n_pumps_installed() == 3
    assert s_pad.recommended_flow_per_pump() == (7650.0, 18360.0)
    assert s_pad.station_capacity(3) == pytest.approx(55080.0)
    assert s_pad.station_capacity(2) == pytest.approx(36720.0)
    # shut-in head per stage from the 2012 Borets section-test fit
    assert s_pad.head_per_stage(0.0) == pytest.approx(107.22128868404283, rel=1e-9)


def test_s_pad_discharge_pins_3_pumps():
    # Regression pins — exact current values, must survive any refactor.
    assert s_pad.discharge_pressure(0.0, 3) == pytest.approx(
        3886.8752407096904, rel=1e-6
    )
    assert s_pad.discharge_pressure(20000.0, 3) == pytest.approx(
        3695.578661886785, rel=1e-6
    )
    assert s_pad.discharge_pressure(30694.0, 3) == pytest.approx(
        3454.9341903902487, rel=1e-6
    )
    assert s_pad.discharge_pressure(45000.0, 3) == pytest.approx(
        2903.411514895759, rel=1e-6
    )


def test_s_pad_discharge_pins_2_pumps():
    assert s_pad.discharge_pressure(15000.0, 2) == pytest.approx(
        3650.2144326356947, rel=1e-6
    )
    assert s_pad.discharge_pressure(30000.0, 2) == pytest.approx(
        2903.411514895759, rel=1e-6
    )
    assert s_pad.discharge_pressure(36000.0, 2) == pytest.approx(
        2386.8533700242215, rel=1e-6
    )


def test_s_pad_scada_validation_anchor():
    # Meta JSON "validated": 30,694 BPD total -> ~3,460 psi discharge (SCADA ~1%).
    assert s_pad.discharge_pressure(30694.0, 3) == pytest.approx(3460.0, abs=35.0)


def test_s_pad_parallel_split_invariance():
    # Parallel pumps share dP: same per-pump flow -> identical discharge.
    assert s_pad.discharge_pressure(20000.0, 2) == pytest.approx(
        s_pad.discharge_pressure(30000.0, 3), rel=1e-12
    )


def test_s_pad_pressure_falls_with_flow():
    flows = [10000.0, 20000.0, 30000.0, 40000.0, 50000.0]
    pressures = [s_pad.discharge_pressure(q, 3) for q in flows]
    assert all(a > b for a, b in zip(pressures, pressures[1:]))


def test_s_pad_more_pumps_more_pressure():
    assert s_pad.discharge_pressure(30000.0, 3) > s_pad.discharge_pressure(30000.0, 2)


def test_s_pad_overrides():
    base = s_pad.discharge_pressure(30000.0, 3)
    # suction override shifts the curve 1:1
    assert s_pad.discharge_pressure(30000.0, 3, suction_psi=300.0) == pytest.approx(
        base + 80.0, rel=1e-12
    )
    # sg override scales only the dP term: suction + (base - suction) * sg
    assert s_pad.discharge_pressure(30000.0, 3, sg=1.03) == pytest.approx(
        3572.1312453941946, rel=1e-6
    )


def test_s_pad_flow_in_range_window():
    assert s_pad.flow_in_range(55080.0, 3) is True  # exactly at 3 x 18,360
    assert s_pad.flow_in_range(55081.0, 3) is False  # just past the thrust window
    assert s_pad.flow_in_range(20000.0, 3) is False  # per-pump 6,667 < 7,650 floor


def test_s_pad_past_capacity_extrapolates_current_behavior():
    # CURRENT-BEHAVIOR documentation, not an endorsement: discharge_pressure has
    # no capacity guard — past station_capacity it keeps evaluating the cubic
    # (still positive at 70 kBPD, NEGATIVE by 80 kBPD on 3 pumps). Callers are
    # expected to gate on flow_in_range()/station_capacity().
    assert s_pad.discharge_pressure(70000.0, 3) == pytest.approx(
        1068.0950092502458, rel=1e-6
    )
    assert s_pad.discharge_pressure(80000.0, 3) == pytest.approx(
        -60.9902132516541, rel=1e-6
    )


def test_s_pad_zero_pumps_raises():
    with pytest.raises(ValueError):
        s_pad.discharge_pressure(30000.0, 0)


# ---------------------------------------------------------------------------
# I-Pad — LP + HP series VFD train; capability = amp-limited falling frontier
# ---------------------------------------------------------------------------


def test_i_pad_meta_parameters():
    assert i_pad.specific_gravity() == pytest.approx(1.04)
    assert i_pad.suction_psi() == pytest.approx(217.0)  # PF separator (LP intake)
    # shut-off head/stage matches the 2021 factory section tests (242 & 228 avg 235)
    assert i_pad.head_per_stage(0.0) == pytest.approx(234.54721351548812, rel=1e-9)
    assert i_pad.bhp_per_stage(41250.0) == pytest.approx(62.67763656447321, rel=1e-6)


def test_i_pad_live_scada_dP_anchors():
    # Meta "validated" 2026-06-16: model vs live SCADA within ~1%.
    lp_dp = i_pad.pump_dP(26, 32869.0, 57.3)  # live: 1887 psid
    hp_dp = i_pad.pump_dP(17, 32363.0, 59.3)  # live: 1326 psid
    assert lp_dp == pytest.approx(1873.7792252332197, rel=1e-6)  # regression pin
    assert hp_dp == pytest.approx(1336.428409707658, rel=1e-6)  # regression pin
    assert lp_dp == pytest.approx(1887.0, abs=20.0)  # live anchor
    assert hp_dp == pytest.approx(1326.0, abs=15.0)  # live anchor


def test_i_pad_live_scada_amp_anchors():
    # k values are calibrated to these exact live points (LP 190.9 A, HP 140.6 A).
    assert i_pad.pump_amps(0.1435, 26, 32869.0, 57.3) == pytest.approx(
        190.88655365405523, rel=1e-6
    )
    assert i_pad.pump_amps(0.1487, 17, 32363.0, 59.3) == pytest.approx(
        140.59157714174046, rel=1e-6
    )


def test_i_pad_frontier_pins():
    # max_discharge_pressure: bisection on amps -> deterministic, pin tight.
    assert i_pad.max_discharge_pressure(20000.0) == pytest.approx(
        4143.588367044152, rel=1e-6
    )
    assert i_pad.max_discharge_pressure(32000.0) == pytest.approx(
        3516.7906096999523, rel=1e-6
    )
    assert i_pad.max_discharge_pressure(50000.0) == pytest.approx(
        2215.430744262225, rel=1e-6
    )


def test_i_pad_frontier_falls_with_flow():
    flows = [15000.0, 25000.0, 35000.0, 45000.0, 55000.0]
    frontier = [i_pad.max_discharge_pressure(q) for q in flows]
    assert all(f is not None for f in frontier)
    assert all(a > b for a, b in zip(frontier, frontier[1:]))


def test_i_pad_past_capacity_returns_none_current_behavior():
    # CURRENT BEHAVIOR: past the amp-limited ceiling (between 55 and 58 kBPD)
    # the frontier returns None — the emergent capacity limit, no exception.
    assert i_pad.max_discharge_pressure(55000.0) is not None
    assert i_pad.max_discharge_pressure(58000.0) is None
    assert i_pad.max_discharge_pressure(100000.0) is None


def test_i_pad_max_flow_at_pressure_pins():
    # Frontier inverse (optimizer PF budget). 3,408 = live HP discharge psig.
    assert i_pad.max_flow_at_pressure(3408.0) == pytest.approx(
        33905.25759782655, rel=1e-6
    )
    assert i_pad.max_flow_at_pressure(3500.0) == pytest.approx(
        32293.510315834912, rel=1e-6
    )
    # CURRENT BEHAVIOR: unreachable pressure -> 0.0 (not None, no raise).
    assert i_pad.max_flow_at_pressure(5000.0) == 0.0


def test_i_pad_pump_max_dP_pin():
    # Public helper takes the pump dict shape documented in _pumps(); values
    # here are straight from the meta JSON (LP: 26 stg, 192 A drive limit).
    lp = {"name": "P-1021 LP", "n_stages": 26, "amp_limit": 192.0, "k": 0.1435}
    assert i_pad.pump_max_dP(lp, 32000.0) == pytest.approx(1920.8687196276294, rel=1e-6)


def test_i_pad_affinity_scaling():
    # Affinity laws: same Q60 at 0.8x speed -> dP scales by 0.8^2 exactly.
    base = i_pad.pump_dP(17, 20000.0, 60.0)
    scaled = i_pad.pump_dP(17, 16000.0, 48.0)
    assert base == pytest.approx(1552.3721451104786, rel=1e-6)
    assert scaled == pytest.approx(993.5181728707066, rel=1e-6)
    assert scaled == pytest.approx(0.64 * base, rel=1e-9)


def test_i_pad_operating_envelope():
    rows = i_pad.operating_envelope([30000.0, 70000.0])

    at_30k = rows[0]
    assert at_30k["feasible"] is True
    assert at_30k["amp_limited"] is True
    assert at_30k["max_discharge_psi"] == pytest.approx(3632.0706522215387, rel=1e-6)
    lp_row, hp_row = at_30k["pumps"]
    assert lp_row["hz"] == pytest.approx(58.22009065230634, rel=1e-6)
    assert lp_row["amps"] == pytest.approx(192.0, abs=1e-6)  # pinned at the drive limit
    assert hp_row["hz"] == pytest.approx(60.0)  # HP has amp headroom -> max speed
    assert hp_row["dP"] == pytest.approx(1407.2211028889005, rel=1e-6)

    past_cap = rows[1]
    assert past_cap["feasible"] is False
    assert past_cap["max_discharge_psi"] is None
    assert all(p["hz"] is None for p in past_cap["pumps"])


# ---------------------------------------------------------------------------
# M-Pad — 3x parallel VFD HP bank on a fixed 1,400 psig LP-held suction,
# head derated by the field wear factor from the meta JSON
# ---------------------------------------------------------------------------


def test_m_pad_meta_parameters():
    assert m_pad.specific_gravity() == pytest.approx(1.03)
    assert m_pad.wear_factor() == pytest.approx(0.91)
    assert m_pad.hp_suction_psi() == pytest.approx(1400.0)
    assert m_pad.hp_recommended_flow_per_pump() == (8351.0, 34798.0)
    assert m_pad.min_total_flow(3) == pytest.approx(25053.0)
    assert m_pad.max_total_flow(3) == pytest.approx(104394.0)
    assert m_pad.min_total_flow(2) == pytest.approx(16702.0)
    assert m_pad.max_total_flow(2) == pytest.approx(69596.0)


def test_m_pad_datasheet_anchors_as_new():
    # Schlumberger set-up sheet, 60 Hz @ design SG 1.05 (as-new, wear OFF):
    # shut-in 4,470.7 psig at 1,500 suction -> boost 2,970.7 psi.
    assert m_pad.pump_boost(0.0, 60.0, apply_wear=False, sg=1.05) == pytest.approx(
        2970.7, abs=1.0
    )
    # design point: 32,000 BPD -> boost 2,001.53 psi (fit within ~0.5%)
    design = m_pad.pump_boost(32000.0, 60.0, apply_wear=False, sg=1.05)
    assert design == pytest.approx(1991.3211165090906, rel=1e-6)  # regression pin
    assert design == pytest.approx(2001.53, rel=6e-3)  # datasheet anchor


def test_m_pad_wear_derate_applied():
    # The meta's field_head_factor (0.91) must actually derate the head:
    # derated boost == as-new boost * wear_factor, exactly.
    raw = m_pad.pump_boost(16480.0, 57.0, apply_wear=False)
    derated = m_pad.pump_boost(16480.0, 57.0)
    assert raw == pytest.approx(2322.4485264242016, rel=1e-6)
    assert derated == pytest.approx(2113.4281590460237, rel=1e-6)
    assert derated == pytest.approx(raw * m_pad.wear_factor(), rel=1e-12)
    assert derated < raw


def test_m_pad_live_scada_anchors():
    # Live SCADA 2026-06-16: P-4230A at 16.5 kBPD made ~2,103 psid at 57.9 Hz
    # drawing 125.0 A. The (wear-derated) model needs ~56.9 Hz for that boost.
    hz = m_pad.hz_for_boost(16480.0, 3495.0 - 1392.0)
    assert hz == pytest.approx(56.871626231372616, rel=1e-6)  # regression pin
    assert 55.0 <= hz <= 58.5  # brackets the live HP-bank speeds (55.9-57.9 Hz)
    assert m_pad.pump_amps(16480.0, 57.7) == pytest.approx(124.99150787157245, rel=1e-6)
    assert m_pad.pump_amps(16480.0, 57.7) == pytest.approx(
        125.0, abs=3.0
    )  # live anchor


def test_m_pad_frontier_pins_3_pumps():
    assert m_pad.max_discharge_pressure(30000.0, 3) == pytest.approx(
        3988.7679982488353, rel=1e-6
    )
    assert m_pad.max_discharge_pressure(60000.0, 3) == pytest.approx(
        3753.80488190112, rel=1e-6
    )
    assert m_pad.max_discharge_pressure(100000.0, 3) == pytest.approx(
        3193.142196916884, rel=1e-6
    )


def test_m_pad_frontier_pins_2_pumps():
    assert m_pad.max_discharge_pressure(30000.0, 2) == pytest.approx(
        3885.9352873221214, rel=1e-6
    )
    assert m_pad.max_discharge_pressure(50000.0, 2) == pytest.approx(
        3583.8189526139995, rel=1e-6
    )


def test_m_pad_parallel_split_invariance():
    # Same per-pump flow (20 kBPD) -> identical bank frontier.
    assert m_pad.max_discharge_pressure(40000.0, 2) == pytest.approx(
        m_pad.max_discharge_pressure(60000.0, 3), rel=1e-12
    )


def test_m_pad_frontier_falls_with_flow_and_rises_with_pumps():
    flows = [30000.0, 50000.0, 70000.0, 90000.0]
    frontier = [m_pad.max_discharge_pressure(q, 3) for q in flows]
    assert all(f is not None for f in frontier)
    assert all(a > b for a, b in zip(frontier, frontier[1:]))
    assert m_pad.max_discharge_pressure(50000.0, 3) > m_pad.max_discharge_pressure(
        50000.0, 2
    )


def test_m_pad_off_curve_and_degenerate_inputs_current_behavior():
    # CURRENT BEHAVIOR: None (no raise) when per-pump flow is >5% past the
    # curve max (rec_hi * 1.05 = 36,537.9/pump), and for q <= 0 or n <= 0.
    assert m_pad.max_discharge_pressure(110000.0, 3) is None  # 36,667/pump — off curve
    assert m_pad.max_discharge_pressure(80000.0, 2) is None  # 40,000/pump — off curve
    assert m_pad.max_discharge_pressure(0.0, 3) is None
    assert m_pad.max_discharge_pressure(-5.0, 3) is None
    assert m_pad.max_discharge_pressure(30000.0, 0) is None


def test_m_pad_max_flow_at_pressure_pins():
    assert m_pad.max_flow_at_pressure(3500.0, 3) == pytest.approx(
        81234.71935707163, rel=1e-6
    )
    assert m_pad.max_flow_at_pressure(4000.0, 3) == pytest.approx(
        28123.019364247346, rel=1e-6
    )
    assert m_pad.max_flow_at_pressure(3500.0, 2) == pytest.approx(
        54156.47957138106, rel=1e-6
    )
    # An easy pressure clamps at the off-curve ceiling (max_total_flow).
    assert m_pad.max_flow_at_pressure(3000.0, 3) == pytest.approx(
        m_pad.max_total_flow(3), rel=1e-6
    )
    # CURRENT BEHAVIOR: unreachable pressure -> 0.0 (not None, no raise).
    assert m_pad.max_flow_at_pressure(4200.0, 3) == 0.0


def test_m_pad_affinity_scaling():
    # Same Q60 at 0.8x speed -> boost scales by 0.8^2 exactly (as-new path).
    base = m_pad.pump_boost(20000.0, 60.0, apply_wear=False)
    scaled = m_pad.pump_boost(16000.0, 48.0, apply_wear=False)
    assert scaled == pytest.approx(0.64 * base, rel=1e-9)


def test_m_pad_hz_for_boost_unreachable_returns_none():
    # CURRENT BEHAVIOR: boost beyond max speed capability -> None.
    assert m_pad.hz_for_boost(30000.0, 5000.0) is None


def test_m_pad_operating_envelope():
    rows = m_pad.operating_envelope([30000.0, 60000.0, 120000.0], 3)

    at_30k = rows[0]
    assert at_30k["feasible"] is True
    assert at_30k["recirc"] is False
    assert at_30k["speed_capped"] is False
    # capability (3,988.8) exceeds the 3,500 header cap -> capped at 3,500
    assert at_30k["max_discharge_psi"] == pytest.approx(3500.0)
    (pump_row,) = at_30k["pumps"]
    assert pump_row["hz"] == pytest.approx(55.151717074155115, rel=1e-6)
    assert pump_row["amps"] == pytest.approx(92.55289191368294, rel=1e-6)
    assert (
        pump_row["amps"] < pump_row["amp_limit"]
    )  # lots of amp headroom (the M-Pad story)

    at_60k = rows[1]
    assert at_60k["max_discharge_psi"] == pytest.approx(3500.0)
    assert at_60k["pumps"][0]["hz"] == pytest.approx(58.00913527177198, rel=1e-6)

    off_curve = rows[2]
    assert off_curve["feasible"] is False
    assert off_curve["max_discharge_psi"] is None
    assert off_curve["pumps"] == []
