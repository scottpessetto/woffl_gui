"""CFP plant curve — pinned to the source spreadsheet's own table values."""

import pytest

from woffl.assembly.cfp_plant import (
    MACHINE_COEFFS,
    PAD_LINE_DP,
    delivered_pf_pressure,
    machine_flow,
    plant_flow,
    plant_pressure,
)


def test_spreadsheet_anchor_2200():
    # inputs!N3: 125,901 BWPD total at 2,200 psi (machines 41,361/40,782/43,758)
    assert plant_flow(2200.0) == pytest.approx(125901, abs=2)
    assert machine_flow("A", 2200.0) == pytest.approx(41361, abs=2)
    assert machine_flow("B", 2200.0) == pytest.approx(40782, abs=2)
    assert machine_flow("C", 2200.0) == pytest.approx(43758, abs=2)


def test_spreadsheet_anchor_2700():
    # inputs!N8: 101,428 BWPD total at 2,700 psi
    assert plant_flow(2700.0) == pytest.approx(101428, abs=2)


def test_flow_decreases_with_pressure():
    flows = [plant_flow(p) for p in (2200.0, 2400.0, 2600.0, 2800.0)]
    assert all(a > b for a, b in zip(flows, flows[1:]))


def test_pressure_roundtrip():
    for p in (2250.0, 2500.0, 2750.0):
        assert plant_pressure(plant_flow(p)) == pytest.approx(p, abs=0.5)


def test_pressure_clamps_outside_window():
    assert plant_pressure(1e9) == pytest.approx(1800.0)
    assert plant_pressure(0.0) == pytest.approx(3000.0)


def test_machine_above_shutoff_returns_zero():
    # C machine's shutoff head (c coefficient) is ~2,152 psi + small b term;
    # far above any machine's head, flow must be 0, not a math error.
    assert machine_flow("C", 5000.0) == 0.0


def test_delivered_pf():
    # Spreadsheet dPs referenced to the 2,697 psi snapshot
    assert delivered_pf_pressure("J", 2697.0) == pytest.approx(2446.0)
    assert delivered_pf_pressure("G", 2697.0) == pytest.approx(2404.0)
    assert delivered_pf_pressure("B", 2697.0) == pytest.approx(2425.0)
    # C-Pad is boosted — not tied to the plant curve. H's spreadsheet entry
    # was legacy (pre-POPS); H takes no plant PF now.
    assert delivered_pf_pressure("C", 2697.0) is None
    assert delivered_pf_pressure("H", 2697.0) is None


def test_pad_dp_table_complete():
    assert set(PAD_LINE_DP) == {"B", "G", "J"}
    assert set(MACHINE_COEFFS) == {"A", "B", "C"}


# ── the data file and the hardcoded constants must not drift ────────────────
# woffl/jp_data/CFP_Pumps/pump_curve_meta.json is the record + the swap target,
# but cfp_plant.py still hardcodes the numbers. If someone updates one and not
# the other, the README's swap contract silently lies.


def _meta():
    import json
    from pathlib import Path

    p = (
        Path(__file__).resolve().parent.parent
        / "woffl"
        / "jp_data"
        / "CFP_Pumps"
        / "pump_curve_meta.json"
    )
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def test_meta_machine_coeffs_match_the_module():
    meta = _meta()["machine_coeffs"]
    for machine, coeffs in MACHINE_COEFFS.items():
        assert tuple(meta[machine]) == tuple(coeffs), machine


def test_meta_scalars_match_the_module():
    from woffl.assembly.cfp_plant import (
        MACHINE_CURVE_VALIDATED,
        MEASURED_DISCHARGE_PSI,
        MEASURED_PRODUCED_WATER_BWPD,
        PRESSURE_WINDOW,
        TRUSTED_BAND,
    )
    from woffl.assembly.cfp_plant import MAX_DISCHARGE_PSI as MAXD
    from woffl.assembly.cfp_plant import PAD_LINE_DP as DP

    m = _meta()
    assert m["max_discharge_psi"] == MAXD
    assert tuple(m["pressure_window"]) == PRESSURE_WINDOW
    assert tuple(m["trusted_band"]) == TRUSTED_BAND
    assert m["machine_curve_validated"] is MACHINE_CURVE_VALIDATED
    op = m["measured_operating_point_2026_07_29"]
    assert op["discharge_psi"] == MEASURED_DISCHARGE_PSI
    assert op["produced_water_bwpd"] == MEASURED_PRODUCED_WATER_BWPD
    for pad, dp in DP.items():
        assert m["pad_line_dp_psi"][pad] == dp, pad


def test_provisional_curve_over_predicts_the_real_machine_total():
    """Documents WHY everything is badged provisional.

    The acceptance basis is the TOTAL OF THE THREE MACHINES (~86,000 BPD at
    ~2,790 psi), NOT metered produced water (~112,300 BWPD) — only ~77% of
    produced water passes these pumps. Against that basis the fit runs ~11%
    HIGH, not low; an earlier version of this test asserted the opposite,
    because it compared against produced water.

    When the new coefficients land, flip this to assert the acceptance point.
    """
    from woffl.assembly.cfp_plant import (
        MEASURED_DISCHARGE_PSI,
        MEASURED_MACHINE_TOTAL_BPD,
    )

    modeled = plant_flow(MEASURED_DISCHARGE_PSI)
    assert modeled > MEASURED_MACHINE_TOTAL_BPD, "the fit over-predicts"
    ratio = MEASURED_MACHINE_TOTAL_BPD / modeled
    assert 0.85 <= ratio <= 0.95, f"expected ~0.88 scale error, got {ratio:.2f}"
    assert not _meta()["validated"]


def test_machine_flow_tags_are_the_confirmed_ones():
    """MPU_FIC_5488/5489 are a DIFFERENT stream and produced two wrong
    conclusions (a two-machine plant, and a 0.99 curve match). Pin the real
    ones so that can't recur."""
    from woffl.assembly.cfp_plant import (
        GPM_TO_BPD,
        MACHINE_FLOW_GPM_TAGS,
        MACHINE_FLOW_TAGS,
    )

    assert MACHINE_FLOW_TAGS == {
        "A": "MPU_FIC_5419S",
        "B": "MPU_FIC_5420S",
        "C": "MPU_FIC_5421S",
    }
    assert set(MACHINE_FLOW_GPM_TAGS) == set(MACHINE_FLOW_TAGS)
    assert "MPU_FIC_5488" not in MACHINE_FLOW_TAGS.values()
    assert "MPU_FIC_5489" not in MACHINE_FLOW_TAGS.values()
    # 892 GPM on the SCADA screen read 30,582 BPD.
    assert 892 * GPM_TO_BPD == pytest.approx(30583, abs=2)


def test_all_three_machines_are_modeled():
    """All three run — an earlier pass wrongly inferred two."""
    from woffl.assembly.cfp_plant import ALL_MACHINES, MEASURED_MACHINE_TOTAL_BPD

    assert set(ALL_MACHINES) == {"A", "B", "C"}
    # A 29,844 + B 26,453 + C 29,931
    assert MEASURED_MACHINE_TOTAL_BPD == pytest.approx(86228.0)
