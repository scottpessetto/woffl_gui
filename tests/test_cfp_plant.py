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
