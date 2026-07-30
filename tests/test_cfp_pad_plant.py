"""CFPPlant — the PadPlant wrapper over the CFP produced-water machines (W1).

Companion to ``tests/test_cfp_plant.py``, which pins the underlying curve.
This file asserts the INTERFACE contract, and in particular the three things
that would silently produce wrong recommendations if they regressed:

* ``coupling == "free_pressure"``. The original plan had CFP as
  ``fixed_curve`` (iterate water -> pressure to a fixed point). Live data
  refuted that: 120 days of metered throughput vs measured discharge gives a
  -1.8 psi/1,000 BWPD slope at r²=0.03, against the -17.5 the curve implies,
  because operators SET the pressure by throttling the disposal well. If this
  flips back to fixed_curve the model starts predicting ~700 psi of PF sag
  where reality shows 75.
* the 2,900 psi piping trip bounds every pressure the class hands out.
* machine SUBSETS refuse to produce numbers until the per-machine curve is
  validated.
"""

import pytest

from woffl.assembly import cfp_plant as cfp
from woffl.gui.cfp_pad_plant import (
    PLANT,
    CFPMachineSubsetUnvalidated,
    CFPPlant,
)
from woffl.gui.pad_plant_base import (
    PF_CONSTRAINT_MAX_PSI,
    PF_CONSTRAINT_MIN_PSI,
    PadPlant,
)


# ── class contract ──────────────────────────────────────────────────────────


def test_is_a_padplant():
    assert isinstance(PLANT, PadPlant)


def test_coupling_is_free_pressure_not_fixed_curve():
    """The load-bearing design decision — see the module docstring."""
    assert PLANT.coupling == "free_pressure"


def test_pump_options_and_trip():
    assert PLANT.n_pump_options == [3, 2, 1]
    assert PLANT.max_header_psi == cfp.MAX_DISCHARGE_PSI == 2900.0


def test_specific_gravity_is_produced_water():
    assert 1.0 < PLANT.specific_gravity() < 1.1


# ── the 2,900 psi trip bounds everything ────────────────────────────────────


def test_pressure_window_ceiling_is_the_trip():
    lo, hi = PLANT.pressure_window()
    assert hi == cfp.MAX_DISCHARGE_PSI
    assert PF_CONSTRAINT_MIN_PSI <= lo < hi <= PF_CONSTRAINT_MAX_PSI


def test_clamp_window_inside_pf_constraint_band():
    lo, hi = PLANT.clamp_window()
    assert PF_CONSTRAINT_MIN_PSI <= lo < hi <= PF_CONSTRAINT_MAX_PSI
    assert hi == cfp.MAX_DISCHARGE_PSI


@pytest.mark.parametrize("q", [40000.0, 60000.0, 78539.0, 88195.0, 112327.0])
def test_header_at_flow_never_exceeds_the_trip(q):
    """Low flows invert to >2,900 psi on the raw curve; the class must cap them
    (the raw `plant_pressure` will happily return 3,000 psi)."""
    p = PLANT.header_at_flow(q)
    assert p is not None
    assert p <= cfp.MAX_DISCHARGE_PSI


def test_low_flow_would_exceed_the_trip_without_the_cap():
    """Guards the cap by showing the underlying curve really does go over."""
    assert cfp.plant_pressure(60000.0) > cfp.MAX_DISCHARGE_PSI
    assert PLANT.header_at_flow(60000.0) == cfp.MAX_DISCHARGE_PSI


def test_warm_start_is_the_measured_discharge():
    assert PLANT.warm_start_psi() == cfp.MEASURED_DISCHARGE_PSI


def test_match_check_header_is_measured_not_a_unit_error():
    """The base implementation would feed a PF rate into header_at_flow, which
    on this plant expects TOTAL WATER. The override must ignore its argument."""
    assert PLANT.match_check_header(27000.0) == cfp.MEASURED_DISCHARGE_PSI
    assert PLANT.match_check_header(0.0) == cfp.MEASURED_DISCHARGE_PSI


# ── total-water semantics (NOT a PF budget) ─────────────────────────────────


@pytest.mark.parametrize(
    "pressure,expected",
    [(2200.0, 125901.0), (2400.0, 116983.0), (2700.0, 101428.0), (2900.0, 88195.0)],
)
def test_budget_at_pressure_is_total_water(pressure, expected):
    assert PLANT.budget_at_pressure(pressure) == pytest.approx(expected, abs=2)
    # …and it is exactly the underlying plant_flow, i.e. total throughput.
    assert PLANT.budget_at_pressure(pressure) == cfp.plant_flow(pressure)


def test_budget_falls_as_pressure_rises():
    """The pressure/volume trade that IS the CFP optimization."""
    budgets = [PLANT.budget_at_pressure(p) for p in (2200, 2400, 2700, 2900)]
    assert all(a > b for a, b in zip(budgets, budgets[1:]))


def test_flow_window_ceiling_is_window_floor_throughput():
    lo, hi = PLANT.flow_window()
    assert lo == 0.0
    assert hi == pytest.approx(cfp.plant_flow(cfp.PRESSURE_WINDOW[0]))


# ── honest flags: over-capacity, clamping, extrapolation ────────────────────


def test_flags_at_a_normal_operating_point():
    f = PLANT.flags(112327.0)
    assert f["in_range"] and not f["over_capacity"]
    assert f["pinned"] == "interior"


def test_flags_over_capacity_is_reported_not_swallowed():
    f = PLANT.flags(200000.0)
    assert f["over_capacity"] is True
    assert f["in_range"] is False
    assert f["pinned"] == "pinned_low"


def test_header_at_flow_none_past_capability():
    assert PLANT.header_at_flow(200000.0) is None


def test_flags_flag_extrapolation_outside_the_trusted_band():
    """2,200-2,700 psi is all the spreadsheet table covers."""
    assert PLANT.flags(112327.0)["trusted_band"] is True  # ~2,496 psi
    assert PLANT.flags(60000.0)["trusted_band"] is False  # capped to 2,900


def test_envelope_rows_carry_feasibility_and_provenance():
    rows = PLANT.envelope([60000.0, 112327.0, 200000.0])
    assert [r["feasible"] for r in rows] == [True, True, False]
    assert all(r["pumps"] == 3 for r in rows)
    assert all(r["max_discharge_psi"] <= cfp.MAX_DISCHARGE_PSI for r in rows)
    assert rows[0]["machines"] == "A,B,C"
    assert set(rows[0]["per_machine_bwpd"]) == {"A", "B", "C"}


# ── machine subsets are gated ───────────────────────────────────────────────


def test_three_machines_is_allowed():
    assert PLANT.machines_for(3) == ("A", "B", "C")
    assert PLANT.machines_for(None) == ("A", "B", "C")


@pytest.mark.parametrize("n", [1, 2])
def test_machine_subset_refuses_until_validated(n):
    assert cfp.MACHINE_CURVE_VALIDATED is False, "flip means the gate should open"
    with pytest.raises(CFPMachineSubsetUnvalidated):
        PLANT.machines_for(n)


def test_subset_gate_subclasses_valueerror():
    """So the pad pages' existing `except ValueError` paths still catch it."""
    assert issubclass(CFPMachineSubsetUnvalidated, ValueError)


def test_machine_subset_available_reports_the_gate():
    assert PLANT.machine_subset_available() is cfp.MACHINE_CURVE_VALIDATED


def test_subset_opens_when_validated(monkeypatch):
    monkeypatch.setattr(cfp, "MACHINE_CURVE_VALIDATED", True)
    plant = CFPPlant()
    assert plant.machines_for(2) == ("A", "B")
    assert plant.machines_for(1) == ("A",)
    assert plant.budget_at_pressure(2400.0, n_pumps=2) == cfp.plant_flow(
        2400.0, ("A", "B")
    )


def test_bad_pump_count_rejected():
    with pytest.raises(ValueError):
        PLANT.machines_for(4)


# ── plant -> pad delivery ───────────────────────────────────────────────────


def test_c_pad_is_not_plant_supplied():
    """C-Pad is boosted on-pad (~3,400 psi measured) — its PF is an input."""
    assert PLANT.delivered_pf_for_pad("C", 2792.0) is None
    assert "C" not in PLANT.plant_supplied_pads()
    assert PLANT.plant_supplied_pads() == ("B", "G", "J")


@pytest.mark.parametrize("pad,dp", [("B", 272.0), ("G", 293.0), ("J", 251.0)])
def test_table_dp_fallback(pad, dp):
    assert PLANT.delivered_pf_for_pad(pad, 2792.0) == pytest.approx(2792.0 - dp)


def test_measured_anchor_beats_the_table():
    """B-pad's 5 header wells cluster at ~2,623 psi, implying ~169 psi of line
    loss, not the table's 272 — so the anchored value must win and must be
    ~103 psi higher than the table fallback."""
    anchored = PLANT.delivered_pf_for_pad("B", 2792.0, measured_pad_pf=2623.0)
    table = PLANT.delivered_pf_for_pad("B", 2792.0)
    assert anchored == pytest.approx(2623.0)
    assert anchored - table == pytest.approx(103.0)


def test_measured_anchor_tracks_discharge_one_for_one():
    base = PLANT.delivered_pf_for_pad("J", cfp.MEASURED_DISCHARGE_PSI,
                                     measured_pad_pf=2682.0)
    up = PLANT.delivered_pf_for_pad("J", cfp.MEASURED_DISCHARGE_PSI + 100.0,
                                    measured_pad_pf=2682.0)
    assert base == pytest.approx(2682.0)
    assert up - base == pytest.approx(100.0)


def test_pad_key_is_case_and_format_tolerant():
    for spec in ("b", "B", "MPB-28", "b-pad"):
        assert PLANT.delivered_pf_for_pad(spec, 2792.0) == pytest.approx(2520.0)


def test_well_names_do_not_all_collapse_to_m_pad():
    """The trap: naive spec[0] on "MPB-28" yields "M", so every well name would
    resolve to M-Pad and quietly take the fallback path."""
    from woffl.gui.cfp_pad_plant import pad_letter

    assert pad_letter("MPB-28") == "B"
    assert pad_letter("MPJ-29") == "J"
    assert pad_letter("MPG-18") == "G"
    assert pad_letter("MPC-45") == "C"
    # A genuine non-CFP pad still resolves to itself and gets no plant delivery.
    assert pad_letter("MPM-19") == "M"
    assert PLANT.delivered_pf_for_pad("MPM-19", 2792.0) is None
    assert PLANT.delivered_pf_for_pad("MPJ-29", 2792.0) == pytest.approx(2541.0)
