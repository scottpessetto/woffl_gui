"""PadPlant base-class tests (R-1 Phase A).

The legacy plant modules stay the numeric authority — their pins live in
``tests/test_pad_plants.py`` and must never change. This file asserts:

* PARITY: the uniform ``PadPlant`` interface returns exactly (``==``, not
  approx) the same numbers as the legacy module functions across a grid of
  points on each pad — so Phase B can swap the pages onto the interface
  without any numeric drift.
* the coupling / n_pump_options / max_header_psi contract per pad
* ``pressure_window`` always lands inside PowerFluidConstraint's
  [1000, 5000] band (the cross-cutting constraint from the review)
* ``FixedHeaderPlant`` — the null plant for pads with no booster model
* the legacy shims still expose the private hooks the pages import
  (``i_pad_plant._max_valid_flow`` is called by ``i_pad_page``)
"""

import math

import pytest

import woffl.gui.i_pad_plant as i_pad
import woffl.gui.m_pad_plant as m_pad
import woffl.gui.s_pad_plant as s_pad
from woffl.gui.pad_plant_base import (
    PF_CONSTRAINT_MAX_PSI,
    PF_CONSTRAINT_MIN_PSI,
    FixedHeaderPlant,
    IPadPlant,
    MPadPlant,
    SPadPlant,
    clamp_to_pf_constraint,
    poly_eval,
)

S_PLANT = SPadPlant()
I_PLANT = IPadPlant()
M_PLANT = MPadPlant()

# grids straddle each pad's interesting territory: shut-in, thrust window,
# SCADA anchors, past-capacity extrapolation (S) / None frontier (I, M)
S_FLOWS = [0.0, 15000.0, 20000.0, 30694.0, 45000.0, 55080.0, 70000.0]
I_FLOWS = [6000.0, 20000.0, 32000.0, 50000.0, 55000.0, 58000.0, 100000.0]
M_FLOWS = [26000.0, 30000.0, 60000.0, 100000.0, 110000.0]
I_PRESSURES = [2215.0, 3408.0, 3500.0, 5000.0]
M_PRESSURES = [3000.0, 3500.0, 4000.0, 4200.0]


# ---------------------------------------------------------------------------
# class contract
# ---------------------------------------------------------------------------


def test_coupling_and_pump_options():
    assert S_PLANT.coupling == "fixed_curve"
    assert I_PLANT.coupling == "free_pressure"
    assert M_PLANT.coupling == "free_pressure"
    assert S_PLANT.n_pump_options == [3, 2]
    assert I_PLANT.n_pump_options == []  # fixed LP+HP series train
    assert M_PLANT.n_pump_options == [3, 2, 1]
    assert S_PLANT.max_header_psi is None
    assert I_PLANT.max_header_psi == 3500.0
    assert M_PLANT.max_header_psi == 3500.0


def test_specific_gravity():
    assert S_PLANT.specific_gravity() == pytest.approx(1.0)
    assert I_PLANT.specific_gravity() == i_pad.specific_gravity()
    assert M_PLANT.specific_gravity() == m_pad.specific_gravity()


def test_clamp_to_pf_constraint():
    assert clamp_to_pf_constraint(800.0) == PF_CONSTRAINT_MIN_PSI
    assert clamp_to_pf_constraint(5600.0) == PF_CONSTRAINT_MAX_PSI
    assert clamp_to_pf_constraint(3400.0) == 3400.0


def test_poly_eval_is_the_shared_evaluator():
    # the legacy modules' _poly must BE the base evaluator (no copies left)
    assert i_pad._poly is poly_eval
    assert m_pad._poly is poly_eval
    coeffs = {"c0": 2.0, "c1": -1.0, "c2": 0.5}
    assert poly_eval(coeffs, 3.0) == 2.0 + -1.0 * 3.0 + 0.5 * 9.0


# ---------------------------------------------------------------------------
# S-Pad parity — fixed_curve
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_pumps", [3, 2])
@pytest.mark.parametrize("q", S_FLOWS)
def test_s_pad_header_at_flow_parity(q, n_pumps):
    assert S_PLANT.header_at_flow(q, n_pumps) == s_pad.discharge_pressure(q, n_pumps)


def test_s_pad_default_n_pumps_is_three():
    assert S_PLANT.header_at_flow(30000.0) == s_pad.discharge_pressure(30000.0, 3)
    assert S_PLANT.budget_at_pressure(3400.0) == s_pad.station_capacity(3)


@pytest.mark.parametrize("n_pumps", [3, 2])
def test_s_pad_budget_is_pressure_independent(n_pumps):
    cap = s_pad.station_capacity(n_pumps)
    for p in [1200.0, 3400.0, 4800.0]:
        assert S_PLANT.budget_at_pressure(p, n_pumps) == cap


@pytest.mark.parametrize("n_pumps", [3, 2])
def test_s_pad_flow_window_parity(n_pumps):
    lo, hi = s_pad.recommended_flow_per_pump()
    assert S_PLANT.flow_window(n_pumps) == (lo * n_pumps, hi * n_pumps)
    assert S_PLANT.flow_window(n_pumps)[1] == s_pad.station_capacity(n_pumps)


def test_s_pad_flags():
    assert S_PLANT.flags(30000.0, 3) == {
        "in_range": True,
        "recirc": False,
        "over_capacity": False,
    }
    # per-pump 6,667 < the 7,650 thrust floor — out of range but not over cap
    assert S_PLANT.flags(20000.0, 3) == {
        "in_range": False,
        "recirc": False,
        "over_capacity": False,
    }
    assert S_PLANT.flags(60000.0, 3) == {
        "in_range": False,
        "recirc": False,
        "over_capacity": True,
    }


def test_s_pad_envelope_parity():
    rows = S_PLANT.envelope(S_FLOWS, 3)
    assert len(rows) == len(S_FLOWS)
    for q, row in zip(S_FLOWS, rows):
        assert row["flow"] == q
        assert row["max_discharge_psi"] == s_pad.discharge_pressure(q, 3)
        assert row["per_pump_bpd"] == s_pad.per_pump_flow(q, 3)
        assert row["in_range"] == s_pad.flow_in_range(q, 3)
        assert row["feasible"] == (q <= s_pad.station_capacity(3))
        assert row["pumps"] == []  # fixed speed — no per-pump speed/amp state


def test_s_pad_pressure_window_in_pf_band():
    for n in (3, 2):
        floor, ceiling = S_PLANT.pressure_window(n)
        assert PF_CONSTRAINT_MIN_PSI <= floor < ceiling <= PF_CONSTRAINT_MAX_PSI
        # the window is the curve over the thrust band (clamped)
        lo_q, hi_q = S_PLANT.flow_window(n)
        assert floor == clamp_to_pf_constraint(s_pad.discharge_pressure(hi_q, n))
        assert ceiling == clamp_to_pf_constraint(s_pad.discharge_pressure(lo_q, n))


# ---------------------------------------------------------------------------
# I-Pad parity — free_pressure, fixed train
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", I_FLOWS)
def test_i_pad_header_at_flow_parity(q):
    # includes the None region past the amp-limited ceiling (58k, 100k)
    assert I_PLANT.header_at_flow(q) == i_pad.max_discharge_pressure(q)


@pytest.mark.parametrize("p", I_PRESSURES)
def test_i_pad_budget_at_pressure_parity(p):
    assert I_PLANT.budget_at_pressure(p) == i_pad.max_flow_at_pressure(p)


def test_i_pad_budget_pins_still_hold_through_interface():
    # same numbers the legacy pins assert — the unified _grow_and_bisect
    # must reproduce the old doubling loop exactly
    assert I_PLANT.budget_at_pressure(3408.0) == pytest.approx(
        33905.25759782655, rel=1e-9
    )
    assert I_PLANT.budget_at_pressure(5000.0) == 0.0


def test_i_pad_flow_window():
    lo, hi = I_PLANT.flow_window()
    assert lo == 0.0
    # ceiling = throughput at barely-above-suction (the page's eval cap)
    assert hi == i_pad.max_flow_at_pressure(i_pad.suction_psi() + 200.0)
    assert 50000.0 < hi < 60000.0  # the emergent ~57k amp ceiling


def test_i_pad_flags():
    assert I_PLANT.flags(30000.0) == {
        "in_range": True,
        "recirc": False,
        "over_capacity": False,
    }
    assert I_PLANT.flags(100000.0) == {
        "in_range": False,
        "recirc": False,
        "over_capacity": True,
    }


def test_i_pad_envelope_parity():
    assert I_PLANT.envelope([30000.0, 70000.0]) == i_pad.operating_envelope(
        [30000.0, 70000.0]
    )


def test_i_pad_pressure_window_in_pf_band():
    floor, ceiling = I_PLANT.pressure_window()
    assert PF_CONSTRAINT_MIN_PSI <= floor < ceiling <= PF_CONSTRAINT_MAX_PSI
    assert ceiling <= I_PLANT.max_header_psi  # operational cap respected


def test_i_pad_page_private_hook_still_works():
    # i_pad_page.py sweeps `while q <= 4.0 * plant._max_valid_flow()`
    assert i_pad._max_valid_flow() == I_PLANT.max_valid_flow()
    assert i_pad._max_valid_flow() > 0


# ---------------------------------------------------------------------------
# M-Pad parity — free_pressure, parallel bank with pump-count choice
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_pumps", [3, 2])
@pytest.mark.parametrize("q", M_FLOWS)
def test_m_pad_header_at_flow_parity(q, n_pumps):
    assert M_PLANT.header_at_flow(q, n_pumps) == m_pad.max_discharge_pressure(
        q, n_pumps
    )


@pytest.mark.parametrize("n_pumps", [3, 2])
@pytest.mark.parametrize("p", M_PRESSURES)
def test_m_pad_budget_at_pressure_parity(p, n_pumps):
    assert M_PLANT.budget_at_pressure(p, n_pumps) == m_pad.max_flow_at_pressure(
        p, n_pumps
    )


def test_m_pad_default_n_pumps_is_three():
    assert M_PLANT.header_at_flow(60000.0) == m_pad.max_discharge_pressure(60000.0, 3)
    assert M_PLANT.budget_at_pressure(3500.0) == m_pad.max_flow_at_pressure(3500.0, 3)


@pytest.mark.parametrize("n_pumps", [3, 2, 1])
def test_m_pad_flow_window_parity(n_pumps):
    assert M_PLANT.flow_window(n_pumps) == (
        m_pad.min_total_flow(n_pumps),
        m_pad.max_total_flow(n_pumps),
    )


def test_m_pad_flags():
    assert M_PLANT.flags(30000.0, 3) == {
        "in_range": True,
        "recirc": False,
        "over_capacity": False,
    }
    # below the recirc floor (25,053 on 3 pumps)
    assert M_PLANT.flags(20000.0, 3) == {
        "in_range": False,
        "recirc": True,
        "over_capacity": False,
    }
    # off-curve: 110k/3 pumps is past rec_hi * 1.05
    assert M_PLANT.flags(110000.0, 3) == {
        "in_range": False,
        "recirc": False,
        "over_capacity": True,
    }
    # degenerate zero flow: recirc, not over-capacity
    assert M_PLANT.flags(0.0, 3) == {
        "in_range": False,
        "recirc": True,
        "over_capacity": False,
    }


def test_m_pad_envelope_parity_default_cap():
    flows = [30000.0, 60000.0, 120000.0]
    assert M_PLANT.envelope(flows, 3) == m_pad.operating_envelope(
        flows, 3, header_cap=3500.0
    )


def test_m_pad_envelope_parity_at_pressure():
    flows = [30000.0, 60000.0]
    assert M_PLANT.envelope(flows, 3, at_pressure=3300.0) == m_pad.operating_envelope(
        flows, 3, header_cap=3300.0
    )


def test_m_pad_pressure_window_in_pf_band():
    for n in (3, 2):
        floor, ceiling = M_PLANT.pressure_window(n)
        assert PF_CONSTRAINT_MIN_PSI <= floor < ceiling <= PF_CONSTRAINT_MAX_PSI
        assert ceiling <= M_PLANT.max_header_psi
        # floor sits above the LP-held suction (1,400 + 300 = 1,700)
        assert floor == pytest.approx(1700.0)


# ---------------------------------------------------------------------------
# FixedHeaderPlant — the null plant for pads with no booster model
# ---------------------------------------------------------------------------


def test_fixed_header_plant_contract():
    plant = FixedHeaderPlant(3400.0)
    assert plant.coupling == "fixed_curve"
    assert plant.n_pump_options == []
    assert plant.max_header_psi is None
    assert plant.specific_gravity() == 1.0


def test_fixed_header_plant_constant_header():
    plant = FixedHeaderPlant(3400.0)
    for q in [0.0, 10000.0, 500000.0]:
        assert plant.header_at_flow(q) == 3400.0


def test_fixed_header_plant_unbounded_budget_and_flow():
    plant = FixedHeaderPlant(3400.0)
    assert math.isinf(plant.budget_at_pressure(4999.0))
    lo, hi = plant.flow_window()
    assert lo == 0.0 and math.isinf(hi)


def test_fixed_header_plant_pressure_window_clamped():
    assert FixedHeaderPlant(3400.0).pressure_window() == (3400.0, 3400.0)
    # a no-booster pad below/above the PF band must still hand the optimizer
    # a legal pressure (PowerFluidConstraint rejects outside [1000, 5000])
    assert FixedHeaderPlant(800.0).pressure_window() == (
        PF_CONSTRAINT_MIN_PSI,
        PF_CONSTRAINT_MIN_PSI,
    )
    assert FixedHeaderPlant(5600.0).pressure_window() == (
        PF_CONSTRAINT_MAX_PSI,
        PF_CONSTRAINT_MAX_PSI,
    )


def test_fixed_header_plant_no_flags():
    plant = FixedHeaderPlant(3400.0)
    assert plant.flags(1e9) == {
        "in_range": True,
        "recirc": False,
        "over_capacity": False,
    }


def test_fixed_header_plant_envelope():
    rows = FixedHeaderPlant(3400.0).envelope([10000.0, 20000.0])
    assert [r["flow"] for r in rows] == [10000.0, 20000.0]
    assert all(r["max_discharge_psi"] == 3400.0 for r in rows)
    assert all(r["feasible"] and r["pumps"] == [] for r in rows)


# ---------------------------------------------------------------------------
# shims delegate to the singletons (no second copy of the physics)
# ---------------------------------------------------------------------------


def test_module_singletons_are_the_base_subclasses():
    assert isinstance(s_pad._PLANT, SPadPlant)
    assert isinstance(i_pad._PLANT, IPadPlant)
    assert isinstance(m_pad._PLANT, MPadPlant)


def test_meta_cache_shared_across_instances():
    # class-level cache mirrors the old lru_cache(1): one JSON read per pad,
    # so fresh instances (parity tests, Phase B specs) don't re-read disk
    assert SPadPlant()._meta() is s_pad._meta()
    assert IPadPlant()._meta() is i_pad._meta()
    assert MPadPlant()._meta() is m_pad._meta()
