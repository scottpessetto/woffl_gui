"""M-Pad (Moose Pad, Mod 42) booster — HP-bank delivered-pressure frontier.

M-Pad is a HYBRID station (different from both S-Pad and I-Pad):

    produced water -> [3x P-4220 LP, PARALLEL] -> ~1,400 psig header
                      -> [3x P-4230 HP, PARALLEL] -> ~3,500 psig Power Fluid header

Parallel WITHIN each bank (flows add, share dP), SERIES BETWEEN banks. v1
models the HP bank only on a fixed LP-held suction; the binding limits are
MIN-FLOW (recirculation shutdown) and VFD max speed, NOT amps, and installed
head is derated by ``field_head_factor`` (~0.91 solids wear). Physics
validated to the psi against live SCADA 2026-06-16.

The physics now lives in :class:`woffl.gui.pad_plant_base.MPadPlant` (R-1
Phase A unification); this module keeps the original public API as thin
delegations to a singleton so the M-Pad page and the pinned tests in
``tests/test_pad_plants.py`` import it unchanged.

GUI-only (no upstream PR), like the other pad plants. Loads
``woffl/jp_data/M_Pad_Pumps/*meta*.json``.
"""

from typing import Optional

from woffl.gui.pad_plant_base import _FT_TO_PSI_DIVISOR  # noqa: F401
from woffl.gui.pad_plant_base import MPadPlant, poly_eval

_PLANT = MPadPlant()

# public handle for the unified pad optimizer (R-1 Phase B: woffl.gui.pad_optimize)
PLANT = _PLANT

# original module constants, kept for back-compat/introspection
_PUMP_DIR = MPadPlant._PUMP_DIR
_HP_SUCTION_DEFAULT = (
    MPadPlant._HP_SUCTION_DEFAULT
)  # LP-held header (PIC-4221 setpoint)

# the shared Horner-style evaluator (was this module's _poly; M-Pad polys are
# TOTAL-pump head/BHP, cubic)
_poly = poly_eval


def _meta() -> dict:
    return _PLANT._meta()


def specific_gravity() -> float:
    return _PLANT.specific_gravity()


def wear_factor() -> float:
    """Field head derate (~0.91): installed pumps make less head than as-new."""
    return _PLANT.wear_factor()


def hp_suction_psi() -> float:
    return _PLANT.hp_suction_psi()


def _hp() -> dict:
    return _PLANT.hp()


def _head_ft(pump: dict, q60: float) -> float:
    return MPadPlant._head_ft(pump, q60)


def _bhp(pump: dict, q60: float) -> float:
    return MPadPlant._bhp(pump, q60)


def pump_boost(
    q_per_pump: float, hz: float, *, apply_wear: bool = True, sg: float | None = None
) -> float:
    """HP-pump differential pressure (psi) at a per-pump flow + speed (affinity).
    Head is field-derated by the wear factor; pass apply_wear=False for as-new."""
    return _PLANT.pump_boost(q_per_pump, hz, apply_wear=apply_wear, sg=sg)


def pump_amps(q_per_pump: float, hz: float) -> float:
    """HP-pump motor amps at a per-pump flow + speed (as-new BHP curve, no wear)."""
    return _PLANT.pump_amps(q_per_pump, hz)


def hp_recommended_flow_per_pump() -> tuple[float, float]:
    return _PLANT.hp_recommended_flow_per_pump()


def min_total_flow(n_pumps: int = 3) -> float:
    """Recirculation floor: below this total PF the HP pumps drop under their
    min recommended per-pump flow (high-diff / low-flow shutdown)."""
    return _PLANT.min_total_flow(n_pumps)


def max_total_flow(n_pumps: int = 3) -> float:
    """Off-curve ceiling: above this the per-pump flow exceeds the curve range."""
    return _PLANT.max_total_flow(n_pumps)


def hz_for_boost(q_per_pump: float, target_boost: float, *, sg: float | None = None):
    """Speed (Hz, <= hz_max) at which the HP pump makes ``target_boost`` at the
    given per-pump flow, or None if it can't reach it even at max speed."""
    return _PLANT.hz_for_boost(q_per_pump, target_boost, sg=sg)


def max_discharge_pressure(
    total_flow_bpd: float, n_pumps: int = 3, sg: float | None = None
) -> Optional[float]:
    """Highest HP-discharge (power-fluid header) pressure the bank can deliver at
    the given total PF flow, all n pumps at max speed. The page caps it at the
    operational limit (3,500). Falls with flow. None if off-curve."""
    return _PLANT.max_discharge_pressure(total_flow_bpd, n_pumps, sg)


def max_flow_at_pressure(
    pressure: float, n_pumps: int = 3, sg: float | None = None
) -> float:
    """Largest total PF the bank can deliver at >= ``pressure`` (frontier inverse),
    for the optimizer's PF budget at a candidate header pressure."""
    return _PLANT.max_flow_at_pressure(pressure, n_pumps, sg)


def operating_envelope(
    flows_bpd, n_pumps: int = 3, *, header_cap: float = 3500.0
) -> list[dict]:
    """For a sweep of total flows, the deliverable header (capability capped at
    ``header_cap``) + the HP pumps' operating speed/amps/headroom and a recirc
    flag — for the results viz."""
    return _PLANT.operating_envelope(flows_bpd, n_pumps, header_cap=header_cap)
