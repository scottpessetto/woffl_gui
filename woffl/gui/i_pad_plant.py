"""I-Pad booster train — amp-limited delivered-pressure frontier.

I-Pad is a **two-pump series** produced-water / power-fluid injection train:

    PF separator (~217 psig) -> P-1021 LP (26 stg) -> P-0901 HP (17 stg) -> wells

Both units are VFD-driven and built from the same Summit SN35000 stage; the
real ceiling is **motor amps**, so the train's capability is a falling frontier
``max_discharge_pressure(total_flow)`` the optimizer may ride on/under
(Scott's call, 2026-06-16). Physics validated to the psi / amp against live
SCADA 2026-06-16.

The physics now lives in :class:`woffl.gui.pad_plant_base.IPadPlant` (R-1
Phase A unification); this module keeps the original public API as thin
delegations to a singleton so the I-Pad page and the pinned tests in
``tests/test_pad_plants.py`` import it unchanged.

GUI-only (MPU data + optimizer coupling), no upstream library PR — like
``s_pad_plant``. Loads ``woffl/jp_data/I_Pad_Pumps/*meta*.json``.
"""

from typing import Optional

from woffl.gui.pad_plant_base import _FT_TO_PSI_DIVISOR  # noqa: F401
from woffl.gui.pad_plant_base import IPadPlant, poly_eval

_PLANT = IPadPlant()

# public handle for the unified pad optimizer (R-1 Phase B: woffl.gui.pad_optimize)
PLANT = _PLANT

# original module constants, kept for back-compat/introspection
_PUMP_DIR = IPadPlant._PUMP_DIR
_HZ_MAX = IPadPlant._HZ_MAX  # curve is normalized to 60 Hz; drives run at/below it
_HZ_FLOOR = IPadPlant._HZ_FLOOR  # never search below this regardless of flow

# the shared Horner-style evaluator (was this module's _poly)
_poly = poly_eval


def _meta() -> dict:
    return _PLANT._meta()


def specific_gravity() -> float:
    return _PLANT.specific_gravity()


def suction_psi() -> float:
    """Train intake (LP suction) — the PF separator pressure."""
    return _PLANT.suction_psi()


def _max_valid_flow() -> float:
    # still called by i_pad_page's envelope sweep — keep the private name
    return _PLANT.max_valid_flow()


def head_per_stage(q60_bpd: float) -> float:
    """Head (ft) per stage at 60 Hz for the given (60-Hz-equivalent) flow."""
    return _PLANT.head_per_stage(q60_bpd)


def bhp_per_stage(q60_bpd: float) -> float:
    """Water BHP per stage at 60 Hz for the given (60-Hz-equivalent) flow."""
    return _PLANT.bhp_per_stage(q60_bpd)


def _pumps() -> list[dict]:
    """[LP, HP] in series order, each {name, n_stages, amp_limit, k}."""
    return _PLANT.pumps()


def pump_dP(
    n_stages: int, flow_bpd: float, hz: float, sg: float | None = None
) -> float:
    """Differential pressure (psi) a pump makes at a flow + speed (affinity)."""
    return _PLANT.pump_dP(n_stages, flow_bpd, hz, sg)


def pump_amps(k: float, n_stages: int, flow_bpd: float, hz: float) -> float:
    """Motor amps a pump draws at a flow + speed (affinity, live-calibrated k)."""
    return _PLANT.pump_amps(k, n_stages, flow_bpd, hz)


def _hz_at_amp_limit(pump: dict, flow_bpd: float) -> Optional[float]:
    return _PLANT.hz_at_amp_limit(pump, flow_bpd)


def pump_max_dP(
    pump: dict, flow_bpd: float, sg: float | None = None
) -> Optional[float]:
    """Max dP (psi) this pump can deliver at the given flow within its amp limit,
    or None if the flow isn't deliverable (over amps even at minimum head)."""
    return _PLANT.pump_max_dP(pump, flow_bpd, sg)


def max_discharge_pressure(
    total_flow_bpd: float, sg: float | None = None
) -> Optional[float]:
    """Highest header (HP-discharge) pressure the series train can deliver at the
    given total PF flow, both pumps at their amp limits. None past the
    amp-limited flow ceiling (the emergent capacity — no arbitrary hard cap)."""
    return _PLANT.max_discharge_pressure(total_flow_bpd, sg)


def max_flow_at_pressure(pressure: float, sg: float | None = None) -> float:
    """Largest total PF flow the train can deliver at >= ``pressure`` (the frontier
    inverted) — the optimizer's PF budget at a candidate header pressure.
    Returns 0.0 if even minimal flow can't reach the pressure."""
    return _PLANT.max_flow_at_pressure(pressure, sg)


def operating_envelope(flows_bpd) -> list[dict]:
    """For a sweep of total flows, the frontier pressure + each pump's limiting
    speed/amps — for the results-screen viz and headroom readout. ``feasible``
    is False past the amp-limited flow ceiling."""
    return _PLANT.operating_envelope(flows_bpd)
