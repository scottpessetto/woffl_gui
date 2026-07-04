"""S-Pad booster pump curve — delivered power-fluid (header) pressure vs. flow.

Three identical Weatherford/Borets ESPi675TJ-12000 booster pumps (P-800 B-1/2/3)
feed a common discharge header on S-Pad. They run in **parallel**: same
differential pressure, flows add, so each online pump carries
``total_flow / n_online``. The delivered header pressure that every S-Pad jet
pump sees is ``suction + dP`` — the single common PF surface pressure for the
whole pad.

The physics now lives in :class:`woffl.gui.pad_plant_base.SPadPlant` (R-1
Phase A unification); this module keeps the original public API as thin
delegations to a singleton so the pad pages and the pinned tests in
``tests/test_pad_plants.py`` import it unchanged. Fit + provenance:
``woffl/jp_data/S_Pad_Pumps/`` (validated vs. SCADA within ~1%, 2026-06-15).

GUI-only (MPU-specific data + the optimizer coupling lives in the S-Pad page),
so no upstream library PR — unlike the assembly-level ``cfp_plant.py``.
"""

from woffl.gui.pad_plant_base import _FT_TO_PSI_DIVISOR, SPadPlant  # noqa: F401

_PLANT = SPadPlant()

# public handle for the unified pad optimizer (R-1 Phase B: woffl.gui.pad_optimize)
PLANT = _PLANT

# original module constants, kept for back-compat/introspection
_META_PATH = SPadPlant._META_PATH
_FALLBACK = SPadPlant._FALLBACK


def _meta() -> dict:
    return _PLANT._meta()


def n_pumps_installed() -> int:
    return _PLANT.n_pumps_installed()


def recommended_flow_per_pump() -> tuple[float, float]:
    return _PLANT.recommended_flow_per_pump()


def head_per_stage(q_per_pump_bpd: float) -> float:
    """Head (ft) produced by one stage at the given per-pump flow (BPD)."""
    return _PLANT.head_per_stage(q_per_pump_bpd)


def discharge_pressure(
    total_flow_bpd: float,
    n_pumps: int = 3,
    *,
    sg: float | None = None,
    suction_psi: float | None = None,
) -> float:
    """Common header discharge pressure (psi) for a given total station flow.

    Args:
        total_flow_bpd: total power-fluid rate through the booster station (sum
            of every S-Pad well's nozzle/lift-water flow), BPD.
        n_pumps: number of booster pumps online (2 or 3). Flows split evenly.
        sg / suction_psi: override the meta defaults if needed.
    """
    return _PLANT.discharge_pressure(
        total_flow_bpd, n_pumps, sg=sg, suction_psi=suction_psi
    )


def per_pump_flow(total_flow_bpd: float, n_pumps: int = 3) -> float:
    return _PLANT.per_pump_flow(total_flow_bpd, n_pumps)


def flow_in_range(total_flow_bpd: float, n_pumps: int = 3) -> bool:
    """True if per-pump flow sits inside the recommended (thrust) window."""
    return _PLANT.flow_in_range(total_flow_bpd, n_pumps)


def station_capacity(n_pumps: int = 3) -> float:
    """Upper hydraulic flow ceiling (BPD) — recommended max per pump × n."""
    return _PLANT.station_capacity(n_pumps)
