"""S-Pad booster pump curve — delivered power-fluid (header) pressure vs. flow.

Three identical Weatherford/Borets ESPi675TJ-12000 booster pumps (P-800 B-1/2/3)
feed a common discharge header on S-Pad. They run in **parallel**: same
differential pressure, flows add, so each online pump carries
``total_flow / n_online``. The delivered header pressure that every S-Pad jet
pump sees is ``suction + dP`` — the single common PF surface pressure for the
whole pad.

The fit + provenance live in ``woffl/jp_data/S_Pad_Pumps/`` (validated vs. SCADA
within ~1%, 2026-06-15): a head-per-stage cubic, 79 stages/unit, 3500 RPM,
SG 1.0, ~220 psi suction. We load the polynomial + parameters from
``pump_curve_meta.json`` so re-running Scott's ``build_pump_curves.py`` flows
through here with no code change.

GUI-only (MPU-specific data + the optimizer coupling lives in the S-Pad page),
so no upstream library PR — unlike the assembly-level ``cfp_plant.py``.
"""

import json
from functools import lru_cache
from pathlib import Path

_META_PATH = (
    Path(__file__).resolve().parent.parent
    / "jp_data" / "S_Pad_Pumps" / "pump_curve_meta.json"
)

# Fallback if the JSON is ever missing — the 2026-06-15 SCADA-validated fit.
_FALLBACK = {
    "n_stages": 79,
    "n_pumps_parallel": 3,
    "specific_gravity": 1.0,
    "suction_psi": 220.0,
    "recommended_flow_per_pump_bpd": [7650, 18360],
    "head_per_stage_poly": {
        "c0": 107.22128868404283,
        "c1": -0.00036333360428150995,
        "c2": -4.557184603974567e-08,
        "c3": -3.867646778059496e-12,
    },
}

_FT_TO_PSI_DIVISOR = 2.31  # ft of head -> psi at SG 1.0 (README/meta convention)


@lru_cache(maxsize=1)
def _meta() -> dict:
    try:
        with open(_META_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return _FALLBACK


def n_pumps_installed() -> int:
    return int(_meta().get("n_pumps_parallel", 3))


def recommended_flow_per_pump() -> tuple[float, float]:
    lo, hi = _meta().get("recommended_flow_per_pump_bpd", [7650, 18360])
    return float(lo), float(hi)


def head_per_stage(q_per_pump_bpd: float) -> float:
    """Head (ft) produced by one stage at the given per-pump flow (BPD)."""
    p = _meta().get("head_per_stage_poly", _FALLBACK["head_per_stage_poly"])
    q = q_per_pump_bpd
    return p["c0"] + p["c1"] * q + p["c2"] * q**2 + p["c3"] * q**3


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
    if n_pumps <= 0:
        raise ValueError("n_pumps must be >= 1")
    meta = _meta()
    sg = float(meta.get("specific_gravity", 1.0)) if sg is None else sg
    suction = float(meta.get("suction_psi", 220.0)) if suction_psi is None else suction_psi
    n_stages = int(meta.get("n_stages", 79))

    q_pp = total_flow_bpd / n_pumps
    dp = head_per_stage(q_pp) * n_stages * sg / _FT_TO_PSI_DIVISOR
    return suction + dp


def per_pump_flow(total_flow_bpd: float, n_pumps: int = 3) -> float:
    return total_flow_bpd / n_pumps


def flow_in_range(total_flow_bpd: float, n_pumps: int = 3) -> bool:
    """True if per-pump flow sits inside the recommended (thrust) window."""
    lo, hi = recommended_flow_per_pump()
    q_pp = per_pump_flow(total_flow_bpd, n_pumps)
    return lo <= q_pp <= hi


def station_capacity(n_pumps: int = 3) -> float:
    """Upper hydraulic flow ceiling (BPD) — recommended max per pump × n."""
    _lo, hi = recommended_flow_per_pump()
    return hi * n_pumps
