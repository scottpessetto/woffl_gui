"""I-Pad booster train — amp-limited delivered-pressure frontier.

I-Pad is a **two-pump series** produced-water / power-fluid injection train, NOT
the parallel fixed-speed station S-Pad runs:

    PF separator (~217 psig) -> P-1021 LP (26 stg) -> P-0901 HP (17 stg) -> wells

Both units are VFD-driven and built from the same Summit SN35000 stage, so one
per-stage curve drives both (scaled x26 LP, x17 HP). Because they're on VFDs the
delivered pressure isn't a fixed-speed curve of flow — the drives modulate speed,
and the real ceiling is **motor amps** (LP pinned at its 192 A drive limit, HP at
154 A). At a fixed amp budget a pump trades head for flow, so the train's
capability is a falling frontier ``max_discharge_pressure(total_flow)``: the
optimizer may sit anywhere on/under it and intentionally drop pressure to move
more PF (Scott's call, 2026-06-16). There is no arbitrary flow cap — the ceiling
is wherever a pump can no longer pass the flow within its amp limit.

Physics (validated to the psi / amp against live SCADA 2026-06-16):
    Q60   = Q_actual * 60 / Hz                      (affinity: flow ~ Hz)
    head  = n_stages * head_per_stage(Q60) * (Hz/60)^2
    dP    = head * SG / 2.31
    BHP   = n_stages * bhp_per_stage(Q60) * (Hz/60)^3   (water)
    amps  = k * BHP                                  (k per pump, live-calibrated)
A pump's max dP at a given flow is the dP at the speed where amps hit its limit
(capped at 60 Hz, floored where Q60 reaches the curve's max valid flow); the
train frontier is suction + dP_LP_max + dP_HP_max.

GUI-only (MPU data + optimizer coupling), no upstream library PR — like
``s_pad_plant``. Loads ``woffl/jp_data/I_Pad_Pumps/*meta*.json``.
"""

import json
from functools import lru_cache
from glob import glob
from pathlib import Path
from typing import Optional

_PUMP_DIR = Path(__file__).resolve().parent.parent / "jp_data" / "I_Pad_Pumps"
_FT_TO_PSI_DIVISOR = 2.31
_HZ_MAX = 60.0  # curve is normalized to 60 Hz; drives run at/below it
_HZ_FLOOR = 10.0  # never search below this regardless of flow


@lru_cache(maxsize=1)
def _meta() -> dict:
    hits = sorted(glob(str(_PUMP_DIR / "*meta*.json")))
    if not hits:
        raise FileNotFoundError(f"no I-Pad pump meta json under {_PUMP_DIR}")
    with open(hits[0], "r", encoding="utf-8") as fh:
        return json.load(fh)


def _poly(coeffs: dict, q: float) -> float:
    """Evaluate sum(c[i]*q**i) for c0..cN (arbitrary order)."""
    total, qp = 0.0, 1.0
    i = 0
    while f"c{i}" in coeffs:
        total += coeffs[f"c{i}"] * qp
        qp *= q
        i += 1
    return total


def specific_gravity() -> float:
    return float(_meta().get("compute", {}).get("specific_gravity_used_in_tables", 1.04))


def suction_psi() -> float:
    """Train intake (LP suction) — the PF separator pressure."""
    lp = _meta()["pumps"]["P-1021_LP_Booster"]
    return float(lp["live_operating_point_2026-06-16"]["intake_psig"])


def _max_valid_flow() -> float:
    return float(_meta()["stage"].get("max_valid_flow_bpd", 60000.0))


def head_per_stage(q60_bpd: float) -> float:
    """Head (ft) per stage at 60 Hz for the given (60-Hz-equivalent) flow."""
    return _poly(_meta()["stage"]["head_per_stage_poly"], q60_bpd)


def bhp_per_stage(q60_bpd: float) -> float:
    """Water BHP per stage at 60 Hz for the given (60-Hz-equivalent) flow."""
    return _poly(_meta()["stage"]["bhp_per_stage_poly"], q60_bpd)


def _pumps() -> list[dict]:
    """[LP, HP] in series order, each {name, n_stages, amp_limit, k}."""
    m = _meta()["pumps"]
    lp = m["P-1021_LP_Booster"]
    hp = m["P-0901_HP_Booster"]
    return [
        {
            "name": "P-1021 LP",
            "n_stages": int(lp["n_stages"]),
            "amp_limit": float(lp["motor_current_limits_A"]["vfd_drive_limit"]),
            "k": float(lp["amps_per_bhp_est"]),
        },
        {
            "name": "P-0901 HP",
            "n_stages": int(hp["n_stages"]),
            "amp_limit": float(hp["motor_current_limits_A"]["fla_and_trip"]),
            "k": float(hp["amps_per_bhp_est"]),
        },
    ]


def pump_dP(n_stages: int, flow_bpd: float, hz: float, sg: float | None = None) -> float:
    """Differential pressure (psi) a pump makes at a flow + speed (affinity)."""
    sg = specific_gravity() if sg is None else sg
    q60 = flow_bpd * 60.0 / hz
    head = n_stages * head_per_stage(q60) * (hz / 60.0) ** 2
    return head * sg / _FT_TO_PSI_DIVISOR


def pump_amps(k: float, n_stages: int, flow_bpd: float, hz: float) -> float:
    """Motor amps a pump draws at a flow + speed (affinity, live-calibrated k)."""
    q60 = flow_bpd * 60.0 / hz
    return k * n_stages * bhp_per_stage(q60) * (hz / 60.0) ** 3


def _hz_at_amp_limit(pump: dict, flow_bpd: float) -> Optional[float]:
    """Speed (Hz, <= 60) at which this pump hits its amp limit at the given flow,
    or None if the pump physically can't pass that flow within its amp limit.

    The speed range is floored where the 60-Hz-equivalent flow (Q60 = Q*60/Hz)
    reaches the pump's max curve flow — below that the poly extrapolates into
    nonsense and the pump isn't on its curve. Amps rise monotonically with speed
    across the valid range, so we bisect for amps == limit.
    """
    k, n, lim = pump["k"], pump["n_stages"], pump["amp_limit"]
    hz_min = max(_HZ_FLOOR, flow_bpd * 60.0 / _max_valid_flow())
    if hz_min >= _HZ_MAX:
        return None  # flow exceeds the curve's max valid flow even at 60 Hz
    if pump_amps(k, n, flow_bpd, _HZ_MAX) <= lim:
        return _HZ_MAX  # not amp-limited here — capped by max speed
    if pump_amps(k, n, flow_bpd, hz_min) > lim:
        return None  # over the limit even at minimum head — flow too high to pass
    lo, hi = hz_min, _HZ_MAX
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if pump_amps(k, n, flow_bpd, mid) > lim:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def pump_max_dP(pump: dict, flow_bpd: float, sg: float | None = None) -> Optional[float]:
    """Max dP (psi) this pump can deliver at the given flow within its amp limit,
    or None if the flow isn't deliverable (over amps even at minimum head)."""
    hz = _hz_at_amp_limit(pump, flow_bpd)
    if hz is None:
        return None
    return pump_dP(pump["n_stages"], flow_bpd, hz, sg)


def max_discharge_pressure(total_flow_bpd: float, sg: float | None = None) -> Optional[float]:
    """Highest header (HP-discharge) pressure the series train can deliver at the
    given total PF flow, both pumps at their amp limits. The capability frontier;
    the optimizer rides on/under it (lower pressure -> more deliverable flow).
    Returns None when the flow exceeds what the train can pass within amps (the
    emergent capacity ceiling — there's no arbitrary hard cap)."""
    sg = specific_gravity() if sg is None else sg
    dps = [pump_max_dP(p, total_flow_bpd, sg) for p in _pumps()]
    if any(d is None for d in dps):
        return None
    return suction_psi() + sum(dps)


def max_flow_at_pressure(pressure: float, sg: float | None = None) -> float:
    """Largest total PF flow the train can deliver at >= ``pressure`` (the frontier
    inverted). The optimizer uses this as the PF budget at a candidate header
    pressure: pick a pressure, this is how much flow the pumps can push at it
    within amps. Returns 0.0 if even minimal flow can't reach the pressure.
    """
    def ok(q: float) -> bool:
        f = max_discharge_pressure(q, sg)
        return f is not None and f >= pressure

    lo = 100.0
    if not ok(lo):
        return 0.0  # pumps can't make this pressure even at low flow
    hi, cap = 200.0, 4.0 * _max_valid_flow()
    while hi < cap and ok(2.0 * hi):
        hi *= 2.0
    hi = min(2.0 * hi, cap)  # ok(lo) True, ok(hi) False (frontier falls with flow)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def operating_envelope(flows_bpd) -> list[dict]:
    """For a sweep of total flows, the frontier pressure + each pump's limiting
    speed/amps — for the results-screen viz and headroom readout. ``feasible``
    is False past the amp-limited flow ceiling."""
    sg = specific_gravity()
    rows = []
    for q in flows_bpd:
        pumps, feasible = [], True
        for p in _pumps():
            hz = _hz_at_amp_limit(p, q)
            if hz is None:
                feasible = False
                pumps.append({"name": p["name"], "hz": None, "dP": None,
                              "amps": None, "amp_limit": p["amp_limit"]})
                continue
            pumps.append({
                "name": p["name"], "hz": hz,
                "dP": pump_dP(p["n_stages"], q, hz, sg),
                "amps": pump_amps(p["k"], p["n_stages"], q, hz),
                "amp_limit": p["amp_limit"],
            })
        rows.append({
            "flow": q,
            "max_discharge_psi": (suction_psi() + sum(p["dP"] for p in pumps))
                                 if feasible else None,
            "pumps": pumps,
            "feasible": feasible,
            "amp_limited": feasible and any(p["hz"] < _HZ_MAX - 0.01 for p in pumps),
        })
    return rows
