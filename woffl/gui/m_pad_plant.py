"""M-Pad (Moose Pad, Mod 42) booster — HP-bank delivered-pressure frontier.

M-Pad is a HYBRID station (different from both S-Pad and I-Pad):

    produced water -> [3x P-4220 LP, PARALLEL] -> ~1,400 psig header
                         (also feeds Disposal + Kuparuk injection)
                      -> [3x P-4230 HP, PARALLEL] -> ~3,500 psig Power Fluid header

Parallel WITHIN each bank (flows add, share dP), SERIES BETWEEN banks (LP -> HP).
Two different REDA pumps (LP N1400N 13-stg, HP M675 41-stg), both VFD, field
SG ~1.03. The head/BHP polynomials are TOTAL-pump (not per-stage), cubic in BPD.

For power-fluid delivery to the jet pumps, the relevant unit is the **HP bank**
(3 parallel M675), fed at the LP-held ~1,400 psig. Two things make this unlike
I-Pad's amp-frontier:
  * It's PARALLEL — n online pumps each carry total/n; the bank shares one dP.
  * Amps have lots of headroom (~50-70% of FLA). The binding limits are
    MIN-FLOW (recirculation / high-diff shutdown — the pumps sit near it today)
    and VFD max speed, NOT amps.
And the installed pumps make ~10% less head than the as-new curve (solids wear),
so head is derated by ``field_head_factor`` (~0.91); amps stay on the as-new
curve (the wear is a head deficit at on-curve power).

v1 models the HP bank only, with a fixed LP-held suction. LP contention with the
disposal/Kuparuk headers (a bigger water balance) is a future add — ``hp_suction``
is a parameter so it slots in later. Physics validated to the psi against live
SCADA 2026-06-16. GUI-only (no upstream PR), like the other pad plants.
"""

import json
from functools import lru_cache
from glob import glob
from pathlib import Path
from typing import Optional

_PUMP_DIR = Path(__file__).resolve().parent.parent / "jp_data" / "M_Pad_Pumps"
_FT_TO_PSI_DIVISOR = 2.31
_HP_SUCTION_DEFAULT = 1400.0  # LP-held header (PIC-4221 setpoint) feeding the HP bank


@lru_cache(maxsize=1)
def _meta() -> dict:
    hits = sorted(glob(str(_PUMP_DIR / "*meta*.json")))
    if not hits:
        raise FileNotFoundError(f"no Moose Pad pump meta json under {_PUMP_DIR}")
    with open(hits[0], "r", encoding="utf-8") as fh:
        return json.load(fh)


def _poly(coeffs: dict, q: float) -> float:
    """sum(c[i]*q**i) for c0..cN. M-Pad polys are TOTAL-pump head/BHP, cubic."""
    total, qp, i = 0.0, 1.0, 0
    while f"c{i}" in coeffs:
        total += coeffs[f"c{i}"] * qp
        qp *= q
        i += 1
    return total


def specific_gravity() -> float:
    return float(_meta().get("compute", {}).get("specific_gravity_field", 1.03))


def wear_factor() -> float:
    """Field head derate (~0.91): installed pumps make less head than as-new."""
    return float(_meta().get("field_head_factor", {}).get("value", 0.91))


def hp_suction_psi() -> float:
    return _HP_SUCTION_DEFAULT


def _hp() -> dict:
    p = _meta()["pumps"]["B_HP_4230"]
    lo, hi = p["recommended_flow_bpd_60hz"]
    return {
        "name": "P-4230 HP",
        "head_poly": p["head_ft_poly_60hz"],
        "bhp_poly": p["bhp_poly_60hz"],
        "amp_limit": float(_meta()["motor_current_limits_A"]["trip"]),
        "k": float(p["amps_per_hp_est"]),
        "rec_lo": float(lo),   # per-pump min recommended (recirc floor) at 60 Hz
        "rec_hi": float(hi),   # per-pump max recommended (off-curve ceiling) at 60 Hz
        "hz_max": float(p["freq_range_hz"][1]),
        "n_default": 3,
    }


def _head_ft(pump: dict, q60: float) -> float:
    return _poly(pump["head_poly"], q60)


def _bhp(pump: dict, q60: float) -> float:
    return _poly(pump["bhp_poly"], q60)


def pump_boost(q_per_pump: float, hz: float, *, apply_wear: bool = True,
               sg: float | None = None) -> float:
    """HP-pump differential pressure (psi) at a per-pump flow + speed (affinity).
    Head is field-derated by the wear factor; pass apply_wear=False for as-new."""
    sg = specific_gravity() if sg is None else sg
    hp = _hp()
    q60 = q_per_pump * 60.0 / hz
    head = _head_ft(hp, q60) * (hz / 60.0) ** 2
    if apply_wear:
        head *= wear_factor()
    return head * sg / _FT_TO_PSI_DIVISOR


def pump_amps(q_per_pump: float, hz: float) -> float:
    """HP-pump motor amps at a per-pump flow + speed (as-new BHP curve, no wear)."""
    hp = _hp()
    q60 = q_per_pump * 60.0 / hz
    return hp["k"] * _bhp(hp, q60) * (hz / 60.0) ** 3


def hp_recommended_flow_per_pump() -> tuple[float, float]:
    hp = _hp()
    return hp["rec_lo"], hp["rec_hi"]


def min_total_flow(n_pumps: int = 3) -> float:
    """Recirculation floor: below this total PF the HP pumps drop under their
    min recommended per-pump flow (high-diff / low-flow shutdown)."""
    return _hp()["rec_lo"] * n_pumps


def max_total_flow(n_pumps: int = 3) -> float:
    """Off-curve ceiling: above this the per-pump flow exceeds the curve range."""
    return _hp()["rec_hi"] * n_pumps


def hz_for_boost(q_per_pump: float, target_boost: float, *, sg: float | None = None):
    """Speed (Hz, <= hz_max) at which the HP pump makes ``target_boost`` at the
    given per-pump flow, or None if it can't reach it even at max speed. Boost
    rises with speed, so we bisect."""
    hp = _hp()
    hz_max = hp["hz_max"]
    if pump_boost(q_per_pump, hz_max, sg=sg) < target_boost:
        return None
    lo, hi = 20.0, hz_max
    if pump_boost(q_per_pump, lo, sg=sg) > target_boost:
        return lo
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if pump_boost(q_per_pump, mid, sg=sg) > target_boost:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def max_discharge_pressure(total_flow_bpd: float, n_pumps: int = 3,
                           sg: float | None = None) -> Optional[float]:
    """Highest HP-discharge (power-fluid header) pressure the bank can deliver at
    the given total PF flow, all n pumps at max speed. The capability frontier;
    the page caps it at the operational limit (3,500). Falls with flow. None if
    the per-pump flow is past the curve's max (off-curve)."""
    if n_pumps <= 0 or total_flow_bpd <= 0:
        return None
    q_pp = total_flow_bpd / n_pumps
    if q_pp > _hp()["rec_hi"] * 1.05:  # a touch past recommended max = off curve
        return None
    return hp_suction_psi() + pump_boost(q_pp, _hp()["hz_max"], sg=sg)


def max_flow_at_pressure(pressure: float, n_pumps: int = 3,
                         sg: float | None = None) -> float:
    """Largest total PF the bank can deliver at >= ``pressure`` (frontier inverse),
    for the optimizer's PF budget at a candidate header pressure."""
    def ok(q: float) -> bool:
        f = max_discharge_pressure(q, n_pumps, sg)
        return f is not None and f >= pressure

    lo = max(min_total_flow(n_pumps), 100.0)
    if not ok(lo):
        return 0.0
    hi, cap = lo, max_total_flow(n_pumps)
    while hi < cap and ok(min(2.0 * hi, cap)):
        hi = min(2.0 * hi, cap)
        if hi >= cap:
            break
    hi = min(2.0 * hi, cap)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def operating_envelope(flows_bpd, n_pumps: int = 3, *, header_cap: float = 3500.0) -> list[dict]:
    """For a sweep of total flows, the deliverable header (capability capped at
    ``header_cap``) + the HP pumps' operating speed/amps/headroom and a recirc
    flag — for the results viz. At each flow the pumps run the speed that holds
    the capped header (or max speed if they can't reach it)."""
    hp = _hp()
    sg = specific_gravity()
    rows = []
    for q in flows_bpd:
        q_pp = q / n_pumps if n_pumps else 0.0
        cap_pressure = max_discharge_pressure(q, n_pumps, sg)
        if cap_pressure is None:
            rows.append({"flow": q, "max_discharge_psi": None, "feasible": False,
                         "recirc": q < min_total_flow(n_pumps), "pumps": []})
            continue
        header = min(header_cap, cap_pressure)
        target_boost = header - hp_suction_psi()
        hz = hz_for_boost(q_pp, target_boost, sg=sg) or hp["hz_max"]
        amps = pump_amps(q_pp, hz)
        rows.append({
            "flow": q, "max_discharge_psi": header, "per_pump_bpd": q_pp,
            "feasible": True, "recirc": q < min_total_flow(n_pumps),
            "speed_capped": cap_pressure < header_cap,
            "pumps": [{"name": hp["name"], "n": n_pumps, "hz": hz, "amps": amps,
                       "amp_limit": hp["amp_limit"]}],
        })
    return rows
