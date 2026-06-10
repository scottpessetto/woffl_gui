"""CFP produced-water plant hydraulics — pump curves + plant→pad delivery.

Source: ``woffl/jp_data/CFP Water Discharge Correlation.xlsx`` (Scott,
2026-06). The plant's three PW pumps (A/B/C machines) run in parallel at a
common discharge pressure; each follows a fitted quadratic

    DischP = a*flow^2 + b*flow + c        (flow in BWPD, P in psi)

so total plant throughput at discharge pressure P is the sum of the three
per-machine flows from inverting the quadratic on its descending limb.
Spreadsheet anchor points: 125,901 BWPD @ 2,200 psi; 101,428 BWPD @ 2,700.

The PW discharge is the power-fluid supply to the CFP-side pads: delivered
PF pressure at a pad = plant discharge − a rule-of-thumb line dP (the
spreadsheet's small table, referenced to a 2,697 psi plant snapshot). The
dP is treated as constant per pad — Scott's caveat: in reality it grows
with volume, so this is a first-order model. C-Pad has its own booster
pump, so its delivered PF is NOT tied to the plant curve.

This couples the CFP optimization to itself: more water through the plant
→ lower discharge pressure → lower delivered PF at B/G/J → less jet-pump
lift. The optimizer iterates allocation ↔ throughput ↔ PF pressure to a
fixed point.
"""

import math

# Fitted quadratic coefficients per machine: DischP = a*q² + b*q + c
MACHINE_COEFFS: dict[str, tuple[float, float, float]] = {
    "A": (-6.60950602540299e-07, -0.0154897118692163, 3971.4005378139),
    "B": (-1.00084923741945e-06, 0.0151043740922211, 3248.56932497937),
    "C": (-1.75394995854e-06, 0.0778673263080865, 2151.05205961576),
}

# Rule-of-thumb plant→pad line dP (psi), from the spreadsheet's table
# (referenced to a 2,697 psi plant discharge snapshot). Constant per pad —
# known simplification, real dP rises with volume. Only the CFP pads whose
# PF rides the plant discharge appear here: B, G, J. C-Pad is boosted
# on-pad; the spreadsheet's H entry was legacy (H predates its POPS
# install and no longer takes plant PF — Scott, 2026-06-10).
PAD_LINE_DP: dict[str, float] = {
    "B": 272.0,
    "G": 293.0,
    "J": 251.0,
}

# Operational ceilings (Scott, 2026-06-10): the CFP PW pumps can be run up
# to 2,900 psi discharge; POPS pads' on-pad charge pumps up to 3,500 psi.
MAX_DISCHARGE_PSI = 2900.0
POPS_MAX_PF_PSI = 3500.0

# Discharge-pressure window the fit is trusted over (the spreadsheet table
# spans ~2,200–2,700 psi; flows go unphysical well outside it).
PRESSURE_WINDOW = (1800.0, 3000.0)


def machine_flow(machine: str, disch_p: float) -> float:
    """Flow (BWPD) of one machine at a discharge pressure, descending limb.

    Inverts DischP = a*q² + b*q + c for q:
        q = (-b - sqrt(b² - 4a(c - P))) / (2a)
    Returns 0.0 when the pressure is above the machine's shutoff head
    (no real root) — the machine can't push against it.
    """
    a, b, c = MACHINE_COEFFS[machine]
    disc = b * b - 4.0 * a * (c - disch_p)
    if disc < 0.0:
        return 0.0
    q = (-b - math.sqrt(disc)) / (2.0 * a)
    return max(q, 0.0)


def plant_flow(disch_p: float) -> float:
    """Total plant throughput (BWPD) at a common discharge pressure (psi)."""
    return sum(machine_flow(m, disch_p) for m in MACHINE_COEFFS)


def plant_pressure(total_flow: float) -> float:
    """Common discharge pressure (psi) at a total plant throughput (BWPD).

    Inverts ``plant_flow`` by bisection over PRESSURE_WINDOW (plant_flow is
    monotone decreasing in pressure). Flows outside the window's range are
    clamped to its edge pressures.
    """
    lo, hi = PRESSURE_WINDOW
    if total_flow >= plant_flow(lo):
        return lo
    if total_flow <= plant_flow(hi):
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if plant_flow(mid) > total_flow:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def delivered_pf_pressure(pad: str, disch_p: float) -> float | None:
    """PF pressure delivered to a pad at a plant discharge pressure.

    Returns None for pads not supplied off the plant curve (e.g. C-Pad,
    which has its own booster and holds its PF pressure independently).
    """
    dp = PAD_LINE_DP.get(pad)
    if dp is None:
        return None
    return disch_p - dp
