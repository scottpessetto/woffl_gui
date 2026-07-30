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
lift.

**How that coupling actually works (Scott, 2026-07-29).** The discharge
pressure is not a passive consequence of production — operators *set* it by
"opening or closing our disposal well", which moves the total flow through
the pumps and therefore where they ride on the curve. The pumps must pass
everything arriving, so the water arriving sets a MINIMUM flow, which sets a
MAXIMUM achievable discharge (capped by ``MAX_DISCHARGE_PSI``, the piping
rating — above it the pumps trip). Cutting controllable water at B/G/C/J
lowers the required flow and buys discharge pressure, hence PF, hence lift.
So the optimizer sweeps discharge as a decision variable
(``coupling="free_pressure"``, see ``woffl.gui.cfp_pad_plant.CFPPlant``)
rather than iterating a passive fixed point.

**Provisional fit — checked against the real per-machine SCADA tags
2026-07-30. Shape good, magnitude ~12% high.** All THREE machines run
(``MACHINE_FLOW_TAGS``): 120-day means A 29,844 · B 26,453 · C 29,931 =
86,228 BPD at 2,787 psi. Against that:

* observed / curve throughput = **0.90** at 2,877 psi and **0.87** at 2,639
  (per machine 0.84-0.96) — a fairly uniform over-prediction, so a single
  ~0.88 scale factor would largely correct it;
* measured slope of discharge on total machine flow is **-13.69 psi per 1,000
  BPD (r²=0.54)** against the curve's -17.5 — same family, modestly steep.

**Acceptance test for replacement coefficients: pass ~86,000 BPD at ~2,790
psi** (total of the three machines — NOT the ~112,300 BWPD of metered
produced water, only ~77% of which passes these pumps).

Earlier notes in this file claimed the plant ran two machines and that the
curve matched A+B to 1%. Both came from ``MPU_FIC_5488/5489``, which are a
different stream, and are void — see ``MACHINE_FLOW_TAGS``.
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

# Discharge-pressure window the fit is INVERTIBLE over. Deliberately wider
# than the band the data actually covers so `plant_pressure` has room to
# bisect; note its upper bound (3,000) sits ABOVE MAX_DISCHARGE_PSI, so this
# is a math bound, not an operational one. The 2,900 psi trip is enforced one
# layer up, in `woffl.gui.cfp_pad_plant.CFPPlant` (max_header_psi /
# clamp_window), which keeps this module's pinned values bit-identical.
PRESSURE_WINDOW = (1800.0, 3000.0)

# The band the spreadsheet table actually spans. Outside it the fit is
# extrapolation: `plant_flow` stays monotone (so the solver behaves) but the
# per-machine shapes are curve-fitting artifacts — machine C's fitted parabola
# even RISES to a 3,015 psi vertex at 22,198 BWPD, and its q=0 intercept
# (2,151 psi) sits below the whole operating window. Report, don't reject.
TRUSTED_BAND = (2200.0, 2700.0)

# Per-machine coefficients are a fit to TOTAL plant behavior over TRUSTED_BAND;
# no machine was validated on its own. Flip to True only once each machine's
# curve is confirmed independently — `CFPPlant` refuses machine SUBSETS while
# this is False, because a 1- or 2-machine curve is pure extrapolation of a fit
# that was never per-machine (see machine C above).
MACHINE_CURVE_VALIDATED = False

# Metered reality at the plant, from reporting.historian (120-day means to
# 2026-07-30). Anchor for measured-PF delivery and the acceptance target for
# replacement coefficients.
MEASURED_DISCHARGE_PSI = 2792.0
MEASURED_PRODUCED_WATER_BWPD = 112327.0
DISCHARGE_TAG = "MPU_PIC_5418"
PRODUCED_WATER_TAG = "MPU_MOD 54_ProdWaterAvgFlowRate_Calc"

# THE per-machine flow tags (BPD). Confirmed against Scott's SCADA screen
# 2026-07-30 (A 30,582 · B 23,589 · C 27,994 BPD at ~2,828 psi) and by the
# GPM twins MPU_FIC_5419/5420/5421 x 1440/42, which reproduce these to 0.1%.
#
# NOT MPU_FIC_5488/5489 — an earlier pass used those and they are a DIFFERENT
# stream (they match a separate column on the same screen reading
# 29,198/29,004/40,563). Any conclusion drawn from them is void.
MACHINE_FLOW_TAGS: dict[str, str] = {
    "A": "MPU_FIC_5419S",
    "B": "MPU_FIC_5420S",
    "C": "MPU_FIC_5421S",
}
MACHINE_FLOW_GPM_TAGS: dict[str, str] = {
    "A": "MPU_FIC_5419",
    "B": "MPU_FIC_5420",
    "C": "MPU_FIC_5421",
}
GPM_TO_BPD = 1440.0 / 42.0  # 34.2857

# 120-day means of the three machines: A 29,844 · B 26,453 · C 29,931.
# ALL THREE RUN — an earlier pass wrongly inferred a two-machine plant.
MEASURED_MACHINE_TOTAL_BPD = 86228.0

# Measured slope of discharge on TOTAL machine flow: -13.69 psi per 1,000 BPD
# over 79,371-94,399 BPD, r²=0.54 (120 days). The curve implies -17.5, so the
# shape is right and modestly steep.
MEASURED_PSI_PER_KBPD = 13.69

# The fit over-predicts throughput by a fairly UNIFORM ~12% (observed/curve =
# 0.90 at 2,877 psi, 0.87 at 2,639; per machine 0.84-0.96). A single scale
# factor would largely correct it — the shape is sound, the magnitude is high.
MEASURED_CURVE_SCALE = 0.88

# Only ~77% of metered produced water passes these machines, and machine flow
# tracks produced water only weakly (d/d = 0.23, r²=0.15) — consistent with
# operators setting the routing rather than production driving it.
MACHINE_SHARE_OF_PRODUCED_WATER = 0.77

ALL_MACHINES: tuple[str, ...] = ("A", "B", "C")


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


def resolve_machines(machines=None) -> tuple[str, ...]:
    """Normalize a machine selection to a validated tuple of machine keys.

    ``None`` (the default everywhere) means all three — every existing caller
    keeps its exact behavior. Raises on unknown keys or an empty selection.
    """
    if machines is None:
        return ALL_MACHINES
    picked = tuple(str(m).strip().upper() for m in machines)
    if not picked:
        raise ValueError("machines must name at least one machine")
    unknown = [m for m in picked if m not in MACHINE_COEFFS]
    if unknown:
        raise ValueError(
            f"unknown machine(s) {unknown}; expected any of {list(MACHINE_COEFFS)}"
        )
    return picked


def plant_flow(disch_p: float, machines=None) -> float:
    """Total throughput (BWPD) of the running machines at a common discharge.

    ``machines=None`` sums all three (the original behavior, bit-identical).
    Pass a subset (e.g. ``("A", "B")``) to model a machine being down — but
    see ``MACHINE_CURVE_VALIDATED``: the per-machine coefficients were fitted
    to TOTAL behavior, so a subset is extrapolation until validated.
    """
    return sum(machine_flow(m, disch_p) for m in resolve_machines(machines))


def plant_pressure(total_flow: float, machines=None) -> float:
    """Common discharge pressure (psi) at a total plant throughput (BWPD).

    Inverts ``plant_flow`` by bisection over PRESSURE_WINDOW (plant_flow is
    monotone decreasing in pressure). Flows outside the window's range are
    clamped to its edge pressures — use :func:`plant_pressure_detail` when the
    caller needs to know a clamp happened.
    """
    return plant_pressure_detail(total_flow, machines)[0]


def plant_pressure_detail(total_flow: float, machines=None) -> tuple[float, str]:
    """``plant_pressure`` plus WHY it returned that value.

    Returns ``(pressure, status)`` where status is:

    * ``"interior"``  — a genuine inversion of the curve.
    * ``"pinned_low"``  — demanded flow is at/above what the plant can pass even
      at the window floor, so the pressure is a CLAMP, not a physics result.
      This is the over-capacity case.
    * ``"pinned_high"`` — demanded flow is at/below the window-ceiling flow, so
      the pressure is clamped down to the ceiling.

    Exists because the plain function silently clamps, and with a soft capacity
    basis the solver can wander into either end. A clamp reported as physics is
    how a four-pad run produces plausible numbers that mean nothing.
    """
    picked = resolve_machines(machines)
    lo, hi = PRESSURE_WINDOW
    if total_flow >= plant_flow(lo, picked):
        return lo, "pinned_low"
    if total_flow <= plant_flow(hi, picked):
        return hi, "pinned_high"
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if plant_flow(mid, picked) > total_flow:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), "interior"


def in_trusted_band(disch_p: float) -> bool:
    """Is this discharge inside the band the spreadsheet table actually covers?"""
    lo, hi = TRUSTED_BAND
    return lo <= disch_p <= hi


def delivered_pf_pressure(pad: str, disch_p: float) -> float | None:
    """PF pressure delivered to a pad at a plant discharge pressure.

    Returns None for pads not supplied off the plant curve (e.g. C-Pad,
    which has its own booster and holds its PF pressure independently).
    """
    dp = PAD_LINE_DP.get(pad)
    if dp is None:
        return None
    return disch_p - dp
