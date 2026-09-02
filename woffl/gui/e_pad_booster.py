"""E-Pad power-fluid booster candidate comparison at a required differential.

The screen this feeds answers one question: **at a required dP across the
booster, how much water can each candidate build move into the 3,400 psig
E-Pad power-fluid header, inside its recommended operating range, and what
amps does that pull?**

Two candidates, both Summit ESP mixed-flow stages on a VFD, both digitized
from the catalog performance pages that came with the workbooks
(``woffl/jp_data/E_Pad_Pumps/``):

* ``SM25000_26STG`` - 875 series, 26 stages. The build in the well.
* ``SN35000_18STG`` - 950 series, 18 stages. The alternative.

The model is the affinity-law sheet the two workbooks implement, evaluated
the other way round. The workbooks fix a speed and read off dP; here the dP
is the input and the SPEED is solved, because that is the decision a VFD
actually makes::

    Q60   = Q * 60 / Hz                              (index the 60 Hz table)
    head  = condition * n_stages * head_stg(Q60) * (Hz/60)^2
    dP    = head * SG / 2.31
    BHP   = SG * n_stages * bhp_stg(Q60) * (Hz/60)^3
    amps  = amps_per_bhp * BHP
    ROR   = [xrc_lo, xrc_hi] * Hz/60           (the range moves with speed)

At a fixed dP the required speed rises with flow, so a candidate's answer is
a *flow window*: below it the pump is under its recommended range (running
too slow to hold the dP at that little flow), above it the recommended range,
the amp limit, or the 60 Hz capability wall cuts in. ``solve_candidate``
returns that window, the duty point at its top, and the whole constant-dP
locus for plotting.

Not enforced, deliberately - see ``housing_pressure_caveat`` /
``not_enforced`` in the meta json: the catalog HOUSING pressure limit (the
SN35000's 2,800 psi is a downhole-housing number and the same stage runs at
3,408 psig discharge on I-Pad today), the shaft HP limit (neither build comes
near it), and NPSH (not published on the supplied pages).

Fork-only, MPU-specific pump data - the same kind of module as
``pad_plant_base``, whose affinity/efficiency statics it reuses rather than
re-deriving. Not a ``PadPlant``: E-Pad has no optimizer run, and stubbing six
optimizer hooks nobody calls would be worse than not claiming the interface.
No upstream PR.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

from woffl.gui.pad_plant_base import PadPlant

_META_PATH = (
    Path(__file__).resolve().parent.parent
    / "jp_data"
    / "E_Pad_Pumps"
    / "E-Pad_booster_pump_meta.json"
)

# Samples in every reported curve / locus, both ends inclusive - the same
# contract the pad curve_report payloads carry.
CURVE_POINTS = 61

# Edge-finding grid. Coarser than a bisection over the whole range and finer
# than the reported curve, so a feasible window is always bracketed before it
# is refined; 4x the reported resolution has never missed one of these bands.
_SCAN_POINTS = 241

# Iso-speed lines the capability sheet draws (filtered to the speed cap).
_CURVE_SPEEDS_HZ = (45.0, 50.0, 55.0, 60.0)

# Speeds the fixed-speed table lists (filtered to the speed cap). Wider and
# coarser than the drawn family: this table is read as an operating ladder,
# so it goes down to where a low-dP duty actually lands.
_SPEED_TABLE_HZ = (35.0, 40.0, 45.0, 50.0, 55.0, 60.0)

# Numerical speed floor, NOT an operating limit: below it the 60-Hz-equivalent
# flow walks off the digitized table. The recommended operating range binds
# long before this at any useful flow.
_HZ_FLOOR = 20.0

_BISECT_STEPS = 60

# Relative slack on a recommended-range edge test, for float boundaries only.
_ROR_EDGE_TOL = 1e-9

# BPD x psi -> hydraulic HP: 3960 * (1440/42) / 2.31. Used only to price the
# pressure burned across a throttling choke.
_BPD_PSI_TO_HP = 3960.0 * (1440.0 / 42.0) / 2.31

# Blocking reasons, in the order they are tested.
BLOCK_NO_SPEED = "no speed makes the dP"
BLOCK_ROR_LOW = "below recommended range"
BLOCK_ROR_HIGH = "above recommended range"
BLOCK_AMPS = "over amp limit"
BLOCK_DP_UNREACHABLE = "this speed cannot make the dP"

# What caps the top of a feasible window.
LIMIT_ROR_HIGH = "Recommended range (high)"
LIMIT_AMPS = "Amp limit"
LIMIT_CAPABILITY = "Capability at max speed"
LIMIT_OVER_DELIVERS = "Over-delivers dP past here"


def _load_meta() -> dict:
    with open(_META_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


_META_CACHE: Optional[dict] = None


def meta() -> dict:
    """The E-Pad booster meta json (loaded once)."""
    global _META_CACHE
    if _META_CACHE is None:
        _META_CACHE = _load_meta()
    return _META_CACHE


def defaults() -> dict:
    """Screen defaults: sg, suction_psi, target_discharge_psi, condition,
    hz_min/hz_max, amps_per_bhp, and the notes explaining each."""
    return dict(meta()["defaults"])


def _interp(table: list[list[float]], q: float, col: int) -> float:
    """Linear interpolation of ``col`` against column 0 of an ascending table.

    Clamped at both ends. Clamping is a safety net only: every caller floors
    the speed so the 60-Hz-equivalent flow stays inside the table (past the
    last row a digitized head curve would run negative).
    """
    if q <= table[0][0]:
        return float(table[0][col])
    if q >= table[-1][0]:
        return float(table[-1][col])
    lo = 0
    hi = len(table) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if table[mid][0] <= q:
            lo = mid
        else:
            hi = mid
    q0, q1 = table[lo][0], table[hi][0]
    v0, v1 = table[lo][col], table[hi][col]
    return float(v0 + (v1 - v0) * (q - q0) / (q1 - q0))


class EPadBooster:
    """One candidate build: a Summit stage type at a stage count, on a VFD.

    Args:
        key (str): meta ``pumps`` key, e.g. ``"SM25000_26STG"``.
        spec (dict): that pump's meta block.
    """

    def __init__(self, key: str, spec: dict) -> None:
        self.key = key
        self.spec = spec
        self.n_stages = int(spec["n_stages"])
        self.table: list[list[float]] = [
            [float(c) for c in row] for row in spec["stage_table"]
        ]
        self.max_valid_flow = float(spec["max_valid_flow_bpd_60hz"])
        self.bep = float(spec["bep_flow_bpd_60hz"])
        lo, hi = spec["xrc_operating_range_bpd_60hz"]
        self.ror_60hz = (float(lo), float(hi))
        self.installed = bool(spec["installed"])
        self.label = str(spec["label"])
        # Flow (60 Hz) where the digitized head curve peaks. Mixed-flow stages
        # climb off shut-off, so the head curve is only falling to the RIGHT of
        # this; every "where does this speed cross that dP" bisection has to
        # start here or it can land on the rising branch and return a flow the
        # pump would never be operated at.
        peak = max(self.table, key=lambda row: row[1])
        self.q_head_peak = float(
            max(row[0] for row in self.table if row[1] == peak[1])
        )

    # -- stage curve ---------------------------------------------------------

    def head_per_stage(self, q60_bpd: float) -> float:
        """Head (ft) per stage at 60 Hz for a 60-Hz-equivalent flow."""
        return _interp(self.table, q60_bpd, 1)

    def bhp_per_stage(self, q60_bpd: float) -> float:
        """Water BHP (SG 1.0) per stage at 60 Hz for a 60-Hz-equivalent flow."""
        return _interp(self.table, q60_bpd, 2)

    # -- performance at a flow + speed --------------------------------------

    def head_ft(self, flow_bpd: float, hz: float, condition: float = 1.0) -> float:
        """Total pump head (ft) at a flow + speed, wear-derated by ``condition``."""
        q60 = PadPlant._affinity_q60(flow_bpd, hz)
        head_60 = self.n_stages * self.head_per_stage(q60)
        return condition * PadPlant._affinity_head(head_60, hz)

    def dp_psi(
        self, flow_bpd: float, hz: float, sg: float, condition: float = 1.0
    ) -> float:
        """Differential pressure (psid) at a flow + speed."""
        return PadPlant._head_ft_to_psi(self.head_ft(flow_bpd, hz, condition), sg)

    def bhp(self, flow_bpd: float, hz: float, sg: float) -> float:
        """Shaft power (BHP) at a flow + speed - the motor's load, and the
        workbooks' "HP load" column. Wear does not lighten the shaft, so
        ``condition`` deliberately does not enter."""
        q60 = PadPlant._affinity_q60(flow_bpd, hz)
        bhp_60 = self.n_stages * self.bhp_per_stage(q60)
        return sg * PadPlant._affinity_bhp(bhp_60, hz)

    def amps(
        self, flow_bpd: float, hz: float, sg: float, amps_per_bhp: float
    ) -> float:
        """Motor amps at a flow + speed, ``amps = k * BHP`` (the convention the
        I-Pad and M-Pad plant models use). ``k`` is a transferred estimate for
        E-Pad - see ``defaults()["amps_per_bhp_note"]``."""
        return amps_per_bhp * self.bhp(flow_bpd, hz, sg)

    def ror(self, hz: float) -> tuple[float, float]:
        """Recommended operating range (BPD) at a speed - the catalog XRC range
        scaled by Hz/60, because flow scales with speed."""
        lo, hi = self.ror_60hz
        f = hz / 60.0
        return lo * f, hi * f

    def hz_window_in_ror(
        self, flow_bpd: float, hz_max: float
    ) -> Optional[tuple[float, float]]:
        """Speeds (Hz) at which ``flow_bpd`` sits INSIDE the recommended range.

        The range scales with speed, so a fixed flow is in range only over a
        speed band: run too slow and the flow is off the right end of the
        curve, too fast and it is off the left end (recirculation). None when
        the band is empty - that flow cannot be passed in range at any speed up
        to ``hz_max``.
        """
        lo60, hi60 = self.ror_60hz
        # q <= ror_hi(hz) -> hz >= q*60/hi60 ; q >= ror_lo(hz) -> hz <= q*60/lo60
        hz_lo = max(flow_bpd * 60.0 / hi60, self._hz_floor(flow_bpd, hz_max))
        hz_hi = min(hz_max, flow_bpd * 60.0 / lo60 if lo60 > 0 else hz_max)
        return (hz_lo, hz_hi) if hz_hi >= hz_lo else None

    def flow_at_dp_and_speed(
        self, hz: float, dp_target: float, sg: float, condition: float
    ) -> Optional[float]:
        """Flow (BPD) where the iso-speed curve at ``hz`` crosses ``dp_target``
        - what the pump actually passes if you pin the drive at that speed and
        the system holds that differential.

        None when this speed cannot make the dP at all (its whole curve sits
        below it). Bisected on the falling branch only, from the head peak out
        to where the digitized table ends at this speed.
        """
        q_lo = self.q_head_peak * hz / 60.0
        q_hi = self.max_valid_flow * hz / 60.0
        if self.dp_psi(q_lo, hz, sg, condition) < dp_target:
            return None
        if self.dp_psi(q_hi, hz, sg, condition) >= dp_target:
            return q_hi  # still over the dP where the curve data runs out
        for _ in range(_BISECT_STEPS):
            mid = 0.5 * (q_lo + q_hi)
            if self.dp_psi(mid, hz, sg, condition) >= dp_target:
                q_lo = mid
            else:
                q_hi = mid
        return 0.5 * (q_lo + q_hi)

    def max_dp_at_flow(
        self,
        flow_bpd: float,
        sg: float,
        condition: float,
        hz_max: float,
        amps_per_bhp: float,
        amp_limit: Optional[float],
    ) -> Optional[float]:
        """Highest dP (psid) this build can make at ``flow_bpd`` while the flow
        stays inside the recommended range and the motor stays under its cap.

        This is the plant capability frontier read flow-first, and it is NOT
        monotone in flow: below ``ror_lo * hz_max/60`` the range floor forces a
        lower speed, so deliverable pressure collapses toward shut-in. None
        when the flow cannot be passed in range, or not within amps.
        """
        window = self.hz_window_in_ror(flow_bpd, hz_max)
        if window is None:
            return None
        hz_lo, hz_hi = window
        if amp_limit is not None:
            # amps rise with speed at fixed flow (BHP ~ (Hz/60)^3)
            if self.amps(flow_bpd, hz_lo, sg, amps_per_bhp) > amp_limit:
                return None
            if self.amps(flow_bpd, hz_hi, sg, amps_per_bhp) > amp_limit:
                lo, hi = hz_lo, hz_hi
                for _ in range(_BISECT_STEPS):
                    mid = 0.5 * (lo + hi)
                    if self.amps(flow_bpd, mid, sg, amps_per_bhp) > amp_limit:
                        hi = mid
                    else:
                        lo = mid
                hz_hi = lo
        return self.dp_psi(flow_bpd, hz_hi, sg, condition)

    def throttled_duty(
        self,
        dp_target: float,
        sg: float,
        condition: float,
        hz_max: float,
        amps_per_bhp: float,
        amp_limit: Optional[float],
        suction_psi: float,
    ) -> Optional[dict]:
        """The other operating policy: run flat out and choke off the surplus.

        Holding the required dP EXACTLY means slowing the drive until the pump
        makes only that much head, and the recommended range shrinks with the
        speed - which is what caps the deliverable rate. The alternative is to
        run at ``hz_max``, pass the most flow the range allows there, and burn
        the extra pressure across a choke. That moves more water for more
        shaft power and a throttling loss, and it is the trade an engineer
        deciding "do I run the pump slower?" is actually making.

        Returns:
            dict | None: ``{q_bpd, hz, dp_made_psid, discharge_psi,
                throttle_psid, throttle_hhp, bhp, amps, amp_headroom_a,
                eff_pct, in_ror}``, or None when the speed cap cannot make the
                required dP at any in-range flow (nothing to throttle).
        """
        q = self.ror_60hz[1] * hz_max / 60.0
        if amp_limit is not None:
            if self.amps(0.0, hz_max, sg, amps_per_bhp) > amp_limit:
                return None  # over the cap even at shut-off
            if self.amps(q, hz_max, sg, amps_per_bhp) > amp_limit:
                # BHP rises with flow at fixed speed, so bisect the flow down
                lo, hi = 0.0, q
                for _ in range(_BISECT_STEPS):
                    mid = 0.5 * (lo + hi)
                    if self.amps(mid, hz_max, sg, amps_per_bhp) > amp_limit:
                        hi = mid
                    else:
                        lo = mid
                q = lo
        ror_lo, ror_hi = self.ror(hz_max)
        if q < ror_lo:
            return None
        dp_made = self.dp_psi(q, hz_max, sg, condition)
        if dp_made < dp_target:
            return None  # flat out it still cannot reach the duty at this flow
        bhp = self.bhp(q, hz_max, sg)
        amps = amps_per_bhp * bhp
        throttle = dp_made - dp_target
        return {
            "q_bpd": q,
            "hz": hz_max,
            "dp_made_psid": dp_made,
            "discharge_psi": suction_psi + dp_made,
            "throttle_psid": throttle,
            # Hydraulic power burned across the choke: BPD x psi / 58,766.
            "throttle_hhp": q * throttle / (_BPD_PSI_TO_HP),
            "bhp": bhp,
            "amps": amps,
            "amp_headroom_a": None if amp_limit is None else amp_limit - amps,
            "eff_pct": PadPlant.hydraulic_efficiency(
                q, self.head_ft(q, hz_max, condition), bhp, sg
            ),
            "in_ror": ror_lo <= q <= ror_hi * (1.0 + _ROR_EDGE_TOL),
        }

    # -- inverse: the speed that holds a required dP ------------------------

    def _hz_floor(self, flow_bpd: float, hz_max: float) -> float:
        """Lowest speed whose 60-Hz-equivalent flow still sits on the table."""
        return max(_HZ_FLOOR, flow_bpd * 60.0 / self.max_valid_flow)

    def hz_for_dp(
        self,
        flow_bpd: float,
        dp_target: float,
        sg: float,
        condition: float,
        hz_max: float,
    ) -> Optional[float]:
        """Speed (Hz, <= ``hz_max``) at which this build makes ``dp_target`` at
        the given flow, or None when it cannot.

        dP rises with speed at fixed flow: the ``(Hz/60)^2`` factor swamps the
        1.3 pct head rise the mixed-flow curve shows off shut-off, so the
        bisection below is safe on both stages. None means either the flow is
        past what the pump can pass at ``hz_max``, or ``hz_max`` cannot reach
        the dP, or the dP is so low it would need a speed under the numerical
        floor (a real pump would be throttled there, and pretending otherwise
        would report a duty point the model never solved).
        """
        hz_lo = self._hz_floor(flow_bpd, hz_max)
        if hz_lo >= hz_max:
            return None  # flow is off the table even at max speed
        if self.dp_psi(flow_bpd, hz_max, sg, condition) < dp_target:
            return None  # cannot reach the dP at max speed
        if self.dp_psi(flow_bpd, hz_lo, sg, condition) > dp_target:
            return None  # would need to run below the modeled speed floor
        lo, hi = hz_lo, hz_max
        for _ in range(_BISECT_STEPS):
            mid = 0.5 * (lo + hi)
            if self.dp_psi(flow_bpd, mid, sg, condition) < dp_target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def max_flow_for_dp(
        self, dp_target: float, sg: float, condition: float, hz_max: float
    ) -> float:
        """Largest flow (BPD) at which the build can HOLD ``dp_target`` - the
        wall the constant-dP locus ends at, ignoring range and amps.

        Two walls, whichever comes first. Above one the pump cannot REACH the
        dP even at ``hz_max``. Above the other it would OVER-DELIVER: the
        lowest speed that keeps the flow on the digitized table already makes
        more than the required dP, so holding exactly that dP would mean
        running off the curve. On the SM25000 at the 600 psid header duty the
        second wall binds first (38.3 k BPD, not the table's 41 k).

        Scanned then bisected rather than assumed monotone: the mixed-flow head
        rise off shut-off can leave a high dP unreachable at zero flow and
        reachable a few thousand BPD up, which a plain "is it reachable at
        q=0" pretest would call infeasible.

        Returns:
            float: that flow, or 0.0 when no flow can hold the dP.
        """
        q_top = self.max_valid_flow * hz_max / 60.0

        def solvable(q: float) -> bool:
            return self.hz_for_dp(q, dp_target, sg, condition, hz_max) is not None

        grid = PadPlant._curve_grid(q_top, _SCAN_POINTS)
        hits = [i for i, q in enumerate(grid) if solvable(q)]
        if not hits:
            return 0.0
        top = hits[-1]
        if top == len(grid) - 1:
            return grid[top]
        lo, hi = grid[top], grid[top + 1]
        for _ in range(_BISECT_STEPS):
            mid = 0.5 * (lo + hi)
            if solvable(mid):
                lo = mid
            else:
                hi = mid
        return lo

    def wall_kind(
        self,
        q_wall: float,
        dp_target: float,
        sg: float,
        condition: float,
        hz_max: float,
    ) -> str:
        """Which of the two walls ``q_wall`` is: out of speed, or over-delivering.

        Reads the speed the dP needs AT the wall. At ``hz_max`` the pump has
        run out of head; anywhere below it, the wall is the table's validity
        floor and the pump would over-deliver past it.
        """
        hz = self.hz_for_dp(q_wall, dp_target, sg, condition, hz_max)
        if hz is None or hz >= hz_max - 1e-6:
            return LIMIT_CAPABILITY
        return LIMIT_OVER_DELIVERS

    # -- one point on the constant-dP locus ---------------------------------

    def point_at_dp(
        self,
        flow_bpd: float,
        dp_target: float,
        sg: float,
        condition: float,
        hz_max: float,
        amps_per_bhp: float,
        amp_limit: Optional[float],
        suction_psi: float,
    ) -> dict:
        """One flow on the constant-dP locus: the speed that holds the dP there
        and everything that follows from it.

        Returns:
            dict: ``{q_bpd, hz, dp_psid, discharge_psi, head_ft, bhp, amps,
                amp_headroom_a, eff_pct, pct_of_bep, ror_lo, ror_hi, in_ror,
                amp_ok, ok, blocked_by}``. ``hz`` and everything downstream of
                it are None when no speed makes the dP at this flow.
        """
        hz = self.hz_for_dp(flow_bpd, dp_target, sg, condition, hz_max)
        if hz is None:
            return {
                "q_bpd": flow_bpd,
                "hz": None,
                "dp_psid": None,
                "discharge_psi": None,
                "head_ft": None,
                "bhp": None,
                "amps": None,
                "amp_headroom_a": None,
                "eff_pct": None,
                "pct_of_bep": None,
                "ror_lo": None,
                "ror_hi": None,
                "in_ror": False,
                "amp_ok": False,
                "ok": False,
                "blocked_by": BLOCK_NO_SPEED,
            }

        head = self.head_ft(flow_bpd, hz, condition)
        bhp = self.bhp(flow_bpd, hz, sg)
        amps = amps_per_bhp * bhp
        ror_lo, ror_hi = self.ror(hz)
        in_ror = ror_lo <= flow_bpd <= ror_hi
        amp_ok = amp_limit is None or amps <= amp_limit

        if not in_ror:
            blocked = BLOCK_ROR_LOW if flow_bpd < ror_lo else BLOCK_ROR_HIGH
        elif not amp_ok:
            blocked = BLOCK_AMPS
        else:
            blocked = None

        return {
            "q_bpd": flow_bpd,
            "hz": hz,
            "dp_psid": dp_target,
            "discharge_psi": suction_psi + dp_target,
            "head_ft": head,
            "bhp": bhp,
            "amps": amps,
            "amp_headroom_a": None if amp_limit is None else amp_limit - amps,
            "eff_pct": PadPlant.hydraulic_efficiency(flow_bpd, head, bhp, sg),
            "pct_of_bep": (
                100.0 * flow_bpd / (self.bep * hz / 60.0) if self.bep > 0 else None
            ),
            "ror_lo": ror_lo,
            "ror_hi": ror_hi,
            "in_ror": in_ror,
            "amp_ok": amp_ok,
            "ok": blocked is None,
            "blocked_by": blocked,
        }

    def speed_table(
        self,
        dp_target: float,
        sg: float,
        condition: float,
        hz_max: float,
        amps_per_bhp: float,
        amp_limit: Optional[float],
        suction_psi: float,
        duty_hz: Optional[float] = None,
    ) -> list[dict]:
        """Pin the drive at a speed: what flow comes out at the required dP?

        The other views answer "what is the most water I can move at this dP"
        and leave the speed to the solve. This one answers the question an
        operator actually asks - *I am going to run it at 55 Hz, what do I
        get?* - and it is the view that shows WHY the deliverable rate caps
        where it does: at each speed the crossing flow is compared with the
        recommended range AT THAT SPEED, and past the duty speed the crossing
        runs off the right end of the range faster than the range grows.

        Args:
            dp_target (float): required differential (psid).
            sg (float): pumped-fluid specific gravity.
            condition (float): head-only wear derate.
            hz_max (float): speed cap; speeds above it are not listed.
            amps_per_bhp (float): amps per shaft BHP.
            amp_limit (float | None): motor amp cap, or None.
            suction_psi (float): booster suction (psig).
            duty_hz (float | None): the solved duty speed, listed as its own
                row and flagged, so the table explains the headline number.

        Returns:
            list[dict]: one row per speed, ascending -
                ``{hz, is_duty, q_bpd, discharge_psi, ror_lo, ror_hi,
                pct_of_ror_hi, in_ror, bhp, amps, amp_ok, eff_pct,
                blocked_by}``. ``q_bpd`` and everything downstream are None
                where that speed's whole curve sits below the required dP.
        """
        speeds = sorted(
            {float(h) for h in _SPEED_TABLE_HZ if float(h) <= hz_max} | {hz_max}
        )
        if duty_hz is not None and not any(abs(h - duty_hz) < 0.05 for h in speeds):
            # The EXACT solved speed, never a rounded copy: rounding it moves
            # the crossing flow a hair past the range ceiling and the duty row
            # then reports itself out of range.
            speeds = sorted(speeds + [duty_hz])
        rows = []
        for hz in speeds:
            is_duty = duty_hz is not None and abs(hz - duty_hz) < 1e-9
            q = self.flow_at_dp_and_speed(hz, dp_target, sg, condition)
            ror_lo, ror_hi = self.ror(hz)
            if q is None:
                rows.append(
                    {
                        "hz": hz,
                        "is_duty": is_duty,
                        "q_bpd": None,
                        "discharge_psi": None,
                        "ror_lo": ror_lo,
                        "ror_hi": ror_hi,
                        "pct_of_ror_hi": None,
                        "in_ror": False,
                        "bhp": None,
                        "amps": None,
                        "amp_ok": False,
                        "eff_pct": None,
                        "blocked_by": BLOCK_DP_UNREACHABLE,
                    }
                )
                continue
            head = self.head_ft(q, hz, condition)
            bhp = self.bhp(q, hz, sg)
            amps = amps_per_bhp * bhp
            # Relative tolerance, not a fudge: the duty flow IS the range
            # ceiling by construction, and an independent bisection lands a
            # few ULPs the wrong side of it. 1e-9 of 22,869 BPD is 0.00002 BPD
            # - it cannot mask an operationally real excursion.
            in_ror = (
                ror_lo * (1.0 - _ROR_EDGE_TOL) <= q <= ror_hi * (1.0 + _ROR_EDGE_TOL)
            )
            amp_ok = amp_limit is None or amps <= amp_limit
            if not in_ror:
                blocked = BLOCK_ROR_LOW if q < ror_lo else BLOCK_ROR_HIGH
            elif not amp_ok:
                blocked = BLOCK_AMPS
            else:
                blocked = None
            rows.append(
                {
                    "hz": hz,
                    "is_duty": is_duty,
                    "q_bpd": q,
                    "discharge_psi": suction_psi + dp_target,
                    "ror_lo": ror_lo,
                    "ror_hi": ror_hi,
                    "pct_of_ror_hi": 100.0 * q / ror_hi if ror_hi > 0 else None,
                    "in_ror": in_ror,
                    "bhp": bhp,
                    "amps": amps,
                    "amp_ok": amp_ok,
                    "eff_pct": PadPlant.hydraulic_efficiency(q, head, bhp, sg),
                    "blocked_by": blocked,
                }
            )
        return rows

    # -- iso-speed sheet ----------------------------------------------------

    def speed_curves(
        self,
        sg: float,
        condition: float,
        hz_max: float,
        amps_per_bhp: float,
        speeds: Iterable[float] = _CURVE_SPEEDS_HZ,
    ) -> list[dict]:
        """dP / BHP / amps / efficiency vs flow at each iso-speed line the
        capability sheet draws, capped at ``hz_max``.

        Returns:
            list[dict]: ``{hz, label, points}`` with points
                ``[flow_bpd, dp_psid, bhp, amps, eff_pct]``. Each line stops
                where the digitized table does at that speed.
        """
        wanted = sorted({float(s) for s in speeds if float(s) <= hz_max} | {hz_max})
        out = []
        for hz in wanted:
            points = []
            for q in PadPlant._curve_grid(self.max_valid_flow * hz / 60.0):
                head = self.head_ft(q, hz, condition)
                bhp = self.bhp(q, hz, sg)
                points.append(
                    [
                        q,
                        PadPlant._head_ft_to_psi(head, sg),
                        bhp,
                        amps_per_bhp * bhp,
                        PadPlant.hydraulic_efficiency(q, head, bhp, sg),
                    ]
                )
            out.append({"hz": hz, "label": f"{hz:.0f} Hz", "points": points})
        return out

    def machine_curve(self, condition: float) -> dict:
        """The vendor sheet: whole-pump head, BHP and efficiency vs flow at
        60 Hz on water, in ``schemas.PumpMachineCurve`` shape so the existing
        machine-chart renderer draws it unchanged.

        As-new and on water deliberately - this panel is the catalog page, not
        the scenario. The wear derate rides along as ``head_derated`` when the
        engineer models one.
        """
        points = []
        derated = []
        for q in PadPlant._curve_grid(self.max_valid_flow):
            head = self.n_stages * self.head_per_stage(q)
            bhp = self.n_stages * self.bhp_per_stage(q)
            points.append(
                [q, head, bhp, PadPlant.hydraulic_efficiency(q, head, bhp, 1.0)]
            )
            derated.append(
                [
                    q,
                    condition * head,
                    bhp,
                    PadPlant.hydraulic_efficiency(q, condition * head, bhp, 1.0),
                ]
            )
        wear = abs(condition - 1.0) > 1e-9
        return {
            "label": f"{self.spec['model']} - {self.n_stages} stg at 60 Hz",
            "hz": 60.0,
            "points": points,
            "head_derated": derated if wear else None,
            "derate_note": (
                f"Head at condition {condition:.2f}" if wear else None
            ),
            "bep": self.bep,
            "por": PadPlant._por(self.bep),
            "aor": list(self.ror_60hz),
            # The catalog gives one XRC range and no separate min-continuous
            # figure, so the AOR shading already carries the low end.
            "min_flow": None,
        }

    def nameplate(self, amp_limit: Optional[float], amps_per_bhp: float) -> dict:
        """Identity block: what the engineer reads before trusting the curve."""
        s = self.spec
        shaft = s["shaft_limit_hp"]
        housing = s["housing_pressure_limit_psi"]
        return {
            "key": self.key,
            "label": self.label,
            "installed": self.installed,
            "model": str(s["model"]),
            "stage_type": str(s["stage_type"]),
            "series_housing": str(s["series_housing"]),
            "arrangement": str(s["arrangement"]),
            "n_stages": self.n_stages,
            "motor": str(s["motor"]),
            "amp_limit_a": amp_limit,
            "amps_per_bhp": amps_per_bhp,
            "shaft_limit_hp": float(shaft["standard"]),
            "housing_pressure_limit_psi": float(housing["standard"]),
            "source": str(s["source"]),
        }


def candidates() -> list[EPadBooster]:
    """Both E-Pad candidates, installed build first."""
    pumps = meta()["pumps"]
    builds = [EPadBooster(k, v) for k, v in pumps.items()]
    return sorted(builds, key=lambda b: (not b.installed, b.key))


def _refine_edge(ok, q_bad: float, q_good: float) -> float:
    """Bisect a bracketed feasibility edge down to the boundary flow, returning
    the last flow on the FEASIBLE side."""
    for _ in range(_BISECT_STEPS):
        mid = 0.5 * (q_bad + q_good)
        if ok(mid):
            q_good = mid
        else:
            q_bad = mid
    return q_good


def _limit_above(
    build: EPadBooster,
    q: float,
    dp_target: float,
    sg: float,
    condition: float,
    hz_max: float,
    amps_per_bhp: float,
    amp_limit: Optional[float],
    suction_psi: float,
    q_ceiling: float,
) -> str:
    """Name the constraint that fails just above a feasible window's top."""
    probe = min(q + max(1.0, 0.002 * max(q, 1.0)), q_ceiling * 1.0000001)
    row = build.point_at_dp(
        probe, dp_target, sg, condition, hz_max, amps_per_bhp, amp_limit, suction_psi
    )
    if row["blocked_by"] == BLOCK_ROR_HIGH:
        return LIMIT_ROR_HIGH
    if row["blocked_by"] == BLOCK_AMPS:
        return LIMIT_AMPS
    if row["blocked_by"] == BLOCK_NO_SPEED:
        return build.wall_kind(q, dp_target, sg, condition, hz_max)
    return LIMIT_CAPABILITY


def solve_candidate(
    build: EPadBooster,
    *,
    dp_target: float,
    suction_psi: float,
    sg: float,
    condition: float,
    hz_max: float,
    amps_per_bhp: float,
    amp_limit: Optional[float],
) -> dict:
    """One candidate's answer at the required dP.

    Args:
        build (EPadBooster): the candidate.
        dp_target (float): required differential pressure (psid).
        suction_psi (float): booster suction (psig); discharge = suction + dP.
        sg (float): pumped-fluid specific gravity.
        condition (float): head-only wear derate (1.00 = as-new).
        hz_max (float): VFD speed cap (Hz).
        amps_per_bhp (float): amps per shaft BHP.
        amp_limit (float | None): motor amp cap. None = report amps, enforce
            nothing.

    Returns:
        dict: ``{nameplate, bep_60hz, ror_60hz, max_valid_flow_60hz,
            q_ceiling, duty, min_duty, window, limited_by, infeasible_reason,
            locus, curves, machine}``. ``duty`` is the MAX-flow feasible point
            (the deliverable rate at this dP); ``min_duty`` the bottom of the
            window (the turndown). Both None, with ``infeasible_reason`` set,
            when no flow satisfies every constraint.
    """
    q_ceiling = build.max_flow_for_dp(dp_target, sg, condition, hz_max)

    def row(q: float) -> dict:
        return build.point_at_dp(
            q, dp_target, sg, condition, hz_max, amps_per_bhp, amp_limit, suction_psi
        )

    def ok(q: float) -> bool:
        return bool(row(q)["ok"])

    def speed_table(duty_hz: Optional[float]) -> list[dict]:
        return build.speed_table(
            dp_target,
            sg,
            condition,
            hz_max,
            amps_per_bhp,
            amp_limit,
            suction_psi,
            duty_hz=duty_hz,
        )

    base = {
        "nameplate": build.nameplate(amp_limit, amps_per_bhp),
        "bep_60hz": build.bep,
        "ror_60hz": list(build.ror_60hz),
        "max_valid_flow_60hz": build.max_valid_flow,
        "q_ceiling": q_ceiling,
        "curves": build.speed_curves(sg, condition, hz_max, amps_per_bhp),
        "machine": build.machine_curve(condition),
        "throttled": build.throttled_duty(
            dp_target,
            sg,
            condition,
            hz_max,
            amps_per_bhp,
            amp_limit,
            suction_psi,
        ),
    }

    if q_ceiling <= 0.0:
        shut_off = build.dp_psi(0.0, hz_max, sg, condition)
        return {
            **base,
            "duty": None,
            "min_duty": None,
            "window": None,
            "limited_by": LIMIT_CAPABILITY,
            "infeasible_reason": (
                f"{build.n_stages} stg makes at most {shut_off:,.0f} psid at "
                f"{hz_max:.0f} Hz (shut-off), short of the {dp_target:,.0f} psid "
                "asked for"
            ),
            "locus": [],
            "speed_table": speed_table(None),
        }

    # Reported locus on the contract grid, out to the capability wall.
    locus = [row(q) for q in PadPlant._curve_grid(q_ceiling, CURVE_POINTS)]

    # Feasible band: scan finer than the reported grid, then refine both edges.
    scan = PadPlant._curve_grid(q_ceiling, _SCAN_POINTS)
    hits = [i for i, q in enumerate(scan) if ok(q)]
    if not hits:
        # Blame the highest flow the pump can actually HOLD the dP at: exactly
        # at the wall the bisected boundary can read a hair under the target
        # and report "no speed", which says nothing about why the band is
        # empty. Fall back to the wall row only if no scan point has a speed.
        held = [row(q) for q in reversed(scan)]
        top = next((r for r in held if r["hz"] is not None), held[0])
        why = top["blocked_by"] or BLOCK_NO_SPEED
        return {
            **base,
            "duty": None,
            "min_duty": None,
            "window": None,
            "limited_by": {
                BLOCK_ROR_HIGH: LIMIT_ROR_HIGH,
                BLOCK_ROR_LOW: LIMIT_ROR_HIGH,
                BLOCK_AMPS: LIMIT_AMPS,
            }.get(why, LIMIT_CAPABILITY),
            "infeasible_reason": (
                f"no flow up to the {q_ceiling:,.0f} BPD capability wall holds "
                f"{dp_target:,.0f} psid inside the recommended range"
                + ("" if amp_limit is None else f" and under {amp_limit:,.0f} A")
                + f" ({why} at {top['q_bpd']:,.0f} BPD)"
            ),
            "locus": locus,
            "speed_table": speed_table(None),
        }

    # Take the contiguous run containing the highest feasible flow - the
    # headline is the deliverable rate, and a stray low-flow island (amps and
    # range are not co-monotone in general) must not widen the window.
    hi_i = hits[-1]
    lo_i = hi_i
    while lo_i - 1 in hits:
        lo_i -= 1

    q_lo = scan[lo_i] if lo_i == 0 else _refine_edge(ok, scan[lo_i - 1], scan[lo_i])
    if hi_i == len(scan) - 1:
        q_hi = scan[hi_i]
        limited_by = build.wall_kind(q_ceiling, dp_target, sg, condition, hz_max)
    else:
        q_hi = _refine_edge(ok, scan[hi_i + 1], scan[hi_i])
        limited_by = _limit_above(
            build,
            q_hi,
            dp_target,
            sg,
            condition,
            hz_max,
            amps_per_bhp,
            amp_limit,
            suction_psi,
            q_ceiling,
        )

    duty = row(q_hi)
    min_duty = row(q_lo)
    return {
        **base,
        "duty": duty,
        "min_duty": min_duty,
        "window": [q_lo, q_hi],
        "limited_by": limited_by,
        "infeasible_reason": None,
        "locus": locus,
        "speed_table": speed_table(duty["hz"]),
    }


def capability_report(
    *,
    dp_psid: float,
    suction_psi: float,
    sg: float,
    condition: float,
    hz_max: float,
    amps_per_bhp: float,
    amp_limit: Optional[float],
) -> dict[str, Any]:
    """Both candidates' answers at one required dP (JSON-safe).

    Args:
        dp_psid (float): required differential pressure across the booster.
        suction_psi (float): booster suction (psig).
        sg (float): pumped-fluid specific gravity.
        condition (float): head-only wear derate (1.00 = as-new).
        hz_max (float): VFD speed cap (Hz).
        amps_per_bhp (float): amps per shaft BHP.
        amp_limit (float | None): motor amp cap; None enforces nothing.

    Returns:
        dict: ``{pad, target{}, notes{}, candidates[]}``; see
            ``schemas.EPadBoosterResponse``.
    """
    m = meta()
    d = m["defaults"]
    return {
        "pad": "E",
        "target": {
            "dp_psid": dp_psid,
            "suction_psi": suction_psi,
            "discharge_psi": suction_psi + dp_psid,
            "sg": sg,
            "condition": condition,
            "hz_max": hz_max,
            "amps_per_bhp": amps_per_bhp,
            "amp_limit_a": amp_limit,
            "header_default_psi": float(d["target_discharge_psi"]),
        },
        "notes": {
            "amps": str(d["amps_per_bhp_note"]),
            "condition": str(m["reference"]["condition_note"]),
            "housing_pressure": str(m["housing_pressure_caveat"]),
            "not_enforced": [str(x) for x in m["not_enforced"]],
            "stage_table": str(m["reference"]["stage_table_note"]),
        },
        "candidates": [
            solve_candidate(
                build,
                dp_target=dp_psid,
                suction_psi=suction_psi,
                sg=sg,
                condition=condition,
                hz_max=hz_max,
                amps_per_bhp=amps_per_bhp,
                amp_limit=amp_limit,
            )
            for build in candidates()
        ],
    }
