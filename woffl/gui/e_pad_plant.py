"""E-Pad booster plant — the single VFD Summit unit as a ``PadPlant``.

E-Pad boosts on-pad into the 3,400 psig power-fluid header
(``server/services/wells.py:_PAD_PF_DEFAULTS``). One 26-stage Summit SM25000
on a VFD (the E-41 surface kit), taking ~2,700 psi suction from the CFP water
header: a single 26-stage unit makes at most ~1,500 psid, so it cannot lift
separator water to 3,400 by itself and is the HP/final stage of a train, the
same architecture as I-Pad's LP -> HP pair.

Field-calibrated (2026-09-24 rate test, meta ``field_test``): the installed
unit current-limits at 889 A at 29,491 BWPD, 53.1 Hz and 3,400 psi from
2,704 psi suction. By default the plant carries that motor (amps per BHP and
the 889 A limit) and, for the installed build, the head derate and the upper
operating range derived from that point (``e_pad_booster.field_calibration``),
so its capacity at 3,400 psi is the tested rate.

``coupling="free_pressure"``: the delivered header is a decision variable
bounded above by a capability frontier, like I-Pad and M-Pad. The frontier
here is shaped by the RECOMMENDED OPERATING RANGE rather than by motor amps,
and that makes it **unimodal in flow**, which the I/M frontiers are not:

* Above ``ror_hi * hz_max/60`` no speed keeps the flow on the curve at all.
* Below ``ror_lo * hz_max/60`` the range FLOOR binds instead - the drive has
  to slow down to keep the flow off the left end of the curve, and deliverable
  pressure collapses with the square of the speed.

So the frontier rises to a knee at ``ror_lo * hz_max/60`` and falls from there
to the flow ceiling. Every inverse in here scans then bisects the falling
branch instead of assuming a monotone curve.

``budget_at_pressure`` is "the most PF the plant can push at AT LEAST this
header", the same semantics the I and M plants use. That is the run-flat-out
policy: at 3,400 psi the installed build has pressure to spare, so the surplus
is choked off. The no-throttle alternative - slow the drive until it makes
exactly the required dP, which shrinks the range and the deliverable rate -
is priced on the E-Pad booster screen (``e_pad_booster.throttled_duty`` vs
``solve_candidate``), because it is an operating decision, not a plant fact.

Physics and the vendor data live in ``woffl.gui.e_pad_booster``; this module is
only the ``PadPlant`` face on it. Fork-only, MPU pump data, no upstream PR.
"""

from typing import Iterable, Optional
from math import isfinite

from woffl.gui.e_pad_booster import EPadBooster, candidates, defaults, field_calibration
from woffl.gui.pad_plant_base import (
    PF_CONSTRAINT_MIN_PSI,
    PadPlant,
    clamp_to_pf_constraint,
)

# Installed build. The alternative is a candidate on the booster screen, not
# something the optimizer may silently assume is in the ground.
INSTALLED_BUILD = "SM25000_26STG"

# Scan resolution for the frontier inverses. The frontier is unimodal, so a
# bracketing scan comes first and only then a bisection.
_SCAN_POINTS = 241
_BISECT_STEPS = 60

# Operational discharge cap (psi), adopted from I-Pad's `_MAX_HEADER_PSI`
# pending an E-Pad-specific piping/wellhead number. NOT a pump limit: the
# installed build's own frontier peaks near 4,560 psi, so above this cap it is
# the cap, not the booster, that limits the header.
_MAX_HEADER_PSI = 3500.0

# Lift above suction the optimizer sweep starts at, mirroring I-Pad's floor.
_SWEEP_FLOOR_LIFT_PSI = 200.0

# "Not given": take the field-measured motor current limit. Distinct from an
# explicit None, which enforces no cap.
_FIELD_DEFAULT = object()


class EPadPlant(PadPlant):
    """E-Pad's booster as the uniform plant interface.

    Args:
        build_key (str): meta ``pumps`` key; defaults to the installed build.
        suction_psi (float | None): booster suction (psig). None takes the
            meta default (2,704, CFP header suction measured at the tested
            maximum rate).
        sg (float | None): pumped-fluid SG. None takes the meta default.
        condition (float | None): head-only wear derate (1.00 = as-new).
            None takes the rate-test calibration for the installed build
            (about 0.77) and 1.00 for any other build.
        hz_max (float): VFD speed cap (Hz).
        amps_per_bhp (float | None): amps per shaft BHP. None takes the meta
            default, calibrated to the E-41 motor in the rate test.
        amp_limit (float | None): motor amp cap. Omitted takes the E-41
            drive's measured current limit (889 A); an explicit None
            enforces no cap.
        field_calibrated (bool): apply the rate-test head derate and upper
            range to the installed build (default). False models an as-new
            unit on the catalog curve and range, e.g. a replacement.
    """

    coupling = "free_pressure"
    # E-Pad pumps handle formation AND lift water (Scott, 2026-09-02;
    # well_sort_engine.POPS_PUMP_HANDLES["E"] == "total"): budget and
    # price TOTAL water. See docs/optimization_redesign_2026-09.md section 5.
    water_key = "totl_wat"
    n_pump_options = []  # single machine — no pump-count choice
    max_header_psi = _MAX_HEADER_PSI

    infeasible_sweep_msg = (
        "No feasible header pressure found — the E-Pad booster couldn't deliver "
        "any well's power-fluid demand inside its recommended operating range. "
        "Check the IPRs and the booster suction."
    )

    def __init__(
        self,
        build_key: str = INSTALLED_BUILD,
        *,
        suction_psi: Optional[float] = None,
        sg: Optional[float] = None,
        condition: Optional[float] = None,
        hz_max: float = 60.0,
        max_header_psi: Optional[float] = None,
        amps_per_bhp: Optional[float] = None,
        amp_limit=_FIELD_DEFAULT,
        field_calibrated: bool = True,
    ) -> None:
        d = defaults()
        hits = [b for b in candidates() if b.key == build_key]
        if not hits:
            raise ValueError(f"unknown E-Pad booster build '{build_key}'")
        self.build: EPadBooster = hits[0]  # a fresh instance: safe to calibrate
        # The rate test ran on the installed unit only. Its demonstrated
        # upper range always applies to that unit (it ran there); its head
        # derate is the default unless the caller states a condition.
        self.field_calibration = (field_calibration(build_key)
                                  if self.build.installed and field_calibrated else None)
        if self.field_calibration is not None:
            self.build.ror_60hz = (self.build.ror_60hz[0], self.field_calibration["ror_hi_60hz"])
        self._suction = float(d["suction_psi"] if suction_psi is None else suction_psi)
        self._sg = float(d["sg"] if sg is None else sg)
        if condition is None:
            condition = (self.field_calibration["condition"] if self.field_calibration is not None
                         else d["condition"])
        self.condition = float(condition)
        self.hz_max = float(hz_max)
        # Instance attribute deliberately shadows the class default: the
        # operational cap is an E-Pad piping number nobody has handed over, so
        # a run must be able to say what it really is.
        self.max_header_psi = float(
            _MAX_HEADER_PSI if max_header_psi is None else max_header_psi
        )
        self.amps_per_bhp = float(
            d["amps_per_bhp"] if amps_per_bhp is None else amps_per_bhp
        )
        if amp_limit is _FIELD_DEFAULT:
            amp_limit = d.get("amp_limit_a")
        self.amp_limit = None if amp_limit is None else float(amp_limit)
        if not all(isfinite(v) for v in (self._suction, self._sg, self.condition,
                                         self.hz_max, self.max_header_psi, self.amps_per_bhp)):
            raise ValueError("E-Pad plant inputs must be finite")
        if not 0 <= self._suction < self.max_header_psi <= 5000 or self.max_header_psi < 1000:
            raise ValueError("E-Pad header cap must exceed suction and stay within 1000-5000 psi")
        if min(self._sg, self.condition, self.hz_max, self.amps_per_bhp) <= 0:
            raise ValueError("E-Pad fluid, condition, speed and amp conversion must be positive")
        if self.amp_limit is not None and (not isfinite(self.amp_limit) or self.amp_limit <= 0):
            raise ValueError("E-Pad amp limit must be finite and positive")

    # -- pad physics ---------------------------------------------------------

    def specific_gravity(self) -> float:
        return self._sg

    def suction_psi(self) -> float:
        """Booster suction (psig) — the upstream stage's discharge."""
        return self._suction

    def knee_flow(self) -> float:
        """Flow (BPD) where the recommended-range FLOOR stops binding, i.e.
        the frontier's peak. Below it the drive must slow to keep the flow off
        the left end of the curve and deliverable pressure collapses."""
        return self.build.ror_60hz[0] * self.hz_max / 60.0

    def flow_ceiling(self) -> float:
        """Flow (BPD) above which no speed keeps the flow inside the
        recommended range — the hydraulic throughput limit."""
        return self.build.ror_60hz[1] * self.hz_max / 60.0

    def max_discharge_pressure(self, total_flow_bpd: float) -> Optional[float]:
        """Highest header (psi) the booster can deliver at a total PF flow,
        inside its recommended range and under its amp cap. None past either
        end of the range."""
        dp = self.build.max_dp_at_flow(
            total_flow_bpd,
            self._sg,
            self.condition,
            self.hz_max,
            self.amps_per_bhp,
            self.amp_limit,
        )
        return None if dp is None else self._suction + dp

    def max_flow_at_pressure(self, pressure: float) -> float:
        """Largest total PF the booster can push at >= ``pressure`` — the
        frontier inverted, which is the optimizer's PF budget at a candidate
        header. 0.0 when no in-range flow reaches the pressure.

        Scanned then bisected on the falling branch: the frontier is unimodal,
        so a plain monotone bisection from zero flow would report 0.0 whenever
        the pressure is above what the collapsed low-flow branch can make.
        """
        top = self.flow_ceiling()
        grid = PadPlant._curve_grid(top, _SCAN_POINTS)

        def ok(q: float) -> bool:
            psi = self.max_discharge_pressure(q)
            return psi is not None and psi >= pressure

        hits = [i for i, q in enumerate(grid) if ok(q)]
        if not hits:
            return 0.0
        hi_i = hits[-1]
        if hi_i == len(grid) - 1:
            return grid[hi_i]
        lo, hi = grid[hi_i], grid[hi_i + 1]
        for _ in range(_BISECT_STEPS):
            mid = 0.5 * (lo + hi)
            if ok(mid):
                lo = mid
            else:
                hi = mid
        return lo

    # -- uniform interface ---------------------------------------------------

    def header_at_flow(
        self, q_total: float, n_pumps: int | None = None
    ) -> Optional[float]:
        return self.max_discharge_pressure(q_total)  # single machine — n/a

    def budget_at_pressure(self, pressure: float, n_pumps: int | None = None) -> float:
        return self.max_flow_at_pressure(pressure)

    def warm_start_psi(self, n_pumps: int | None = None) -> float:
        # The live PF header setpoint, capped operationally.
        return min(self.max_header_psi, float(defaults()["target_discharge_psi"]))

    def match_check_header(self, total_pf: float, n_pumps: int | None = None) -> float:
        header = self.max_discharge_pressure(total_pf) if total_pf > 0 else None
        if header is None:
            header = self.warm_start_psi(n_pumps)
        # Cap at the operational discharge limit like I-Pad: uncapped, a small
        # measured total PF puts the frontier near 4,560 psi and every well
        # gets a spurious pass (the P0-7 family).
        return min(self.max_header_psi, header)

    def match_check_budget_bpd(
        self, total_pf: float, n_pumps: int | None = None
    ) -> float:
        return max(total_pf * 1.5, self.flow_ceiling())

    def flow_window(self, n_pumps: int | None = None) -> tuple[float, float]:
        """(recirc floor, throughput ceiling) in total PF BPD. The floor is the
        frontier knee: below it the booster can still run, but only by slowing
        down, and it can no longer hold a useful header."""
        return self.knee_flow(), self.flow_ceiling()

    def pressure_window(self, n_pumps: int | None = None) -> tuple[float, float]:
        floor = max(self._suction + _SWEEP_FLOOR_LIFT_PSI, PF_CONSTRAINT_MIN_PSI)
        # Always scan: an amp limit moves the maximum off the nominal knee
        # even when the knee itself still solves (40 A: true peak 3,928 psi
        # at 6,480 BPD vs 3,870 at the knee), which cut the sweep short.
        available = [self.max_discharge_pressure(q) for q in
                     [self.knee_flow(), *PadPlant._curve_grid(self.flow_ceiling(), _SCAN_POINTS)]]
        peak = max((p for p in available if p is not None), default=None)
        if peak is None:
            raise ValueError("E-Pad has no operating pressure inside its speed, amp and flow limits")
        ceiling = clamp_to_pf_constraint(
            min(self.max_header_psi, peak)
        )
        if ceiling <= max(self._suction, PF_CONSTRAINT_MIN_PSI):
            raise ValueError("E-Pad has no boosted operating pressure within its header cap")
        floor = min(floor, ceiling)
        return floor, ceiling

    def flags(self, q_total: float, n_pumps: int | None = None) -> dict:
        in_range = self.max_discharge_pressure(q_total) is not None
        # Two different failures, and the pad page says which: too little flow
        # (range floor, recirculation) or too much (range ceiling).
        recirc = not in_range and q_total < self.knee_flow()
        return {
            "in_range": in_range,
            "recirc": recirc,
            "over_capacity": not in_range and not recirc,
        }

    def envelope(
        self,
        flows: Iterable[float],
        n_pumps: int | None = None,
        at_pressure: float | None = None,
    ) -> list[dict]:
        """One consistent speed/head/amp point per flow. The represented
        policy runs at available speed and throttles surplus pressure; a
        requested header above the frontier or cap is explicitly infeasible."""
        rows = []
        for q in flows:
            hz = self.build.max_hz_at_flow(q, self._sg, self.hz_max, self.amps_per_bhp, self.amp_limit)
            psi = self.max_discharge_pressure(q)
            if hz is None or psi is None:
                rows.append(
                    {
                        "flow": q,
                        "max_discharge_psi": None,
                        "per_pump_bpd": q,
                        "feasible": False,
                        "recirc": q < self.knee_flow(),
                        "pumps": [],
                    }
                )
                continue
            feasible = at_pressure is None or (
                self._suction <= at_pressure <= self.max_header_psi and at_pressure <= psi + 1e-7)
            rows.append(
                {
                    "flow": q,
                    "max_discharge_psi": psi,
                    "per_pump_bpd": q,
                    "feasible": feasible,
                    "delivered_header_psi": min(psi, self.max_header_psi if at_pressure is None else at_pressure),
                    "throttle_psi": max(0., psi - (self.max_header_psi if at_pressure is None else at_pressure)),
                    "recirc": False,
                    "pumps": [
                        {
                            "name": self.build.label,
                            "n": 1,
                            "hz": hz,
                            "dP": psi - self._suction,
                            "amps": self.build.amps(q, hz, self._sg, self.amps_per_bhp),
                            "amp_limit": self.amp_limit,
                        }
                    ],
                }
            )
        return rows

    # -- curve report --------------------------------------------------------

    def _validation_note(self) -> str:
        """What the curves rest on, stated on the nameplate."""
        cal = self.field_calibration
        if cal is not None:
            p = cal["point"]
            return (
                "Calibrated to one E-41 rate-test point: current limit "
                f"{p['amps']:,.0f} A at {p['rate_bwpd']:,.0f} BWPD, {p['hz']:g} Hz, "
                f"{p['discharge_psi']:,.0f} psi from {p['suction_psi']:,.0f} psi suction "
                f"(head condition {self.condition:.2f}). Other rates and headers are "
                "the catalog curve derated to that point, not measured."
            )
        return (
            "Catalog stage curve plus the Summit workbook's affinity sheet (as new); "
            f"suction {self._suction:,.0f} psi. The installed unit's rate test is "
            "applied only to the field-calibrated installed build."
        )

    def curve_report(self, n_pumps: int | None = None) -> dict:
        """Station + machine curves for the E-Pad booster.

        Args:
            n_pumps (int | None): carried into the payload only — one machine.

        Returns:
            dict: the ``PadPlant.curve_report`` payload. The station family is
                one iso-speed line per drawn speed; the frontier is the
                range-limited capability the optimizer rides, unimodal in flow.
        """
        build = self.build
        curves = []
        for line in build.speed_curves(
            self._sg, self.condition, self.hz_max, self.amps_per_bhp
        ):
            curves.append(
                {
                    "label": line["label"],
                    "n_pumps": None,
                    "hz": line["hz"],
                    "active": line["hz"] == self.hz_max,
                    # station axis is DELIVERED header, not differential
                    "points": [[p[0], self._suction + p[1]] for p in line["points"]],
                }
            )

        front_pts = []
        for q in PadPlant._curve_grid(self.flow_ceiling()):
            psi = self.max_discharge_pressure(q)
            if psi is not None:
                front_pts.append([q, psi])

        machine = build.machine_curve(self.condition)
        return {
            "pad": "E",
            "coupling": self.coupling,
            "n_pumps": self._n(n_pumps),
            "sg": self._sg,
            "suction_psi": self._suction,
            "max_header_psi": self.max_header_psi,
            "nameplate": {
                "equipment": build.label,
                "model": f"{build.spec['model']} {build.spec['stage_type']}, "
                f"{build.n_stages} stg",
                "arrangement": "1 x VFD, HP/final stage into the PF header",
                "speed": f"3,500 RPM at 60 Hz (VFD), capped {self.hz_max:.0f} Hz",
                "source": str(build.spec["source"]),
                "validated": self._validation_note(),
            },
            "station": {
                "curves": curves,
                "frontier": {
                    "label": (
                        f"Recommended range limit ({build.ror_60hz[0]:,.0f}-"
                        f"{build.ror_60hz[1]:,.0f} BPD at 60 Hz)"
                    ),
                    "n_pumps": None,
                    "hz": None,
                    "active": False,
                    "points": front_pts,
                },
                "bep": build.bep,
                "por": PadPlant._por(build.bep),
                "aor": list(build.ror_60hz),
                "min_flow": self.knee_flow(),
                "header_cap": self.max_header_psi,
            },
            "pumps": [machine],
        }


# Public handle for the unified pad optimizer, mirroring s/i/m_pad_plant.
PLANT = EPadPlant()
