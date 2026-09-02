"""E-Pad booster plant — the single VFD Summit unit as a ``PadPlant``.

E-Pad boosts on-pad into the 3,400 psig power-fluid header
(``server/services/wells.py:_PAD_PF_DEFAULTS``). One 26-stage Summit SM25000
on a VFD, taking ~2,800 psi suction from the upstream stage: a single 26-stage
unit makes at most ~1,500 psid, so it cannot lift separator water to 3,400 by
itself and is the HP/final stage of a train, the same architecture as I-Pad's
LP -> HP pair.

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

from woffl.gui.e_pad_booster import EPadBooster, candidates, defaults
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


class EPadPlant(PadPlant):
    """E-Pad's booster as the uniform plant interface.

    Args:
        build_key (str): meta ``pumps`` key; defaults to the installed build.
        suction_psi (float | None): booster suction (psig). None takes the
            meta default (2,800, the Summit workbook's suction cell).
        sg (float | None): pumped-fluid SG. None takes the meta default.
        condition (float): head-only wear derate (1.00 = as-new).
        hz_max (float): VFD speed cap (Hz).
        amps_per_bhp (float | None): amps per shaft BHP. None takes the meta
            default (a transferred estimate - see the booster module).
        amp_limit (float | None): motor amp cap. None enforces nothing, which
            is the default because no E-Pad motor nameplate exists in the
            vendor data.
    """

    coupling = "free_pressure"
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
        condition: float = 1.0,
        hz_max: float = 60.0,
        max_header_psi: Optional[float] = None,
        amps_per_bhp: Optional[float] = None,
        amp_limit: Optional[float] = None,
    ) -> None:
        d = defaults()
        hits = [b for b in candidates() if b.key == build_key]
        if not hits:
            raise ValueError(f"unknown E-Pad booster build '{build_key}'")
        self.build: EPadBooster = hits[0]
        self._suction = float(d["suction_psi"] if suction_psi is None else suction_psi)
        self._sg = float(d["sg"] if sg is None else sg)
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
        self.amp_limit = None if amp_limit is None else float(amp_limit)

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
        peak = self.max_discharge_pressure(self.knee_flow())
        ceiling = clamp_to_pf_constraint(
            min(self.max_header_psi, peak if peak is not None else self.max_header_psi)
        )
        if ceiling <= floor:
            ceiling = floor + 500.0
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
        """Per-flow frontier rows: the deliverable header and the speed / amps
        it takes. ``n_pumps`` and ``at_pressure`` are ignored - one machine,
        and the frontier already is the pressure."""
        rows = []
        for q in flows:
            window = self.build.hz_window_in_ror(q, self.hz_max)
            psi = self.max_discharge_pressure(q)
            if window is None or psi is None:
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
            hz = window[1]
            rows.append(
                {
                    "flow": q,
                    "max_discharge_psi": psi,
                    "per_pump_bpd": q,
                    "feasible": True,
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
                "validated": (
                    "NOT validated against live E-Pad SCADA - catalog stage curve "
                    "plus the Summit workbook's affinity sheet. Suction "
                    f"{self._suction:,.0f} psi is the workbook's cell, not a "
                    "measured tag."
                ),
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
