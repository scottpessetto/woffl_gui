"""CFP produced-water plant as a :class:`~woffl.gui.pad_plant_base.PadPlant`.

Wraps :mod:`woffl.assembly.cfp_plant` so the four CFP-side pads (J/G/C/B) can
be optimized against the central plant through the same interface the S/I/M
pad plants use. GUI-only (MPU-specific plant data), no upstream library PR.

WHY ``free_pressure`` AND NOT ``fixed_curve``
---------------------------------------------
The original plan modeled CFP as ``fixed_curve``: water production sets plant
throughput, throughput sets discharge, the optimizer iterates to a fixed point.
A 120-day regression of metered throughput against measured discharge killed
that: the slope is -1.8 psi per 1,000 BWPD at r²=0.03, versus the -17.5 the
curve implies. Throughput swung 41,591 BWPD while discharge moved 75 psi.

The reason (Scott, 2026-07-29): **operators set the discharge pressure by
opening or closing the disposal well**, which moves total flow through the
pumps and so where they ride on the curve. Pressure is a decision, not an
outcome — which is exactly what ``free_pressure`` models (cf. the I-Pad VFD
train and M-Pad HP bank). The physical constraint is that the pumps must pass
everything arriving: **water arriving sets a minimum flow, which caps the
achievable discharge**, and ``MAX_DISCHARGE_PSI`` (the piping rating, above
which the pumps trip) caps it again. Cutting controllable water at B/G/C/J
lowers the required flow and buys discharge pressure, hence PF, hence lift.

TWO INTERFACE SEMANTICS THAT DIFFER FROM S/I/M — READ BEFORE USE
----------------------------------------------------------------
1. ``header_at_flow`` / ``budget_at_pressure`` / ``flow_window`` are in
   **TOTAL WATER**, not power fluid. On a POPS pad the pad pump's capacity is
   the PF budget and it binds. Here PF is a small slice of throughput
   (~27,000 BWPD of PF against ~112,300 BWPD metered), so the PF volume never
   binds — the binding quantity is total water, because that is what sets the
   pressure. A caller that treats ``budget_at_pressure`` as a PF budget will
   think it has 88,195 BWPD of power fluid available. It does not.
2. ``header_at_flow`` returns **PLANT DISCHARGE**, not delivered pad pressure.
   The plant→pad step is per-pad and lives in :meth:`delivered_pf_for_pad`,
   because B/G/J each have their own line loss and C-Pad is boosted on-pad and
   ignores the plant discharge entirely.

Every absolute number is PROVISIONAL — see the acceptance test in
``woffl.assembly.cfp_plant``'s docstring (the replacement curve must pass
~112,000 BWPD at ~2,790 psi; today's fit passes only ~95,000 there).
"""

from typing import Iterable, Optional

from woffl.assembly import cfp_plant as _cfp
from woffl.assembly.network_optimizer import derive_pad
from woffl.gui.pad_plant_base import (
    PF_CONSTRAINT_MIN_PSI,
    PadPlant,
    clamp_to_pf_constraint,
)


class CFPMachineSubsetUnvalidated(ValueError):
    """Raised when a <3-machine curve is requested before per-machine validation.

    Subclasses ``ValueError`` so existing ``except ValueError`` handlers in the
    pad pages keep working. The GUI catches this and shows the message rather
    than emitting numbers from a fit that was never per-machine (machine C's
    fitted parabola rises to a 3,015 psi vertex and its q=0 intercept is
    2,151 psi — below the whole operating window).
    """


# n_pumps (machines online) -> which machines. Ordered A, B, C so a 2-machine
# case is the two whose individual fits are least pathological (A is properly
# monotone; C is the artifact-heavy one).
_MACHINES_BY_COUNT: dict[int, tuple[str, ...]] = {
    3: ("A", "B", "C"),
    2: ("A", "B"),
    1: ("A",),
}

# Produced water. The plant moves formation + returned power fluid; SG sits a
# little above fresh water. Matches the wat_sg the well models use.
_PRODUCED_WATER_SG = 1.02

# Produced-water header (pump suction). Low-pressure — the real number does not
# matter much because ``clamp_window`` floors the iterate band at
# PowerFluidConstraint's 1,000 psi anyway; it is here so the interface has an
# honest value rather than a magic one.
_SUCTION_PSI = 50.0


# Pad-letter resolution is deliberately NOT reimplemented here — it delegates to
# the single copy in network_optimizer so pad identity can't drift between the
# optimizer, the plant models and the PF seeding (the duplicate-lift-classifier
# lesson in AGENTS.md). Re-exported so callers of this module have it to hand.
pad_letter = derive_pad


class CFPPlant(PadPlant):
    """The CFP's three parallel produced-water machines (A/B/C).

    See the module docstring for the two interface semantics that differ from
    the POPS pad plants (total water rather than PF; plant discharge rather
    than delivered pad pressure).
    """

    coupling = "free_pressure"
    n_pump_options = [3, 2, 1]
    max_header_psi = _cfp.MAX_DISCHARGE_PSI  # 2,900 psi piping trip

    infeasible_sweep_msg = (
        "No feasible plant discharge found — at every trial pressure the "
        "machines could not pass the water arriving. Reduce the exogenous "
        "water, bring another machine online, or check the IPRs."
    )

    # -- machine selection ---------------------------------------------------

    def machines_for(self, n_pumps: int | None = None) -> tuple[str, ...]:
        """Machines online for a pump count, refusing unvalidated subsets."""
        n = self._n(n_pumps)
        if n is None:
            return _cfp.ALL_MACHINES
        try:
            picked = _MACHINES_BY_COUNT[int(n)]
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                f"n_pumps must be one of {sorted(_MACHINES_BY_COUNT)}, got {n!r}"
            ) from None
        if len(picked) < len(_cfp.ALL_MACHINES) and not _cfp.MACHINE_CURVE_VALIDATED:
            raise CFPMachineSubsetUnvalidated(
                f"Running {len(picked)} of {len(_cfp.ALL_MACHINES)} machines is not "
                "modelable yet: the per-machine coefficients were fitted to TOTAL "
                "plant behavior over 2,200-2,700 psi, never validated individually "
                "(machine C's fit implies a 2,151 psi shutoff head, below the whole "
                "operating window). Set cfp_plant.MACHINE_CURVE_VALIDATED = True "
                "once each machine's curve is confirmed on its own."
            )
        return picked

    def machine_subset_available(self) -> bool:
        """True when <3-machine cases can be modeled (per-machine curve validated)."""
        return bool(_cfp.MACHINE_CURVE_VALIDATED)

    # -- PadPlant surface ----------------------------------------------------

    def specific_gravity(self) -> float:
        return _PRODUCED_WATER_SG

    def suction_psi(self) -> float:
        return _SUCTION_PSI

    def header_at_flow(
        self, q_total: float, n_pumps: int | None = None
    ) -> Optional[float]:
        """PLANT DISCHARGE (psi) when the machines must pass ``q_total`` TOTAL
        WATER — capped at the piping trip.

        ``None`` when the flow is past the plant's capability (the inversion
        pinned at the window floor), matching the PadPlant contract.
        """
        machines = self.machines_for(n_pumps)
        pressure, status = _cfp.plant_pressure_detail(q_total, machines)
        if status == "pinned_low":
            return None  # cannot pass this much water at any modelable pressure
        return min(pressure, self.max_header_psi)

    def budget_at_pressure(self, pressure: float, n_pumps: int | None = None) -> float:
        """TOTAL WATER (BWPD) the machines pass at that discharge — NOT a PF
        budget (see the module docstring)."""
        return _cfp.plant_flow(pressure, self.machines_for(n_pumps))

    def flow_window(self, n_pumps: int | None = None) -> tuple[float, float]:
        """(0, max total water) — the ceiling is throughput at the window floor."""
        machines = self.machines_for(n_pumps)
        return 0.0, _cfp.plant_flow(_cfp.PRESSURE_WINDOW[0], machines)

    def pressure_window(self, n_pumps: int | None = None) -> tuple[float, float]:
        """Discharge band the optimizer sweeps: invertible floor → piping trip."""
        self.machines_for(n_pumps)  # validate the subset even though the band is fixed
        floor = max(_cfp.PRESSURE_WINDOW[0], PF_CONSTRAINT_MIN_PSI)
        ceiling = clamp_to_pf_constraint(self.max_header_psi)
        if ceiling <= floor:
            ceiling = floor + 500.0
        return clamp_to_pf_constraint(floor), ceiling

    def flags(self, q_total: float, n_pumps: int | None = None) -> dict:
        """Standard three flags plus the two CFP-specific honesty flags.

        ``trusted_band`` False means the resulting discharge is outside
        2,200-2,700 psi, i.e. extrapolation of the spreadsheet fit.
        ``pinned`` carries the raw inversion status so a caller can say
        "this pressure is a clamp" rather than presenting it as physics.
        """
        machines = self.machines_for(n_pumps)
        pressure, status = _cfp.plant_pressure_detail(q_total, machines)
        over = status == "pinned_low"
        capped = pressure > self.max_header_psi
        effective = min(pressure, self.max_header_psi)
        return {
            "in_range": not over,
            # No recirc floor is modeled: disposal throttling IS the plant's
            # minimum-flow mechanism, and it lives outside this curve.
            "recirc": False,
            "over_capacity": over,
            "trusted_band": _cfp.in_trusted_band(effective),
            "pinned": status,
            "trip_capped": capped,
        }

    def envelope(
        self,
        flows: Iterable[float],
        n_pumps: int | None = None,
        at_pressure: float | None = None,
    ) -> list[dict]:
        """Per-flow operating rows. ``at_pressure`` is ignored (the discharge is
        derived from the flow, not imposed on it)."""
        machines = self.machines_for(n_pumps)
        rows = []
        for q in flows:
            pressure, status = _cfp.plant_pressure_detail(float(q), machines)
            effective = min(pressure, self.max_header_psi)
            rows.append(
                {
                    "flow": float(q),
                    "max_discharge_psi": effective,
                    "feasible": status != "pinned_low",
                    "pumps": len(machines),
                    "machines": ",".join(machines),
                    "pinned": status,
                    "trusted_band": _cfp.in_trusted_band(effective),
                    "per_machine_bwpd": {
                        m: _cfp.machine_flow(m, effective) for m in machines
                    },
                }
            )
        return rows

    def warm_start_psi(self, n_pumps: int | None = None) -> float:
        """Start at the plant's MEASURED discharge rather than the sweep ceiling
        — it is where the plant actually sits, so the first iterate is already
        near the answer."""
        lo, hi = self.pressure_window(n_pumps)
        return min(max(_cfp.MEASURED_DISCHARGE_PSI, lo), hi)

    def match_check_header(self, total_pf: float, n_pumps: int | None = None) -> float:
        """Model the pre-flight check at the plant's MEASURED discharge.

        The base implementation derives a header from the pad's measured PF via
        ``header_at_flow``, but here that argument is power fluid while
        ``header_at_flow`` wants total water — feeding one to the other would be
        a unit error. The measured discharge is both correct and more honest.
        """
        lo, hi = self.clamp_window(n_pumps)
        return min(max(_cfp.MEASURED_DISCHARGE_PSI, lo), hi)

    # -- plant -> pad delivery ----------------------------------------------

    def delivered_pf_for_pad(
        self,
        pad: str,
        disch_p: float,
        measured_pad_pf: float | None = None,
    ) -> Optional[float]:
        """PF pressure delivered to a pad at a given plant discharge.

        Prefers a **measured anchor**: hold the pad's own gauge reading and move
        it by the CHANGE in discharge from the metered baseline. The hardcoded
        ``PAD_LINE_DP`` constants are only the fallback, because all three were
        measured wrong on 2026-07-29 (discharge 2,792 psi vs live per-well PF):

        ==== ========= ======== ==============================================
        pad  table dP  real dP  evidence
        ==== ========= ======== ==============================================
        B    272       ~169     5 wells clustered 2,619-2,630 psi (11 psi band)
        G    293       ~44      only MPG-18 (2,748) is on the header
        J    251       ~110     MPJ-27/29/32 clustered 2,678-2,687
        ==== ========= ======== ==============================================

        So the table under-delivers PF by 100-250 psi, which biases every CFP
        jet-pump simulation pessimistic. Note the pad *median* is the wrong
        statistic on these pads — B/G/J/C are ESP-heavy, so only a minority of
        wells sit on the JP PF header and the high CLUSTER is the header (see
        ``cfp_pad_pf`` for the cluster resolver).

        Returns ``None`` for a pad not supplied off the plant discharge — C-Pad,
        which is boosted on-pad (~3,400 psi measured) and holds its PF
        independently. Callers supply C-Pad's pressure themselves.
        """
        key = pad_letter(pad)
        if key not in _cfp.PAD_LINE_DP:
            return None
        if measured_pad_pf is not None:
            return float(measured_pad_pf) + (
                float(disch_p) - _cfp.MEASURED_DISCHARGE_PSI
            )
        return float(disch_p) - _cfp.PAD_LINE_DP[key]

    def plant_supplied_pads(self) -> tuple[str, ...]:
        """Pads whose PF rides the plant discharge (C-Pad is boosted on-pad)."""
        return tuple(sorted(_cfp.PAD_LINE_DP))


# Module-level singleton, mirroring the s/i/m_pad_plant delegation pattern.
PLANT = CFPPlant()
