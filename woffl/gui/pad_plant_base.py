"""Pad booster-plant base — one `PadPlant` interface over the three pad models.

R-1 Phase A (docs/code_review_2026-07-01.md): the physics that used to live as
module-level functions in ``s_pad_plant`` / ``i_pad_plant`` / ``m_pad_plant``
now lives in the subclasses here, moved verbatim so every number is
bit-identical (the pins in ``tests/test_pad_plants.py`` are the harness). The
legacy modules keep their public APIs as thin delegations to a module-level
singleton — the pad pages are untouched in this phase.

Two coupling kinds:

* ``fixed_curve`` — the delivered PF header pressure is a *curve of flow* (the
  S-Pad fixed-speed parallel station; also ``FixedHeaderPlant`` where it's a
  constant). The optimizer iterates flow -> pressure to a fixed point.
* ``free_pressure`` — pressure is a *decision variable* bounded above by an
  amp/speed capability frontier (I-Pad series VFD train, M-Pad parallel VFD HP
  bank). The optimizer sweeps candidate pressures; ``budget_at_pressure`` is
  the PF budget the plant can push at each.

Cross-cutting constraint (the review's P0-7 family): the optimizer's
``PowerFluidConstraint`` rejects any pressure outside [1000, 5000] psi, so
every pressure this interface hands back for optimizer use — the
``pressure_window()`` floor and ceiling in particular — is clamped into that
band regardless of what the physical curve says (the S-Pad curve sits near
3,900 psi at shut-in and can extrapolate anywhere; a no-booster pad could sit
below 1,000).

GUI-only (MPU-specific pump data), no upstream library PR — like the plant
modules it unifies.
"""

import json
from glob import glob
from pathlib import Path
from typing import Iterable, Optional

_JP_DATA_DIR = Path(__file__).resolve().parent.parent / "jp_data"

_FT_TO_PSI_DIVISOR = 2.31  # ft of head -> psi at SG 1.0 (README/meta convention)

_CURVE_POINTS = 61  # samples per reported curve, both ends inclusive
_BPD_TO_GPM = 42.0 / 1440.0  # 42 gal/bbl over 1440 min/day

# woffl.assembly.network_optimizer.PowerFluidConstraint validates
# 1000 <= pressure <= 5000 psi at construction.
PF_CONSTRAINT_MIN_PSI = 1000.0
PF_CONSTRAINT_MAX_PSI = 5000.0


def clamp_to_pf_constraint(pressure: float) -> float:
    """Clamp a header pressure into PowerFluidConstraint's [1000, 5000] band."""
    return min(max(pressure, PF_CONSTRAINT_MIN_PSI), PF_CONSTRAINT_MAX_PSI)


def poly_eval(coeffs: dict, q: float) -> float:
    """Evaluate sum(c[i]*q**i) for c0..cN (arbitrary order).

    Verbatim the ``_poly`` the I-Pad and M-Pad modules each carried — the
    running-power accumulation is kept exactly so results stay bit-identical.
    """
    total, qp, i = 0.0, 1.0, 0
    while f"c{i}" in coeffs:
        total += coeffs[f"c{i}"] * qp
        qp *= q
        i += 1
    return total


class PadPlant:
    """Uniform interface over a pad's booster plant.

    Subclasses hold the pad-specific data and physics; the interface below is
    what the (Phase B) unified pad page / optimizer will consume:

    * ``header_at_flow(q_total, n_pumps)`` — delivered/deliverable PF header
      (psi) at a total station flow. ``None`` past the plant's capability.
    * ``budget_at_pressure(pressure, n_pumps)`` — max total PF (BPD) the plant
      can push at >= that header pressure (pressure-independent hydraulic
      ceiling for ``fixed_curve`` plants).
    * ``flow_window(n_pumps)`` — (lo, hi) recommended/feasible total-flow band.
    * ``pressure_window(n_pumps)`` — (floor, ceiling) header band for the
      optimizer sweep, always inside PowerFluidConstraint's [1000, 5000].
    * ``flags(q_total, n_pumps)`` — {in_range, recirc, over_capacity}; each
      plant computes the ones that apply, the others are False.
    * ``envelope(flows, n_pumps, at_pressure)`` — per-flow operating rows
      wrapping the legacy ``operating_envelope`` shapes (keys beyond ``flow``
      / ``max_discharge_psi`` / ``feasible`` / ``pumps`` are pad-specific).
    * ``curve_report(n_pumps)`` -- industry-format pump-curve payload: the
      station curve family + capability frontier, and the per-machine head /
      BHP / efficiency sheet with BEP, POR and AOR.
    """

    coupling: str = "free_pressure"  # or "fixed_curve"
    n_pump_options: list = []  # selectable online-pump counts; [] = fixed train
    max_header_psi: Optional[float] = None  # operational discharge cap
    # Which water stream this pad's machines handle and the optimizer budgets:
    # "lift_wat" (power fluid only - formation water passes to the plant) or
    # "totl_wat" (lift + formation - a full-POPS pad pump). Every pad is
    # "lift_wat" until its owner says otherwise (see well_sort_engine.
    # POPS_PUMP_HANDLES for the candidates; redesign doc §5).
    water_key: str = "lift_wat"

    def delivered_header(
        self,
        q_total: float,
        setpoint: Optional[float] = None,
        n_pumps: int | None = None,
    ) -> tuple[Optional[float], bool]:
        """Header the pad actually runs at for a total PF draw: SETPOINT
        below the knee, frontier above (docs/optimization_redesign_2026-09.md
        §3).

        Operators run a header setpoint with bypass / recirculation; the
        booster only fails to hold it when the frontier at that flow is
        below it. A fixed-speed station (``coupling == "fixed_curve"``) has
        no setpoint - its header follows the curve.

        Args:
            q_total: total PF draw, BPD
            setpoint: operator header setpoint, psi (None = the plant's
                operational cap ``max_header_psi``; None on a fixed-curve
                plant means "the curve")
            n_pumps: pumps online

        Returns:
            (header_psi, over_capacity): header None and over_capacity True
            when the frontier cannot carry the flow at all.
        """
        frontier = self.header_at_flow(q_total, n_pumps) if q_total > 0 else None
        if self.coupling == "fixed_curve":
            return (frontier, False) if (q_total <= 0 or frontier is not None) else (None, True)
        cap = setpoint if setpoint is not None else self.max_header_psi
        if q_total > 0 and frontier is None:
            # Either BELOW the machine's minimum flow (it recirculates and
            # holds the setpoint - E-Pad's unimodal frontier is None there)
            # or ABOVE its capability (over capacity).
            try:
                q_min = float(self.flow_window(n_pumps)[0])
            except Exception:  # noqa: BLE001 - a plant without a window
                q_min = 0.0
            if q_total < q_min and cap is not None:
                return float(cap), False
            return None, True
        if frontier is None:  # no draw: the plant holds its setpoint
            return cap, False
        if cap is None:
            return float(frontier), False
        return min(float(cap), float(frontier)), False

    # -- pad-specific: subclasses must implement ----------------------------

    def specific_gravity(self) -> float:
        raise NotImplementedError

    def header_at_flow(
        self, q_total: float, n_pumps: int | None = None
    ) -> Optional[float]:
        raise NotImplementedError

    def budget_at_pressure(self, pressure: float, n_pumps: int | None = None) -> float:
        raise NotImplementedError

    def flow_window(self, n_pumps: int | None = None) -> tuple[float, float]:
        raise NotImplementedError

    def pressure_window(self, n_pumps: int | None = None) -> tuple[float, float]:
        raise NotImplementedError

    def flags(self, q_total: float, n_pumps: int | None = None) -> dict:
        raise NotImplementedError

    def envelope(
        self,
        flows: Iterable[float],
        n_pumps: int | None = None,
        at_pressure: float | None = None,
    ) -> list[dict]:
        raise NotImplementedError

    def curve_report(self, n_pumps: int | None = None) -> dict:
        """Industry-format booster pump curves for this plant (JSON-safe).

        Args:
            n_pumps (int | None): online pump count; None resolves through
                ``_n`` to the plant's own default.

        Returns:
            dict: ``{pad, coupling, n_pumps, sg, suction_psi, max_header_psi,
                nameplate{}, station{curves, frontier, bep, por, aor,
                min_flow, header_cap}, pumps[]}``. Station curve points are
                ``[total_flow_bpd, discharge_psi]``; machine curve points are
                ``[flow_bpd, head_ft, bhp, eff_pct]`` at PER-PUMP flow. Plain
                dicts, lists, floats and strings only.

        Implemented by ``SPadPlant`` (fixed-speed parallel family),
        ``IPadPlant`` (iso-speed series train + amp frontier) and
        ``MPadPlant`` (iso-speed parallel VFD bank). ``FixedHeaderPlant`` has
        no pumps and does not implement it.
        """
        raise NotImplementedError

    # -- optimizer policy (R-1 Phase B: consumed by woffl.gui.pad_optimize) --
    #
    # Each hook is a page-level constant/derivation the S/I/M pages carried,
    # moved verbatim so the unified compute core reproduces every number.

    # User-facing message when the free_pressure sweep finds no feasible
    # header at all (the page shows it via st.error).
    infeasible_sweep_msg = (
        "No feasible header pressure found — the boosters couldn't deliver "
        "any well's power-fluid demand. Check the IPRs."
    )

    def suction_psi(self) -> float:
        """Reference suction/intake pressure (psi) — what the settled header
        collapses to when the demanded flow is past the plant's capability."""
        raise NotImplementedError

    def warm_start_psi(self, n_pumps: int | None = None) -> float:
        """Initial trial header for the fixed-point loops (each page's
        historical warm start). Default: fixed_curve plants start on the
        curve at 60% of the flow ceiling (the S-Pad rule); free_pressure
        plants at the sweep ceiling (I/M override with their own probes)."""
        if self.coupling == "fixed_curve":
            return self.header_at_flow(0.6 * self.flow_window(n_pumps)[1], n_pumps)
        return self.pressure_window(n_pumps)[1]

    def clamp_window(self, n_pumps: int | None = None) -> tuple[float, float]:
        """Band the fixed-point ITERATES are clamped into — distinct from
        ``pressure_window`` (the optimizer sweep band). Floor:
        PowerFluidConstraint's 1000 psi, or the plant suction when that sits
        higher (M-Pad's LP-held 1,400); ceiling: the operational cap when the
        plant has one, else the constraint's 5000."""
        return (
            clamp_to_pf_constraint(self.suction_psi()),
            (
                self.max_header_psi
                if self.max_header_psi is not None
                else PF_CONSTRAINT_MAX_PSI
            ),
        )

    def match_check_header(self, total_pf: float, n_pumps: int | None = None) -> float:
        """Header the pre-flight match check models the wells at, from the
        pad's measured total PF (warm start when there's no measured PF).
        Default reproduces the S-Pad page; I/M override verbatim."""
        header = self.header_at_flow(total_pf, n_pumps) if total_pf > 0 else None
        if header is None:
            header = self.warm_start_psi(n_pumps)
        lo, hi = self.clamp_window(n_pumps)
        return min(max(header, lo), hi)

    def match_check_budget_bpd(
        self, total_pf: float, n_pumps: int | None = None
    ) -> float:
        """Non-binding PF budget for the match-check constraint (the check
        reads specific pumps via get_pump_performance; it never optimizes)."""
        return self.flow_window(n_pumps)[1]

    # -- shared machinery ----------------------------------------------------

    def _n(self, n_pumps: int | None) -> Optional[int]:
        """Resolve an n_pumps argument: default = first n_pump_options entry
        (None for fixed trains, whose physics ignore it)."""
        if n_pumps is not None:
            return n_pumps
        return self.n_pump_options[0] if self.n_pump_options else None

    # one meta load per subclass (mirrors the legacy modules' lru_cache(1))
    _meta_cache: Optional[dict] = None

    def _load_meta(self) -> dict:
        raise NotImplementedError

    def _meta(self) -> dict:
        cls = type(self)
        if cls._meta_cache is None:
            cls._meta_cache = self._load_meta()
        return cls._meta_cache

    @staticmethod
    def _load_meta_glob(pump_dir: Path, label: str) -> dict:
        """Load the (single) ``*meta*.json`` under a pad's pump-data dir."""
        hits = sorted(glob(str(pump_dir / "*meta*.json")))
        if not hits:
            raise FileNotFoundError(f"no {label} pump meta json under {pump_dir}")
        with open(hits[0], "r", encoding="utf-8") as fh:
            return json.load(fh)

    # affinity laws + unit conversion (I/M carried these inline, identically)

    @staticmethod
    def _affinity_q60(flow_bpd: float, hz: float) -> float:
        """60-Hz-equivalent flow: Q60 = Q * 60 / Hz."""
        return flow_bpd * 60.0 / hz

    @staticmethod
    def _affinity_head(head_60hz: float, hz: float) -> float:
        """Head scales with speed squared."""
        return head_60hz * (hz / 60.0) ** 2

    @staticmethod
    def _affinity_bhp(bhp_60hz: float, hz: float) -> float:
        """Shaft power scales with speed cubed."""
        return bhp_60hz * (hz / 60.0) ** 3

    @staticmethod
    def _head_ft_to_psi(head_ft: float, sg: float) -> float:
        return head_ft * sg / _FT_TO_PSI_DIVISOR

    # -- curve reporting (head / BHP / efficiency sheets) --------------------

    @staticmethod
    def hydraulic_efficiency(
        q_bpd: float, head_ft: float, bhp: float, sg: float
    ) -> float:
        """Pump hydraulic efficiency (percent) at one point on a curve.

        ``eff = Q_gpm * H_ft * SG / (3960 * BHP)``, water horsepower over
        shaft horsepower.

        Args:
            q_bpd (float): flow through the machine (BPD).
            head_ft (float): head the machine makes at that flow (ft).
            bhp (float): shaft power at that flow (BHP).
            sg (float): the SG THE BHP CURVE WAS FIT AT, which is not always
                the SG used for the head-to-psi conversion. S and I fit BHP on
                water so they read at 1.0 (I converts psi at 1.04); M's
                datasheet power is the design SG 1.05 while its psi conversion
                uses the field 1.03. Only with those values do S and I
                reproduce the ``eff_pct`` / ``stage_eff_pct`` columns of the
                committed CSVs and M the Schlumberger datasheet efficiency.

        Returns:
            float: efficiency (percent, 0-100). 0.0 at zero flow and wherever
                the power curve is non-positive (efficiency is undefined
                there); clamped into [0, 100] everywhere else.
        """
        if q_bpd <= 0.0:
            return 0.0
        if bhp <= 0.0:
            return 0.0
        eff = 100.0 * (q_bpd * _BPD_TO_GPM) * head_ft * sg / (3960.0 * bhp)
        return min(max(eff, 0.0), 100.0)

    @staticmethod
    def _curve_grid(q_max: float, n: int = _CURVE_POINTS) -> list[float]:
        """``n`` evenly spaced flows (BPD) from shut-off (0) to ``q_max``,
        both ends inclusive."""
        return [q_max * i / (n - 1) for i in range(n)]

    @staticmethod
    def _por(bep: Optional[float]) -> Optional[list[float]]:
        """Preferred operating region [lo, hi] (BPD) about a BEP flow, 0.70x
        to 1.20x. None when there is no BEP to hang it on."""
        return None if not bep else [0.70 * bep, 1.20 * bep]

    @staticmethod
    def _grow_and_bisect(ok, lo: float, hi: float, cap: float) -> float:
        """Doubling + 50-step bisection inverse of a falling frontier.

        ``ok(q)`` must be monotone non-increasing in q (True below the
        frontier). Returns 0.0 when even ``lo`` fails; otherwise the largest
        q (capped at ``cap``) where ``ok`` holds. This one implementation
        reproduces both legacy ``max_flow_at_pressure`` loops (I-Pad started
        hi at 200, M-Pad at its recirc floor) bit-for-bit: below ``cap`` the
        doubling probes the same points, and at/past ``cap`` both collapse to
        bisecting the same [lo, cap] interval.
        """
        if not ok(lo):
            return 0.0
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


class SPadPlant(PadPlant):
    """S-Pad: 3x parallel fixed-speed Borets ESPi675TJ-12000 boosters.

    Same differential pressure across parallel pumps, flows add, so each
    online pump carries ``total_flow / n_online``; delivered header =
    suction + dP(per-pump flow). Fixed speed means the delivered pressure is
    a CURVE of flow (``coupling="fixed_curve"``) — the optimizer iterates
    flow -> pressure to a fixed point rather than choosing a pressure.

    Fit + provenance: ``woffl/jp_data/S_Pad_Pumps/`` (head-per-stage cubic,
    79 stages/unit, 3500 RPM, SG 1.0, ~220 psi suction; validated vs. SCADA
    within ~1%, 2026-06-15). Physics moved verbatim from ``s_pad_plant``.
    """

    coupling = "fixed_curve"
    n_pump_options = [3, 2]
    max_header_psi = None  # no operational cap; pages clamp to the PF band

    _META_PATH = _JP_DATA_DIR / "S_Pad_Pumps" / "pump_curve_meta.json"

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
        "bhp_per_stage_poly": {
            "c0": 5.49354336433901,
            "c1": 0.0003478302405478312,
            "c2": 6.361293991469625e-09,
            "c3": -3.505560158491784e-13,
        },
    }

    def _load_meta(self) -> dict:
        try:
            with open(self._META_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return self._FALLBACK

    # -- pad physics (verbatim from s_pad_plant) ----------------------------

    def n_pumps_installed(self) -> int:
        return int(self._meta().get("n_pumps_parallel", 3))

    def specific_gravity(self) -> float:
        return float(self._meta().get("specific_gravity", 1.0))

    def suction_psi(self) -> float:
        """Booster suction (PF separator side), from the meta — ~220 psi."""
        return float(self._meta().get("suction_psi", 220.0))

    def recommended_flow_per_pump(self) -> tuple[float, float]:
        lo, hi = self._meta().get("recommended_flow_per_pump_bpd", [7650, 18360])
        return float(lo), float(hi)

    def head_per_stage(self, q_per_pump_bpd: float) -> float:
        """Head (ft) produced by one stage at the given per-pump flow (BPD).

        Kept as the original explicit cubic (NOT poly_eval) — ``q**2``/``q**3``
        vs. running-power multiplication can differ in the last ulp, and the
        pins demand bit-identical numbers.
        """
        p = self._meta().get(
            "head_per_stage_poly", self._FALLBACK["head_per_stage_poly"]
        )
        q = q_per_pump_bpd
        return p["c0"] + p["c1"] * q + p["c2"] * q**2 + p["c3"] * q**3

    def bhp_per_stage(self, q_per_pump_bpd: float) -> float:
        """Shaft power (BHP) one stage draws at the given per-pump flow (BPD).

        Explicit cubic in the same style as ``head_per_stage`` (NOT poly_eval)
        so the two curve accessors stay term-for-term identical.
        """
        p = self._meta().get("bhp_per_stage_poly", self._FALLBACK["bhp_per_stage_poly"])
        q = q_per_pump_bpd
        return p["c0"] + p["c1"] * q + p["c2"] * q**2 + p["c3"] * q**3

    def discharge_pressure(
        self,
        total_flow_bpd: float,
        n_pumps: int = 3,
        *,
        sg: float | None = None,
        suction_psi: float | None = None,
    ) -> float:
        """Common header discharge pressure (psi) for a given total station flow.

        No capacity guard — past station_capacity it keeps evaluating the
        cubic (current behavior; callers gate on flow_in_range/station_capacity).
        """
        if n_pumps <= 0:
            raise ValueError("n_pumps must be >= 1")
        meta = self._meta()
        sg = float(meta.get("specific_gravity", 1.0)) if sg is None else sg
        suction = (
            float(meta.get("suction_psi", 220.0))
            if suction_psi is None
            else suction_psi
        )
        n_stages = int(meta.get("n_stages", 79))

        q_pp = total_flow_bpd / n_pumps
        dp = self._head_ft_to_psi(self.head_per_stage(q_pp) * n_stages, sg)
        return suction + dp

    def per_pump_flow(self, total_flow_bpd: float, n_pumps: int = 3) -> float:
        return total_flow_bpd / n_pumps

    def flow_in_range(self, total_flow_bpd: float, n_pumps: int = 3) -> bool:
        """True if per-pump flow sits inside the recommended (thrust) window."""
        lo, hi = self.recommended_flow_per_pump()
        q_pp = self.per_pump_flow(total_flow_bpd, n_pumps)
        return lo <= q_pp <= hi

    def station_capacity(self, n_pumps: int = 3) -> float:
        """Upper hydraulic flow ceiling (BPD) — recommended max per pump x n."""
        _lo, hi = self.recommended_flow_per_pump()
        return hi * n_pumps

    # -- uniform interface ----------------------------------------------------

    def header_at_flow(self, q_total: float, n_pumps: int | None = None) -> float:
        return self.discharge_pressure(q_total, self._n(n_pumps))

    def budget_at_pressure(self, pressure: float, n_pumps: int | None = None) -> float:
        # fixed-curve: delivered pressure follows the flow, so the budget is
        # the hydraulic (thrust) ceiling regardless of the requested pressure
        return self.station_capacity(self._n(n_pumps))

    def flow_window(self, n_pumps: int | None = None) -> tuple[float, float]:
        n = self._n(n_pumps)
        lo, hi = self.recommended_flow_per_pump()
        return lo * n, hi * n

    def pressure_window(self, n_pumps: int | None = None) -> tuple[float, float]:
        # delivered pressure falls with flow: the curve spans [P(flow hi),
        # P(flow lo)] over the thrust window, clamped into the PF band
        n = self._n(n_pumps)
        lo_q, hi_q = self.flow_window(n)
        return (
            clamp_to_pf_constraint(self.discharge_pressure(hi_q, n)),
            clamp_to_pf_constraint(self.discharge_pressure(lo_q, n)),
        )

    def flags(self, q_total: float, n_pumps: int | None = None) -> dict:
        n = self._n(n_pumps)
        return {
            "in_range": self.flow_in_range(q_total, n),
            "recirc": False,
            "over_capacity": q_total > self.station_capacity(n),
        }

    def envelope(
        self,
        flows: Iterable[float],
        n_pumps: int | None = None,
        at_pressure: float | None = None,
    ) -> list[dict]:
        # fixed-speed parallel station: no per-pump speed/amp state to report,
        # and no header cap — ``at_pressure`` is ignored
        n = self._n(n_pumps)
        return [
            {
                "flow": q,
                "max_discharge_psi": self.discharge_pressure(q, n),
                "per_pump_bpd": self.per_pump_flow(q, n),
                "feasible": q <= self.station_capacity(n),
                "in_range": self.flow_in_range(q, n),
                "pumps": [],
            }
            for q in flows
        ]

    # -- curve report ---------------------------------------------------------

    def curve_report(self, n_pumps: int | None = None) -> dict:
        """Station + machine pump curves for the S-Pad boosters.

        Args:
            n_pumps (int | None): online pump count (BPD physics is per-pump
                flow = total / n); None = the plant default.

        Returns:
            dict: the ``PadPlant.curve_report`` payload. The station family is
                one fixed-speed delivered-header curve per online-pump count;
                the single machine sheet is one 79-stage unit at 3,500 RPM.
        """
        meta = self._meta()
        n = self._n(n_pumps)
        installed = self.n_pumps_installed()
        # the family always highlights a curve the station actually has, even
        # if a caller asks for a count it does not
        n_on = min(max(int(n or installed), 1), installed)
        rec_lo, rec_hi = self.recommended_flow_per_pump()
        n_stages = int(meta.get("n_stages", 79))
        rpm = int(meta.get("speed_rpm", 3500))
        bep_pp = float(meta.get("bep_flow_per_pump_bpd", 12000))

        curves = []
        for count in range(1, installed + 1):
            # 1.12x the recommended max mirrors the Streamlit curve plot: it
            # carries the curve past the thrust window into up-thrust
            q_max = rec_hi * 1.12 * count
            curves.append(
                {
                    "label": f"{count} pump{'s' if count != 1 else ''}",
                    "n_pumps": count,
                    "hz": 60.0,
                    "active": count == n_on,
                    "points": [
                        [q, self.discharge_pressure(q, count)]
                        for q in self._curve_grid(q_max)
                    ],
                }
            )

        pump_pts = []
        for q in self._curve_grid(21000.0):  # the head poly's stated valid range
            head = n_stages * self.head_per_stage(q)
            bhp = n_stages * self.bhp_per_stage(q)
            # the stage BHP poly is a water fit, so efficiency reads at SG 1.0
            eff = self.hydraulic_efficiency(q, head, bhp, 1.0)
            pump_pts.append([q, head, bhp, eff])

        model = f"Weatherford/Borets ESPi675TJ-12000 (675 series), {n_stages} stg"
        return {
            "pad": "S",
            "coupling": self.coupling,
            "n_pumps": n,
            "sg": self.specific_gravity(),
            "suction_psi": self.suction_psi(),
            "max_header_psi": self.max_header_psi,
            "nameplate": {
                "equipment": "P-800 B-1/B-2/B-3",
                "model": model,
                "arrangement": f"{installed} x parallel, fixed speed",
                "speed": f"{rpm:,} RPM (60 Hz, no VFD)",
                "source": str(meta.get("source", "")),
                "validated": str(meta.get("validated", "")),
            },
            "station": {
                "curves": curves,
                # fixed speed: the curve family IS the capability
                "frontier": None,
                "bep": bep_pp * n_on,
                "por": self._por(bep_pp * n_on),
                "aor": [rec_lo * n_on, rec_hi * n_on],
                "min_flow": None,
                "header_cap": None,
            },
            "pumps": [
                {
                    "label": f"P-800 B-x - {n_stages} stg at {rpm:,} RPM",
                    "hz": 60.0,
                    "points": pump_pts,
                    "head_derated": None,
                    "derate_note": None,
                    "bep": bep_pp,
                    "por": self._por(bep_pp),
                    "aor": [rec_lo, rec_hi],
                    "min_flow": None,
                }
            ],
        }


class IPadPlant(PadPlant):
    """I-Pad: 2-pump SERIES VFD train — amp-limited delivered-pressure frontier.

    PF separator (~217 psig) -> P-1021 LP (26 stg) -> P-0901 HP (17 stg) ->
    wells. Both units share one Summit SN35000 per-stage curve; VFD-driven, so
    the real ceiling is motor amps (LP at its 192 A drive limit, HP at 154 A).
    At a fixed amp budget a pump trades head for flow — the capability is a
    falling frontier ``max_discharge_pressure(total_flow)`` the optimizer may
    ride on/under (``coupling="free_pressure"``). No arbitrary flow cap: the
    ceiling is wherever a pump can no longer pass the flow within amps.

        Q60   = Q_actual * 60 / Hz                      (affinity: flow ~ Hz)
        head  = n_stages * head_per_stage(Q60) * (Hz/60)^2
        dP    = head * SG / 2.31
        BHP   = n_stages * bhp_per_stage(Q60) * (Hz/60)^3   (water)
        amps  = k * BHP                                  (k live-calibrated)

    Validated to the psi / amp against live SCADA 2026-06-16. Physics moved
    verbatim from ``i_pad_plant``; loads ``woffl/jp_data/I_Pad_Pumps``.
    """

    coupling = "free_pressure"
    n_pump_options = []  # fixed LP+HP series train — no pump-count choice
    max_header_psi = 3500.0  # operational discharge cap (pages' _MAX_HEADER_PSI)

    infeasible_sweep_msg = (
        "No feasible header pressure found — the pumps couldn't deliver any "
        "well's power-fluid demand within their amp limits. Check the IPRs."
    )

    _PUMP_DIR = _JP_DATA_DIR / "I_Pad_Pumps"
    _HZ_MAX = 60.0  # curve is normalized to 60 Hz; drives run at/below it
    _HZ_FLOOR = 10.0  # never search below this regardless of flow
    # iso-speed lines the station panel draws (both units at the same speed)
    _CURVE_SPEEDS_HZ = (45.0, 50.0, 55.0, 60.0)

    def _load_meta(self) -> dict:
        return self._load_meta_glob(self._PUMP_DIR, "I-Pad")

    # -- pad physics (verbatim from i_pad_plant) ----------------------------

    def specific_gravity(self) -> float:
        return float(
            self._meta().get("compute", {}).get("specific_gravity_used_in_tables", 1.04)
        )

    def suction_psi(self) -> float:
        """Train intake (LP suction) — the PF separator pressure."""
        lp = self._meta()["pumps"]["P-1021_LP_Booster"]
        return float(lp["live_operating_point_2026-06-16"]["intake_psig"])

    def max_valid_flow(self) -> float:
        return float(self._meta()["stage"].get("max_valid_flow_bpd", 60000.0))

    def head_per_stage(self, q60_bpd: float) -> float:
        """Head (ft) per stage at 60 Hz for the given (60-Hz-equivalent) flow."""
        return poly_eval(self._meta()["stage"]["head_per_stage_poly"], q60_bpd)

    def bhp_per_stage(self, q60_bpd: float) -> float:
        """Water BHP per stage at 60 Hz for the given (60-Hz-equivalent) flow."""
        return poly_eval(self._meta()["stage"]["bhp_per_stage_poly"], q60_bpd)

    def pumps(self) -> list[dict]:
        """[LP, HP] in series order, each {name, n_stages, amp_limit, k}."""
        m = self._meta()["pumps"]
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

    def pump_dP(
        self, n_stages: int, flow_bpd: float, hz: float, sg: float | None = None
    ) -> float:
        """Differential pressure (psi) a pump makes at a flow + speed (affinity)."""
        sg = self.specific_gravity() if sg is None else sg
        q60 = self._affinity_q60(flow_bpd, hz)
        head = self._affinity_head(n_stages * self.head_per_stage(q60), hz)
        return self._head_ft_to_psi(head, sg)

    def pump_amps(self, k: float, n_stages: int, flow_bpd: float, hz: float) -> float:
        """Motor amps a pump draws at a flow + speed (affinity, live-calibrated k)."""
        q60 = self._affinity_q60(flow_bpd, hz)
        return self._affinity_bhp(k * n_stages * self.bhp_per_stage(q60), hz)

    def hz_at_amp_limit(self, pump: dict, flow_bpd: float) -> Optional[float]:
        """Speed (Hz, <= 60) at which this pump hits its amp limit at the given
        flow, or None if the pump physically can't pass that flow within its
        amp limit.

        The speed range is floored where the 60-Hz-equivalent flow reaches the
        pump's max curve flow — below that the poly extrapolates into nonsense.
        Amps rise monotonically with speed across the valid range, so we bisect
        for amps == limit.
        """
        k, n, lim = pump["k"], pump["n_stages"], pump["amp_limit"]
        hz_min = max(self._HZ_FLOOR, flow_bpd * 60.0 / self.max_valid_flow())
        if hz_min >= self._HZ_MAX:
            return None  # flow exceeds the curve's max valid flow even at 60 Hz
        if self.pump_amps(k, n, flow_bpd, self._HZ_MAX) <= lim:
            return self._HZ_MAX  # not amp-limited here — capped by max speed
        if self.pump_amps(k, n, flow_bpd, hz_min) > lim:
            return None  # over the limit even at minimum head — flow too high
        lo, hi = hz_min, self._HZ_MAX
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if self.pump_amps(k, n, flow_bpd, mid) > lim:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    def pump_max_dP(
        self, pump: dict, flow_bpd: float, sg: float | None = None
    ) -> Optional[float]:
        """Max dP (psi) this pump can deliver at the given flow within its amp
        limit, or None if the flow isn't deliverable."""
        hz = self.hz_at_amp_limit(pump, flow_bpd)
        if hz is None:
            return None
        return self.pump_dP(pump["n_stages"], flow_bpd, hz, sg)

    def max_discharge_pressure(
        self, total_flow_bpd: float, sg: float | None = None
    ) -> Optional[float]:
        """Highest header (HP-discharge) pressure the series train can deliver
        at the given total PF flow, both pumps at their amp limits. Returns
        None past the amp-limited flow ceiling (no arbitrary hard cap)."""
        sg = self.specific_gravity() if sg is None else sg
        dps = [self.pump_max_dP(p, total_flow_bpd, sg) for p in self.pumps()]
        if any(d is None for d in dps):
            return None
        return self.suction_psi() + sum(dps)

    def max_flow_at_pressure(self, pressure: float, sg: float | None = None) -> float:
        """Largest total PF flow the train can deliver at >= ``pressure`` (the
        frontier inverted) — the optimizer's PF budget at a candidate header.
        Returns 0.0 if even minimal flow can't reach the pressure."""

        def ok(q: float) -> bool:
            f = self.max_discharge_pressure(q, sg)
            return f is not None and f >= pressure

        return self._grow_and_bisect(ok, 100.0, 200.0, 4.0 * self.max_valid_flow())

    def operating_envelope(self, flows_bpd) -> list[dict]:
        """For a sweep of total flows, the frontier pressure + each pump's
        limiting speed/amps. ``feasible`` is False past the amp-limited flow
        ceiling."""
        sg = self.specific_gravity()
        rows = []
        for q in flows_bpd:
            pumps, feasible = [], True
            for p in self.pumps():
                hz = self.hz_at_amp_limit(p, q)
                if hz is None:
                    feasible = False
                    pumps.append(
                        {
                            "name": p["name"],
                            "hz": None,
                            "dP": None,
                            "amps": None,
                            "amp_limit": p["amp_limit"],
                        }
                    )
                    continue
                pumps.append(
                    {
                        "name": p["name"],
                        "hz": hz,
                        "dP": self.pump_dP(p["n_stages"], q, hz, sg),
                        "amps": self.pump_amps(p["k"], p["n_stages"], q, hz),
                        "amp_limit": p["amp_limit"],
                    }
                )
            rows.append(
                {
                    "flow": q,
                    "max_discharge_psi": (
                        (self.suction_psi() + sum(p["dP"] for p in pumps))
                        if feasible
                        else None
                    ),
                    "pumps": pumps,
                    "feasible": feasible,
                    "amp_limited": feasible
                    and any(p["hz"] < self._HZ_MAX - 0.01 for p in pumps),
                }
            )
        return rows

    # -- uniform interface ----------------------------------------------------

    def header_at_flow(
        self, q_total: float, n_pumps: int | None = None
    ) -> Optional[float]:
        return self.max_discharge_pressure(q_total)  # fixed train — n_pumps N/A

    def budget_at_pressure(self, pressure: float, n_pumps: int | None = None) -> float:
        return self.max_flow_at_pressure(pressure)

    def warm_start_psi(self, n_pumps: int | None = None) -> float:
        # the page's historical warm start: the frontier near shut-in flow
        return self.max_discharge_pressure(6000.0) or 3400.0

    def match_check_header(self, total_pf: float, n_pumps: int | None = None) -> float:
        header = self.max_discharge_pressure(total_pf) if total_pf > 0 else None
        if header is None:
            # no current PF, or beyond frontier — model at the live setpoint
            header = self.max_discharge_pressure(8000.0) or 3400.0
        # Cap at the operational discharge limit — every other I-Pad path
        # does. Uncapped, a small measured total PF put the frontier at
        # ~4,400-4,750 psi and produced spurious "✗ BUST" verdicts (P0-7).
        return min(self.max_header_psi, header)

    def match_check_budget_bpd(
        self, total_pf: float, n_pumps: int | None = None
    ) -> float:
        return max(total_pf * 1.5, 60000.0)

    def flow_window(self, n_pumps: int | None = None) -> tuple[float, float]:
        # no recirc floor modeled; ceiling = throughput at barely-above-suction
        # (the page's eval cap: max_flow_at_pressure(suction + 200))
        return 0.0, self.max_flow_at_pressure(self.suction_psi() + 200.0)

    def pressure_window(self, n_pumps: int | None = None) -> tuple[float, float]:
        # the page sweep's band: a lift floor above suction up to the
        # operational cap (the frontier at low flow can sit above it)
        floor = max(self.suction_psi() + 300.0, 1600.0, PF_CONSTRAINT_MIN_PSI)
        ceiling = clamp_to_pf_constraint(
            min(self.max_header_psi, self.max_discharge_pressure(6000.0) or 4000.0)
        )
        if ceiling <= floor:
            ceiling = floor + 500.0
        return floor, ceiling

    def flags(self, q_total: float, n_pumps: int | None = None) -> dict:
        over = self.max_discharge_pressure(q_total) is None
        return {"in_range": not over, "recirc": False, "over_capacity": over}

    def envelope(
        self,
        flows: Iterable[float],
        n_pumps: int | None = None,
        at_pressure: float | None = None,
    ) -> list[dict]:
        # frontier report — the I envelope has no header-cap or pump-count
        # concept, so ``n_pumps`` / ``at_pressure`` are ignored
        return self.operating_envelope(flows)

    # -- curve report ---------------------------------------------------------

    def curve_report(self, n_pumps: int | None = None) -> dict:
        """Station + machine pump curves for the I-Pad series train.

        Args:
            n_pumps (int | None): carried into the payload only -- the LP+HP
                train is fixed, so the physics ignores it.

        Returns:
            dict: the ``PadPlant.curve_report`` payload. The station family is
                one iso-speed line per ``_CURVE_SPEEDS_HZ`` entry with BOTH
                units at that speed; the frontier is the amp-limited
                capability the optimizer rides. Two machine sheets, LP then HP.
        """
        meta = self._meta()
        stage = meta["stage"]
        sg = self.specific_gravity()
        suction = self.suction_psi()
        pumps = self.pumps()
        q_valid = self.max_valid_flow()
        bep = float(stage["bep_flow_bpd"])
        aor = [float(q) for q in stage["recommended_operating_range_bpd"]]
        min_flow = float(stage["min_continuous_flow_bpd"])

        curves = []
        for hz in self._CURVE_SPEEDS_HZ:
            # affinity: the stage poly is only valid out to max_valid_flow in
            # 60-Hz-equivalent flow, so the ceiling scales with speed
            points = []
            for q in self._curve_grid(q_valid * hz / 60.0):
                dp = sum(self.pump_dP(p["n_stages"], q, hz, sg) for p in pumps)
                points.append([q, suction + dp])
            curves.append(
                {
                    "label": f"{hz:.0f} Hz (both units)",
                    "n_pumps": None,
                    "hz": hz,
                    "active": hz == self._HZ_MAX,
                    "points": points,
                }
            )

        lp_amps, hp_amps = pumps[0]["amp_limit"], pumps[1]["amp_limit"]
        front_pts = []
        for q in self._curve_grid(self.flow_window()[1]):
            psi = self.max_discharge_pressure(q, sg)
            if psi is not None:  # past the amp ceiling the train has no point
                front_pts.append([q, psi])

        machines = []
        for p in pumps:
            pump_pts = []
            for q in self._curve_grid(q_valid):
                head = p["n_stages"] * self.head_per_stage(q)
                bhp = p["n_stages"] * self.bhp_per_stage(q)
                # the stage BHP poly is a water fit, so efficiency reads at
                # SG 1.0 even though the psi conversion uses 1.04
                eff = self.hydraulic_efficiency(q, head, bhp, 1.0)
                pump_pts.append([q, head, bhp, eff])
            machines.append(
                {
                    "label": f"{p['name']} - {p['n_stages']} stg at 60 Hz",
                    "hz": 60.0,
                    "points": pump_pts,
                    "head_derated": None,
                    "derate_note": None,
                    # one shared stage curve, so both units carry it
                    "bep": bep,
                    "por": self._por(bep),
                    "aor": list(aor),
                    "min_flow": min_flow,
                }
            )

        lp_stg, hp_stg = pumps[0]["n_stages"], pumps[1]["n_stages"]
        model = f"Summit ESP SN35000 XRC CCW (950 series), {lp_stg} stg + {hp_stg} stg"
        rpm = int(stage.get("ref_speed_rpm", 3500))
        return {
            "pad": "I",
            "coupling": self.coupling,
            "n_pumps": self._n(n_pumps),
            "sg": sg,
            "suction_psi": suction,
            "max_header_psi": self.max_header_psi,
            "nameplate": {
                "equipment": f"{pumps[0]['name']} / {pumps[1]['name']}",
                "model": model,
                "arrangement": "2 x series (LP feeds HP), VFD",
                "speed": f"{rpm:,} RPM at 60 Hz (VFD)",
                "source": str(stage.get("source", "")),
                "validated": str(meta.get("validated", "")),
            },
            "station": {
                "curves": curves,
                "frontier": {
                    "label": f"Amp limit (LP {lp_amps:.0f} A / HP {hp_amps:.0f} A)",
                    # same shape as a station curve: the capability bound is
                    # not an operating line, and each unit rides its own
                    # amp-limited speed, so n_pumps/hz/active stay empty
                    "n_pumps": None,
                    "hz": None,
                    "active": False,
                    "points": front_pts,
                },
                # series train: one flow through both units, nothing to scale
                "bep": bep,
                "por": self._por(bep),
                "aor": list(aor),
                "min_flow": min_flow,
                "header_cap": self.max_header_psi,
            },
            "pumps": machines,
        }


class MPadPlant(PadPlant):
    """M-Pad (Moose Pad, Mod 42): 3x parallel VFD REDA M675 HP bank.

    Hybrid station: 3x P-4220 LP (parallel) hold a ~1,400 psig header that
    feeds 3x P-4230 HP (parallel) up to the ~3,500 psig PF header. v1 models
    the HP bank only, on a fixed LP-held suction. Parallel bank: n online
    pumps each carry total/n and share one dP. Amps have big headroom — the
    binding limits are MIN-FLOW (recirculation shutdown) and VFD max speed —
    and installed head is derated by ``field_head_factor`` (~0.91 wear; amps
    stay on the as-new curve). Polynomials are TOTAL-pump (not per-stage).

    ``coupling="free_pressure"``: capability frontier + inverse, like I-Pad.
    Validated to the psi against live SCADA 2026-06-16. Physics moved verbatim
    from ``m_pad_plant``; loads ``woffl/jp_data/M_Pad_Pumps``.
    """

    coupling = "free_pressure"
    # M-Pad pumps handle formation AND lift water (Scott, 2026-09-02;
    # well_sort_engine.POPS_PUMP_HANDLES["M"] == "total"): budget and
    # price TOTAL water. See docs/optimization_redesign_2026-09.md section 5.
    water_key = "totl_wat"
    n_pump_options = [3, 2, 1]
    max_header_psi = 3500.0  # PF-header discharge cap (PIC-4231 setpoint)

    infeasible_sweep_msg = (
        "No feasible header pressure found — the HP bank couldn't deliver any "
        "well's power-fluid demand. Check the IPRs and pump count."
    )

    _PUMP_DIR = _JP_DATA_DIR / "M_Pad_Pumps"
    _HP_SUCTION_DEFAULT = 1400.0  # LP-held header (PIC-4221 setpoint)
    # iso-speed lines the station panel draws. The B_HP_4230 datasheet
    # freq_range_hz is [51, 61], but the bank is not operated above nominal
    # speed - the model's max (and the top drawn line) is capped at 60 Hz.
    _HZ_OPERATING_MAX = 60.0
    _CURVE_SPEEDS_HZ = (51.0, 55.0, 58.0, 60.0)

    def _load_meta(self) -> dict:
        return self._load_meta_glob(self._PUMP_DIR, "Moose Pad")

    # -- pad physics (verbatim from m_pad_plant) ----------------------------

    def specific_gravity(self) -> float:
        return float(
            self._meta().get("compute", {}).get("specific_gravity_field", 1.03)
        )

    def wear_factor(self) -> float:
        """Field head derate (~0.91): installed pumps make less head than as-new."""
        return float(self._meta().get("field_head_factor", {}).get("value", 0.91))

    def hp_suction_psi(self) -> float:
        return self._HP_SUCTION_DEFAULT

    def hp(self) -> dict:
        p = self._meta()["pumps"]["B_HP_4230"]
        lo, hi = p["recommended_flow_bpd_60hz"]
        return {
            "name": "P-4230 HP",
            "head_poly": p["head_ft_poly_60hz"],
            "bhp_poly": p["bhp_poly_60hz"],
            "amp_limit": float(self._meta()["motor_current_limits_A"]["trip"]),
            "k": float(p["amps_per_hp_est"]),
            "rec_lo": float(lo),  # per-pump min recommended (recirc floor) at 60 Hz
            "rec_hi": float(
                hi
            ),  # per-pump max recommended (off-curve ceiling) at 60 Hz
            "hz_max": min(
                self._HZ_OPERATING_MAX, float(p["freq_range_hz"][1])
            ),
            "n_default": 3,
        }

    @staticmethod
    def _head_ft(pump: dict, q60: float) -> float:
        return poly_eval(pump["head_poly"], q60)

    @staticmethod
    def _bhp(pump: dict, q60: float) -> float:
        return poly_eval(pump["bhp_poly"], q60)

    def pump_boost(
        self,
        q_per_pump: float,
        hz: float,
        *,
        apply_wear: bool = True,
        sg: float | None = None,
    ) -> float:
        """HP-pump differential pressure (psi) at a per-pump flow + speed.
        Head is field-derated by the wear factor; apply_wear=False for as-new."""
        sg = self.specific_gravity() if sg is None else sg
        hp = self.hp()
        q60 = self._affinity_q60(q_per_pump, hz)
        head = self._affinity_head(self._head_ft(hp, q60), hz)
        if apply_wear:
            head *= self.wear_factor()
        return self._head_ft_to_psi(head, sg)

    def pump_amps(self, q_per_pump: float, hz: float) -> float:
        """HP-pump motor amps at a per-pump flow + speed (as-new BHP, no wear)."""
        hp = self.hp()
        q60 = self._affinity_q60(q_per_pump, hz)
        return self._affinity_bhp(hp["k"] * self._bhp(hp, q60), hz)

    def hp_recommended_flow_per_pump(self) -> tuple[float, float]:
        hp = self.hp()
        return hp["rec_lo"], hp["rec_hi"]

    def min_total_flow(self, n_pumps: int = 3) -> float:
        """Recirculation floor: below this total PF the HP pumps drop under
        their min recommended per-pump flow (high-diff / low-flow shutdown)."""
        return self.hp()["rec_lo"] * n_pumps

    def max_total_flow(self, n_pumps: int = 3) -> float:
        """Off-curve ceiling: above this the per-pump flow exceeds the curve range."""
        return self.hp()["rec_hi"] * n_pumps

    def hz_for_boost(
        self, q_per_pump: float, target_boost: float, *, sg: float | None = None
    ):
        """Speed (Hz, <= hz_max) at which the HP pump makes ``target_boost`` at
        the given per-pump flow, or None if it can't reach it even at max
        speed. Boost rises with speed, so we bisect."""
        hp = self.hp()
        hz_max = hp["hz_max"]
        if self.pump_boost(q_per_pump, hz_max, sg=sg) < target_boost:
            return None
        lo, hi = 20.0, hz_max
        if self.pump_boost(q_per_pump, lo, sg=sg) > target_boost:
            return lo
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if self.pump_boost(q_per_pump, mid, sg=sg) > target_boost:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    def max_discharge_pressure(
        self, total_flow_bpd: float, n_pumps: int = 3, sg: float | None = None
    ) -> Optional[float]:
        """Highest HP-discharge (PF header) pressure the bank can deliver at
        the given total PF flow, all n pumps at max speed. The page caps it at
        the operational limit (3,500). None if the per-pump flow is off-curve."""
        if n_pumps <= 0 or total_flow_bpd <= 0:
            return None
        q_pp = total_flow_bpd / n_pumps
        if (
            q_pp > self.hp()["rec_hi"] * 1.05
        ):  # a touch past recommended max = off curve
            return None
        return self.hp_suction_psi() + self.pump_boost(q_pp, self.hp()["hz_max"], sg=sg)

    def max_flow_at_pressure(
        self, pressure: float, n_pumps: int = 3, sg: float | None = None
    ) -> float:
        """Largest total PF the bank can deliver at >= ``pressure`` (frontier
        inverse), for the optimizer's PF budget at a candidate header."""

        def ok(q: float) -> bool:
            f = self.max_discharge_pressure(q, n_pumps, sg)
            return f is not None and f >= pressure

        lo = max(self.min_total_flow(n_pumps), 100.0)
        return self._grow_and_bisect(ok, lo, lo, self.max_total_flow(n_pumps))

    def operating_envelope(
        self, flows_bpd, n_pumps: int = 3, *, header_cap: float = 3500.0
    ) -> list[dict]:
        """For a sweep of total flows, the deliverable header (capability
        capped at ``header_cap``) + the HP pumps' speed/amps/headroom and a
        recirc flag. At each flow the pumps run the speed that holds the
        capped header (or max speed if they can't reach it)."""
        hp = self.hp()
        sg = self.specific_gravity()
        rows = []
        for q in flows_bpd:
            q_pp = q / n_pumps if n_pumps else 0.0
            cap_pressure = self.max_discharge_pressure(q, n_pumps, sg)
            if cap_pressure is None:
                rows.append(
                    {
                        "flow": q,
                        "max_discharge_psi": None,
                        "feasible": False,
                        "recirc": q < self.min_total_flow(n_pumps),
                        "pumps": [],
                    }
                )
                continue
            header = min(header_cap, cap_pressure)
            target_boost = header - self.hp_suction_psi()
            hz = self.hz_for_boost(q_pp, target_boost, sg=sg) or hp["hz_max"]
            amps = self.pump_amps(q_pp, hz)
            rows.append(
                {
                    "flow": q,
                    "max_discharge_psi": header,
                    "per_pump_bpd": q_pp,
                    "feasible": True,
                    "recirc": q < self.min_total_flow(n_pumps),
                    "speed_capped": cap_pressure < header_cap,
                    "pumps": [
                        {
                            "name": hp["name"],
                            "n": n_pumps,
                            "hz": hz,
                            "amps": amps,
                            "amp_limit": hp["amp_limit"],
                        }
                    ],
                }
            )
        return rows

    # -- uniform interface ----------------------------------------------------

    def header_at_flow(
        self, q_total: float, n_pumps: int | None = None
    ) -> Optional[float]:
        return self.max_discharge_pressure(q_total, self._n(n_pumps))

    def budget_at_pressure(self, pressure: float, n_pumps: int | None = None) -> float:
        return self.max_flow_at_pressure(pressure, self._n(n_pumps))

    def suction_psi(self) -> float:
        return self.hp_suction_psi()

    def warm_start_psi(self, n_pumps: int | None = None) -> float:
        return float(self.max_header_psi)  # the PIC-4231 setpoint

    def match_check_header(self, total_pf: float, n_pumps: int | None = None) -> float:
        # the page's _frontier_header(total, cap, n): the operational cap when
        # there's no measured PF, the LP-held suction when the bank can't push
        # the measured PF at any pressure
        if total_pf <= 0:
            return float(self.max_header_psi)
        p = self.max_discharge_pressure(total_pf, self._n(n_pumps))
        return self.hp_suction_psi() if p is None else min(self.max_header_psi, p)

    def match_check_budget_bpd(
        self, total_pf: float, n_pumps: int | None = None
    ) -> float:
        return max(total_pf * 1.5, 80000.0)

    def flow_window(self, n_pumps: int | None = None) -> tuple[float, float]:
        n = self._n(n_pumps)
        return self.min_total_flow(n), self.max_total_flow(n)

    def pressure_window(self, n_pumps: int | None = None) -> tuple[float, float]:
        # the page sweep's band: floor above the LP-held suction, ceiling =
        # frontier just above the recirc floor, capped at the operational limit
        n = self._n(n_pumps)
        floor = max(self.hp_suction_psi() + 300.0, 1600.0, PF_CONSTRAINT_MIN_PSI)
        top = self.max_discharge_pressure(self.min_total_flow(n) * 1.2, n)
        ceiling = clamp_to_pf_constraint(
            min(self.max_header_psi, top or self.max_header_psi)
        )
        if ceiling <= floor:
            ceiling = floor + 500.0
        return floor, ceiling

    def flags(self, q_total: float, n_pumps: int | None = None) -> dict:
        n = self._n(n_pumps)
        over = q_total > 0 and self.max_discharge_pressure(q_total, n) is None
        recirc = q_total < self.min_total_flow(n)
        return {
            "in_range": not over and not recirc,
            "recirc": recirc,
            "over_capacity": over,
        }

    def envelope(
        self,
        flows: Iterable[float],
        n_pumps: int | None = None,
        at_pressure: float | None = None,
    ) -> list[dict]:
        cap = self.max_header_psi if at_pressure is None else at_pressure
        return self.operating_envelope(flows, self._n(n_pumps), header_cap=cap)

    # -- curve report ---------------------------------------------------------

    def curve_report(self, n_pumps: int | None = None) -> dict:
        """Station + machine pump curves for the M-Pad HP bank.

        Args:
            n_pumps (int | None): online HP pumps; None = the plant default.

        Returns:
            dict: the ``PadPlant.curve_report`` payload. The station family is
                one iso-speed line per ``_CURVE_SPEEDS_HZ`` entry at the
                resolved pump count, WEAR-DERATED because that is the curve
                the optimizer's frontier rides. The machine sheet is as-new
                with the worn head carried alongside in ``head_derated``.
        """
        meta = self._meta()
        hp = self.hp()
        n = self._n(n_pumps)
        n_on = max(int(n or hp["n_default"]), 1)
        sg = self.specific_gravity()
        wear = self.wear_factor()
        suction = self.hp_suction_psi()
        b = meta["pumps"]["B_HP_4230"]
        stages = int(b.get("stages", 41))
        bep_pp = float(b["bep_flow_bpd_60hz"])
        rec_lo, rec_hi = self.hp_recommended_flow_per_pump()

        curves = []
        for hz in self._CURVE_SPEEDS_HZ:
            points = []
            for q in self._curve_grid(rec_hi * (hz / 60.0) * n_on):
                # pump_boost applies the wear derate by default: the family is
                # field-representative, not the as-new catalog curve
                points.append([q, suction + self.pump_boost(q / n_on, hz, sg=sg)])
            curves.append(
                {
                    "label": f"{n_on} pump{'s' if n_on != 1 else ''} at {hz:.0f} Hz",
                    "n_pumps": n_on,
                    "hz": hz,
                    "active": hz == hp["hz_max"],
                    "points": points,
                }
            )

        # the machine sheet is the vendor curve: as-new head, datasheet power,
        # and efficiency at the SG the BHP poly was fit at (design 1.05)
        sg_design = float(meta.get("compute", {}).get("specific_gravity_design", 1.05))
        pump_pts, derated = [], []
        for q in self._curve_grid(rec_hi):
            head = self._head_ft(hp, q)
            bhp = self._bhp(hp, q)
            eff = self.hydraulic_efficiency(q, head, bhp, sg_design)
            pump_pts.append([q, head, bhp, eff])
            derated.append([q, head * wear])

        freq_lo, freq_hi = b["freq_range_hz"]
        nominal_hz = int(b.get("nominal_freq_hz", 60))
        return {
            "pad": "M",
            "coupling": self.coupling,
            "n_pumps": n,
            "sg": sg,
            "suction_psi": suction,
            "max_header_psi": self.max_header_psi,
            "nameplate": {
                "equipment": str(b.get("tags", "P-4230A / P-4230B / P-4230C")),
                "model": f"Schlumberger REDA M675-A, 862 series, {stages} stg",
                "arrangement": (
                    f"{hp['n_default']} x parallel VFD, on the LP-held "
                    f"{suction:,.0f} psig header"
                ),
                "speed": (
                    f"{int(freq_lo)} to {int(freq_hi)} Hz (VFD), "
                    f"nominal {nominal_hz} Hz, "
                    f"operated at <= {int(hp['hz_max'])} Hz"
                ),
                "source": str(meta.get("manufacturer", "")) + " data sheet"
                " + Mod 42 HPS set-up sheets",
                "validated": str(meta.get("validated", "")),
            },
            "station": {
                # the max-speed iso-speed line already IS the capability
                "frontier": None,
                "curves": curves,
                "bep": bep_pp * n_on,
                "por": self._por(bep_pp * n_on),
                "aor": [rec_lo * n_on, rec_hi * n_on],
                "min_flow": self.min_total_flow(n_on),
                "header_cap": self.max_header_psi,
            },
            "pumps": [
                {
                    "label": f"P-4230 HP - REDA M675-A, {stages} stg at 60 Hz",
                    "hz": 60.0,
                    "points": pump_pts,
                    "head_derated": derated,
                    "derate_note": f"field derated x{wear:.2f} (wear)",
                    "bep": bep_pp,
                    "por": self._por(bep_pp),
                    "aor": [rec_lo, rec_hi],
                    "min_flow": rec_lo,
                }
            ],
        }


class FixedHeaderPlant(PadPlant):
    """Null plant for a pad with no booster model: constant delivered header,
    unbounded PF budget, no operating flags. Lets the (Phase B) unified pad
    page run any pad by just naming its PF header pressure."""

    coupling = "fixed_curve"
    n_pump_options = []
    max_header_psi = None

    def __init__(self, header_psi: float):
        self.header_psi = float(header_psi)

    def specific_gravity(self) -> float:
        return 1.0

    def suction_psi(self) -> float:
        return self.header_psi  # constant header — nothing collapses anywhere

    def clamp_window(self, n_pumps: int | None = None) -> tuple[float, float]:
        return (PF_CONSTRAINT_MIN_PSI, PF_CONSTRAINT_MAX_PSI)

    def header_at_flow(self, q_total: float, n_pumps: int | None = None) -> float:
        return self.header_psi

    def budget_at_pressure(self, pressure: float, n_pumps: int | None = None) -> float:
        return float("inf")

    def flow_window(self, n_pumps: int | None = None) -> tuple[float, float]:
        return 0.0, float("inf")

    def pressure_window(self, n_pumps: int | None = None) -> tuple[float, float]:
        p = clamp_to_pf_constraint(self.header_psi)
        return p, p

    def flags(self, q_total: float, n_pumps: int | None = None) -> dict:
        return {"in_range": True, "recirc": False, "over_capacity": False}

    def envelope(
        self,
        flows: Iterable[float],
        n_pumps: int | None = None,
        at_pressure: float | None = None,
    ) -> list[dict]:
        return [
            {
                "flow": q,
                "max_discharge_psi": self.header_psi,
                "per_pump_bpd": None,
                "feasible": True,
                "pumps": [],
            }
            for q in flows
        ]
