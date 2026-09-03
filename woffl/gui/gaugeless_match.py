"""Gaugeless test match: infer a well's flowing BHP from its power-fluid rate.

A well test without a downhole gauge still carries a pressure measurement.
The nozzle passes power fluid in proportion to the square root of the
pressure drop across it, so the test's PF rate at the test's PF pressure
pins the throat-entry pressure, which is the suction BHP within the
entrance loss. This module turns that into a fit:

* the IPR is anchored on the test's OWN oil rate at a trial ``pwf``;
* the throat and diffuser coefficients (``kth``, ``kdi``) are free, exactly
  as in the single-point BHP match (``fric_calibration``);
* Nelder-Mead moves ``(pwf, kth, kdi)`` to minimize the root-mean-square
  fractional error of modeled oil and modeled PF rate against the test.

The anchor passes through ``(oil_test, pwf)`` by construction, so the oil
target is met exactly when the solved suction equals the anchor ``pwf``;
the PF target is met when that suction is the one the nozzle equation
implies. Two targets, one identifiable pressure, and a pair of discharge
coefficients that are only identifiable in combination (the same limit the
gauge-based single-point match has).

Two things one test cannot separate, reported as caveats, never hidden:

* a worn nozzle passes more PF at the same drop, so with the nozzle held at
  its catalog area a washed-out nozzle reads as a LOW inferred BHP;
* a choked throat (sonic) makes ``kth``/``kdi`` inert - they come back at
  their seeds with a message, like ``fric_calibration``'s pinned branch;
* when the test's PF rate lies outside what the catalog nozzle can pass at
  ANY BHP at the test's PF pressure (``pf_reachable`` False), the BHP is
  NOT identified: the PF target would only drag it onto a rail. The result
  is capped at "poor", keeps the closest point, says what to check
  (test-day PF pressure, nozzle wear, pump identity) and reports the
  nozzle AREA factor that would explain the gap (``area_factor_needed``).

The forward model is the SAME ``jetpump_solver`` the Solver page runs, so a
"good" match here reproduces on the page bit-for-bit. Nothing is written:
the caller lays the result over the sidebar and an explicit save keeps it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, sqrt
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
from scipy.optimize import minimize

from woffl.assembly.solopump import jetpump_solver
from woffl.geometry.jetpump import JetPump
from woffl.gui.fric_calibration import KDI_BOUNDS, KTH_BOUNDS

if TYPE_CHECKING:  # the factory's contract, not a runtime import
    from woffl.flow.inflow import InFlow

# The anchor BHP search window: never below the floor (a Vogel curve with
# pwf near zero is a vertical line), never within the margin of reservoir
# pressure (zero drawdown makes the anchor rate unreachable).
PWF_FLOOR = 100.0
PWF_PRES_MARGIN = 25.0

# Seed scan over the BHP window at the caller's discharge coefficients.
SCAN_POINTS = 7
SCAN_LO_FRAC = 0.20
SCAN_HI_FRAC = 0.95

# RMS fractional error grades (2 % / 5 %).
GOOD_FRAC = 0.02
FAIR_FRAC = 0.05
MULTISTART_THRESHOLD = FAIR_FRAC

# Discharge-coefficient starts tried when the neutral start is not "fair":
# the seed pair first, then the corners of the (kth, kdi) box.
EXTRA_STARTS: list[tuple[float, float]] = [
    (0.10, 0.10),
    (0.60, 0.60),
    (0.10, 0.60),
    (0.60, 0.10),
]

SOLVER_FAIL_PENALTY = 1e6
PF_REACH_TOL = 0.02  # the scan's PF range, widened 2 % each way, must contain the test
# BHP resolution: the BHP change worth one GOOD_FRAC of PF along the scan's
# PF-vs-BHP slope. Past this the PF rate barely sees the BHP and the block
# says so (the nozzle drop is dominated by the PF pressure, not the suction).
WEAK_RESOLUTION_PSI = 250.0
BOUND_TOL = 1e-3
MAXITER = 200

NOZZLE_CAVEAT = (
    "One test cannot separate nozzle wear from BHP: the nozzle is held at its "
    "catalog area, so a washed-out nozzle would make this BHP read low."
)


@dataclass
class GaugelessMatchResult:
    well_name: str
    # the test
    oil_test: float
    water_test: float
    pf_test: float
    # the fit
    pwf: float                      # inferred anchor BHP, psi
    qwf_liq: float                  # anchor TOTAL LIQUID (oil + formation water), BLPD
    form_wc: float                  # the test's formation water cut
    kth: float
    kdi: float
    ken: float                      # held at the caller's value
    knz: float                      # held
    # the model at the fit
    modeled_bhp: float
    modeled_oil: float
    modeled_water: float
    modeled_pf: float
    score: float                    # RMS fractional error (oil, PF)
    oil_error_pct: float            # (model − test) / test × 100
    pf_error_pct: float
    match_quality: str              # good / fair / poor / failed
    converged: bool
    seed_pwf: Optional[float] = None
    scan: list[dict] = field(default_factory=list)
    bounded: bool = False
    sonic: bool = False
    iterations: int = 0
    starts_tried: int = 0
    message: Optional[str] = None
    caveat: str = NOZZLE_CAVEAT
    # PF reachability: the model's PF range over the BHP scan and whether the
    # test's PF falls inside it. False = the BHP is not identified.
    pf_reachable: bool = True
    pf_model_min: Optional[float] = None
    pf_model_max: Optional[float] = None
    # When the PF is unreachable: the nozzle AREA factor that would let the
    # catalog nozzle pass the test's PF at the fitted point. Nozzle flow
    # scales with area, so it is (pf_test / modeled_pf) ** 2 - the same
    # quantity the sidebar's nozzle_area_factor knob holds (the wear
    # mechanics scale the DIAMETER by its square root). None when the PF is
    # reachable or the fit has no finite modeled PF.
    area_factor_needed: Optional[float] = None
    # How well the PF rate resolves the BHP on this well: the BHP change
    # worth a GOOD_FRAC PF error along the scan slope (psi). None when the
    # scan has no slope.
    bhp_resolution_psi: Optional[float] = None
    pf_per_100psi: Optional[float] = None


def _grade(score: float) -> str:
    if not isfinite(score):
        return "failed"
    if score <= GOOD_FRAC:
        return "good"
    if score <= FAIR_FRAC:
        return "fair"
    return "poor"


def _rms(oil: float, pf: float, oil_test: float, pf_test: float) -> float:
    return sqrt(0.5 * (((oil - oil_test) / oil_test) ** 2 + ((pf - pf_test) / pf_test) ** 2))


def match_test(
    *,
    well_name: str,
    oil_test: float,
    water_test: float,
    pf_test: float,
    pres: float,
    make_inflow: Callable[[float, float], "InFlow"],
    pwh: float,
    tsu: float,
    ppf_surf: float,
    nozzle: str,
    throat: str,
    knz: float,
    ken: float,
    seed_kth: float,
    seed_kdi: float,
    wellbore,
    wellprof,
    prop_su,
    prop_pf,
    jpump_direction: str = "reverse",
    nozzle_area_factor: float = 1.0,
    mach_crit: float = 1.0,
) -> GaugelessMatchResult:
    """Fit ``(pwf, kth, kdi)`` so the installed pump reproduces the test's
    oil and PF rates.

    Args:
        oil_test / water_test / pf_test: the test's oil (STBOPD), formation
            water (BWPD) and power-fluid (BWPD) rates. Oil and PF must be
            positive; water may be zero.
        pres: reservoir pressure the IPR is drawn to, psi.
        make_inflow: ``(oil_rate, pwf) -> InFlow`` - the caller's inflow
            factory, so the anchor is built exactly like a solve (oil basis).
        pwh / tsu / ppf_surf: test-day wellhead pressure, suction temperature
            and PF surface pressure.
        nozzle / throat / knz / ken: the installed pump; ``knz`` and ``ken``
            are held.
        seed_kth / seed_kdi: the caller's current discharge coefficients -
            the first start, and what a choked (sonic) fit hands back.
        wellbore / wellprof / prop_su / prop_pf: prebuilt physics objects.
        nozzle_area_factor / mach_crit: the well's persisted wear factor and
            slip closure, applied so the fit and the page run the same pump.

    Returns:
        GaugelessMatchResult. ``match_quality == "failed"`` with a message
        when no trial solved.
    """
    if oil_test <= 0 or pf_test <= 0:
        raise ValueError("the test must carry positive oil and power-fluid rates")
    water_test = max(0.0, float(water_test))
    liquid_test = float(oil_test) + water_test
    form_wc = water_test / liquid_test

    pwf_lo = PWF_FLOOR
    pwf_hi = float(pres) - PWF_PRES_MARGIN
    if pwf_hi <= pwf_lo:
        raise ValueError("reservoir pressure too low to draw an IPR")
    bounds_u = [(pwf_lo / pres, pwf_hi / pres), KTH_BOUNDS, KDI_BOUNDS]

    def _solve(pwf: float, kth: float, kdi: float):
        """(psu, oil, water, pf, sonic, ok) at one trial."""
        try:
            jp = JetPump(nozzle, throat, knz=knz, ken=ken, kth=kth, kdi=kdi)
            if nozzle_area_factor and nozzle_area_factor != 1.0:
                jp.dnz = jp.dnz * float(np.sqrt(nozzle_area_factor))
            ipr = make_inflow(float(oil_test), float(pwf))
            psu, sonic, qoil, fwat, qnz, _mach = jetpump_solver(
                pwh=pwh,
                tsu=tsu,
                ppf_surf=ppf_surf,
                jpump=jp,
                wellbore=wellbore,
                wellprof=wellprof,
                ipr_su=ipr,
                prop_su=prop_su,
                prop_pf=prop_pf,
                jpump_direction=jpump_direction,
                mach_crit=float(mach_crit or 1.0),
            )
            vals = (psu, qoil, fwat, qnz)
            if any(v is None or not isfinite(float(v)) for v in vals):
                return np.nan, np.nan, np.nan, np.nan, False, False
            return float(psu), float(qoil), float(fwat), float(qnz), bool(sonic), True
        except Exception:  # noqa: BLE001 - a failed trial is a penalty, not a crash
            return np.nan, np.nan, np.nan, np.nan, False, False

    # ── seed: scan the BHP window, pick where the PF rate crosses the test ──
    scan: list[dict] = []
    for pwf in np.linspace(SCAN_LO_FRAC * pres, SCAN_HI_FRAC * pres, SCAN_POINTS):
        pwf = float(min(max(pwf, pwf_lo), pwf_hi))
        psu, qoil, fwat, qnz, sonic, ok = _solve(pwf, seed_kth, seed_kdi)
        scan.append(
            {
                "pwf": pwf,
                "psu": psu if ok else None,
                "oil": qoil if ok else None,
                "pf": qnz if ok else None,
                "sonic": sonic if ok else None,
            }
        )
    usable = [s for s in scan if s["pf"] is not None]
    if not usable:
        return GaugelessMatchResult(
            well_name=well_name,
            oil_test=float(oil_test),
            water_test=water_test,
            pf_test=float(pf_test),
            pwf=float("nan"),
            qwf_liq=liquid_test,
            form_wc=form_wc,
            kth=seed_kth,
            kdi=seed_kdi,
            ken=ken,
            knz=knz,
            modeled_bhp=float("nan"),
            modeled_oil=float("nan"),
            modeled_water=float("nan"),
            modeled_pf=float("nan"),
            score=float("inf"),
            oil_error_pct=float("nan"),
            pf_error_pct=float("nan"),
            match_quality="failed",
            converged=False,
            scan=scan,
            message="the pump model found no operating point anywhere in the BHP window",
        )
    pf_model_min = min(s["pf"] for s in usable)
    pf_model_max = max(s["pf"] for s in usable)
    pf_reachable = (
        pf_model_min * (1.0 - PF_REACH_TOL) <= pf_test <= pf_model_max * (1.0 + PF_REACH_TOL)
    )
    # PF-vs-BHP slope over the scan (least squares) -> BHP resolution
    bhp_resolution: Optional[float] = None
    pf_per_100psi: Optional[float] = None
    if len(usable) >= 2:
        xs = np.array([u["pwf"] for u in usable])
        ys = np.array([u["pf"] for u in usable])
        slope = float(np.polyfit(xs, ys, 1)[0])  # BWPD per psi
        if abs(slope) > 1e-9:
            pf_per_100psi = slope * 100.0
            bhp_resolution = GOOD_FRAC * float(pf_test) / abs(slope)
    # nearest scan point on PF, refined by linear interpolation when the
    # neighbours bracket the test's PF rate
    best = min(usable, key=lambda s: abs(s["pf"] - pf_test))
    seed_pwf = best["pwf"]
    for a, b in zip(usable, usable[1:]):
        if (a["pf"] - pf_test) * (b["pf"] - pf_test) < 0 and a["pf"] != b["pf"]:
            t = (pf_test - a["pf"]) / (b["pf"] - a["pf"])
            seed_pwf = a["pwf"] + t * (b["pwf"] - a["pwf"])
            break
    seed_pwf = float(min(max(seed_pwf, pwf_lo), pwf_hi))

    # ── Nelder-Mead over scaled (pwf/pres, kth, kdi) ──
    evals = {"n": 0}

    def objective(u):
        for value, (lo, hi) in zip(u, bounds_u):
            if not (lo <= value <= hi):
                return SOLVER_FAIL_PENALTY
        evals["n"] += 1
        _psu, qoil, _fwat, qnz, _sonic, ok = _solve(u[0] * pres, u[1], u[2])
        if not ok:
            return SOLVER_FAIL_PENALTY
        return _rms(qoil, qnz, oil_test, pf_test)

    def _clip(u) -> list[float]:
        return [float(np.clip(v, lo, hi)) for v, (lo, hi) in zip(u, bounds_u)]

    def run_start(kth0: float, kdi0: float) -> tuple[list[float], int]:
        u0 = _clip([seed_pwf / pres, kth0, kdi0])
        res = minimize(
            objective,
            u0,
            method="Nelder-Mead",
            bounds=bounds_u,
            options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": MAXITER},
        )
        u = _clip(res.x)
        return u, int(getattr(res, "nit", 0) or 0)

    starts = [(seed_kth, seed_kdi)] + EXTRA_STARTS
    best_u: Optional[list[float]] = None
    best_score = float("inf")
    iterations = 0
    starts_tried = 0
    for kth0, kdi0 in starts:
        u, nit = run_start(kth0, kdi0)
        starts_tried += 1
        iterations += nit
        _psu, qoil, _fwat, qnz, _sonic, ok = _solve(u[0] * pres, u[1], u[2])
        score = _rms(qoil, qnz, oil_test, pf_test) if ok else float("inf")
        if score < best_score:
            best_score, best_u = score, u
        if best_score <= MULTISTART_THRESHOLD:
            break

    if best_u is None or not isfinite(best_score):
        return GaugelessMatchResult(
            well_name=well_name,
            oil_test=float(oil_test),
            water_test=water_test,
            pf_test=float(pf_test),
            pwf=seed_pwf,
            qwf_liq=liquid_test,
            form_wc=form_wc,
            kth=seed_kth,
            kdi=seed_kdi,
            ken=ken,
            knz=knz,
            modeled_bhp=float("nan"),
            modeled_oil=float("nan"),
            modeled_water=float("nan"),
            modeled_pf=float("nan"),
            score=float("inf"),
            oil_error_pct=float("nan"),
            pf_error_pct=float("nan"),
            match_quality="failed",
            converged=False,
            seed_pwf=seed_pwf,
            scan=scan,
            iterations=iterations,
            starts_tried=starts_tried,
            message="every start ended on a failed solve",
            pf_reachable=pf_reachable,
            pf_model_min=pf_model_min,
            pf_model_max=pf_model_max,
            bhp_resolution_psi=bhp_resolution,
            pf_per_100psi=pf_per_100psi,
        )

    # narrowed to three floats by the guard above; named so the arithmetic
    # below is float x float rather than "whatever came out of the optimizer"
    u_opt: list[float] = [float(v) for v in best_u]
    pwf_opt, kth_opt, kdi_opt = u_opt[0] * pres, u_opt[1], u_opt[2]
    psu, qoil, fwat, qnz, sonic, _ok = _solve(pwf_opt, kth_opt, kdi_opt)
    message: Optional[str] = None
    if sonic:
        # choked throat: kth/kdi have no gradient - hand back the seeds and
        # keep the one thing the test did identify, the BHP
        kth_opt, kdi_opt = seed_kth, seed_kdi
        psu, qoil, fwat, qnz, sonic, _ok = _solve(pwf_opt, kth_opt, kdi_opt)
        message = (
            "Throat is choked at this point (sonic): the discharge coefficients "
            "cannot be identified and are left at their seeds. The inferred BHP "
            "is the choke floor."
        )
    score = _rms(qoil, qnz, oil_test, pf_test)

    bounded = (
        abs(pwf_opt - pwf_lo) < 1.0
        or abs(pwf_opt - pwf_hi) < 1.0
        or any(
            abs(v - lo) < BOUND_TOL or abs(v - hi) < BOUND_TOL
            for v, (lo, hi) in zip((kth_opt, kdi_opt), (KTH_BOUNDS, KDI_BOUNDS))
        )
    )
    if bounded and message is None:
        message = "A fitted value sits on its search bound - treat the match as provisional."
    quality = _grade(score)
    area_factor_needed: Optional[float] = None
    if not pf_reachable:
        # The PF target is off the model's whole range: the anchor BHP was
        # pulled toward a rail chasing it and is NOT an identification.
        # The one hardware number that WOULD explain it: nozzle flow scales
        # with area, so the area factor that closes the PF gap at this point
        # is the squared rate ratio.
        if isfinite(qnz) and qnz > 0:
            area_factor_needed = (float(pf_test) / float(qnz)) ** 2
        side = "more" if pf_test > pf_model_max else "less"
        message = (
            f"The pump cannot pass {pf_test:,.0f} BWPD at any BHP with a catalog nozzle at "
            f"{ppf_surf:,.0f} psi PF pressure (model range {pf_model_min:,.0f} to "
            f"{pf_model_max:,.0f} BWPD); the test wants {side}. The BHP below is the closest "
            "point, not an identification - check the test-day PF pressure, nozzle wear "
            "(area factor) and the installed pump identity."
        ) + (f" {message}" if message else "")
        if quality in ("good", "fair"):
            quality = "poor"
    elif bhp_resolution is not None and bhp_resolution > WEAK_RESOLUTION_PSI:
        message = (
            f"The PF rate barely sees the BHP on this well ({abs(pf_per_100psi or 0.0):,.0f} BWPD per "
            f"100 psi): a {GOOD_FRAC * 100:.0f} % PF error is worth about {bhp_resolution:,.0f} psi, "
            "so treat the inferred BHP as a bracket, not a point."
        ) + (f" {message}" if message else "")

    return GaugelessMatchResult(
        well_name=well_name,
        oil_test=float(oil_test),
        water_test=water_test,
        pf_test=float(pf_test),
        pwf=float(pwf_opt),
        qwf_liq=liquid_test,
        form_wc=form_wc,
        kth=float(kth_opt),
        kdi=float(kdi_opt),
        ken=ken,
        knz=knz,
        modeled_bhp=psu,
        modeled_oil=qoil,
        modeled_water=fwat,
        modeled_pf=qnz,
        score=score,
        oil_error_pct=(qoil - oil_test) / oil_test * 100.0,
        pf_error_pct=(qnz - pf_test) / pf_test * 100.0,
        match_quality=quality,
        converged=True,
        seed_pwf=seed_pwf,
        scan=scan,
        bounded=bounded,
        sonic=sonic,
        iterations=iterations,
        starts_tried=starts_tried,
        message=message,
        pf_reachable=pf_reachable,
        pf_model_min=pf_model_min,
        pf_model_max=pf_model_max,
        bhp_resolution_psi=bhp_resolution,
        pf_per_100psi=pf_per_100psi,
        area_factor_needed=area_factor_needed,
    )
