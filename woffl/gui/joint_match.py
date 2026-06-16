"""Joint oil + power-fluid auto-match for a single jet-pump well.

Given a well's MEASURED test oil rate and power-fluid (lift-water) rate, find the
IPR + jet-pump parameters that make the model reproduce BOTH at once — so the
model is trustworthy in absolute terms before it's optimized.

Knobs (the levers an engineer would actually turn):
  - IPR productivity (Vogel qmax)  → primarily sets OIL.
  - PF surface pressure (ppf_surf) → primarily sets the PF (nozzle) rate.
  - Friction coefs ken/kth/kdi     → added degrees of freedom; also let the BHP
    be matched when a measured BHP is supplied, and help close a gap the two
    primary knobs can't.
The installed nozzle/throat is FIXED — we match the pump that was in the well.

Strategy: bounded least-squares (scipy ``least_squares``, TRF) on the residual
vector [oil_err, pf_err, (bhp_err)] with multi-start; a solver failure at a trial
point returns a large penalty so the optimizer steps away from it. We try a
2-knob match (qmax, ppf) first; if that can't close both targets we expand to the
5-knob match (add friction). When even that can't match, ``diagnose()`` classifies
WHY — PF-limited, oil-limited, friction-bound, or non-convergent — with the
physical reason, so the engineer knows what to fix (no silent failure).

Pure of Streamlit; the GUI builds wellbore/well_profile and seeds the sidebar
from the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from woffl.flow.inflow import InFlow
from woffl.gui.utils import (
    create_jetpump,
    create_reservoir_mix,
    run_jetpump_solver,
)

_PENALTY = 5.0  # per-residual penalty when the solver can't solve a trial point
_MATCH_TOL = 0.10  # |err| <= 10% counts as matched (per target)
_PPF_LO, _PPF_HI = 800.0, 5500.0
_KEN_LO, _KEN_HI = 0.005, 0.40
_KTH_LO, _KTH_HI = 0.05, 1.0
_KDI_LO, _KDI_HI = 0.05, 1.0


@dataclass
class JointMatchResult:
    ok: bool                      # both oil and PF matched within tolerance
    status: str                   # "matched" / "partial" / "failed"
    # Matched parameters (to seed the sidebar):
    qwf_oil: float                # sidebar qwf (OIL) anchor
    pwf: float                    # inferred flowing BHP (sidebar pwf)
    pres: float
    ken: float
    kth: float
    kdi: float
    ppf_surf: float
    # Modeled outcome at the matched params:
    modeled_oil: float
    modeled_pf: float
    modeled_bhp: float
    oil_err_pct: float            # (modeled-target)/target * 100
    pf_err_pct: float
    bhp_err_pct: Optional[float]
    sonic: bool
    diagnostic: str               # human-readable why (esp. when not matched)
    iterations: int


@dataclass
class BatchMatchRow:
    """One well's outcome in a batch auto-match (see :func:`batch_match`)."""
    well: str
    status: str                      # "matched" / "partial" / "failed" / "error"
    ok: bool
    oil_err_pct: Optional[float]
    pf_err_pct: Optional[float]
    ppf_surf: Optional[float]
    diagnostic: str
    result: Optional[JointMatchResult]  # full result (None on "error") for "apply"


# Sort order for a batch health table — worst first so the engineer triages the
# wells that need attention before the ones that already match.
_BATCH_ORDER = {"error": 0, "failed": 1, "partial": 2, "matched": 3}


def batch_match(wells) -> list[BatchMatchRow]:
    """Run :func:`joint_match` over many wells with per-well error isolation.

    This is the pure (Streamlit-free) core a "match all wells" button is a thin
    glue layer over: the GUI builds each well's kwargs from the review store +
    its measured test (oil + PF), and renders the returned rows as a health
    table. Keeping the loop here means it's unit-tested and a single bad well
    can never break the batch.

    Args:
        wells: iterable of ``(label, kwargs)`` where ``kwargs`` is a dict passed
            straight to ``joint_match(**kwargs)``.

    Returns:
        One :class:`BatchMatchRow` per well, sorted worst-first
        (error → failed → partial → matched, then by label). A well whose match
        raises becomes a ``status="error"`` row rather than aborting the batch.
    """
    rows: list[BatchMatchRow] = []
    for label, kwargs in wells:
        try:
            r = joint_match(**kwargs)
            rows.append(BatchMatchRow(
                well=label, status=r.status, ok=r.ok, oil_err_pct=r.oil_err_pct,
                pf_err_pct=r.pf_err_pct, ppf_surf=r.ppf_surf,
                diagnostic=r.diagnostic, result=r,
            ))
        except Exception as e:  # one well must never break the batch
            rows.append(BatchMatchRow(
                well=label, status="error", ok=False, oil_err_pct=None,
                pf_err_pct=None, ppf_surf=None,
                diagnostic=f"Match crashed: {type(e).__name__}: {e}", result=None,
            ))
    rows.sort(key=lambda x: (_BATCH_ORDER.get(x.status, 0), x.well))
    return rows


def batch_summary(rows: list[BatchMatchRow]) -> dict:
    """Counts by status for a batch result — ``{matched, partial, failed, error}``."""
    out = {"matched": 0, "partial": 0, "failed": 0, "error": 0}
    for r in rows:
        out[r.status] = out.get(r.status, 0) + 1
    return out


def _inflow_from_qmax(qmax: float, pres: float) -> InFlow:
    """Build an InFlow with the given Vogel qmax (productivity), anchored at a
    reference drawdown. ``oil_flow(psu) = qmax * vogel(psu)``, so qmax is the
    single productivity knob; the operating oil falls out of the solve."""
    pwf_ref = 0.5 * pres
    f = 1.0 - 0.2 * (pwf_ref / pres) - 0.8 * (pwf_ref / pres) ** 2  # = 0.7
    return InFlow(qwf=max(qmax * f, 1.0), pwf=pwf_ref, pres=pres)


def joint_match(
    *,
    oil_target: float,
    pf_target: float,
    pres: float,
    nozzle: str,
    throat: str,
    surf_pres: float,
    form_temp: float,
    rho_pf: float,
    ppf_surf0: float,
    wellbore,
    well_profile,
    form_wc: float,
    form_gor: float,
    ken0: float = 0.03,
    kth0: float = 0.3,
    kdi0: float = 0.4,
    field_model: Optional[str] = None,
    jpump_direction: str = "reverse",
    bhp_target: Optional[float] = None,
    tune_friction: bool = True,
    max_nfev: int = 120,
) -> JointMatchResult:
    """Find IPR + JP params so the model reproduces oil_target AND pf_target."""
    from scipy.optimize import least_squares

    res_mix = create_reservoir_mix(form_wc, form_gor, form_temp, field_model)

    state = {"n": 0}

    def model(qmax, ppf, ken, kth, kdi):
        state["n"] += 1
        try:
            jp = create_jetpump(nozzle, throat, ken, kth, kdi)
        except Exception:
            return None
        out = run_jetpump_solver(
            surf_pres, form_temp, rho_pf, float(np.clip(ppf, _PPF_LO, _PPF_HI)),
            jp, wellbore, well_profile, _inflow_from_qmax(qmax, pres), res_mix,
            field_model=field_model, jpump_direction=jpump_direction, quiet=True,
        )
        if out is None:
            return None
        psu, sonic, qoil, _fwat, qnz, mach = out
        if psu is None or np.isnan(psu) or np.isnan(qoil) or np.isnan(qnz):
            return None
        return {"bhp": float(psu), "oil": float(qoil), "pf": float(qnz),
                "sonic": bool(sonic), "mach": float(mach),
                "nozzle": f"{nozzle}{throat}"}

    def residuals(m):
        r = [(m["oil"] - oil_target) / max(oil_target, 1.0),
             (m["pf"] - pf_target) / max(pf_target, 1.0)]
        if bhp_target:
            r.append((m["bhp"] - bhp_target) / max(bhp_target, 1.0))
        return r

    n_pen = 3 if bhp_target else 2

    # ── Stage 1: 2-knob match (qmax, ppf) ────────────────────────────────
    qmax_lo, qmax_hi = max(oil_target, 10.0), oil_target * 12.0 + 1000.0

    def resid2(x):
        m = model(x[0], x[1], ken0, kth0, kdi0)
        return [_PENALTY] * n_pen if m is None else residuals(m)

    starts2 = [
        (oil_target * 1.5, ppf_surf0),
        (oil_target * 2.2, ppf_surf0),
        (oil_target * 1.25, min(ppf_surf0 * 1.15, _PPF_HI)),
        (oil_target * 3.0, max(ppf_surf0 * 0.85, 1600.0)),
    ]
    best = _run_starts(
        least_squares, resid2, starts2,
        bounds=([qmax_lo, _PPF_LO], [qmax_hi, _PPF_HI]), max_nfev=max_nfev,
    )
    bx = best["x"]
    bm = model(bx[0], bx[1], ken0, kth0, kdi0)
    bken, bkth, bkdi, bppf = ken0, kth0, kdi0, bx[1]

    matched = bm is not None and _within(bm, oil_target, pf_target, bhp_target)

    # ── Stage 2: 5-knob match (add friction) if Stage 1 couldn't close it ──
    if tune_friction and not matched:
        def resid5(x):
            m = model(x[0], x[1], x[2], x[3], x[4])
            return [_PENALTY] * n_pen if m is None else residuals(m)

        starts5 = [(bx[0], bx[1], ken0, kth0, kdi0),
                   (bx[0], bx[1], 0.10, 0.30, 0.30),
                   (oil_target * 1.5, ppf_surf0, 0.03, 0.50, 0.50)]
        best5 = _run_starts(
            least_squares, resid5, starts5,
            bounds=([qmax_lo, _PPF_LO, _KEN_LO, _KTH_LO, _KDI_LO],
                    [qmax_hi, _PPF_HI, _KEN_HI, _KTH_HI, _KDI_HI]),
            max_nfev=max_nfev,
        )
        m5 = model(*best5["x"])
        if m5 is not None and (bm is None or best5["cost"] < best["cost"]):
            best, bm = best5, m5
            bx = best5["x"]
            bken, bkth, bkdi, bppf = bx[2], bx[3], bx[4], bx[1]
            matched = _within(bm, oil_target, pf_target, bhp_target)

    # ── Probe the binding constraint (only when we couldn't match) ────────
    # Re-solve at the pump's physical ceilings so the diagnostic can state the
    # real limit ("this nozzle tops out at X BPD") instead of guessing from how
    # close a knob landed to its bound.
    probes = {}
    if bm is not None and not matched:
        m_ipr = model(qmax_hi, bppf, bken, bkth, bkdi)        # max productivity, same PF
        m_ipr_pf = model(qmax_hi, _PPF_HI, bken, bkth, bkdi)  # max productivity + max PF
        m_pf = model(bx[0], _PPF_HI, bken, bkth, bkdi)        # max PF pressure
        probes = {
            "oil_at_max_ipr": (m_ipr or {}).get("oil"),
            "oil_at_max_ipr_pf": (m_ipr_pf or {}).get("oil"),
            "pf_at_max_ppf": (m_pf or {}).get("pf"),
        }

    return _build_result(
        bm, bx[0], bppf, bken, bkth, bkdi, pres, oil_target, pf_target,
        bhp_target, matched, state["n"], qmax_hi, probes,
    )


def _run_starts(least_squares, resid, starts, *, bounds, max_nfev):
    """Run least_squares from several seeds; keep the lowest-cost result."""
    best = None
    for x0 in starts:
        try:
            r = least_squares(resid, list(x0), bounds=bounds, method="trf",
                              loss="soft_l1", max_nfev=max_nfev, xtol=1e-4, ftol=1e-4)
        except Exception:
            continue
        if best is None or r.cost < best["cost"]:
            best = {"x": r.x, "cost": r.cost}
        if best["cost"] < 1e-4:
            break
    if best is None:  # everything threw — return the first seed unsolved
        best = {"x": np.array(starts[0]), "cost": float("inf")}
    return best


def _within(m, oil_t, pf_t, bhp_t) -> bool:
    if m is None:
        return False
    ok = (abs(m["oil"] - oil_t) / max(oil_t, 1.0) <= _MATCH_TOL
          and abs(m["pf"] - pf_t) / max(pf_t, 1.0) <= _MATCH_TOL)
    if bhp_t:
        ok = ok and abs(m["bhp"] - bhp_t) / max(bhp_t, 1.0) <= _MATCH_TOL
    return ok


def _build_result(m, qmax, ppf, ken, kth, kdi, pres, oil_t, pf_t, bhp_t,
                  matched, iters, qmax_hi, probes=None) -> JointMatchResult:
    if m is None:
        return JointMatchResult(
            ok=False, status="failed", qwf_oil=oil_t, pwf=0.5 * pres, pres=pres,
            ken=ken, kth=kth, kdi=kdi, ppf_surf=ppf, modeled_oil=0.0,
            modeled_pf=0.0, modeled_bhp=0.0, oil_err_pct=float("nan"),
            pf_err_pct=float("nan"), bhp_err_pct=None, sonic=False,
            diagnostic=("The model could not solve this pump/IPR/PF combination at "
                        "all (no operating point). Check the pump identity, GOR, and "
                        "reservoir pressure."),
            iterations=iters,
        )
    oil_err = (m["oil"] - oil_t) / max(oil_t, 1.0)
    pf_err = (m["pf"] - pf_t) / max(pf_t, 1.0)
    bhp_err = ((m["bhp"] - bhp_t) / max(bhp_t, 1.0)) if bhp_t else None
    status = "matched" if matched else "partial"
    diag = diagnose(m, oil_err, pf_err, ppf, oil_t, pf_t, matched, probes or {})
    return JointMatchResult(
        ok=matched, status=status,
        qwf_oil=float(m["oil"]),        # anchor the sidebar IPR at the modeled oil
        pwf=float(m["bhp"]),            # at the operating BHP -> reproduces the IPR
        pres=float(pres), ken=float(ken), kth=float(kth), kdi=float(kdi),
        ppf_surf=float(ppf), modeled_oil=float(m["oil"]), modeled_pf=float(m["pf"]),
        modeled_bhp=float(m["bhp"]), oil_err_pct=oil_err * 100.0,
        pf_err_pct=pf_err * 100.0,
        bhp_err_pct=(bhp_err * 100.0 if bhp_err is not None else None),
        sonic=m["sonic"], diagnostic=diag, iterations=iters,
    )


def diagnose(m, oil_err, pf_err, ppf, oil_t, pf_t, matched, probes) -> str:
    """Classify why a match did/didn't land, with the physical reason + fix.

    Uses the constraint probes (oil at max IPR, oil at max IPR+PF, PF at max
    pressure) to state the *real* physical ceiling rather than inferring it from
    how close a knob landed to its numeric bound.
    """
    if matched:
        return "Matched oil and PF within 10%."
    bits = []
    pf_max = probes.get("pf_at_max_ppf")
    oil_max_ipr = probes.get("oil_at_max_ipr")
    oil_max_ipr_pf = probes.get("oil_at_max_ipr_pf")

    # ── PF (power-fluid / nozzle rate) ────────────────────────────────────
    if abs(pf_err) > _MATCH_TOL:
        if pf_err < 0:  # modeled PF below target
            if pf_max is not None and pf_max < pf_t * (1 - _MATCH_TOL):
                bits.append(
                    f"PF-limited: the {('sonic-choked ' if m['sonic'] else '')}nozzle "
                    f"delivers at most {pf_max:,.0f} BPD even at {_PPF_HI:,.0f} psi, vs "
                    f"the {pf_t:,.0f} BPD target. The nozzle is too small, or the real "
                    "delivered PF pressure is higher than modeled — verify the pump "
                    "size and header pressure."
                )
            else:
                bits.append(
                    f"PF short by {pf_err * 100:+.0f}% ({m['pf']:,.0f} vs {pf_t:,.0f}) — "
                    "more PF pressure would close it; it traded off against the oil "
                    "target. Consider matching at a higher header pressure."
                )
        else:  # modeled PF above target
            if ppf <= _PPF_LO + 50:
                bits.append(
                    f"PF too HIGH: the nozzle pushes {m['pf']:,.0f} BPD even at the "
                    f"{_PPF_LO:,.0f} psi floor — the modeled PF pressure is likely "
                    "lower than real, or the nozzle is larger than modeled."
                )
            else:
                bits.append(
                    f"PF over by {pf_err * 100:+.0f}% ({m['pf']:,.0f} vs {pf_t:,.0f})."
                )

    # ── Oil ───────────────────────────────────────────────────────────────
    if abs(oil_err) > _MATCH_TOL:
        if oil_err < 0:  # modeled oil below target
            if oil_max_ipr_pf is not None and oil_max_ipr_pf < oil_t * (1 - _MATCH_TOL):
                bits.append(
                    f"Pump-capacity-limited: with unlimited reservoir AND max PF "
                    f"pressure the {m.get('nozzle','this')} pump still tops out at "
                    f"{oil_max_ipr_pf:,.0f} BOPD < the {oil_t:,.0f} target. This pump "
                    "physically can't make the oil — fit a bigger nozzle/throat."
                )
            elif oil_max_ipr is not None and oil_max_ipr < oil_t * (1 - _MATCH_TOL):
                bits.append(
                    f"PF-pressure-limited on oil: at {ppf:,.0f} psi the pump lifts at "
                    f"most {oil_max_ipr:,.0f} BOPD, but it reaches the {oil_t:,.0f} "
                    "target at higher PF pressure. Raise the delivered header pressure."
                )
            else:
                bits.append(
                    f"Oil short by {oil_err * 100:+.0f}% ({m['oil']:,.0f} vs "
                    f"{oil_t:,.0f}) — a stronger IPR reaches it; the optimizer settled "
                    "in a local min. Re-run or nudge reservoir pressure."
                )
        else:  # modeled oil above target — IPR too productive / drawdown too high
            bits.append(
                f"Oil over by {oil_err * 100:+.0f}% ({m['oil']:,.0f} vs {oil_t:,.0f}) — "
                "the IPR is more productive than the test shows."
            )
    return "  ".join(bits) or "Partial match — see the errors above."
