"""Pure compute helpers for the Header Pressure Impact tab.

This module holds the side-effect-free logic behind ``header_impact`` so it can
be unit-tested without a Streamlit or Databricks runtime. The Streamlit render
layer (``header_impact.render_tab``) imports these and supplies the IO.

Currently houses:
  - ``resolve_pad_pf`` / ``pf_map_from_selected`` — per-pad power-fluid pressure
    resolution (the v2 "each pad runs at a different PF" feature).
  - ``_chosen_method`` — the physics/empirical/auto ΔOil picker.
  - ``_verdict`` — the sonic-aware per-well response label.

The larger solver/empirical extraction (``_solve_at_whp``, ``_empirical_columns``,
``_solve_nonjp_row``) is queued for a follow-up so it can be verified against the
live UI rather than blind.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd


# ── per-pad power-fluid pressure ─────────────────────────────────────────────


def resolve_pad_pf(
    pads: list[str],
    stored: dict | None,
    default_fn: Callable[[str], float],
) -> dict[str, int]:
    """Resolve the per-pad PF pressure dict for exactly the selected pads.

    Each pad runs its jet pumps at a different power-fluid pressure, so PF is a
    per-pad input to the modeled baseline (not one global value). This merges
    previously-stored edits with pad defaults:

    - pads still selected keep their stored (possibly user-edited) value;
    - newly selected pads are seeded from ``default_fn(pad)``;
    - deselected pads are dropped.

    Args:
        pads: the currently selected pad letters (e.g. ``["G", "H", "I", "J"]``).
        stored: prior ``{pad: pf}`` mapping from session state (may be None or
            contain pads no longer selected).
        default_fn: callable mapping a pad letter to its default PF (psi).

    Returns:
        ``{pad: int_pf}`` for exactly ``pads``, values coerced to int.
    """
    stored = stored or {}
    out: dict[str, int] = {}
    for pad in pads:
        val = stored.get(pad)
        if val is None:
            val = default_fn(pad)
        try:
            out[pad] = int(round(float(val)))
        except (TypeError, ValueError):
            out[pad] = int(default_fn(pad))
    return out


def pf_map_from_selected(selected_df: pd.DataFrame | None) -> dict[str, float]:
    """Build a ``{well: pf_held}`` map from the selected input rows.

    Used to feed each well's own PF pressure into the gaugeless BHP back-calc
    (replacing the old single global ``test_pf_pres``). Rows without a numeric
    ``PF held (psi)`` (e.g. non-JP wells) are skipped.
    """
    out: dict[str, float] = {}
    if selected_df is None or "PF held (psi)" not in getattr(selected_df, "columns", []):
        return out
    for _, r in selected_df.iterrows():
        wn = r.get("Well")
        pf = r.get("PF held (psi)")
        if wn is None or pd.isna(pf):
            continue
        try:
            out[str(wn)] = float(pf)
        except (TypeError, ValueError):
            continue
    return out


# ── ΔOil method picker (physics / empirical / auto) ──────────────────────────


def _chosen_method(phys_doil, emp_doil, method: str):
    """Pick the ΔOil to trust for a well + a label, per the global method.

    auto = keep physics unless the empirical ΔOil is available AND materially
    disagrees (Scott's "override physics when reality differs"); empirical = use
    the field value wherever a responsive empirical slope produced one.
    """
    phys = float(phys_doil) if pd.notna(phys_doil) else np.nan
    emp = float(emp_doil) if pd.notna(emp_doil) else np.nan
    if pd.isna(phys):  # non-JP well (no physics) — empirical is the only estimate
        return (emp, "empirical") if pd.notna(emp) else (np.nan, "—")
    if method == "physics":
        return phys, "physics"
    if method == "empirical":
        return (emp, "empirical") if pd.notna(emp) else (phys, "physics (emp N/A)")
    # auto
    if pd.notna(emp) and pd.notna(phys) and abs(phys - emp) >= max(20.0, 0.3 * abs(phys)):
        return emp, "empirical (override)"
    return phys, "physics"


# ── sonic-aware per-well verdict ─────────────────────────────────────────────


def _verdict(
    sonic_now: bool,
    sonic_scen: bool,
    delta_oil: float,
    emp_class: str | None,
    compare_emp: bool,
    resp_thresh: float = 5.0,
) -> str:
    """Combine the physics sonic flag + ΔOil with the empirical class into an
    actionable per-well label (Scott's sonic-decoupling insight made explicit).

    A jet pump already sonic at the current WHP is choked — a header move can't
    propagate past the critical throat, so it won't respond regardless of what
    the IPR suggests. Non-sonic wells that the physics says respond are then
    confirmed or contradicted by the empirical slope.
    """
    if sonic_now:
        return "sonic-decoupled"          # already choked — header won't help
    if sonic_scen:
        return "chokes when lowered"       # responds, then sonic-limits
    responds = pd.notna(delta_oil) and abs(delta_oil) >= resp_thresh
    if not responds:
        return "no response"
    if not compare_emp or emp_class in (None, "no tag", "no data", "insufficient"):
        return "responsive (physics)"
    if emp_class == "responsive":
        return "responsive ✓"
    if emp_class == "slugging":
        return "disagree — check"
    return "responsive (physics)"


# ── sensitivity roll-up (per-pad + overall) ──────────────────────────────────


def verdict_bucket(verdict, sonic_now: bool = False) -> str:
    """Collapse the 6-way verdict into a response class for roll-ups.

    responsive / sonic / no-response / other. ``sonic_now`` (the physics flag)
    forces ``sonic`` even if the verdict text doesn't say so.
    """
    v = str(verdict or "")
    if sonic_now or v in ("sonic-decoupled", "chokes when lowered"):
        return "sonic"
    if v.startswith("responsive"):
        return "responsive"
    if v == "no response":
        return "no-response"
    return "other"


def summarize_sensitivity(
    results_df: pd.DataFrame, delta_p: float, doil_col: str = "Chosen ΔOil (BOPD)"
):
    """Per-pad + overall sensitivity roll-up for the header move.

    Returns ``(per_pad_df, overall)``: one row per pad plus an ``overall`` dict
    for the whole selection. Each carries well counts by response class, summed
    ΔOil, post-move oil, and a normalized BOPD-per-100-psi lever (so a -50 psi
    run and a -100 psi run are comparable).
    """
    empty_overall: dict = {}
    if results_df is None or results_df.empty:
        return pd.DataFrame(), empty_overall
    df = results_df.copy()
    if doil_col not in df.columns:
        doil_col = "ΔOil (BOPD)"
    if "Sonic now" in df.columns:
        sonic = df["Sonic now"].fillna(False)
    else:
        sonic = pd.Series(False, index=df.index)
    verdicts = df["Verdict"] if "Verdict" in df.columns else pd.Series("", index=df.index)
    df["_bucket"] = [verdict_bucket(v, bool(s)) for v, s in zip(verdicts, sonic)]
    norm = (100.0 / abs(delta_p)) if delta_p else None

    def _agg(g: pd.DataFrame) -> dict:
        doil = float(g[doil_col].sum())
        oil_now = float(g["Oil now (BOPD)"].sum()) if "Oil now (BOPD)" in g.columns else np.nan
        return {
            "Wells": int(len(g)),
            "Responsive": int((g["_bucket"] == "responsive").sum()),
            "Sonic": int((g["_bucket"] == "sonic").sum()),
            "No-response": int((g["_bucket"] == "no-response").sum()),
            "Oil now (BOPD)": oil_now,
            "ΔOil (BOPD)": doil,
            "Oil after Δ (BOPD)": oil_now + doil,
            "BOPD per 100 psi": (doil * norm) if norm is not None else np.nan,
        }

    rows = [{"Pad": pad, **_agg(g)} for pad, g in df.groupby("Pad")]
    per_pad = pd.DataFrame(rows).sort_values("Pad").reset_index(drop=True)
    overall = {"Pad": "ALL", **_agg(df)}
    return per_pad, overall


# ── modeled-vs-reality sense check ───────────────────────────────────────────


def sense_check_table(
    results_df: pd.DataFrame, test_oil_map: dict | None = None
) -> pd.DataFrame:
    """Per-well modeled-vs-reality table.

    Shows Phys / Emp / Chosen ΔOil side by side, plus the modeled Oil-now against
    the well's own measured test oil (a validation residual that the tab didn't
    surface before — only the single-well Solver tab did).
    """
    if results_df is None or results_df.empty:
        return pd.DataFrame()
    test_oil_map = test_oil_map or {}
    df = results_df.copy()

    def col(name, default=np.nan):
        return df[name] if name in df.columns else pd.Series(default, index=df.index)

    out = pd.DataFrame(
        {
            "Well": col("Well", ""),
            "Pad": col("Pad", ""),
            "Lift": col("Lift", ""),
            "Phys ΔOil (BOPD)": col("ΔOil (BOPD)"),
            "Emp ΔOil (BOPD)": col("Emp ΔOil (BOPD)"),
            "Chosen ΔOil (BOPD)": col("Chosen ΔOil (BOPD)"),
            "Method used": col("Method used", ""),
            "Emp class": col("Emp class", ""),
            "Oil now modeled (BOPD)": col("Oil now (BOPD)"),
        }
    )
    out["Oil now measured (BOPD)"] = out["Well"].map(lambda w: test_oil_map.get(w, np.nan))
    out["Oil residual (BOPD)"] = out["Oil now modeled (BOPD)"] - out["Oil now measured (BOPD)"]
    measured = out["Oil now measured (BOPD)"]
    out["Oil residual %"] = np.where(
        measured.abs() > 0, out["Oil residual (BOPD)"] / measured * 100.0, np.nan
    )
    return out


def _bias_flag(b) -> str:
    if b is None or (isinstance(b, float) and np.isnan(b)):
        return "—"
    if b < 0.8:
        return "physics optimistic (<0.8)"
    if b > 1.2:
        return "physics conservative (>1.2)"
    return "ok"


def bias_by_pad(results_df: pd.DataFrame) -> pd.DataFrame:
    """Per-pad empirical-vs-physics ΔOil bias for responsive wells.

    bias = mean(Emp ΔOil / Phys ΔOil) over responsive wells that have both
    estimates and non-zero physics. A soft optimism flag (not a hard gate):
    <0.8 = physics looks optimistic vs the field, >1.2 = conservative. Returns a
    per-pad frame with an appended ``ALL`` row; empty if nothing qualifies.
    """
    if results_df is None or results_df.empty:
        return pd.DataFrame()
    df = results_df.copy()
    if "ΔOil (BOPD)" not in df.columns or "Emp ΔOil (BOPD)" not in df.columns:
        return pd.DataFrame()
    verdicts = df["Verdict"].astype(str) if "Verdict" in df.columns else pd.Series("", index=df.index)
    resp = df[verdicts.str.startswith("responsive")].copy()
    resp = resp[resp["ΔOil (BOPD)"].notna() & resp["Emp ΔOil (BOPD)"].notna()]
    resp = resp[resp["ΔOil (BOPD)"].abs() > 1e-6]
    if resp.empty:
        return pd.DataFrame()
    resp["_ratio"] = resp["Emp ΔOil (BOPD)"] / resp["ΔOil (BOPD)"]
    rows = [
        {"Pad": pad, "Responsive wells": int(len(g)), "Emp/Phys bias": float(g["_ratio"].mean())}
        for pad, g in resp.groupby("Pad")
    ]
    out = pd.DataFrame(rows).sort_values("Pad").reset_index(drop=True)
    out["Flag"] = out["Emp/Phys bias"].apply(_bias_flag)
    total_bias = float(resp["_ratio"].mean())
    total = {
        "Pad": "ALL",
        "Responsive wells": int(len(resp)),
        "Emp/Phys bias": total_bias,
        "Flag": _bias_flag(total_bias),
    }
    return pd.concat([out, pd.DataFrame([total])], ignore_index=True)


# ── model-vs-observed response sense check ───────────────────────────────────


def slope_agreement(
    model_slope, obs_slope, rel_tol: float = 0.30, abs_tol: float = 0.15
) -> tuple[str, float]:
    """Does the historian's observed dBHP/dWHP match the model's predicted one?

    The model says BHP should move ``model_slope`` psi per psi of wellhead change;
    the within-day historian fit says it actually moved ``obs_slope``. This is the
    heart of "did we see what the model expects?".

    Agreement when ``|obs − model| ≤ max(abs_tol, rel_tol·|model|)`` — a blended
    absolute + relative band so near-zero slopes aren't judged on ratio alone.
    Outside the band: the field moved LESS than the model (model over-predicts the
    response — e.g. a near-sonic / slugging well) or MORE (model under-predicts).

    Returns ``(label, obs/model ratio)``. ``ratio`` is NaN when it isn't defined.
    """
    m = float(model_slope) if model_slope is not None and not pd.isna(model_slope) else np.nan
    o = float(obs_slope) if obs_slope is not None and not pd.isna(obs_slope) else np.nan
    if np.isnan(m) and np.isnan(o):
        return "no data", np.nan
    if np.isnan(o):
        return "no observed slope", np.nan
    if np.isnan(m):
        return "empirical only", np.nan
    ratio = (o / m) if abs(m) > 1e-9 else np.nan
    if abs(o - m) <= max(abs_tol, rel_tol * abs(m)):
        return "confirmed ✓", ratio
    if o < m:
        return "model over-predicts", ratio   # field moved less than the model
    return "model under-predicts", ratio        # field moved more than the model


def sense_check_response(
    results_df: pd.DataFrame, rel_tol: float = 0.30, abs_tol: float = 0.15
) -> pd.DataFrame:
    """Per-well model-vs-observed response sense check.

    For each well: the model's BHP sensitivity (``Phys dBHP/dWHP``) vs the observed
    historian slope (``Emp dBHP/dWHP``), with a verdict from :func:`slope_agreement`.
    Sonic-decoupled wells (physics says the header can't propagate past the choke)
    are reported separately, and we note whether the history is correspondingly flat
    (agrees) or shows a response the physics can't (worth a look).
    """
    if results_df is None or results_df.empty:
        return pd.DataFrame()
    flat_classes = ("slugging", "insufficient", "no data", "no tag", "")
    rows = []
    for _, r in results_df.iterrows():
        model_s = r.get("Phys dBHP/dWHP", np.nan)
        obs_s = r.get("Emp dBHP/dWHP", np.nan)
        emp_class = str(r.get("Emp class", ""))
        if bool(r.get("Sonic now", False)):
            label, ratio = "sonic-decoupled", np.nan
            if emp_class == "responsive":
                note = "history shows response — check"
            elif emp_class in flat_classes:
                note = "history flat — agrees"
            else:
                note = ""
        else:
            label, ratio = slope_agreement(model_s, obs_s, rel_tol, abs_tol)
            note = ""
        rows.append(
            {
                "Well": r.get("Well"), "Pad": r.get("Pad"), "Lift": r.get("Lift"),
                "WHP now (psi)": r.get("WHP now (psi)"),
                "BHP now (psi)": r.get("BHP now (psi)"),
                "Model dBHP/dWHP": model_s, "Obs dBHP/dWHP": obs_s,
                "Obs/Model": ratio, "Emp class": emp_class,
                "Sense-check": label, "Note": note,
            }
        )
    return pd.DataFrame(rows)


def pad_updown_lever(curve: dict | None, pad: str, ref: int = 100) -> tuple[float, float]:
    """The pad's oil lever DOWN vs UP from today, summed over its wells.

    From the swept response curve, ``(down, up)`` = oil change for a ``ref``-psi
    header drop and a ``ref``-psi rise (BOPD). The two differ when the response is
    nonlinear — e.g. wells choke sonic on the way down, flattening that side. NaN
    where the curve lacks the ±ref or 0 points, or no pad well has data there.
    """
    if not curve:
        return (np.nan, np.nan)
    deltas = list(curve.get("deltas", []))
    wells = curve.get("wells", {}) or {}
    if 0 not in deltas or ref not in deltas or -ref not in deltas:
        return (np.nan, np.nan)
    i0, idn, iup = deltas.index(0), deltas.index(-ref), deltas.index(ref)

    def pad_oil(i: int) -> float:
        tot, seen = 0.0, False
        for wd in wells.values():
            if str(wd.get("pad")) != str(pad):
                continue
            o = wd.get("oil", [])
            if i < len(o) and o[i] is not None and not pd.isna(o[i]):
                tot += float(o[i])
                seen = True
        return tot if seen else np.nan

    o0, odn, oup = pad_oil(i0), pad_oil(idn), pad_oil(iup)
    down = (odn - o0) if not (np.isnan(odn) or np.isnan(o0)) else np.nan
    up = (oup - o0) if not (np.isnan(oup) or np.isnan(o0)) else np.nan
    return (down, up)


# ── long-horizon back-test (real header move vs the model) ───────────────────


def backtest_anchors(
    test_well: pd.DataFrame, frac: float = 0.30,
    date_col: str = "WtDate", whp_col: str = "whp", bhp_col: str = "BHP",
    oil_col: str = "WtOilVol", liquid_col: str = "WtTotalFluid",
) -> dict:
    """Then/now operating-point anchors for the long-horizon back-test.

    Splits a well's allocated tests (sorted by date) into an early ``frac`` slice
    and a late ``frac`` slice, taking the median of each pressure/rate per slice —
    robust to single-test noise. Returns medians, the Δ between them, test count
    and day span; ``{}`` if fewer than 2 tests. Liquid falls back to oil when
    total-fluid is absent.
    """
    if test_well is None or test_well.empty or date_col not in test_well.columns:
        return {}
    d = test_well.dropna(subset=[date_col]).sort_values(date_col)
    n = len(d)
    if n < 2:
        return {}
    k = max(1, int(round(n * frac)))
    early, late = d.iloc[:k], d.iloc[-k:]

    def med(g, col):
        if col in g.columns:
            v = pd.to_numeric(g[col], errors="coerce").median()
            return float(v) if pd.notna(v) else np.nan
        return np.nan

    liq_then, liq_now = med(early, liquid_col), med(late, liquid_col)
    if np.isnan(liq_then):
        liq_then = med(early, oil_col)
    if np.isnan(liq_now):
        liq_now = med(late, oil_col)
    out = {
        "n_tests": int(n), "k": int(k),
        "whp_then": med(early, whp_col), "whp_now": med(late, whp_col),
        "bhp_then": med(early, bhp_col), "bhp_now": med(late, bhp_col),
        "oil_then": med(early, oil_col), "oil_now": med(late, oil_col),
        "liquid_then": liq_then, "liquid_now": liq_now,
    }
    try:
        span = late[date_col].median() - early[date_col].median()
        out["days"] = float(getattr(span, "days", np.nan))
    except Exception:
        out["days"] = np.nan
    out["d_whp"] = out["whp_now"] - out["whp_then"]
    out["d_bhp"] = out["bhp_now"] - out["bhp_then"]
    out["d_oil"] = out["oil_now"] - out["oil_then"]
    out["d_liquid"] = liq_now - liq_then
    return out


def predict_dbhp_from_curve(curve_well: dict | None, whp_then, whp_now):
    """Model-predicted ΔBHP for the *actual* header move, read off the swept
    BHP-vs-WHP curve: ``BHP_model(whp_now) − BHP_model(whp_then)`` by interpolation.

    Returns ``(dbhp, extrapolated)``; ``extrapolated`` is True when either WHP is
    outside the swept range (np.interp clamps to the endpoints, so the prediction
    is a floor/ceiling, not a true solve).
    """
    if not curve_well:
        return (np.nan, False)
    pts = sorted(
        (float(x), float(y))
        for x, y in zip(curve_well.get("whp") or [], curve_well.get("bhp") or [])
        if x is not None and y is not None and not pd.isna(x) and not pd.isna(y)
    )
    if len(pts) < 2 or pd.isna(whp_then) or pd.isna(whp_now):
        return (np.nan, False)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    extr = not (xs[0] <= whp_then <= xs[-1] and xs[0] <= whp_now <= xs[-1])
    return (float(np.interp(whp_now, xs, ys)) - float(np.interp(whp_then, xs, ys)), extr)


def backpressure_consistency(
    d_whp, d_bhp, d_liquid, whp_eps: float = 15.0, bhp_eps: float = 15.0,
    q_eps: float = 20.0,
) -> str:
    """Diagnose the observed window move against the nodal back-pressure signature:
    WHP↑ ⇒ BHP↑ (less drawdown) ⇒ liquid↓.

    Separates a genuine back-pressure response from reservoir depletion (rate falls
    with BHP flat — the IPR moved, not the operating point) and from contradictions
    (BHP moved opposite the header — gauge / pump-change / sonic). ``*_eps`` are
    dead-bands so noise near zero reads as 'flat', not a move.
    """
    def sgn(x, eps):
        if x is None or pd.isna(x):
            return None
        return 1 if x > eps else (-1 if x < -eps else 0)

    sw, sb, sq = sgn(d_whp, whp_eps), sgn(d_bhp, bhp_eps), sgn(d_liquid, q_eps)
    if sw is None or sw == 0:
        return "header ~flat (n/a)"
    if sb is None:                                   # gaugeless: lean on liquid only
        if sq == -sw:
            return "liquid consistent (no BHP)"
        return "liquid ~flat (no BHP)" if sq == 0 else "inconclusive (no BHP)"
    if sb == -sw:
        return "contradicts (BHP vs WHP)"
    if sb == sw:                                     # BHP tracked WHP — response link holds
        if sq == -sb:
            return "back-pressure-consistent ✓"
        return "BHP tracks WHP; liquid ~flat" if sq == 0 else "BHP tracks WHP; liquid wrong way"
    return "depletion-like / BHP flat"               # sb == 0: BHP flat despite WHP move


# ── windowed Vogel fits: back into a pseudo-pr that's allowed to deplete ──────

# Physical ceiling on the backed-out pseudo reservoir pressure, by formation —
# clustered/sparse tests can otherwise drive the LS fit to an unphysical pr. MPU
# caps (Scott, 2026-06-02): Schrader ~2200 psi, Kuparuk ~4200 psi (deeper/higher).
PR_MAX_DEFAULT = 3500.0
FORMATION_PR_MAX = {"Schrader": 2200.0, "Kuparuk": 4200.0}


def pr_hi_for_formation(formation) -> float:
    """Upper bound for the pseudo-Pr fit given a well's formation (defaults to
    ``PR_MAX_DEFAULT`` for unknown/missing)."""
    return float(FORMATION_PR_MAX.get(str(formation), PR_MAX_DEFAULT))


def vogel_oil(bhp, qmax: float, pr: float) -> float:
    """Vogel oil rate at a bottomhole pressure for a given qmax & reservoir pressure
    (bhp clamped to [0, pr]). The pure inverse of :func:`fit_vogel_ipr`, used to
    predict the oil change for a BHP move off a fitted pseudo-Pr IPR."""
    if pr is None or pd.isna(pr) or pr <= 0 or pd.isna(qmax) or pd.isna(bhp):
        return float("nan")
    r = min(max(float(bhp), 0.0), float(pr)) / float(pr)
    return float(qmax) * (1.0 - 0.2 * r - 0.8 * r * r)


def fit_vogel_ipr(pwf, q, pr_hi: float = PR_MAX_DEFAULT, n_grid: int = 240) -> dict | None:
    """Fit a Vogel IPR — qmax and a *pseudo* reservoir pressure ``pr`` — to a cluster
    of ``(pwf, q)`` test points by least squares (Standing-style back-out).

    Vogel: ``q = qmax·[1 − 0.2(pwf/pr) − 0.8(pwf/pr)²]``. We don't measure pr, so we
    back into it: for each candidate pr (> max pwf, so the Vogel factor stays > 0),
    ``qmax`` is the closed-form LS scale and we keep the pr minimizing the residual.

    Returns ``dict(qmax, pr, rmse, n, pwf_spread, pr_at_bound)`` or ``None`` for < 2
    usable points. With little pwf spread the pr is weakly constrained — small
    ``pwf_spread`` / ``pr_at_bound`` tell the caller the fit is soft (don't trust the
    absolute pr, only its trend across windows).
    """
    P = np.asarray(pwf, dtype=float)
    Q = np.asarray(q, dtype=float)
    m = np.isfinite(P) & np.isfinite(Q) & (P > 0) & (Q > 0)
    P, Q = P[m], Q[m]
    if P.size < 2:
        return None
    spread = float(P.max() - P.min())
    pr_lo = float(P.max()) * 1.02 + 1.0
    if pr_hi <= pr_lo:
        pr_hi = pr_lo + 500.0
    grid = np.linspace(pr_lo, pr_hi, n_grid)
    best = None
    for pr in grid:
        r = P / pr
        f = 1.0 - 0.2 * r - 0.8 * r * r
        denom = float(np.sum(f * f))
        if np.any(f <= 0) or denom <= 0:
            continue
        qmax = float(np.sum(Q * f) / denom)
        sse = float(np.sum((Q - qmax * f) ** 2))
        if best is None or sse < best[0]:
            best = (sse, qmax, float(pr))
    if best is None:
        return None
    sse, qmax, pr = best
    step = grid[1] - grid[0]
    return {
        "qmax": qmax, "pr": pr, "rmse": float(np.sqrt(sse / P.size)),
        "n": int(P.size), "pwf_spread": spread,
        "pr_at_bound": bool(abs(pr - grid[-1]) < step),
    }


def windowed_ipr_fits(
    test_well: pd.DataFrame, min_tests: int = 3, min_days: int = 25,
    max_days: int = 95, pr_hi: float = PR_MAX_DEFAULT, date_col: str = "WtDate",
    bhp_col: str = "BHP", oil_col: str = "WtOilVol",
) -> list[dict]:
    """Fit a Vogel IPR per time window so the curve is allowed to SHIFT over the
    lookback (depletion), instead of forcing one IPR on 12 months of data.

    Windows grow until they hold ``min_tests`` AND span ``min_days`` (hard-capped at
    ``max_days`` once they have ≥2 points), so they adapt to test density — ~1 month
    where tests are frequent, up to ~3 where sparse. Each window backs into a pseudo
    pr via :func:`fit_vogel_ipr`. The **pr trend across windows is the depletion
    signal**; points sliding along a single window's curve is the back-pressure
    response. Returns a time-ordered list of window dicts (start/end/mid/n_window +
    the fit fields); unfittable windows are dropped.
    """
    if test_well is None or test_well.empty or date_col not in test_well.columns:
        return []
    if bhp_col not in test_well.columns or oil_col not in test_well.columns:
        return []
    d = test_well.dropna(subset=[date_col, bhp_col, oil_col]).sort_values(date_col)
    n = len(d)
    if n < 2:
        return []
    ts = list(pd.to_datetime(d[date_col]))
    groups: list[tuple[int, int]] = []
    start = 0
    for k in range(1, n):
        cnt = k - start                       # window = indices [start .. k-1]
        span = (ts[k - 1] - ts[start]).days
        if (cnt >= min_tests and span >= min_days) or (span >= max_days and cnt >= 2):
            groups.append((start, k - 1))
            start = k
    groups.append((start, n - 1))
    # fold a too-small trailing window into its predecessor
    if len(groups) >= 2 and (groups[-1][1] - groups[-1][0] + 1) < 2:
        s0 = groups[-2][0]
        e1 = groups[-1][1]
        groups = groups[:-2] + [(s0, e1)]

    out = []
    for s, e in groups:
        sub = d.iloc[s:e + 1]
        fit = fit_vogel_ipr(sub[bhp_col].tolist(), sub[oil_col].tolist(), pr_hi=pr_hi)
        if fit is not None:
            sd, ed = ts[s], ts[e]
            out.append(
                {"start": sd, "end": ed, "mid": sd + (ed - sd) / 2,
                 "n_window": e - s + 1, **fit}
            )
    return out


def fit_well_ipr(
    test_well: pd.DataFrame, pr_hi: float = PR_MAX_DEFAULT, min_pts: int = 3,
    bhp_col: str = "BHP", rate_col: str = "WtTotalFluid",
) -> dict | None:
    """Single Vogel IPR fit to ALL a well's (BHP, rate) test points.

    Fit on **total liquid** (``WtTotalFluid``), NOT oil — Vogel is a liquid inflow
    relationship and the rest of the system (compute_vogel_coefficients,
    build_well_config) is liquid→WC→oil; fitting oil directly is a units mismatch
    when water cut varies. Convert the resulting ΔLiquid to ΔOil with (1−WC) downstream.

    Field-tested choice over per-window fitting: using every point maximizes the BHP
    spread (~280 vs ~50 psi/window). The absolute pr is still soft for many wells (it
    pins to the formation cap) — trust the IPR *shape*, read depletion from
    :func:`depletion_signature`. Returns ``fit_vogel_ipr``'s dict or ``None``.
    """
    if test_well is None or test_well.empty:
        return None
    if bhp_col not in test_well.columns or rate_col not in test_well.columns:
        return None
    d = test_well.dropna(subset=[bhp_col, rate_col])
    if len(d) < min_pts:
        return None
    return fit_vogel_ipr(d[bhp_col].tolist(), d[rate_col].tolist(), pr_hi=pr_hi)


def depletion_signature(
    test_well: pd.DataFrame, corr_min: float = 0.35, rate_eps: float = 10.0,
    min_pts: int = 4, date_col: str = "WtDate", bhp_col: str = "BHP",
    rate_col: str = "WtTotalFluid",
) -> dict:
    """Geometric depletion read from a well's operating points over time.

    Uses **total liquid** (``WtTotalFluid``) as the rate, to match the liquid IPR.
    On a FIXED IPR, rate and BHP are inversely related (Vogel: more drawdown → more
    rate). If instead BHP and rate move **together** (positive correlation), the IPR
    itself is shifting — down = depletion (Scott's "horizontal line"), up = rising.
    Robust because it keys on the *sign* of the BHP↔rate correlation, not a fragile
    absolute-pr fit.

    Returns ``{corr, bhp_per_mo, rate_per_mo, rate_bhp_slope, n, verdict}``:
      - ``back-pressure`` corr < -corr_min (riding a fixed IPR),
      - ``depleting``   corr > corr_min AND rate falling > rate_eps/mo (BHP not rising),
      - ``improving``   corr > corr_min AND rate rising > rate_eps/mo (BHP not falling),
      - ``flat/mixed``  correlated-but-small-trend, |corr| ≤ corr_min, or mixed trends,
      - ``insufficient`` < min_pts paired points or no BHP/rate spread.
    """
    out = {"corr": np.nan, "bhp_per_mo": np.nan, "rate_per_mo": np.nan,
           "rate_bhp_slope": np.nan, "n": 0, "verdict": "insufficient"}
    if test_well is None or test_well.empty:
        return out
    if not all(c in test_well.columns for c in (date_col, bhp_col, rate_col)):
        return out
    d = test_well.dropna(subset=[date_col, bhp_col, rate_col]).sort_values(date_col)
    bhp = pd.to_numeric(d[bhp_col], errors="coerce").to_numpy(dtype=float)
    rate = pd.to_numeric(d[rate_col], errors="coerce").to_numpy(dtype=float)
    days = (pd.to_datetime(d[date_col]) - pd.to_datetime(d[date_col]).min()).dt.days.to_numpy(dtype=float)
    ok = np.isfinite(bhp) & np.isfinite(rate) & np.isfinite(days)
    bhp, rate, days = bhp[ok], rate[ok], days[ok]
    out["n"] = int(bhp.size)
    if bhp.size < min_pts or np.std(bhp) < 1.0 or np.std(rate) < 1.0:
        return out
    out["corr"] = float(np.corrcoef(bhp, rate)[0, 1])
    # Direct rate sensitivity dLiquid/dBHP = linear slope of liquid on BHP (BPD/psi).
    # For a back-pressure well (corr<0) this IS the IPR slope at the operating range —
    # robust, no fragile Vogel pr. (For a depleting well it's the depletion line.)
    out["rate_bhp_slope"] = float(np.polyfit(bhp, rate, 1)[0])
    if np.ptp(days) > 0:
        out["bhp_per_mo"] = float(np.polyfit(days, bhp, 1)[0] * 30.0)
        out["rate_per_mo"] = float(np.polyfit(days, rate, 1)[0] * 30.0)
    corr, bt, ot = out["corr"], out["bhp_per_mo"], out["rate_per_mo"]
    if corr <= -corr_min:
        out["verdict"] = "back-pressure"
    elif corr >= corr_min and pd.notna(ot):
        # correlated → the rate trend's direction & size says depleting vs improving
        if ot < -rate_eps and (pd.isna(bt) or bt < 0):
            out["verdict"] = "depleting"
        elif ot > rate_eps and (pd.isna(bt) or bt > 0):
            out["verdict"] = "improving"
        else:
            out["verdict"] = "flat/mixed"     # correlated but no material/clean trend
    else:
        out["verdict"] = "flat/mixed"
    return out


# ── the deterministic per-well header-impact chain ───────────────────────────


def estimate_header_impact(
    d_whp, dbhp_dwhp, qmax, pr, bhp_now, wc, sonic: bool = False,
) -> dict:
    """Scott's per-well chain for a header change:

        ΔWHP ──①dBHP/dWHP──▶ ΔBHP ──②liquid Vogel IPR──▶ ΔLiquid ──③×(1−WC)──▶ ΔOil

    ① ``dbhp_dwhp`` is the ALREADY-RESOLVED WHP→BHP correlation (the well's own if good,
       else a donor / group average). Pass ``sonic=True`` to force ΔBHP = 0 — a jet pump
       in critical flow is choked, so the header can't propagate to BHP.
    ② ``qmax, pr`` are the well's liquid Vogel IPR (its own, or a similar-reservoir
       donor's). ΔLiquid is the FULL nonlinear delta off the curve, so a large move
       curves correctly.
    ③ ``wc`` is the water cut; ΔOil = ΔLiquid × (1 − WC).

    Returns ``{dbhp, bhp_scen, dliquid, doil}`` (NaN where inputs are missing). A sonic
    well returns dbhp=0, dliquid=0, doil=0 — explicitly "no header lever".
    """
    out = {"dbhp": np.nan, "bhp_scen": np.nan, "dliquid": np.nan, "doil": np.nan}
    if pd.isna(d_whp) or pd.isna(bhp_now):
        return out
    slope = 0.0 if sonic else (float(dbhp_dwhp) if pd.notna(dbhp_dwhp) else np.nan)
    if pd.isna(slope):
        return out
    dbhp = slope * float(d_whp)
    bhp_scen = float(bhp_now) + dbhp
    out["dbhp"] = dbhp
    out["bhp_scen"] = bhp_scen
    if pd.notna(qmax) and pd.notna(pr) and pr and pr > 0:
        dliq = vogel_oil(bhp_scen, qmax, pr) - vogel_oil(bhp_now, qmax, pr)
        out["dliquid"] = dliq
        out["doil"] = dliq * (1.0 - (float(wc) if pd.notna(wc) else 0.0))
    return out


def group_correlation_stats(wells: dict) -> dict:
    """Per-lift mean & std of the responsive wells' own dBHP/dWHP — the group
    correlation and its spread (drives the donor default and the field CI)."""
    def _ok(x):
        return x is not None and not (isinstance(x, float) and np.isnan(x))
    by_lift: dict = {}
    for b in wells.values():
        if _ok(b.get("own_corr")):
            by_lift.setdefault(b.get("lift", "?"), []).append(float(b["own_corr"]))
    return {k: (float(np.mean(v)), float(np.std(v)) if len(v) > 1 else 0.0)
            for k, v in by_lift.items() if v}


def estimate_header_impacts(wells: dict, d_whp, group_corr_override: dict | None = None) -> dict:
    """Resolve donors across a set of wells, then run :func:`estimate_header_impact`
    on each — the field-level orchestration of Scott's chain.

    ``wells`` = ``{name: {own_corr, sonic, lift, qmax, pr, pad, formation, bhp_now, wc,
    corr_donor, ipr_donor}}``. ``own_corr`` is the well's OWN dBHP/dWHP if good (else None);
    ``sonic`` the jet-pump critical-flow flag; ``qmax/pr`` its OWN liquid IPR (else None);
    ``corr_donor`` / ``ipr_donor`` (optional) a SPECIFIC well to borrow from.

    ① correlation ladder: **specific-well donor → own → (JP & sonic ⇒ 0) → group-avg by
       lift**. ② IPR ladder: **specific-well donor → own → group-avg by pad+formation**.
    ``group_corr_override`` ({lift: corr}) swaps the group correlation (used for the CI).

    Returns ``{name: {dbhp, bhp_scen, dliquid, doil, corr, corr_src, ipr_src, conf, qmax, pr}}``.
    """
    def _ok(x):
        return x is not None and not (isinstance(x, float) and np.isnan(x))

    grp_corr = {k: m for k, (m, _s) in group_correlation_stats(wells).items()}
    if group_corr_override:
        grp_corr = {**grp_corr, **group_corr_override}

    grp_ipr: dict = {}
    for b in wells.values():
        if _ok(b.get("qmax")) and _ok(b.get("pr")):
            grp_ipr.setdefault((b.get("pad"), b.get("formation")), []).append(
                (float(b["qmax"]), float(b["pr"])))
    grp_ipr = {k: (float(np.mean([q for q, _ in v])), float(np.mean([p for _, p in v])))
               for k, v in grp_ipr.items() if v}

    out: dict = {}
    for name, b in wells.items():
        lift, sonic = b.get("lift", "?"), bool(b.get("sonic", False))
        cd = b.get("corr_donor")
        if cd and _ok(wells.get(cd, {}).get("own_corr")):
            corr, csrc = float(wells[cd]["own_corr"]), "donor"     # specific-well override wins
        elif _ok(b.get("own_corr")):
            corr, csrc = float(b["own_corr"]), "own"
        elif sonic and lift == "JP":
            corr, csrc = 0.0, "sonic"
        elif lift in grp_corr:
            corr, csrc = grp_corr[lift], "group"
        else:
            corr, csrc = np.nan, "none"

        idn = b.get("ipr_donor")
        if idn and _ok(wells.get(idn, {}).get("qmax")) and _ok(wells.get(idn, {}).get("pr")):
            qmax, pr, isrc = float(wells[idn]["qmax"]), float(wells[idn]["pr"]), "donor"
        elif _ok(b.get("qmax")) and _ok(b.get("pr")):
            qmax, pr, isrc = b["qmax"], b["pr"], "own"
        elif (b.get("pad"), b.get("formation")) in grp_ipr:
            qmax, pr, isrc = *grp_ipr[(b.get("pad"), b.get("formation"))], "group"
        else:
            qmax, pr, isrc = np.nan, np.nan, "none"

        est = estimate_header_impact(d_whp, corr, qmax, pr, b.get("bhp_now"),
                                     b.get("wc"), sonic=(csrc == "sonic"))
        if csrc == "sonic":
            conf = "sonic"
        elif csrc == "none" or isrc == "none":
            conf = "none"
        elif csrc == "own" and isrc == "own":
            conf = "high"
        else:
            conf = "med"
        out[name] = {**est, "corr": corr, "corr_src": csrc, "ipr_src": isrc,
                     "conf": conf, "qmax": qmax, "pr": pr}
    return out


# ── response-curve aggregation (per-pad + overall) ───────────────────────────


def aggregate_response_curve(curve: dict | None) -> dict:
    """Aggregate per-well swept oil into per-pad and overall ΔOil-vs-header curves.

    ``curve`` = ``{"deltas": [...], "wells": {well: {"pad": p, "oil": [oil/delta]}}}``.
    Returns ``{"deltas": [...], "pads": {pad: [Δoil/delta]}, "ALL": [Δoil/delta]}``
    where Δoil is relative to the delta==0 baseline (nearest-to-zero if 0 absent).
    Wells whose oil list length doesn't match ``deltas`` are skipped.
    """
    if not curve or "wells" not in curve or "deltas" not in curve:
        return {}
    deltas = list(curve["deltas"])
    if not deltas:
        return {}
    try:
        zero_idx = deltas.index(0)
    except ValueError:
        zero_idx = min(range(len(deltas)), key=lambda i: abs(deltas[i]))
    pads: dict[str, list[float]] = {}
    all_curve = [0.0] * len(deltas)
    for _well, wd in curve["wells"].items():
        oils = wd.get("oil") or []
        if len(oils) != len(deltas):
            continue
        base = oils[zero_idx]
        if base is None or (isinstance(base, float) and np.isnan(base)):
            continue
        pad = str(wd.get("pad", "?"))
        pad_arr = pads.setdefault(pad, [0.0] * len(deltas))
        for i, o in enumerate(oils):
            if o is None or (isinstance(o, float) and np.isnan(o)):
                continue
            d = o - base
            pad_arr[i] += d
            all_curve[i] += d
    return {"deltas": deltas, "pads": pads, "ALL": all_curve}


# ── like-wells donor assignment (G3) ─────────────────────────────────────────

OWN_TOKEN = "(own)"
GROUP_PADLIFT = "(group: pad+lift)"
GROUP_PADFORMATION = "(group: pad+formation)"
GROUP_FORMATION = "(group: formation)"
GROUP_LIFT = "(group: lift)"
DONOR_GROUP_TOKENS = (GROUP_PADLIFT, GROUP_PADFORMATION, GROUP_FORMATION, GROUP_LIFT)


def donor_tokens(well_names: list[str]) -> list[str]:
    """The selectbox options for a donor column: own, the group schemes, then
    every well (so a specific donor can be picked).

    Group schemes — pad+lift / pad+formation borrow from nearby analogs on the
    same pad; formation / lift borrow field-wide. The coverage panel auto-fills
    pad+formation for wells with no IPR test data and lift for wells with no
    usable WHP→BHP correlation (the widest pool of like-lift analogs)."""
    return [
        OWN_TOKEN, GROUP_PADLIFT, GROUP_PADFORMATION, GROUP_FORMATION, GROUP_LIFT
    ] + sorted(well_names)


def donor_member_wells(target: str, token: str, rows_meta: dict | None):
    """Resolve a donor token to the source wells whose IPR/correlation to average.

    ``rows_meta`` = ``{well: {"pad":.., "lift":.., "formation":..}}`` for the
    selected set. Returns ``None`` for OWN_TOKEN (caller uses the well's own);
    otherwise a non-empty list of member wells (group averages include the target
    itself, which is fine — it's the group's representative value). A specific
    well token returns ``[that_well]`` (so a single named donor is used directly).
    """
    if not token or token == OWN_TOKEN:
        return None
    meta = rows_meta or {}
    tmeta = meta.get(target, {})
    if token == GROUP_PADLIFT:
        members = [
            w for w, m in meta.items()
            if m.get("pad") == tmeta.get("pad") and m.get("lift") == tmeta.get("lift")
        ]
        return members or None
    if token == GROUP_PADFORMATION:
        members = [
            w for w, m in meta.items()
            if m.get("pad") == tmeta.get("pad") and m.get("formation") == tmeta.get("formation")
        ]
        return members or None
    if token == GROUP_FORMATION:
        members = [w for w, m in meta.items() if m.get("formation") == tmeta.get("formation")]
        return members or None
    if token == GROUP_LIFT:
        members = [w for w, m in meta.items() if m.get("lift") == tmeta.get("lift")]
        return members or None
    return [token] if token in meta else None


def average_vogel_rows(rows: list[dict] | None) -> dict | None:
    """Average a list of Vogel/IPR rows into one (mean of ResP/qwf/pwf/form_wc/fgor).

    Returns ``None`` unless the average has the keys ``build_well_config`` needs
    (ResP, qwf, pwf), so callers can safely fall back to the well's own IPR.
    """
    rows = [r for r in (rows or []) if r]
    if not rows:
        return None
    out: dict = {}
    for k in ("ResP", "qwf", "pwf", "form_wc", "fgor"):
        vals = [float(r[k]) for r in rows if k in r and not pd.isna(r.get(k))]
        if vals:
            out[k] = sum(vals) / len(vals)
    if not all(k in out for k in ("ResP", "qwf", "pwf")):
        return None
    return out


def average_slope(slopes) -> float | None:
    """Mean of the usable (non-NaN) within-day BHP~WHP slopes; None if none."""
    vals = [float(s) for s in (slopes or []) if s is not None and not pd.isna(s)]
    return (sum(vals) / len(vals)) if vals else None


def describe_donor(token: str, n_members: int = 0) -> str:
    """Short provenance label for a donor column in the results table."""
    if not token or token == OWN_TOKEN:
        return "own"
    if token == GROUP_PADLIFT:
        return f"group pad+lift (n={n_members})"
    if token == GROUP_PADFORMATION:
        return f"group pad+formation (n={n_members})"
    if token == GROUP_FORMATION:
        return f"group formation (n={n_members})"
    if token == GROUP_LIFT:
        return f"group lift (n={n_members})"
    return f"donor {token}"


def corr_display_plan(results_df, available_wells) -> dict:
    """Decide, per well used in the impact, which well's WHP→BHP trend to draw in
    the correlation grid and a title note — so EVERY well gets a panel (its own, a
    donor's, or a labeled 'using group correlation' note).

    ``available_wells`` = wells that actually have a usable within-day trend.
    Reads the ``Corr donor`` provenance produced by ``describe_donor``. Returns
    ``{well: {"source": well_or_None, "note": str, "slope": float}}``.
    """
    avail = set(available_wells or [])
    plan: dict = {}
    if results_df is None or "Well" not in getattr(results_df, "columns", []):
        return plan
    for _, r in results_df.iterrows():
        wn = r["Well"]
        donor = str(r.get("Corr donor", "own"))
        slope = r.get("Emp dBHP/dWHP")
        if donor.startswith("donor "):
            src = donor[len("donor "):].strip()
            plan[wn] = {"source": (src if src in avail else None),
                        "note": f"using {src} corr", "slope": slope}
        elif donor.startswith("group"):
            plan[wn] = {"source": None, "note": f"using {donor}", "slope": slope}
        else:  # own / — / blank
            plan[wn] = {"source": (wn if wn in avail else None), "note": "", "slope": slope}
    return plan
