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
GROUP_FORMATION = "(group: formation)"
DONOR_GROUP_TOKENS = (GROUP_PADLIFT, GROUP_FORMATION)


def donor_tokens(well_names: list[str]) -> list[str]:
    """The selectbox options for a donor column: own, the two group schemes,
    then every well (so a specific donor can be picked)."""
    return [OWN_TOKEN, GROUP_PADLIFT, GROUP_FORMATION] + sorted(well_names)


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
    if token == GROUP_FORMATION:
        members = [w for w, m in meta.items() if m.get("formation") == tmeta.get("formation")]
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
    if token == GROUP_FORMATION:
        return f"group formation (n={n_members})"
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
