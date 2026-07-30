"""Live power-fluid surface pressure from mpu.wells.vw_pressure_daily.

The view carries one row per well(bore) per day with tubing_prs, inn_ann_prs,
out_ann_prs, btmhole_prs (daily samples back to 2024-09-25). Which column is
the power fluid depends on the well's circulation direction:

  * reverse circ (standard JP): PF goes down the ANNULUS — inn_ann_prs is the
    PF pressure (~2500-3400 psi) and the tubing carries production
    (~170-400 psi).
  * forward circ: PF goes down the TUBING — tubing_prs is the PF pressure and
    reads ABOVE inn_ann_prs (e.g. F-002/F-021/F-058/F-077 at ~2750, E-17 at
    ~1815, L-20 at ~1830).

:func:`resolve_pf_pressure` encodes that rule, so callers never pick a column
themselves. Data quirks handled here so callers don't have to:

  * the view can return MULTIPLE rows per enthid+sample_date (same wellbore,
    repeated samples — seen on I-24/I-26/S-53); queries aggregate max() per
    well+day,
  * dead/absent gauges read 0 or NULL (~10% of recent rows) — a reading below
    PF_MIN_VALID can't be power fluid and is treated as missing,
  * columns are SQL decimal — everything is cast to float.
"""

from typing import Optional, Tuple

import pandas as pd

from woffl.assembly.databricks_client import execute_query

# A reading below this can't be the power fluid: production tubing runs
# ~170-400 psi and dead gauges read 0, while the PF headers run ~1300-3400 psi.
# Matches the solver/auto-match floor (_PPF_LO) and the sidebar widget minimum
# (sidebar.SEED_BOUNDS["ppf_surf"][0]) so a resolved value is always seedable.
PF_MIN_VALID = 800.0

# Human-readable labels for the pf_source codes, for captions/tooltips.
PF_SOURCE_LABELS = {
    "annulus": "annulus (reverse circ)",
    "tubing": "tubing (forward circ)",
}


def _valid_pf(value) -> Optional[float]:
    """Cast to float; None for NULL/NaN/dead-gauge readings below PF_MIN_VALID."""
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(val) or val < PF_MIN_VALID:
        return None
    return val


def resolve_pf_pressure(tubing_prs, inn_ann_prs) -> Tuple[Optional[float], Optional[str]]:
    """Resolve the PF surface pressure from a day's tubing/annulus readings.

    Rule: PF is inn_ann_prs (reverse circ) unless the well is forward circ, in
    which case tubing_prs > inn_ann_prs and the tubing carries the power fluid.

    Returns ``(pressure_psi, source)`` where source is ``"annulus"`` /
    ``"tubing"`` (see PF_SOURCE_LABELS), or ``(None, None)`` when neither
    reading is a plausible PF pressure.
    """
    tub = _valid_pf(tubing_prs)
    ann = _valid_pf(inn_ann_prs)
    if tub is not None and (ann is None or tub > ann):
        return tub, "tubing"
    if ann is not None:
        return ann, "annulus"
    return None, None


def add_pf_columns(
    df: pd.DataFrame,
    tubing_col: str = "pf_tubing_prs",
    ann_col: str = "pf_inn_ann_prs",
) -> pd.DataFrame:
    """Add resolved ``pf_press`` / ``pf_source`` columns to a DataFrame in place.

    Tolerates missing input columns (e.g. mocked test frames without the
    vw_pressure_daily join) — the pf columns are then all-NaN/None, so
    consumers can uniformly check ``pf_press``.
    """
    if tubing_col in df.columns or ann_col in df.columns:
        tub = df.get(tubing_col, pd.Series(index=df.index, dtype=float))
        ann = df.get(ann_col, pd.Series(index=df.index, dtype=float))
        resolved = [resolve_pf_pressure(t, a) for t, a in zip(tub, ann)]
        df["pf_press"] = [r[0] for r in resolved]
        df["pf_source"] = [r[1] for r in resolved]
    else:
        df["pf_press"] = None
        df["pf_source"] = None
    df["pf_press"] = pd.to_numeric(df["pf_press"], errors="coerce")
    return df


# max() per well+day collapses the duplicate-sample rows and prefers an
# operating reading over a shut-in/dead 0 taken the same day.
PF_LATEST_QUERY = """\
SELECT
    well_name,
    sample_date,
    max(tubing_prs) AS tubing_prs,
    max(inn_ann_prs) AS inn_ann_prs
FROM mpu.wells.vw_pressure_daily
WHERE sample_date >= date_sub(current_date(), {days_back})
GROUP BY well_name, sample_date
"""


def fetch_pf_latest(days_back: int = 30) -> pd.DataFrame:
    """Latest valid PF surface pressure per well from vw_pressure_daily.

    Pulls the last ``days_back`` days for ALL wells in one query, resolves the
    PF pressure per day, and keeps each well's most recent day with a valid
    reading — so a well whose gauge dropped out yesterday still reports its
    last good value instead of nothing.

    Returns a DataFrame with columns ``well`` (GUI-normalized, e.g. MPB-28),
    ``pf_press``, ``pf_source`` ("annulus"/"tubing"), ``pf_date``,
    ``tubing_prs``, ``inn_ann_prs``. Empty DataFrame when the view returns
    nothing.
    """
    # Lazy import: well_test_client imports from this module, so a top-level
    # import here would be circular.
    from woffl.assembly.well_test_client import _normalize_well_name

    df = execute_query(PF_LATEST_QUERY.format(days_back=int(days_back)))
    cols = ["well", "pf_press", "pf_source", "pf_date", "tubing_prs", "inn_ann_prs"]
    if df.empty:
        return pd.DataFrame(columns=cols)

    for col in ("tubing_prs", "inn_ann_prs"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = add_pf_columns(df, tubing_col="tubing_prs", ann_col="inn_ann_prs")
    df = df[df["pf_press"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=cols)

    df["sample_date"] = pd.to_datetime(df["sample_date"], errors="coerce")
    # tail(1), not .last(): .last() takes the last non-NaN value per COLUMN,
    # which can stitch together readings from different days.
    df = df.sort_values("sample_date").groupby("well_name", as_index=False).tail(1)
    df["well"] = df["well_name"].astype(str).str.strip().map(_normalize_well_name)
    df = df.rename(columns={"sample_date": "pf_date"})
    return df[cols].reset_index(drop=True)


def pad_pf_medians(
    pf_latest: pd.DataFrame, sources: tuple = ("annulus",)
) -> dict[str, int]:
    """Per-pad median of the latest live PF pressures, rounded to 50 psi.

    Keyed by pad letter ("B", "I", ...). Defaults to annulus-source wells
    only: an annulus reading >= PF_MIN_VALID is unambiguously JP power fluid,
    while a tubing-source value can be an ESP's flowing tubing pressure — on
    ESP-majority pads (e.g. J) those drag the all-source median down to ~900.
    Pads with no qualifying well are absent (callers fall back to pad
    defaults). Per-WELL seeds still use both sources via resolve_pf_pressure;
    this filter is only for pad-level broadcast defaults.
    """
    if pf_latest is None or pf_latest.empty:
        return {}
    df = pf_latest[pf_latest["pf_press"].notna()].copy()
    if sources:
        df = df[df["pf_source"].isin(sources)]
    if df.empty:
        return {}
    pads = (
        df["well"].astype(str).str.replace("MP", "", n=1).str.split("-").str[0]
    )
    medians = df.groupby(pads)["pf_press"].median()
    return {pad: int(round(val / 50.0) * 50) for pad, val in medians.items() if pad}


def pad_pf_cluster(
    pf_latest: pd.DataFrame,
    tol_psi: float = 150.0,
    sources: tuple = ("annulus",),
) -> dict[str, dict]:
    """Per-pad PF header pressure from the HIGH CLUSTER rather than the median.

    :func:`pad_pf_medians` is right for JP-majority pads (S, I) — most wells sit
    on the PF header, so the median IS the header. On **mixed-lift pads it is
    wrong**, because only a minority do and the median averages header wells
    against unrelated ones. Measured on the CFP pads 2026-07-29:

    * G-Pad had two annulus readings, 2,748 and 1,420 psi. The "median" of 2,084
      is just their midpoint — a pressure no well is at.
    * C-Pad's median of 1,208 hid its real ~3,404 psi on-pad booster.

    The header is the TIGHT CLUSTER at the top: take the highest valid reading
    and keep everything within ``tol_psi`` of it. On B-Pad that recovers 5 wells
    spanning 11 psi (2,619-2,630) and rejects two tubing readings near 900.

    Returns ``{pad: {"psi", "n_cluster", "n_pad", "max", "min"}}`` — the counts
    matter: a one-well cluster (G) is a far weaker basis than a five-well one
    (B), and the caller should be able to say so rather than being handed a bare
    number. Pads with no qualifying well are absent.
    """
    if pf_latest is None or pf_latest.empty:
        return {}
    df = pf_latest[pf_latest["pf_press"].notna()].copy()
    if sources and "pf_source" in df.columns:
        df = df[df["pf_source"].isin(sources)]
    df = df[df["pf_press"] >= PF_MIN_VALID]
    if df.empty:
        return {}

    df["_pad"] = (
        df["well"].astype(str).str.replace("MP", "", n=1).str.split("-").str[0]
    )
    out: dict[str, dict] = {}
    for pad, grp in df.groupby("_pad"):
        if not pad:
            continue
        top = float(grp["pf_press"].max())
        cluster = grp[grp["pf_press"] >= top - float(tol_psi)]["pf_press"]
        out[pad] = {
            "psi": float(cluster.median()),
            "n_cluster": int(len(cluster)),
            "n_pad": int(len(grp)),
            "max": top,
            "min": float(cluster.min()),
        }
    return out
