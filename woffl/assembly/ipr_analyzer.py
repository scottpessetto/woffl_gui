"""IPR Analyzer

Computes Vogel IPR parameters from merged well test + BHP data.
Estimates optimal reservoir pressure per well and generates
Vogel coefficients suitable for the multi-well optimization template.

Adapted from header_pressure_impact/process_data/calc_PI_RP.py and bhp_liq.py.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from woffl.flow.inflow import InFlow

logger = logging.getLogger(__name__)


def _axis_scales(
    bhp: np.ndarray, fluid: np.ndarray
) -> tuple[float, float]:
    """Candidate-independent normalization scales for the two plot axes.

    Data-cloud spread with a floor of 5% of the cloud's magnitude, so a
    perfectly flat axis (all tests at the same BHP) can't blow residuals up
    to infinity, and neither axis dominates purely by its units.
    """
    q_scale = max(float(np.std(fluid)), 0.05 * float(np.median(np.abs(fluid))), 1.0)
    p_scale = max(float(np.std(bhp)), 0.05 * float(np.median(np.abs(bhp))), 1.0)
    return q_scale, p_scale


def _normalized_curve_sse(
    bhp: np.ndarray,
    fluid: np.ndarray,
    pres: float,
    qmax: np.ndarray,
    q_scale: float,
    p_scale: float,
) -> np.ndarray:
    """Axis-normalized point-to-curve SSE for one candidate RP, per anchor.

    For each anchored Vogel curve (one qmax per anchor) and each test point,
    SUMS the squared RATE-direction residual (predicted rate at the point's
    BHP) and the squared PRESSURE-direction residual (Vogel inverted for BHP
    at the point's rate), both axis-normalized: ``d² = u² + v²``.

    Why the SUM and not an orthogonal-distance formula: the old rate-only
    SSE railed the RP search to the field cap on flat test clouds (BHP
    barely moves, rate scatter is allocation noise — e.g. B-28 2026-07:
    corr(BHP, rate) = −0.05), because a flatter curve (higher RP) always
    shrinks rate residuals. A pressure-only SSE rails the other way (RP →
    just above max BHP, Qmax exploding). The harmonic/orthogonal combination
    ``u²v²/(u²+v²)`` degenerates to whichever residual is smaller and rails
    low too (verified on B-28). Summing penalizes BOTH failure modes — the
    steep low-RP curve blows up the rate term, the flat high-RP curve blows
    up the pressure term — so the total has an interior minimum, and on
    informative clouds both terms minimize at the true RP together. For
    points beyond the curve's reach (rate > qmax) the vertical residual is
    taken to the curve's p=0 endpoint.

    Returns an ``(m,)`` array: total SSE per anchor. Points with
    BHP >= pres contribute the legacy 1e8 penalty each.
    """
    ratio = bhp / pres
    vogel = 1.0 - 0.2 * ratio - 0.8 * ratio**2
    valid_j = bhp < pres

    pred_q = np.outer(qmax, np.where(valid_j, vogel, 0.0))  # (m, n)
    du = (pred_q - fluid[None, :]) / q_scale

    # Invert Vogel for the pressure-direction residual: for exact on-curve
    # data disc = (0.2 + 1.6·ratio)², so r_pred reproduces ratio exactly.
    with np.errstate(divide="ignore", invalid="ignore"):
        qratio = fluid[None, :] / qmax[:, None]
    disc = 0.04 + 3.2 * (1.0 - qratio)
    r_pred = (-0.2 + np.sqrt(np.clip(disc, 0.0, None))) / 1.6
    pred_p = np.where(disc >= 0.0, r_pred * pres, 0.0)
    dv = (pred_p - bhp[None, :]) / p_scale

    d2 = du * du + dv * dv
    d2[:, ~valid_j] = 1e8  # penalty for BHP >= RP, same as the original
    return d2.sum(axis=1)


def _calculate_global_sse(
    bhp_values: np.ndarray, fluid_values: np.ndarray, pres: float
) -> float:
    """Calculate the fit error for a candidate reservoir pressure.

    For each test point used as the anchor, computes the Vogel curve and
    measures the axis-normalized distance (see :func:`_normalized_curve_sse`)
    to ALL test points. Returns the minimum SSE across all possible anchor
    points — the RP where some anchored Vogel curve best passes through the
    cloud as drawn on the plot.

    Args:
        bhp_values: Array of BHP values for the well
        fluid_values: Array of total fluid rate values for the well
        pres: Candidate reservoir pressure (must be > max BHP)

    Returns:
        Minimum normalized SSE across all anchor choices
    """
    n = len(bhp_values)
    if n < 2:
        return float("inf")

    bhp = np.asarray(bhp_values, dtype=float)
    fluid = np.asarray(fluid_values, dtype=float)

    ratio = bhp / pres
    vogel = 1.0 - 0.2 * ratio - 0.8 * ratio**2

    # Valid anchors: BHP below the candidate RP, positive rate, positive
    # Vogel denominator (the denom guard mirrors the original loop).
    valid_anchor = (bhp < pres) & (fluid > 0) & (vogel > 0)
    if not valid_anchor.any():
        return float("inf")

    qmax = fluid[valid_anchor] / vogel[valid_anchor]  # (m,)
    q_scale, p_scale = _axis_scales(bhp, fluid)
    return float(
        _normalized_curve_sse(bhp, fluid, pres, qmax, q_scale, p_scale).min()
    )


def _calculate_r_squared(
    bhp_values: np.ndarray,
    fluid_values: np.ndarray,
    pres: float,
    anchor_bhp: float,
    anchor_fluid: float,
) -> float:
    """Calculate R² goodness-of-fit for a Vogel curve.

    Args:
        bhp_values: Array of BHP values
        fluid_values: Array of total fluid rate values
        pres: Reservoir pressure
        anchor_bhp: BHP of the anchor test point
        anchor_fluid: Fluid rate of the anchor test point

    Returns:
        R² value (1.0 = perfect fit, 0.0 = no better than mean)
    """
    if len(fluid_values) < 2 or anchor_bhp >= pres:
        return 0.0

    try:
        ratio = anchor_bhp / pres
        denom = 1.0 - 0.2 * ratio - 0.8 * ratio**2
        if denom <= 0:
            return 0.0
        qmax = anchor_fluid / denom

        # Accumulate residuals AND the total-sum-of-squares over the SAME subset
        # (points below RP). Previously ss_res skipped bhp>=pres points while
        # ss_tot/mean were over all points, biasing R² whenever any test sat at
        # or above the fitted RP.
        ss_res = 0.0
        used_fluid = []
        for j in range(len(bhp_values)):
            if bhp_values[j] >= pres:
                continue
            ratio_j = bhp_values[j] / pres
            predicted = qmax * (1.0 - 0.2 * ratio_j - 0.8 * ratio_j**2)
            ss_res += (predicted - fluid_values[j]) ** 2
            used_fluid.append(fluid_values[j])

        if len(used_fluid) < 2:
            return 0.0
        used_arr = np.asarray(used_fluid, dtype=float)
        ss_tot = np.sum((used_arr - used_arr.mean()) ** 2)

        if ss_tot == 0:
            return 0.0

        return 1.0 - (ss_res / ss_tot)
    except (ValueError, ZeroDivisionError):
        return 0.0


def estimate_reservoir_pressure(
    merged_data: pd.DataFrame,
    max_pres_schrader: int = 1800,
    max_pres_kuparuk: int = 3000,
    jp_chars: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Estimate optimal reservoir pressure for each well.

    Uses a global least-squares approach: for each candidate RP, tries
    every test point as the Vogel anchor and picks the combination that
    minimizes total squared error across ALL test points.

    The search starts just above the max BHP (not +100) to allow
    lower RP values that fit the data more tightly.

    Args:
        merged_data: DataFrame with 'well', 'BHP', 'WtTotalFluid' columns
        max_pres_schrader: Maximum reservoir pressure for Schrader wells
        max_pres_kuparuk: Maximum reservoir pressure for Kuparuk wells
        jp_chars: Optional DataFrame with well characteristics (for is_sch flag)

    Returns:
        Input DataFrame with added 'Optimal_RP' and 'PI' columns
    """
    df = merged_data.copy()
    unique_wells = df["well"].unique()
    optimal_pres = {}

    # Build lookup for field model
    is_schrader = {}
    if jp_chars is not None and "Well" in jp_chars.columns:
        for _, row in jp_chars.iterrows():
            is_schrader[row["Well"]] = bool(row.get("is_sch", True))

    for well in unique_wells:
        well_data = df[df["well"] == well].dropna(subset=["BHP", "WtTotalFluid"])
        max_bhp = well_data["BHP"].max()

        if pd.isna(max_bhp) or well_data.empty:
            print(f"Warning: No valid BHP data for well {well}. Skipping.")
            continue

        bhp_values = well_data["BHP"].values.astype(float)
        fluid_values = well_data["WtTotalFluid"].values.astype(float)

        # Determine max pressure based on field model
        if well in is_schrader:
            max_pres = max_pres_schrader if is_schrader[well] else max_pres_kuparuk
        else:
            max_pres = max_pres_schrader

        min_sse = float("inf")
        best_pres = None

        # Start search just above max BHP (not +100) to allow tighter fits
        start_pres = int(max_bhp) + 10
        end_pres = max_pres

        if start_pres >= end_pres:
            # Clamp to the field cap — int(max_bhp)+50 could exceed max_pres
            # (e.g. a Schrader well with max BHP ~1795 -> 1845 > the 1800 cap).
            best_pres = min(int(max_bhp) + 50, max_pres)
        else:
            # Search with fine resolution near max_bhp, coarser further out
            for pres in range(start_pres, end_pres, 5):
                sse = _calculate_global_sse(bhp_values, fluid_values, float(pres))
                if sse < min_sse:
                    min_sse = sse
                    best_pres = pres

        # If no candidate produced a finite SSE (e.g. degenerate all-<=0 fluid),
        # best_pres is still None -> NaN Optimal_RP -> the well silently drops
        # out downstream. Fall back to a sensible RP just above max BHP instead.
        if best_pres is None:
            best_pres = min(int(max_bhp) + 50, max_pres)

        optimal_pres[well] = best_pres

    df["Optimal_RP"] = df["well"].map(optimal_pres)

    # Calculate productivity index
    def calc_pi(row):
        if pd.isna(row["Optimal_RP"]) or pd.isna(row["BHP"]):
            return np.nan
        denom = row["Optimal_RP"] - row["BHP"]
        if denom <= 0:
            return np.nan
        return row["WtTotalFluid"] / denom

    df["PI"] = df.apply(calc_pi, axis=1)
    return df


def compute_vogel_coefficients(
    merged_data_with_rp: pd.DataFrame,
    resp_modifier: int = 0,
) -> pd.DataFrame:
    """Compute Vogel IPR coefficients for each well.

    For each well, creates Vogel curves using:
    - Most recent test point
    - Lowest BHP test point
    - Median BHP test point

    Adapted from header_pressure_impact/process_data/bhp_liq.py plot_bhp_liquidrate_r2().

    Args:
        merged_data_with_rp: DataFrame with 'well', 'BHP', 'WtTotalFluid',
            'WtDate', 'Optimal_RP' columns
        resp_modifier: Offset to add to optimal RP (default 150 psi)

    Returns:
        DataFrame with columns: Well, ResP, QMax_recent, QMax_lowest_bhp,
        QMax_median, qwf, pwf, form_wc, num_tests
    """
    df = merged_data_with_rp.copy()
    df["date"] = pd.to_datetime(df["WtDate"])
    current_date = pd.to_datetime("today")
    df["days_since"] = (current_date - df["date"]).dt.days

    unique_wells = df["well"].unique()
    coeffs_list = []

    for well in sorted(unique_wells):
        well_data = df[df["well"] == well].copy()
        well_data = well_data.dropna(subset=["BHP", "WtTotalFluid", "Optimal_RP"])

        if well_data.empty or len(well_data) < 1:
            continue

        optimal_res_p = well_data["Optimal_RP"].iloc[0] + resp_modifier

        if pd.isna(optimal_res_p):
            continue

        try:
            # Sort by recency (most recent first)
            well_data = well_data.sort_values(by="days_since")

            # Most recent test
            recent_fluid = well_data["WtTotalFluid"].iloc[0]
            recent_bhp = well_data["BHP"].iloc[0]
            vogel_recent = InFlow(recent_fluid, recent_bhp, optimal_res_p)
            qmax_recent = vogel_recent.vogel_qmax(
                recent_fluid, recent_bhp, optimal_res_p
            )

            # Lowest BHP test
            well_data_by_bhp = well_data.sort_values(by="BHP")
            lowest_fluid = well_data_by_bhp["WtTotalFluid"].iloc[0]
            lowest_bhp = well_data_by_bhp["BHP"].iloc[0]
            vogel_lowest = InFlow(lowest_fluid, lowest_bhp, optimal_res_p)
            qmax_lowest = vogel_lowest.vogel_qmax(
                lowest_fluid, lowest_bhp, optimal_res_p
            )

            # Median BHP test
            median_bhp_val = well_data["BHP"].median()
            closest_row = well_data.iloc[
                (well_data["BHP"] - median_bhp_val).abs().argsort()[:1]
            ]
            median_fluid = closest_row["WtTotalFluid"].values[0]
            median_bhp = closest_row["BHP"].values[0]
            vogel_median = InFlow(median_fluid, median_bhp, optimal_res_p)
            qmax_median = vogel_median.vogel_qmax(
                median_fluid, median_bhp, optimal_res_p
            )

            # Calculate watercut from most recent test
            wc = 0.5  # default
            if (
                "WtWaterVol" in well_data.columns
                and "WtTotalFluid" in well_data.columns
            ):
                total = well_data["WtTotalFluid"].iloc[0]
                water = (
                    well_data["WtWaterVol"].iloc[0]
                    if "WtWaterVol" in well_data.columns
                    else 0
                )
                if total > 0:
                    wc = water / total

            # Calculate R² fit quality using the median anchor
            bhp_vals = well_data["BHP"].values.astype(float)
            fluid_vals = well_data["WtTotalFluid"].values.astype(float)
            r_squared = _calculate_r_squared(
                bhp_vals, fluid_vals, optimal_res_p, median_bhp, median_fluid
            )

            coeffs_list.append(
                {
                    "Well": well,
                    "ResP": optimal_res_p,
                    "QMax_recent": qmax_recent,
                    "QMax_lowest_bhp": qmax_lowest,
                    "QMax_median": qmax_median,
                    "qwf": recent_fluid,
                    "pwf": recent_bhp,
                    "form_wc": round(wc, 3),
                    "fgor": (
                        well_data["fgor"].iloc[0]
                        if "fgor" in well_data.columns
                        and pd.notna(well_data["fgor"].iloc[0])
                        else 250
                    ),
                    "num_tests": len(well_data),
                    "most_recent_date": well_data["date"].iloc[0],
                    "R2": round(r_squared, 3),
                }
            )
        except (ValueError, ZeroDivisionError, KeyError, IndexError) as e:
            # Expected per-well data problems (e.g. InFlow pwf>=pres, bad rows):
            # skip this well but keep going. Unexpected exceptions (e.g. a column
            # rename after an upstream sync) propagate so a SYSTEMIC failure is
            # visible instead of silently degrading every well.
            logger.warning("Skipping Vogel coefficients for %s: %s", well, e)
            continue

    return pd.DataFrame(coeffs_list)


def generate_ipr_curves(
    vogel_coeffs: pd.DataFrame,
) -> Dict[str, Dict]:
    """Generate IPR curve data for plotting.

    For each well, generates arrays of BHP vs fluid rate for the
    most recent, lowest BHP, and median Vogel curves.

    Args:
        vogel_coeffs: DataFrame from compute_vogel_coefficients()

    Returns:
        Dictionary mapping well name to dict with:
        - bhp_array: array of BHP values
        - fluid_recent: fluid rates for most recent test curve
        - fluid_lowest: fluid rates for lowest BHP test curve
        - fluid_median: fluid rates for median test curve
        - res_pres: reservoir pressure
        - qmax_recent: max flow rate (recent)
    """
    ipr_data = {}

    for _, row in vogel_coeffs.iterrows():
        well = row["Well"]
        res_p = row["ResP"]

        if pd.isna(res_p) or res_p <= 0:
            continue

        bhp_array = np.arange(0, int(res_p), 10)

        try:
            # Recent curve
            vogel_recent = InFlow(row["qwf"], row["pwf"], res_p)
            fluid_recent = [vogel_recent.oil_flow(bhp, "vogel") for bhp in bhp_array]

            ipr_data[well] = {
                "bhp_array": bhp_array,
                "fluid_recent": fluid_recent,
                "res_pres": res_p,
                "qmax_recent": row["QMax_recent"],
            }
        except (ValueError, ZeroDivisionError, KeyError, IndexError) as e:
            logger.warning("Skipping IPR curve for %s: %s", well, e)
            continue

    return ipr_data


def export_optimization_template(
    vogel_coeffs: pd.DataFrame,
    jp_chars_path: Optional[str] = None,
    jp_chars: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Generate a CSV matching the well_optimization_template.csv format.

    Merges Vogel-derived qwf/pwf/res_pres with jp_chars data for
    tubing, temperature, TVD, and field model information.

    Args:
        vogel_coeffs: DataFrame from compute_vogel_coefficients()
        jp_chars_path: Path to jp_chars.csv. If None, auto-detects.
        jp_chars: Live well-characteristics DataFrame (e.g. from
            load_well_characteristics()). Takes precedence over the CSV —
            the legacy jp_chars.csv is stale and missing newer wells, and
            its values would clobber the live Databricks data when the
            template is loaded back via load_wells_from_dataframe.

    Returns:
        DataFrame matching the well_optimization_template.csv format
    """
    # Live chars take precedence; the CSV path is the legacy fallback.
    if jp_chars is not None and not jp_chars.empty and "Well" in jp_chars.columns:
        jp_lookup = jp_chars.set_index("Well").to_dict("index")
    else:
        if jp_chars_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            jp_chars_path = os.path.join(current_dir, "..", "jp_data", "jp_chars.csv")

        try:
            jp_chars_csv = pd.read_csv(jp_chars_path)
            jp_lookup = jp_chars_csv.set_index("Well").to_dict("index")
        except FileNotFoundError:
            jp_lookup = {}

    rows = []
    for _, coeff_row in vogel_coeffs.iterrows():
        well = coeff_row["Well"]
        jp_data = jp_lookup.get(well, {})

        # Determine field model
        is_sch = jp_data.get("is_sch", True)
        if isinstance(is_sch, str):
            is_sch = is_sch.lower() in ("true", "1", "yes")
        field_model = "Schrader" if is_sch else "Kuparuk"

        row = {
            "Well": well,
            "res_pres": coeff_row["ResP"],
            "form_temp": jp_data.get("form_temp", 75 if is_sch else 170),
            # pd.NA (not "") for missing depths: load_wells_from_dataframe
            # only skips template cells that fail pd.notna — an empty string
            # passes that check and CLOBBERED the well's live Databricks
            # JP_TVD, silently dropping it to the 4000-ft default.
            "JP_TVD": jp_data.get("JP_TVD", pd.NA),
            "JP_MD": jp_data.get("JP_MD", pd.NA),
            "out_dia": jp_data.get("out_dia", 4.5),
            "thick": jp_data.get("thick", 0.271),
            "casing_od": 6.875,
            "casing_thick": 0.5,
            "form_wc": coeff_row.get("form_wc", 0.5),
            "form_gor": coeff_row.get("fgor", 250),
            "field_model": field_model,
            "surf_pres": 210,
            "qwf_blpd": coeff_row["qwf"],
            "pwf": coeff_row["pwf"],
            "comments": f"IPR from well tests ({coeff_row['num_tests']} tests)",
        }
        rows.append(row)

    return pd.DataFrame(rows)
