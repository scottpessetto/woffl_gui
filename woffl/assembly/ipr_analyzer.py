"""IPR Analyzer

Computes Vogel IPR parameters from merged well test + BHP data.
Estimates optimal reservoir pressure per well and generates
Vogel coefficients suitable for the multi-well optimization template.

Adapted from header_pressure_impact/process_data/calc_PI_RP.py and bhp_liq.py.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from woffl.flow.inflow import InFlow


def _calculate_global_sse(bhp_values: np.ndarray, fluid_values: np.ndarray, pres: float) -> float:
    """Calculate sum of squared errors for a candidate reservoir pressure.

    For each test point used as the anchor, computes the Vogel curve and
    measures how well it predicts ALL other test points. Returns the
    minimum SSE across all possible anchor points.

    This approach finds the RP where the Vogel curve best passes through
    the cloud of test data, regardless of which point anchors the curve.

    Args:
        bhp_values: Array of BHP values for the well
        fluid_values: Array of total fluid rate values for the well
        pres: Candidate reservoir pressure (must be > max BHP)

    Returns:
        Minimum sum of squared errors across all anchor choices
    """
    n = len(bhp_values)
    if n < 2:
        return float("inf")

    best_sse = float("inf")

    for anchor_idx in range(n):
        anchor_bhp = bhp_values[anchor_idx]
        anchor_fluid = fluid_values[anchor_idx]

        # Skip if anchor BHP >= candidate RP (invalid for Vogel)
        if anchor_bhp >= pres or anchor_fluid <= 0:
            continue

        try:
            # Compute qmax from this anchor point
            ratio = anchor_bhp / pres
            denom = 1.0 - 0.2 * ratio - 0.8 * ratio**2
            if denom <= 0:
                continue
            qmax = anchor_fluid / denom

            # Predict fluid rate for all points and compute SSE
            sse = 0.0
            for j in range(n):
                if bhp_values[j] >= pres:
                    sse += 1e8  # Penalty for BHP >= RP
                    continue
                ratio_j = bhp_values[j] / pres
                predicted = qmax * (1.0 - 0.2 * ratio_j - 0.8 * ratio_j**2)
                sse += (predicted - fluid_values[j]) ** 2

            if sse < best_sse:
                best_sse = sse

        except (ValueError, ZeroDivisionError):
            continue

    return best_sse


def _calculate_r_squared(
    bhp_values: np.ndarray, fluid_values: np.ndarray, pres: float, anchor_bhp: float, anchor_fluid: float
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

        ss_res = 0.0
        for j in range(len(bhp_values)):
            if bhp_values[j] >= pres:
                continue
            ratio_j = bhp_values[j] / pres
            predicted = qmax * (1.0 - 0.2 * ratio_j - 0.8 * ratio_j**2)
            ss_res += (predicted - fluid_values[j]) ** 2

        mean_fluid = np.mean(fluid_values)
        ss_tot = np.sum((fluid_values - mean_fluid) ** 2)

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
            best_pres = int(max_bhp) + 50
        else:
            # Search with fine resolution near max_bhp, coarser further out
            for pres in range(start_pres, end_pres, 5):
                sse = _calculate_global_sse(bhp_values, fluid_values, float(pres))
                if sse < min_sse:
                    min_sse = sse
                    best_pres = pres

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
            qmax_recent = vogel_recent.vogel_qmax(recent_fluid, recent_bhp, optimal_res_p)

            # Lowest BHP test
            well_data_by_bhp = well_data.sort_values(by="BHP")
            lowest_fluid = well_data_by_bhp["WtTotalFluid"].iloc[0]
            lowest_bhp = well_data_by_bhp["BHP"].iloc[0]
            vogel_lowest = InFlow(lowest_fluid, lowest_bhp, optimal_res_p)
            qmax_lowest = vogel_lowest.vogel_qmax(lowest_fluid, lowest_bhp, optimal_res_p)

            # Median BHP test
            median_bhp_val = well_data["BHP"].median()
            closest_row = well_data.iloc[(well_data["BHP"] - median_bhp_val).abs().argsort()[:1]]
            median_fluid = closest_row["WtTotalFluid"].values[0]
            median_bhp = closest_row["BHP"].values[0]
            vogel_median = InFlow(median_fluid, median_bhp, optimal_res_p)
            qmax_median = vogel_median.vogel_qmax(median_fluid, median_bhp, optimal_res_p)

            # Calculate watercut from most recent test
            wc = 0.5  # default
            if "WtWaterVol" in well_data.columns and "WtTotalFluid" in well_data.columns:
                total = well_data["WtTotalFluid"].iloc[0]
                water = well_data["WtWaterVol"].iloc[0] if "WtWaterVol" in well_data.columns else 0
                if total > 0:
                    wc = water / total

            # Calculate R² fit quality using the median anchor
            bhp_vals = well_data["BHP"].values.astype(float)
            fluid_vals = well_data["WtTotalFluid"].values.astype(float)
            r_squared = _calculate_r_squared(bhp_vals, fluid_vals, optimal_res_p, median_bhp, median_fluid)

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
                    "num_tests": len(well_data),
                    "most_recent_date": well_data["date"].iloc[0],
                    "R2": round(r_squared, 3),
                }
            )
        except Exception as e:
            print(f"Error computing Vogel coefficients for {well}: {e}")
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
        except Exception as e:
            print(f"Error generating IPR curve for {well}: {e}")
            continue

    return ipr_data


def export_optimization_template(
    vogel_coeffs: pd.DataFrame,
    jp_chars_path: Optional[str] = None,
) -> pd.DataFrame:
    """Generate a CSV matching the well_optimization_template.csv format.

    Merges Vogel-derived qwf/pwf/res_pres with jp_chars data for
    tubing, temperature, TVD, and field model information.

    Args:
        vogel_coeffs: DataFrame from compute_vogel_coefficients()
        jp_chars_path: Path to jp_chars.csv. If None, auto-detects.

    Returns:
        DataFrame matching the well_optimization_template.csv format
    """
    # Load jp_chars for well characteristics
    if jp_chars_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        jp_chars_path = os.path.join(current_dir, "..", "jp_data", "jp_chars.csv")

    try:
        jp_chars = pd.read_csv(jp_chars_path)
        jp_lookup = jp_chars.set_index("Well").to_dict("index")
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
            "JP_TVD": jp_data.get("JP_TVD", ""),
            "JP_MD": jp_data.get("JP_MD", ""),
            "out_dia": jp_data.get("out_dia", 4.5),
            "thick": jp_data.get("thick", 0.271),
            "casing_od": 6.875,
            "casing_thick": 0.5,
            "form_wc": coeff_row.get("form_wc", 0.5),
            "form_gor": 250,
            "field_model": field_model,
            "surf_pres": 210,
            "qwf_bopd": coeff_row["qwf"],
            "pwf": coeff_row["pwf"],
            "comments": f"IPR from well tests ({coeff_row['num_tests']} tests)",
        }
        rows.append(row)

    return pd.DataFrame(rows)
