"""Utility functions for the WOFFL GUI

This module contains helper functions for the Streamlit GUI.
"""

import numpy as np
import streamlit as st
from woffl.assembly.batchrun import BatchPump
from woffl.assembly.curvefit import exp_model, rev_exp_deriv
from woffl.assembly.sysops import jetpump_solver
from woffl.flow import jetgraphs as jg
from woffl.flow import jetplot as jplt
from woffl.flow.inflow import InFlow
from woffl.geometry.jetpump import JetPump
from woffl.geometry.pipe import Annulus, Pipe
from woffl.geometry.wellprofile import WellProfile
from woffl.pvt.blackoil import BlackOil
from woffl.pvt.formgas import FormGas
from woffl.pvt.formwat import FormWater
from woffl.pvt.resmix import ResMix


def create_jetpump(nozzle_no, area_ratio, ken, kth, kdi):
    """Create a JetPump object with the given parameters."""
    return JetPump(nozzle_no=nozzle_no, area_ratio=area_ratio, ken=ken, kth=kth, kdi=kdi)


def create_reservoir_mix(wc, gor, temp, field_model=None):
    """Create a ResMix object with the given parameters."""
    # Default to schrader if field_model is None or an unknown model
    if field_model is None:
        field_model = "schrader"

    field_model = field_model.lower()  # Convert to lowercase for case-insensitive comparison

    if field_model == "schrader":
        oil = BlackOil.schrader()
        water = FormWater.schrader()
        gas = FormGas.schrader()
    elif field_model == "kuparuk":
        oil = BlackOil.kuparuk()
        water = FormWater.kuparuk()
        gas = FormGas.kuparuk()
    else:
        # Default to schrader if an unknown model is specified
        oil = BlackOil.schrader()
        water = FormWater.schrader()
        gas = FormGas.schrader()

    return ResMix(wc=wc, fgor=gor, oil=oil, wat=water, gas=gas)


def create_well_profile(field_model=None, jpump_tvd=None):
    """Create a WellProfile object with the given field model and jetpump TVD.

    Args:
        field_model (str, optional): The field model to use ("schrader" or "kuparuk").
            If None, defaults to "schrader".
        jpump_tvd (float, optional): The jetpump true vertical depth in feet.
            If provided, the well profile will be adjusted to have this TVD.
            If not provided, the default jetpump MD from the field model will be used.

    Returns:
        WellProfile: A WellProfile object
    """
    # Default to schrader if field_model is None
    if field_model is None:
        field_model = "schrader"

    field_model = field_model.lower()  # Convert to lowercase for case-insensitive comparison

    # First create a well profile with the default jetpump MD
    if field_model == "schrader":
        well_profile = WellProfile.schrader()
    elif field_model == "kuparuk":
        well_profile = WellProfile.kuparuk()
    else:
        # Default to schrader if an unknown model is specified
        well_profile = WellProfile.schrader()

    # If jpump_tvd is provided, create a new well profile with the correct jetpump MD
    if jpump_tvd is not None:
        try:
            # Convert TVD to MD using the well profile's interpolation
            jpump_md = well_profile.md_interp(jpump_tvd)

            # Create a new well profile with the same MD/VD arrays but with the new jetpump MD
            well_profile = WellProfile(md_list=well_profile.md_ray, vd_list=well_profile.vd_ray, jetpump_md=jpump_md)
        except ValueError as e:
            # If the TVD is outside the well profile's range, log a warning and use the default
            print(f"Warning: {e}. Using default jetpump MD.")

    return well_profile


def create_pipes(tubing_od=4.5, tubing_thickness=0.5, casing_od=6.875, casing_thickness=0.5):
    """Create tubing, casing, and annulus objects."""
    tube = Pipe(out_dia=tubing_od, thick=tubing_thickness)
    case = Pipe(out_dia=casing_od, thick=casing_thickness)
    ann = Annulus(inn_pipe=tube, out_pipe=case)
    return tube, case, ann


def create_inflow(qwf, pwf, pres):
    """Create an InFlow object with the given parameters."""
    return InFlow(qwf=qwf, pwf=pwf, pres=pres)


def generate_choked_figures(form_temp, rho_pf, ppf_surf, jetpump, tube, well_profile, inflow, res_mix):
    """Generate choked figures using the jetgraphs module."""
    return jg.choked_figures(form_temp, rho_pf, ppf_surf, jetpump, tube, well_profile, inflow, res_mix)


def generate_discharge_check(surf_pres, form_temp, rho_pf, ppf_surf, jetpump, tube, well_profile, inflow, res_mix):
    """Generate discharge check using the jetgraphs module."""
    return jg.discharge_check(surf_pres, form_temp, rho_pf, ppf_surf, jetpump, tube, well_profile, inflow, res_mix)


def generate_multi_throat_entry_books(psu_ray, form_temp, ken, ate, inflow, res_mix):
    """Generate multi throat entry books using the jetplot module."""
    return jplt.multi_throat_entry_books(psu_ray, form_temp, ken, ate, inflow, res_mix)


def generate_multi_suction_graphs(qoil_list, book_list):
    """Generate multi suction graphs using the jetplot module."""
    return jplt.multi_suction_graphs(qoil_list, book_list)


def run_jetpump_solver(surf_pres, form_temp, rho_pf, ppf_surf, jetpump, tube, well_profile, inflow, res_mix):
    """Run the jetpump solver and return the results.

    This function uses the jetpump_solver from sysops to find a solution for the jetpump system
    that factors in the wellhead pressure and reservoir conditions.

    Returns:
        tuple or None: (psu, sonic_status, qoil_std, fwat_bwpd, qnz_bwpd, mach_te) if successful,
                       None if the solver fails
    """
    try:
        return jetpump_solver(
            pwh=surf_pres,
            tsu=form_temp,
            rho_pf=rho_pf,
            ppf_surf=ppf_surf,
            jpump=jetpump,
            wellbore=tube,  # Note: jetpump_solver expects wellbore, not tube
            wellprof=well_profile,
            ipr_su=inflow,
            prop_su=res_mix,
        )
    except ValueError as e:
        # Handle the case where the well cannot lift at max suction pressure
        st.error(f"Solver error: {str(e)}")
        return None


def recommend_jetpump(batch_pump, marginal_watercut, water_type="lift"):
    """Recommend Jet Pump Based on Marginal Watercut

    Analyzes batch run results to recommend a jet pump configuration where the
    marginal watercut is closest to but still below the specified threshold.
    This represents the economic limit where additional water handling is justified.

    Args:
        batch_pump (BatchPump): BatchPump object with processed results
        marginal_watercut (float): Threshold for marginal watercut (bbl water / (bbl water + bbl oil))
        water_type (str): "lift" or "total" depending on the desired analysis

    Returns:
        dict: Recommended jet pump configuration and performance metrics
            {
                'nozzle': str,
                'throat': str,
                'qoil_std': float,
                'water_rate': float,
                'marginal_ratio': float,
                'recommendation_type': str
            }
    """
    # Validate inputs
    if not hasattr(batch_pump, "df") or batch_pump.df.empty:
        raise ValueError("Batch pump has no results to analyze")

    if not hasattr(batch_pump, "coeff_totl") or not hasattr(batch_pump, "coeff_lift"):
        raise ValueError("Batch pump curve fitting has not been performed")

    # Validate water type
    water_type = _validate_water_type(water_type)

    # Get semi-finalist pumps
    semi_df = batch_pump.df[batch_pump.df["semi"]].copy()
    if semi_df.empty:
        raise ValueError("No semi-finalist jet pumps found")

    # Sort by oil rate (which correlates with water rate for semi-finalists)
    semi_df = semi_df.sort_values(by="qoil_std", ascending=True)

    # Get the appropriate coefficients and water rates
    coeff = batch_pump.coeff_lift if water_type == "lift" else batch_pump.coeff_totl
    water_col = "lift_wat" if water_type == "lift" else "totl_wat"
    marg_col = "molwr" if water_type == "lift" else "motwr"

    # Get water rates and calculate marginal watercuts for semi-finalists
    water_rates = semi_df[water_col].values

    # Convert marginal oil-water ratios to marginal watercuts
    # Original ratios are (bbl oil / bbl water)
    # We need (bbl water / (bbl water + bbl oil))
    original_ratios = semi_df[marg_col].values
    marginal_watercuts = 1 / (1 + original_ratios)

    # Check if any pumps meet the threshold (below the watercut threshold)
    below_threshold = marginal_watercuts <= marginal_watercut

    # If no pumps meet the threshold, recommend the one with lowest marginal watercut
    if not any(below_threshold):
        best_idx = np.argmin(marginal_watercuts)
        recommendation = {
            "nozzle": semi_df.iloc[best_idx]["nozzle"],
            "throat": semi_df.iloc[best_idx]["throat"],
            "qoil_std": semi_df.iloc[best_idx]["qoil_std"],
            "water_rate": semi_df.iloc[best_idx][water_col],
            "marginal_ratio": marginal_watercuts[best_idx],
            "recommendation_type": "best_available",
        }
        return recommendation

    # Find theoretical optimal water rate using curve fit
    # This is where the marginal watercut equals the threshold
    # We need to convert the watercut threshold to an oil-water ratio first
    # watercut = water / (water + oil)
    # oil / water = (1 - watercut) / watercut
    oil_water_ratio = (1 - marginal_watercut) / marginal_watercut

    a, b, c = coeff
    optimal_water_rate = rev_exp_deriv(oil_water_ratio, b, c)
    optimal_oil_rate = exp_model(optimal_water_rate, a, b, c)

    # Find the closest semi-finalist pump just below the threshold
    # First, filter to only those below threshold
    valid_indices = np.where(below_threshold)[0]

    # If we have the theoretical point, find the closest actual pump
    if valid_indices.size > 0:
        # Calculate distances to the theoretical optimal point
        distances = []
        for idx in valid_indices:
            pump_water = water_rates[idx]
            pump_oil = semi_df.iloc[idx]["qoil_std"]
            # Calculate Euclidean distance in the oil-water space
            distance = np.sqrt((pump_water - optimal_water_rate) ** 2 + (pump_oil - optimal_oil_rate) ** 2)
            distances.append(distance)

        # Find the closest pump
        closest_idx = valid_indices[np.argmin(distances)]

        recommendation = {
            "nozzle": semi_df.iloc[closest_idx]["nozzle"],
            "throat": semi_df.iloc[closest_idx]["throat"],
            "qoil_std": semi_df.iloc[closest_idx]["qoil_std"],
            "water_rate": semi_df.iloc[closest_idx][water_col],
            "marginal_ratio": marginal_watercuts[closest_idx],
            "recommendation_type": "optimal",
            "theoretical_water_rate": optimal_water_rate,
            "theoretical_oil_rate": optimal_oil_rate,
        }
        return recommendation

    # This should never happen if below_threshold check is done correctly
    raise ValueError("Could not determine recommended jet pump")


def _validate_water_type(water_type):
    """Validate Type of Water String

    Checks that the string passed into a method or argument fits the required description.
    This is used when the water type wants to be defined as lift or total.

    Args:
        water_type (str): "lift" or "total" depending on the desired analysis

    Returns:
        str: Properly formatted as either "lift" or "total"
    """
    # Validate the 'water' argument
    if water_type not in {"lift", "total", "totl"}:
        raise ValueError(f"Invalid value for 'water_type': {water_type}. Expected 'lift', 'total', or 'totl'.")

    # Standardize "totl" to "total"
    if water_type == "totl":
        water_type = "total"
    return water_type


def highlight_recommended_pump(ax, recommendation, water_type="lift"):
    """Highlight the recommended pump on a performance plot

    Args:
        ax (matplotlib.axes.Axes): The axes object to plot on
        recommendation (dict): The recommendation dictionary from recommend_jetpump
        water_type (str): "lift" or "total" depending on the plot type
    """
    if recommendation is None:
        return

    water_type = _validate_water_type(water_type)

    # Extract values from recommendation
    water_rate = recommendation["water_rate"]
    oil_rate = recommendation["qoil_std"]
    pump_label = recommendation["nozzle"] + recommendation["throat"]

    # Highlight the recommended pump with a star marker
    ax.plot(
        water_rate,
        oil_rate,
        marker="*",
        markersize=15,
        markerfacecolor="gold",
        markeredgecolor="black",
        markeredgewidth=1.5,
        linestyle="none",
        label=f"Recommended: {pump_label}",
    )

    # Add annotation
    ax.annotate(
        f"Recommended: {pump_label}",
        xy=(water_rate, oil_rate),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", alpha=0.8),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
    )


def run_batch_pump(
    surf_pres,
    form_temp,
    rho_pf,
    ppf_surf,
    tube,
    well_profile,
    inflow,
    res_mix,
    nozzle_options,
    throat_options,
    wellname="Test Well",
):
    """Run a batch pump simulation with multiple nozzle and throat combinations.

    Args:
        surf_pres: Surface pressure (psi)
        form_temp: Formation temperature (°F)
        rho_pf: Power fluid density (lbm/ft³)
        ppf_surf: Power fluid surface pressure (psi)
        tube: Tubing pipe object
        well_profile: Well profile object
        inflow: Inflow performance object
        res_mix: Reservoir mixture object
        nozzle_options: List of nozzle sizes to test
        throat_options: List of throat ratios to test
        wellname: Name of the well for display purposes

    Returns:
        BatchPump: A BatchPump object with results, or None if processing fails
    """
    # Create a list of jet pumps with all combinations of nozzles and throats
    jp_list = BatchPump.jetpump_list(nozzle_options, throat_options)

    # Create a BatchPump object
    batch_pump = BatchPump(
        pwh=surf_pres,
        tsu=form_temp,
        rho_pf=rho_pf,
        ppf_surf=ppf_surf,
        wellbore=tube,
        wellprof=well_profile,
        ipr_su=inflow,
        prop_su=res_mix,
        wellname=wellname,
    )

    # Run the batch simulation
    batch_pump.batch_run(jp_list)

    # Process the results
    try:
        batch_pump.process_results()
        return batch_pump
    except (ValueError, RuntimeError) as e:
        error_msg = str(e)
        if "Optimal parameters not found" in error_msg:
            st.error("Could not fit curve to the data. Try selecting more nozzle sizes and throat ratios.")
            st.info("The batch run results are still available in the data table, but the curve fitting failed.")
            # Return the batch_pump object without curve fitting
            return batch_pump
        else:
            st.error(f"Error processing batch results: {error_msg}")
            return None


def run_power_fluid_range_batch(
    surf_pres,
    form_temp,
    rho_pf,
    power_fluid_min,
    power_fluid_max,
    power_fluid_step,
    tube,
    well_profile,
    inflow,
    res_mix,
    nozzle_options,
    throat_options,
    wellname="Test Well",
):
    """Run a comprehensive batch pump simulation across a range of power fluid pressures.

    Args:
        surf_pres: Surface pressure (psi)
        form_temp: Formation temperature (°F)
        rho_pf: Power fluid density (lbm/ft³)
        power_fluid_min: Minimum power fluid pressure (psi)
        power_fluid_max: Maximum power fluid pressure (psi)
        power_fluid_step: Step size for power fluid pressure (psi)
        tube: Tubing pipe object
        well_profile: Well profile object
        inflow: Inflow performance object
        res_mix: Reservoir mixture object
        nozzle_options: List of nozzle sizes to test
        throat_options: List of throat ratios to test
        wellname: Name of the well for display purposes

    Returns:
        pandas.DataFrame: Comprehensive results across all power fluid pressures
    """
    import pandas as pd

    # Create pressure range
    pressure_range = np.arange(power_fluid_min, power_fluid_max + power_fluid_step, power_fluid_step)

    # Create a list of jet pumps with all combinations of nozzles and throats
    jp_list = BatchPump.jetpump_list(nozzle_options, throat_options)

    all_results = []
    total_combinations = len(pressure_range) * len(jp_list)

    # Create a progress bar
    progress_bar = st.progress(0)
    current_combination = 0

    for pressure in pressure_range:
        # Create a BatchPump object for this pressure
        batch_pump = BatchPump(
            pwh=surf_pres,
            tsu=form_temp,
            rho_pf=rho_pf,
            ppf_surf=pressure,
            wellbore=tube,
            wellprof=well_profile,
            ipr_su=inflow,
            prop_su=res_mix,
            wellname=wellname,
        )

        # Run the batch simulation for this pressure
        batch_pump.batch_run(jp_list, debug=False)

        # Add power fluid pressure to the results
        batch_pump.df["power_fluid_pressure"] = pressure

        # Add the results to our comprehensive dataset
        all_results.append(batch_pump.df)

        # Update progress bar
        current_combination += len(jp_list)
        progress_bar.progress(current_combination / total_combinations)

    # Combine all results into a single dataframe
    comprehensive_df = pd.concat(all_results, ignore_index=True)

    # Clear progress bar
    progress_bar.empty()

    return comprehensive_df
