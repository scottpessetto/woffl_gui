"""Utility functions for the WOFFL GUI

This module contains helper functions for the Streamlit GUI.
"""

import numpy as np
import streamlit as st

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


def create_reservoir_mix(wc, gor, temp, field_model="schrader"):
    """Create a ResMix object with the given parameters."""
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


def create_well_profile(field_model="schrader", jpump_tvd=None):
    """Create a WellProfile object with the given field model and jetpump TVD.

    Args:
        field_model (str): The field model to use ("schrader" or "kuparuk")
        jpump_tvd (float, optional): The jetpump true vertical depth in feet.
            If provided, the well profile will be adjusted to have this TVD.
            If not provided, the default jetpump MD from the field model will be used.

    Returns:
        WellProfile: A WellProfile object
    """
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
