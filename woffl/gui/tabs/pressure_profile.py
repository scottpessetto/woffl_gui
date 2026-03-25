"""Tab: Pressure Profile

Computes and plots tubing and annulus pressure vs depth, and the
differential between them.  Useful for evaluating casing/tubing
loads and packer differential.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from woffl.flow import jetflow as jf
from woffl.flow import outflow as of
from woffl.flow import singlephase as sp
from woffl.gui.params import SimulationParams
from woffl.gui.utils import create_pvt_components, run_jetpump_solver
from woffl.pvt.resmix import ResMix


def _powerfluid_pressure_profile(ppf_surf, tsu, qnz_bwpd, prop_pf, wellbore, wellprof, flowpath):
    """Compute segmented power-fluid pressure profile from surface to JP.

    Power fluid is single-phase water flowing DOWN.  Gravity increases
    pressure; friction opposes flow and reduces the gain.

    Returns:
        vd_seg: TVD array (ft)
        prs_array: Pressure at each TVD point (psi)
    """
    if flowpath == "tubing":
        hyd_dia = wellbore.tube_hyd_dia
        area = wellbore.tube_area
        abs_ruff = wellbore.tube_abs_ruff
    else:
        hyd_dia = wellbore.ann_hyd_dia
        area = wellbore.ann_area
        abs_ruff = wellbore.ann_abs_ruff

    md_seg, vd_seg = wellprof.outflow_spacing(100)

    # Single-phase parameters (constant for incompressible fluid)
    prop_cond = prop_pf.condition(ppf_surf, tsu)
    rho = prop_cond.density
    visc = prop_cond.viscosity

    qwat_fts = sp.bpd_to_ft3s(qnz_bwpd)
    vel = sp.velocity(qwat_fts, area)
    n_re = sp.reynolds(rho, vel, hyd_dia, visc)
    rel_ruff = sp.relative_roughness(hyd_dia, abs_ruff)
    ff = sp.ffactor_darcy(n_re, rel_ruff)

    # Height convention matches production_top_down_press (negative = going down).
    # Length is NOT negated (positive = with flow direction, i.e. downward).
    vd_diff = np.diff(vd_seg) * -1
    md_diff = np.diff(md_seg)  # positive — flow goes down with calculation

    prs_list = [ppf_surf]
    for length, height in zip(md_diff, vd_diff):
        dp_stat = sp.diff_press_static(rho, height)
        dp_fric = sp.diff_press_friction(ff, rho, vel, hyd_dia, length)
        prs_list.append(prs_list[-1] - dp_stat - dp_fric)

    return vd_seg, np.array(prs_list)


def render_tab(params: SimulationParams, jetpump, wellbore, well_profile, inflow, res_mix) -> None:
    """Render the Pressure Profile tab."""
    st.subheader("Pressure Profile")

    # Flow path labels
    if params.jpump_direction == "reverse":
        prod_path = "tubing"
        pf_path = "annulus"
        prod_label = "Tubing (Production)"
        pf_label = "Annulus (Power Fluid)"
    else:
        prod_path = "annulus"
        pf_path = "tubing"
        prod_label = "Annulus (Production)"
        pf_label = "Tubing (Power Fluid)"

    # Solve for operating point
    solver_results = run_jetpump_solver(
        params.surf_pres,
        params.form_temp,
        params.rho_pf,
        params.ppf_surf,
        jetpump,
        wellbore,
        well_profile,
        inflow,
        res_mix,
        field_model=params.field_model,
        jpump_direction=params.jpump_direction,
    )

    if not solver_results:
        st.warning("Could not compute pressure profiles — solver did not converge.")
        return

    psu, sonic_status, qoil_std, fwat_bwpd, qnz_bwpd, mach_te = solver_results

    # Build mixed production fluid (formation + power fluid) — same as discharge_residual
    _, prop_pf, _ = create_pvt_components(params.field_model)
    prop_pf = prop_pf.condition(0, 60)

    wc_tm, _ = jf.throat_wc(qoil_std, res_mix.wc, qnz_bwpd)
    prop_tm = ResMix(wc_tm, res_mix.fgor, res_mix.oil, res_mix.wat, res_mix.gas)

    # Production pressure profile (top-down from wellhead)
    md_seg, prod_prs, _ = of.production_top_down_press(
        params.surf_pres, params.form_temp, qoil_std, prop_tm,
        wellbore, well_profile, prod_path,
    )
    _, vd_seg = well_profile.outflow_spacing(100)

    # Power-fluid pressure profile (top-down from PF surface pressure)
    _, pf_prs = _powerfluid_pressure_profile(
        params.ppf_surf, params.form_temp, qnz_bwpd, prop_pf,
        wellbore, well_profile, pf_path,
    )

    # Differential: power-fluid minus production
    differential = pf_prs - prod_prs

    # --- Plotly figure ---
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Pressure vs Depth", "Differential vs Depth"),
        horizontal_spacing=0.12,
    )

    jp_tvd = params.jpump_tvd
    hover_psi = "TVD: %{y:.0f} ft<br>Pressure: %{x:.0f} psi<extra></extra>"
    hover_diff = "TVD: %{y:.0f} ft<br>Differential: %{x:.0f} psi<extra></extra>"

    fig.add_trace(
        go.Scatter(x=prod_prs, y=vd_seg, name=prod_label,
                   line=dict(color="blue"), hovertemplate=hover_psi),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=pf_prs, y=vd_seg, name=pf_label,
                   line=dict(color="red"), hovertemplate=hover_psi),
        row=1, col=1,
    )

    # Suction pressure marker at JP depth
    fig.add_trace(
        go.Scatter(
            x=[psu], y=[jp_tvd], name=f"Suction ({psu:.0f} psi)",
            mode="markers", marker=dict(color="purple", size=10, symbol="diamond"),
            hovertemplate="Suction Pressure: %{x:.0f} psi<extra></extra>",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(x=differential, y=vd_seg, name="PF \u2212 Production",
                   line=dict(color="green"), hovertemplate=hover_diff),
        row=1, col=2,
    )
    # Zero-line on differential plot
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=2)

    # Jet pump depth marker on both subplots
    fig.add_hline(
        y=jp_tvd, line_dash="dot", line_color="orange", opacity=0.6,
        annotation_text="JP depth", annotation_position="top left",
        row=1, col=1,
    )
    fig.add_hline(
        y=jp_tvd, line_dash="dot", line_color="orange", opacity=0.6,
        row=1, col=2,
    )

    fig.update_yaxes(autorange="reversed", title_text="TVD (ft)", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_xaxes(title_text="Pressure (psi)", row=1, col=1)
    fig.update_xaxes(title_text="Differential (psi)", row=1, col=2)
    fig.update_layout(height=600, legend=dict(orientation="h", y=-0.15))

    st.plotly_chart(fig, use_container_width=True)

    # Summary metrics at JP depth
    st.markdown("**At Jet Pump Depth:**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Discharge Pressure", f"{prod_prs[-1]:.0f} psi")
    with col2:
        st.metric("Suction Pressure", f"{psu:.0f} psi")
    with col3:
        st.metric(pf_label, f"{pf_prs[-1]:.0f} psi")
    with col4:
        st.metric("Differential (PF \u2212 Prod)", f"{differential[-1]:.0f} psi")

    st.caption(
        "The tubing pressure at JP depth is the **discharge** pressure — what the pump "
        "must produce to lift mixed fluid to surface. The **suction** pressure is below "
        "the pump where formation fluid enters. The jet pump boosts pressure from suction "
        "to discharge. Power fluid profile is single-phase water; production uses Beggs & Brill."
    )
