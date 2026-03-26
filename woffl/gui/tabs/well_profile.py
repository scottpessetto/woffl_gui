"""Tab 4: Well Profile Visualization

Renders the well trajectory visualization including TVD vs MD plots,
deviation profiles, and inclination data when survey data is available.
"""

import plotly.graph_objects as go
import streamlit as st

from woffl.gui.params import SimulationParams
from woffl.gui.utils import get_well_survey_data


def render_tab(params: SimulationParams, well_profile) -> None:
    """Render the Well Profile Visualization tab.

    Args:
        params: Simulation parameters from sidebar
        well_profile: WellProfile object
    """
    st.subheader("Well Profile Visualization")

    selected_well = params.selected_well
    jpump_tvd = params.jpump_tvd
    field_model = params.field_model

    # Display well profile information
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Well Profile Information:**")
        well_label = (
            selected_well if selected_well != "Custom" else f"Generic {field_model}"
        )
        st.write(f"- Well Name: {well_label}")
        st.write(f"- Number of Survey Points: {len(well_profile.md_ray)}")
        st.write(
            f"- MD Range: {well_profile.md_ray[0]:.1f} to {well_profile.md_ray[-1]:.1f} ft"
        )
        st.write(
            f"- TVD Range: {well_profile.vd_ray[0]:.1f} to {well_profile.vd_ray[-1]:.1f} ft"
        )
        st.write(f"- Jetpump MD: {well_profile.jetpump_md:.1f} ft")
        st.write(f"- Jetpump TVD: {jpump_tvd:.1f} ft")

        max_deviation = max(
            abs(well_profile.md_ray[i] - well_profile.vd_ray[i])
            for i in range(len(well_profile.md_ray))
        )
        st.write(f"- Max Deviation: {max_deviation:.1f} ft")

    with col2:
        survey_data = (
            get_well_survey_data(selected_well) if selected_well != "Custom" else None
        )

        if survey_data is not None and not survey_data.empty:
            st.success("✅ Using Actual Survey Data")
            st.write(f"- Survey Points: {len(survey_data)}")
            if "inclination" in survey_data.columns:
                max_inc = survey_data["inclination"].max()
                st.write(f"- Max Inclination: {max_inc:.2f}°")
            if "azimuth" in survey_data.columns:
                st.write(
                    f"- Azimuth Range: {survey_data['azimuth'].min():.1f}°"
                    f" to {survey_data['azimuth'].max():.1f}°"
                )
        else:
            st.info(f"ℹ️ Using Default {field_model} Model")
            st.write("- Generic well profile")
            st.write("- Simplified trajectory")

    # Well trajectory plots
    _render_trajectory_plots(well_profile, jpump_tvd)

    # Inclination plot if survey data is available
    if (
        survey_data is not None
        and not survey_data.empty
        and "inclination" in survey_data.columns
    ):
        _render_inclination_plot(survey_data, well_profile)

    # Explanation
    st.markdown("""
    **Well Profile Explanation:**
    - **TVD vs MD Plot**: Shows the well's path from surface to total depth
    - **Deviation Plot**: Shows horizontal offset from vertical at each depth
    - **Inclination Plot**: Shows well angle from vertical (0° = vertical, 90° = horizontal)
    - **Red star**: Jetpump location
    - **Dashed lines**: Jetpump MD and TVD reference lines

    A vertical well would show MD = TVD (45° line on first plot).
    Deviation indicates how far the well has moved horizontally from the surface location.
    """)


def _render_trajectory_plots(well_profile, jpump_tvd: int) -> None:
    """Render the TVD vs MD and deviation plots."""
    st.subheader("Well Trajectory")

    col1, col2 = st.columns(2)

    with col1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=well_profile.md_ray.tolist(),
            y=well_profile.vd_ray.tolist(),
            mode="lines",
            name="Well Path",
            line=dict(color="blue", width=2),
            hovertemplate="MD: %{x:.0f} ft<br>TVD: %{y:.0f} ft<extra></extra>",
        ))
        fig1.add_hline(
            y=jpump_tvd,
            line=dict(color="red", dash="dash", width=2),
            annotation_text=f"Jetpump TVD ({jpump_tvd:.0f} ft)",
            annotation_position="top left",
        )
        fig1.add_vline(
            x=well_profile.jetpump_md,
            line=dict(color="green", dash="dash", width=2),
            annotation_text=f"Jetpump MD ({well_profile.jetpump_md:.0f} ft)",
            annotation_position="top right",
        )
        fig1.add_trace(go.Scatter(
            x=[well_profile.jetpump_md],
            y=[jpump_tvd],
            mode="markers",
            name="Jetpump Location",
            marker=dict(color="red", size=14, symbol="star"),
            hovertemplate="Jetpump<br>MD: %{x:.0f} ft<br>TVD: %{y:.0f} ft<extra></extra>",
        ))
        fig1.update_layout(
            title=dict(text="Well Profile: TVD vs MD", font=dict(size=14)),
            xaxis_title="Measured Depth (ft)",
            yaxis_title="True Vertical Depth (ft)",
            yaxis=dict(autorange="reversed"),
            height=500,
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        deviation = [
            well_profile.md_ray[i] - well_profile.vd_ray[i]
            for i in range(len(well_profile.md_ray))
        ]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=deviation,
            y=well_profile.vd_ray.tolist(),
            mode="lines",
            name="Deviation",
            line=dict(color="green", width=2),
            hovertemplate="Deviation: %{x:.0f} ft<br>TVD: %{y:.0f} ft<extra></extra>",
        ))
        fig2.add_hline(
            y=jpump_tvd,
            line=dict(color="red", dash="dash", width=2),
            annotation_text=f"Jetpump TVD ({jpump_tvd:.0f} ft)",
            annotation_position="top left",
        )
        fig2.update_layout(
            title=dict(text="Well Deviation Profile", font=dict(size=14)),
            xaxis_title="Horizontal Deviation (ft)",
            yaxis_title="True Vertical Depth (ft)",
            yaxis=dict(autorange="reversed"),
            height=500,
        )
        st.plotly_chart(fig2, use_container_width=True)


def _render_inclination_plot(survey_data, well_profile) -> None:
    """Render the inclination vs measured depth plot."""
    st.subheader("Well Inclination Profile")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=survey_data["inclination"].tolist(),
        y=survey_data["meas_depth"].tolist(),
        mode="lines",
        name="Inclination",
        line=dict(color="blue", width=2),
        hovertemplate="Inclination: %{x:.0f}°<br>MD: %{y:.0f} ft<extra></extra>",
    ))
    fig.add_hline(
        y=well_profile.jetpump_md,
        line=dict(color="red", dash="dash", width=2),
        annotation_text=f"Jetpump MD ({well_profile.jetpump_md:.0f} ft)",
        annotation_position="top left",
    )
    fig.update_layout(
        title=dict(text="Well Inclination vs Measured Depth", font=dict(size=14)),
        xaxis_title="Inclination (degrees)",
        yaxis_title="Measured Depth (ft)",
        yaxis=dict(autorange="reversed"),
        height=1000,
    )
    st.plotly_chart(fig, use_container_width=True)
