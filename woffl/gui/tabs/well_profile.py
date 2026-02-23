"""Tab 4: Well Profile Visualization

Renders the well trajectory visualization including TVD vs MD plots,
deviation profiles, and inclination data when survey data is available.
"""

import matplotlib.pyplot as plt
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
        well_label = selected_well if selected_well != "Custom" else f"Generic {field_model}"
        st.write(f"- Well Name: {well_label}")
        st.write(f"- Number of Survey Points: {len(well_profile.md_ray)}")
        st.write(f"- MD Range: {well_profile.md_ray[0]:.1f} to {well_profile.md_ray[-1]:.1f} ft")
        st.write(f"- TVD Range: {well_profile.vd_ray[0]:.1f} to {well_profile.vd_ray[-1]:.1f} ft")
        st.write(f"- Jetpump MD: {well_profile.jetpump_md:.1f} ft")
        st.write(f"- Jetpump TVD: {jpump_tvd:.1f} ft")

        max_deviation = max(
            abs(well_profile.md_ray[i] - well_profile.vd_ray[i]) for i in range(len(well_profile.md_ray))
        )
        st.write(f"- Max Deviation: {max_deviation:.1f} ft")

    with col2:
        survey_data = get_well_survey_data(selected_well) if selected_well != "Custom" else None

        if survey_data is not None and not survey_data.empty:
            st.success("✅ Using Actual Survey Data")
            st.write(f"- Survey Points: {len(survey_data)}")
            if "inclination" in survey_data.columns:
                max_inc = survey_data["inclination"].max()
                st.write(f"- Max Inclination: {max_inc:.2f}°")
            if "azimuth" in survey_data.columns:
                st.write(
                    f"- Azimuth Range: {survey_data['azimuth'].min():.1f}°" f" to {survey_data['azimuth'].max():.1f}°"
                )
        else:
            st.info(f"ℹ️ Using Default {field_model} Model")
            st.write("- Generic well profile")
            st.write("- Simplified trajectory")

    # Well trajectory plots
    _render_trajectory_plots(well_profile, jpump_tvd)

    # Inclination plot if survey data is available
    if survey_data is not None and not survey_data.empty and "inclination" in survey_data.columns:
        _render_inclination_plot(survey_data, well_profile)

    # Explanation
    st.markdown(
        """
    **Well Profile Explanation:**
    - **TVD vs MD Plot**: Shows the well's path from surface to total depth
    - **Deviation Plot**: Shows horizontal offset from vertical at each depth
    - **Inclination Plot**: Shows well angle from vertical (0° = vertical, 90° = horizontal)
    - **Red star**: Jetpump location
    - **Dashed lines**: Jetpump MD and TVD reference lines
    
    A vertical well would show MD = TVD (45° line on first plot).
    Deviation indicates how far the well has moved horizontally from the surface location.
    """
    )


def _render_trajectory_plots(well_profile, jpump_tvd: int) -> None:
    """Render the TVD vs MD and deviation plots."""
    st.subheader("Well Trajectory")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: TVD vs MD
    ax1.plot(well_profile.md_ray, well_profile.vd_ray, "b-", linewidth=2, label="Well Path")
    ax1.axhline(
        y=jpump_tvd,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Jetpump TVD ({jpump_tvd:.0f} ft)",
    )
    ax1.axvline(
        x=well_profile.jetpump_md,
        color="g",
        linestyle="--",
        linewidth=2,
        label=f"Jetpump MD ({well_profile.jetpump_md:.0f} ft)",
    )
    ax1.scatter(
        [well_profile.jetpump_md],
        [jpump_tvd],
        color="red",
        s=100,
        zorder=5,
        marker="*",
        label="Jetpump Location",
    )
    ax1.set_xlabel("Measured Depth (ft)", fontsize=12)
    ax1.set_ylabel("True Vertical Depth (ft)", fontsize=12)
    ax1.set_title("Well Profile: TVD vs MD", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()

    # Plot 2: Deviation (MD - TVD) vs Depth
    deviation = [well_profile.md_ray[i] - well_profile.vd_ray[i] for i in range(len(well_profile.md_ray))]
    ax2.plot(deviation, well_profile.vd_ray, "g-", linewidth=2)
    ax2.axhline(
        y=jpump_tvd,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Jetpump TVD ({jpump_tvd:.0f} ft)",
    )
    ax2.set_xlabel("Horizontal Deviation (ft)", fontsize=12)
    ax2.set_ylabel("True Vertical Depth (ft)", fontsize=12)
    ax2.set_title("Well Deviation Profile", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_inclination_plot(survey_data, well_profile) -> None:
    """Render the inclination vs measured depth plot."""
    st.subheader("Well Inclination Profile")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(survey_data["meas_depth"], survey_data["inclination"], "b-", linewidth=2)
    ax.axvline(
        x=well_profile.jetpump_md,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Jetpump MD ({well_profile.jetpump_md:.0f} ft)",
    )
    ax.set_xlabel("Measured Depth (ft)", fontsize=12)
    ax.set_ylabel("Inclination (degrees)", fontsize=12)
    ax.set_title("Well Inclination vs Measured Depth", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
