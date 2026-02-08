"""Multi-Well Optimization Page

Standalone page for multi-well jet pump optimization.
"""

import os
import tempfile

import matplotlib.pyplot as plt
import streamlit as st
from woffl.assembly.network_optimizer import PowerFluidConstraint, load_wells_from_csv
from woffl.assembly.optimization_algorithms import optimize
from woffl.gui.optimization_utils import get_template_csv_content
from woffl.gui.optimization_viz import (
    create_efficiency_scatter,
    create_marginal_rate_chart,
    create_oil_rate_bar_chart,
    create_power_fluid_pie_chart,
    create_pump_config_chart,
    create_watercut_comparison,
)


def run_multi_well_optimization_page():
    """Render the multi-well optimization page"""

    st.title("🛢️ Multi-Well Jet Pump Optimization")

    st.markdown(
        """
    Optimize jet pump sizing across multiple wells to maximize field oil production 
    given a constrained power fluid supply.
    """
    )

    # Sidebar for parameters
    with st.sidebar:
        st.header("Optimization Parameters")

        # Power Fluid Settings
        st.subheader("Power Fluid Constraint")
        total_pf = st.number_input(
            "Total Available Power Fluid (bbl/day)",
            min_value=0,
            max_value=50000,
            value=10000,
            step=500,
            help="Total power fluid capacity available for all wells",
        )

        pf_pressure = st.number_input(
            "Power Fluid Pressure (psi)",
            min_value=1000,
            max_value=5000,
            value=3168,
            step=100,
            help="Surface power fluid pressure",
        )

        rho_pf = st.number_input("Power Fluid Density (lbm/ft³)", min_value=50.0, max_value=70.0, value=62.4, step=0.1)

        # Optimization Settings
        st.subheader("Optimization Settings")
        opt_method = st.selectbox(
            "Algorithm",
            ["greedy", "proportional"],
            index=0,
            help="Greedy: Iterative marginal allocation. Proportional: Productivity-based allocation.",
        )

        marginal_wc = st.number_input(
            "Marginal Watercut Threshold", min_value=0.0, max_value=1.0, value=0.94, step=0.01, format="%.2f"
        )

        # Pump Options
        st.subheader("Pump Options to Test")
        nozzle_opts = st.multiselect(
            "Nozzle Sizes",
            ["8", "9", "10", "11", "12", "13", "14", "15"],
            default=["10", "11", "12", "13"],
            help="Select nozzle sizes to test during optimization",
        )

        throat_opts = st.multiselect(
            "Throat Ratios",
            ["X", "A", "B", "C", "D", "E"],
            default=["B", "C", "D"],
            help="Select throat ratios to test during optimization",
        )

    # Main content
    st.write("## Well Configuration")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload Well Configuration CSV",
            type=["csv"],
            help="Upload a CSV file with well configurations. Wells in jp_chars.csv will auto-populate.",
        )

    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        template_csv = get_template_csv_content()
        st.download_button(
            label="📥 Download CSV Template",
            data=template_csv,
            file_name="well_optimization_template.csv",
            mime="text/csv",
            help="Download a template CSV with example wells",
            use_container_width=True,
        )

    if not uploaded_file:
        st.info("👆 Upload a CSV file to begin. Click 'Download CSV Template' to get started.")

        # Show example and instructions
        with st.expander("📖 How to Use", expanded=True):
            st.markdown(
                """
            ### Quick Start Guide
            
            1. **Download the CSV template** using the button above
            2. **Edit the CSV** to include your wells:
               - For wells in the database (jp_chars.csv): Just list the well name
               - For custom wells: Provide all required parameters
            3. **Upload the CSV** and configure power fluid constraints
            4. **Run optimization** to get pump recommendations
            
            ### Example CSV
            ```csv
            Well,res_pres,form_temp,JP_TVD
            MPB-28,,,
            MPE-35,,,
            CustomWell-1,1500,75,4000
            ```
            
            - MPB-28 and MPE-35 auto-load from database
            - CustomWell-1 uses specified custom parameters
            
            ### What You Get
            - Recommended jet pump size for each well
            - Total field oil production prediction
            - Power fluid allocation across wells
            - Comprehensive visualizations and metrics
            """
            )
        return

    # Validation check for pump options
    if not nozzle_opts or not throat_opts:
        st.warning("⚠️ Please select at least one nozzle size and one throat ratio in the sidebar.")
        return

    # Run Optimization
    if st.button("🚀 Run Multi-Well Optimization", type="primary", use_container_width=True):
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as f:
                f.write(uploaded_file.getvalue())
                temp_csv_path = f.name

            # Load wells
            with st.spinner("Loading well configurations..."):
                from woffl.assembly.network_optimizer import NetworkOptimizer

                wells = load_wells_from_csv(temp_csv_path)
                st.success(f"✅ Loaded {len(wells)} wells from CSV")

                # Show wells loaded
                with st.expander("Wells Loaded"):
                    for well in wells:
                        st.write(f"- {well.well_name} ({well.field_model})")

            # Create optimizer
            pf_constraint = PowerFluidConstraint(total_rate=total_pf, pressure=pf_pressure, rho_pf=rho_pf)

            optimizer = NetworkOptimizer(
                wells=wells,
                power_fluid=pf_constraint,
                nozzle_options=nozzle_opts,
                throat_options=throat_opts,
                marginal_watercut=marginal_wc,
            )

            # Run batch simulations
            st.write("### Running Batch Simulations")
            progress_bar = st.progress(0)
            status_text = st.empty()

            def progress_callback(current, total, well_name):
                if total > 0:
                    progress_bar.progress(current / total)
                    status_text.text(f"Processing {well_name}... ({current}/{total})")

            optimizer.run_all_batch_simulations(progress_callback)
            progress_bar.empty()
            status_text.empty()

            st.success(f"✅ Completed batch simulations for {len(wells)} wells")

            # Run optimization
            st.write("### Running Optimization")
            with st.spinner(f"Running {opt_method} optimization..."):
                results = optimize(optimizer, method=opt_method)

            if not results:
                st.warning("⚠️ Optimization did not produce viable results. Try adjusting constraints or pump options.")
                return

            st.success(f"✅ Optimization complete! Allocated pumps to {len(results)} wells")

            # Display results
            st.write("## Optimization Results")

            res_tab1, res_tab2, res_tab3, res_tab4 = st.tabs(
                ["📊 Summary", "📋 Well Details", "📈 Visualizations", "💾 Export"]
            )

            with res_tab1:
                st.write("### Field-Level Metrics")
                metrics = optimizer.calculate_field_metrics()

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Oil Rate", f"{metrics['total_oil_rate']:.1f} BOPD")
                    st.metric("Wells Optimized", metrics["num_wells"])
                with col2:
                    st.metric("Total Water Rate", f"{metrics['total_water_rate']:.1f} BWPD")
                    st.metric("Field Watercut", f"{metrics['field_watercut']:.1%}")
                with col3:
                    st.metric("Power Fluid Used", f"{metrics['total_power_fluid']:.1f} BWPD")
                    st.metric("PF Utilization", f"{metrics['power_fluid_utilization']:.1%}")
                with col4:
                    st.metric("Avg Marginal Oil", f"{metrics['average_marginal_oil']:.3f}")
                    st.metric("Sonic Wells", f"{metrics['num_sonic']}/{metrics['num_wells']}")

            with res_tab2:
                st.write("### Well-Level Results")
                results_df = optimizer.to_dataframe()
                st.dataframe(results_df, use_container_width=True, height=400)

            with res_tab3:
                st.write("### Optimization Visualizations")

                viz_col1, viz_col2 = st.columns(2)

                with viz_col1:
                    st.write("#### Power Fluid Allocation")
                    fig_pie = create_power_fluid_pie_chart(results)
                    st.pyplot(fig_pie)
                    plt.close()

                    st.write("#### Oil Rate by Well")
                    fig_oil = create_oil_rate_bar_chart(results)
                    st.pyplot(fig_oil)
                    plt.close()

                    st.write("#### Oil vs Power Fluid Efficiency")
                    fig_eff = create_efficiency_scatter(results)
                    st.pyplot(fig_eff)
                    plt.close()

                with viz_col2:
                    st.write("#### Pump Configurations")
                    fig_config = create_pump_config_chart(results)
                    st.pyplot(fig_config)
                    plt.close()

                    st.write("#### Watercut Comparison")
                    fig_wc = create_watercut_comparison(results)
                    st.pyplot(fig_wc)
                    plt.close()

                    st.write("#### Marginal Oil Rates")
                    fig_marg = create_marginal_rate_chart(results)
                    st.pyplot(fig_marg)
                    plt.close()

            with res_tab4:
                st.write("### Export Results")
                results_df = optimizer.to_dataframe()
                csv_data = results_df.to_csv(index=False)

                st.download_button(
                    label="📥 Download Optimization Results CSV",
                    data=csv_data,
                    file_name="multi_well_optimization_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

                st.write("### Summary Statistics")
                st.write(f"- Total Wells: {len(results)}")
                st.write(f"- Total Oil: {metrics['total_oil_rate']:.1f} BOPD")
                st.write(f"- Total Power Fluid: {metrics['total_power_fluid']:.1f} BWPD")
                st.write(f"- Power Fluid Utilization: {metrics['power_fluid_utilization']:.1%}")

            # Clean up temp file
            os.unlink(temp_csv_path)

        except Exception as e:
            st.error(f"❌ Error during optimization: {str(e)}")
            st.exception(e)
