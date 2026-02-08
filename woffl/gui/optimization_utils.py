"""Optimization Utilities for GUI

Helper functions for multi-well optimization GUI integration.
"""

import pandas as pd
from woffl.assembly.batchrun import BatchPump
from woffl.assembly.network_optimizer import (
    NetworkOptimizer,
    OptimizationResult,
    PowerFluidConstraint,
    WellConfig,
    create_well_template_csv,
    load_wells_from_csv,
)
from woffl.flow.inflow import InFlow
from woffl.geometry.pipe import Annulus, Pipe
from woffl.geometry.wellprofile import WellProfile
from woffl.pvt.resmix import ResMix


def create_optimizer_from_csv(
    csv_path: str,
    total_power_fluid: float,
    power_fluid_pressure: float,
    rho_pf: float,
    nozzle_options: list[str],
    throat_options: list[str],
    marginal_watercut: float = 0.94,
) -> NetworkOptimizer:
    """Create NetworkOptimizer from CSV file

    Args:
        csv_path: Path to CSV with well configurations
        total_power_fluid: Total available power fluid (bbl/day)
        power_fluid_pressure: Power fluid pressure (psi)
        rho_pf: Power fluid density (lbm/ft³)
        nozzle_options: List of nozzle sizes to test
        throat_options: List of throat sizes to test
        marginal_watercut: Economic watercut threshold

    Returns:
        NetworkOptimizer instance

    Raises:
        ValueError: If CSV loading or validation fails
    """
    # Load wells from CSV
    wells = load_wells_from_csv(csv_path)

    # Create power fluid constraint
    power_fluid = PowerFluidConstraint(total_rate=total_power_fluid, pressure=power_fluid_pressure, rho_pf=rho_pf)

    # Create optimizer
    optimizer = NetworkOptimizer(wells, power_fluid, nozzle_options, throat_options, marginal_watercut)

    return optimizer


def format_optimization_results_table(results: list[OptimizationResult]) -> pd.DataFrame:
    """Format optimization results for display

    Args:
        results: List of optimization results

    Returns:
        Formatted DataFrame for display
    """
    data = []
    for r in results:
        data.append(
            {
                "Well": r.well_name,
                "Pump Config": f"{r.recommended_nozzle}{r.recommended_throat}",
                "Power Fluid (BWPD)": f"{r.allocated_power_fluid:.0f}",
                "Oil Rate (BOPD)": f"{r.predicted_oil_rate:.1f}",
                "Form Water (BWPD)": f"{r.predicted_formation_water:.1f}",
                "Total Water (BWPD)": f"{r.predicted_total_water:.1f}",
                "Total WC": f"{r.total_watercut:.2%}",
                "Marginal Oil Rate": f"{r.marginal_oil_rate:.3f}",
                "Sonic": "Yes" if r.sonic_status else "No",
            }
        )

    return pd.DataFrame(data)


def get_template_csv_content() -> str:
    """Get CSV template content for download

    Returns:
        CSV template as string
    """
    return create_well_template_csv()
