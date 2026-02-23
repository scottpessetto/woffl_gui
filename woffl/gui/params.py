"""Simulation Parameters Dataclass

Bundles the many individual parameters that travel together through the GUI
into a single typed container. This replaces the 15+ loose parameters that
were previously threaded through every function call.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SimulationParams:
    """All parameters needed to run a single-well jetpump simulation.

    This dataclass is populated from sidebar widgets and passed to tab renderers,
    replacing the pattern of threading 12+ individual parameters through every call.

    Attributes:
        nozzle_no: Nozzle size string (e.g., "12")
        area_ratio: Throat area ratio (e.g., "B")
        ken: Nozzle loss coefficient
        kth: Throat loss coefficient
        kdi: Diffuser loss coefficient
        tubing_od: Tubing outer diameter, inches
        tubing_thickness: Tubing wall thickness, inches
        casing_od: Casing outer diameter, inches
        casing_thickness: Casing wall thickness, inches
        form_wc: Formation water cut, fraction
        form_gor: Gas-oil ratio, scf/bbl
        form_temp: Formation temperature, °F
        field_model: Field PVT model ("Schrader" or "Kuparuk")
        surf_pres: Surface pressure, psi
        jpump_tvd: Jetpump true vertical depth, ft
        rho_pf: Power fluid density, lbm/ft³
        ppf_surf: Power fluid surface pressure, psi
        qwf: Well flow rate for IPR, bbl/day
        pwf: Flowing bottom hole pressure for IPR, psi
        pres: Reservoir pressure, psi
        nozzle_batch_options: Nozzle sizes for batch analysis
        throat_batch_options: Throat ratios for batch analysis
        water_type: Water type for analysis ("lift" or "total")
        marginal_watercut: Economic threshold for water handling
        power_fluid_min: Min power fluid pressure for range analysis, psi
        power_fluid_max: Max power fluid pressure for range analysis, psi
        power_fluid_step: Step size for power fluid pressure range, psi
        selected_well: Selected well name or "Custom"
        well_data: Well characteristics dict from jp_chars.csv, or None
    """

    # Jetpump parameters
    nozzle_no: str = "12"
    area_ratio: str = "B"
    ken: float = 0.03
    kth: float = 0.3
    kdi: float = 0.4

    # Pipe parameters
    tubing_od: float = 4.5
    tubing_thickness: float = 0.5
    casing_od: float = 6.875
    casing_thickness: float = 0.5

    # Formation parameters
    form_wc: float = 0.50
    form_gor: int = 250
    form_temp: int = 70
    field_model: str = "Schrader"

    # Well parameters
    surf_pres: int = 210
    jpump_tvd: int = 4065
    rho_pf: float = 62.4
    ppf_surf: int = 3168

    # Inflow parameters
    qwf: int = 750
    pwf: int = 500
    pres: int = 1700

    # Batch run parameters
    nozzle_batch_options: list[str] = field(default_factory=lambda: ["9", "10", "11", "12", "13", "14", "15"])
    throat_batch_options: list[str] = field(default_factory=lambda: ["A", "B", "C", "D"])
    water_type: str = "total"
    marginal_watercut: float = 0.94

    # Power fluid range parameters
    power_fluid_min: int = 1800
    power_fluid_max: int = 3600
    power_fluid_step: int = 200

    # Well selection
    selected_well: str = "Custom"
    well_data: Optional[dict] = None
