"""Simulation Parameters Dataclass

Bundles the many individual parameters that travel together through the GUI
into a single typed container. This replaces the 15+ loose parameters that
were previously threaded through every function call.
"""

from dataclasses import dataclass, field
from typing import Optional

NOZZLE_OPTIONS = ["8", "9", "10", "11", "12", "13", "14", "15"]
THROAT_OPTIONS = ["X", "A", "B", "C", "D", "E"]


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
        jpump_direction: Circulation direction ("forward" or "reverse")
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
        qwf: TOTAL LIQUID rate at ``pwf`` (BLPD) — the IPR anchor rate. See
            the RATE CONVENTION note below; use ``qwf_oil`` / ``qwf_water``
            rather than re-deriving them at a call site.
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

    RATE CONVENTION (Scott, 2026-08-03)
    -----------------------------------
    ``qwf`` is the well's TOTAL LIQUID rate at ``pwf`` in BLPD, EXCLUDING
    returned power fluid — formation oil + formation water. It is the measured
    quantity: ``vw_well_test.WtTotalFluid`` is what the Vogel fit is built on
    (``ipr_analyzer.compute_vogel_coefficients`` returns it as ``qwf``), and
    it is what ``prop_hist.ipr_qwf_liq`` stores. Oil and water are DERIVED
    from it through the assumed / 🔒 locked water cut:

        oil   = qwf * (1 - form_wc)      ``qwf_oil``
        water = qwf * form_wc            ``qwf_water``

    It used to run the other way — the sidebar held OIL and the liquid rate
    was back-computed as ``oil / (1 - wc)``. That made the *derived* number
    the persisted one, so B-28's stored 2135.29 BLPD matched no well test
    (it was 363 BOPD / (1 - 0.83)) and every WC edit silently rescaled the
    engineer's anchor. Derive downward from the measurement, never upward.

    ``woffl.flow.inflow.InFlow`` keeps its library contract of a SINGLE-PHASE
    rate, so every construction site feeds it :attr:`inflow_rate`, never
    ``qwf`` raw.
    """

    # Jetpump parameters
    nozzle_no: str = "12"
    area_ratio: str = "B"
    ken: float = 0.03
    kth: float = 0.3
    kdi: float = 0.4
    jpump_direction: str = "reverse"

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
    # Water-pump (dewatering) mode: model a 100%-water, no-oil well to see what
    # suction / power fluid it takes to flow it. When True the sidebar forces
    # form_wc = 1.0 and qwf is the well's WATER deliverability. Default False.
    model_as_water: bool = False

    # PVT overrides (None = use field_model preset)
    oil_api: Optional[float] = None
    gas_sg: Optional[float] = None
    wat_sg: Optional[float] = None
    bubble_point: Optional[float] = None

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
    nozzle_batch_options: list[str] = field(
        default_factory=lambda: ["9", "10", "11", "12", "13", "14", "15"]
    )
    throat_batch_options: list[str] = field(
        default_factory=lambda: ["A", "B", "C", "D"]
    )
    water_type: str = "total"
    marginal_watercut: float = 0.94

    # Power fluid range parameters
    power_fluid_min: int = 1800
    power_fluid_max: int = 3600
    power_fluid_step: int = 200

    # Well selection
    selected_well: str = "Custom"
    well_data: Optional[dict] = None

    @property
    def qwf_oil(self) -> float:
        """Oil rate at ``pwf`` (BOPD) — derived: ``qwf * (1 - form_wc)``."""
        return float(self.qwf) * (1.0 - float(self.form_wc))

    @property
    def qwf_water(self) -> float:
        """Formation water rate at ``pwf`` (BWPD) — ``qwf * form_wc``. Excludes
        returned power fluid (that is a PF-circuit rate, not inflow)."""
        return float(self.qwf) * float(self.form_wc)

    @property
    def inflow_rate(self) -> float:
        """The single-phase rate to hand ``InFlow`` (the IPR is anchored on it).

        Normally the oil rate. In water-pump (dewatering) mode the sidebar
        forces ``form_wc = 1.0``, which would make the oil rate identically
        zero and collapse the curve — there the produced phase IS the liquid,
        so the full ``qwf`` is the anchor.
        """
        return float(self.qwf) if self.model_as_water else self.qwf_oil

