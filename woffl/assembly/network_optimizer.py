"""Network Optimizer

Multi-well jet pump optimization module for maximizing oil production
given constrained power fluid resources.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from woffl.assembly.batchrun import BatchPump
from woffl.flow.inflow import InFlow
from woffl.geometry.jetpump import JetPump
from woffl.geometry.pipe import Pipe
from woffl.geometry.wellprofile import WellProfile
from woffl.pvt.resmix import ResMix


@dataclass
class WellConfig:
    """Configuration for a single well in the optimization

    Attributes:
        well_name (str): Well identifier
        res_pres (float): Reservoir pressure, psi
        form_temp (float): Formation temperature, °F
        jpump_tvd (float): Jet pump true vertical depth, ft
        jpump_md (float): Jet pump measured depth, ft
        tubing_od (float): Tubing outer diameter, inches
        tubing_thickness (float): Tubing wall thickness, inches
        casing_od (float): Casing outer diameter, inches
        casing_thickness (float): Casing wall thickness, inches
        form_wc (float): Formation water cut, fraction
        form_gor (float): Gas-oil ratio, scf/bbl
        field_model (str): Field PVT model ("Schrader" or "Kuparuk")
        surf_pres (float): Surface pressure, psi
        qwf (float): Well flow rate for IPR, bbl/day
        pwf (float): Flowing bottom hole pressure for IPR, psi
        use_survey (bool): Whether to use survey data if available
    """

    well_name: str
    res_pres: float
    form_temp: float
    jpump_tvd: float
    jpump_md: Optional[float] = None
    tubing_od: float = 4.5
    tubing_thickness: float = 0.271
    casing_od: float = 6.875
    casing_thickness: float = 0.5
    form_wc: float = 0.5
    form_gor: float = 250.0
    field_model: str = "Schrader"
    surf_pres: float = 210.0
    qwf: float = 750.0
    pwf: float = 500.0
    use_survey: bool = True

    def __post_init__(self):
        """Validate configuration on initialization"""
        if self.jpump_md is None:
            self.jpump_md = self.jpump_tvd

        # Validate field model
        if self.field_model not in ["Schrader", "Kuparuk"]:
            raise ValueError(f"field_model must be 'Schrader' or 'Kuparuk', got '{self.field_model}'")

        # Validate ranges
        if not (400 <= self.res_pres <= 5000):
            raise ValueError(f"res_pres must be between 400-5000 psi, got {self.res_pres}")
        if not (32 <= self.form_temp <= 350):
            raise ValueError(f"form_temp must be between 32-350 °F, got {self.form_temp}")
        if not (2500 <= self.jpump_tvd <= 8000):
            raise ValueError(f"jpump_tvd must be between 2500-8000 ft, got {self.jpump_tvd}")
        if not (0.0 <= self.form_wc <= 1.0):
            raise ValueError(f"form_wc must be between 0.0-1.0, got {self.form_wc}")


@dataclass
class PowerFluidConstraint:
    """Power fluid system constraint

    Attributes:
        total_rate (float): Total available power fluid, bbl/day
        pressure (float): Power fluid pressure at surface, psi
        rho_pf (float): Power fluid density, lbm/ft³
    """

    total_rate: float
    pressure: float
    rho_pf: float = 62.4

    def __post_init__(self):
        """Validate constraints"""
        if self.total_rate <= 0:
            raise ValueError(f"total_rate must be positive, got {self.total_rate}")
        if not (1000 <= self.pressure <= 5000):
            raise ValueError(f"pressure must be between 1000-5000 psi, got {self.pressure}")
        if not (50.0 <= self.rho_pf <= 70.0):
            raise ValueError(f"rho_pf must be between 50-70 lbm/ft³, got {self.rho_pf}")


@dataclass
class OptimizationResult:
    """Results for a single well in optimization

    Attributes:
        well_name (str): Well identifier
        recommended_nozzle (str): Recommended nozzle size
        recommended_throat (str): Recommended throat size
        allocated_power_fluid (float): Allocated power fluid rate, bbl/day
        predicted_oil_rate (float): Predicted oil rate, bbl/day
        predicted_formation_water (float): Predicted formation water rate, bbl/day
        predicted_lift_water (float): Predicted lift water rate, bbl/day
        suction_pressure (float): Suction pressure, psi
        marginal_oil_rate (float): Marginal oil per power fluid (dOil/dPF)
        sonic_status (bool): Whether pump is operating at sonic conditions
        mach_te (float): Mach number at throat entry
    """

    well_name: str
    recommended_nozzle: str
    recommended_throat: str
    allocated_power_fluid: float
    predicted_oil_rate: float
    predicted_formation_water: float
    predicted_lift_water: float
    suction_pressure: float
    marginal_oil_rate: float
    sonic_status: bool
    mach_te: float

    @property
    def predicted_total_water(self) -> float:
        """Calculate total water rate"""
        return self.predicted_formation_water + self.predicted_lift_water

    @property
    def total_watercut(self) -> float:
        """Calculate total watercut"""
        total_liquid = self.predicted_oil_rate + self.predicted_total_water
        if total_liquid > 0:
            return self.predicted_total_water / total_liquid
        return 0.0


class NetworkOptimizer:
    """Container for multi-well optimization

    This class coordinates the optimization of jet pump sizing across
    multiple wells given a constrained power fluid supply.
    """

    def __init__(
        self,
        wells: list[WellConfig],
        power_fluid: PowerFluidConstraint,
        nozzle_options: list[str],
        throat_options: list[str],
        marginal_watercut: float = 0.94,
    ):
        """Initialize network optimizer

        Args:
            wells: List of well configurations
            power_fluid: Power fluid constraint
            nozzle_options: Available nozzle sizes to test
            throat_options: Available throat sizes to test
            marginal_watercut: Economic watercut threshold
        """
        self.wells = wells
        self.power_fluid = power_fluid
        self.nozzle_options = nozzle_options
        self.throat_options = throat_options
        self.marginal_watercut = marginal_watercut

        # Results storage
        self.batch_results: dict[str, BatchPump] = {}
        self.optimization_results: Optional[list[OptimizationResult]] = None

    def get_well_by_name(self, well_name: str) -> Optional[WellConfig]:
        """Get well configuration by name"""
        for well in self.wells:
            if well.well_name == well_name:
                return well
        return None

    def _create_well_objects(self, well: WellConfig):
        """Create simulation objects for a well

        Args:
            well: Well configuration

        Returns:
            tuple: (tube, wellprofile, inflow, res_mix)
        """
        from woffl.flow.inflow import InFlow
        from woffl.geometry.pipe import Pipe
        from woffl.geometry.wellprofile import WellProfile
        from woffl.pvt.blackoil import BlackOil
        from woffl.pvt.formgas import FormGas
        from woffl.pvt.formwat import FormWater
        from woffl.pvt.resmix import ResMix

        # Create tubing
        tube = Pipe(out_dia=well.tubing_od, thick=well.tubing_thickness)

        # Create well profile
        field_model_lower = well.field_model.lower()
        if field_model_lower == "schrader":
            well_profile = WellProfile.schrader()
        else:
            well_profile = WellProfile.kuparuk()

        # Adjust well profile for jetpump depth
        # Use the well's jpump_md directly (it was set from jpump_tvd in __post_init__ if not provided)
        try:
            #  Recreate well profile with correct jetpump MD
            well_profile = WellProfile(
                md_list=well_profile.md_ray, vd_list=well_profile.vd_ray, jetpump_md=well.jpump_md
            )
        except Exception:
            # If this fails, use the default well profile as-is
            pass

        # Create inflow
        inflow = InFlow(qwf=well.qwf, pwf=well.pwf, pres=well.res_pres)

        # Create reservoir mix
        if field_model_lower == "schrader":
            oil = BlackOil.schrader()
            water = FormWater.schrader()
            gas = FormGas.schrader()
        else:
            oil = BlackOil.kuparuk()
            water = FormWater.kuparuk()
            gas = FormGas.kuparuk()

        res_mix = ResMix(wc=well.form_wc, fgor=well.form_gor, oil=oil, wat=water, gas=gas)

        return tube, well_profile, inflow, res_mix

    def run_all_batch_simulations(self, progress_callback=None) -> dict[str, BatchPump]:
        """Run batch simulations for all wells

        Args:
            progress_callback: Optional callback function(current, total, well_name)
                             for tracking progress

        Returns:
            Dictionary mapping well_name to BatchPump objects
        """
        self.batch_results = {}
        total_wells = len(self.wells)

        for idx, well in enumerate(self.wells):
            if progress_callback:
                progress_callback(idx, total_wells, well.well_name)

            # Create well-specific objects
            tube, well_profile, inflow, res_mix = self._create_well_objects(well)

            # Create jet pump list
            jp_list = BatchPump.jetpump_list(self.nozzle_options, self.throat_options)

            # Create and run BatchPump
            batch_pump = BatchPump(
                pwh=well.surf_pres,
                tsu=well.form_temp,
                rho_pf=self.power_fluid.rho_pf,
                ppf_surf=self.power_fluid.pressure,
                wellbore=tube,
                wellprof=well_profile,
                ipr_su=inflow,
                prop_su=res_mix,
                wellname=well.well_name,
            )

            # Run batch simulation (don't raise errors, capture them)
            batch_pump.batch_run(jp_list, debug=False)

            # Process results
            try:
                batch_pump.process_results()
            except (ValueError, RuntimeError) as e:
                # If curve fitting fails, that's okay - we can still use raw results
                print(f"Warning: Curve fitting failed for {well.well_name}: {e}")

            self.batch_results[well.well_name] = batch_pump

        if progress_callback:
            progress_callback(total_wells, total_wells, "Complete")

        return self.batch_results

    def get_power_fluid_requirement(self, well_name: str, nozzle: str, throat: str) -> Optional[float]:
        """Get power fluid requirement for specific pump configuration

        Args:
            well_name: Name of the well
            nozzle: Nozzle size
            throat: Throat ratio

        Returns:
            Power fluid rate in bbl/day, or None if not found
        """
        if well_name not in self.batch_results:
            return None

        batch_pump = self.batch_results[well_name]

        # Find the row with matching nozzle and throat
        mask = (batch_pump.df["nozzle"] == nozzle) & (batch_pump.df["throat"] == throat)
        matching_rows = batch_pump.df[mask]

        if matching_rows.empty:
            return None

        # Return lift water rate (power fluid)
        lift_wat = matching_rows.iloc[0]["lift_wat"]

        # Return None if NaN
        if pd.isna(lift_wat):
            return None

        return float(lift_wat)

    def get_pump_performance(self, well_name: str, nozzle: str, throat: str) -> Optional[dict]:
        """Get complete performance metrics for specific pump configuration

        Args:
            well_name: Name of the well
            nozzle: Nozzle size
            throat: Throat ratio

        Returns:
            Dictionary with performance metrics, or None if not found
        """
        if well_name not in self.batch_results:
            return None

        batch_pump = self.batch_results[well_name]

        # Find the row with matching nozzle and throat
        mask = (batch_pump.df["nozzle"] == nozzle) & (batch_pump.df["throat"] == throat)
        matching_rows = batch_pump.df[mask]

        if matching_rows.empty:
            return None

        row = matching_rows.iloc[0]

        # Return None if oil rate is NaN (simulation failed)
        if pd.isna(row["qoil_std"]):
            return None

        return {
            "oil_rate": float(row["qoil_std"]),
            "formation_water": float(row["form_wat"]),
            "lift_water": float(row["lift_wat"]),
            "total_water": float(row["totl_wat"]),
            "suction_pressure": float(row["psu_solv"]),
            "sonic_status": bool(row["sonic_status"]),
            "mach_te": float(row["mach_te"]),
            "marginal_oil_lift_water": float(row.get("molwr", 0.0)) if pd.notna(row.get("molwr")) else 0.0,
            "marginal_oil_total_water": float(row.get("motwr", 0.0)) if pd.notna(row.get("motwr")) else 0.0,
        }

    def validate_allocation(self, allocation: dict[str, tuple[str, str]]) -> tuple[bool, float, dict]:
        """Validate if allocation is feasible given power fluid constraint

        Args:
            allocation: Dict mapping well_name to (nozzle, throat) tuple

        Returns:
            Tuple of (is_feasible, total_power_fluid_used, details_dict)

            details_dict contains:
                - 'total_pf': Total  power fluid used
                - 'available_pf': Available power fluid
                - 'utilization': Fraction of available power fluid used
                - 'well_allocations': Dict of {well_name: power_fluid_rate}
                - 'failed_wells': List of wells with invalid configurations
        """
        total_pf = 0.0
        well_allocations = {}
        failed_wells = []

        for well_name, (nozzle, throat) in allocation.items():
            pf_req = self.get_power_fluid_requirement(well_name, nozzle, throat)

            if pf_req is None:
                failed_wells.append(well_name)
            else:
                well_allocations[well_name] = pf_req
                total_pf += pf_req

        is_feasible = total_pf <= self.power_fluid.total_rate and len(failed_wells) == 0

        details = {
            "total_pf": total_pf,
            "available_pf": self.power_fluid.total_rate,
            "utilization": total_pf / self.power_fluid.total_rate if self.power_fluid.total_rate > 0 else 0.0,
            "well_allocations": well_allocations,
            "failed_wells": failed_wells,
        }

        return is_feasible, total_pf, details

    def calculate_field_metrics(self, results: Optional[list[OptimizationResult]] = None) -> dict:
        """Calculate aggregate field-level metrics

        Args:
            results: Optimization results (uses self.optimization_results if None)

        Returns:
            Dictionary of field metrics
        """
        if results is None:
            results = self.optimization_results

        if not results:
            return {}

        total_oil = sum(r.predicted_oil_rate for r in results)
        total_form_water = sum(r.predicted_formation_water for r in results)
        total_lift_water = sum(r.allocated_power_fluid for r in results)
        total_water = total_form_water + total_lift_water
        total_liquid = total_oil + total_water

        return {
            "total_oil_rate": total_oil,
            "total_formation_water": total_form_water,
            "total_lift_water": total_lift_water,
            "total_water_rate": total_water,
            "total_power_fluid": total_lift_water,
            "field_watercut": total_water / total_liquid if total_liquid > 0 else 0.0,
            "field_oil_cut": total_oil / total_liquid if total_liquid > 0 else 0.0,
            "average_marginal_oil": np.mean([r.marginal_oil_rate for r in results]),
            "num_wells": len(results),
            "num_sonic": sum(1 for r in results if r.sonic_status),
            "power_fluid_utilization": total_lift_water / self.power_fluid.total_rate,
        }

    def to_dataframe(self, results: Optional[list[OptimizationResult]] = None) -> pd.DataFrame:
        """Convert optimization results to DataFrame

        Args:
            results: Optimization results (uses self.optimization_results if None)

        Returns:
            DataFrame with optimization results
        """
        if results is None:
            results = self.optimization_results

        if not results:
            return pd.DataFrame()

        data = []
        for r in results:
            data.append(
                {
                    "Well": r.well_name,
                    "Nozzle": r.recommended_nozzle,
                    "Throat": r.recommended_throat,
                    "Power Fluid (BWPD)": r.allocated_power_fluid,
                    "Oil Rate (BOPD)": r.predicted_oil_rate,
                    "Formation Water (BWPD)": r.predicted_formation_water,
                    "Lift Water (BWPD)": r.predicted_lift_water,
                    "Total Water (BWPD)": r.predicted_total_water,
                    "Suction Pressure (psi)": r.suction_pressure,
                    "Marginal Oil Rate": r.marginal_oil_rate,
                    "Sonic": r.sonic_status,
                    "Mach Number": r.mach_te,
                    "Total Watercut": r.total_watercut,
                }
            )

        return pd.DataFrame(data)


def load_jp_chars(jp_chars_path: Optional[str] = None) -> dict:
    """Load jet pump characteristics database

    Args:
        jp_chars_path: Path to jp_chars.csv (auto-detected if None)

    Returns:
        Dictionary mapping well name to characteristics
    """
    if jp_chars_path is None:
        # Auto-detect path relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        jp_chars_path = os.path.join(current_dir, "..", "jp_data", "jp_chars.csv")

    try:
        df = pd.read_csv(jp_chars_path)
        return df.set_index("Well").to_dict("index")
    except FileNotFoundError:
        print(f"Warning: jp_chars.csv not found at {jp_chars_path}")
        return {}


def load_wells_from_csv(csv_path: str, jp_chars_path: Optional[str] = None) -> list[WellConfig]:
    """Load well configurations from CSV with fallback to jp_chars database

    Args:
        csv_path: Path to CSV file with well configurations
        jp_chars_path: Path to jp_chars.csv database (auto-detected if None)

    Returns:
        List of WellConfig objects

    Raises:
        ValueError: If required fields are missing or invalid
        FileNotFoundError: If CSV file not found
    """
    # Load jp_chars database
    jp_chars_dict = load_jp_chars(jp_chars_path)

    # Read input CSV
    input_df = pd.read_csv(csv_path)

    # Validate required column
    if "Well" not in input_df.columns:
        raise ValueError("CSV must have 'Well' column")

    well_configs = []
    errors = []

    for idx, row in input_df.iterrows():
        well_name = row["Well"]

        # Skip empty rows
        if pd.isna(well_name) or str(well_name).strip() == "":
            continue

        try:
            # Start with database values if well exists
            if well_name in jp_chars_dict:
                base_config = jp_chars_dict[well_name].copy()
            else:
                base_config = {}

            # Override with CSV values (if provided and not NaN)
            for col in input_df.columns:
                if col != "Well" and col != "comments" and pd.notna(row[col]):
                    base_config[col] = row[col]

            # Map database fields to WellConfig parameters with proper defaults
            # Check for required fields first
            if "res_pres" not in base_config:
                raise ValueError(f"res_pres is required but not found in database or CSV")
            if "JP_TVD" not in base_config and "jpump_tvd" not in base_config:
                raise ValueError(f"JP_TVD is required but not found in database or CSV")

            config_params = {
                "well_name": well_name,
                "res_pres": float(base_config["res_pres"]),
                "form_temp": float(base_config.get("form_temp", 70)),
                "jpump_tvd": float(base_config.get("JP_TVD") or base_config.get("jpump_tvd", 4000)),
                "jpump_md": float(
                    base_config.get("JP_MD") or base_config.get("JP_TVD") or base_config.get("jpump_tvd", 4000)
                ),
                "tubing_od": float(base_config.get("out_dia") or base_config.get("tubing_od", 4.5)),
                "tubing_thickness": float(base_config.get("thick") or base_config.get("tubing_thickness", 0.271)),
                "casing_od": float(base_config.get("casing_od", 6.875)),
                "casing_thickness": float(base_config.get("casing_thick") or base_config.get("casing_thickness", 0.5)),
                "form_wc": float(base_config.get("form_wc", 0.5)),
                "form_gor": float(base_config.get("form_gor", 250)),
                "field_model": (
                    "Schrader"
                    if base_config.get("is_sch", base_config.get("field_model", "Schrader"))
                    in [True, "TRUE", "True", "Schrader"]
                    else "Kuparuk"
                ),
                "surf_pres": float(base_config.get("surf_pres", 210)),
                "qwf": float(base_config.get("qwf", 750)),
                "pwf": float(base_config.get("pwf", 500)),
            }

            # Create WellConfig (validation happens in __post_init__)
            config = WellConfig(**config_params)
            well_configs.append(config)

        except KeyError as e:
            errors.append(f"Row {idx+2} (Well {well_name}): Missing required field {e}")
        except ValueError as e:
            errors.append(f"Row {idx+2} (Well {well_name}): {str(e)}")
        except Exception as e:
            errors.append(f"Row {idx+2} (Well {well_name}): Unexpected error - {str(e)}")

    if errors:
        raise ValueError("Errors loading well configurations:\n" + "\n".join(errors))

    if not well_configs:
        raise ValueError("No valid well configurations found in CSV")

    return well_configs


def create_well_template_csv() -> str:
    """Create CSV template string for well configurations

    Returns:
        CSV template as string
    """
    template = """Well,res_pres,form_temp,JP_TVD,JP_MD,out_dia,thick,casing_od,casing_thick,form_wc,form_gor,field_model,surf_pres,qwf,pwf,comments
MPB-28,,,,,,,,,,,,,,,Auto-populated from jp_chars.csv
MPB-30,,,,,,,,,,,,,,,Auto-populated from jp_chars.csv
MPE-35,,,,,,,,,,,,,,,Auto-populated from jp_chars.csv
CustomWell-1,1500,75,4000,4500,4.5,0.271,6.875,0.5,0.45,250,Schrader,210,800,500,Example custom well
CustomWell-2,1600,80,4200,4700,4.5,0.271,6.875,0.5,0.50,300,Kuparuk,220,750,550,Example custom well"""
    return template


def validate_well_config(config: WellConfig) -> tuple[bool, list[str]]:
    """Validate well configuration

    Args:
        config: Well configuration to validate

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    try:
        # Validation happens in WellConfig __post_init__
        # This function can add additional business logic validation

        # Example: Check for unrealistic combinations
        if config.jpump_tvd > 6000 and config.tubing_od < 3.0:
            errors.append("Deep wells (>6000 ft) typically require larger tubing (>=3.5 inches)")

        if config.form_wc > 0.95 and config.form_gor < 100:
            errors.append("High watercut wells (>95%) with low GOR (<100) may have poor performance")

    except Exception as e:
        errors.append(str(e))

    return (len(errors) == 0, errors)
