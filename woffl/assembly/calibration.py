"""Per-Well Model Calibration

Computes calibration factors by comparing model predictions to actual production
for each well's current jet pump configuration. Factors are applied post-optimization
to scale predicted rates without altering the optimizer's internal ranking.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from woffl.assembly.network_optimizer import NetworkOptimizer, OptimizationResult


@dataclass
class CalibrationResult:
    """Calibration data for a single well.

    Attributes:
        well_name: Well identifier
        current_nozzle: Current installed nozzle size
        current_throat: Current installed throat ratio
        model_oil: Model-predicted oil rate for current config (BOPD)
        actual_oil: Actual oil rate from well test (BOPD)
        model_pf: Model-predicted power fluid for current config (BWPD)
        actual_pf: Actual power fluid rate (BWPD), or None
        model_bhp: Model-predicted suction pressure (psi)
        actual_bhp: Actual BHP from well test (psi), or None
        calibration_factor: actual_oil / model_oil, clamped 0.3–2.0
    """

    well_name: str
    current_nozzle: str
    current_throat: str
    model_oil: float
    actual_oil: float
    model_pf: Optional[float] = None
    actual_pf: Optional[float] = None
    model_bhp: Optional[float] = None
    actual_bhp: Optional[float] = None
    calibration_factor: float = 1.0

    @property
    def oil_error_pct(self) -> float:
        """Absolute percentage error between model and actual oil."""
        if self.actual_oil > 0:
            return abs(self.model_oil - self.actual_oil) / self.actual_oil * 100
        return 0.0

    @property
    def quality_grade(self) -> str:
        """Calibration quality: 'good' <15%, 'fair' <30%, 'poor' >=30%."""
        err = self.oil_error_pct
        if err < 15:
            return "good"
        elif err < 30:
            return "fair"
        return "poor"


def run_calibration(
    optimizer: "NetworkOptimizer",
    actual_oil_map: dict[str, float],
    actual_pf_map: dict[str, float],
    actual_bhp_map: dict[str, float],
    current_jp_map: dict[str, tuple[str, str]],
) -> dict[str, CalibrationResult]:
    """Compute per-well calibration factors.

    For each well with actual data and a current JP config present in batch results,
    compare model prediction to actual oil rate and compute a scaling factor.

    Args:
        optimizer: NetworkOptimizer with completed batch simulations
        actual_oil_map: well_name -> actual oil rate (BOPD)
        actual_pf_map: well_name -> actual power fluid rate (BWPD)
        actual_bhp_map: well_name -> actual BHP (psi)
        current_jp_map: well_name -> (nozzle, throat) from JP history

    Returns:
        Dict mapping well_name to CalibrationResult
    """
    calibration = {}

    for well in optimizer.wells:
        name = well.well_name

        # Need both actual oil and current JP config
        if name not in actual_oil_map or name not in current_jp_map:
            continue

        nozzle, throat = current_jp_map[name]
        if nozzle is None or throat is None:
            continue

        actual_oil = actual_oil_map[name]
        if actual_oil <= 0:
            continue

        # Look up model prediction for this config
        perf = optimizer.get_pump_performance(name, nozzle, throat)
        if perf is None:
            # Config not in batch results — skip, factor stays 1.0
            continue

        model_oil = perf["oil_rate"]
        if model_oil <= 0:
            continue

        # Compute and clamp factor
        raw_factor = actual_oil / model_oil
        factor = max(0.3, min(2.0, raw_factor))

        calibration[name] = CalibrationResult(
            well_name=name,
            current_nozzle=nozzle,
            current_throat=throat,
            model_oil=model_oil,
            actual_oil=actual_oil,
            model_pf=perf.get("lift_water"),
            actual_pf=actual_pf_map.get(name),
            model_bhp=perf.get("suction_pressure"),
            actual_bhp=actual_bhp_map.get(name),
            calibration_factor=factor,
        )

    return calibration


def apply_calibration(
    results: list["OptimizationResult"],
    calibration: dict[str, CalibrationResult],
) -> list["OptimizationResult"]:
    """Apply calibration factors to optimization results.

    Creates new OptimizationResult objects with scaled oil and formation water.
    PF, suction pressure, sonic status, and mach are unchanged.

    Args:
        results: Original optimization results
        calibration: Dict mapping well_name to CalibrationResult

    Returns:
        New list of OptimizationResult with calibrated rates
    """
    from woffl.assembly.network_optimizer import OptimizationResult

    calibrated = []
    for r in results:
        factor = calibration[r.well_name].calibration_factor if r.well_name in calibration else 1.0
        calibrated.append(
            OptimizationResult(
                well_name=r.well_name,
                recommended_nozzle=r.recommended_nozzle,
                recommended_throat=r.recommended_throat,
                allocated_power_fluid=r.allocated_power_fluid,
                predicted_oil_rate=r.predicted_oil_rate * factor,
                predicted_formation_water=r.predicted_formation_water * factor,
                predicted_lift_water=r.predicted_lift_water,
                suction_pressure=r.suction_pressure,
                marginal_oil_rate=r.marginal_oil_rate,
                sonic_status=r.sonic_status,
                mach_te=r.mach_te,
            )
        )
    return calibrated


def compute_field_calibration_summary(calibration: dict[str, CalibrationResult]) -> dict:
    """Compute aggregate calibration statistics.

    Args:
        calibration: Dict mapping well_name to CalibrationResult

    Returns:
        Dict with median_factor, mean_factor, num_calibrated, num_skipped,
        worst_well, best_well
    """
    if not calibration:
        return {
            "median_factor": 1.0,
            "mean_factor": 1.0,
            "num_calibrated": 0,
            "num_skipped": 0,
            "worst_well": None,
            "best_well": None,
        }

    import statistics

    factors = [c.calibration_factor for c in calibration.values()]
    errors = {name: c.oil_error_pct for name, c in calibration.items()}

    worst = max(errors, key=errors.get) if errors else None
    best = min(errors, key=errors.get) if errors else None

    return {
        "median_factor": statistics.median(factors),
        "mean_factor": statistics.mean(factors),
        "num_calibrated": len(calibration),
        "num_skipped": 0,  # caller can override with total - calibrated
        "worst_well": worst,
        "best_well": best,
    }
