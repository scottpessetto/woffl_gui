"""Optimization Results Analysis

Utilities for analyzing and comparing optimization results.
"""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from woffl.assembly.network_optimizer import NetworkOptimizer, OptimizationResult


def compare_scenarios(
    baseline_results: list["OptimizationResult"], optimized_results: list["OptimizationResult"]
) -> dict:
    """Compare baseline vs optimized scenarios

    Args:
        baseline_results: Baseline optimization results
        optimized_results: Optimized results to compare against baseline

    Returns:
        Dictionary with comparison metrics
    """
    # Calculate totals for baseline
    baseline_oil = sum(r.predicted_oil_rate for r in baseline_results)
    baseline_water = sum(r.predicted_total_water for r in baseline_results)
    baseline_pf = sum(r.allocated_power_fluid for r in baseline_results)

    # Calculate totals for optimized
    optimized_oil = sum(r.predicted_oil_rate for r in optimized_results)
    optimized_water = sum(r.predicted_total_water for r in optimized_results)
    optimized_pf = sum(r.allocated_power_fluid for r in optimized_results)

    # Calculate improvements
    oil_improvement = optimized_oil - baseline_oil
    oil_improvement_pct = (oil_improvement / baseline_oil * 100) if baseline_oil > 0 else 0

    water_change = optimized_water - baseline_water
    water_change_pct = (water_change / baseline_water * 100) if baseline_water > 0 else 0

    pf_change = optimized_pf - baseline_pf
    pf_change_pct = (pf_change / baseline_pf * 100) if baseline_pf > 0 else 0

    return {
        "baseline_oil_rate": baseline_oil,
        "optimized_oil_rate": optimized_oil,
        "oil_improvement": oil_improvement,
        "oil_improvement_pct": oil_improvement_pct,
        "baseline_water_rate": baseline_water,
        "optimized_water_rate": optimized_water,
        "water_change": water_change,
        "water_change_pct": water_change_pct,
        "baseline_power_fluid": baseline_pf,
        "optimized_power_fluid": optimized_pf,
        "power_fluid_change": pf_change,
        "power_fluid_change_pct": pf_change_pct,
        "num_baseline_wells": len(baseline_results),
        "num_optimized_wells": len(optimized_results),
    }


def sensitivity_analysis(
    optimizer: "NetworkOptimizer", power_fluid_range: tuple[float, float], num_points: int = 10, method: str = "greedy"
) -> pd.DataFrame:
    """Run sensitivity analysis on power fluid availability

    Tests different power fluid availability levels to see how total oil
    production varies.

    Args:
        optimizer: NetworkOptimizer instance (with batch results already run)
        power_fluid_range: (min_pf, max_pf) tuple in bbl/day
        num_points: Number of power fluid levels to test
        method: Optimization method to use

    Returns:
        DataFrame with columns: power_fluid, total_oil, total_water, num_wells, utilization
    """
    from woffl.assembly.network_optimizer import PowerFluidConstraint
    from woffl.assembly.optimization_algorithms import optimize

    if not optimizer.batch_results:
        raise ValueError("Must run batch simulations before sensitivity analysis")

    min_pf, max_pf = power_fluid_range
    pf_levels = np.linspace(min_pf, max_pf, num_points)

    # Store original constraint
    original_constraint = optimizer.power_fluid

    results = []
    for pf_level in pf_levels:
        # Update power fluid constraint
        optimizer.power_fluid = PowerFluidConstraint(
            total_rate=pf_level, pressure=original_constraint.pressure, rho_pf=original_constraint.rho_pf
        )

        # Run optimization at this power fluid level
        opt_results = optimize(optimizer, method=method)

        # Calculate metrics
        total_oil = sum(r.predicted_oil_rate for r in opt_results)
        total_water = sum(r.predicted_total_water for r in opt_results)
        total_pf_used = sum(r.allocated_power_fluid for r in opt_results)
        utilization = total_pf_used / pf_level if pf_level > 0 else 0

        results.append(
            {
                "power_fluid_available": pf_level,
                "total_oil_rate": total_oil,
                "total_water_rate": total_water,
                "power_fluid_used": total_pf_used,
                "num_wells": len(opt_results),
                "utilization": utilization,
                "oil_per_pf": total_oil / total_pf_used if total_pf_used > 0 else 0,
            }
        )

    # Restore original constraint
    optimizer.power_fluid = original_constraint

    return pd.DataFrame(results)


def analyze_well_contributions(results: list["OptimizationResult"]) -> pd.DataFrame:
    """Analyze individual well contributions to field production

    Args:
        results: List of optimization results

    Returns:
        DataFrame with well-level contribution analysis
    """
    data = []

    # Calculate totals
    total_oil = sum(r.predicted_oil_rate for r in results)
    total_pf = sum(r.allocated_power_fluid for r in results)

    for r in results:
        oil_share = r.predicted_oil_rate / total_oil if total_oil > 0 else 0
        pf_share = r.allocated_power_fluid / total_pf if total_pf > 0 else 0

        data.append(
            {
                "well_name": r.well_name,
                "pump_config": f"{r.recommended_nozzle}{r.recommended_throat}",
                "oil_rate": r.predicted_oil_rate,
                "oil_share_pct": oil_share * 100,
                "power_fluid": r.allocated_power_fluid,
                "pf_share_pct": pf_share * 100,
                "efficiency": r.predicted_oil_rate / r.allocated_power_fluid if r.allocated_power_fluid > 0 else 0,
                "watercut": r.total_watercut,
                "sonic": r.sonic_status,
            }
        )

    df = pd.DataFrame(data)
    df = df.sort_values("oil_rate", ascending=False)

    return df


def identify_bottlenecks(optimizer: "NetworkOptimizer") -> dict:
    """Identify optimization bottlenecks

    Analyzes which wells are constraining overall performance and why.

    Args:
        optimizer: NetworkOptimizer with optimization results

    Returns:
        Dictionary with bottleneck analysis
    """
    if not optimizer.optimization_results:
        return {"error": "No optimization results available"}

    results = optimizer.optimization_results

    # Check power fluid utilization
    metrics = optimizer.calculate_field_metrics()
    pf_utilization = metrics["power_fluid_utilization"]

    # Find wells with low marginal oil rates
    marginal_rates = [r.marginal_oil_rate for r in results]
    avg_marginal = np.mean(marginal_rates)

    low_performers = [r.well_name for r in results if r.marginal_oil_rate < avg_marginal * 0.5]

    # Find wells operating at sonic conditions
    sonic_wells = [r.well_name for r in results if r.sonic_status]

    # Find wells with high watercut
    high_watercut_wells = [r.well_name for r in results if r.total_watercut > 0.9]

    # Determine primary bottleneck
    if pf_utilization >= 0.95:
        primary_bottleneck = "power_fluid_constraint"
        recommendation = "Increase available power fluid to boost production"
    elif len(low_performers) > len(results) * 0.5:
        primary_bottleneck = "well_productivity"
        recommendation = "Focus on wells with higher productivity indices"
    elif len(sonic_wells) > len(results) * 0.3:
        primary_bottleneck = "sonic_limitations"
        recommendation = "Consider larger throat sizes to avoid sonic choking"
    else:
        primary_bottleneck = "balanced"
        recommendation = "System is well-balanced"

    return {
        "primary_bottleneck": primary_bottleneck,
        "recommendation": recommendation,
        "pf_utilization": pf_utilization,
        "low_performing_wells": low_performers,
        "sonic_wells": sonic_wells,
        "high_watercut_wells": high_watercut_wells,
        "avg_marginal_oil_rate": avg_marginal,
    }


def calculate_incremental_value(
    optimizer: "NetworkOptimizer",
    oil_price: float = 80.0,  # $/bbl
    water_cost: float = 2.0,  # $/bbl
    pf_cost: float = 1.0,  # $/bbl
) -> pd.DataFrame:
    """Calculate economic value of each well in optimization

    Args:
        optimizer: NetworkOptimizer with optimization results
        oil_price: Revenue per barrel of oil ($/bbl)
        water_cost: Cost to handle water ($/bbl)
        pf_cost: Cost to pump power fluid ($/bbl)

    Returns:
        DataFrame with economic analysis per well
    """
    if not optimizer.optimization_results:
        return pd.DataFrame()

    results = optimizer.optimization_results
    data = []

    for r in results:
        # Daily revenue
        oil_revenue = r.predicted_oil_rate * oil_price

        # Daily costs
        water_handling_cost = r.predicted_total_water * water_cost
        pf_pumping_cost = r.allocated_power_fluid * pf_cost
        total_cost = water_handling_cost + pf_pumping_cost

        # Net value
        net_daily_value = oil_revenue - total_cost

        # Annual value
        net_annual_value = net_daily_value * 365

        data.append(
            {
                "well_name": r.well_name,
                "pump_config": f"{r.recommended_nozzle}{r.recommended_throat}",
                "oil_revenue": oil_revenue,
                "water_cost": water_handling_cost,
                "pf_cost": pf_pumping_cost,
                "total_cost": total_cost,
                "net_daily_value": net_daily_value,
                "net_annual_value": net_annual_value,
                "value_per_pf": net_daily_value / r.allocated_power_fluid if r.allocated_power_fluid > 0 else 0,
            }
        )

    df = pd.DataFrame(data)
    df = df.sort_values("net_annual_value", ascending=False)

    return df
