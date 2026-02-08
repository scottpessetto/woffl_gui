"""Optimization Algorithms for Multi-Well Jet Pump Sizing

This module contains algorithms for optimizing jet pump sizing across multiple
wells subject to power fluid constraints.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from woffl.assembly.network_optimizer import NetworkOptimizer, OptimizationResult


def greedy_optimization(optimizer: "NetworkOptimizer") -> list["OptimizationResult"]:
    """Greedy allocation algorithm

    Iteratively allocates to the well/pump combination with the highest
    marginal oil production rate until power fluid is exhausted.

    Args:
        optimizer: NetworkOptimizer instance with batch results already run

    Returns:
        List of OptimizationResult objects

    Raises:
        ValueError: If batch results haven't been run
    """
    from woffl.assembly.network_optimizer import OptimizationResult

    if not optimizer.batch_results:
        raise ValueError("Must run batch simulations before optimization")

    # Initialize allocation (no pump assigned)
    allocation = {well.well_name: None for well in optimizer.wells}
    remaining_pf = optimizer.power_fluid.total_rate

    # Track what we've tried
    tried_combinations = set()

    # Iteratively find the best pump to add
    while True:
        best_well = None
        best_nozzle = None
        best_throat = None
        best_marginal_oil = -np.inf
        best_pf_required = 0

        # Evaluate all possible additions/changes
        for well in optimizer.wells:
            well_name = well.well_name
            batch_pump = optimizer.batch_results[well_name]

            # Get all successful pump configurations
            successful_configs = batch_pump.df[~batch_pump.df["qoil_std"].isna()]

            for _, row in successful_configs.iterrows():
                nozzle = row["nozzle"]
                throat = row["throat"]

                combo_key = (well_name, nozzle, throat)

                # Skip if already tried
                if combo_key in tried_combinations:
                    continue

                # Get performance
                perf = optimizer.get_pump_performance(well_name, nozzle, throat)
                if perf is None:
                    continue

                pf_required = perf["lift_water"]
                oil_rate = perf["oil_rate"]

                # Calculate marginal oil considering current allocation
                current_oil = 0
                current_pf = 0
                if allocation[well_name] is not None:
                    current_nozzle, current_throat = allocation[well_name]
                    current_perf = optimizer.get_pump_performance(well_name, current_nozzle, current_throat)
                    if current_perf:
                        current_oil = current_perf["oil_rate"]
                        current_pf = current_perf["lift_water"]

                # Net change in oil and power fluid
                delta_oil = oil_rate - current_oil
                delta_pf = pf_required - current_pf

                # Check if we have enough power fluid for this change
                if delta_pf > remaining_pf + 0.1:  # Small tolerance for rounding
                    continue

                # Calculate marginal oil rate
                if delta_pf > 0:
                    marginal_oil = delta_oil / delta_pf
                else:
                    # If using same or less PF but getting more oil, that's infinite marginal value
                    marginal_oil = np.inf if delta_oil > 0 else -np.inf

                # Update best if this is better
                if marginal_oil > best_marginal_oil:
                    best_marginal_oil = marginal_oil
                    best_well = well_name
                    best_nozzle = nozzle
                    best_throat = throat
                    best_pf_required = pf_required

        # If no improvement found, we're done
        if best_well is None:
            break

        # Apply the best allocation
        old_allocation = allocation[best_well]
        allocation[best_well] = (best_nozzle, best_throat)
        tried_combinations.add((best_well, best_nozzle, best_throat))

        # Update remaining power fluid
        if old_allocation is not None:
            old_nozzle, old_throat = old_allocation
            old_perf = optimizer.get_pump_performance(best_well, old_nozzle, old_throat)
            if old_perf:
                remaining_pf += old_perf["lift_water"]

        remaining_pf -= best_pf_required

    # Convert allocation to OptimizationResult objects
    results = []
    for well in optimizer.wells:
        well_name = well.well_name

        if allocation[well_name] is None:
            # No pump allocated - use smallest viable configuration or skip
            continue

        nozzle, throat = allocation[well_name]
        perf = optimizer.get_pump_performance(well_name, nozzle, throat)

        if perf is None:
            continue

        result = OptimizationResult(
            well_name=well_name,
            recommended_nozzle=nozzle,
            recommended_throat=throat,
            allocated_power_fluid=perf["lift_water"],
            predicted_oil_rate=perf["oil_rate"],
            predicted_formation_water=perf["formation_water"],
            predicted_lift_water=perf["lift_water"],
            suction_pressure=perf["suction_pressure"],
            marginal_oil_rate=perf["marginal_oil_lift_water"],
            sonic_status=perf["sonic_status"],
            mach_te=perf["mach_te"],
        )
        results.append(result)

    optimizer.optimization_results = results
    return results


def simple_proportional_allocation(optimizer: "NetworkOptimizer") -> list["OptimizationResult"]:
    """Simple proportional allocation

    Allocates power fluid proportionally based on well productivity index.
    This is a simpler alternative to greedy for initial testing.

    Args:
        optimizer: NetworkOptimizer instance with batch results already run

    Returns:
        List of OptimizationResult objects
    """
    from woffl.assembly.network_optimizer import OptimizationResult

    if not optimizer.batch_results:
        raise ValueError("Must run batch simulations before optimization")

    # For each well, find the best pump configuration ignoring constraints
    well_best_configs = {}
    well_productivities = {}

    for well in optimizer.wells:
        well_name = well.well_name
        batch_pump = optimizer.batch_results[well_name]

        # Find configuration with highest oil rate
        successful = batch_pump.df[~batch_pump.df["qoil_std"].isna()]
        if successful.empty:
            continue

        best_row = successful.loc[successful["qoil_std"].idxmax()]
        well_best_configs[well_name] = (best_row["nozzle"], best_row["throat"])

        # Use oil rate as productivity metric
        well_productivities[well_name] = best_row["qoil_std"]

    # If no valid configurations, return empty
    if not well_best_configs:
        return []

    # Allocate proportionally
    total_productivity = sum(well_productivities.values())

    results = []
    for well_name, (nozzle, throat) in well_best_configs.items():
        # Calculate this well's share of power fluid
        share = well_productivities[well_name] / total_productivity
        allocated_pf = optimizer.power_fluid.total_rate * share

        perf = optimizer.get_pump_performance(well_name, nozzle, throat)
        if perf is None:
            continue

        # Check if this configuration uses reasonable power fluid
        # If not, find a smaller pump
        if perf["lift_water"] > allocated_pf * 1.5:
            # Find a smaller pump that fits the allocation better
            batch_pump = optimizer.batch_results[well_name]
            successful = batch_pump.df[~batch_pump.df["qoil_std"].isna()]
            successful = successful[successful["lift_wat"] <= allocated_pf * 1.2]

            if not successful.empty:
                best_row = successful.loc[successful["qoil_std"].idxmax()]
                nozzle = best_row["nozzle"]
                throat = best_row["throat"]
                perf = optimizer.get_pump_performance(well_name, nozzle, throat)

        if perf is not None:
            result = OptimizationResult(
                well_name=well_name,
                recommended_nozzle=nozzle,
                recommended_throat=throat,
                allocated_power_fluid=perf["lift_water"],
                predicted_oil_rate=perf["oil_rate"],
                predicted_formation_water=perf["formation_water"],
                predicted_lift_water=perf["lift_water"],
                suction_pressure=perf["suction_pressure"],
                marginal_oil_rate=perf["marginal_oil_lift_water"],
                sonic_status=perf["sonic_status"],
                mach_te=perf["mach_te"],
            )
            results.append(result)

    optimizer.optimization_results = results
    return results


# Note: Linear programming and genetic algorithm implementations would go here
# For now, greedy is the most practical and performs well for this type of problem


def optimize(optimizer: "NetworkOptimizer", method: str = "greedy") -> list["OptimizationResult"]:
    """Main optimization dispatcher

    Args:
        optimizer: NetworkOptimizer instance
        method: Optimization method ('greedy', 'proportional')

    Returns:
        List of OptimizationResult objects

    Raises:
        ValueError: If method is not recognized
    """
    if method == "greedy":
        return greedy_optimization(optimizer)
    elif method == "proportional":
        return simple_proportional_allocation(optimizer)
    else:
        raise ValueError(f"Unknown optimization method: {method}. Use 'greedy' or 'proportional'")
