"""Optimization Algorithms for Multi-Well Jet Pump Sizing

This module contains algorithms for optimizing jet pump sizing across multiple
wells subject to power fluid constraints.

Methods:
    milp: Mixed-integer linear programming via scipy (exact solver)
    mckp: Multi-choice knapsack via OR-Tools CP-SAT (Kaelin's upstream solver)
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from woffl.assembly.network_optimizer import NetworkOptimizer, OptimizationResult


def milp_optimization(optimizer: "NetworkOptimizer") -> list["OptimizationResult"]:
    """Optimal allocation via mixed-integer linear programming.

    Formulates pump selection as a multiple-choice knapsack problem and
    solves it exactly using MILP.  Each well may be assigned at most one
    pump configuration (nozzle/throat).  The solver maximizes total oil
    production subject to the power-fluid budget constraint.

    Args:
        optimizer: NetworkOptimizer instance with batch results already run

    Returns:
        List of OptimizationResult objects

    Raises:
        ValueError: If batch results haven't been run
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csc_array
    from scipy.sparse import vstack as sp_vstack

    from woffl.assembly.network_optimizer import OptimizationResult

    if not optimizer.batch_results:
        raise ValueError("Must run batch simulations before optimization")

    # ── Build decision-variable list ──────────────────────────────────────
    # Each variable x[k] ∈ {0,1} represents selecting config k for a well.
    configs: list[dict] = []
    well_names: list[str] = []

    for well in optimizer.wells:
        wn = well.well_name
        if wn not in optimizer.batch_results:
            continue
        batch_pump = optimizer.batch_results[wn]
        successful = batch_pump.df[~batch_pump.df["qoil_std"].isna()]

        for _, row in successful.iterrows():
            perf = optimizer.get_pump_performance(wn, row["nozzle"], row["throat"])
            if perf is None:
                continue
            configs.append(
                {
                    "well_name": wn,
                    "nozzle": row["nozzle"],
                    "throat": row["throat"],
                    "perf": perf,
                }
            )

        if wn not in well_names:
            well_names.append(wn)

    if not configs:
        optimizer.optimization_results = []
        return []

    n = len(configs)
    n_wells = len(well_names)
    well_idx = {wn: i for i, wn in enumerate(well_names)}

    # ── Objective: maximize Σ oil  →  minimize -Σ oil ─────────────────────
    c = np.array([-cfg["perf"]["oil_rate"] for cfg in configs])

    # ── Constraint 1: at most one config per well ─────────────────────────
    row_ids, col_ids, vals = [], [], []
    for k, cfg in enumerate(configs):
        row_ids.append(well_idx[cfg["well_name"]])
        col_ids.append(k)
        vals.append(1.0)
    A_well = csc_array((vals, (row_ids, col_ids)), shape=(n_wells, n))

    # ── Constraint 2: Σ pf[k]*x[k] ≤ budget ─────────────────────────────
    pf_vals = np.array([[cfg["perf"]["lift_water"] for cfg in configs]])
    A_pf = csc_array(pf_vals)

    A = sp_vstack([A_well, A_pf], format="csc")
    b_upper = np.concatenate([np.ones(n_wells), [optimizer.power_fluid.total_rate]])
    b_lower = np.full(n_wells + 1, -np.inf)

    constraints = LinearConstraint(A, lb=b_lower, ub=b_upper)

    # ── Bounds & integrality ──────────────────────────────────────────────
    bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))
    integrality = np.ones(n)  # all binary

    result = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)

    if not result.success:
        optimizer.optimization_results = []
        return []

    # ── Extract selected configurations ───────────────────────────────────
    selected = np.where(result.x > 0.5)[0]

    results = []
    for k in selected:
        cfg = configs[k]
        perf = cfg["perf"]
        results.append(
            OptimizationResult(
                well_name=cfg["well_name"],
                recommended_nozzle=cfg["nozzle"],
                recommended_throat=cfg["throat"],
                allocated_power_fluid=perf["lift_water"],
                predicted_oil_rate=perf["oil_rate"],
                predicted_formation_water=perf["formation_water"],
                predicted_lift_water=perf["lift_water"],
                suction_pressure=perf["suction_pressure"],
                marginal_oil_rate=perf["marginal_oil_lift_water"],
                sonic_status=perf["sonic_status"],
                mach_te=perf["mach_te"],
            )
        )

    optimizer.optimization_results = results
    return results


def mckp_optimization(optimizer: "NetworkOptimizer") -> list["OptimizationResult"]:
    """Optimal allocation via Multi-Choice Knapsack (OR-Tools CP-SAT).

    Bridges the GUI's NetworkOptimizer interface to Kaelin's upstream
    WellNetwork.optimize() which uses the CP-SAT constraint solver.

    Each well picks exactly one jet pump from its semi-finalists to maximize
    total oil production subject to the power-fluid budget constraint.

    Args:
        optimizer: NetworkOptimizer instance with batch results already run

    Returns:
        List of OptimizationResult objects

    Raises:
        ValueError: If batch results haven't been run
    """
    from woffl.assembly.network import optimize_jet_pumps
    from woffl.assembly.network_optimizer import OptimizationResult

    if not optimizer.batch_results:
        raise ValueError("Must run batch simulations before optimization")

    # Collect BatchPump objects in well order
    batch_pumps = []
    for well in optimizer.wells:
        wn = well.well_name
        if wn not in optimizer.batch_results:
            continue
        batch_pumps.append(optimizer.batch_results[wn])

    if not batch_pumps:
        optimizer.optimization_results = []
        return []

    # Call upstream MCKP solver
    mckp_df = optimize_jet_pumps(
        well_list=batch_pumps,
        qpf_tot=optimizer.power_fluid.total_rate,
        water_key="lift_wat",
        allow_shutin=False,
    )

    # Convert MCKP result DataFrame to OptimizationResult objects
    results = []
    for _, row in mckp_df.iterrows():
        well_name = row["wellname"]
        nozzle = str(row["nozzle"])
        throat = str(row["throat"])

        # Look up full performance from batch results
        perf = optimizer.get_pump_performance(well_name, nozzle, throat)
        if perf is None:
            continue

        results.append(
            OptimizationResult(
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
        )

    optimizer.optimization_results = results
    return results


def optimize(
    optimizer: "NetworkOptimizer", method: str = "milp"
) -> list["OptimizationResult"]:
    """Main optimization dispatcher

    Args:
        optimizer: NetworkOptimizer instance
        method: Optimization method ('milp' or 'mckp')

    Returns:
        List of OptimizationResult objects

    Raises:
        ValueError: If method is not recognized
    """
    if method == "milp":
        return milp_optimization(optimizer)
    elif method == "mckp":
        return mckp_optimization(optimizer)
    else:
        raise ValueError(f"Unknown optimization method: {method}. Use 'milp' or 'mckp'")
