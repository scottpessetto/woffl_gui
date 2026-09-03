"""Jet Pump Network Solver

Add mutliple BatchPumps to a network and provide a shared resource. The shared
resource can be either lift water (power fluid) or total water.
"""

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from woffl.assembly.batchpump import BatchPump

SCALE = 100  # CP-SAT requires integers; multiply floats by this before rounding


def optimize_jet_pumps(
    well_list: list[BatchPump],
    qpf_tot: float,
    water_key: str = "lift_wat",
    allow_shutin: bool = False,
    water_price: float = 0.0,
    all_configs: bool = False,
) -> pd.DataFrame:
    """Optimize Jet Pumps via Multiple-Choice Knapsack

    Each well picks exactly one jet pump from its semi-finalists to maximize
    total oil production subject to a shared power fluid capacity constraint.
    Uses the CP-SAT solver from ortools.

    Args:
        well_list (list[BatchPump]): Wells with batch_run() and process_results() already called
        qpf_tot (float): Total surface pump capacity, BWPD
        water_key (str): Column for capacity constraint, "lift_wat" or "totl_wat"
        allow_shutin (bool): If True, solver may shut in a well when its water is better used elsewhere
        water_price (float): λ, bbl oil per bbl of ``water_key`` water. The
            objective becomes Σ (oil − λ·water); 0.0 (default) is the
            original oil-only objective, bit-identical.
        all_configs (bool): If True, every converged row (``error == "na"``,
            or all rows when there is no ``error`` column) is a candidate
            instead of the ``semi`` subset. The GUI fork passes True so this
            solver and the MILP see the same candidate set.

    Returns:
        df (DataFrame): One row per well with selected pump and rates

    Raises:
        ValueError: If any well has no semi-finalists
        RuntimeError: If the problem is infeasible (capacity too small)
    """
    # [LIBRARY change -> upstream PR to kwellis/woffl] water_price / all_configs:
    # the fork prices machine water in the objective and hands both solvers
    # one candidate set (docs/optimization_redesign_2026-09.md). Defaults keep
    # the upstream behaviour exactly.
    candidates = []
    for well in well_list:
        if all_configs:
            df_c = well.df
            if "error" in df_c.columns:
                err = df_c["error"]
                df_c = df_c[err.isna() | err.astype(str).str.strip().isin(("na", ""))]
            df_c = df_c[df_c["qoil_std"].notna()].reset_index(drop=True)
            if df_c.empty:
                raise ValueError(f"Well '{well.wellname}' has no converged config")
        else:
            df_c = well.df[well.df["semi"]].reset_index(drop=True)
            if df_c.empty:
                raise ValueError(f"Well '{well.wellname}' has no semi-finalists")
        candidates.append(df_c)

    model = cp_model.CpModel()

    # decision variables: x[i][j] = 1 if well i selects semi-finalist j
    x = []
    for i, df_semi in enumerate(candidates):
        well_vars = [model.new_bool_var(f"w{i}_p{j}") for j in range(len(df_semi))]
        x.append(well_vars)
        if allow_shutin:
            model.add_at_most_one(well_vars)
        else:
            model.add_exactly_one(well_vars)

    # capacity constraint: total water <= qpf_tot
    capacity_scaled = int(np.floor(qpf_tot * SCALE))
    water_terms = []
    for i, df_semi in enumerate(candidates):
        for j, wat in enumerate(df_semi[water_key]):
            water_terms.append(int(np.ceil(wat * SCALE)) * x[i][j])
    model.add(sum(water_terms) <= capacity_scaled)

    # objective: maximize Σ (oil − λ·water); λ = 0 is the original oil-only
    # objective. CP-SAT needs integers, so the value is scaled by SCALE and
    # floored (a negative value floors away from zero, which is conservative).
    oil_terms = []
    for i, df_semi in enumerate(candidates):
        for j, (oil, wat) in enumerate(zip(df_semi["qoil_std"], df_semi[water_key])):
            value = float(oil) - float(water_price) * float(wat)
            oil_terms.append(int(np.floor(value * SCALE)) * x[i][j])
    model.maximize(sum(oil_terms))

    # solve
    solver = cp_model.CpSolver()
    status = solver.solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        min_water = sum(df[water_key].min() for df in candidates)
        raise RuntimeError(
            f"MCKP infeasible: {qpf_tot:.0f} bwpd capacity cannot serve all wells. "
            f"Minimum required: {min_water:.0f} bwpd."
        )

    # extract solution
    results = []
    for i, (well, df_semi) in enumerate(zip(well_list, candidates)):
        selected = False
        for j in range(len(df_semi)):
            if solver.value(x[i][j]):
                row = df_semi.iloc[j]
                results.append(
                    {
                        "wellname": well.wellname,
                        "nozzle": row["nozzle"],
                        "throat": row["throat"],
                        "qoil_std": row["qoil_std"],
                        "lift_wat": row["lift_wat"],
                        "form_wat": row["form_wat"],
                        "totl_wat": row["totl_wat"],
                    }
                )
                selected = True
                break
        if not selected:
            results.append(
                {
                    "wellname": well.wellname,
                    "nozzle": "off",
                    "throat": "off",
                    "qoil_std": 0.0,
                    "lift_wat": 0.0,
                    "form_wat": 0.0,
                    "totl_wat": 0.0,
                }
            )

    return pd.DataFrame(results)
