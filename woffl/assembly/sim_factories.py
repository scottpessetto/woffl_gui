"""Simulation object factories - the Streamlit-free canonical copies.

FORK-ONLY module (like ``parallelism.py`` and the ``*_client.py`` glue): not
upstream physics, just the construction helpers every caller needs.

These bodies lived in ``woffl/gui/utils.py``, which does ``import streamlit``
at module scope. That made them unreachable without Streamlit, which was a
problem in two directions:

* ``network_optimizer._create_well_objects`` - LIBRARY code - imported
  ``woffl.gui.utils`` for ``create_pvt_components``, so the library depended
  on the GUI.
* The FastAPI server could not use them at all, so ``server/services/
  factories.py`` grew a parallel "faithful copy ... minus the Streamlit
  dependencies" of the same physics. Two implementations of one set of
  factories is a divergence waiting to happen; this module is the one both
  now share.

Bodies are lifted VERBATIM from ``woffl/gui/utils.py`` so every existing
caller keeps getting bit-identical objects. The single behavioural difference
is in :func:`run_jetpump_solver`, which no longer paints an ``st.error`` box
on failure - it just returns None, exactly as the ``quiet=True`` path did.
"""

from __future__ import annotations

from typing import Any, Optional

from woffl.assembly.solopump import jetpump_solver
from woffl.flow.inflow import InFlow
from woffl.geometry import JetPump, Pipe, PipeInPipe
from woffl.pvt import BlackOil, FormGas, FormWater, ResMix


def create_jetpump(nozzle_no, area_ratio, ken, kth, kdi) -> JetPump:
    """Create a JetPump object with the given parameters."""
    return JetPump(
        nozzle_no=nozzle_no, area_ratio=area_ratio, ken=ken, kth=kth, kdi=kdi
    )


def create_pvt_components(
    field_model=None,
    oil_api=None,
    gas_sg=None,
    wat_sg=None,
    bubble_point=None,
) -> tuple[BlackOil, FormWater, FormGas]:
    """Create PVT components (oil, water, gas) for the given field model.

    This is the single source of truth for Schrader/Kuparuk PVT model
    selection. Used by create_reservoir_mix(), network_optimizer.
    _create_well_objects(), and the server's factories.

    Args:
        field_model: "Schrader" or "Kuparuk" (case-insensitive). Provides
            defaults for any oil_api/gas_sg/wat_sg/bubble_point not
            explicitly supplied.
        oil_api, gas_sg, wat_sg, bubble_point: Optional per-well overrides
            (e.g., from Databricks vw_prop_resvr). When provided, these
            replace the field_model preset values.

    Returns:
        tuple: (BlackOil, FormWater, FormGas) instances
    """
    if field_model is None:
        field_model = "schrader"
    field_model = field_model.lower()

    if field_model == "kuparuk":
        oil_default = BlackOil.kuparuk()
        wat_default = FormWater.kuparuk()
        gas_default = FormGas.kuparuk()
    else:
        oil_default = BlackOil.schrader()
        wat_default = FormWater.schrader()
        gas_default = FormGas.schrader()

    final_api = oil_api if oil_api is not None else oil_default.oil_api
    final_pbp = bubble_point if bubble_point is not None else oil_default.pbp
    final_oil_sg = gas_sg if gas_sg is not None else oil_default.gas_sg
    final_gas_sg = gas_sg if gas_sg is not None else gas_default.gas_sg
    final_wat_sg = wat_sg if wat_sg is not None else wat_default.wat_sg

    oil = BlackOil(oil_api=final_api, bubblepoint=final_pbp, gas_sg=final_oil_sg)
    water = FormWater(wat_sg=final_wat_sg)
    gas = FormGas(gas_sg=final_gas_sg)
    return oil, water, gas


def create_reservoir_mix(
    wc,
    gor,
    temp,
    field_model=None,
    oil_api=None,
    gas_sg=None,
    wat_sg=None,
    bubble_point=None,
    model_as_water=False,
) -> ResMix:
    """Create a ResMix object with the given parameters.

    model_as_water (bool): opt-in water-pump mode for a 100%-water (no-oil)
        well; passed through to ResMix. Default False = oil-anchored behavior.
    """
    oil, water, gas = create_pvt_components(
        field_model=field_model,
        oil_api=oil_api,
        gas_sg=gas_sg,
        wat_sg=wat_sg,
        bubble_point=bubble_point,
    )
    return ResMix(
        wc=wc, fgor=gor, oil=oil, wat=water, gas=gas, model_as_water=model_as_water
    )


def create_pipes(
    tubing_od=4.5, tubing_thickness=0.5, casing_od=6.875, casing_thickness=0.5
) -> tuple[Pipe, Pipe, PipeInPipe]:
    """Create tubing, casing, and wellbore (PipeInPipe) objects."""
    tube = Pipe(out_dia=tubing_od, thick=tubing_thickness)
    case = Pipe(out_dia=casing_od, thick=casing_thickness)
    wellbore = PipeInPipe(inn_pipe=tube, out_pipe=case)
    return tube, case, wellbore


def create_inflow(qwf, pwf, pres) -> InFlow:
    """Create an InFlow object from a SINGLE-PHASE rate at ``pwf``.

    ``qwf`` keeps the ``InFlow`` library contract: ONE phase - the oil rate
    normally, the water rate in dewatering mode - NOT the total liquid rate
    that ``SimulationParams.qwf`` holds (RATE CONVENTION, gui/params.py).
    Callers with a ``SimulationParams`` MUST pass ``params.inflow_rate``,
    which already picks the right phase for the mode.
    """
    return InFlow(qwf=qwf, pwf=pwf, pres=pres)


def run_jetpump_solver(
    surf_pres,
    form_temp,
    rho_pf,
    ppf_surf,
    jetpump,
    wellbore,
    well_profile,
    inflow,
    res_mix,
    field_model=None,
    jpump_direction="reverse",
) -> Optional[tuple[Any, ...]]:
    """Run the jetpump solver and return the results.

    Finds a solution for the jetpump system that factors in the wellhead
    pressure and reservoir conditions.

    Returns:
        tuple or None: (psu, sonic_status, qoil_std, fwat_bwpd, qnz_bwpd,
        mach_te) if successful, None if the solver fails.

    Raises:
        ThroatEntryNoSolution: re-raised (it subclasses both ValueError and
            IndexError) so callers that auto-recover from a too-low GOR still
            fire. Other call sites guard it explicitly. DO NOT "fix" this
            re-raise - the recovery depends on it.
    """
    from woffl.flow.errors import ThroatEntryNoSolution

    # Create power fluid properties from field model water
    _, prop_pf, _ = create_pvt_components(field_model)
    prop_pf = prop_pf.condition(0, 60)

    try:
        return jetpump_solver(
            pwh=surf_pres,
            tsu=form_temp,
            ppf_surf=ppf_surf,
            jpump=jetpump,
            wellbore=wellbore,
            wellprof=well_profile,
            ipr_su=inflow,
            prop_su=res_mix,
            prop_pf=prop_pf,
            jpump_direction=jpump_direction,
        )
    except ThroatEntryNoSolution:
        raise
    except ValueError:
        # The well cannot lift at max suction pressure. The GUI used to paint
        # an st.error box here; callers now surface their own message.
        return None
