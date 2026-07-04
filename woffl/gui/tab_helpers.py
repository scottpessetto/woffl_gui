"""Shared helpers for the single-well tab modules (Batch Run, PF Range).

Deliberately streamlit-free so the helpers are trivially unit-testable:
signature building and pump-at-date matching are pure functions of their
inputs.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from woffl.gui.params import NOZZLE_OPTIONS, THROAT_OPTIONS, SimulationParams


def physical_sweep_signature(params: SimulationParams) -> tuple:
    """Signature tuple of every SimulationParams field that affects the physics of a sweep.

    Single source of truth for the heavy-sweep session-state memo keys —
    Batch Run (``_batch_sweep_cache``) and PF Range (``_pf_range_cache``)
    both build their cache signature as ``physical_sweep_signature(params) +
    (tab-specific extras...)`` (batch nozzle/throat option lists; PF
    min/max/step).

    CLAUDE.md rule: **if you add an input that affects the sweep, ADD IT
    HERE.** A field missing from this tuple makes the cache silently serve
    results computed under the old value of that field (P1-4:
    ``model_as_water`` was missing from both tabs' local tuples, so at
    form_wc = 1.0 toggling "Model as 100% water" served the other mode's
    sweep).
    """
    return (
        params.selected_well,
        params.field_model,
        params.jpump_direction,
        params.nozzle_no,
        params.area_ratio,
        params.ken,
        params.kth,
        params.kdi,
        params.tubing_od,
        params.tubing_thickness,
        params.casing_od,
        params.casing_thickness,
        params.jpump_tvd,
        params.form_wc,
        params.form_gor,
        params.form_temp,
        params.model_as_water,
        params.oil_api,
        params.gas_sg,
        params.wat_sg,
        params.bubble_point,
        params.surf_pres,
        params.rho_pf,
        params.ppf_surf,
        params.qwf,
        params.pwf,
        params.pres,
    )


def pump_at_test_matches(
    jp_hist, well: str, test_date, nozzle, throat
) -> Optional[bool]:
    """Whether the pump installed at ``test_date`` is the given nozzle/throat.

    Resolves the installed pump via :func:`woffl.assembly.jp_history.
    get_pump_at_date` — set-to-set tenure; ``Date Pulled`` is never consulted
    (JPCOs are same-day pull+set and the tracker's Date Pulled lags by
    days-to-weeks). Mirrors the Solver's ``pump_differs`` guard semantics
    (``jetpump_solver._pump_at_test_date`` + ``_is_valid_pump_code``):

    - True/False when the pump at the test date resolves to a valid National
      pump code and can be compared to ``(nozzle, throat)``.
    - None (fail open — the caller must treat this as "can't tell, don't
      block") when history/date is unavailable, no install predates the
      test, or the resolved record isn't a real pump (e.g. S-17's
      throat-letter-in-the-nozzle-column rows, which ``get_pump_at_date``
      returns with ``nozzle_no=None``).
    """
    if jp_hist is None or not nozzle or not throat:
        return None
    if test_date is None or pd.isna(test_date):
        return None

    from woffl.assembly.jp_history import get_pump_at_date

    pump = get_pump_at_date(jp_hist, well, test_date)
    if pump is None:
        return None

    hist_nozzle = pump.get("nozzle_no")
    hist_throat = pump.get("throat_ratio")
    # Same guard as the Solver's _is_valid_pump_code: a corrupt JP-history
    # record must not be treated as a real test pump.
    if str(hist_nozzle) not in NOZZLE_OPTIONS or str(hist_throat) not in THROAT_OPTIONS:
        return None

    return (str(nozzle), str(throat).strip()) == (hist_nozzle, hist_throat)
