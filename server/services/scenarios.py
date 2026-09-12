"""Canonical sensitivity cases; no fitting, persistence or physics changes."""

from __future__ import annotations

from typing import Any, Literal

from server.schemas import SimParams
from woffl.assembly.pump_candidates import CLEAN_PUMP

WcBasis = Literal["fixed_oil_ipr", "anchor_measurement"]


def scenario_params(
    base: SimParams, update: dict[str, Any], wc_basis: WcBasis = "fixed_oil_ipr",
) -> SimParams:
    """Build an independent case with explicit composition and hardware scope.

    A WC change normally preserves the oil IPR. If qwf is deliberately varied
    too, that value is a liquid anchor at the BASE WC, then converted to the
    scenario WC. Anchor-measurement mode instead retains the liquid anchor and
    lets uncertain WC alter the inferred oil IPR. GOR stays independently fixed
    unless explicitly varied; this does not assume constant operating gas rate.
    """
    values = {**base.model_dump(), **update}
    if wc_basis not in ("fixed_oil_ipr", "anchor_measurement"):
        raise ValueError("Unknown WC sensitivity basis")
    if "form_wc" in update and not base.model_as_water and wc_basis == "fixed_oil_ipr":
        wc = float(values["form_wc"])
        if not 0 <= wc < 1 or not 0 <= base.form_wc < 1:
            raise ValueError("Fixed oil IPR requires WC below 100%")
        if wc != base.form_wc:
            values["qwf"] = float(values["qwf"]) * (1 - base.form_wc) / (1 - wc)
    changed_hardware = any(values[k] != getattr(base, k) for k in ("nozzle_no", "area_ratio"))
    if changed_hardware or values["pump_state"] == "replacement":
        values.update(CLEAN_PUMP, pump_state="replacement")
    elif values["hydraulics_model"] != base.hydraulics_model:
        values.update(CLEAN_PUMP)
    if float(values["pwf"]) >= float(values["pres"]):
        raise ValueError("IPR anchor BHP must be below reservoir pressure")
    return SimParams.model_validate(values)


def scenario_patch(base: SimParams, candidate: SimParams) -> dict[str, Any]:
    """Include automatic oil-anchor/hardware changes in a reproducible Apply."""
    before = base.model_dump()
    return {key: value for key, value in candidate.model_dump().items() if value != before[key]}
