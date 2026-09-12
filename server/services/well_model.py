"""Stable dependency identity for fixed-oil-IPR calibration and replay.

No warehouse access: the caller supplies an already hydrated WellConfig.
Operating test WC/GOR/PF pressure and fitted pump coefficients are intentionally
outside this identity. The normalized oil curve, PVT and bore assumptions are
shared by every observation; installation and training controls bind separately.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path

from woffl.flow.hydraulics import physics_model
from woffl.flow.inflow import InFlow

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = "fixed-oil-ipr-v1"
DEPENDENCIES = (
    "well_name", "res_pres", "form_temp", "jpump_tvd", "jpump_md",
    "tubing_od", "tubing_thickness", "casing_od", "casing_thickness",
    "field_model", "surf_pres", "oil_api", "gas_sg", "wat_sg", "rho_pf",
    "bubble_point", "jpump_direction",
)


def _number(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise ValueError("Non-finite well-model dependency")
        # Equivalent liquid representations have small binary roundoff. This
        # changes only hash normalization, never simulation or saved precision.
        return float(format(value, ".14g"))
    return value


def describe(config):
    """Return a stable fingerprint and reviewable normalized model inputs."""
    curve = InFlow(config.qwf * (1-config.form_wc), config.pwf, config.res_pres)
    inputs = {key: _number(getattr(config, key, None)) for key in DEPENDENCIES}
    inputs.update(contract=CONTRACT,
                  oil_qmax=_number(curve.vogel_qmax(curve.qwf, curve.pwf, curve.pres)),
                  physics_model=physics_model(config.hydraulics_model))
    survey = ROOT / "woffl" / "jp_data" / "well_surveys" / f"{config.well_name} Deviation Survey.csv"
    if survey.parent.resolve() != (ROOT / "woffl" / "jp_data" / "well_surveys").resolve():
        raise ValueError("Invalid well survey identity")
    try:
        inputs["survey_sha256"] = sha256(survey.read_bytes()).hexdigest()
    except FileNotFoundError:
        inputs["survey_sha256"] = None  # the solver uses its field profile
    payload = json.dumps(inputs, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {"fingerprint": sha256(payload.encode()).hexdigest()[:32], "inputs": inputs}


def from_context(context, hydraulics_model=None):
    """Use the optimizer's canonical conversion, with measured MD included."""
    from server.services.optimizer_runs import _config_from_seeds
    if context.get("geometry_issue"):
        raise ValueError(context["geometry_issue"])
    seeds = dict(context["seeds"])
    if context.get("jpump_md") is not None:
        seeds["jpump_md"] = context["jpump_md"]
    if hydraulics_model is not None:
        seeds["hydraulics_model"] = hydraulics_model
    return describe(_config_from_seeds(context["well"], "", seeds))
