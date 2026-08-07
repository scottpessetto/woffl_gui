"""Booster-pump curves for the S / I / M pad plants.

Thin read-only wrapper over ``PadPlant.curve_report`` - the plant models
(``woffl/gui/pad_plant_base.py``) own the physics and the vendor data files;
this module only resolves the pad to its plant and hands the payload up.

Pure static physics: no Databricks, no saved fits, no run state. The curve set
for a pad changes only when its data files do, so it is cached hard and can be
served before an optimization run has been started.
"""

from __future__ import annotations

from typing import Any, Optional

from server import config
from server.cache import ttl_cache


@ttl_cache(config.TTL_PUMP_CURVE, maxsize=16)
def pump_curve(pad: str, n_pumps: Optional[int]) -> dict[str, Any]:
    """Industry-format curve set for one pad's booster plant.

    The payload is exactly ``PadPlant.curve_report``'s contract (nameplate,
    station curve family with BEP/POR/AOR and capability frontier, and the
    per-machine head / BHP / efficiency curves) - plain JSON-safe nested
    dicts and lists.

    Args:
        pad (str): pad letter, "S", "I" or "M".
        n_pumps (int | None): pumps online for the station family. None means
            the plant's own default.

    Returns:
        dict: curve_report payload; see schemas.PumpCurveResponse.

    Raises:
        ValueError: unknown pad (propagated from the plant lookup).
    """
    from server.services.optimizer_runs import _pad_plant

    plant = _pad_plant(pad)
    return plant.curve_report(n_pumps)
