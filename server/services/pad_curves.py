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


@ttl_cache(config.TTL_PUMP_CURVE, maxsize=64)
def pump_curve(
    pad: str,
    n_pumps: Optional[int],
    build: Optional[str] = None,
    suction_psi: Optional[float] = None,
    hz_max: Optional[float] = None,
    max_header_psi: Optional[float] = None,
) -> dict[str, Any]:
    """Industry-format curve set for one pad's booster plant.

    The payload is exactly ``PadPlant.curve_report``'s contract (nameplate,
    station curve family with BEP/POR/AOR and capability frontier, and the
    per-machine head / BHP / efficiency curves) - plain JSON-safe nested
    dicts and lists.

    Args:
        pad (str): pad letter, "S", "I", "M" or "E".
        n_pumps (int | None): pumps online for the station family. None means
            the plant's own default.
        build (str | None): E-Pad only - which booster build is in the ground.
        suction_psi (float | None): E-Pad only - booster suction (psig).
        hz_max (float | None): E-Pad only - VFD speed cap (Hz).
        max_header_psi (float | None): E-Pad only - operational header cap.

    Returns:
        dict: curve_report payload; see schemas.PumpCurveResponse.

    Raises:
        ValueError: unknown pad, or an unknown E-Pad build.
    """
    from server.services.optimizer_runs import _pad_plant

    if pad == "E":
        # Configured per call: none of these four is a measured E-Pad tag, so
        # the sheet must show the booster the caller is actually assuming.
        from woffl.gui.e_pad_plant import INSTALLED_BUILD, EPadPlant

        plant = EPadPlant(
            build or INSTALLED_BUILD,
            suction_psi=suction_psi,
            hz_max=60.0 if hz_max is None else hz_max,
            max_header_psi=max_header_psi,
        )
    else:
        plant = _pad_plant(pad)
    report = plant.curve_report(n_pumps)
    # Ride the plant's selectable online-pump counts along (outside
    # curve_report - its key set is contract-pinned by the plant tests) so
    # the client can offer a "pumps online" control; [] = fixed train.
    report["n_pump_options"] = [int(n) for n in plant.n_pump_options]
    return report
