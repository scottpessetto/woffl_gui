"""E-Pad booster candidate capability at a required differential pressure.

Thin read-only wrapper over ``woffl.gui.e_pad_booster.capability_report`` -
that module owns the physics and the vendor data files; this one only pins the
cache and hands the payload up.

Pure static physics: no Databricks, no saved fits, no run state. The answer
depends only on the request scalars and the files on disk, so it is cached on
the whole argument tuple and can be served before any optimization run.
"""

from __future__ import annotations

from typing import Any, Optional

from server import config
from server.cache import ttl_cache


@ttl_cache(config.TTL_PUMP_CURVE, maxsize=64)
def capability(
    dp_psid: float,
    suction_psi: float,
    sg: float,
    condition: float,
    hz_max: float,
    amps_per_bhp: float,
    amp_limit_a: Optional[float],
) -> dict[str, Any]:
    """Both E-Pad booster candidates at one required dP.

    Args:
        dp_psid (float): required differential pressure across the booster.
        suction_psi (float): booster suction (psig).
        sg (float): pumped-fluid specific gravity.
        condition (float): head-only wear derate (1.00 = as-new).
        hz_max (float): VFD speed cap (Hz).
        amps_per_bhp (float): amps per shaft BHP.
        amp_limit_a (float | None): motor amp cap; None enforces nothing.

    Returns:
        dict: ``capability_report`` payload; see schemas.EPadBoosterResponse.
    """
    from woffl.gui import e_pad_booster

    return e_pad_booster.capability_report(
        dp_psid=dp_psid,
        suction_psi=suction_psi,
        sg=sg,
        condition=condition,
        hz_max=hz_max,
        amps_per_bhp=amps_per_bhp,
        amp_limit=amp_limit_a,
    )
