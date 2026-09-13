"""Keep installed hardware calibration separate from clean replacements."""

from woffl.geometry.jetpump import JetPump

# [LIBRARY change -> upstream PR to kwellis/woffl]
# These are model reference coefficients, not evidence that a new pump is perfect.
CLEAN_PUMP = {"ken": 0.03, "kth": 0.3, "kdi": 0.4, "nozzle_area_factor": 1.0}


def scoped_pumps(nozzles, throats, installed=None, coefficients=None):
    """Return an installed candidate plus clean catalog candidates.

    The two candidates may have identical catalog sizes but different performance.
    Unknown installation identity never distributes fitted coefficients to the catalog.
    The installed candidate is always included when its identity is known;
    the requested grid limits clean replacement choices only.
    """
    result = []
    coefs = {**CLEAN_PUMP, **(coefficients or {})}
    # [LIBRARY change -> upstream PR to kwellis/woffl] Patch 46: restricting
    # replacement sizes must not remove the no-change operating choice.
    if installed and all(installed):
        jp = JetPump(installed[0], installed[1], ken=coefs["ken"], kth=coefs["kth"], kdi=coefs["kdi"])
        jp.dnz *= coefs["nozzle_area_factor"] ** 0.5
        jp.pump_state = "installed"
        result.append(jp)
    for nozzle in nozzles:
        for throat in throats:
            jp = JetPump(nozzle, throat, ken=CLEAN_PUMP["ken"], kth=CLEAN_PUMP["kth"], kdi=CLEAN_PUMP["kdi"])
            jp.pump_state = "replacement"
            result.append(jp)
    return result
