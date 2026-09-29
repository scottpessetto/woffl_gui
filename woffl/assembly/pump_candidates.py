"""Keep installed hardware calibration separate from clean replacements."""

from woffl.geometry.jetpump import JetPump

# [LIBRARY change -> upstream PR to kwellis/woffl]
# These are model reference coefficients, not evidence that a new pump is perfect.
CLEAN_PUMP = {"ken": 0.03, "kth": 0.3, "kdi": 0.4, "nozzle_area_factor": 1.0}


def scoped_pumps(nozzles, throats, installed=None, coefficients=None, rejected=None):
    """Return an installed candidate plus clean catalog candidates.

    The two candidates may have identical catalog sizes but different performance.
    Unknown installation identity never distributes fitted coefficients to the catalog.
    The installed candidate is always included when its identity is known;
    the requested grid limits clean replacement choices only.

    Args:
        nozzles (list): Replacement nozzle sizes.
        throats (list): Replacement throat ratios.
        installed (tuple | None): Installed ``(nozzle, throat)`` identity.
        coefficients (dict | None): The installed pump's own fitted losses.
        rejected (list | None): Receives ``(nozzle, throat, reason)`` when a
            known installed identity is not a catalog pump. That candidate is
            skipped so the replacement options (and other wells) still run.

    Returns:
        list: JetPumps with ``pump_state`` "installed" or "replacement".
    """
    result = []
    coefs = {**CLEAN_PUMP, **(coefficients or {})}
    # [LIBRARY change -> upstream PR to kwellis/woffl] Patch 46: restricting
    # replacement sizes must not remove the no-change operating choice.
    if installed and all(installed):
        # [LIBRARY change -> upstream PR to kwellis/woffl] Patch 50: a malformed
        # tracker identity ("12.0", "20E", "12F") must not abort a pad batch.
        try:
            jp = JetPump(installed[0], installed[1], ken=coefs["ken"], kth=coefs["kth"], kdi=coefs["kdi"])
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            if rejected is not None:
                rejected.append((str(installed[0]), str(installed[1]),
                                 f"installed pump identity not in catalog: {exc!r}"))
        else:
            jp.dnz *= coefs["nozzle_area_factor"] ** 0.5
            jp.pump_state = "installed"
            result.append(jp)
    for nozzle in nozzles:
        for throat in throats:
            jp = JetPump(nozzle, throat, ken=CLEAN_PUMP["ken"], kth=CLEAN_PUMP["kth"], kdi=CLEAN_PUMP["kdi"])
            jp.pump_state = "replacement"
            result.append(jp)
    return result


# [LIBRARY change -> upstream PR to kwellis/woffl] Patch 51: candidate identity
# for explicit subset solves and the installed/clean twin shortcut.
def pump_key(nozzle, throat, state=None):
    """Batch-row identity ``(nozzle, throat, pump_state)`` of one candidate.

    Nozzle and throat are normalized the way ``JetPump`` stores them
    (string nozzle, upper-case throat) so a key built from a caller's choice
    matches the row the batch writes. ``state`` is None for legacy
    (unscoped) candidates, which carry no ``pump_state``.

    Args:
        nozzle (str | int): Nozzle number.
        throat (str): Throat (area-ratio) letter.
        state (str | None): "installed", "replacement" or None.

    Returns:
        tuple: ``(str nozzle, str throat, state)``.
    """
    return (str(nozzle), str(throat).upper(), state)


def jetpump_key(jp):
    """``pump_key`` of a built ``JetPump`` (its optional ``pump_state`` included)."""
    return pump_key(jp.noz_no, jp.rat_ar, getattr(jp, "pump_state", None))


def identical_twins(jetpumps):
    """Replacement candidates that are exactly the installed candidate.

    An installed pump whose losses and nozzle area are the clean reference
    values is, input for input, the same ``JetPump`` as the clean replacement
    of its size. Both rows stay in the batch (they are different hardware
    actions), but the model need only be solved once. Every instance
    attribute except ``pump_state`` must be equal, so a fitted coefficient or
    area factor anywhere keeps the two candidates independent.

    Args:
        jetpumps (list): Candidates from ``scoped_pumps``.

    Returns:
        dict: ``{replacement index: installed index}`` for exact twins.
    """
    def physics(jp):
        return {k: v for k, v in vars(jp).items() if k != "pump_state"}

    installed = [(i, physics(jp)) for i, jp in enumerate(jetpumps)
                 if getattr(jp, "pump_state", None) == "installed"]
    twins = {}
    if not installed:
        return twins
    for i, jp in enumerate(jetpumps):
        if getattr(jp, "pump_state", None) != "replacement":
            continue
        inputs = physics(jp)
        for j, source in installed:
            if inputs == source:
                twins[i] = j
                break
    return twins
