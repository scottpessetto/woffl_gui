"""Back-solver: infer flowing BHP (pwf) from a KNOWN test oil rate.

For wells **without a BHP gauge** the engineer knows the test's oil rate and
the installed pump, but not the flowing bottomhole pressure. They currently
hand-tweak the IPR (qwf / pwf / ResP) until the modeled jet pump reproduces the
test oil rate — slow, and it often lands on a non-convergent IPR (one whose
productivity the pump can't physically meet at that PF, so there's no operating
point).

This module inverts that. Given the installed pump + the test's conditions
(PF pressure, ResMix from WC/GOR, geometry, WHP) and the reservoir pressure
``pres``, it finds the ``pwf*`` such that the modeled pump reproduces the known
oil rate, using the IPR ``InFlow(qwf=qoil_test, pwf, pres)``.

The IPR's ``qwf`` is the OIL rate (BOPD) — matching ``InFlow``'s contract and
the GUI sidebar convention (sidebar ``qwf`` is oil; see
``woffl/flow/inflow.py``). The optimizer's ``WellConfig.qwf`` is total-liquid,
but we are in oil units here.

Monotonicity
------------
At fixed reservoir pressure ``pres``, the residual

    g(pwf) = modeled_oil(pwf) - qoil_test

is monotonic in ``pwf``: a *lower* ``pwf`` builds a tighter Vogel IPR (lower
qmax), so the pump delivers *less* oil; as ``pwf -> pres`` the IPR loosens and
the pump delivers more. So we scan a grid of ``pwf`` for a sign change in g,
then bisect within that sub-bracket.

Pump-limited case
-----------------
If even the most-productive feasible IPR (``pwf`` near ``pres``) still produces
*less* oil than the test, the pump can't reach the test rate at this PF — the
classic "increase oil -> won't converge" the engineer hits. We return a clear
pump-limited result rather than a number.

GUI-only — no upstream library change. ``run_jetpump_solver`` is treated as a
black box.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from woffl.gui.utils import (
    create_inflow,
    create_jetpump,
    create_reservoir_mix,
    run_jetpump_solver,
)


@dataclass
class BackMatchResult:
    """Outcome of a Match-test-oil-rate back-solve.

    Attributes:
        ok: True when a ``pwf*`` reproducing the test oil rate was found.
        pwf: The inferred flowing BHP (psig). None on failure / pump-limited.
        qwf_oil: The target oil rate the IPR is anchored on (BOPD).
        pres: Reservoir pressure the solve held fixed (psig).
        modeled_oil: Modeled oil rate at ``pwf*`` (BOPD). None on failure.
        message: Human-readable explanation (success or why it failed).
        pump_limited: True when the pump can't reach the test rate at this PF
            (the most-productive feasible IPR still produces less oil).
        iterations: Total solver evaluations (grid scan + bisection).
    """

    ok: bool
    pwf: float | None
    qwf_oil: float
    pres: float
    modeled_oil: float | None
    message: str
    pump_limited: bool = False
    iterations: int = 0
    # Best (closest-to-target) feasible point found, for diagnostics even when
    # the target itself was out of reach.
    best_pwf: float | None = field(default=None)
    best_oil: float | None = field(default=None)


def match_oil_rate(
    *,
    qoil_test: float,
    pres: float,
    nozzle: str,
    throat: str,
    surf_pres: float,
    form_temp: float,
    rho_pf: float,
    ppf_surf: float,
    wellbore,
    well_profile,
    form_wc: float,
    form_gor: float,
    ken: float,
    kth: float,
    kdi: float,
    field_model: str | None = None,
    jpump_direction: str = "reverse",
    tol_bopd: float = 3.0,
    tol_frac: float = 0.01,
    grid_points: int = 14,
    max_bisect: int = 24,
    pwf_lo_frac: float = 0.15,
    pwf_hi_frac: float = 0.97,
    pwf_floor: float = 100.0,
) -> BackMatchResult:
    """Infer the flowing BHP that makes the modeled pump match ``qoil_test``.

    Builds the same solver inputs the friction calibration uses (the installed
    pump + the test's conditions), but with a *trial* IPR ``InFlow(qoil_test,
    pwf, pres)``. Scans ``pwf`` over ``[pwf_lo_frac*pres, pwf_hi_frac*pres]``
    (clamped at ``pwf_floor``) for a sign change in
    ``g(pwf) = modeled_oil - qoil_test``, then bisects.

    Args:
        qoil_test: Known test OIL rate to reproduce (BOPD).
        pres: Reservoir pressure to hold fixed (psig).
        nozzle, throat: Installed pump identity (e.g. "12", "B").
        surf_pres: Wellhead / surface pressure for the solve (psi).
        form_temp: Formation temperature (deg F).
        rho_pf: Power-fluid density (passed through to the solver wrapper).
        ppf_surf: Power-fluid surface pressure (psi).
        wellbore, well_profile: Geometry objects (built once by the caller).
        form_wc: Formation water cut (fraction) for the ResMix.
        form_gor: Gas-oil ratio (scf/bbl) for the ResMix.
        ken, kth, kdi: Loss coefficients for the JetPump.
        field_model, jpump_direction: Passed through to the solver wrapper.
        tol_bopd / tol_frac: Convergence tolerance — stop when
            ``|modeled_oil - qoil_test|`` is within ``tol_bopd`` BOPD OR
            ``tol_frac`` of the target, whichever is looser.
        grid_points: Number of pwf samples in the initial scan.
        max_bisect: Max bisection iterations within the bracket.
        pwf_lo_frac / pwf_hi_frac: Scan range as fractions of ``pres``.
        pwf_floor: Lowest pwf to ever evaluate (psi).

    Returns:
        BackMatchResult.
    """
    if qoil_test is None or qoil_test <= 0:
        return BackMatchResult(
            ok=False, pwf=None, qwf_oil=float(qoil_test or 0), pres=float(pres),
            modeled_oil=None, iterations=0,
            message="No positive test oil rate to match against.",
        )
    if pres is None or pres <= 0:
        return BackMatchResult(
            ok=False, pwf=None, qwf_oil=float(qoil_test), pres=float(pres or 0),
            modeled_oil=None, iterations=0,
            message="Reservoir pressure must be greater than zero.",
        )

    pres = float(pres)
    qoil_test = float(qoil_test)
    tol = max(tol_bopd, tol_frac * qoil_test)

    # Geometry / fluid objects are reused across every solver call. ResMix is
    # conditioned in place by the solver, so build a fresh one per call to be
    # safe against shared-state surprises (see CLAUDE.md ResMix note).
    jetpump = create_jetpump(nozzle, throat, ken, kth, kdi)

    eval_count = 0
    # Track the closest feasible (solvable) point so we can report a useful
    # "best the pump can do" even when the target is unreachable.
    best_pwf: float | None = None
    best_oil: float | None = None
    best_gap = float("inf")

    def _modeled_oil(pwf: float) -> float | None:
        """Modeled oil rate for InFlow(qoil_test, pwf, pres); None on failure."""
        nonlocal eval_count, best_pwf, best_oil, best_gap
        # InFlow requires 0 <= pwf < pres; clamp/skip degenerate values.
        if pwf >= pres or pwf < 0:
            return None
        eval_count += 1
        try:
            inflow = create_inflow(qoil_test, pwf, pres)
        except ValueError:
            return None
        res_mix = create_reservoir_mix(
            form_wc, form_gor, form_temp, field_model
        )
        try:
            result = run_jetpump_solver(
                surf_pres,
                form_temp,
                rho_pf,
                ppf_surf,
                jetpump,
                wellbore,
                well_profile,
                inflow,
                res_mix,
                field_model=field_model,
                jpump_direction=jpump_direction,
                quiet=True,
            )
        except (ValueError, IndexError, ZeroDivisionError):
            # ThroatEntryNoSolution (subclasses both) and other solver
            # brittleness on marginal IPRs — treat as "no point here".
            return None
        if result is None:
            return None
        qoil_std = result[2]  # (psu, sonic, qoil_std, fwat, qnz, mach_te)
        if qoil_std is None:
            return None
        gap = abs(qoil_std - qoil_test)
        if gap < best_gap:
            best_gap, best_pwf, best_oil = gap, pwf, qoil_std
        return qoil_std

    # --- 1. Grid scan for a sign change in g(pwf) = modeled_oil - qoil_test. --
    lo = max(pwf_floor, pwf_lo_frac * pres)
    hi = pwf_hi_frac * pres
    if hi <= lo:
        return BackMatchResult(
            ok=False, pwf=None, qwf_oil=qoil_test, pres=pres,
            modeled_oil=None, iterations=eval_count,
            message=(
                f"Reservoir pressure {pres:.0f} psi is too low to scan a "
                "flowing-BHP range."
            ),
        )

    step = (hi - lo) / (grid_points - 1)
    grid = [lo + i * step for i in range(grid_points)]

    # Evaluate g across the grid, remembering feasible (solvable) samples.
    samples: list[tuple[float, float]] = []  # (pwf, g)
    for pwf in grid:
        mo = _modeled_oil(pwf)
        if mo is None:
            continue
        samples.append((pwf, mo - qoil_test))

    if not samples:
        return BackMatchResult(
            ok=False, pwf=None, qwf_oil=qoil_test, pres=pres,
            modeled_oil=None, iterations=eval_count,
            message=(
                f"The installed pump {nozzle}{throat} produced no solvable "
                f"operating point across the flowing-BHP range at PF "
                f"{ppf_surf:.0f} psi. Check the pump/throat, PF pressure, GOR, "
                "or reservoir pressure."
            ),
        )

    # Find an adjacent pair of feasible samples that brackets a root of g.
    bracket: tuple[float, float, float, float] | None = None  # (p_a,g_a,p_b,g_b)
    for (p_a, g_a), (p_b, g_b) in zip(samples, samples[1:]):
        if g_a == 0.0:
            bracket = (p_a, g_a, p_a, g_a)
            break
        if g_a * g_b <= 0.0:
            bracket = (p_a, g_a, p_b, g_b)
            break

    if bracket is None:
        # No sign change. Two flavors:
        #   * every feasible g < 0  -> even the loosest IPR underproduces ->
        #     pump-limited (the engineer's "increase oil -> won't converge").
        #   * every feasible g > 0  -> even the tightest IPR overproduces ->
        #     the test rate is below anything the pump will make here.
        all_neg = all(g < 0 for _, g in samples)
        all_pos = all(g > 0 for _, g in samples)
        if all_neg:
            msg = (
                f"The installed pump {nozzle}{throat} can't reach "
                f"{qoil_test:,.0f} BOPD at PF {ppf_surf:,.0f} psi"
            )
            if best_oil is not None:
                msg += (
                    f" - the most it can lift at this PF is ~{best_oil:,.0f} "
                    f"BOPD (at BHP ~{best_pwf:,.0f} psi)"
                )
            msg += ". Raise PF pressure, or check the pump/throat."
            return BackMatchResult(
                ok=False, pwf=None, qwf_oil=qoil_test, pres=pres,
                modeled_oil=None, iterations=eval_count, pump_limited=True,
                best_pwf=best_pwf, best_oil=best_oil, message=msg,
            )
        if all_pos:
            msg = (
                f"Even at the tightest feasible IPR, {nozzle}{throat} models "
                f"more than {qoil_test:,.0f} BOPD at PF {ppf_surf:,.0f} psi"
            )
            if best_oil is not None:
                msg += f" (best ~{best_oil:,.0f} BOPD at BHP ~{best_pwf:,.0f} psi)"
            msg += (
                ". The test rate looks low for this pump/PF — double-check the "
                "test oil rate, PF pressure, or reservoir pressure."
            )
            return BackMatchResult(
                ok=False, pwf=None, qwf_oil=qoil_test, pres=pres,
                modeled_oil=None, iterations=eval_count,
                best_pwf=best_pwf, best_oil=best_oil, message=msg,
            )
        # Mixed/sparse feasibility without a clean adjacent sign change — fall
        # back to the closest feasible point if it's within tolerance.
        if best_oil is not None and best_gap <= tol:
            return BackMatchResult(
                ok=True, pwf=best_pwf, qwf_oil=qoil_test, pres=pres,
                modeled_oil=best_oil, iterations=eval_count,
                best_pwf=best_pwf, best_oil=best_oil,
                message=(
                    f"Inferred flowing BHP = {best_pwf:,.0f} psi so "
                    f"{nozzle}{throat} models {best_oil:,.0f} BOPD "
                    f"(target {qoil_test:,.0f})."
                ),
            )
        return BackMatchResult(
            ok=False, pwf=None, qwf_oil=qoil_test, pres=pres,
            modeled_oil=None, iterations=eval_count,
            best_pwf=best_pwf, best_oil=best_oil,
            message=(
                f"Could not bracket a flowing BHP that reproduces "
                f"{qoil_test:,.0f} BOPD with {nozzle}{throat} at PF "
                f"{ppf_surf:,.0f} psi (solver feasibility was patchy across the "
                "range). Try adjusting GOR or PF pressure."
            ),
        )

    # --- 2. Bisect within the bracket. -------------------------------------
    p_a, g_a, p_b, g_b = bracket
    if g_a == 0.0:  # exact grid hit (rare)
        return BackMatchResult(
            ok=True, pwf=p_a, qwf_oil=qoil_test, pres=pres,
            modeled_oil=qoil_test, iterations=eval_count,
            best_pwf=best_pwf, best_oil=best_oil,
            message=(
                f"Inferred flowing BHP = {p_a:,.0f} psi so {nozzle}{throat} "
                f"models {qoil_test:,.0f} BOPD."
            ),
        )

    pwf_star = 0.5 * (p_a + p_b)
    g_star = 0.0
    for _ in range(max_bisect):
        pwf_star = 0.5 * (p_a + p_b)
        mo = _modeled_oil(pwf_star)
        if mo is None:
            # Solver blinked inside the bracket — nudge toward the feasible
            # endpoint (the one whose g we trust) and retry.
            pwf_star = 0.5 * (pwf_star + (p_a if abs(g_a) <= abs(g_b) else p_b))
            mo = _modeled_oil(pwf_star)
            if mo is None:
                break
        g_star = mo - qoil_test
        if abs(g_star) <= tol or (p_b - p_a) <= 2.0:
            break
        # Keep the sub-interval that still brackets the root.
        if g_a * g_star <= 0.0:
            p_b, g_b = pwf_star, g_star
        else:
            p_a, g_a = pwf_star, g_star

    modeled_oil = g_star + qoil_test
    if abs(g_star) <= tol:
        return BackMatchResult(
            ok=True, pwf=pwf_star, qwf_oil=qoil_test, pres=pres,
            modeled_oil=modeled_oil, iterations=eval_count,
            best_pwf=best_pwf, best_oil=best_oil,
            message=(
                f"Inferred flowing BHP = {pwf_star:,.0f} psi so "
                f"{nozzle}{throat} models {modeled_oil:,.0f} BOPD "
                f"(target {qoil_test:,.0f})."
            ),
        )

    # Bisection stalled (solver brittleness inside the bracket) but we may have
    # a close-enough best point from the scan/bisection.
    if best_oil is not None and best_gap <= tol:
        return BackMatchResult(
            ok=True, pwf=best_pwf, qwf_oil=qoil_test, pres=pres,
            modeled_oil=best_oil, iterations=eval_count,
            best_pwf=best_pwf, best_oil=best_oil,
            message=(
                f"Inferred flowing BHP = {best_pwf:,.0f} psi so "
                f"{nozzle}{throat} models {best_oil:,.0f} BOPD "
                f"(target {qoil_test:,.0f})."
            ),
        )
    return BackMatchResult(
        ok=False, pwf=pwf_star, qwf_oil=qoil_test, pres=pres,
        modeled_oil=modeled_oil, iterations=eval_count,
        best_pwf=best_pwf, best_oil=best_oil,
        message=(
            f"Found a flowing-BHP bracket but couldn't tighten it to "
            f"{qoil_test:,.0f} BOPD (closest ~{modeled_oil:,.0f} BOPD at "
            f"~{pwf_star:,.0f} psi) — the solver was unstable inside the "
            "bracket. Try a small GOR or PF-pressure change."
        ),
    )
