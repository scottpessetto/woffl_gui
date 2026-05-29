"""Test Harness cases.

Each case is a function that takes no arguments and returns a :class:`CaseResult`.
Cases reuse the same ``@st.cache_data`` fetchers as the rest of the app, so
they run against today's live Databricks data (warm cache → instant; cold
cache → first call fetches from Databricks).

The Test Harness page lives at ``woffl/gui/scotts_tools/test_harness.py``
and iterates over :data:`ALL_CASES` to render pass/fail + drilldown panels.

Add new cases by:
1. Writing a ``case_xxx()`` function returning a :class:`CaseResult`.
2. Appending it to :data:`ALL_CASES`.

Conventions:
- Cases should be self-contained — no shared mutable state between cases.
- Wrap risky logic in try/except inside the case so one bad case doesn't
  hide the others (the page also wraps each call defensively, but local
  handling lets the case return its own diagnostic ``details``).
- ``passed=False`` should always come with a clear ``summary`` so the
  user can see what broke without opening the drilldown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class CaseResult:
    """Outcome of a single harness case.

    Attributes:
        name: Short identifier, shown in the case row.
        description: One-paragraph explanation of what the case checks.
        passed: Pass/fail flag.
        summary: One-line outcome, shown next to the name in the page.
        details: Free-form dict surfaced in the drilldown. Should include
            both expected and actual values so the user can read the
            comparison directly.
        error: Exception text if the case crashed (``passed=False``).
    """

    name: str
    description: str
    passed: bool
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


# ---------------------------------------------------------------------------
# Marginal WC sanity
# ---------------------------------------------------------------------------


def case_field_marginal_wc_in_range() -> CaseResult:
    """Field-wide marginal WC at the 2% threshold should sit in a plausible
    operating range (0.80–1.00). Catches data-pipeline breakage that would
    produce nonsense values (e.g. all NaN → exception, or a stripper with
    a 0.99 WC dominating the result)."""
    from woffl.gui.scotts_tools.well_sort import compute_field_marginal_wc

    try:
        result = compute_field_marginal_wc(threshold_pct=2.0)
    except Exception as e:
        return CaseResult(
            name="Field Marginal WC",
            description=case_field_marginal_wc_in_range.__doc__ or "",
            passed=False,
            summary=f"compute_field_marginal_wc raised: {type(e).__name__}",
            error=str(e),
        )

    if result is None:
        return CaseResult(
            name="Field Marginal WC",
            description=case_field_marginal_wc_in_range.__doc__ or "",
            passed=False,
            summary="No online non-POPs wells available — check Well Sort.",
        )

    wc = result["marginal_wc"]
    in_range = 0.80 <= wc <= 1.00
    return CaseResult(
        name="Field Marginal WC",
        description=case_field_marginal_wc_in_range.__doc__ or "",
        passed=in_range,
        summary=(
            f"marginal_wc={wc:.3f} (well={result['well']}, "
            f"{result['well_count']} wells, "
            f"{result['total_field_water']:,.0f} BWPD total)"
        ),
        details={
            "expected_range": [0.80, 1.00],
            "actual_marginal_wc": wc,
            "marginal_well": result["well"],
            "marginal_pad": result["pad"],
            "well_count": result["well_count"],
            "total_field_water_bwpd": result["total_field_water"],
            "threshold_pct": result["threshold_pct"],
        },
    )


def case_per_pad_marginal_wc_runs() -> CaseResult:
    """Every POPs pad in PUMP_LIMIT_PRESETS should compute a per-pad
    marginal WC against today's data. We expect at least 4 of the 6
    POPs pads to have online wells and produce a usable result —
    fewer than that is a bust signal (Databricks query regression, a
    bad shut-in log, or a missing column)."""
    from woffl.gui.scotts_tools.well_sort import (
        PUMP_LIMIT_PRESETS,
        compute_pad_marginal_wc,
    )

    min_passing_pads = 4

    per_pad: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    ok_count = 0
    for pad, preset in sorted(PUMP_LIMIT_PRESETS.items()):
        try:
            res = compute_pad_marginal_wc(pad=pad, pump_limit=preset)
        except Exception as e:
            failures.append(f"{pad}: {type(e).__name__}: {e}")
            per_pad[pad] = {"error": str(e)}
            continue
        if res is None:
            per_pad[pad] = {"status": "no online wells"}
            continue
        per_pad[pad] = {
            "marginal_wc": res["marginal_wc"],
            "well": res["well"],
            "pad_water": res["pad_water"],
            "pump_limit": res["pump_limit"],
            "headroom": res["headroom"],
            "water_basis": res["water_basis"],
            "well_count": res["well_count"],
        }
        ok_count += 1

    passed = (not failures) and (ok_count >= min_passing_pads)
    return CaseResult(
        name="Per-Pad Marginal WC",
        description=case_per_pad_marginal_wc_runs.__doc__ or "",
        passed=passed,
        summary=(
            f"{ok_count}/{len(per_pad)} pads OK"
            + (f"; {len(failures)} failed" if failures else "")
            + (
                f" (need ≥{min_passing_pads})"
                if ok_count < min_passing_pads
                else ""
            )
        ),
        details={
            "min_passing_pads": min_passing_pads,
            "ok_count": ok_count,
            "per_pad": per_pad,
            "failures": failures,
        },
    )


# ---------------------------------------------------------------------------
# Online wells sanity
# ---------------------------------------------------------------------------


def case_online_well_count_in_range() -> CaseResult:
    """Total online non-POPs wells should be within a plausible range
    (currently 30–120). Catches Databricks pipeline regressions where the
    shut-in log or XV view returns a degenerate result."""
    from woffl.gui.scotts_tools.well_sort import _build_online_non_pops

    try:
        df = _build_online_non_pops()
    except Exception as e:
        return CaseResult(
            name="Online Well Count",
            description=case_online_well_count_in_range.__doc__ or "",
            passed=False,
            summary=f"_build_online_non_pops raised: {type(e).__name__}",
            error=str(e),
        )

    n = len(df)
    in_range = 30 <= n <= 120
    return CaseResult(
        name="Online Well Count",
        description=case_online_well_count_in_range.__doc__ or "",
        passed=in_range,
        summary=f"{n} online non-POPs wells (expected 30–120)",
        details={
            "expected_range": [30, 120],
            "actual_count": n,
            "columns": list(df.columns) if not df.empty else [],
        },
    )


def case_field_oil_total_sanity() -> CaseResult:
    """Sum of online non-POPs Oil should be in a plausible MPU range
    (15,000–80,000 BOPD). Catches unit / column regressions that would
    show oil rates off by 1000x or zero."""
    from woffl.gui.scotts_tools.well_sort import _build_online_non_pops

    try:
        df = _build_online_non_pops()
    except Exception as e:
        return CaseResult(
            name="Field Oil Total",
            description=case_field_oil_total_sanity.__doc__ or "",
            passed=False,
            summary=f"_build_online_non_pops raised: {type(e).__name__}",
            error=str(e),
        )

    if df.empty or "Oil" not in df.columns:
        return CaseResult(
            name="Field Oil Total",
            description=case_field_oil_total_sanity.__doc__ or "",
            passed=False,
            summary="No Oil column / no online wells.",
        )

    total = float(df["Oil"].dropna().sum())
    in_range = 15_000 <= total <= 80_000
    return CaseResult(
        name="Field Oil Total",
        description=case_field_oil_total_sanity.__doc__ or "",
        passed=in_range,
        summary=f"{total:,.0f} BOPD across non-POPs wells (expected 15,000–80,000)",
        details={
            "expected_range": [15_000, 80_000],
            "actual_total_oil_bopd": total,
            "well_count": int(df["Oil"].notna().sum()),
        },
    )


# ---------------------------------------------------------------------------
# Configuration / presets consistency
# ---------------------------------------------------------------------------


def case_pump_handler_presets_consistent() -> CaseResult:
    """Every POPs pad must have entries in BOTH PUMP_LIMIT_PRESETS and
    POPS_PUMP_HANDLES. A mismatch means the per-pad calculator will fall
    back to a default water basis silently, producing wrong numbers."""
    from woffl.gui.scotts_tools.well_sort import (
        POPS_PUMP_HANDLES,
        PUMP_LIMIT_PRESETS,
    )

    limit_pads = set(PUMP_LIMIT_PRESETS.keys())
    handle_pads = set(POPS_PUMP_HANDLES.keys())

    missing_handle = sorted(limit_pads - handle_pads)
    missing_limit = sorted(handle_pads - limit_pads)
    passed = not missing_handle and not missing_limit

    return CaseResult(
        name="Pad Preset Consistency",
        description=case_pump_handler_presets_consistent.__doc__ or "",
        passed=passed,
        summary=(
            "Consistent"
            if passed
            else (
                f"Mismatch — missing handlers: {missing_handle}, "
                f"missing limits: {missing_limit}"
            )
        ),
        details={
            "limit_pads": sorted(limit_pads),
            "handle_pads": sorted(handle_pads),
            "missing_handler": missing_handle,
            "missing_limit": missing_limit,
        },
    )


# ---------------------------------------------------------------------------
# Registry — append new cases here.
# ---------------------------------------------------------------------------

def case_header_impact_universe() -> CaseResult:
    """The Header Impact tool's producer universe + lift classifier should
    find a sensible mix on today's data (≥80 producers, ≥30 ESPs, ≥30 JPs).
    Catches a schema break in vw_well_test / vw_well_header / jp_history that
    would empty the input table or misclassify lift types."""
    from woffl.assembly.databricks_client import fetch_jp_history
    from woffl.gui.scotts_tools import header_impact as hi

    try:
        ov = hi.fetch_well_overview(6)
        jp_hist = fetch_jp_history()
    except Exception as e:
        return CaseResult(
            name="Header Impact universe",
            description=case_header_impact_universe.__doc__ or "",
            passed=False,
            summary=f"raised: {type(e).__name__}",
            error=str(e),
        )

    if ov is None or ov.empty:
        return CaseResult(
            name="Header Impact universe",
            description=case_header_impact_universe.__doc__ or "",
            passed=False,
            summary="fetch_well_overview returned empty",
        )

    ov = ov.copy()
    ov["lift"] = [hi._classify_lift(r["well"], jp_hist, r) for _, r in ov.iterrows()]
    n = int(len(ov))
    mix = ov["lift"].value_counts().to_dict()
    n_esp = int(mix.get("ESP", 0))
    n_jp = int(mix.get("JP", 0))
    ok = n >= 80 and n_esp >= 30 and n_jp >= 30

    return CaseResult(
        name="Header Impact universe",
        description=case_header_impact_universe.__doc__ or "",
        passed=ok,
        summary=(
            f"{n} producers (ESP {n_esp}, JP {n_jp})"
            if ok
            else f"below floor: {n} producers (ESP {n_esp}, JP {n_jp})"
        ),
        details={
            "total_producers": n,
            "lift_mix": mix,
            "expected_floor": {"total": 80, "ESP": 30, "JP": 30},
        },
    )


ALL_CASES: list[Callable[[], CaseResult]] = [
    case_field_marginal_wc_in_range,
    case_per_pad_marginal_wc_runs,
    case_online_well_count_in_range,
    case_field_oil_total_sanity,
    case_pump_handler_presets_consistent,
    case_header_impact_universe,
]
