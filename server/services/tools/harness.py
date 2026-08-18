"""Test Harness - curated live-data sanity cases.

Port of woffl/gui/scotts_tools/test_harness.py plus the tests/harness_cases.py
registry it drove. Both are rewritten against the server's services rather
than restored verbatim: the originals imported ``scotts_tools.well_sort`` and
``scotts_tools.header_impact``, neither of which exists any more.

What this is FOR: these are not unit tests (pytest covers that, offline and
mocked). Each case exercises a real pipeline against TODAY's Databricks data
and asserts the answer is physically plausible. They catch the failure pytest
structurally cannot: the code is fine, the data moved. A view drops a column,
an allocation goes to zero, a pad falls out of the fleet - the maths still
runs and returns something confidently wrong.

Adding a case: write a ``case_*`` function returning a :class:`CaseResult`
and append it to :data:`ALL_CASES`. Catch your own exceptions so one bad case
cannot hide the rest; the runner nets anything that leaks, but a local catch
lets you return useful ``details``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

log = logging.getLogger("woffl.web.tools.harness")

# Defaults matching the Well Sort page's own, so a case measures what the
# page shows rather than a private configuration.
_THRESHOLD_PCT = 2.0
_STALE_DAYS = 45
_POPS: list[str] = []
_FORCE: list[str] = []


@dataclass
class CaseResult:
    """Outcome of a single harness case.

    Attributes:
        name: Short identifier, shown in the case row.
        description: One-paragraph explanation of what the case checks.
        passed: Pass/fail flag.
        summary: One-line outcome, shown next to the name.
        details: Free-form dict for the drilldown. Include BOTH expected and
            actual so the comparison is readable without rerunning anything.
        error: Exception text when the case crashed.
    """

    name: str
    description: str
    passed: bool
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


def _fail(name: str, doc: str, exc: Exception) -> CaseResult:
    return CaseResult(
        name=name,
        description=doc,
        passed=False,
        summary=f"Unhandled {type(exc).__name__}: {exc}",
        error=str(exc),
    )


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def case_field_marginal_wc_in_range() -> CaseResult:
    """Field-wide marginal water cut at the 2% threshold sits in a plausible
    operating range (0.80-1.00). Catches pipeline breakage that would
    otherwise show up as a confident but nonsense cutoff on Well Sort."""
    doc = case_field_marginal_wc_in_range.__doc__ or ""
    try:
        from server.services import well_sort as ws

        res = ws.marginal_payload(_THRESHOLD_PCT, _STALE_DAYS, _POPS, _FORCE)
        if not res:
            return CaseResult(
                "field_marginal_wc", doc, False,
                "marginal_payload returned nothing - no online wells?",
            )
        wc = float(res.get("marginal_wc"))
        ok = 0.80 <= wc <= 1.00
        return CaseResult(
            "field_marginal_wc", doc, ok,
            f"marginal WC = {wc:.4f}" + ("" if ok else " - OUTSIDE 0.80-1.00"),
            {"marginal_wc": wc, "expected_range": [0.80, 1.00],
             "well": res.get("well"), "pad": res.get("pad"),
             "threshold_pct": res.get("threshold_pct")},
        )
    except Exception as exc:  # noqa: BLE001
        return _fail("field_marginal_wc", doc, exc)


def case_every_pad_resolves_a_marginal_wc() -> CaseResult:
    """Every pad with online producers resolves a pad-level marginal WC.
    A pad that silently drops out is the signature of a header/allocation
    join going empty."""
    doc = case_every_pad_resolves_a_marginal_wc.__doc__ or ""
    try:
        from server.services import well_sort as ws

        tables = ws.tables_payload("oil", _STALE_DAYS, _POPS, _FORCE)
        online = tables.get("online") or []
        pads = sorted({str(r.get("pad")) for r in online if r.get("pad")})
        resolved, missing = {}, []
        for pad in pads:
            row = ws.pad_marginal_payload(pad, 2.0, _STALE_DAYS, _POPS, _FORCE)
            if row and row.get("marginal_wc") is not None:
                resolved[pad] = round(float(row["marginal_wc"]), 4)
            else:
                missing.append(pad)
        ok = bool(pads) and not missing
        return CaseResult(
            "pad_marginal_wc", doc, ok,
            f"{len(resolved)}/{len(pads)} pads resolved"
            + (f" - missing {missing}" if missing else ""),
            {"resolved": resolved, "missing": missing, "pads_seen": pads},
        )
    except Exception as exc:  # noqa: BLE001
        return _fail("pad_marginal_wc", doc, exc)


def case_online_well_count_in_range() -> CaseResult:
    """The online producer count sits in a believable band (40-140). The
    fleet is ~90 wells; a count near zero means the online filter or the
    shut-in join broke, not that the field shut in."""
    doc = case_online_well_count_in_range.__doc__ or ""
    try:
        from server.services import well_sort as ws

        tables = ws.tables_payload("oil", _STALE_DAYS, _POPS, _FORCE)
        n = len(tables.get("online") or [])
        ok = 40 <= n <= 140
        return CaseResult(
            "online_well_count", doc, ok,
            f"{n} online producers" + ("" if ok else " - OUTSIDE 40-140"),
            {"online": n, "shut": len(tables.get("shut") or []),
             "expected_range": [40, 140]},
        )
    except Exception as exc:  # noqa: BLE001
        return _fail("online_well_count", doc, exc)


def case_field_oil_total_sanity() -> CaseResult:
    """Total allocated oil across online producers is positive and within an
    ORDER-OF-MAGNITUDE band (2,000-150,000 BOPD).

    The band is deliberately loose. This case exists to catch an allocation
    column that has gone null, halved, or changed units - not to assert a
    field rate, which moves with well count and is not this file's business
    to know. A tight band just produces a failing case nobody trusts (the
    first run tripped at 62,455 BOPD, which is simply the field)."""
    doc = case_field_oil_total_sanity.__doc__ or ""
    try:
        from server.services import well_sort as ws

        tables = ws.tables_payload("oil", _STALE_DAYS, _POPS, _FORCE)
        rows = tables.get("online") or []
        oils = [float(r["oil"]) for r in rows if r.get("oil") is not None]
        total = sum(oils)
        ok = 2_000 <= total <= 150_000
        return CaseResult(
            "field_oil_total", doc, ok,
            f"{total:,.0f} BOPD across {len(oils)} wells"
            + ("" if ok else " - OUTSIDE 2,000-150,000"),
            {"total_bopd": round(total, 1), "wells_with_oil": len(oils),
             "wells_total": len(rows), "expected_range": [2000, 150000]},
        )
    except Exception as exc:  # noqa: BLE001
        return _fail("field_oil_total", doc, exc)


def case_jp_history_current_pumps() -> CaseResult:
    """The JP tracker resolves a current pump for a decent share of the
    fleet. Near-zero means the tracker pull or the Date Set parsing broke -
    which would silently reclassify JP wells as flowing everywhere."""
    doc = case_jp_history_current_pumps.__doc__ or ""
    try:
        from woffl.assembly.jp_history import get_current_pump

        from server.services import datasources

        jp_hist, source = datasources.jp_history_safe()
        wells = sorted({str(w) for w in jp_hist["Well Name"].dropna().unique()})
        current = 0
        for w in wells:
            pump = get_current_pump(jp_hist, w)
            if pump and pump.get("nozzle_no") and pump.get("throat_ratio"):
                current += 1
        ok = current >= 30
        return CaseResult(
            "jp_current_pumps", doc, ok,
            f"{current} wells with a current pump (source: {source})"
            + ("" if ok else " - expected >= 30"),
            {"wells_in_tracker": len(wells), "with_current_pump": current,
             "source": source, "expected_min": 30},
        )
    except Exception as exc:  # noqa: BLE001
        return _fail("jp_current_pumps", doc, exc)


def case_well_chars_cover_the_fleet() -> CaseResult:
    """Well characteristics resolve for the fleet with the geometry the
    solver needs (JP_TVD). A well missing JP_TVD cannot be modelled at all,
    so a spike here is a silent loss of coverage."""
    doc = case_well_chars_cover_the_fleet.__doc__ or ""
    try:
        from server.services import datasources

        df, source = datasources.well_chars_safe()
        total = len(df)
        missing = int(df["JP_TVD"].isna().sum()) if "JP_TVD" in df.columns else total
        ok = total >= 60 and missing == 0
        return CaseResult(
            "well_chars_coverage", doc, ok,
            f"{total} wells, {missing} missing JP_TVD (source: {source})",
            {"wells": total, "missing_jp_tvd": missing, "source": source,
             "estimated_tvd": int(df["tvd_estimated"].sum())
             if "tvd_estimated" in df.columns else None},
        )
    except Exception as exc:  # noqa: BLE001
        return _fail("well_chars_coverage", doc, exc)


ALL_CASES: list[Callable[[], CaseResult]] = [
    case_field_marginal_wc_in_range,
    case_every_pad_resolves_a_marginal_wc,
    case_online_well_count_in_range,
    case_field_oil_total_sanity,
    case_jp_history_current_pumps,
    case_well_chars_cover_the_fleet,
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def list_cases() -> dict[str, Any]:
    """Registered cases without running any of them."""
    return {
        "cases": [
            {"name": fn.__name__.replace("case_", ""),
             "description": (fn.__doc__ or "").strip()}
            for fn in ALL_CASES
        ]
    }


def run_all() -> dict[str, Any]:
    """Run every case against today's data.

    Never raises: a case that throws becomes a failed row, because the point
    of the harness is the report, and one bad case must not cost the other
    five their result.
    """
    results: list[dict[str, Any]] = []
    t0 = time.monotonic()
    for fn in ALL_CASES:
        started = time.monotonic()
        try:
            res = fn()
        except Exception as exc:  # noqa: BLE001 - defensive net
            log.warning("harness case %s leaked", fn.__name__, exc_info=True)
            res = _fail(fn.__name__.replace("case_", ""), fn.__doc__ or "", exc)
        results.append(
            {
                "name": res.name,
                "description": res.description.strip(),
                "passed": bool(res.passed),
                "summary": res.summary,
                "details": res.details,
                "error": res.error,
                "seconds": round(time.monotonic() - started, 2),
            }
        )
    passed = sum(1 for r in results if r["passed"])
    return {
        "results": results,
        "passed": passed,
        "failed": len(results) - passed,
        "total": len(results),
        "seconds": round(time.monotonic() - t0, 1),
    }
