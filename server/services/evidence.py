"""Field-evidence suction response mined from daily pressure history.

The M-Pad choke model's PF hydraulics are validated, but its cavitation
floor is contradicted by measured BHPs on roughly half the pad: the model
freezes psu at a floor 50-145 psi ABOVE where the gauges actually flow, and
field PF cuts move BHP (~+35 psi per -400 psi PF) where the model says it
cannot move at all. This module mines mpu.wells.vw_pressure_daily for the
measured suction response so pad_optimize can substitute field data exactly
where the model is contradicted.

Per well the evidence is a PLAIN DICT (woffl/gui never imports server code):

    {"floor": float|None, "psu_ref": float|None, "beta": float|None,
     "beta_source": "well"|"pad"|"default", "n_days": int, "n_pairs": int,
     "window": [str, str]}    # ISO dates, provenance

- floor:   min(p5 of flowing daily BHP, min well-test BHP) - the measured
           cavitation floor.
- psu_ref: median BHP over the last PSU_REF_DAYS flowing days - today's
           operating suction anchor.
- beta:    -median(dBHP/dPpf) over qualifying flowing-day pairs (Theil-Sen
           style median of pairwise slopes), clamped to BETA_CLAMP.

btmhole_prs in vw_pressure_daily IS the daily BHP series and is fully
populated (verified 31/31 days on M-064 over the last 30 days). If it ever
goes sparse, fall back to joining mpu.wells.vw_bhp_daily_clean on
enthid + tag_date - that join is already written in
server/services/history.py (_EXTENDED_TEST_QUERY / _BHP_DAILY_QUERY).
"""

from __future__ import annotations

import logging
from statistics import median
from typing import Any, Iterable, Optional

import pandas as pd

from server import config
from server.cache import ttl_cache

log = logging.getLogger("woffl.web.evidence")

# ---------------------------------------------------------------------------
# Tunables (one block - see the evidence-suction spec)
# ---------------------------------------------------------------------------

FLOOR_PCTL = 5              # flowing-BHP percentile taken as the measured floor
MIN_PAIRS = 5               # pairs needed before a well's own beta is trusted
PAIR_WINDOW_DAYS = (3, 30)  # beta pair spacing: close enough to be one regime,
                            # far enough apart to not be gauge noise
DPF_MIN_PSI = 100.0         # minimum |dPpf| for a pair to carry signal
BETA_CLAMP = (0.0, 0.5)     # physical bounds on -dBHP/dPpf
BETA_DEFAULT = 0.09         # measured MPM-64 Nov slope - the fleet prior
PSU_REF_DAYS = 14           # flowing days behind the psu_ref median
BHP_GLITCH_PSI = 50.0       # daily BHP at/below this is a dead/glitching gauge
PPF_MAX_PSI = 5500.0        # resolved PF above this is not a real header

# Fleet query - pattern lifted from pf_pressure.fetch_pf_latest: max() per
# well+day collapses repeated samples and prefers an operating reading over a
# same-day shut-in zero.
_FLEET_QUERY = """\
SELECT well_name, sample_date,
       max(tubing_prs) AS tubing_prs,
       max(inn_ann_prs) AS inn_ann_prs,
       max(btmhole_prs) AS btmhole_prs
FROM mpu.wells.vw_pressure_daily
WHERE sample_date >= date_sub(current_date(), 365)
GROUP BY well_name, sample_date
"""


# ---------------------------------------------------------------------------
# Fleet frame (cached) + per-well side inputs (all fail-soft)
# ---------------------------------------------------------------------------


@ttl_cache(config.TTL_EXTENDED_TESTS, maxsize=4)
def _fleet_pressure_daily() -> pd.DataFrame:
    """365 days of daily tubing/annulus/bottomhole pressure for ALL wells.

    One query for the fleet (the view names wells like "M-064"); the ``well``
    column carries the app-normalized name ("MPM-64") so callers slice by the
    optimizer's names directly. Raises on Databricks failure - pad_evidence's
    caller owns the fail-soft.
    """
    from woffl.assembly.databricks_client import execute_query
    from woffl.assembly.well_test_client import _normalize_well_name

    df = execute_query(_FLEET_QUERY)
    if df.empty:
        return pd.DataFrame(
            columns=["well", "sample_date", "tubing_prs", "inn_ann_prs", "btmhole_prs"]
        )
    for col in ("tubing_prs", "inn_ann_prs", "btmhole_prs"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["sample_date"] = pd.to_datetime(df["sample_date"], errors="coerce")
    df["well"] = df["well_name"].astype(str).str.strip().map(_normalize_well_name)
    return df


def _min_test_bhp(well: str) -> Optional[float]:
    """Lowest credible well-test BHP over the trailing year; None fail-soft."""
    try:
        from server.services import tests as tests_svc

        df = tests_svc.tests_for_well(well, 12, 0)
        if df is None or df.empty or "BHP" not in df.columns:
            return None
        bhp = pd.to_numeric(df["BHP"], errors="coerce")
        bhp = bhp[bhp > BHP_GLITCH_PSI]
        if bhp.empty:
            return None
        return float(bhp.min())
    except Exception:
        return None


def _install_dates_by_well() -> dict[str, list[pd.Timestamp]]:
    """JP install dates per well from the tracker; {} when unavailable.

    Used as the JPCO guard: a beta pair spanning a Date Set measures the
    pump changeout, not the PF response. No guard beats no evidence, so a
    missing tracker fails soft to {}.
    """
    try:
        from server.services import datasources

        jp_hist, source = datasources.jp_history_safe()
        if jp_hist is None or source is None:
            return {}
        if "Well Name" not in jp_hist.columns or "Date Set" not in jp_hist.columns:
            return {}
        frame = jp_hist.dropna(subset=["Date Set"])
        dates = pd.to_datetime(frame["Date Set"], errors="coerce")
        out: dict[str, list[pd.Timestamp]] = {}
        for well, when in zip(frame["Well Name"], dates):
            if pd.isna(when):
                continue
            out.setdefault(str(well), []).append(when.normalize())
        return out
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Pure per-well assembly (no I/O - the unit-test surface)
# ---------------------------------------------------------------------------


def _clamp_beta(value: float) -> float:
    lo, hi = BETA_CLAMP
    return min(max(float(value), lo), hi)


def well_evidence(
    daily: "pd.DataFrame | list[dict[str, Any]]",
    min_test_bhp: Optional[float] = None,
    res_pres: Optional[float] = None,
    install_dates: Optional[Iterable[Any]] = None,
) -> Optional[dict[str, Any]]:
    """Evidence dict for one well from its daily pressure rows. PURE - no I/O.

    Args:
        daily: rows with sample_date, tubing_prs, inn_ann_prs, btmhole_prs
            (one row per day - the fleet query's max() aggregation).
        min_test_bhp: lowest credible well-test BHP, or None.
        res_pres: saved-fit reservoir pressure; days with BHP at/above it are
            shut-in buildup, not flowing evidence.
        install_dates: JP install dates (anything pd.Timestamp accepts); beta
            pairs spanning one are dropped (JPCO response != PF response).

    Returns:
        The plain evidence dict, or None when no flowing day survives the
        filter chain (the well is then absent from pad_evidence's result).
        beta_source is "well" when the well earns its own slope (>= MIN_PAIRS
        pairs) else "default"; pad_evidence upgrades "default" to "pad" when
        siblings supply a pad median.
    """
    from woffl.assembly.pf_pressure import resolve_pf_pressure

    df = daily if isinstance(daily, pd.DataFrame) else pd.DataFrame(list(daily))
    needed = {"sample_date", "btmhole_prs"}
    if df is None or df.empty or not needed <= set(df.columns):
        return None

    df = df.copy()
    df["sample_date"] = pd.to_datetime(df["sample_date"], errors="coerce")
    df["btmhole_prs"] = pd.to_numeric(df["btmhole_prs"], errors="coerce")
    df = df.dropna(subset=["sample_date", "btmhole_prs"]).sort_values("sample_date")

    # Filter chain (order matters only for readability - filters are ANDed):
    # 1. dead/glitching gauge days (kills e.g. MPM-34's 29.5 psi row),
    df = df[df["btmhole_prs"] > BHP_GLITCH_PSI]
    # 2. days without a valid resolved PF pressure (shut-in / dead PF gauge),
    ppf = [
        resolve_pf_pressure(t, a)[0]
        for t, a in zip(df.get("tubing_prs", pd.Series(index=df.index, dtype=float)),
                        df.get("inn_ann_prs", pd.Series(index=df.index, dtype=float)))
    ]
    df["ppf"] = pd.to_numeric(pd.Series(ppf, index=df.index), errors="coerce")
    df = df[df["ppf"].notna() & (df["ppf"] <= PPF_MAX_PSI)]
    # 3. shut-in buildup days (BHP at/above reservoir pressure is not flowing).
    if res_pres is not None and float(res_pres) > 0:
        df = df[df["btmhole_prs"] < float(res_pres)]

    if df.empty:
        return None

    dates = list(df["sample_date"])
    bhps = [float(v) for v in df["btmhole_prs"]]
    ppfs = [float(v) for v in df["ppf"]]
    n_days = len(df)

    floor = float(df["btmhole_prs"].quantile(FLOOR_PCTL / 100.0))
    if min_test_bhp is not None:
        floor = min(floor, float(min_test_bhp))

    psu_ref = float(median(bhps[-PSU_REF_DAYS:]))

    installs: list[pd.Timestamp] = []
    for when in install_dates or ():
        stamp = pd.to_datetime(when, errors="coerce")
        if not pd.isna(stamp):
            installs.append(stamp.normalize())

    lo_days, hi_days = PAIR_WINDOW_DAYS
    slopes: list[float] = []
    for i in range(n_days):
        for j in range(i + 1, n_days):
            span = (dates[j] - dates[i]).days
            if span < lo_days:
                continue
            if span > hi_days:
                break  # dates are sorted: every later j is farther still
            dppf = ppfs[j] - ppfs[i]
            if abs(dppf) < DPF_MIN_PSI:
                continue
            if any(dates[i] < ins <= dates[j] for ins in installs):
                continue  # pair spans a JPCO - it measures the pump, not PF
            slopes.append((bhps[j] - bhps[i]) / dppf)

    n_pairs = len(slopes)
    if n_pairs >= MIN_PAIRS:
        beta = _clamp_beta(-median(slopes))
        beta_source = "well"
    else:
        beta = BETA_DEFAULT
        beta_source = "default"

    return {
        "floor": floor,
        "psu_ref": psu_ref,
        "beta": beta,
        "beta_source": beta_source,
        "n_days": n_days,
        "n_pairs": n_pairs,
        "window": [dates[0].date().isoformat(), dates[-1].date().isoformat()],
    }


def _apply_pad_fallback(rows: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Upgrade "default" betas to the pad median of well-earned betas.

    Fallback chain per the spec: n_pairs >= MIN_PAIRS -> the well's own beta
    ("well"); else the median of sibling wells' earned betas ("pad"); else
    BETA_DEFAULT stays ("default"). Mutates and returns ``rows``.
    """
    earned = [
        r["beta"]
        for r in rows.values()
        if r.get("beta_source") == "well" and r.get("beta") is not None
    ]
    if not earned:
        return rows
    pad_beta = _clamp_beta(median(earned))
    for r in rows.values():
        if r.get("beta_source") == "default":
            r["beta"] = pad_beta
            r["beta_source"] = "pad"
    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pad_evidence(
    well_names: list[str],
    res_pres: Optional[dict[str, float]] = None,
) -> dict[str, dict[str, Any]]:
    """Evidence dicts for a pad's wells, keyed by app well name (MPM-64).

    Wells without usable data (no daily rows, nothing survives the filter
    chain, or a per-well assembly error) are simply ABSENT - never fatal.
    Only the fleet fetch itself raises; the caller owns that fail-soft.

    Args:
        well_names: app-normalized names (the optimizer's WellConfig names).
        res_pres: well -> saved-fit reservoir pressure, for the shut-in
            buildup filter. Missing/None entries skip that filter.
    """
    fleet = _fleet_pressure_daily()
    installs = _install_dates_by_well()
    res_map = res_pres or {}

    out: dict[str, dict[str, Any]] = {}
    for well in well_names:
        try:
            sub = fleet[fleet["well"] == well] if not fleet.empty else fleet
            if sub.empty:
                continue
            row = well_evidence(
                sub,
                min_test_bhp=_min_test_bhp(well),
                res_pres=res_map.get(well),
                install_dates=installs.get(well),
            )
        except Exception:
            log.warning("suction evidence assembly failed for %s", well, exc_info=True)
            continue
        if row is not None:
            out[well] = row
    return _apply_pad_fallback(out)
