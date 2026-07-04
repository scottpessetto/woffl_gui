"""Pump Report Card — historical best-pump analysis for one well.

Answers "historically this well has best performed with a XX pump because…"
from the data that already flows through the app:

  * pump tenures from JP history (set-to-set — the JPCO rule; never
    ``Date Pulled``), coalesced into **pump eras**: consecutive installs of
    the same pump size are one era, and the number of slickline re-sets
    within an era is itself a reliability signal (B-28 ran 12B five times in
    11 months),
  * well tests per era: oil, water cut, lift water, test-day BHP, and the
    test-day power-fluid pressure joined from vw_pressure_daily,
  * the daily BHP series for time-in-good-state continuity between tests.

The verdict is deliberately transparent: a weighted composite of visible
components plus explicit confounder caveats (water-cut shift between eras,
small test counts, different PF regimes, eras predating PF/BHP coverage).
Raw oil comparison across eras is confounded — MPE-24's same 12C pump
medianed 279 BOPD at 62% WC in 2024 and 109 BOPD at 88% WC in 2025 — so the
reasons must always carry the context.

Pure pandas — no Streamlit, no Databricks. The GUI (tabs/jp_history_tab.py)
feeds it the frames it already fetches.
"""

from typing import Optional

import pandas as pd

# Ranking gates. Eras with fewer than MIN_TESTS_TO_RANK tests carry too little
# evidence to rank at all (still shown, flagged); SMALL_N_TESTS just caveats.
MIN_TESTS_TO_RANK = 3
SMALL_N_TESTS = 5

# Confounder thresholds for verdict caveats.
WC_SHIFT_CAVEAT = 0.10  # water-cut delta between eras that taints oil deltas
PF_SHIFT_CAVEAT_PSI = 300.0  # PF-regime delta that taints oil deltas

# Composite weights (renormalized over the components an era actually has —
# an era predating PF/BHP coverage is scored on what it does have).
SCORE_WEIGHTS = {
    "oil": 0.40,  # median oil vs the well's best era
    "stability": 0.25,  # time-in-good-state (tests + daily BHP)
    "longevity": 0.20,  # era length, capped at a year
    "efficiency": 0.15,  # oil per barrel of power fluid
}
LONGEVITY_CAP_DAYS = 365.0


def format_pump(nozzle, throat) -> str:
    """Format nozzle + throat into a label like '12B'.

    Tolerant of tracker quirks: a float nozzle ('12.0' → '12'), a missing
    nozzle (S-17 rows carry only the throat letter → 'D'), both missing → '?'.
    """
    parts = ""
    if pd.notna(nozzle):
        try:
            parts += str(int(float(nozzle)))
        except (ValueError, TypeError):
            parts += str(nozzle).strip()
    if pd.notna(throat):
        parts += str(throat).strip()
    return parts or "?"


# ---------------------------------------------------------------------------
# Era building
# ---------------------------------------------------------------------------


def build_pump_eras(well_jp: pd.DataFrame, end_date=None) -> list[dict]:
    """Coalesce a well's JP installs into pump eras.

    Args:
        well_jp: JP-history rows for ONE well with ``Date Set``,
            ``Nozzle Number``, ``Throat Ratio`` columns. Rows without a
            ``Date Set`` are ignored.
        end_date: end of the analysis window (default: today) — closes the
            last (active) era.

    Returns:
        Era dicts sorted by start: ``pump``, ``start``, ``end``, ``days``,
        ``installs`` (slickline sets within the era — JPCO churn),
        ``active`` (True for the era still in the hole).
    """
    if well_jp is None or well_jp.empty:
        return []
    df = well_jp.dropna(subset=["Date Set"]).sort_values("Date Set")
    if df.empty:
        return []
    end_date = pd.Timestamp(end_date) if end_date is not None else pd.Timestamp.now()

    eras: list[dict] = []
    for _, row in df.iterrows():
        pump = format_pump(row.get("Nozzle Number"), row.get("Throat Ratio"))
        start = pd.Timestamp(row["Date Set"])
        if eras and eras[-1]["pump"] == pump:
            eras[-1]["installs"] += 1  # same pump re-set (JPCO) — same era
        else:
            eras.append({"pump": pump, "start": start, "installs": 1})

    for i, era in enumerate(eras):
        era["end"] = eras[i + 1]["start"] if i + 1 < len(eras) else end_date
        era["days"] = max(int((era["end"] - era["start"]).days), 0)
        era["active"] = i == len(eras) - 1
    return eras


# ---------------------------------------------------------------------------
# Per-era metrics
# ---------------------------------------------------------------------------


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Numeric column as a Series; empty Series when the column is absent
    (DataFrame.get returns a scalar for frames with no columns)."""
    if name not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[name], errors="coerce")


def _median(series) -> Optional[float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.median()) if len(s) else None


def _thirds_drift(dates: pd.Series, values: pd.Series) -> Optional[float]:
    """median(last third) − median(first third), date-ordered. None under 6 pts."""
    df = pd.DataFrame({"d": dates, "v": pd.to_numeric(values, errors="coerce")})
    df = df.dropna().sort_values("d")
    n = len(df)
    if n < 6:
        return None
    k = n // 3
    return float(df["v"].tail(k).median() - df["v"].head(k).median())


def _longest_run_days(dates: pd.Series, good: pd.Series) -> int:
    """Longest span (days, first→last) of consecutive 'good' points."""
    df = pd.DataFrame({"d": dates, "g": good.astype(bool)}).dropna().sort_values("d")
    best = 0
    run_start = None
    prev_d = None
    for d, g in zip(df["d"], df["g"]):
        if g:
            if run_start is None:
                run_start = d
            prev_d = d
            best = max(best, int((prev_d - run_start).days))
        else:
            run_start = None
    return best


def era_metrics(
    era: dict,
    tests: pd.DataFrame,
    bhp_daily: Optional[pd.DataFrame],
    good_oil: Optional[float],
    good_bhp: Optional[float],
) -> dict:
    """Compute one era's report-card row.

    Args:
        era: dict from :func:`build_pump_eras`.
        tests: the well's full test frame (``WtDate``, ``WtOilVol``,
            ``WtWaterVol``, ``lift_wat``, ``BHP``, optional ``pf_press``).
        bhp_daily: daily BHP frame (``tag_date``, ``bhp``) or None.
        good_oil: "good test" oil threshold (BOPD); None disables.
        good_bhp: "good BHP" ceiling (psi, good = at/below); None disables.

    Missing data degrades gracefully: eras before PF/BHP coverage get None
    in those columns and the verdict names the gap instead of guessing.
    """
    m = dict(era)  # pump/start/end/days/installs/active

    t = tests[(tests["WtDate"] >= era["start"]) & (tests["WtDate"] < era["end"])]
    m["n_tests"] = int(len(t))

    oil = _col(t, "WtOilVol")
    wat = _col(t, "WtWaterVol")
    m["med_oil"] = _median(oil)
    m["p90_oil"] = float(oil.dropna().quantile(0.9)) if oil.notna().any() else None
    m["oil_drift"] = _thirds_drift(t["WtDate"], oil) if len(t) else None

    total = oil + wat
    wc = (wat / total).where(total > 0)
    m["med_wc"] = _median(wc)
    m["wc_drift"] = _thirds_drift(t["WtDate"], wc) if len(t) else None

    pf = _col(t, "pf_press")
    m["med_pf"] = _median(pf)
    m["n_pf"] = int(pf.notna().sum())

    m["med_bhp"] = _median(_col(t, "BHP"))

    lift = _col(t, "lift_wat")
    eff = (oil / lift).where(lift > 0)
    m["oil_per_pf"] = _median(eff)

    # Time-in-good-state — tests carry the oil signal, the daily series the
    # BHP signal (tests alone are too sparse to say the well *stayed* good).
    m["good_test_frac"] = None
    m["good_run_days"] = None
    if good_oil is not None and oil.notna().any():
        good = oil >= float(good_oil)
        m["good_test_frac"] = float(good[oil.notna()].mean())
        m["good_run_days"] = _longest_run_days(t["WtDate"], good.fillna(False))

    m["good_bhp_frac"] = None
    m["good_bhp_run_days"] = None
    if good_bhp is not None and bhp_daily is not None and not bhp_daily.empty:
        b = bhp_daily[
            (bhp_daily["tag_date"] >= era["start"])
            & (bhp_daily["tag_date"] < era["end"])
        ]
        vals = pd.to_numeric(b.get("bhp"), errors="coerce")
        if vals.notna().any():
            good_b = vals <= float(good_bhp)
            m["good_bhp_frac"] = float(good_b[vals.notna()].mean())
            m["good_bhp_run_days"] = _longest_run_days(
                b["tag_date"], good_b.fillna(False)
            )
    return m


def default_good_thresholds(
    tests: pd.DataFrame, bhp_daily: Optional[pd.DataFrame]
) -> tuple[Optional[int], Optional[int]]:
    """Auto-seed the two 'good' thresholds from the well's own history.

    good-oil = 75% of the P75 oil rate over the window (rounded to 10 BOPD);
    good-BHP = the median of the daily BHP series (rounded to 25 psi), falling
    back to test-day BHPs. None when there's no data to seed from.
    """
    good_oil = None
    oil = _col(tests, "WtOilVol").dropna()
    if len(oil):
        good_oil = int(round(0.75 * float(oil.quantile(0.75)) / 10.0) * 10)

    good_bhp = None
    src = None
    if bhp_daily is not None and not bhp_daily.empty:
        src = _col(bhp_daily, "bhp").dropna()
    if src is None or not len(src):
        src = _col(tests, "BHP").dropna()
    if src is not None and len(src):
        good_bhp = int(round(float(src.median()) / 25.0) * 25)
    return good_oil, good_bhp


# ---------------------------------------------------------------------------
# Ranking + verdict
# ---------------------------------------------------------------------------


def rank_eras(metrics: list[dict]) -> list[dict]:
    """Score each era 0–100 and sort best-first (unranked eras last).

    The composite is a weighted mean over the components an era actually has
    (SCORE_WEIGHTS, renormalized) — so an era predating PF coverage isn't
    punished for the missing efficiency term, it just isn't credited either.
    Eras with < MIN_TESTS_TO_RANK tests get ``ranked=False`` and no score.
    """
    if not metrics:
        return []
    max_oil = max((m["med_oil"] or 0.0) for m in metrics) or None
    max_eff = max((m["oil_per_pf"] or 0.0) for m in metrics) or None

    for m in metrics:
        m["ranked"] = m["n_tests"] >= MIN_TESTS_TO_RANK and m["med_oil"] is not None
        m["score"] = None
        if not m["ranked"]:
            continue
        comps: dict[str, float] = {}
        if max_oil:
            comps["oil"] = (m["med_oil"] or 0.0) / max_oil
        stab = [v for v in (m["good_test_frac"], m["good_bhp_frac"]) if v is not None]
        if stab:
            comps["stability"] = sum(stab) / len(stab)
        comps["longevity"] = min(m["days"], LONGEVITY_CAP_DAYS) / LONGEVITY_CAP_DAYS
        if max_eff and m["oil_per_pf"] is not None:
            comps["efficiency"] = m["oil_per_pf"] / max_eff
        wsum = sum(SCORE_WEIGHTS[k] for k in comps)
        m["score"] = round(
            100.0 * sum(SCORE_WEIGHTS[k] * v for k, v in comps.items()) / wsum, 1
        )

    return sorted(
        metrics,
        key=lambda m: (m["score"] is not None, m["score"] or 0.0),
        reverse=True,
    )


def _fmt_span(era: dict) -> str:
    end = "today" if era.get("active") else f"{era['end']:%b %Y}"
    return f"{era['start']:%b %Y} → {end}"


def _era_label(m: dict) -> str:
    """'12C (Sep 2023)' — dated so two eras of the SAME pump size stay
    distinguishable in caveat text (E-24 ran 12C twice, split by a 10C)."""
    return f"{m['pump']} ({m['start']:%b %Y})"


def build_verdict(ranked: list[dict], well_name: str) -> Optional[dict]:
    """The 'historically best' statement with explicit reasons AND caveats.

    Returns ``{"best": era_metrics, "reasons": [...], "caveats": [...]}`` or
    None when no era carries enough evidence to rank.
    """
    scored = [m for m in ranked if m.get("score") is not None]
    if not scored:
        return None
    best = scored[0]

    reasons: list[str] = []
    reasons.append(
        f"{best['med_oil']:,.0f} BOPD median across {best['n_tests']} tests "
        f"over {best['days']} days ({_fmt_span(best)})"
    )
    if best.get("good_test_frac") is not None:
        run = (
            f" (longest good run {best['good_run_days']} days)"
            if best.get("good_run_days")
            else ""
        )
        reasons.append(
            f"held 'good' oil on {best['good_test_frac']:.0%} of tests{run}"
        )
    if best.get("good_bhp_frac") is not None:
        run = (
            f" (longest streak {best['good_bhp_run_days']} days)"
            if best.get("good_bhp_run_days")
            else ""
        )
        reasons.append(
            f"BHP at/below target on {best['good_bhp_frac']:.0%} of days{run}"
        )
    if best.get("med_pf") is not None:
        reasons.append(f"at a median test-day PF of {best['med_pf']:,.0f} psi")
    if best.get("oil_per_pf") is not None:
        reasons.append(f"{best['oil_per_pf']:.2f} bbl oil per bbl power fluid")
    if best.get("installs", 1) > 1:
        reasons.append(
            f"re-set {best['installs']}× within the era (slickline churn — "
            "watch wash-out)"
        )

    caveats: list[str] = []
    runner = scored[1] if len(scored) > 1 else None
    if runner is not None:
        if (
            best.get("med_wc") is not None
            and runner.get("med_wc") is not None
            and abs(best["med_wc"] - runner["med_wc"]) > WC_SHIFT_CAVEAT
        ):
            caveats.append(
                f"water cut shifted {abs(best['med_wc'] - runner['med_wc']) * 100:.0f} "
                f"pts between the {_era_label(best)} and {_era_label(runner)} eras — "
                "part of the oil delta is reservoir change, not the pump"
            )
        if (
            best.get("med_pf") is not None
            and runner.get("med_pf") is not None
            and abs(best["med_pf"] - runner["med_pf"]) > PF_SHIFT_CAVEAT_PSI
        ):
            caveats.append(
                f"PF regimes differ ({best['med_pf']:,.0f} vs "
                f"{runner['med_pf']:,.0f} psi) — oil deltas partly reflect "
                "available power fluid"
            )
    if best["n_tests"] < SMALL_N_TESTS:
        caveats.append(
            f"only {best['n_tests']} tests back the winning era — thin evidence"
        )
    no_pf = [m["pump"] for m in ranked if m.get("n_pf", 0) == 0 and m["n_tests"] > 0]
    if no_pf:
        caveats.append(
            "no test-day PF data for era(s) "
            + ", ".join(dict.fromkeys(no_pf))
            + " (pre-dates vw_pressure_daily coverage) — compared on oil only"
        )
    # Compare the in-hole pump on OIL, not the composite — a young era is
    # dragged down by longevity by design, which is exactly the case where
    # "best historical" shouldn't be read as "pull the current pump".
    active = next((m for m in ranked if m.get("active")), None)
    if (
        active is not None
        and active is not best
        and active.get("med_oil")
        and best.get("med_oil")
        and active["med_oil"] >= 0.85 * best["med_oil"]
    ):
        gap = (best["med_oil"] - active["med_oil"]) / best["med_oil"] * 100.0
        how = (
            "at or above the best era's median oil"
            if gap <= 0
            else f"within {gap:.0f}% of the best era's median oil"
        )
        caveats.append(
            f"the current pump ({active['pump']}, {active['med_oil']:,.0f} BOPD "
            f"median) is {how} — young eras score low on longevity; keep "
            "watching before pulling it"
        )

    return {"best": best, "reasons": reasons, "caveats": caveats}


def build_report(
    well_jp: pd.DataFrame,
    tests: pd.DataFrame,
    bhp_daily: Optional[pd.DataFrame],
    good_oil: Optional[float],
    good_bhp: Optional[float],
    end_date=None,
) -> tuple[list[dict], Optional[dict]]:
    """One-call assembly: eras → metrics → ranking → verdict.

    Returns ``(ranked_metrics, verdict_or_None)``.
    """
    eras = build_pump_eras(well_jp, end_date=end_date)
    if not eras or tests is None or tests.empty:
        return [], None
    metrics = [
        era_metrics(e, tests, bhp_daily, good_oil, good_bhp) for e in eras
    ]
    ranked = rank_eras(metrics)
    well = str(tests["well"].iloc[0]) if "well" in tests else ""
    return ranked, build_verdict(ranked, well)
