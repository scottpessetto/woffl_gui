"""Well Sort decision engine - pure DataFrame math, no Streamlit, no I/O.

Extracted from ``woffl.gui.scotts_tools.well_sort`` so the FastAPI web app
can reuse the SINGLE canonical implementation (R-8 / P1-30: exactly one
marginal-WC walk exists; do not re-add another). The Streamlit module keeps
thin wrappers with its original signatures - both GUIs call into here.

Inputs are the frames produced by ``well_sort_client.build_online_table`` /
``build_shut_in_table`` (after ``apply_pops_pad``); every function is a pure
function of its arguments.
"""

from __future__ import annotations

import pandas as pd

# Pads with on-pad production separation (POPs) - the default selection for
# the Wells tab's config UI; wells on these pads get PopsPad=True.
DEFAULT_POPS_PADS = ["E", "F", "H", "I", "M", "S"]

# Per-POPs-pad water handling capacity (BWPD). Starter presets for the
# Per-Pad Marginal WC calculator; the UI lets users override each value.
# Update these numbers here if the field hardware changes.
PUMP_LIMIT_PRESETS: dict[str, int] = {
    "E": 20_000,
    "F": 28_000,
    "H": 30_000,
    "I": 32_000,
    "M": 55_000,
    "S": 35_000,
}

# Which water stream the pad's pump actually handles. Some POPs pads only
# separate power-fluid water (formation water passes through to the central
# facility); others handle the full produced stream. The marginal-WC calc
# measures cumulative water against the pad's pump limit using whichever
# stream the pad actually constrains.
POPS_PUMP_HANDLES: dict[str, str] = {
    "E": "total",  # full POPs: formation + lift water over the pad pump
    "F": "total",
    "M": "total",
    "I": "lift",  # PF-only: pad pump sees power-fluid water only
    "H": "lift",
    "S": "lift",
}


def field_marginal_wc(non_pops: pd.DataFrame, threshold_pct: float = 2.0) -> dict | None:
    """Field-wide marginal WC at the given cumulative-water threshold.

    Sort online non-POPs wells by TotalWC descending, walk the list adding
    TotalWater. The marginal WC is the WC of the first well at which the
    cumulative water crosses the threshold fraction of field water. This
    rejects single-well noise (a 0.99-WC stripper making 40 BWPD lifts the
    cumulative by ~zero, so the marginal lands on a meaningful well).

    Returns a dict with the headline result + the ranked dataframe for
    display, or None when no online non-POPs wells with valid TotalWC /
    TotalWater are available.
    """
    if non_pops.empty:
        return None

    valid = non_pops[non_pops["TotalWC"].notna() & non_pops["TotalWater"].notna()].copy()
    valid = valid[valid["TotalWater"] > 0].copy()
    if valid.empty:
        return None

    valid = valid.sort_values("TotalWC", ascending=False).reset_index(drop=True)
    total_field_water = float(valid["TotalWater"].sum())
    valid["CumWater"] = valid["TotalWater"].cumsum()
    valid["CumWaterPct"] = (valid["CumWater"] / total_field_water) * 100.0

    above_mask = valid["CumWaterPct"] >= threshold_pct
    if above_mask.any():
        marg_idx = int(above_mask.idxmax())
    else:
        # Threshold higher than 100% (impossible by construction). Use the
        # bottom well as a defensive fallback.
        marg_idx = len(valid) - 1

    marg_row = valid.iloc[marg_idx]
    return {
        "marginal_wc": float(marg_row["TotalWC"]),
        "well": str(marg_row["Well"]),
        "pad": str(marg_row.get("Pad", "-")),
        "total_field_water": total_field_water,
        "well_count": int(len(valid)),
        "threshold_pct": float(threshold_pct),
        "marg_idx": marg_idx,
        "ranked_df": valid,
    }


def pad_marginal_wc(online_full: pd.DataFrame, pad: str, pump_limit: float) -> dict | None:
    """Per-pad marginal WC + headroom against the pad pump's capacity.

    The pad's pump only sees one water stream:
      * **lift** for PF-only POPs (I, H, S) - pump only handles power-fluid
        water; formation water passes through to the central facility.
      * **total** for full POPs (E, F, M) - pump handles formation + PF.

    Per-well "pad WC" = ``water / (water + oil)`` using whichever stream
    the pad pump actually sees. For PF-only pads this is a new metric
    ("PFWC"); for full POPs it equals the standard TotalWC.

    The pad's **marginal WC** is just the max of that per-well WC - i.e.
    the worst-performing well on the pad pump. The pump limit doesn't
    influence the marginal; instead it gives the **headroom** number
    (positive = "X BWPD available to allocate", negative = "OVER by X").

    Returns dict with the headline result + the ranked dataframe (sorted
    by pad WC descending), or None if the pad has no online wells with a
    usable water + oil pair.
    """
    if online_full.empty:
        return None

    pad_df = online_full[online_full["Pad"] == pad].copy()
    if pad_df.empty:
        return None

    water_basis = POPS_PUMP_HANDLES.get(pad, "total")
    water_col = "LiftWater" if water_basis == "lift" else "TotalWater"
    if water_col not in pad_df.columns or "Oil" not in pad_df.columns:
        return None

    valid = pad_df[pad_df[water_col].notna() & pad_df["Oil"].notna()].copy()
    # Drop rows where both rates are zero - no per-well WC defined.
    valid = valid[(valid[water_col] > 0) | (valid["Oil"] > 0)].copy()
    if valid.empty:
        return None

    # Per-well WC measured against the pad pump's stream.
    denom = valid[water_col] + valid["Oil"]
    valid["WC_pad"] = valid[water_col] / denom.where(denom > 0, 1.0)

    # Sort worst-first so the marginal lands at the top of the table.
    valid = valid.sort_values("WC_pad", ascending=False).reset_index(drop=True)

    total_pad_water = float(valid[water_col].sum())
    marg_idx = 0
    marg_row = valid.iloc[marg_idx]
    marginal_wc = float(marg_row["WC_pad"])

    pump_limit_f = float(pump_limit) if pump_limit and pump_limit > 0 else 0.0
    headroom = pump_limit_f - total_pad_water if pump_limit_f > 0 else None

    return {
        "marginal_wc": marginal_wc,
        "well": str(marg_row["Well"]),
        "pad": pad,
        "pad_water": total_pad_water,
        "pump_limit": pump_limit_f,
        "headroom": float(headroom) if headroom is not None else None,
        "well_count": int(len(valid)),
        "marg_idx": marg_idx,
        "ranked_df": valid,
        "water_basis": water_basis,
        "water_col": water_col,
    }


# ---------------------------------------------------------------------------
# Triage - keep / SI / BOL decision engine
# ---------------------------------------------------------------------------
#
# Driving rule, per Scott: a well's water cut vs the field MARGINAL WC sets
# the lean -
#   * online well, WC above marginal  -> shut-in (SI) candidate
#   * shut well,   WC below marginal  -> bring-on-line (BOL) candidate
# A poor LATEST test against a healthy recent HISTORY is deliberately NOT
# acted on: it's flagged to verify / BOL-trial, because a single bad test
# often recovers (Scott BOLs these to check). The history signal reuses what
# the engine already computes - the 2-month outlier deviation (online) and
# the 90-day near-last-test average (shut).


def effective_wc(row) -> tuple[float, str | None]:
    """Effective last-test WC and the basis it was read on.

    Returns ``(wc, basis)`` - basis ``"total"`` (``totl_wc``, the same basis
    as the marginal line), ``"form"`` (formation-water fallback when the test
    carries no total WC - reads LOW vs the total-WC line on lifted wells, so
    callers must flag it rather than judge silently, P1-27), or ``None`` when
    no WC exists at all.
    """
    v = row.get("TotalWC")
    if pd.notna(v):
        return float(v), "total"
    v = row.get("WC")
    if pd.notna(v):
        return float(v), "form"
    return float("nan"), None


def form_basis_note(wc_basis: str | None) -> str:
    """Why-suffix flagging a decision made on form-basis WC (P1-27)."""
    if wc_basis != "form":
        return ""
    return " [form-basis WC - test has no total WC; reads low vs the total-WC line]"


def add_online_decision(online_df: pd.DataFrame, marginal_wc: float) -> pd.DataFrame:
    """Augment the online table with a keep/SI decision vs the marginal WC.

    Adds: Decision (emoji-tagged display string), DecisionCode (stable ASCII
    code for programmatic consumers: pops / verify_stale / keep / verify_si /
    si), Why (plain-language reason), WCvsMarginal (Total WC - marginal),
    WCBasis ("total"/"form" - which WC the decision used; a form-basis
    fallback reads low vs the total-WC line and is flagged in Why, P1-27),
    and a hidden ``_rank`` for sorting.

    Rule:
      POPS-pad well                              -> POPS (own handling)
      stale / no test                            -> Verify (unknown state)
      WC <= marginal                             -> Keep online
      WC > marginal AND latest test anomalous    -> Verify before SI
        (oil outlier-LOW or water outlier-HIGH - either way one fluky test
         shouldn't condemn the well, P1-28)
      WC > marginal otherwise                    -> SI candidate
    """
    from woffl.assembly.well_sort_client import OUTLIER_PCT

    df = online_df.copy()
    if df.empty:
        for c in ("WCvsMarginal", "_rank"):
            df[c] = pd.Series(dtype="float")
        for c in ("Decision", "DecisionCode", "Why", "WCBasis"):
            df[c] = pd.Series(dtype="object")
        return df

    decisions, codes, whys, deltas, bases, ranks = [], [], [], [], [], []
    for _, r in df.iterrows():
        wc, wc_basis = effective_wc(r)
        deltas.append(wc - marginal_wc if pd.notna(wc) else float("nan"))
        bases.append(wc_basis)
        basis_note = form_basis_note(wc_basis)
        pops = bool(r.get("PopsPad"))
        stale = bool(r.get("StaleTest"))
        oil_dev = r.get("OilDev")
        wat_dev = r.get("WatDev")
        if pops:
            decisions.append("⚪ POPS (own handling)")
            codes.append("pops")
            whys.append(
                "On a POPs pad - water separated on-pad; judge with the "
                "per-pad Marginal WC calc, not the field line."
            )
            ranks.append(4)
        elif pd.isna(wc) or stale:
            decisions.append("⚠️ Verify - stale/no test")
            codes.append("verify_stale")
            whys.append("No recent representative test; re-test before any SI call.")
            ranks.append(1)
        elif wc <= marginal_wc:
            decisions.append("✅ Keep online")
            codes.append("keep")
            whys.append(
                f"WC {wc * 100:.0f}% <= marginal {marginal_wc * 100:.0f}% - worth its water."
                + basis_note
            )
            ranks.append(2)
        else:
            # Outlier protection: a single anomalous test - oil reading LOW
            # or water reading HIGH - inflates the apparent WC; verify with a
            # re-test instead of condemning the well on it (P1-28).
            oil_down = pd.notna(oil_dev) and float(oil_dev) < -OUTLIER_PCT
            wat_up = pd.notna(wat_dev) and float(wat_dev) > OUTLIER_PCT
            if oil_down or wat_up:
                anomalies = []
                if oil_down:
                    anomalies.append(
                        f"oil is {abs(float(oil_dev)) * 100:.0f}% below the 2-mo avg"
                    )
                if wat_up:
                    anomalies.append(
                        f"water is {float(wat_dev) * 100:.0f}% above the 2-mo avg"
                    )
                decisions.append("⚠️ Verify before SI")
                codes.append("verify_si")
                whys.append(
                    f"WC {wc * 100:.0f}% > marginal {marginal_wc * 100:.0f}%, but the "
                    f"latest test looks anomalous - {' and '.join(anomalies)} - "
                    "confirm with a re-test before SI (may recover)." + basis_note
                )
                ranks.append(1)
            else:
                decisions.append("🔴 SI candidate")
                codes.append("si")
                whys.append(
                    f"WC {wc * 100:.0f}% > marginal {marginal_wc * 100:.0f}% - "
                    "water not worth the oil." + basis_note
                )
                ranks.append(0)
    df["WCvsMarginal"] = deltas
    df["WCBasis"] = bases
    df["Decision"] = decisions
    df["DecisionCode"] = codes
    df["Why"] = whys
    df["_rank"] = ranks
    return df


def add_shut_decision(shut_df: pd.DataFrame, marginal_wc: float) -> pd.DataFrame:
    """Augment the shut-in (offline) table with a BOL decision vs marginal WC.

    Adds: Decision, DecisionCode (verify_no_test / bol / bol_trial /
    verify_form_hist / leave_shut), Why, WCvsMarginal, NearAvgWC (90-day
    history WC), NearAvgWCBasis / WCBasis ("total"/"form"), and a hidden
    ``_rank``.

    The 90-day history WC is compared to a TOTAL-WC marginal line, so it is
    computed on the same basis wherever possible: the near-window SQL only
    averages FORMATION water (``AVG(form_wat)``), so when the last test
    carries lift-water data it is folded in as the window's lift proxy. Where
    only form-basis history exists it is NOT allowed to grant a BOL trial -
    form WC reads systematically low vs the total-WC line on lifted wells -
    the row is flagged to verify instead (P1-27).

    Rule:
      no test                                       -> Verify (no test)
      last WC <= marginal                           -> BOL candidate
      last WC > marginal BUT 90-day TOTAL-basis
        hist WC <= marg                             -> BOL trial (recovery?)
      last WC > marginal, form-basis hist <= marg   -> Verify (basis unreliable)
      last WC > marginal and history also high      -> Leave shut
    """
    df = shut_df.copy()
    if df.empty:
        for c in ("WCvsMarginal", "NearAvgWC", "_rank"):
            df[c] = pd.Series(dtype="float")
        for c in ("Decision", "DecisionCode", "Why", "WCBasis", "NearAvgWCBasis"):
            df[c] = pd.Series(dtype="object")
        return df

    decisions, codes, whys, deltas = [], [], [], []
    near_wcs, near_bases, wc_bases, ranks = [], [], [], []
    for _, r in df.iterrows():
        wc, wc_basis = effective_wc(r)
        deltas.append(wc - marginal_wc if pd.notna(wc) else float("nan"))
        wc_bases.append(wc_basis)
        basis_note = form_basis_note(wc_basis)
        # 90-day near-last-test history WC: the "was it healthy recently?"
        # signal that protects against condemning a well on one bad test.
        # NearAvgWater is formation water only, but the marginal line is
        # total WC - when the last test has lift-water data, fold it in so
        # history and line share a basis; a form-only history is flagged
        # below rather than decided on (P1-27).
        na_oil, na_wat = r.get("NearAvgOil"), r.get("NearAvgWater")
        lift_wat = r.get("LiftWater")
        nwc, nwc_basis = float("nan"), None
        if pd.notna(na_oil) and pd.notna(na_wat):
            hist_oil, hist_wat = float(na_oil), float(na_wat)
            if pd.notna(lift_wat):
                hist_wat += float(lift_wat)
                nwc_basis = "total"
            else:
                nwc_basis = "form"
            if (hist_oil + hist_wat) > 0:
                nwc = hist_wat / (hist_oil + hist_wat)
            else:
                nwc_basis = None
        near_wcs.append(nwc)
        near_bases.append(nwc_basis)

        if pd.isna(wc):
            decisions.append("⚠️ Verify - no test")
            codes.append("verify_no_test")
            whys.append("No usable test on record; test before BOL.")
            ranks.append(2)
        elif wc <= marginal_wc:
            decisions.append("🟢 BOL candidate")
            codes.append("bol")
            whys.append(
                f"Last WC {wc * 100:.0f}% <= marginal {marginal_wc * 100:.0f}% - "
                "worth bringing on." + basis_note
            )
            ranks.append(0)
        elif pd.notna(nwc) and nwc <= marginal_wc:
            if nwc_basis == "total":
                decisions.append("🔬 BOL trial")
                codes.append("bol_trial")
                whys.append(
                    f"Last WC {wc * 100:.0f}% > marginal, but the 90-day history WC "
                    f"{nwc * 100:.0f}% (total basis, incl lift water) was below - "
                    "BOL to see if the oil rate has recovered." + basis_note
                )
                ranks.append(1)
            else:
                # Form-basis history reads LOW vs the total-WC line - don't
                # grant a BOL trial on it; flag for a re-test instead.
                decisions.append("⚠️ Verify - form-basis history")
                codes.append("verify_form_hist")
                whys.append(
                    f"Last WC {wc * 100:.0f}% > marginal {marginal_wc * 100:.0f}%; "
                    f"the 90-day history WC {nwc * 100:.0f}% is below the line but "
                    "formation-water basis only (no lift-water data) - unreliable "
                    "vs the total-WC line. Re-test before any BOL call."
                )
                ranks.append(2)
        else:
            # Safe on either basis: a form-basis history above the line
            # implies the total-basis history is above it too.
            decisions.append("⏸️ Leave shut")
            codes.append("leave_shut")
            whys.append(
                f"WC {wc * 100:.0f}% > marginal {marginal_wc * 100:.0f}% (history too) - "
                "water not worth it." + basis_note
            )
            ranks.append(3)
    df["WCvsMarginal"] = deltas
    df["NearAvgWC"] = near_wcs
    df["NearAvgWCBasis"] = near_bases
    df["WCBasis"] = wc_bases
    df["Decision"] = decisions
    df["DecisionCode"] = codes
    df["Why"] = whys
    df["_rank"] = ranks
    return df
