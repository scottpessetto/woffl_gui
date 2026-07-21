"""Per-well reviewed-state store for the pad optimization workflow.

The pad-scoped "Review & Calibrate Wells" step lets an engineer step through
every well on a pad in the single-well Solver, verify/adjust the match, and
**Save** the reviewed state. This module is the pure-data backbone for that:

  - ``snapshot_from_params`` captures the sidebar ``SimulationParams`` (+ the
    calibrated friction coefficients + provenance) into one serializable
    per-well entry dict.
  - ``to_well_config`` turns an entry into the optimizer's ``WellConfig``.
  - ``store_to_dataframe`` / ``dataframe_to_store`` round-trip the whole store
    to/from a CSV the engineer can download and re-upload for future reruns
    or edits (the eventual Databricks save target shares this schema).

No Streamlit here — session-state hydration lives in the step module so this
stays unit-testable.

UNIT GOTCHA (verified against the live code, do not "simplify" away):
  - The sidebar/Solver ``qwf`` is **OIL** rate (BOPD). ``InFlow.qwf`` is oil
    (``woffl/flow/inflow.py``: "The oil rate is used in conjunction with a
    reservoir mixture's wc and gor to calculate other components"), and the
    Solver feeds ``params.qwf`` straight in.
  - ``WellConfig.qwf`` is **TOTAL LIQUID** (BLPD); ``NetworkOptimizer`` converts
    it back to oil via ``oil = qwf * (1 - form_wc)``
    (``network_optimizer.py``: "convert total fluid IPR rate to oil rate").
  So a snapshot must convert OIL -> TOTAL LIQUID (``qwf / (1 - wc)``); the
  optimizer's reverse conversion then reproduces the exact IPR the engineer
  reviewed. ``OIL_RATE_FIELD`` preserves the as-reviewed oil rate for display
  and provenance so the round-trip is auditable.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from woffl.assembly.network_optimizer import WellConfig

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

# WellConfig-bound fields carried verbatim into the optimizer's WellConfig.
_WELLCONFIG_FLOAT_FIELDS = (
    "res_pres",
    "form_temp",
    "jpump_tvd",
    "tubing_od",
    "tubing_thickness",
    "casing_od",
    "casing_thickness",
    "form_wc",
    "form_gor",
    "surf_pres",
    "qwf",  # TOTAL LIQUID (BLPD) — see module docstring
    "pwf",
)

# Optional floats: missing/NaN must round-trip to None (not 0.0), so the
# library falls back to its field_model presets / defaults.
_OPTIONAL_FLOAT_FIELDS = (
    "jpump_md",
    "oil_api",
    "gas_sg",
    "wat_sg",
    "bubble_point",
    "ppf_surf_well",
    "knz_well",
    "ken_well",
    "kth_well",
    "kdi_well",
)

# As-reviewed oil rate (BOPD) kept alongside the total-liquid qwf for audit.
OIL_RATE_FIELD = "qwf_oil_review"

_STRING_FIELDS = (
    "well_name",
    "field_model",
    "jpump_direction",
    "review_nozzle",
    "review_throat",
    "ipr_source",
    "bhp_source",
    "gauge_note",
    "notes",
)

_BOOL_FIELDS = ("is_hypothetical", "reviewed", "offline")

# Provenance vocabularies (kept small + explicit so the UI can badge them).
IPR_SOURCES = ("vogel", "single_test", "forced", "hypothetical")
BHP_SOURCES = ("gauged", "assumed")

# Ordered CSV columns. well_name first; provenance/meta last.
CSV_COLUMNS = (
    ("well_name",)
    + _WELLCONFIG_FLOAT_FIELDS
    + (OIL_RATE_FIELD,)
    + _OPTIONAL_FLOAT_FIELDS
    + (
        "field_model",
        "jpump_direction",
        "review_nozzle",
        "review_throat",
        "ipr_source",
        "bhp_source",
        "gauge_note",
        "is_hypothetical",
        "reviewed",
        "offline",
        "notes",
    )
)

# WellConfig hard validation bounds (mirror network_optimizer.WellConfig.__post_init__)
# so a snapshot can never construct an invalid config.
_WELLCONFIG_BOUNDS = {
    "res_pres": (400.0, 5000.0),
    "form_temp": (32.0, 350.0),
    "jpump_tvd": (2500.0, 8000.0),
    "form_wc": (0.0, 1.0),
}

# The oil optimizer converts qwf (liquid) back to oil via oil = qwf * (1 - wc);
# above this water cut the round trip degenerates (at wc=1.0 the well's oil is
# identically zero and every config NaNs — the well then LOOKS "optimizer shut
# in" when really it was never modelable). Such wells must be saved offline
# (dewatering) instead of silently corrupting the plan. See P0-2 in
# docs/code_review_2026-07-01.md (the S-03 bug).
MAX_MODELABLE_WC = 0.99


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _opt_float(raw: Any) -> Optional[float]:
    """Coerce a CSV cell to float, mapping None/NaN/blank/non-numeric -> None."""
    if raw is None:
        return None
    try:
        if pd.isna(raw):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(raw, str) and not raw.strip():
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _str(raw: Any, default: str = "") -> str:
    if raw is None:
        return default
    try:
        if pd.isna(raw):
            return default
    except (TypeError, ValueError):
        pass
    return str(raw)


def _nozzle_str(raw: Any) -> str:
    """Normalize a nozzle label that may have been read from CSV as a float
    ('10.0' -> '10'), so it matches the integer-string NOZZLE_OPTIONS."""
    s = _str(raw).strip()
    if not s:
        return ""
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return s


def _direction(raw: Any) -> str:
    """Normalize a circulation direction to 'reverse'/'forward'.

    Sidebar radio values are 'Reverse'/'Forward'; older stores/CSVs have no
    column at all. Anything unrecognized falls back to 'reverse' (the
    standard configuration) — same default WellConfig applies.
    """
    d = _str(raw).strip().lower()
    return d if d in ("reverse", "forward") else "reverse"


def _bool(raw: Any, default: bool = False) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return default
    try:
        if pd.isna(raw):
            return default
    except (TypeError, ValueError):
        pass
    if isinstance(raw, str):
        return raw.strip().lower() in {"true", "1", "yes", "y"}
    try:
        return bool(int(raw))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Snapshot: SimulationParams -> store entry
# ---------------------------------------------------------------------------


def snapshot_from_params(
    params,
    *,
    ipr_source: str,
    bhp_source: str = "gauged",
    is_hypothetical: bool = False,
    offline: bool = False,
    gauge_note: str = "",
    jpump_md: Optional[float] = None,
    pin_pf_pressure: bool = True,
    notes: str = "",
) -> dict:
    """Snapshot the reviewed sidebar state into a serializable per-well entry.

    Args:
        params: the live ``SimulationParams`` for the well being reviewed.
        ipr_source: provenance tag (see ``IPR_SOURCES``) — how the IPR anchor
            was established (Vogel fit, single test, forced best-guess, or a
            fully synthetic hypothetical well).
        bhp_source: ``"gauged"`` (a measured BHP backed the match) or
            ``"assumed"`` (no gauge — pwf is an engineering estimate).
        is_hypothetical: a future/what-if well with no Databricks backing.
        jpump_md: jet-pump measured depth if known (defaults to TVD downstream).
        pin_pf_pressure: when True (default) the reviewed surface PF pressure is
            stored as the well's ``ppf_surf_well`` override, pinning the well to
            the pressure it was matched at. The Phase-B pad pump curve overrides
            this pad-wide with the common header pressure; the reviewed
            ``ken/kth/kdi`` are always kept.
        notes: free-text engineer note carried into the CSV.

    Returns:
        A plain dict keyed by ``CSV_COLUMNS``.
    """
    if ipr_source not in IPR_SOURCES:
        raise ValueError(f"ipr_source must be one of {IPR_SOURCES}, got {ipr_source!r}")
    if bhp_source not in BHP_SOURCES:
        raise ValueError(f"bhp_source must be one of {BHP_SOURCES}, got {bhp_source!r}")

    wc = float(params.form_wc)
    qwf_oil = float(params.qwf)
    if wc > MAX_MODELABLE_WC:
        if not offline:
            raise ValueError(
                f"{params.selected_well}: water cut {wc:.0%} is above the "
                f"{MAX_MODELABLE_WC:.0%} the oil optimizer can model — its oil "
                "rate degenerates to zero and the well would falsely show as "
                "'optimizer shut in'. Save it as Offline (dewatering) to keep it "
                "accounted for on the pad, or lower the water cut."
            )
        # Offline dewatering well: in water-pump-mode reviews the sidebar qwf IS
        # the water/total rate, so carry it as the liquid rate (oil ~ 0). Never
        # fed to the optimizer (active_entries filters offline).
        qwf_liquid = qwf_oil
    else:
        # OIL (BOPD) -> TOTAL LIQUID (BLPD). Exact inverse of the optimizer's
        # oil = qwf * (1 - wc), so the engineer's reviewed oil rate survives the
        # round trip bit-for-bit.
        qwf_liquid = qwf_oil / (1.0 - wc)

    return {
        "well_name": params.selected_well,
        "res_pres": float(params.pres),
        "form_temp": float(params.form_temp),
        "jpump_tvd": float(params.jpump_tvd),
        "tubing_od": float(params.tubing_od),
        "tubing_thickness": float(params.tubing_thickness),
        "casing_od": float(params.casing_od),
        "casing_thickness": float(params.casing_thickness),
        "form_wc": wc,
        "form_gor": float(params.form_gor),
        "surf_pres": float(params.surf_pres),
        "qwf": qwf_liquid,
        "pwf": float(params.pwf),
        OIL_RATE_FIELD: qwf_oil,
        "jpump_md": float(jpump_md) if jpump_md is not None else None,
        "oil_api": params.oil_api,
        "gas_sg": params.gas_sg,
        "wat_sg": params.wat_sg,
        "bubble_point": params.bubble_point,
        # Per-well calibration overrides. knz is fixed (0.01) in the friction
        # calibration and has no sidebar field, so it stays None (library default).
        "ppf_surf_well": float(params.ppf_surf) if pin_pf_pressure else None,
        "knz_well": None,
        "ken_well": float(params.ken),
        "kth_well": float(params.kth),
        "kdi_well": float(params.kdi),
        # Circulation direction from the sidebar radio (live-PF-seeded on well
        # selection): "reverse" = PF down the annulus, "forward" = PF down the
        # tubing (e.g. MPS-17). Carried into WellConfig so the optimizer's
        # BatchPump models the correct conduits.
        "jpump_direction": _direction(getattr(params, "jpump_direction", None)),
        # Review metadata. The optimizer re-chooses nozzle/throat, so the
        # reviewed pump is informational (drives the "reviewed vs optimized"
        # comparison in Results), not a constraint.
        "field_model": params.field_model,
        "review_nozzle": params.nozzle_no,
        "review_throat": params.area_ratio,
        "ipr_source": ipr_source,
        "bhp_source": bhp_source,
        # Free-text provenance, e.g. "Gauge-backed IPR — <window>, <n> samples".
        # Non-empty ⇒ the BHP/IPR was created from memory-gauge data.
        "gauge_note": gauge_note,
        "is_hypothetical": bool(is_hypothetical),
        # offline = pulled / down: kept in the store (accounted for) but excluded
        # from the optimization run by the host page.
        "offline": bool(offline),
        "reviewed": True,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Entry -> WellConfig (for the optimizer)
# ---------------------------------------------------------------------------


def to_well_config(entry: dict) -> WellConfig:
    """Build the optimizer's ``WellConfig`` from a reviewed entry.

    Float fields with hard library bounds are clamped so a slightly-out-of-range
    reviewed value (or an edited CSV) can't trip ``WellConfig.__post_init__``.
    """
    wc_val = float(entry["form_wc"])
    if wc_val > MAX_MODELABLE_WC and not entry.get("offline"):
        raise ValueError(
            f"{entry.get('well_name', '?')}: water cut {wc_val:.0%} exceeds the "
            f"{MAX_MODELABLE_WC:.0%} the oil optimizer can model — mark the well "
            "offline (dewatering) or lower the water cut."
        )

    clamped = {}
    for fld in _WELLCONFIG_FLOAT_FIELDS:
        val = float(entry[fld])
        if fld in _WELLCONFIG_BOUNDS:
            lo, hi = _WELLCONFIG_BOUNDS[fld]
            val = _clamp(val, lo, hi)
        clamped[fld] = val

    return WellConfig(
        well_name=entry["well_name"],
        res_pres=clamped["res_pres"],
        form_temp=clamped["form_temp"],
        jpump_tvd=clamped["jpump_tvd"],
        jpump_md=entry.get("jpump_md"),
        tubing_od=clamped["tubing_od"],
        tubing_thickness=clamped["tubing_thickness"],
        casing_od=clamped["casing_od"],
        casing_thickness=clamped["casing_thickness"],
        form_wc=clamped["form_wc"],
        form_gor=clamped["form_gor"],
        field_model=entry["field_model"],
        surf_pres=clamped["surf_pres"],
        qwf=clamped["qwf"],
        pwf=clamped["pwf"],
        oil_api=entry.get("oil_api"),
        gas_sg=entry.get("gas_sg"),
        wat_sg=entry.get("wat_sg"),
        bubble_point=entry.get("bubble_point"),
        ppf_surf_well=entry.get("ppf_surf_well"),
        knz_well=entry.get("knz_well"),
        ken_well=entry.get("ken_well"),
        kth_well=entry.get("kth_well"),
        kdi_well=entry.get("kdi_well"),
        jpump_direction=_direction(entry.get("jpump_direction")),
    )


def store_to_well_configs(store: dict[str, dict]) -> list[WellConfig]:
    """Convert a whole store ({well_name: entry}) into a WellConfig list."""
    return [to_well_config(entry) for entry in store.values()]


def hypothetical_entry(
    name: str,
    *,
    field_model: str,
    res_pres: float,
    oil_bopd: float,
    pwf: float,
    form_wc: float,
    form_gor: float,
    form_temp: float,
    jpump_tvd: float,
    nozzle: str = "",
    throat: str = "",
    tubing_od: float = 4.5,
    tubing_thickness: float = 0.5,
    surf_pres: float = 210.0,
    jpump_direction: str = "reverse",
    offline: bool = False,
    notes: str = "hypothetical",
) -> dict:
    """Build a fully-synthetic hypothetical well entry.

    Single source of truth for BOTH hypothetical creation paths — the
    Review-stage form and the Configure-screen "new well from scratch (no
    analog)" form. ``nozzle``/``throat`` become the reviewed pump so the
    well is usable in fixed-pump tools (Base vs Future needs a pump);
    ``offline=True`` parks it as a FUTURE well (out of the base case and the
    optimization until brought online or picked as future).

    ``form_wc`` is capped at 0.99 — the store's oil↔total-liquid round trip
    degenerates at 1.0 (see MAX_MODELABLE_WC).
    """
    wc = min(float(form_wc), MAX_MODELABLE_WC)
    oil_frac = 1.0 - wc
    return {
        "well_name": name,
        "res_pres": float(res_pres),
        "form_temp": float(form_temp),
        "jpump_tvd": float(jpump_tvd),
        "tubing_od": float(tubing_od),
        "tubing_thickness": float(tubing_thickness),
        "casing_od": 6.875,
        "casing_thickness": 0.5,
        "form_wc": wc,
        "form_gor": float(form_gor),
        "surf_pres": float(surf_pres),
        "qwf": float(oil_bopd) / oil_frac,
        "pwf": float(pwf),
        OIL_RATE_FIELD: float(oil_bopd),
        "jpump_md": float(jpump_tvd),
        "oil_api": None,
        "gas_sg": None,
        "wat_sg": None,
        "bubble_point": None,
        "ppf_surf_well": None,
        "knz_well": None,
        "ken_well": None,
        "kth_well": None,
        "kdi_well": None,
        "jpump_direction": _direction(jpump_direction),
        "field_model": field_model,
        "review_nozzle": str(nozzle or ""),
        "review_throat": str(throat or ""),
        "ipr_source": "hypothetical",
        "bhp_source": "assumed",
        "gauge_note": "",
        "is_hypothetical": True,
        "offline": bool(offline),
        "reviewed": True,
        "notes": notes,
    }


def clone_entry(entry: dict, new_name: str, *, source_well: str) -> dict:
    """A placeholder copy of an existing well's reviewed entry.

    Backs the Configure-screen "add a placeholder well like X" flow: copies
    every physical/calibration field (IPR, geometry, friction coefs, PF
    direction, review pump), renames, and flags the copy hypothetical so
    provenance stays honest — it feeds the optimizer like a real well but
    can never be mistaken for one. The source entry is not touched.
    """
    import copy

    out = copy.deepcopy(entry)
    out["well_name"] = new_name
    out["is_hypothetical"] = True
    out["ipr_source"] = "hypothetical"
    out["bhp_source"] = "assumed"
    out["reviewed"] = True
    out["offline"] = False
    out["gauge_note"] = ""
    out["notes"] = f"placeholder — cloned from {source_well}"
    return out


def active_entries(store: dict[str, dict]) -> dict[str, dict]:
    """Entries that should feed the optimizer — excludes offline/pulled wells."""
    return {k: v for k, v in store.items() if not v.get("offline")}


def validate_store(store: dict[str, dict]) -> dict[str, list[str]]:
    """Per-well issues for a loaded/edited store, rendered after a CSV upload
    so holes can't silently become plausible defaults (P0-10): a missing
    res_pres used to coerce to 0.0 and then clamp to 400 psi in
    ``to_well_config`` — the run proceeded on garbage with no warning.
    """
    issues: dict[str, list[str]] = {}

    def add(wn: str, msg: str) -> None:
        issues.setdefault(wn, []).append(msg)

    for wn, e in store.items():
        wc = float(e.get("form_wc") or 0.0)
        for fld, (lo, hi) in _WELLCONFIG_BOUNDS.items():
            if fld == "form_wc":
                continue
            v = float(e.get(fld) or 0.0)
            if not (lo <= v <= hi):
                add(
                    wn,
                    f"{fld}={v:g} outside [{lo:g}, {hi:g}] — would be "
                    "silently clamped at run time",
                )
        for fld in ("tubing_od", "tubing_thickness", "casing_od", "casing_thickness"):
            if float(e.get(fld) or 0.0) <= 0:
                add(wn, f"{fld} is missing/zero")
        qwf = float(e.get("qwf") or 0.0)
        pwf = float(e.get("pwf") or 0.0)
        rp = float(e.get("res_pres") or 0.0)
        if qwf <= 0:
            add(wn, "qwf (total liquid) is missing/zero")
        if pwf <= 0:
            add(wn, "pwf is missing/zero")
        elif rp and pwf >= rp:
            add(
                wn,
                f"pwf={pwf:g} ≥ res_pres={rp:g} (degenerate IPR — the well "
                "would fail every pump combo)",
            )
        if wc > MAX_MODELABLE_WC and not e.get("offline"):
            add(
                wn,
                f"water cut {wc:.0%} is above the modelable "
                f"{MAX_MODELABLE_WC:.0%} — mark the well offline (dewatering)",
            )
        oil = e.get(OIL_RATE_FIELD)
        if oil is not None and qwf > 0 and 0 < wc <= MAX_MODELABLE_WC:
            expected = float(oil) / (1.0 - wc)
            if expected > 0 and abs(qwf - expected) / expected > 0.02:
                add(
                    wn,
                    f"qwf={qwf:g} inconsistent with oil={float(oil):g} at "
                    f"WC {wc:.2f} (expected ≈{expected:.0f}) — was the WC "
                    "edited without re-deriving qwf?",
                )
    return issues


def store_signature(store: dict[str, dict]) -> tuple:
    """Order-independent signature of the optimizer-relevant store state.

    Stamped into a run's meta so Results can detect that wells were added,
    edited, or toggled online/offline AFTER the run — without it, a well
    added post-run rendered as "Optimizer shut in" (P0-3). Only physical /
    calibration fields participate; notes, provenance, and the reviewed pump
    label don't change the optimization.
    """
    # jpump_direction changes the modeled conduits, so it participates.
    fields = (
        _WELLCONFIG_FLOAT_FIELDS
        + _OPTIONAL_FLOAT_FIELDS
        + ("field_model", "jpump_direction")
    )

    def _norm(v):
        if v is None:
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return str(v)
        return None if f != f else round(f, 6)

    return tuple(
        (wn,) + tuple(_norm(store[wn].get(f)) for f in fields) for wn in sorted(store)
    )


# ---------------------------------------------------------------------------
# CSV round-trip
# ---------------------------------------------------------------------------


def store_to_dataframe(store: dict[str, dict]) -> pd.DataFrame:
    """Serialize the store to a DataFrame with stable ``CSV_COLUMNS`` order."""
    rows = [{col: entry.get(col) for col in CSV_COLUMNS} for entry in store.values()]
    return pd.DataFrame(rows, columns=list(CSV_COLUMNS))


def dataframe_to_store(df: pd.DataFrame) -> dict[str, dict]:
    """Parse an uploaded CSV/DataFrame back into a store keyed by well_name.

    Tolerant of missing optional columns and NaN cells (optional floats and
    blank strings come back as None/"" rather than NaN). ``well_name`` is
    required; rows without one are skipped.
    """
    store: dict[str, dict] = {}
    for _, row in df.iterrows():
        well_name = _str(row.get("well_name")).strip()
        if not well_name:
            continue

        entry: dict[str, Any] = {"well_name": well_name}
        for fld in _WELLCONFIG_FLOAT_FIELDS:
            entry[fld] = float(_opt_float(row.get(fld)) or 0.0)
        entry[OIL_RATE_FIELD] = _opt_float(row.get(OIL_RATE_FIELD))
        for fld in _OPTIONAL_FLOAT_FIELDS:
            entry[fld] = _opt_float(row.get(fld))
        # Normalize case so a hand-edited "schrader"/"KUPARUK" can't trip
        # WellConfig.__post_init__ at run time.
        fm = (_str(row.get("field_model"), "Schrader") or "Schrader").strip().title()
        entry["field_model"] = fm if fm in ("Schrader", "Kuparuk") else "Schrader"
        entry["jpump_direction"] = _direction(row.get("jpump_direction"))
        entry["review_nozzle"] = _nozzle_str(row.get("review_nozzle"))
        entry["review_throat"] = _str(row.get("review_throat")).strip()
        entry["ipr_source"] = _str(row.get("ipr_source"), "vogel") or "vogel"
        entry["bhp_source"] = _str(row.get("bhp_source"), "gauged") or "gauged"
        entry["gauge_note"] = _str(row.get("gauge_note"))
        entry["is_hypothetical"] = _bool(row.get("is_hypothetical"))
        entry["offline"] = _bool(row.get("offline"))
        entry["reviewed"] = _bool(row.get("reviewed"), default=True)
        entry["notes"] = _str(row.get("notes"))
        store[well_name] = entry
    return store
