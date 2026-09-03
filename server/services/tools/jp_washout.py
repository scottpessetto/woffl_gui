"""JP Wash-Out detection - which pumps need more PF pressure than we have.

Port of the retired Streamlit tool.

The question: for every JP producer whose latest test measured ``lift_wat``,
what power-fluid SURFACE pressure would the model need to produce that much
nozzle flow, at the well's own friction coefficients and the pump that was
installed WHEN THE TEST RAN? A pump that has washed out passes more water
than its catalog throat should, so the pressure required to reproduce the
measurement drops well below the pad's actual PF - or, read the other way, a
pump needing MORE than the surface infrastructure can deliver (default
3,400 psi) is not going to make its numbers.

Two subtleties carried over verbatim because both were bug fixes:

* The pump used is ``get_pump_at_date`` for the TEST's date, not the current
  one - the lift water was measured through whatever was in the hole then.
* ``PumpChangedSinceTest`` is propagated into the result row. It used to be
  computed and dropped, so a well whose pump had already been swapped since
  the scanned test was still flagged for a changeout that had happened.

Fan-out is the shared process pool, so a fleet scan uses both vCPUs and does
not hold the GIL against the rest of the app.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from server import pool
from server.services import datasources
from server.services import tests as tests_svc
from server.services.tools import _common

log = logging.getLogger("woffl.web.tools.washout")

# Surface PF infrastructure cap: above this the pump cannot be fed hard
# enough to make the observed rate, so the pump (not the pressure) is the
# problem. The tab exposed it as an editable threshold; so does the page.
DEFAULT_PPF_LIMIT = 3400.0

# Required-PF over test-day-PF above which a pump is flagged as washed
# (needs this much more pressure than it actually ran at to make its lift
# water). 15% clears allocation noise on lift_wat and PF gauge scatter.
WASHOUT_RATIO = 1.15

DEFAULT_MONTHS = 6


def _build_scan_input(months_back: int) -> Optional[pd.DataFrame]:
    """Latest lift_wat-bearing test per current-JP well, ready to calibrate."""
    from woffl.assembly.jp_history import get_current_pump, get_pump_at_date

    raw = tests_svc.fetch_all_well_tests(months_back)
    if raw is None or raw.empty or "lift_wat" not in raw.columns:
        return None
    valid = raw.dropna(subset=["lift_wat"]).copy()
    valid = valid[valid["lift_wat"] > 0]
    if valid.empty:
        return None
    latest = valid.sort_values("WtDate").groupby("well").tail(1).copy()

    jp_hist, _source = datasources.jp_history_safe()
    if jp_hist is None or jp_hist.empty:
        return None

    chars_map = _common.well_chars_map()
    if not chars_map:
        return None

    # Vogel IPR per well: without it the calibration runs against the generic
    # WC=0.5 / GOR=250 / qwf=750 defaults and PpfRequired absorbs the IPR error.
    vogel_map = _common.get_vogel_for_wells(
        latest["well"].unique().tolist(), months_back
    )

    rows: list[dict[str, Any]] = []
    for _, t in latest.iterrows():
        wn = t["well"]
        pump = get_current_pump(jp_hist, wn)
        if pump is None or not pump.get("nozzle_no") or not pump.get("throat_ratio"):
            continue
        chars = chars_map.get(wn)
        if not chars:
            continue
        pump_at = get_pump_at_date(jp_hist, wn, t["WtDate"])
        if not (pump_at and pump_at.get("nozzle_no") and pump_at.get("throat_ratio")):
            pump_at = pump
        pump_changed = str(pump_at["nozzle_no"]) != str(pump["nozzle_no"]) or str(
            pump_at["throat_ratio"]
        ) != str(pump["throat_ratio"])
        whp = float(t["whp"]) if pd.notna(t.get("whp")) else 210.0
        rows.append(
            {
                "Well": wn,
                "Pad": _common.pad_from_mp_name(wn),
                "Pump": f"{pump_at['nozzle_no']}{pump_at['throat_ratio']}",
                "Nozzle": str(pump_at["nozzle_no"]),
                "Throat": str(pump_at["throat_ratio"]),
                "PumpChangedSinceTest": pump_changed,
                "_vogel": vogel_map.get(wn),
                "WtDate": t["WtDate"],
                "Oil": float(t.get("WtOilVol") or 0.0),
                "Water": float(t.get("WtWaterVol") or 0.0),
                "Gas": float(t.get("WtGasVol") or 0.0),
                "LiftWat": float(t["lift_wat"]),
                "WHP": whp,
                "BHP": float(t["BHP"]) if pd.notna(t.get("BHP")) else None,
                # The PF pressure the pump ACTUALLY ran at on test day (test-
                # day vw_pressure_daily join; PF_MIN_VALID filter). The
                # washout verdict compares required-vs-actual (EVID-F24).
                "PfAtTest": (
                    float(t["pf_press"])
                    if pd.notna(t.get("pf_press")) and float(t["pf_press"]) >= 800.0
                    else None
                ),
                "_chars": chars,
            }
        )
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values(["Pad", "Well"]).reset_index(drop=True)


def calibrate_one(row_dict: dict) -> dict:
    """One well's PF calibration. Module-level and picklable: pool worker.

    Errors are caught HERE so a single bad well cannot kill the scan - the
    row comes back with ``Status="error"`` and the rest of the fleet lands.
    """
    from woffl.assembly.pf_calibration import calibrate_pf_for_lift

    wn = row_dict["Well"]
    chars = row_dict["_chars"]
    pump_changed = bool(row_dict.get("PumpChangedSinceTest", False))
    base = {
        "Well": wn,
        "Pad": row_dict["Pad"],
        "Pump": row_dict["Pump"],
        "PumpChangedSinceTest": pump_changed,
        "WtDate": row_dict["WtDate"],
        "Oil": row_dict["Oil"],
        "Water": row_dict["Water"],
        "Gas": row_dict["Gas"],
        "LiftWat": row_dict["LiftWat"],
        "BHP": row_dict["BHP"],
        "WHP": row_dict["WHP"],
    }
    try:
        wc = _common.build_well_config(
            wn, {wn: chars}, row_dict.get("_vogel"), surf_pres=row_dict["WHP"]
        )
        wellbore, well_profile, inflow, res_mix, prop_pf = _common.create_well_objects(wc)
        cur = _common.friction_coefs_from_chars(chars)
        result = calibrate_pf_for_lift(
            well_name=wn,
            target_lift=row_dict["LiftWat"],
            pwh=row_dict["WHP"],
            tsu=wc.form_temp,
            nozzle=row_dict["Nozzle"],
            throat=row_dict["Throat"],
            knz=cur.get("knz", 0.01),
            ken=cur.get("ken", 0.03),
            kth=cur.get("kth", 0.30),
            kdi=cur.get("kdi", 0.30),
            wellbore=wellbore,
            wellprof=well_profile,
            ipr_su=inflow,
            prop_su=res_mix,
            prop_pf=prop_pf,
            jpump_direction=wc.jpump_direction,  # EVID-F22
        )
        return {
            **base,
            "PpfRequired": result.ppf_surf,
            "ModeledQnz": result.modeled_qnz,
            "LiftResidual": result.lift_residual,
            "Converged": result.converged,
            "Bounded": result.bounded,
            "Sonic": result.sonic,
            "Iterations": result.iterations,
            "Status": "ok",
            "Error": "",
        }
    except Exception as exc:  # noqa: BLE001 - one bad well is a row, not a 500
        return {
            **base,
            "PpfRequired": None,
            "ModeledQnz": None,
            "LiftResidual": None,
            "Converged": False,
            "Bounded": False,
            "Sonic": False,
            "Iterations": 0,
            "Status": "error",
            "Error": str(exc)[:200],
        }


def scan(months_back: int = DEFAULT_MONTHS, ppf_limit: float = DEFAULT_PPF_LIMIT) -> dict[str, Any]:
    """Scan the fleet for pumps that cannot make their measured lift water.

    Args:
        months_back: Test lookback window.
        ppf_limit: Surface PF cap, psi. Rows above it are flagged.

    Returns:
        dict: ``{"rows": [...], "flagged": n, "scanned": n, "errors": n,
        "ppf_limit": float, "months_back": int}``. Rows carry the full
        calibration outcome so the page can show why a well was flagged.
    """
    from server.services import frames

    inp = _build_scan_input(months_back)
    if inp is None or inp.empty:
        return {
            "rows": [], "flagged": 0, "scanned": 0, "errors": 0,
            "ppf_limit": ppf_limit, "months_back": months_back,
        }

    jobs = [(row.to_dict(),) for _, row in inp.iterrows()]
    results = pool.submit_all(calibrate_one, jobs)
    if results is None:  # no pool, or it broke - identical work, serially
        results = [calibrate_one(job[0]) for job in jobs]

    rows = frames.records(pd.DataFrame(results))
    for r in rows:
        ppf = r.get("PpfRequired")
        ok = r.get("Status") == "ok" and ppf is not None
        pf_at_test = r.get("PfAtTest")
        # Required-over-actual: the washout signal is a pump that needs MORE
        # PF than it was actually fed to make the lift water the test
        # allocated it. Flagging against a fixed 3,400 psi cap missed a
        # 2,200-psi-pad pump needing 3,000 (+36%) and flagged an M-pad pump
        # needing 3,450 (+1.5%) - review 2026-09-01, EVID-F24. The cap is
        # kept as its own column (infrastructure feasibility), not the flag.
        ratio = (
            float(ppf) / float(pf_at_test)
            if ok and pf_at_test not in (None, 0) and float(pf_at_test) > 0
            else None
        )
        r["PpfRatio"] = round(ratio, 3) if ratio is not None else None
        r["OverLimit"] = bool(ok and float(ppf) > ppf_limit)
        if ratio is not None:
            r["Flagged"] = bool(ratio > WASHOUT_RATIO)
            r["FlagBasis"] = "vs measured PF"
        else:
            r["Flagged"] = bool(r["OverLimit"])
            r["FlagBasis"] = "vs limit (no test-day PF)"
    rows.sort(
        key=lambda r: (r.get("PpfRequired") is None, -(r.get("PpfRequired") or 0.0))
    )
    return {
        "rows": rows,
        "flagged": sum(1 for r in rows if r["Flagged"]),
        "scanned": len(rows),
        "errors": sum(1 for r in rows if r.get("Status") == "error"),
        "ppf_limit": ppf_limit,
        "months_back": months_back,
    }
