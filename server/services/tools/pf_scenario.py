"""PF Scenario - oil and BHP response to a power-fluid pressure change.

Port of woffl/gui/scotts_tools/pf_scenario.py (engine only).

Sibling of Header Impact: this sweeps power-fluid pressure (ppf_surf) holding
wellhead pressure fixed; that one sweeps wellhead pressure holding PF fixed.
IPR resolution is the three-tier chain both share: Vogel from a BHP gauge ->
JP back-calc for gaugeless wells -> jp_chars defaults.
"""


import numpy as np
import pandas as pd

from woffl.assembly.batchpump import BatchPump
from woffl.assembly.jp_history import get_current_pump, get_pump_at_date
from woffl.assembly.network_optimizer import WellConfig
from woffl.geometry.jetpump import JetPump
from woffl.geometry.pipe import Pipe, PipeInPipe
from woffl.assembly.sim_factories import create_pvt_components
from woffl.pvt.resmix import ResMix

from server.services.tools import _common
from server.services.tools._common import (
    build_well_config,
    casing_dims_from_chars,
    create_well_objects,
    fetch_well_tests_raw,
    friction_coefs_from_chars,
    get_latest_whp_per_well,
    get_vogel_for_wells,
    normalize_short_name,
    pad_from_mp_name,
)


def _resolve_pump_for_test(
    jp_hist: pd.DataFrame,
    well_name: str,
    test_date,
    current_pump: dict | None = None,
) -> tuple[dict | None, bool]:
    """Resolve the pump installed AT a historical test's date.

    [P1-29 fix] Historical BHP/production tests must be paired with the pump
    that was actually in the hole on the test date via ``get_pump_at_date``
    (Date Set -> next Date Set tenure; JPCOs are same-day pull+set so Date
    Pulled is never used — see ``jp_history.get_pump_at_date``), not today's
    current pump. Mirrors the pattern already used by ``jp_fric_trend.py`` /
    ``jp_washout.py`` (also duplicated in ``jp_calibration.py`` — each
    Scott's Tools tab is independently maintained).

    Falls back to ``current_pump`` (looked up via ``get_current_pump`` when
    not supplied) when no install record covers the test date.

    Returns ``(pump_dict_or_None, pump_changed)``. ``pump_changed`` is True
    when the resolved at-test pump differs from the well's current pump, or
    when the current pump can't be determined at all.
    """
    if current_pump is None:
        current_pump = get_current_pump(jp_hist, well_name)

    pump_at = get_pump_at_date(jp_hist, well_name, test_date)
    if not (pump_at and pump_at.get("nozzle_no") and pump_at.get("throat_ratio")):
        pump_at = current_pump

    if (
        pump_at is None
        or not pump_at.get("nozzle_no")
        or not pump_at.get("throat_ratio")
    ):
        return None, False

    if current_pump is None:
        return pump_at, True

    pump_changed = str(pump_at["nozzle_no"]) != str(
        current_pump.get("nozzle_no")
    ) or str(pump_at["throat_ratio"]) != str(current_pump.get("throat_ratio"))
    return pump_at, pump_changed


def _latest_bhp_with_date_per_well(
    months_back: int,
) -> dict[str, tuple[float, "pd.Timestamp"]]:
    """Return ``{well: (bhp, test_date)}`` for the well's latest measured-BHP test.

    ``get_latest_bhp_per_well`` (shared helper) returns just the BHP value,
    which is enough for the calibration target but not enough to resolve
    which pump was in the hole on that test's date. This local variant
    keeps the date alongside the value so the friction-coef calibration
    below can call ``get_pump_at_date`` / ``_resolve_pump_for_test`` — using
    today's current pump against a historical BHP test silently miscalibrates
    when the pump has been changed out since (see jp_fric_trend.py /
    jp_washout.py / jp_calibration.py).
    """
    raw = fetch_well_tests_raw(months_back)
    if raw is None or raw.empty or "BHP" not in raw.columns:
        return {}
    valid = raw.dropna(subset=["BHP"]).sort_values("WtDate")
    if valid.empty:
        return {}
    latest = valid.groupby("well").tail(1)
    return {
        row["well"]: (float(row["BHP"]), row["WtDate"]) for _, row in latest.iterrows()
    }


# ── CSV parsing ────────────────────────────────────────────────────────────


def _parse_scenario_csv(uploaded_file) -> pd.DataFrame | None:
    """Parse scenario CSV.

    Column A: well names  (e.g. B-30, I-15)
    Column B: Scenario A PF pressures (psi)
    Column C: Scenario B PF pressures (psi)

    Returns DataFrame with columns: short_name, well_name, pad, pf_pres_a, pf_pres_b
    """
    df = pd.read_csv(uploaded_file, header=None)
    if len(df.columns) < 3:
        raise ValueError(
            "CSV must have 3 columns: well name, Scenario A pressure, Scenario B pressure."
        )

    records = []
    notes: list[str] = []
    for _, row in df.iterrows():
        raw = str(row.iloc[0]).strip()
        if not raw or raw.lower() == "nan":
            continue
        try:
            pf_a = float(row.iloc[1])
            pf_b = float(row.iloc[2])
        except (ValueError, TypeError):
            notes.append(f"Skipped '{raw}': non-numeric PF pressure.")
            continue
        mp = normalize_short_name(raw)
        records.append(
            {
                "short_name": raw,
                "well_name": mp,
                "pad": pad_from_mp_name(mp),
                "pf_pres_a": pf_a,
                "pf_pres_b": pf_b,
            }
        )
    if not records:
        return None
    return pd.DataFrame(records)


# ── gaugeless BHP estimation ──────────────────────────────────────────────


def _discharge_residual_fixed_rate(
    psu,
    qoil_std,
    pwh,
    tsu,
    ppf_surf,
    jpump,
    wellbore,
    wellprof,
    prop_su,
    prop_pf,
    jpump_direction="reverse",
):
    """Jet pump discharge residual with a FIXED oil rate (no IPR).

    Mirrors solopump.discharge_residual but replaces the IPR lookup with
    a known oil rate so we can back-calculate suction pressure.
    """
    import woffl.assembly.solopump as so
    from woffl.flow import jetflow as jf
    from woffl.flow import outflow as of
    from woffl.flow import singlephase as sp
    from woffl.flow.jetplot import JetBook

    prod_path = "tubing" if jpump_direction == "reverse" else "annulus"
    pf_path = "annulus" if jpump_direction == "reverse" else "tubing"
    ate = jpump.ath - jpump.anz

    # Throat entry with fixed oil rate (replaces throat_entry_zero_tde)
    prop_c = prop_su.condition(psu, tsu)
    qtot = sum(prop_c.insitu_volm_flow(qoil_std))
    vte = sp.velocity(qtot, ate)
    te_book = JetBook(
        psu, vte, prop_c.rho_mix(), prop_c.cmix(), jf.enterance_ke(jpump.ken, vte)
    )

    pdec = 25
    while (te_book.tde_ray[-1] > 0) and (te_book.prs_ray[-1] > 50):
        pte_step = te_book.prs_ray[-1] - pdec
        prop_c = prop_c.condition(pte_step, tsu)
        qtot = sum(prop_c.insitu_volm_flow(qoil_std))
        vte = sp.velocity(qtot, ate)
        te_book.append(
            pte_step,
            vte,
            prop_c.rho_mix(),
            prop_c.cmix(),
            jf.enterance_ke(jpump.ken, vte),
        )

    pte, vte, rho_te, mach_te = te_book.dete_zero()

    # Power fluid iteration (identical to solopump.discharge_residual)
    dp_stat = sp.diff_press_static(prop_pf.density, -1 * wellprof.jetpump_vd)
    qpf_list = [2000.0, 3000.0]
    res_pf = []
    for qpf in qpf_list:
        r, vnz, pni = so.powerfluid_residual(
            qpf,
            pte,
            ppf_surf,
            tsu,
            dp_stat,
            jpump,
            wellbore,
            wellprof,
            prop_pf,
            pf_path,
        )
        res_pf.append(r)
    from woffl.flow.errors import ConvergenceError

    n = 0
    while abs(res_pf[-1]) > 5:
        qpf = so.qpf_secant(qpf_list[-2], qpf_list[-1], res_pf[-2], res_pf[-1])
        r, vnz, pni = so.powerfluid_residual(
            qpf,
            pte,
            ppf_surf,
            tsu,
            dp_stat,
            jpump,
            wellbore,
            wellprof,
            prop_pf,
            pf_path,
        )
        qpf_list.append(qpf)
        res_pf.append(r)
        n += 1
        if n == 20:  # uncapped, this loop could hang the whole app process
            raise ConvergenceError("power fluid rate did not converge")

    qnz = qpf_list[-1]
    wc_tm, _ = jf.throat_wc(qoil_std, prop_su.wc, qnz)

    prop_tm = ResMix(wc_tm, prop_su.fgor, prop_su.oil, prop_su.wat, prop_su.gas)
    ptm = jf.throat_discharge(
        pte, tsu, jpump.kth, vnz, jpump.anz, prop_pf.density, vte, ate, rho_te, prop_tm
    )
    _, pdi_jp = jf.diffuser_discharge(
        ptm, tsu, jpump.kdi, jpump.ath, wellbore.inn_pipe.inn_area, qoil_std, prop_tm
    )

    # Outflow
    _, prs_ray, _ = of.production_top_down_press(
        pwh, tsu, qoil_std, prop_tm, wellbore, wellprof, prod_path
    )
    pdi_of = prs_ray[-1]

    return pdi_jp - pdi_of


def _estimate_bhp(
    qoil_std,
    wc,
    fgor,
    pwh,
    tsu,
    ppf_surf,
    jpump,
    wellbore,
    wellprof,
    prop_pf,
    field_model="Schrader",
    psu_max=1800.0,
    jpump_direction="reverse",
):
    """Estimate suction pressure (BHP) from known production + jet pump conditions.

    Binary-searches suction pressure until the jet pump discharge matches
    the outflow requirement at the known oil rate. Returns None when the
    bracket does not contain a root: the pump cannot lift the rate at any
    suction, OR the residual is positive across the whole bracket. The latter
    used to return the bracket EDGE (100 psi) as if it were a solved BHP,
    and the caller then seeded a synthetic Vogel anchored at 100 psi
    (review 2026-09-01, EVID-F25) - a wall is not an estimate.
    """
    oil, water, gas = create_pvt_components(field_model)
    prop_su = ResMix(wc=wc, fgor=fgor, oil=oil, wat=water, gas=gas)

    psu_lo, psu_hi = 100.0, psu_max

    def _safe_residual(psu):
        try:
            return _discharge_residual_fixed_rate(
                psu,
                qoil_std,
                pwh,
                tsu,
                ppf_surf,
                jpump,
                wellbore,
                wellprof,
                prop_su,
                prop_pf,
                jpump_direction=jpump_direction,
            )
        except Exception:
            return None

    res_lo = _safe_residual(psu_lo)
    res_hi = _safe_residual(psu_hi)
    if res_lo is None or res_hi is None:
        return None
    # Both negative → pump can't lift this rate at any psu
    if res_lo < 0 and res_hi < 0:
        return None
    # Both positive → no root in the bracket; the floor is psu_min from the
    # throat-entry sweep, not this bracket's lower edge. Not an estimate.
    if res_lo > 0 and res_hi > 0:
        return None

    for _ in range(30):
        psu_mid = (psu_lo + psu_hi) / 2
        if abs(psu_hi - psu_lo) < 5:
            return psu_mid
        res_mid = _safe_residual(psu_mid)
        if res_mid is None:
            psu_hi = psu_mid
            continue
        # Residual (pdi_jp - pdi_of) is monotonically INCREASING in suction
        # pressure (same convention as solopump.discharge_residual): res<0 below
        # the root, res>0 above it. So a negative midpoint means the root lies
        # ABOVE psu_mid -> raise the lower bound; positive -> lower the upper
        # bound. (Previously these were swapped, collapsing the solve onto a
        # bracket endpoint instead of the root.)
        if res_mid < 0:
            psu_lo = psu_mid
        else:
            psu_hi = psu_mid

    return (psu_lo + psu_hi) / 2


def _estimate_gaugeless_ipr(
    missing_wells,
    months_back,
    test_pf_pres,
    jp_hist,
    jp_chars_dict,
    whp_map: dict[str, float] | None = None,
    pf_map: dict[str, float] | None = None,
):
    """For wells without BHP gauges, estimate BHP from production + jet pump physics.

    Returns dict of {well_name: synthetic_vogel_row} compatible with the
    Vogel dict used elsewhere.

    ``pf_map`` optionally supplies a per-well power-fluid pressure (psi); a well
    absent from it falls back to the scalar ``test_pf_pres``. The Header Pressure
    Impact tab passes per-pad PF here so the back-calc matches each pad's actual
    power-fluid pressure instead of one global value.
    """
    from woffl.assembly.network_optimizer import _load_well_profile

    try:
        raw_df = fetch_well_tests_raw(months_back)
    except Exception:
        return {}
    if raw_df is None or raw_df.empty:
        return {}

    whp_map = whp_map or {}
    results = {}
    for wn in missing_wells:
        well_tests = raw_df[raw_df["well"] == wn].copy()
        # Keep only rows that have production but NO BHP
        well_tests = well_tests[
            well_tests["BHP"].isna() & well_tests["WtTotalFluid"].notna()
        ]
        if well_tests.empty:
            continue

        # Most recent test
        latest = well_tests.sort_values("WtDate", ascending=False).iloc[0]
        oil_rate = latest.get("WtOilVol", 0)
        total_fluid = latest["WtTotalFluid"]
        if total_fluid <= 0 or pd.isna(oil_rate) or oil_rate <= 0:
            continue

        wc = latest.get("form_wc", np.nan)
        if pd.isna(wc) and total_fluid > 0:
            water = latest.get("WtWaterVol", 0) or 0
            wc = water / total_fluid if total_fluid > 0 else 0.5
        fgor = latest.get("fgor", 250)
        if pd.isna(fgor):
            fgor = 250

        # Get pump config — [P1-29 fix] resolve the pump that was actually
        # installed AT this test's date, not today's current pump. The
        # back-calculated BHP is a physics inversion through the jet pump's
        # own nozzle/throat areas at the time production was measured; using
        # today's (possibly changed-out) pump silently poisons the estimate.
        pump, pump_changed = _resolve_pump_for_test(jp_hist, wn, latest["WtDate"])
        if pump is None or pump["nozzle_no"] is None:
            continue

        # Get well geometry
        chars = jp_chars_dict.get(wn)
        if not chars:
            continue

        # Per-well friction coefficients (vw_prop_mech), fall back to JetPump defaults
        fric_coefs = friction_coefs_from_chars(chars)
        jpump = JetPump(pump["nozzle_no"], pump["throat_ratio"], **fric_coefs)

        def _cnum(key, default):
            """NaN-safe numeric lookup — Databricks chars carry missing values
            as NaN under a *present* key, so dict.get's default never fires and
            float(nan) silently poisons the solve (NaN Pipe areas -> NaN
            residual -> _estimate_bhp's bound checks skipped)."""
            v = chars.get(key)
            if v is None:
                return default
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return default
            return default if pd.isna(fv) else fv

        is_sch = chars.get("is_sch", True)
        if isinstance(is_sch, str):
            is_sch = is_sch.lower() in ("true", "1", "yes")
        elif pd.isna(is_sch):
            is_sch = True
        fm = "Schrader" if is_sch else "Kuparuk"
        tsu = _cnum("form_temp", 75 if is_sch else 170)
        jpump_md = _cnum("JP_MD", _cnum("JP_TVD", 4000))

        tube = Pipe(out_dia=_cnum("out_dia", 4.5), thick=_cnum("thick", 0.271))
        casing_od, casing_thk = casing_dims_from_chars(chars)
        case = Pipe(out_dia=casing_od, thick=casing_thk)
        wellbore = PipeInPipe(inn_pipe=tube, out_pipe=case)
        well_profile = _load_well_profile(wn, jpump_md, fm)
        _, water_obj, _ = create_pvt_components(fm)
        prop_pf = water_obj.condition(0, 60)

        # Per-well WHP: use the test row's whp first (most consistent with the
        # BHP we're back-calculating), then map fallback, then 210
        test_whp = latest.get("whp")
        if test_whp is None or pd.isna(test_whp):
            pwh = float(whp_map.get(wn, 210.0))
        else:
            pwh = float(test_whp)

        # Use oil rate (not total fluid) as qoil_std for the solver
        psu_max = _cnum("res_pres", 1800)
        pf_for_well = float((pf_map or {}).get(wn, test_pf_pres))
        bhp_est = _estimate_bhp(
            oil_rate,
            wc,
            fgor,
            pwh,
            tsu,
            pf_for_well,
            jpump,
            wellbore,
            well_profile,
            prop_pf,
            field_model=fm,
            psu_max=psu_max,
            jpump_direction=_common.detect_jpump_direction(wn),  # EVID-F22
        )
        if bhp_est is None:
            continue

        # Estimate reservoir pressure as BHP + cushion (simple heuristic)
        res_pres = max(bhp_est + 200, psu_max)

        results[wn] = {
            "ResP": res_pres,
            "qwf": total_fluid,
            "pwf": bhp_est,
            "form_wc": round(wc, 3),
            "fgor": fgor,
            "QMax_recent": 0,
            "QMax_lowest_bhp": 0,
            "QMax_median": 0,
            "num_tests": len(well_tests),
            "R2": 0,
            "_bhp_estimated": True,
            # [P1-29 guard] surfaced by render_tab so the engineer knows this
            # well's back-calc used the pump that WAS in the hole at test
            # time, which differs from what's in the well today.
            "_pump_changed": pump_changed,
        }

    return results


# ── solver ────────────────────────────────────────────────────────────────


def _solve_at_pf(
    wc: WellConfig,
    well_objects: tuple,
    nozzle: str,
    throat: str,
    pf_pres: float,
    fric_coefs: dict | None = None,
) -> dict:
    """Run solver for one well / one pump / one PF pressure.

    Args:
        fric_coefs: Optional {knz, ken, kth, kdi} from vw_prop_mech.
            Missing keys fall through to JetPump class defaults.
    """
    wellbore, well_profile, inflow, res_mix, prop_pf = well_objects
    jp = JetPump(nozzle, throat, **(fric_coefs or {}))
    batch = BatchPump(
        pwh=wc.surf_pres,
        tsu=wc.form_temp,
        ppf_surf=pf_pres,
        wellbore=wellbore,
        wellprof=well_profile,
        ipr_su=inflow,
        prop_su=res_mix,
        prop_pf=prop_pf,
        jpump_direction=wc.jpump_direction,
        wellname=wc.well_name,
    )
    result_df = batch.batch_run([jp])
    if result_df.empty:
        return dict(oil=np.nan, pf_rate=np.nan, psu=np.nan, sonic=False, mach=np.nan)
    r = result_df.iloc[0]
    return dict(
        oil=float(r["qoil_std"]) if pd.notna(r["qoil_std"]) else np.nan,
        pf_rate=float(r["lift_wat"]) if pd.notna(r["lift_wat"]) else np.nan,
        psu=float(r["psu_solv"]) if pd.notna(r["psu_solv"]) else np.nan,
        sonic=bool(r["sonic_status"]) if pd.notna(r.get("sonic_status")) else False,
        mach=float(r["mach_te"]) if pd.notna(r.get("mach_te")) else np.nan,
    )


# ── main tab ───────────────────────────────────────────────────────────────


# ── results display ───────────────────────────────────────────────────────




def compare_one(
    well_name: str,
    chars: dict,
    pump: dict,
    vogel_row: dict | None,
    whp: float,
    pf_a: float,
    pf_b: float,
) -> dict | None:
    """One well at two PF pressures. Module-level and picklable: pool worker.

    Lifted from the tab's render loop. Returns None when the well cannot be
    set up or solved, so one bad well drops out instead of failing the run.
    """
    base = {"Well": well_name, "Pad": _common.pad_from_mp_name(well_name)}
    try:
        wc = _common.build_well_config(
            well_name, {well_name: chars}, vogel_row, surf_pres=whp
        )
        well_objs = _common.create_well_objects(wc)
    except Exception as exc:  # noqa: BLE001 - a skipped well must SAY why
        return {**base, "Status": "error", "Error": f"setup: {str(exc)[:160]}"}

    nozzle = str(pump["nozzle_no"])
    throat = str(pump["throat_ratio"])
    fric = _common.friction_coefs_from_chars(chars)
    try:
        res_a = _solve_at_pf(wc, well_objs, nozzle, throat, float(pf_a), fric)
        res_b = _solve_at_pf(wc, well_objs, nozzle, throat, float(pf_b), fric)
    except Exception as exc:  # noqa: BLE001 - a skipped well must SAY why
        return {**base, "Status": "error", "Error": f"solver: {str(exc)[:160]}"}

    return {
        **base,
        "Status": "ok",
        "Error": "",
        "Nozzle": nozzle,
        "Throat": throat,
        "WHP": int(round(whp)),
        "kth": fric.get("kth"),
        "kdi": fric.get("kdi"),
        "PfA": int(round(pf_a)),
        "OilA": res_a["oil"],
        "PfRateA": res_a["pf_rate"],
        "BhpA": res_a["psu"],
        "SonicA": res_a["sonic"],
        "MachA": res_a["mach"],
        "PfB": int(round(pf_b)),
        "OilB": res_b["oil"],
        "PfRateB": res_b["pf_rate"],
        "BhpB": res_b["psu"],
        "SonicB": res_b["sonic"],
        "MachB": res_b["mach"],
        "DeltaOil": res_b["oil"] - res_a["oil"],
        "DeltaPfRate": res_b["pf_rate"] - res_a["pf_rate"],
        "DeltaBhp": res_b["psu"] - res_a["psu"],
    }
