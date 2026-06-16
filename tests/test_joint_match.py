"""Validation of woffl.gui.joint_match (joint oil + power-fluid auto-match).

Databricks isn't reachable offline, so we validate the ALGORITHM (not real wells)
by round-trip: pick truth params -> forward-solve to get (oil, pf[, bhp]) targets
-> ask joint_match to recover a match. The pytest functions guard the core
behaviours; the ``__main__`` block prints a verbose table for manual runs.

    WOFFL_MAX_WORKERS=1 PYTHONPATH=. ./venv/Scripts/python.exe -m pytest tests/test_joint_match.py
    WOFFL_MAX_WORKERS=1 PYTHONPATH=. ./venv/Scripts/python.exe -m tests.test_joint_match
"""

import os

os.environ.setdefault("WOFFL_MAX_WORKERS", "1")

from woffl.gui.joint_match import _inflow_from_qmax, joint_match  # noqa: E402
from woffl.gui.utils import (  # noqa: E402
    create_jetpump,
    create_pipes,
    create_reservoir_mix,
    create_well_profile,
    run_jetpump_solver,
)

FIELD = "schrader"
SURF, TEMP, RHO_PF = 250.0, 120.0, 62.4
_, _, WELLBORE = create_pipes()
WELLPROF = create_well_profile(field_model=FIELD, jpump_tvd=5000.0)


def _targets(qmax, ppf, nozzle, throat, wc, gor=800, pres=2000, ken=0.03,
             kth=0.3, kdi=0.4, direction="reverse"):
    """Forward-solve a 'truth' well -> (oil, pf, bhp) measured targets, or None."""
    rm = create_reservoir_mix(wc, gor, TEMP, FIELD)
    jp = create_jetpump(nozzle, throat, ken, kth, kdi)
    out = run_jetpump_solver(SURF, TEMP, RHO_PF, ppf, jp, WELLBORE, WELLPROF,
                             _inflow_from_qmax(qmax, pres), rm, field_model=FIELD,
                             jpump_direction=direction, quiet=True)
    if out is None:
        return None
    psu, sonic, qoil, _fwat, qnz, _mach = out
    return {"bhp": psu, "oil": qoil, "pf": qnz, "sonic": sonic}


def _match(oil, pf, nozzle, throat, wc, *, gor=800, pres=2000, bhp=None,
           direction="reverse", oil_mult=1.0, pf_mult=1.0):
    return joint_match(
        oil_target=oil * oil_mult, pf_target=pf * pf_mult, pres=pres,
        nozzle=nozzle, throat=throat, surf_pres=SURF, form_temp=TEMP,
        rho_pf=RHO_PF, ppf_surf0=2500.0, wellbore=WELLBORE, well_profile=WELLPROF,
        form_wc=wc, form_gor=gor, field_model=FIELD, bhp_target=bhp,
        jpump_direction=direction,
    )


# ── pytest guards ─────────────────────────────────────────────────────────

def test_roundtrip_reverse_matches():
    for qmax, ppf, nz, th, wc in [
        (1200, 3200, "12", "B", 0.5),
        (900, 2400, "12", "B", 0.5),
        (2200, 3600, "12", "B", 0.5),
        (1500, 3000, "14", "B", 0.5),
        (1000, 3200, "12", "B", 0.85),
    ]:
        t = _targets(qmax, ppf, nz, th, wc)
        assert t is not None
        r = _match(t["oil"], t["pf"], nz, th, wc)
        assert r.ok, f"{nz}{th} wc{wc}: {r.status} {r.diagnostic}"
        assert abs(r.oil_err_pct) <= 10 and abs(r.pf_err_pct) <= 10


def test_forward_circulating_matches():
    t = _targets(1200, 3200, "12", "B", 0.7, direction="forward")
    if t is None:
        return  # forward truth not solvable in this config — skip
    r = _match(t["oil"], t["pf"], "12", "B", 0.7, direction="forward")
    assert r.ok, f"forward: {r.status} {r.diagnostic}"


def test_bhp_as_third_target():
    """Gauged well: oil + PF + BHP all matched."""
    t = _targets(1200, 3200, "12", "B", 0.8)
    assert t is not None
    r = _match(t["oil"], t["pf"], "12", "B", 0.8, bhp=t["bhp"])
    assert r.ok
    assert r.bhp_err_pct is not None and abs(r.bhp_err_pct) <= 10


def test_marginal_pump_recovers():
    """Target generated from the newly-fixed marginal regime (12B, 94% WC)."""
    t = _targets(1000, 3000, "12", "B", 0.94)
    assert t is not None, "marginal truth should solve after the walk-inward fix"
    r = _match(t["oil"], t["pf"], "12", "B", 0.94)
    assert r.ok, f"marginal: {r.status} {r.diagnostic}"


def test_pf_too_high_is_flagged_not_silently_matched():
    """An unreachable PF target -> partial + a PF-limited diagnostic, not a fake
    'matched'. Guards the probe-based diagnostics."""
    t = _targets(1200, 3200, "12", "B", 0.5)
    assert t is not None
    r = _match(t["oil"], t["pf"], "12", "B", 0.5, pf_mult=4.0)
    assert not r.ok and r.status == "partial"
    assert "PF" in r.diagnostic


def test_oil_too_high_reports_pump_capacity():
    t = _targets(1200, 3200, "12", "B", 0.5)
    assert t is not None
    r = _match(t["oil"], t["pf"], "12", "B", 0.5, oil_mult=5.0)
    assert not r.ok
    assert "capacity" in r.diagnostic.lower() or "oil" in r.diagnostic.lower()


def test_batch_match_isolates_errors_and_sorts():
    """The batch core runs many wells, isolates a bad one as an 'error' row
    (never aborting the batch), and sorts worst-first for triage."""
    from woffl.gui.joint_match import batch_match, batch_summary

    t = _targets(1200, 3200, "12", "B", 0.7)
    assert t is not None
    good = dict(
        oil_target=t["oil"], pf_target=t["pf"], pres=2000, nozzle="12",
        throat="B", surf_pres=SURF, form_temp=TEMP, rho_pf=RHO_PF,
        ppf_surf0=2500, wellbore=WELLBORE, well_profile=WELLPROF,
        form_wc=0.7, form_gor=800, field_model=FIELD,
    )
    partial = dict(good, pf_target=t["pf"] * 4.0)   # unreachable PF -> partial
    bad = {k: v for k, v in good.items() if k != "nozzle"}  # missing kw -> error

    rows = batch_match([("GOOD", good), ("PARTIAL", partial), ("BAD", bad)])
    by = {r.well: r for r in rows}
    assert by["GOOD"].status == "matched" and by["GOOD"].ok
    assert by["PARTIAL"].status == "partial" and not by["PARTIAL"].ok
    assert by["BAD"].status == "error" and by["BAD"].result is None
    # worst-first ordering: error -> partial -> matched
    assert [r.well for r in rows] == ["BAD", "PARTIAL", "GOOD"]
    assert batch_summary(rows) == {"matched": 1, "partial": 1, "failed": 0, "error": 1}


# ── verbose manual runner ─────────────────────────────────────────────────

def _case(name, *, qmax, ppf, nozzle="12", throat="B", wc=0.5, gor=800,
          pres=2000, oil_mult=1.0, pf_mult=1.0):
    t = _targets(qmax, ppf, nozzle, throat, wc, gor, pres)
    if t is None:
        print(f"[{name}] truth did not solve — skip")
        return
    r = _match(t["oil"], t["pf"], nozzle, throat, wc, gor=gor, pres=pres,
               oil_mult=oil_mult, pf_mult=pf_mult)
    print(f"[{name}] truth oil={t['oil']:.0f} pf={t['pf']:.0f} -> {r.status} "
          f"oil_err={r.oil_err_pct:+.1f}% pf_err={r.pf_err_pct:+.1f}% "
          f"ppf={r.ppf_surf:.0f} iters={r.iterations}")
    print(f"      diag: {r.diagnostic}")


if __name__ == "__main__":
    print("=== joint_match synthetic validation ===\n")
    _case("roundtrip-mid", qmax=1200, ppf=3200)
    _case("roundtrip-high-oil", qmax=2200, ppf=3600)
    _case("roundtrip-highwc", qmax=1000, ppf=3200, wc=0.85)
    _case("pf-too-high", qmax=1200, ppf=3200, pf_mult=4.0)
    _case("oil-too-high", qmax=1200, ppf=3200, oil_mult=5.0)
    print("\n=== done ===")
