"""match_test: infer the anchor BHP from a test's PF rate with a fake solver.

The real jetpump_solver is replaced by an analytic pump so the seed scan,
the scaled Nelder-Mead and the branch handling run for real:

* SUBSONIC: the pump's suction is set by the IPR anchor and the discharge
  coefficients; the nozzle passes PF as sqrt(ppf_nozzle − psu). A test
  generated from a known (pwf*, kth*, kdi*) must be recovered.
* SONIC: psu is pinned on a floor regardless of coefficients - the fit
  keeps the seeds for kth/kdi and says so.
* FAILED: the solver never converges - a graded "failed" result, no raise.
"""

from __future__ import annotations

from math import sqrt

import pytest

import woffl.gui.gaugeless_match as gm

PRES = 1800.0
PNI = 4200.0  # nozzle-inlet pressure of the fake pump (PF + hydrostatic)


class _FakePump:
    def __init__(self, nozzle, throat, knz=0.01, ken=0.03, kth=0.3, kdi=0.3):
        self.nozzle, self.throat, self.knz = nozzle, throat, knz
        self.ken, self.kth, self.kdi = ken, kth, kdi
        self.dnz = 0.2099


class _FakeInflow:
    """Straight-line PI anchored on (oil, pwf) to PRES."""

    def __init__(self, oil, pwf, pres):
        self.pi = oil / (pres - pwf)
        self.pres = pres

    def oil(self, psu):
        return max(0.0, self.pi * (self.pres - psu))


def _make_inflow(oil, pwf):
    return _FakeInflow(oil, pwf, PRES)


def _subsonic_solver(*, jpump, ipr_su, **kwargs):
    """Suction = the drawdown the pump can pull at these discharge losses,
    balanced against the IPR: psu solves oil(psu) = demand(psu) where the
    pump's demand falls with psu and rises with kth/kdi (more loss = less
    lift = higher suction needed). Closed form for a linear IPR."""
    # pump demand curve: oil_demand = D0 − D1·psu − 400·(kth + kdi)
    d0, d1 = 900.0, 0.5
    loss = 400.0 * (jpump.kth + jpump.kdi)
    # oil(psu) = pi·(pres − psu)  ==  d0 − loss − d1·psu
    pi = ipr_su.pi
    psu = (d0 - loss - pi * PRES) / (d1 - pi) if abs(d1 - pi) > 1e-9 else 500.0
    psu = min(max(psu, 150.0), PRES - 50.0)
    oil = ipr_su.oil(psu)
    if oil <= 0:
        return float("nan"), False, float("nan"), float("nan"), float("nan"), 0.0
    wat = oil * 2.86  # a fixed 74 % water cut mixture
    qnz = 60.0 * sqrt(max(PNI - psu, 1.0))
    return psu, False, oil, wat, qnz, 0.3


def _sonic_solver(*, jpump, ipr_su, **kwargs):
    psu = 600.0
    oil = ipr_su.oil(psu)
    return psu, True, oil, oil * 2.86, 60.0 * sqrt(PNI - psu), 1.0


def _dead_solver(**kwargs):
    raise RuntimeError("no crossing")


def _run(monkeypatch, solver, *, oil, water, pf, seed_kth=0.30, seed_kdi=0.30):
    monkeypatch.setattr(gm, "JetPump", _FakePump)
    monkeypatch.setattr(gm, "jetpump_solver", solver)
    return gm.match_test(
        well_name="MPE-19",
        oil_test=oil,
        water_test=water,
        pf_test=pf,
        pres=PRES,
        make_inflow=_make_inflow,
        pwh=250.0,
        tsu=120.0,
        ppf_surf=3168.0,
        nozzle="11",
        throat="A",
        knz=0.01,
        ken=0.03,
        seed_kth=seed_kth,
        seed_kdi=seed_kdi,
        wellbore=None,
        wellprof=None,
        prop_su=None,
        prop_pf=None,
    )


def _truth(pwf, kth, kdi, oil):
    """The test a well with this (pwf, kth, kdi) would produce - generated
    through the same fake so the fit has an exact answer."""
    ipr = _FakeInflow(oil, pwf, PRES)
    pump = _FakePump("11", "A", kth=kth, kdi=kdi)
    psu, _s, o, w, q, _m = _subsonic_solver(jpump=pump, ipr_su=ipr)
    return psu, o, w, q


def test_recovers_the_bhp_that_made_the_test(monkeypatch):
    # a well whose anchor (171 BOPD at pwf*) and losses (kth*, kdi*) are
    # consistent: at pwf* the fake pump's suction IS pwf*
    kth_star, kdi_star = 0.35, 0.45
    # find the self-consistent pwf* for these losses by bisection on the fake
    lo, hi = 200.0, 1600.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        psu, *_ = _truth(mid, kth_star, kdi_star, 171.0)
        lo, hi = (mid, hi) if psu > mid else (lo, mid)
    pwf_star = 0.5 * (lo + hi)
    psu, oil_t, wat_t, pf_t = _truth(pwf_star, kth_star, kdi_star, 171.0)
    assert abs(psu - pwf_star) < 0.5  # self-consistent by construction

    res = _run(monkeypatch, _subsonic_solver, oil=oil_t, water=wat_t, pf=pf_t)
    assert res.match_quality == "good", (res.score, res.message)
    assert res.converged and not res.sonic
    assert res.pwf == pytest.approx(pwf_star, abs=15.0)
    assert res.modeled_bhp == pytest.approx(res.pwf, abs=15.0)
    assert res.modeled_oil == pytest.approx(oil_t, rel=0.02)
    assert res.modeled_pf == pytest.approx(pf_t, rel=0.02)
    # the anchor the client applies: TOTAL liquid and the test's water cut
    assert res.qwf_liq == pytest.approx(oil_t + wat_t)
    assert res.form_wc == pytest.approx(wat_t / (oil_t + wat_t))
    assert res.ken == 0.03 and res.knz == 0.01  # held
    assert res.seed_pwf is not None and len(res.scan) == gm.SCAN_POINTS
    assert "nozzle" in res.caveat


def test_seed_comes_from_the_pf_crossing(monkeypatch):
    # the scan alone (before Nelder-Mead) lands near the PF crossing
    pwf_star, kth_star, kdi_star = 900.0, 0.30, 0.30
    _psu, oil_t, wat_t, pf_t = _truth(pwf_star, kth_star, kdi_star, 171.0)
    res = _run(monkeypatch, _subsonic_solver, oil=oil_t, water=wat_t, pf=pf_t)
    # the scan brackets the test PF, so the interpolated seed is inside the
    # window and the PF at the seed is close to the test
    assert gm.PWF_FLOOR < res.seed_pwf < PRES
    pfs = [s["pf"] for s in res.scan if s["pf"] is not None]
    assert min(pfs) < pf_t < max(pfs)


def test_sonic_keeps_seed_coefficients_and_says_so(monkeypatch):
    res = _run(monkeypatch, _sonic_solver, oil=171.0, water=489.0, pf=2863.0, seed_kth=0.33, seed_kdi=0.44)
    assert res.sonic is True
    assert (res.kth, res.kdi) == (0.33, 0.44)
    assert res.message and "choked" in res.message
    # the BHP is still the one thing identified: the floor
    assert res.modeled_bhp == pytest.approx(600.0)


def test_dead_solver_is_a_failed_grade_not_a_raise(monkeypatch):
    res = _run(monkeypatch, _dead_solver, oil=171.0, water=489.0, pf=2863.0)
    assert res.match_quality == "failed"
    assert res.converged is False
    assert res.message and "no operating point" in res.message
    assert all(s["pf"] is None for s in res.scan)


def test_rejects_a_test_without_oil_or_pf(monkeypatch):
    with pytest.raises(ValueError, match="positive oil"):
        _run(monkeypatch, _subsonic_solver, oil=0.0, water=100.0, pf=2000.0)
    with pytest.raises(ValueError, match="positive oil"):
        _run(monkeypatch, _subsonic_solver, oil=100.0, water=100.0, pf=0.0)


def test_grades():
    assert gm._grade(0.01) == "good"
    assert gm._grade(0.04) == "fair"
    assert gm._grade(0.2) == "poor"
    assert gm._grade(float("inf")) == "failed"


def test_unreachable_pf_is_not_an_identification(monkeypatch):
    # the fake nozzle passes at most ~60*sqrt(PNI - 150) = 3818 BWPD; ask
    # for far more and the BHP must NOT be reported as identified
    res = _run(monkeypatch, _subsonic_solver, oil=171.0, water=489.0, pf=6000.0)
    assert res.pf_reachable is False
    assert res.pf_model_max is not None and res.pf_model_max < 6000.0
    assert res.match_quality in ("poor", "failed")
    assert res.message and "cannot pass" in res.message and "not an identification" in res.message


def test_unreachable_pf_reports_the_area_factor_that_would_explain_it(monkeypatch):
    # Nozzle flow scales with area, so the factor that closes the PF gap at
    # the fitted point is the SQUARED rate ratio - the one hardware number
    # an engineer can check against the sidebar's 0.8 - 1.3 bound.
    res = _run(monkeypatch, _subsonic_solver, oil=171.0, water=489.0, pf=6000.0)
    assert res.pf_reachable is False
    assert res.modeled_pf > 0
    assert res.area_factor_needed == pytest.approx((6000.0 / res.modeled_pf) ** 2, rel=1e-9)
    assert res.area_factor_needed > 1.0  # the test wants MORE PF than the catalog nozzle passes


def test_reachable_pf_reports_no_area_factor(monkeypatch):
    _psu, oil_t, wat_t, pf_t = _truth(900.0, 0.30, 0.30, 171.0)
    res = _run(monkeypatch, _subsonic_solver, oil=oil_t, water=wat_t, pf=pf_t)
    assert res.pf_reachable is True
    assert res.area_factor_needed is None


def test_failed_match_reports_no_area_factor(monkeypatch):
    res = _run(monkeypatch, _dead_solver, oil=171.0, water=489.0, pf=6000.0)
    assert res.match_quality == "failed"
    assert res.area_factor_needed is None


def test_reachable_pf_keeps_the_grade(monkeypatch):
    _psu, oil_t, wat_t, pf_t = _truth(900.0, 0.30, 0.30, 171.0)
    res = _run(monkeypatch, _subsonic_solver, oil=oil_t, water=wat_t, pf=pf_t)
    assert res.pf_reachable is True
    assert res.pf_model_min <= pf_t <= res.pf_model_max
    assert res.match_quality == "good"


def test_reports_how_well_pf_resolves_bhp(monkeypatch):
    _psu, oil_t, wat_t, pf_t = _truth(900.0, 0.30, 0.30, 171.0)
    res = _run(monkeypatch, _subsonic_solver, oil=oil_t, water=wat_t, pf=pf_t)
    # the slope is taken against the ANCHOR pwf over the scan (the fake's
    # suction is the IPR/pump balance, so its sign is the fake's business);
    # what matters is that it is real and the resolution follows from it
    assert res.pf_per_100psi is not None and res.pf_per_100psi != 0
    assert res.bhp_resolution_psi is not None and res.bhp_resolution_psi > 0
    # resolution = GOOD_FRAC * pf / |slope|
    assert res.bhp_resolution_psi == pytest.approx(
        gm.GOOD_FRAC * pf_t / abs(res.pf_per_100psi / 100.0), rel=1e-6
    )


def test_weak_resolution_is_said_out_loud(monkeypatch):
    # a nozzle whose PF hardly moves with suction: 0.05 BWPD per psi
    def flat_solver(*, jpump, ipr_su, **kwargs):
        d0, d1 = 900.0, 0.5
        loss = 400.0 * (jpump.kth + jpump.kdi)
        pi = ipr_su.pi
        psu = (d0 - loss - pi * PRES) / (d1 - pi) if abs(d1 - pi) > 1e-9 else 500.0
        psu = min(max(psu, 150.0), PRES - 50.0)
        oil = ipr_su.oil(psu)
        return psu, False, oil, oil * 2.86, 2500.0 - 0.05 * psu, 0.3

    monkeypatch.setattr(gm, "JetPump", _FakePump)
    monkeypatch.setattr(gm, "jetpump_solver", flat_solver)
    res = gm.match_test(
        well_name="MPE-19", oil_test=171.0, water_test=489.0, pf_test=2470.0, pres=PRES,
        make_inflow=_make_inflow, pwh=250.0, tsu=120.0, ppf_surf=3168.0, nozzle="11", throat="A",
        knz=0.01, ken=0.03, seed_kth=0.3, seed_kdi=0.3, wellbore=None, wellprof=None,
        prop_su=None, prop_pf=None,
    )
    assert res.pf_reachable is True
    assert res.pf_per_100psi is not None
    assert res.bhp_resolution_psi == pytest.approx(
        gm.GOOD_FRAC * 2470.0 / abs(res.pf_per_100psi / 100.0), rel=1e-6
    )
    assert res.bhp_resolution_psi > gm.WEAK_RESOLUTION_PSI
    assert res.message and "barely sees the BHP" in res.message
