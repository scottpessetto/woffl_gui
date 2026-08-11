"""calibrate_friction_coefs branch behavior with a fake solver.

The real jetpump_solver is monkeypatched (fc.jetpump_solver / fc.JetPump)
with cheap analytic stand-ins so the actual Nelder-Mead multi-start
machinery runs:

- SUBSONIC: psu responds to (ken, kth, kdi), so the optimizer fits them
  exactly as today (quality graded by _classify_match, coefs from the
  optimizer, no message).
- SONIC: psu is pinned on the cavitation floor regardless of coefficients
  (kth/kdi have zero psu gradient; ken would only move the floor), so the
  single-point match is degenerate - the result must carry the SEED
  coefficients, match_quality "pinned" and an explanatory message, never
  the optimizer's railed values (the MPM-64 failure mode: railed at
  ken=0.40/kth=0.05/kdi=0.05, writing the calibration-day gauge BHP into
  the floor).
"""

from __future__ import annotations

import woffl.gui.fric_calibration as fc

FLOOR = 600.0


class _FakePump:
    """Stands in for JetPump - records the coefs the optimizer tried."""

    DNZ0 = 0.2099  # catalog nozzle diameter the multipoint fnz scales from

    def __init__(self, nozzle, throat, knz=0.01, ken=0.03, kth=0.3, kdi=0.3):
        self.nozzle, self.throat, self.knz = nozzle, throat, knz
        self.ken, self.kth, self.kdi = ken, kth, kdi
        self.dnz = self.DNZ0


def _sonic_solver(*, jpump, **kwargs):
    """Cavitation-floor well: psu never moves off FLOOR, always sonic."""
    return FLOOR, True, 180.0, 40.0, 2600.0, 1.0


def _subsonic_solver(*, jpump, **kwargs):
    """Identifiable well: psu is a smooth function of all three coefs."""
    psu = 400.0 + 2500.0 * jpump.ken + 700.0 * jpump.kth + 700.0 * jpump.kdi
    return psu, False, 250.0, 50.0, 2800.0, 0.4


def _calibrate(monkeypatch, solver, *, target, ken_seed=0.08):
    monkeypatch.setattr(fc, "JetPump", _FakePump)
    monkeypatch.setattr(fc, "jetpump_solver", solver)
    return fc.calibrate_friction_coefs(
        well_name="MPM-64",
        target_bhp=target,
        pwh=250.0,
        tsu=120.0,
        ppf_surf=3168.0,
        nozzle="12",
        throat="B",
        knz=0.01,
        ken=ken_seed,
        wellbore=None,
        wellprof=None,
        ipr_su=None,
        prop_su=None,
        prop_pf=None,
        jpump_direction="reverse",
    )


def test_sonic_pinned_returns_seeds(monkeypatch):
    r = _calibrate(monkeypatch, _sonic_solver, target=300.0, ken_seed=0.08)

    assert r.match_quality == "pinned"
    assert r.converged is True  # the solver ran; this is a diagnosis, not a failure
    assert r.sonic is True
    # Seeds, NOT the optimizer's railed values.
    assert (r.best_ken, r.best_kth, r.best_kdi) == (0.08, fc.NEUTRAL_KTH, fc.NEUTRAL_KDI)
    # Modeled BHP is the solve at the seeds - the floor.
    assert r.best_modeled_bhp == FLOOR
    assert r.bhp_error == FLOOR - 300.0
    # bounded is computed on the RETURNED (seed) coefs, which sit mid-range.
    assert r.bounded is False
    assert r.message is not None
    assert "cavitation floor" in r.message
    assert "+300 psi" in r.message  # floor gap = modeled - target


def test_sonic_pinned_clamps_seed_ken_and_flags_bound(monkeypatch):
    # A wild caller seed is clamped into KEN_BOUNDS; landing ON the bound is
    # then honestly reported through the bounded flag.
    r = _calibrate(monkeypatch, _sonic_solver, target=300.0, ken_seed=0.90)
    assert r.match_quality == "pinned"
    assert r.best_ken == fc.KEN_BOUNDS[1]
    assert (r.best_kth, r.best_kdi) == (fc.NEUTRAL_KTH, fc.NEUTRAL_KDI)
    assert r.bounded is True


def test_subsonic_behavior_unchanged(monkeypatch):
    # At the neutral seed the fake gives 400 + 2500*0.08 + 700*0.6 = 1020;
    # target 1000 is reachable inside the bounds, so the optimizer must fit
    # it and the pinned path must stay out of the way.
    r = _calibrate(monkeypatch, _subsonic_solver, target=1000.0, ken_seed=0.08)

    assert r.converged is True
    assert r.sonic is False
    assert r.message is None
    assert r.match_quality == fc._classify_match(abs(r.bhp_error))
    assert abs(r.bhp_error) <= fc.GOOD_PSI
    assert r.match_quality == "good"
    # The coefs are the optimizer's, consistent with the modeled BHP.
    expect = 400.0 + 2500.0 * r.best_ken + 700.0 * r.best_kth + 700.0 * r.best_kdi
    assert abs(expect - r.best_modeled_bhp) < 1e-9


# ---------------------------------------------------------------------------
# calibrate_multipoint - synthetic generator with a known truth set of
# (ken, kth, kdi, fnz, mach_crit)
# ---------------------------------------------------------------------------
# Each parameter gets an orthogonal observable so recovery is well-posed:
# ken = psu offset, kth = psu slope in (3500-ppf), kdi = additive qnz offset,
# fnz = multiplicative qnz scale (the sqrt term varies across ppf, so scale
# and offset separate). A shared u^2 term in psu for kth AND kdi turned out
# nearly collinear over the ppf window - the fitter matched RMS to ~3 psi on
# a kth/kdi trade-off, which is exactly the degeneracy real spread avoids.
# mach_crit lowers a cavitation FLOOR the responsive psu gets clipped to
# (psu = max(resp, floor)); with mach_crit = 1.0 and the classic ppf window
# (ppf < 3500 so u > 0) the floor never binds, keeping the legacy 4-param
# cases byte-identical. High-ppf points (u < 0) drop resp below the
# mach_crit=1.0 floor, so both BHP level and dBHP/dPpf response out there
# are reachable only by fitting mach_crit > 1 - the P3 failure mode.

import math
from types import SimpleNamespace

PPF_UNSOLVABLE = 9999.0  # sentinel ppf: the fake solver raises
RES_PRES = 1500.0


def _gen_floor(ken, mach_crit):
    """Cavitation floor: falls as the fitted critical Mach rises."""
    return 200.0 + 3000.0 * ken - 250.0 * (mach_crit - 1.0)


def _gen_resp(ken, kth, ppf):
    """Responsive (un-clipped) psu."""
    u = 3500.0 - ppf
    return 200.0 + 3000.0 * ken + 0.08 * u * kth


def _gen_psu(ken, kth, kdi, ppf, mach_crit=1.0):
    return max(_gen_resp(ken, kth, ppf), _gen_floor(ken, mach_crit))


def _gen_qnz(fnz, kdi, psu, ppf):
    return 30.0 * fnz * math.sqrt(max(ppf - psu, 25.0)) - 500.0 * kdi


def _mp_solver(*, jpump, ppf_surf, mach_crit=1.0, **kwargs):
    if ppf_surf >= PPF_UNSOLVABLE:
        raise RuntimeError("unsolvable point")
    fnz = (jpump.dnz / _FakePump.DNZ0) ** 2
    resp = _gen_resp(jpump.ken, jpump.kth, ppf_surf)
    floor = _gen_floor(jpump.ken, mach_crit)
    psu = max(resp, floor)
    qnz = _gen_qnz(fnz, jpump.kdi, psu, ppf_surf)
    return psu, resp <= floor, 200.0, 40.0, qnz, 0.4


def _mp_config():
    return SimpleNamespace(
        res_pres=RES_PRES, form_temp=120.0, surf_pres=250.0,
        jpump_direction="reverse",
        ken_well=None, kth_well=None, kdi_well=None,
        field_model="Schrader", oil_api=None, gas_sg=None, wat_sg=None,
        bubble_point=None,
    )


def _mp_points(ppfs, ken, kth, kdi, fnz, mach_crit=1.0):
    pts = []
    for i, ppf in enumerate(ppfs):
        psu = _gen_psu(ken, kth, kdi, ppf, mach_crit)
        pts.append({
            "date": f"2026-07-{i + 1:02d}", "kind": "daily", "ppf": ppf,
            "bhp": psu, "pf_rate": _gen_qnz(fnz, kdi, psu, ppf), "pwh": 250.0,
            "qtot": 900.0, "oil": None, "wc": 0.7, "fgor": 300.0,
            "weight": 1.0,
        })
    return pts


def _patch_mp(monkeypatch):
    monkeypatch.setattr(fc, "JetPump", _FakePump)
    monkeypatch.setattr(fc, "jetpump_solver", _mp_solver)
    monkeypatch.setattr(fc, "_build_well_objects",
                        lambda wc: (None, None, None, None, None))
    monkeypatch.setattr(fc, "_point_pvt_components",
                        lambda wc: (None, None, None))
    monkeypatch.setattr(fc, "_point_res_mix",
                        lambda wc, fgor, pvt: ("mix", wc, fgor))
    monkeypatch.setattr(fc, "_point_inflow",
                        lambda oil, pwf, pres: ("ipr", oil, pwf, pres))


TRUE = dict(ken=0.10, kth=0.45, kdi=0.25, fnz=1.12)
PPFS = [1200.0, 1600.0, 2000.0, 2400.0, 2900.0, 3400.0]


def test_multipoint_recovers_known_params(monkeypatch):
    _patch_mp(monkeypatch)
    pts = _mp_points(PPFS, **TRUE)
    r = fc.calibrate_multipoint(_mp_config(), "12", "B", pts)

    assert r.refusal is None
    # maxiter is contract-capped at 100 Nelder-Mead iterations, which from
    # the default far seed leaves a little slack on the slope-coupled params
    # (deterministic - NM has no randomness). The caller-seed test below
    # shows exact recovery once converged.
    assert abs(r.best_ken - TRUE["ken"]) < 0.01
    assert abs(r.best_kth - TRUE["kth"]) < 0.10
    assert abs(r.best_kdi - TRUE["kdi"]) < 0.06
    assert abs(r.best_fnz - TRUE["fnz"]) < 0.02
    assert r.n_used == len(pts) and r.n_dropped == 0
    assert r.rms_bhp_psi < 10.0 and r.rms_pf_pct < 2.0
    assert r.bounded is False and r.railed == []
    assert r.iterations > 0
    assert len(r.per_point) == len(pts)
    row = r.per_point[0]
    assert set(row) == {"date", "kind", "ppf", "bhp_meas", "bhp_model",
                        "pf_meas", "pf_model"}
    assert row["date"] == "2026-07-01" and row["kind"] == "daily"
    assert abs(row["bhp_model"] - row["bhp_meas"]) < 15.0

    # implied beta = -(dpsu/dppf) between median ppf and median-300, from
    # the same generator at the TRUE params
    med = 2200.0
    beta_true = -(_gen_psu(ppf=med, **{k: TRUE[k] for k in ("ken", "kth", "kdi")})
                  - _gen_psu(ppf=med - 300.0,
                             **{k: TRUE[k] for k in ("ken", "kth", "kdi")})) / 300.0
    assert r.implied_beta is not None
    assert abs(r.implied_beta - beta_true) < 0.02

    assert r.message.startswith(f"fit {len(pts)} points:")
    assert "fnz" in r.message and "washout" in r.message
    assert "mach_crit" in r.message


def test_multipoint_huber_ignores_wild_outlier(monkeypatch):
    # One wild gauge day (bhp +500 psi, a repeat reading at the median ppf)
    # injected into the classic truth set. Under the old squared loss the
    # outlier's pull grows with its miss and drags the fit off the truth
    # surface (verified: kth lands ~0.39 high and kdi rails at its lower
    # bound); the Huber loss caps the pull at a bounded slope, so parameter
    # recovery must stay within the CLEAN-set tolerances of
    # test_multipoint_recovers_known_params.
    _patch_mp(monkeypatch)
    pts = _mp_points(PPFS, **TRUE)
    outlier = dict(pts[2], date="2026-07-15", bhp=pts[2]["bhp"] + 500.0)
    pts = pts + [outlier]
    r = fc.calibrate_multipoint(_mp_config(), "12", "B", pts)

    assert r.refusal is None
    assert abs(r.best_ken - TRUE["ken"]) < 0.01
    assert abs(r.best_kth - TRUE["kth"]) < 0.10
    assert abs(r.best_kdi - TRUE["kdi"]) < 0.06
    assert abs(r.best_fnz - TRUE["fnz"]) < 0.02

    # The outlier is not silently dropped - its per_point row is present -
    # but the fit ignores it: the model stays on the truth surface, so the
    # row keeps essentially the full +500 psi miss.
    assert r.n_used == len(pts) and r.n_dropped == 0
    rows = [row for row in r.per_point if row["date"] == outlier["date"]]
    assert len(rows) == 1
    assert rows[0]["bhp_meas"] - rows[0]["bhp_model"] > 400.0


def test_multipoint_drops_unsolvable_and_prep_bad_points(monkeypatch):
    _patch_mp(monkeypatch)
    pts = _mp_points(PPFS, **TRUE)
    bad_solve = dict(pts[0], ppf=PPF_UNSOLVABLE)          # solver raises
    bad_prep = dict(pts[0], bhp=RES_PRES - 10.0)          # pwf >= pres - 25
    r = fc.calibrate_multipoint(_mp_config(), "12", "B",
                                pts + [bad_solve, bad_prep])

    assert r.refusal is None
    assert r.n_dropped == 2
    assert r.n_used == len(pts)
    assert len(r.per_point) == len(pts)
    assert abs(r.best_fnz - TRUE["fnz"]) < 0.02


def test_multipoint_refuses_when_majority_drop(monkeypatch):
    _patch_mp(monkeypatch)
    pts = _mp_points(PPFS[:2], **TRUE)
    dead = [dict(pts[0], ppf=PPF_UNSOLVABLE, date=f"2026-08-{i:02d}")
            for i in range(1, 5)]
    r = fc.calibrate_multipoint(_mp_config(), "12", "B", pts + dead)

    assert r.refusal is not None
    assert "half" in r.refusal
    assert r.n_dropped == 4


def test_multipoint_mirrors_builder_refusal(monkeypatch):
    _patch_mp(monkeypatch)
    builder = {"points": _mp_points(PPFS, **TRUE),
               "refusal": "young pump era - not identifiable yet"}
    r = fc.calibrate_multipoint(_mp_config(), "12", "B", builder)

    assert r.refusal == "young pump era - not identifiable yet"
    assert r.n_used == 0 and r.per_point == []
    assert r.implied_beta is None
    # seeds echoed back, clipped into bounds
    assert r.best_ken == 0.03 and r.best_fnz == 1.0
    assert r.best_mach_crit == 1.0


def test_multipoint_railed_fnz_detected(monkeypatch):
    _patch_mp(monkeypatch)
    # Generator uses fnz beyond the search bound: the fit must rail at 1.3
    # and say so instead of silently absorbing the miss elsewhere.
    pts = _mp_points(PPFS, ken=0.10, kth=0.45, kdi=0.25, fnz=1.5)
    r = fc.calibrate_multipoint(_mp_config(), "12", "B", pts)

    assert "fnz" in r.railed
    assert r.bounded is True
    assert r.best_fnz <= fc.FNZ_BOUNDS[1] + 1e-9


def test_multipoint_legacy_4tuple_seed_accepted(monkeypatch):
    # Legacy callers pass (ken, kth, kdi, fnz): mach_crit 1.0 is appended,
    # and from the exact-truth seed the fit converges immediately.
    _patch_mp(monkeypatch)
    pts = _mp_points(PPFS, **TRUE)
    r = fc.calibrate_multipoint(
        _mp_config(), "12", "B", pts,
        seed=(TRUE["ken"], TRUE["kth"], TRUE["kdi"], TRUE["fnz"]),
    )
    assert r.refusal is None
    assert r.rms_bhp_psi < 1.0 and r.rms_pf_pct < 0.5
    assert abs(r.best_mach_crit - 1.0) < 0.05


def test_multipoint_difference_term_recovers_response(monkeypatch):
    # The generator's kth term gives the truth a real dBHP/dPpf response
    # (beta = 0.08*kth over the classic u > 0 window). The paired-difference
    # term must make the fit reproduce that slope, not just the BHP levels:
    # implied_beta lands on the generator slope and the pairwise dBHP RMS
    # is small.
    _patch_mp(monkeypatch)
    pts = _mp_points(PPFS, **TRUE)
    r = fc.calibrate_multipoint(_mp_config(), "12", "B", pts)

    assert r.refusal is None
    beta_true = 0.08 * TRUE["kth"]  # -(dpsu/dppf) of _gen_resp
    assert r.implied_beta is not None
    assert abs(r.implied_beta - beta_true) < 0.02
    # 6 points sorted by ppf: 5 consecutive pairs + the (min, max) span,
    # every gap >= 100 psi, so all 6 qualify.
    assert r.rms_dbhp_psi is not None
    assert r.rms_dbhp_psi < 10.0
    # all-pairs combinations of 6 points with qualifying dppf -> 15 pairs
    assert "dBHP" in r.message and "over 15 pairs" in r.message


def test_multipoint_single_ppf_reduces_to_levels_only(monkeypatch):
    # Degenerate spread: every point at ONE ppf, so no pair reaches the
    # 100-psi dppf gate. Contract: the objective quietly reduces to the
    # level terms and rms_dbhp_psi is None (not 0 - zero would claim a
    # perfect response match that was never measured).
    _patch_mp(monkeypatch)
    pts = _mp_points([2000.0] * 4, **TRUE)
    r = fc.calibrate_multipoint(_mp_config(), "12", "B", pts)

    assert r.refusal is None
    assert r.n_used == 4 and r.n_dropped == 0
    assert r.rms_bhp_psi < 10.0 and r.rms_pf_pct < 2.0
    assert r.rms_dbhp_psi is None
    assert "dBHP" not in r.message


# ---------------------------------------------------------------------------
# mach_crit identifiability - the P3 failure mode reproduced synthetically
# ---------------------------------------------------------------------------
# TRUE5 operates at high ppf (u < 0), where the responsive psu sits BELOW
# the mach_crit=1.0 floor: a 4-param fit is pinned flat (level matchable,
# response not), while mach_crit=1.6 drops the floor 150 psi and frees both.

TRUE5 = dict(ken=0.10, kth=0.45, kdi=0.25, fnz=1.12, mach_crit=1.6)
PPFS5 = [3600.0, 4600.0, 5600.0, 6600.0, 7400.0]


def _clamp_mach_crit(monkeypatch, hi=1.0):
    monkeypatch.setattr(fc, "MP_BOUNDS", [
        fc.KEN_BOUNDS, fc.KTH_BOUNDS, fc.KDI_BOUNDS, fc.FNZ_BOUNDS,
        (1.0, hi),
    ])


def test_multipoint_4param_cannot_reproduce_floor_response(monkeypatch):
    # Force mach_crit fixed at 1.0 (a 4-param fit): every TRUE5 point pins
    # to the flat floor, so the fit can match the LEVEL (mean) but not the
    # ppf response - rms_bhp stays high and implied beta collapses to ~0.
    _patch_mp(monkeypatch)
    _clamp_mach_crit(monkeypatch, hi=1.0)
    pts = _mp_points(PPFS5, **TRUE5)
    r = fc.calibrate_multipoint(_mp_config(), "12", "B", pts)

    assert r.refusal is None
    assert r.best_mach_crit == 1.0
    assert r.rms_bhp_psi > 30.0
    assert r.implied_beta is not None
    assert abs(r.implied_beta) < 0.005


def test_multipoint_5param_recovers_mach_crit_floor(monkeypatch):
    # The same truth set, mach_crit free: the fit must escape the floor
    # (mach_crit high enough to un-pin every point), reproduce level AND
    # response, and report the recovered implied beta.
    _patch_mp(monkeypatch)
    pts = _mp_points(PPFS5, **TRUE5)
    r = fc.calibrate_multipoint(_mp_config(), "12", "B", pts)

    assert r.refusal is None
    # Any mach_crit with floor below the lowest operating psu fits equally
    # (the floor is slack above ~1.56), so assert escape, not the exact 1.6.
    assert r.best_mach_crit > 1.5
    assert r.rms_bhp_psi < 5.0 and r.rms_pf_pct < 2.0
    assert abs(r.best_ken - TRUE5["ken"]) < 0.02
    assert abs(r.best_kth - TRUE5["kth"]) < 0.05
    # fnz/kdi ride a scale-vs-offset trade-off (PF rms < 1% either way), so
    # neither is asserted here - the classic recovery test pins them.
    beta_true = 0.08 * TRUE5["kth"]  # responsive slope, -(dpsu/dppf)
    assert r.implied_beta is not None
    assert abs(r.implied_beta - beta_true) < 0.02
    assert f"mach_crit {r.best_mach_crit:.2f}" in r.message


def test_multipoint_mach_crit_railed_at_upper_bound(monkeypatch):
    # Responsive low-ppf points pin ken/kth; pinned high-ppf points sit on
    # a floor only mach_crit=2.8 reaches. The fit must rail at 2.5 and say
    # so instead of silently absorbing the miss elsewhere.
    _patch_mp(monkeypatch)
    truth = dict(ken=0.10, kth=0.9, kdi=0.25, fnz=1.0, mach_crit=2.8)
    pts = _mp_points([1200.0, 2000.0, 2900.0, 3400.0], **truth)
    pts += _mp_points([9800.0, 9900.0], **truth)
    # Seed near the truth with an interior mach_crit: the pinned points pull
    # mach_crit up against the 2.5 bound within the iteration budget.
    r = fc.calibrate_multipoint(_mp_config(), "12", "B", pts,
                                seed=(0.10, 0.9, 0.25, 1.0, 2.2))

    assert "mach_crit" in r.railed
    assert r.bounded is True
    assert r.best_mach_crit <= fc.MACH_CRIT_BOUNDS[1] + 1e-9
