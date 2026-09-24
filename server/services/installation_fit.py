"""Fit one well across all of its pump installations.

Design and evidence: docs/installation_fit_plan_2026-09-23.md.

One oil IPR describes the well (held at the saved curve unless the caller
explicitly refits its single scale). Pump behaviour is pooled: a well-level
(kth, kdi) shared by every installation, a per-installation effective
nozzle-area factor shrunk toward 1, and (model M3) per-installation kth/kdi
offsets shrunk toward the well level. Estimates are MAP values of a robust,
regularized nonlinear least-squares problem:

    sum huber((model - measured) / sigma)  +  sum ((param - prior) / prior_sd)^2 / 2

Speed comes from structure, not from a looser answer:

* Levenberg-Marquardt (IRLS Huber on data rows, Marquardt scaling), not
  Nelder-Mead: a handful of Jacobians instead of hundreds of solves.
* Jacobians by implicit differentiation. The forward model is the root of
  R(psu; params) = pdi_jp - pdi_of = 0, so d psu/d p = -(dR/dp)/(dR/dpsu).
  Each partial is ONE discharge_residual at the converged psu; nothing is
  re-solved per parameter. Sonic (pinned) tests fall back to full-solve
  differences, which are cheap there (one residual per solve).
* Newton polish: the solver stops at 5 psi / 10 psid; one Newton step on the
  same dR/dpsu lands the root well inside that, so LM does not chase noise.
* Per-test physical Jacobians are mapped to global parameters by the chain
  rule, so an installation's parameters only ever touch its own rows.

Nothing here reads Databricks or writes anything. Callers supply the
installations/tests (``server.services.pump_match.assemble``) and a
``mapper`` (serial by default; the job layer passes the process pool).
Pump losses never transfer to a replacement: this module describes the
installed history, it does not change the reference-coefficient rule.
"""
from __future__ import annotations

import math
import time
from copy import copy
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Callable, Optional

import numpy as np

from woffl.flow.entry_energy import scoped_paths

# Physical parameters every test is evaluated at, in this order.
PHYS = ("ipr", "kth", "kdi", "fnz")
IPR, KTH, KDI, FNZ = range(4)
REFERENCE = (1.0, 0.30, 0.40, 1.0)
KEN = 0.03  # held: rails on pinned wells and is not separable from kth here
KNZ = 0.01  # held: separates from fnz only through the entry area
PHYS_BOUNDS = ((0.3, 3.0), (0.05, 1.0), (0.05, 1.0), (0.8, 1.3))

# Measurement scales (the multipoint fitter's, so results are comparable).
SIGMA_BHP_PSI = 50.0
PF_REL = 0.05
OIL_ABS_BOPD = 10.0
OIL_REL = 0.10
HUBER_DELTA = 1.5
FAILED_RESIDUAL = 3.0  # standardized; a failure must cost more than a 3-sigma miss

# Priors (provisional; the plan replaces the taus with empirical Bayes).
PRIOR_SD_K = 0.15  # well-level kth/kdi around reference
TAU_LOG_FNZ = 0.05  # installation nozzle area, log scale (~5%)
TAU_DK = 0.10  # M3 installation kth/kdi offsets

IPR_GATE_MEDIAN = 0.20  # median |curve oil / test oil - 1| at measured BHP
PSU_STEP = 10.0  # psi, central difference for dR/dpsu (>> 1 psi inner noise)
PHYS_STEP = (0.02, 0.03, 0.03, 0.01)  # forward-difference steps at fixed psu
CONTINUATION_STEPS = 3  # Newton corrections from the linear prediction
CONTINUATION_TOL_PSID = 2.0  # |R| accepted (inner loops carry ~1 psi noise)
CONTINUATION_MAX_MOVE_PSI = 40.0  # beyond this, re-solve from scratch
NOT_IDENTIFIED_RATIO = 0.9  # posterior sd / prior sd above this: data added little
MIN_INFORMATION = 1.0  # Fisher information over one prior variance needed to fit kth
COLLINEARITY_MAX = 15.0  # Brun et al. (2001) upper end of the identifiable range
MODELS = ("M0", "M1", "M2", "M3")
MODEL_LABELS = {
    "M0": "Reference losses (IPR only)",
    "M1": "One well-level kth/kdi",
    "M2": "Well level + per-installation nozzle area",
    "M3": "M2 + per-installation kth/kdi offsets",
}


# ---------------------------------------------------------------------------
# Per-test physics (pure, picklable: runs in pool workers)
# ---------------------------------------------------------------------------


def _pump(cfg, phys):
    from woffl.geometry import JetPump

    pump = JetPump(cfg.installed_nozzle, cfg.installed_throat, knz=KNZ, ken=KEN,
                   kth=float(phys[KTH]), kdi=float(phys[KDI]))
    pump.dnz *= math.sqrt(float(phys[FNZ]))
    return pump


def _inflow(cfg, scale):
    from woffl.flow import InFlow

    # WellConfig.qwf is TOTAL liquid; the oil anchor is derived exactly once.
    # Scaling the anchor at fixed pwf/pres scales Vogel qmax: one curve.
    return InFlow(qwf=cfg.qwf * (1 - cfg.form_wc) * float(scale), pwf=cfg.pwf, pres=cfg.res_pres)


@scoped_paths
def _continue(resid, phys, guess) -> Optional[dict]:
    """Predictor-corrector: Newton on R(psu) from the linear prediction.

    ``guess`` = ``{"psu": predicted root, "slope": previous dR/dpsu row}``.
    Returns a state like the full solve's, or None to fall back to it (a
    large correction may mean another branch, e.g. the pump went sonic).
    """
    slope = guess["slope"]
    psu = float(guess["psu"])
    if not (math.isfinite(psu) and math.isfinite(slope[0]) and abs(slope[0]) > 1e-6):
        return None
    moved = 0.0
    r = resid(psu, phys)
    for _ in range(CONTINUATION_STEPS):
        if abs(r[0]) <= CONTINUATION_TOL_PSID:
            return {"sonic": False, "r": r, "slope": slope, "stale_slope": True,
                    "y": (psu, r[1], r[3]), "fwat": r[2]}
        step = -r[0] / slope[0]
        moved += abs(step)
        if moved > CONTINUATION_MAX_MOVE_PSI:
            return None
        psu += step
        r = resid(psu, phys)
    return None


@scoped_paths
def evaluate_test(task: dict, phys, want_jac: bool = False, state: Optional[dict] = None,
                  cols: tuple = (IPR, KTH, KDI, FNZ), guess: Optional[dict] = None) -> dict:
    """Model one test at physical parameters ``phys``.

    Returns ``{"y": (bhp, oil, pf), "liquid", "sonic", "state"}``, plus
    ``"jac"`` (3x4, d(bhp, oil, pf)/d(ipr, kth, kdi, fnz)) when asked, or
    ``{"error": message}``. ``state`` from a previous call at the SAME phys
    skips the solve (LM asks for the Jacobian only at accepted points).
    Only the physical columns in ``cols`` are computed; the rest stay zero.
    """
    from woffl.assembly.network_optimizer import NetworkOptimizer
    from woffl.assembly.solopump import discharge_residual, jetpump_solver

    cfg, pwh, ppf = task["cfg"], task["pwh"], task["ppf"]
    phys = tuple(float(v) for v in phys)
    try:
        bore, profile, _inflow0, mix, prop_pf = NetworkOptimizer._create_well_objects(cfg)
        args = (pwh, cfg.form_temp, ppf)
        kw = dict(hydraulics_model=cfg.hydraulics_model)

        def resid(psu, p):
            r, qoil, fwat, qnz, _m = discharge_residual(
                psu, *args, _pump(cfg, p), bore, profile, _inflow(cfg, p[IPR]), mix, prop_pf,
                cfg.jpump_direction, **kw)
            return float(r), float(qoil), float(fwat), float(qnz)

        def solve(p):
            psu, sonic, qoil, fwat, qnz, _m = jetpump_solver(
                *args, _pump(cfg, p), bore, profile, _inflow(cfg, p[IPR]), mix, prop_pf,
                cfg.jpump_direction, **kw)
            out = tuple(float(v) for v in (psu, qoil, fwat, qnz))
            if not all(math.isfinite(v) for v in out):
                raise ValueError("non-finite solve")
            return out, bool(sonic)

        if state is None and guess is not None:
            state = _continue(resid, phys, guess)
        if state is None:
            (psu, qoil, fwat, qnz), sonic = solve(phys)
            state = {"sonic": sonic}
            if not sonic:
                # dR/dpsu by central difference, then one Newton polish step.
                r0 = resid(psu, phys)
                rp, rm = resid(psu + PSU_STEP, phys), resid(psu - PSU_STEP, phys)
                slope = [(a - b) / (2 * PSU_STEP) for a, b in zip(rp, rm)]
                if math.isfinite(slope[0]) and abs(slope[0]) > 1e-6:
                    step = -r0[0] / slope[0]
                    if abs(step) < 2 * PSU_STEP:
                        r1 = resid(psu + step, phys)
                        if abs(r1[0]) < abs(r0[0]):
                            psu, r0 = psu + step, r1
                qoil, fwat, qnz = r0[1], r0[2], r0[3]
                state.update(r=r0, slope=slope)
            state.update(y=(psu, qoil, qnz), fwat=fwat)
        psu, qoil, qnz = state["y"]
        out = {"y": (psu, qoil, qnz), "liquid": qoil + state["fwat"], "sonic": state["sonic"], "state": state}
        if not want_jac:
            return out

        if not state["sonic"] and state.get("stale_slope"):
            # Continuation reused the previous point's dR/dpsu; the Jacobian
            # needs this point's own.
            rp, rm = resid(psu + PSU_STEP, phys), resid(psu - PSU_STEP, phys)
            state["slope"] = [(a - b) / (2 * PSU_STEP) for a, b in zip(rp, rm)]
            state["stale_slope"] = False
        jac = np.zeros((3, 4))
        base = np.array([psu, qoil, qnz])
        for j in cols:
            p = list(phys)
            h = PHYS_STEP[j] * (phys[j] if j in (IPR, FNZ) else 1.0)
            p[j] = phys[j] + h
            if state["sonic"]:
                (psu_j, qoil_j, _f, qnz_j), _s = solve(p)
                jac[:, j] = (np.array([psu_j, qoil_j, qnz_j]) - base) / h
            else:
                r_j = resid(psu, p)
                slope = state["slope"]
                dpsu = -((r_j[0] - state["r"][0]) / h) / slope[0]
                jac[0, j] = dpsu
                jac[1, j] = slope[1] * dpsu + (r_j[1] - state["r"][1]) / h
                jac[2, j] = slope[3] * dpsu + (r_j[3] - state["r"][3]) / h
        out["jac"] = jac
        return out
    except Exception as exc:  # noqa: BLE001 - a failed test is data, not a crash
        return {"error": f"{type(exc).__name__}: {exc}"}


def evaluate_chunk(items):
    """Pool task: ``[(key, task, phys, want_jac, state, cols, guess), ...] -> [(key, out)]``."""
    return [(key, evaluate_test(task, phys, want_jac, state, cols, guess))
            for key, task, phys, want_jac, state, cols, guess in items]


def serial_mapper(items):
    return evaluate_chunk(items)


def pool_mapper(items):
    """Every test of one evaluation as a few chunks on the shared process
    pool; serial under a CPU token when the pool is down or breaks. The
    per-test work is pure, so pooled and serial results are identical."""
    from server import pool

    workers = pool.workers()
    if workers > 0 and len(items) > 1:
        size = max(1, math.ceil(len(items) / (3 * workers)))
        chunks = [items[i:i + size] for i in range(0, len(items), size)]
        done = pool.submit_all(evaluate_chunk, [(chunk,) for chunk in chunks])
        if done is not None:
            return [pair for chunk in done for pair in chunk]
    with pool.cpu_slot():
        return evaluate_chunk(items)


# ---------------------------------------------------------------------------
# Problem layout
# ---------------------------------------------------------------------------


@dataclass
class Param:
    name: str
    phys: int  # index into PHYS
    inst: Optional[int]  # None = well level (every installation)
    lo: float
    hi: float
    mu: Optional[float] = None  # prior mean (in x space); None = no prior
    sd: Optional[float] = None
    log: bool = False  # x = log(physical value)


@dataclass
class Observation:
    index: int  # row index in the assembled rows
    inst: int
    task: dict
    y: tuple  # (bhp, oil, pf), each None when not usable
    date: str = ""  # ISO day, for chronological folds


def build_layout(model: str, n_inst: int, refit_ipr: bool, active_inst: list[int],
                 fit_kdi: bool = True, fit_kth: bool = True) -> list[Param]:
    """Global parameters of one rung. ``fit_kdi=False`` holds kdi at reference
    (kth then carries the combined throat+diffuser loss); ``fit_kth=False``
    holds both (the data cannot move them). See select_losses."""
    fit_kdi = fit_kdi and fit_kth
    params: list[Param] = []
    if refit_ipr:
        params.append(Param("ipr_scale", IPR, None, math.log(0.3), math.log(3.0), log=True))
    if model in ("M1", "M2", "M3") and fit_kth:
        params.append(Param("kth", KTH, None, *PHYS_BOUNDS[KTH], mu=REFERENCE[KTH], sd=PRIOR_SD_K))
        if fit_kdi:
            params.append(Param("kdi", KDI, None, *PHYS_BOUNDS[KDI], mu=REFERENCE[KDI], sd=PRIOR_SD_K))
    if model in ("M2", "M3"):
        for i in active_inst:
            params.append(Param(f"fnz[{i}]", FNZ, i, math.log(0.8), math.log(1.3), mu=0.0, sd=TAU_LOG_FNZ, log=True))
    if model == "M3" and fit_kth:
        for i in active_inst:
            params.append(Param(f"dkth[{i}]", KTH, i, -0.5, 0.5, mu=0.0, sd=TAU_DK))
            if fit_kdi:
                params.append(Param(f"dkdi[{i}]", KDI, i, -0.5, 0.5, mu=0.0, sd=TAU_DK))
    return params


def x_start(params: list[Param]) -> np.ndarray:
    # Prior means; unregularized parameters (the IPR scale) start at 1 (log 0).
    return np.array([p.mu if p.mu is not None else 0.0 for p in params], dtype=float)


def phys_for(params: list[Param], x: np.ndarray, inst: int) -> tuple[np.ndarray, np.ndarray]:
    """Physical parameters for installation ``inst`` and d phys / d x (4 x n)."""
    phys = np.array(REFERENCE, dtype=float)
    dphys = np.zeros((4, len(params)))
    for k, p in enumerate(params):
        if p.inst is not None and p.inst != inst:
            continue
        if p.log:
            phys[p.phys] = math.exp(x[k])
            dphys[p.phys, k] = phys[p.phys]
        elif p.inst is None:  # well-level value replaces the reference
            phys[p.phys] = x[k]
            dphys[p.phys, k] = 1.0
    for k, p in enumerate(params):  # additive installation offsets last
        if p.inst == inst and not p.log:
            phys[p.phys] += x[k]
            dphys[p.phys, k] = 1.0
    for j, (lo, hi) in enumerate(PHYS_BOUNDS):
        if phys[j] < lo or phys[j] > hi:
            phys[j] = min(max(phys[j], lo), hi)
            dphys[j, :] = 0.0  # pinned at the physical bound
    return phys, dphys


def sigmas(y) -> tuple:
    bhp, oil, pf = y
    return (SIGMA_BHP_PSI if bhp is not None else None,
            max(OIL_ABS_BOPD, OIL_REL * oil) if oil is not None else None,
            PF_REL * pf if pf is not None else None)


# ---------------------------------------------------------------------------
# Residuals, Jacobian and Levenberg-Marquardt
# ---------------------------------------------------------------------------


class Problem:
    """Evaluates standardized data residuals, priors and their Jacobian."""

    def __init__(self, obs: list[Observation], params: list[Param], mapper: Callable = serial_mapper,
                 cancel: Optional[Callable[[], None]] = None):
        self.obs, self.params, self.mapper, self.cancel = obs, params, mapper, cancel
        self.rows = [(t, c) for t, o in enumerate(obs) for c in range(3) if o.y[c] is not None]
        self.sig = [sigmas(o.y) for o in obs]
        self.evaluations = 0
        self.test_evaluations = 0
        # Physical columns any parameter maps to; the others are never needed.
        self.cols = tuple(sorted({p.phys for p in params}))

    def run(self, x: np.ndarray, want_jac: bool, states: Optional[list] = None,
            near: Optional[tuple] = None) -> list[dict]:
        """Evaluate every test at ``x``. ``states`` reuse solves made at this
        same ``x``; ``near = (x_prev, results_prev)`` predicts each non-sonic
        root from the previous Jacobian (continuation) instead of re-solving."""
        if self.cancel is not None:
            self.cancel()
        items = []
        for t, o in enumerate(self.obs):
            phys, _ = phys_for(self.params, x, o.inst)
            state = states[t] if states is not None else None
            guess = None
            if near is not None and state is None:
                prev = near[1][t]
                if "jac" in prev and not prev["sonic"] and prev["state"].get("slope"):
                    old, _ = phys_for(self.params, near[0], o.inst)
                    guess = {"psu": prev["y"][0] + float(prev["jac"][0] @ (phys - old)),
                             "slope": prev["state"]["slope"]}
            items.append((t, o.task, tuple(phys), want_jac, state, self.cols, guess))
        out = dict(self.mapper(items))
        self.evaluations += 1
        self.test_evaluations += len(items)
        return [out[t] for t in range(len(self.obs))]

    def residuals(self, results: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        """Data residuals and a mask of rows whose test failed."""
        r = np.zeros(len(self.rows))
        failed = np.zeros(len(self.rows), dtype=bool)
        for k, (t, c) in enumerate(self.rows):
            res = results[t]
            if "error" in res:
                r[k], failed[k] = FAILED_RESIDUAL, True
            else:
                r[k] = (res["y"][c] - self.obs[t].y[c]) / self.sig[t][c]
        return r, failed

    def jacobian(self, x: np.ndarray, results: list[dict]) -> np.ndarray:
        J = np.zeros((len(self.rows), len(self.params)))
        dphys = [phys_for(self.params, x, o.inst)[1] for o in self.obs]
        for k, (t, c) in enumerate(self.rows):
            res = results[t]
            if "jac" in res:
                J[k] = res["jac"][c] @ dphys[t] / self.sig[t][c]
        return J

    def prior(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = [k for k, p in enumerate(self.params) if p.sd is not None]
        rp = np.array([(x[k] - self.params[k].mu) / self.params[k].sd for k in idx])
        Jp = np.zeros((len(idx), len(self.params)))
        for row, k in enumerate(idx):
            Jp[row, k] = 1.0 / self.params[k].sd
        return rp, Jp


def huber(r: np.ndarray) -> np.ndarray:
    a = np.abs(r)
    return np.where(a <= HUBER_DELTA, 0.5 * r * r, HUBER_DELTA * (a - 0.5 * HUBER_DELTA))


def huber_weights(r: np.ndarray) -> np.ndarray:
    a = np.abs(r)
    return np.where(a <= HUBER_DELTA, 1.0, HUBER_DELTA / np.maximum(a, 1e-12))


def total_cost(r: np.ndarray, rp: np.ndarray) -> float:
    return float(huber(r).sum() + 0.5 * (rp @ rp if rp.size else 0.0))


@dataclass
class FitResult:
    model: str
    params: list[Param]
    x: np.ndarray
    results: list[dict]
    r: np.ndarray
    failed: np.ndarray
    J: np.ndarray
    cost: float
    iterations: int
    converged: bool
    message: str
    evaluations: int
    test_evaluations: int
    seconds: float
    cov: Optional[np.ndarray] = None
    extra: dict = field(default_factory=dict)


def levenberg_marquardt(problem: Problem, x0: np.ndarray, *, max_iter: int = 25, rtol: float = 1e-4,
                        progress: Optional[Callable[[str], None]] = None) -> tuple:
    """Projected LM with Huber IRLS on data rows. Returns (x, results, J, cost, iters, converged, msg)."""
    lo = np.array([p.lo for p in problem.params])
    hi = np.array([p.hi for p in problem.params])
    x = np.clip(x0, lo, hi)
    results = problem.run(x, want_jac=True)
    r, _ = problem.residuals(results)
    rp, Jp = problem.prior(x)
    J = problem.jacobian(x, results)
    cost = total_cost(r, rp)
    if not problem.params:
        return x, results, J, cost, 0, True, "no free parameters"
    lam = 1e-2
    small = 0  # consecutive accepted steps below rtol (one can be solver noise)
    for it in range(1, max_iter + 1):
        w = huber_weights(r)
        A = J.T @ (w[:, None] * J) + Jp.T @ Jp
        g = J.T @ (w * r) + Jp.T @ rp
        D = np.maximum(np.diag(A), 1e-9)
        improved = False
        for _attempt in range(8):
            try:
                dx = np.linalg.solve(A + lam * np.diag(D), -g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            x_new = np.clip(x + dx, lo, hi)
            if np.allclose(x_new, x, rtol=0, atol=1e-10):
                return x, results, J, cost, it, True, "step below resolution"
            trial = problem.run(x_new, want_jac=False, near=(x, results))
            r_new, _ = problem.residuals(trial)
            rp_new, _ = problem.prior(x_new)
            cost_new = total_cost(r_new, rp_new)
            if cost_new < cost:
                states = [res.get("state") for res in trial]
                results = problem.run(x_new, want_jac=True, states=states)
                rel = (cost - cost_new) / max(cost, 1e-12)
                x, r, rp, cost = x_new, r_new, rp_new, cost_new
                Jp = problem.prior(x)[1]
                J = problem.jacobian(x, results)
                lam = max(lam / 3, 1e-7)
                improved = True
                if progress is not None:
                    progress(f"iteration {it}: cost {cost:.2f}")
                small = small + 1 if rel < rtol else 0
                if small >= 2:
                    return x, results, J, cost, it, True, "cost change below tolerance"
                break
            lam *= 4
        if not improved:
            return x, results, J, cost, it, True, "no improving step (local minimum)"
    return x, results, J, cost, max_iter, False, "iteration limit"


def fit_model(model: str, obs: list[Observation], n_inst: int, *, refit_ipr: bool = False,
              mapper: Callable = serial_mapper, x0: Optional[np.ndarray] = None,
              cancel: Optional[Callable[[], None]] = None,
              progress: Optional[Callable[[str], None]] = None,
              fit_kdi: bool = True, fit_kth: bool = True) -> FitResult:
    started = time.perf_counter()
    active = sorted({o.inst for o in obs})
    params = build_layout(model, n_inst, refit_ipr, active, fit_kdi, fit_kth)
    problem = Problem(obs, params, mapper, cancel)
    x, results, J, cost, iters, converged, message = levenberg_marquardt(
        problem, x_start(params) if x0 is None else x0, progress=progress)
    if params:
        # Report the optimum from full solves (continuation only steered LM),
        # so every displayed prediction is exactly the solver's.
        results = problem.run(x, want_jac=True)
        J = problem.jacobian(x, results)
        rp, _ = problem.prior(x)
        cost = total_cost(problem.residuals(results)[0], rp)
    r, failed = problem.residuals(results)
    fit = FitResult(model, params, x, results, r, failed, J, cost, iters, converged, message,
                    problem.evaluations, problem.test_evaluations, time.perf_counter() - started)
    fit.cov = laplace_covariance(problem, fit)
    return fit


# ---------------------------------------------------------------------------
# Uncertainty, identifiability and model comparison
# ---------------------------------------------------------------------------


def laplace_covariance(problem: Problem, fit: FitResult) -> Optional[np.ndarray]:
    """(J^T W J + P)^-1 at the MAP, inflated by the reduced chi-square when > 1."""
    if not fit.params:
        return None
    ok = ~fit.failed
    w = huber_weights(fit.r) * ok
    _rp, Jp = problem.prior(fit.x)
    H = fit.J.T @ (w[:, None] * fit.J) + Jp.T @ Jp
    try:
        cov = np.linalg.pinv(H)
    except np.linalg.LinAlgError:
        return None
    dof = int(ok.sum()) - len(fit.params)
    if dof > 0:
        chi2 = float((w * fit.r * fit.r).sum()) / dof
        cov = cov * max(1.0, chi2)
    return cov


def collinearity_index(J: np.ndarray, mask: Optional[np.ndarray] = None) -> Optional[float]:
    """Brun et al. (2001) collinearity index of the data Jacobian's columns."""
    S = J if mask is None else J[:, mask]
    if S.shape[1] < 2:
        return None
    norms = np.linalg.norm(S, axis=0)
    if np.any(norms <= 0):
        return math.inf
    Sn = S / norms
    lam_min = float(np.linalg.eigvalsh(Sn.T @ Sn)[0])
    return math.inf if lam_min <= 1e-15 else 1.0 / math.sqrt(lam_min)


def select_losses(obs: list[Observation], *, mapper: Callable = serial_mapper,
                  cancel: Optional[Callable[[], None]] = None) -> dict:
    """Which pump losses can these data identify? One Jacobian pass at reference.

    * kth: its Fisher information over one prior variance,
      ||J_kth||^2 * PRIOR_SD_K^2, must reach MIN_INFORMATION. Below that the
      posterior is barely narrower than the prior (sonic/pinned wells: kth
      cannot move suction), so kth and kdi are held at reference.
    * kdi: held at reference when kth and kdi are collinear (Brun et al.
      2001 index above COLLINEARITY_MAX). BHP, oil and PF respond to the two
      in almost fixed proportion, so only their combination is identified;
      kth then carries it. Reported, never hidden.
    """
    problem = Problem(obs, build_layout("M1", 0, False, []), mapper, cancel)
    x = x_start(problem.params)
    results = problem.run(x, want_jac=True)
    _r, failed = problem.residuals(results)
    J = problem.jacobian(x, results)[~failed]
    info_kth = float(J[:, 0] @ J[:, 0]) * PRIOR_SD_K ** 2
    info_kdi = float(J[:, 1] @ J[:, 1]) * PRIOR_SD_K ** 2
    index = collinearity_index(J)
    sonic = sum(1 for res in results if res.get("sonic"))
    out: dict[str, Any] = dict(information_kth=info_kth, information_kdi=info_kdi, collinearity=index,
               sonic_tests=sonic, solved_tests=sum(1 for res in results if "error" not in res),
               fit_kth=True, fit_kdi=True, reason=None)
    if info_kth < MIN_INFORMATION:
        why = (f"{sonic} of {out['solved_tests']} solved tests are sonic-pinned, where these losses cannot "
               "move suction" if sonic * 2 >= max(out["solved_tests"], 1) else
               "the measured BHP, oil and PF barely respond to them within their measurement error")
        out.update(fit_kth=False, fit_kdi=False, reason=(
            f"The tests carry almost no information on throat/diffuser losses ({why}; information "
            f"{info_kth:.2f} < {MIN_INFORMATION:.0f}). kth and kdi stay at reference."))
    elif index is not None and index > COLLINEARITY_MAX:
        out.update(fit_kdi=False, reason=(
            f"kth and kdi move BHP, oil and PF in nearly fixed proportion (collinearity index {index:.0f} > "
            f"{COLLINEARITY_MAX:.0f}); only their combination is identifiable. kdi is held at reference and "
            "kth carries the combined loss."))
    return out


def aicc(fit: FitResult) -> Optional[float]:
    m = int((~fit.failed).sum())
    k = len(fit.params)
    if m - k - 1 <= 0:
        return None
    rss = float((fit.r[~fit.failed] ** 2).sum())
    return m * math.log(max(rss, 1e-12) / m) + 2 * k + 2 * k * (k + 1) / (m - k - 1)


def describe_params(fit: FitResult, inst_ids: list[str]) -> list[dict]:
    rows = []
    for k, p in enumerate(fit.params):
        sd_x = math.sqrt(fit.cov[k, k]) if fit.cov is not None and fit.cov[k, k] >= 0 else None
        value = math.exp(fit.x[k]) if p.log else float(fit.x[k])
        # Delta method for log parameters: sd(value) = value * sd(x).
        sd = (value * sd_x if p.log else sd_x) if sd_x is not None else None
        ratio = (sd_x / p.sd) if (sd_x is not None and p.sd) else None
        at_bound = abs(fit.x[k] - p.lo) < 1e-6 or abs(fit.x[k] - p.hi) < 1e-6
        rows.append(dict(
            name=p.name, physical=PHYS[p.phys], installation_id=inst_ids[p.inst] if p.inst is not None else None,
            value=value, sd=sd, prior_mean=(math.exp(p.mu) if p.log and p.mu is not None else p.mu),
            prior_sd=p.sd, posterior_to_prior=ratio, at_bound=bool(at_bound),
            identified=(ratio is None or ratio < NOT_IDENTIFIED_RATIO),
        ))
    return rows


def ipr_consistency(obs: list[Observation]) -> dict:
    """Oil on the one saved curve at each test's measured BHP vs measured oil."""
    misses = []
    for o in obs:
        bhp, oil, _pf = o.y
        if bhp is None or oil is None or oil <= 0:
            continue
        try:
            curve = _inflow(o.task["cfg"], 1.0).oil_flow(bhp, method="vogel")
        except Exception:  # noqa: BLE001
            continue
        misses.append(curve / oil - 1.0)
    if not misses:
        return dict(count=0, median_abs=None, median_signed=None, passes=True)
    med_abs = median(abs(m) for m in misses)
    return dict(count=len(misses), median_abs=med_abs, median_signed=median(misses),
                passes=med_abs <= IPR_GATE_MEDIAN)


# ---------------------------------------------------------------------------
# Assembly glue
# ---------------------------------------------------------------------------


def prepare(well: str, *, months: int = 24, hydraulics_model: str = "beggs",
            exclude_wt_uids: tuple = ()) -> dict[str, Any]:
    """Load one well's installations and tests exactly as the every-test
    replay does (saved oil IPR, each test's WC/GOR, clean reference losses).

    Reads Databricks (cached fleet frames); writes nothing.
    """
    from server import schemas
    from server.services import pump_match as pm

    request = schemas.PumpMatchRequest(mode="all_tests", months=months, pump_losses="clean_reference",
                                       hydraulics_model=hydraulics_model)
    cfg, tracker, test_frame, source, context, as_of = pm.load_history(well, request)
    if exclude_wt_uids and "wt_uid" in test_frame:
        excluded = {str(u) for u in exclude_wt_uids}
        test_frame = test_frame[~test_frame["wt_uid"].map(pm.code).isin(excluded)]
    eras, rows, work, notes = pm.assemble(cfg, tracker, test_frame, request, as_of)
    obs, inst_ids = observations_from_assembly(eras, rows, work, cfg.res_pres)
    return dict(well=well, cfg=cfg, eras=eras, rows=rows, obs=obs, inst_ids=inst_ids, notes=notes,
                source=source, as_of=as_of, context=context, request=request)


def observations_from_assembly(eras: list[dict], rows: list[dict], work: list, res_pres: float) -> tuple:
    """Turn ``pump_match.assemble(mode='all_tests')`` output into observations.

    ``work`` items are ``(cfg_at_test, controls, row_index)`` with the saved
    oil IPR and the test's WC/GOR already applied. Returns
    ``(observations, installation_ids)``; installations without usable tests
    are omitted (they get no parameters).
    """
    ids = [e["installation_id"] for e in eras]
    obs: list[Observation] = []
    for cfg, controls, index in work:
        row = rows[index]
        inst = ids.index(row["installation_id"])
        bhp = row.get("bhp")
        bhp = bhp if (bhp is not None and 50 < bhp < res_pres - 10) else None
        oil = row.get("oil")
        oil = oil if (oil is not None and oil > 0) else None
        pf = row.get("pf")
        pf = pf if (pf is not None and 0 < pf <= 20000) else None
        if bhp is None and oil is None and pf is None:
            continue
        task = {"cfg": copy(cfg), "pwh": float(controls["pwh"]), "ppf": float(controls["ppf"])}
        obs.append(Observation(index, inst, task, (bhp, oil, pf), row["date"]))
    return obs, ids


def summarize(fit: FitResult, obs: list[Observation], inst_ids: list[str]) -> dict:
    predictions = []
    for o, res in zip(obs, fit.results):
        entry = dict(index=o.index, installation_id=inst_ids[o.inst])
        if "error" in res:
            entry["message"] = res["error"]
        else:
            bhp, oil, pf = res["y"]
            entry.update(predicted_bhp=bhp, predicted_oil=oil, predicted_pf=pf,
                         predicted_liquid=res["liquid"], sonic=res["sonic"])
        predictions.append(entry)
    data_rows = ~fit.failed
    return dict(
        model=fit.model, label=MODEL_LABELS[fit.model], cost=fit.cost, iterations=fit.iterations,
        converged=fit.converged, message=fit.message, evaluations=fit.evaluations,
        test_evaluations=fit.test_evaluations, seconds=fit.seconds,
        n_rows=int(len(fit.r)), n_failed_rows=int(fit.failed.sum()),
        rms_standardized=float(math.sqrt((fit.r[data_rows] ** 2).mean())) if data_rows.any() else None,
        aicc=aicc(fit), collinearity=collinearity_index(fit.J) if fit.params else None,
        params=describe_params(fit, inst_ids) if fit.params else [],
        predictions=predictions,
    )


def fit_ladder(obs: list[Observation], inst_ids: list[str], *, models=("M0", "M1", "M2"),
               refit_ipr: bool = False, mapper: Callable = serial_mapper,
               cancel: Optional[Callable[[], None]] = None,
               progress: Optional[Callable[[str], None]] = None) -> dict[str, Any]:
    """Fit the requested rungs, each warm-started from the previous one.

    The IPR gate runs first: when the one saved curve misses oil at measured
    BHP by more than IPR_GATE_MEDIAN (median), pump parameters are not fitted
    unless the caller asked to refit the curve. Otherwise pump losses would
    absorb inflow error and look like wear.
    """
    gate = ipr_consistency(obs)
    out: dict[str, Any] = {"ipr_gate": gate, "models": {}, "skipped": {}, "losses": None}
    previous: Optional[FitResult] = None
    active = sorted({o.inst for o in obs})
    fit_kdi = fit_kth = True
    for model in models:
        if model != "M0" and not gate["passes"] and not refit_ipr:
            out["skipped"][model] = (
                f"The saved IPR misses test oil at measured BHP by a median {100 * gate['median_abs']:.0f}% "
                f"(gate {100 * IPR_GATE_MEDIAN:.0f}%); pump terms would absorb inflow error. Refit the one IPR first.")
            continue
        if model != "M0" and out["losses"] is None:
            if progress is not None:
                progress("checking which pump losses the data can identify")
            out["losses"] = select_losses(obs, mapper=mapper, cancel=cancel)
            fit_kdi, fit_kth = out["losses"]["fit_kdi"], out["losses"]["fit_kth"]
            if not fit_kth:
                for rest in models[models.index(model):]:
                    if rest in ("M1", "M3"):
                        out["skipped"][rest] = out["losses"]["reason"]
            if model in out["skipped"]:
                continue
        if progress is not None:
            progress(f"fitting {model}: {MODEL_LABELS[model]}")
        layout = build_layout(model, len(inst_ids), refit_ipr, active, fit_kdi, fit_kth)
        x0 = _warm_start(previous, layout) if previous is not None else None
        fit = fit_model(model, obs, len(inst_ids), refit_ipr=refit_ipr, mapper=mapper, x0=x0,
                        cancel=cancel, progress=progress, fit_kdi=fit_kdi, fit_kth=fit_kth)
        if model != "M0" and fit_kdi and fit_kth:
            # Identifiability is a property of the linearization point: judge
            # the kth/kdi pair where the data put them, not at reference.
            names = [p.name for p in fit.params]
            cols = [names.index("kth"), names.index("kdi")]
            index = collinearity_index(fit.J[~fit.failed][:, cols])
            out["losses"]["collinearity_at_fit"] = index
            if index is not None and index > COLLINEARITY_MAX:
                fit_kdi = False
                out["losses"].update(fit_kdi=False, reason=(
                    f"kth and kdi move BHP, oil and PF in nearly fixed proportion at the fitted point "
                    f"(collinearity index {index:.0f} > {COLLINEARITY_MAX:.0f}); only their combination is "
                    "identifiable. kdi is held at reference and kth carries the combined loss."))
                if progress is not None:
                    progress(f"refitting {model} with kdi held at reference (kth/kdi not separable)")
                layout = build_layout(model, len(inst_ids), refit_ipr, active, fit_kdi, fit_kth)
                fit = fit_model(model, obs, len(inst_ids), refit_ipr=refit_ipr, mapper=mapper,
                                x0=_warm_start(fit, layout), cancel=cancel, progress=progress,
                                fit_kdi=fit_kdi, fit_kth=fit_kth)
        out["models"][model] = summarize(fit, obs, inst_ids)
        out.setdefault("_fits", {})[model] = fit
        previous = fit
    out["_flags"] = dict(fit_kdi=fit_kdi, fit_kth=fit_kth, refit_ipr=refit_ipr)
    return out


def _warm_start(previous: FitResult, params: list[Param]) -> np.ndarray:
    x = x_start(params)
    known = {p.name: previous.x[k] for k, p in enumerate(previous.params)}
    for k, p in enumerate(params):
        if p.name in known:
            x[k] = known[p.name]
    return x


# ---------------------------------------------------------------------------
# Rolling-origin cross-validation and model selection
# ---------------------------------------------------------------------------

EMBARGO_DAYS = 3
MAX_FOLDS = 6
MIN_TRAIN_OBS = 3
CV_MAX_ITER = 8
CHANGEOUT_TESTS = 3  # tests each side of a changeout for the response check
DELTA_MIN = {"bhp": 25.0, "oil": 10.0}  # smaller measured changes are "no clear change"


def _day(text: str):
    from datetime import date

    return date.fromisoformat(text[:10])


def fold_origins(obs: list[Observation]) -> list[tuple[str, str]]:
    """Chronological split points ``(day, kind)``: the first test of every
    later installation, then the middle test of long installations.

    A changeout fold asks "does what we learned transfer to a NEW pump?"; a
    mid-installation fold asks "does it hold later on the SAME pump?".
    """
    if not obs:
        return []
    first = min(o.date for o in obs)
    by_inst: dict[int, list[str]] = {}
    for o in obs:
        by_inst.setdefault(o.inst, []).append(o.date)
    origins = []
    mids = []
    for _inst, dates in sorted(by_inst.items()):
        dates = sorted(dates)
        if dates[0] > first:
            origins.append((dates[0], "changeout"))
        if len(dates) >= 6:
            mids.append((len(dates), dates[len(dates) // 2]))
    origins = sorted(origins)[:MAX_FOLDS]
    for _n, d in sorted(mids, reverse=True):
        if len(origins) >= MAX_FOLDS:
            break
        if all(abs((_day(d) - _day(o)).days) > EMBARGO_DAYS for o, _k in origins):
            origins.append((d, "later_same_pump"))
    return sorted(origins)


def _test_loss(r: np.ndarray, rows: list[tuple[int, int]], failed: np.ndarray, n_obs: int) -> np.ndarray:
    """Huber loss summed over each held-out test's channels."""
    clipped = np.where(failed, FAILED_RESIDUAL, r)
    per_row = huber(clipped)
    loss = np.zeros(n_obs)
    for k, (t, _c) in enumerate(rows):
        loss[t] += float(per_row[k])
    return loss


def cross_validate(obs: list[Observation], inst_ids: list[str], ladder: dict, *,
                   mapper: Callable = serial_mapper, cancel: Optional[Callable[[], None]] = None,
                   progress: Optional[Callable[[str], None]] = None) -> dict[str, Any]:
    """Refit every rung on data before each origin (minus the embargo) and
    score its predictions of the tests after it, up to the next origin.

    Held-out predictions never see their own outcomes. A new installation's
    own parameters are unknown before its tests, so it is predicted at the
    prior (nozzle factor 1, no offsets) with the well-level values learned.
    """
    fits: dict[str, FitResult] = ladder.get("_fits", {})
    flags = ladder.get("_flags", {})
    origins = fold_origins(obs)
    folds: list[dict] = []
    per_model_losses: dict[str, list[float]] = {m: [] for m in fits}
    held_predictions: dict[str, list[dict]] = {m: [] for m in fits}
    for f, (origin, kind) in enumerate(origins):
        cutoff = _day(origin)
        nxt = _day(origins[f + 1][0]) if f + 1 < len(origins) else None
        train = [o for o in obs if (cutoff - _day(o.date)).days > EMBARGO_DAYS]
        test = [o for o in obs if _day(o.date) >= cutoff and (nxt is None or _day(o.date) < nxt)]
        fold: dict[str, Any] = dict(origin=origin, kind=kind, n_train=len(train), n_test=len(test), models={})
        if len(train) < MIN_TRAIN_OBS or not test:
            fold["skipped"] = "Too few earlier tests to train on." if test else "No tests after this point."
            folds.append(fold)
            continue
        active = sorted({o.inst for o in train})
        for model, full in fits.items():
            if progress is not None:
                progress(f"held-out check {f + 1}/{len(origins)} (from {origin}): {model}")
            layout = build_layout(model, len(inst_ids), flags.get("refit_ipr", False), active,
                                  flags.get("fit_kdi", True), flags.get("fit_kth", True))
            problem = Problem(train, layout, mapper, cancel)
            x, results, *_rest = levenberg_marquardt(problem, _warm_start(full, layout), max_iter=CV_MAX_ITER)
            held = Problem(test, layout, mapper, cancel)
            held_results = held.run(x, want_jac=False)
            r, failed = held.residuals(held_results)
            losses = _test_loss(r, held.rows, failed, len(test))
            per_model_losses[model].extend(losses.tolist())
            entry = dict(mean_loss=float(losses.mean()), n=len(test),
                         failed=int(sum("error" in h for h in held_results)),
                         **_channel_scores(test, held_results))
            if kind == "changeout":
                entry["changeout"] = _changeout_check(train, results, test, held_results)
            fold["models"][model] = entry
            for o, res in zip(test, held_results):
                held_predictions[model].append(dict(_prediction(o, res, inst_ids), origin=origin))
        folds.append(fold)
    scores = {}
    for model, losses in per_model_losses.items():
        if losses:
            arr = np.array(losses)
            scores[model] = dict(mean_loss=float(arr.mean()),
                                 se=float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else None,
                                 n=len(arr))
    return dict(folds=folds, scores=scores, held_predictions=held_predictions, embargo_days=EMBARGO_DAYS)


def _prediction(o: Observation, res: dict, inst_ids: list[str]) -> dict:
    entry = dict(index=o.index, installation_id=inst_ids[o.inst])
    if "error" in res:
        entry["message"] = res["error"]
    else:
        bhp, oil, pf = res["y"]
        entry.update(predicted_bhp=bhp, predicted_oil=oil, predicted_pf=pf,
                     predicted_liquid=res["liquid"], sonic=res["sonic"])
    return entry


def _channel_scores(test: list[Observation], results: list[dict]) -> dict:
    bhp, oil, pf = [], [], []
    for o, res in zip(test, results):
        if "error" in res:
            continue
        mb, mo, mp = res["y"]
        if o.y[0] is not None:
            bhp.append(mb - o.y[0])
        if o.y[1] is not None:
            oil.append(100 * abs(mo / o.y[1] - 1))
        if o.y[2] is not None:
            pf.append(100 * abs(mp / o.y[2] - 1))
    return dict(bhp_rms=math.sqrt(sum(e * e for e in bhp) / len(bhp)) if bhp else None,
                bhp_bias=sum(bhp) / len(bhp) if bhp else None,
                oil_median_abs_pct=median(oil) if oil else None,
                pf_median_abs_pct=median(pf) if pf else None)


def _changeout_check(train: list[Observation], train_results: list[dict],
                     test: list[Observation], held_results: list[dict]) -> dict:
    """Measured vs predicted change across the pump change (medians of the
    last/first CHANGEOUT_TESTS tests). This is what a sizing decision uses."""
    new_inst = test[0].inst
    before = [(o, r) for o, r in zip(train, train_results) if o.inst != new_inst][-CHANGEOUT_TESTS:]
    after = [(o, r) for o, r in zip(test, held_results) if o.inst == new_inst][:CHANGEOUT_TESTS]
    out: dict[str, Any] = {}
    for c, name in ((0, "bhp"), (1, "oil")):
        mb = [o.y[c] for o, _r in before if o.y[c] is not None]
        ma = [o.y[c] for o, _r in after if o.y[c] is not None]
        pb = [r["y"][c] for _o, r in before if "error" not in r]
        pa = [r["y"][c] for _o, r in after if "error" not in r]
        if not (mb and ma and pb and pa):
            out[name] = None
            continue
        measured, predicted = median(ma) - median(mb), median(pa) - median(pb)
        clear = abs(measured) >= DELTA_MIN[name]
        out[name] = dict(measured=measured, predicted=predicted, clear=clear,
                         direction_correct=((measured > 0) == (predicted > 0)) if clear else None)
    return out


def select_model(ladder: dict, cv: dict) -> dict:
    """Simplest rung within one standard error of the best held-out score.

    Without held-out evidence the lowest AICc is shown and labeled
    unvalidated: a description of the history, not a tested predictor.
    """
    order = [m for m in MODELS if m in ladder["models"]]
    scores = cv.get("scores", {})
    if order and all(m in scores for m in order):
        best = min(order, key=lambda m: scores[m]["mean_loss"])
        se = scores[best]["se"] or 0.0
        limit = scores[best]["mean_loss"] + se
        chosen = next(m for m in order if scores[m]["mean_loss"] <= limit)
        return dict(model=chosen, basis="held_out_1se", best=best, limit=limit, reason=(
            f"{chosen} is the simplest model whose held-out loss ({scores[chosen]['mean_loss']:.2f}) is within "
            f"one standard error of the best ({best}: {scores[best]['mean_loss']:.2f} + {se:.2f})."))
    with_aicc = [m for m in order if ladder["models"][m].get("aicc") is not None]
    if not with_aicc:
        return dict(model=order[0] if order else None, basis="none", reason="No model could be compared.")
    chosen = min(with_aicc, key=lambda m: ladder["models"][m]["aicc"])
    return dict(model=chosen, basis="aicc_unvalidated", reason=(
        f"No held-out tests were available, so {chosen} is shown for its lowest AICc. "
        "It describes the history but was not tested on unseen data."))


# ---------------------------------------------------------------------------
# Background job
# ---------------------------------------------------------------------------

KINDS = ("installation-fit",)


def run(job: dict, well: str, request) -> dict[str, Any]:
    import pickle
    from hashlib import sha256
    from pathlib import Path

    from server import jobs, surface_cache
    from woffl.flow.hydraulics import physics_model

    def cancel():
        jobs.check_cancelled(job)

    def progress(text):
        jobs.set_progress(job, text)

    progress("loading saved well inputs, installations and tests")
    prep = prepare(well, months=request.months, hydraulics_model=request.hydraulics_model,
                   exclude_wt_uids=tuple(request.exclude_wt_uids))
    cancel()
    obs, inst_ids, eras = prep["obs"], prep["inst_ids"], prep["eras"]
    if not obs:
        raise ValueError("No usable tests with BHP, oil or PF on any dated installation in this window.")
    here = Path(__file__)
    source_hash = sha256(here.read_bytes() + here.with_name("pump_match.py").read_bytes()).hexdigest()
    inputs = (source_hash, request.model_dump(), prep["source"], prep["as_of"][:10], eras,
              [(vars(o.task["cfg"]), o.task["pwh"], o.task["ppf"], o.y, o.date, o.inst) for o in obs])
    key, cached = surface_cache.snapshot(prep["cfg"], "installation-fit-v1", inputs)
    if cached is not None:
        return cached

    started = time.perf_counter()
    models = ("M0", "M1", "M2", "M3") if request.include_m3 else ("M0", "M1", "M2")
    ladder = fit_ladder(obs, inst_ids, models=models, refit_ipr=request.refit_ipr,
                        mapper=pool_mapper, cancel=cancel, progress=progress)
    cv = cross_validate(obs, inst_ids, ladder, mapper=pool_mapper, cancel=cancel, progress=progress)
    selection = select_model(ladder, cv)
    cancel()

    counts: dict[int, int] = {}
    spans: dict[int, list] = {}
    for o in obs:
        counts[o.inst] = counts.get(o.inst, 0) + 1
        spans.setdefault(o.inst, []).append(o.task["ppf"])
    era_rows = [dict(
        installation_id=era["installation_id"], date_set=era["date_set"], end=era.get("end"),
        pump=era["pump"], nozzle=era.get("nozzle"), throat=era.get("throat"),
        flags=list(era.get("flags") or []), unavailable=era.get("unavailable"), n_tests=counts.get(i, 0),
        ppf_span=(max(spans[i]) - min(spans[i])) if i in spans else None) for i, era in enumerate(eras)]
    ipr_note = (" The single curve's scale was refitted (explicit request)." if request.refit_ipr
                else " The saved curve is held.")
    notes = [
        "One oil IPR describes the well across every pump; each test uses its own measured WC/GOR." + ipr_note,
        "ken, knz and the Mach limit are held. Replacement pumps still use reference losses; "
        "these fits describe the installed history only.",
        "Held-out predictions come from refits on earlier tests only (3-day embargo). A new pump is predicted "
        "at the prior nozzle factor because its own tests are not yet seen.",
        "Uncertainties are model-conditional (Laplace approximation, inflated by misfit). Structural error, "
        "such as the known discharge deficit, is outside them.",
    ] + list(prep["notes"])
    result = dict(
        well=well, request=request.model_dump(), physics_model=physics_model(request.hydraulics_model),
        as_of=prep["as_of"], source=prep["source"], notes=notes, eras=era_rows, rows=prep["rows"],
        ipr_gate=ladder["ipr_gate"], losses=ladder["losses"], skipped=ladder["skipped"],
        models=ladder["models"], cv=cv, selected=selection,
        seconds=time.perf_counter() - started, validated_for_sizing=False,
        snapshot_id=(key or sha256(pickle.dumps(inputs)).digest()).hex(),
    )
    result = _jsonable(result)
    surface_cache.store_snapshot(key, result)
    return result


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def start(well: str, request) -> str:
    from server import jobs

    return jobs.start(KINDS[0], lambda job: run(job, well, request))
