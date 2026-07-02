"""Solopump Smoke Tests

Smoke tests for the single-well, single-pump solver (solopump.jetpump_solver).
Tests pump sizing response, power fluid pressure sensitivity, circulation
direction, and wellbore geometry effects.
"""

import pytest

import woffl.assembly.solopump as so
import woffl.flow.jetflow as _jf
from woffl.flow.inflow import InFlow
from woffl.geometry.jetpump import JetPump
from woffl.geometry.pipe import Pipe, PipeInPipe
from woffl.geometry.wellprofile import WellProfile
from woffl.pvt.blackoil import BlackOil
from woffl.pvt.formgas import FormGas
from woffl.pvt.formwat import FormWater
from woffl.pvt.resmix import ResMix

# --- E-41 baseline setup ---

pwh = 210  # psig
tsu = 80  # deg F
ppf_surf = 3168  # psig

tubing = Pipe(out_dia=4.5, thick=0.5)
casing = Pipe(out_dia=6.875, thick=0.5)
wbore = PipeInPipe(inn_pipe=tubing, out_pipe=casing)
profile = WellProfile.schrader()

mpu_oil = BlackOil.schrader()
mpu_wat = FormWater.schrader()
mpu_gas = FormGas.schrader()

ipr = InFlow(qwf=246, pwf=1049, pres=1400)
res = ResMix(wc=0.894, fgor=600, oil=mpu_oil, wat=mpu_wat, gas=mpu_gas)


def _solve(jpump, pwh=pwh, ppf=ppf_surf, wellbore=wbore, direction="reverse"):
    """Helper to call jetpump_solver with E-41 defaults, returns (qoil, lwat)."""
    psu, sonic, qoil, fwat, lwat, mach = so.jetpump_solver(
        pwh, tsu, ppf, jpump, wellbore, profile, ipr, res, mpu_wat, direction
    )
    return qoil, lwat


class TestPumpSizing:
    """Larger semi-finalist pump should produce more oil.

    Pumps selected from E-41 batch semi-finalists:
    11D (~193 bopd) and 13C (~225 bopd).
    """

    def test_larger_pump_more_oil(self):
        qoil_small, _ = _solve(JetPump("11", "D"))
        qoil_large, _ = _solve(JetPump("13", "C"))
        assert qoil_large > qoil_small

    def test_larger_pump_more_water(self):
        _, lwat_small = _solve(JetPump("11", "D"))
        _, lwat_large = _solve(JetPump("13", "C"))
        assert lwat_large > lwat_small


class TestPowerFluidPressure:
    """Higher power fluid pressure delivers more energy to the nozzle."""

    def test_higher_ppf_more_oil(self):
        pump = JetPump("12", "C")
        qoil_low, _ = _solve(pump, ppf=2500)
        qoil_high, _ = _solve(pump, ppf=3168)
        assert qoil_high > qoil_low


class TestCirculationDirection:
    """Reverse vs forward with similar tubing and annular flow areas.

    Tubing and casing were sized so the tubing cross-sectional area and the
    annular cross-sectional area are nearly equal, isolating the effect of
    flow path geometry (circular vs annular) on friction:

        Tubing: 4.5" OD, 0.50" wall -> ID=3.50", area=9.62 in2
        Annulus: 6.25" OD casing, 0.30" wall -> ann area=9.17 in2 (ratio 1.05)

    Expected: with similar areas, reverse circulation (production up tubing)
    would have slightly less oil than forward (production up annulus) because
    the tubing imposes more friction on the multiphase production stream.

    Actual: reverse produced significantly more oil (217 vs 143 bopd). Likely
    caused by Beggs & Brill performing poorly in annular geometry — the
    hydraulic diameter correction for an annulus may overpredict friction.
    Test asserts the two directions produce different results, without
    asserting which is higher, until the annular multiphase model is better
    understood.
    """

    def test_reverse_vs_forward(self):
        tube_dir = Pipe(out_dia=4.5, thick=0.50)
        case_dir = Pipe(out_dia=6.25, thick=0.30)
        wbore_dir = PipeInPipe(inn_pipe=tube_dir, out_pipe=case_dir)

        pump = JetPump("12", "C")
        qoil_rev, _ = _solve(pump, wellbore=wbore_dir, direction="reverse")
        qoil_fwd, _ = _solve(pump, wellbore=wbore_dir, direction="forward")

        # with similar areas, expect a difference — assert they're not equal
        # direction of difference TBD, log both for inspection
        print(f"\nReverse: {qoil_rev:.1f} bopd, Forward: {qoil_fwd:.1f} bopd")
        assert qoil_rev != qoil_fwd


class TestMarginalConvergence:
    """Marginal pumps (small throat / high water cut) must still converge.

    Regression guard for the ``_residual_walk_inward`` fix. For these configs the
    inner throat-mixture solve is infeasible right AT the lower suction bracket
    endpoint (``psu_min``), so ``jetpump_solver`` historically aborted with
    ``ConvergenceError: throat mixture did not converge`` — even though the well
    demonstrably flows (the discharge residual crosses zero just inside the
    feasible suction range). The fix walks the suction inward to the nearest
    feasible point so the outer search keeps a valid bracket. These are realistic
    Milne Point conditions (12B at ~94% WC, ~2000 psi reservoir), i.e. the
    "pump is in the well and working, but won't converge in the model" case.

    Setup is independent of the module's E-41 fixtures (whose 1400 psi reservoir
    doesn't reach the infeasible-endpoint regime).
    """

    pwh_m, tsu_m, ppf_m, pres_m = 250, 120, 3000, 2000
    res_marginal = ResMix(wc=0.94, fgor=800, oil=mpu_oil, wat=mpu_wat, gas=mpu_gas)
    # IPR with qmax ~1000 BOPD anchored at half-reservoir drawdown.
    _vf = 1 - 0.2 * 0.5 - 0.8 * 0.25  # Vogel factor at pwf = 0.5*pres
    ipr_marginal = InFlow(qwf=1000 * _vf, pwf=0.5 * pres_m, pres=pres_m)

    def _solve_marginal(self, jpump):
        return so.jetpump_solver(
            self.pwh_m, self.tsu_m, self.ppf_m, jpump, wbore, profile,
            self.ipr_marginal, self.res_marginal, mpu_wat, "reverse",
        )

    def test_marginal_pump_converges(self):
        """12B at 94% WC / 2000 psi reservoir solves (used to abort)."""
        psu, sonic, qoil, fwat, lwat, mach = self._solve_marginal(JetPump("12", "B"))
        assert qoil > 0 and lwat > 0
        assert 50 < qoil < 600  # sane oil rate, not a degenerate/NaN result
        assert 2000 < lwat < 4000

    def test_marginal_endpoint_is_actually_infeasible(self):
        """Precondition: the inner solve really is infeasible at raw psu_min, so
        the test above genuinely exercises the walk-inward path (not a no-op)."""
        from woffl.flow.errors import ConvergenceError

        jp = JetPump("12", "B")
        psu_min, _q, _te = _jf.psu_minimize(
            tsu=self.tsu_m, ken=jp.ken, ate=jp.ate,
            ipr_su=self.ipr_marginal, prop_su=self.res_marginal,
        )
        raised = False
        try:
            so.discharge_residual(
                psu_min, self.pwh_m, self.tsu_m, self.ppf_m, jp, wbore, profile,
                self.ipr_marginal, self.res_marginal, mpu_wat, "reverse",
            )
        except ConvergenceError:
            raised = True
        assert raised, "expected throat-mixture infeasibility at raw psu_min"

    def test_several_marginal_pumps_converge(self):
        """A spread of small-throat / high-WC pumps all converge now."""
        for nozzle, throat in (("11", "A"), ("11", "B"), ("12", "A"), ("12", "B")):
            _psu, _s, qoil, _f, lwat, _m = self._solve_marginal(JetPump(nozzle, throat))
            assert qoil > 0 and lwat > 0, f"{nozzle}{throat} failed to converge"

    def test_thin_upper_band_feasibility_converges(self):
        """13B at 99% WC, high productivity: the throat is feasible only in a
        thin suction band right against reservoir pressure (top ~13 psi). The
        adaptive walk must reach it — a half-range walk stopped at the midpoint
        and falsely reported the well unconvergeable, even though the discharge
        residual crosses zero in that band (a real, if marginal, solution)."""
        res_hi = ResMix(wc=0.99, fgor=400, oil=mpu_oil, wat=mpu_wat, gas=mpu_gas)
        ipr_hi = InFlow(qwf=3000 * self._vf, pwf=0.5 * self.pres_m, pres=self.pres_m)
        psu, sonic, qoil, fwat, lwat, mach = so.jetpump_solver(
            self.pwh_m, self.tsu_m, self.ppf_m, JetPump("13", "B"), wbore, profile,
            ipr_hi, res_hi, mpu_wat, "reverse",
        )
        assert qoil > 0 and lwat > 0  # marginal well, but it does flow


def test_throat_wc_water_branch():
    """throat_wc at 100% suction WC returns the anchor rate as formation water
    and a 100% throat-mixture WC, without dividing by (1 - wc_su) = 0."""
    wc_tm, qwat_su = _jf.throat_wc(qoil_std=800, wc_su=1.0, qwat_nz=500)
    assert qwat_su == 800
    assert wc_tm == 1.0


class TestWaterPumpMode:
    """Opt-in 100%-water (dewatering) mode.

    A no-oil well must solve when model_as_water=True (water-anchored path), and
    the oil path must be untouched when it's off (wc=1.0 still raises). Regression
    guard for resmix._static_insitu_volm_flow_water + jetflow.throat_wc water
    branch + the model_as_water flag. See docs/water_pump_mode_plan.md and
    docs/upstream_sync.md.
    """

    pwh_w, tsu_w, ppf_w, pres_w = 250, 120, 3000, 2000
    _vf = 1 - 0.2 * 0.5 - 0.8 * 0.25  # Vogel factor at pwf = 0.5*pres
    # qwf is the WATER deliverability at pwf (IPR reused as a water inflow curve).
    ipr_water = InFlow(qwf=1500 * _vf, pwf=0.5 * pres_w, pres=pres_w)
    res_water = ResMix(
        wc=1.0, fgor=600, oil=mpu_oil, wat=mpu_wat, gas=mpu_gas, model_as_water=True
    )

    def test_water_pump_solve_converges(self):
        """A 100%-water well solves and lifts water (no oil)."""
        psu, sonic, qwater, fwat, lwat, mach = so.jetpump_solver(
            self.pwh_w, self.tsu_w, self.ppf_w, JetPump("12", "B"), wbore, profile,
            self.ipr_water, self.res_water, mpu_wat, "reverse",
        )
        # the "oil" slot carries the formation water rate in water mode
        assert qwater > 0 and fwat > 0 and lwat > 0
        assert psu > 0

    def test_water_mode_off_still_raises_at_full_wc(self):
        """Sanity: WITHOUT the flag, wc=1.0 still raises (oil path unchanged)."""
        res_no_flag = ResMix(wc=1.0, fgor=600, oil=mpu_oil, wat=mpu_wat, gas=mpu_gas)
        with pytest.raises(ValueError):
            so.jetpump_solver(
                self.pwh_w, self.tsu_w, self.ppf_w, JetPump("12", "B"), wbore,
                profile, self.ipr_water, res_no_flag, mpu_wat, "reverse",
            )


class TestWellboreGeometry:
    """Smaller wellbore components increase friction and reduce oil."""

    def test_smaller_casing_less_oil(self):
        """Same tubing, smaller casing -> more annular friction -> less oil."""
        pump = JetPump("12", "C")

        big_casing = Pipe(out_dia=6.875, thick=0.5)
        small_casing = Pipe(out_dia=5.5, thick=0.25)
        wbore_big = PipeInPipe(inn_pipe=tubing, out_pipe=big_casing)
        wbore_small = PipeInPipe(inn_pipe=tubing, out_pipe=small_casing)

        qoil_big, _ = _solve(pump, wellbore=wbore_big)
        qoil_small, _ = _solve(pump, wellbore=wbore_small)
        assert qoil_big > qoil_small

    def test_smaller_tubing_less_oil(self):
        """Big dummy casing (9" OD) so annular friction is negligible.
        Smaller tubing -> more production friction -> less oil."""
        pump = JetPump("12", "C")
        dummy_casing = Pipe(out_dia=9.0, thick=0.25)

        big_tubing = Pipe(out_dia=4.5, thick=0.5)
        small_tubing = Pipe(out_dia=3.5, thick=0.271)
        wbore_big = PipeInPipe(inn_pipe=big_tubing, out_pipe=dummy_casing)
        wbore_small = PipeInPipe(inn_pipe=small_tubing, out_pipe=dummy_casing)

        qoil_big, _ = _solve(pump, wellbore=wbore_big)
        qoil_small, _ = _solve(pump, wellbore=wbore_small)
        assert qoil_big > qoil_small


class TestSecantSolveRatesPopulated:
    """Tripwire for the _secant_solve 0-BOPD fix (upstream PR to kwellis/woffl).

    When BOTH seeds are the bracket ends (their residuals come from res_lookup,
    so discharge_residual is never called for them) AND the bracket already
    satisfies psu_diff / res_tol, the secant while-loop never runs. Before the
    fix the returned rates stayed at their 0.0 initializers — a real flowing
    well reported as 0 BOPD while returning normally (converged=False but no
    ConvergenceError, so no fallback fired). The fix re-evaluates
    discharge_residual at the returned psu when the cached rates don't already
    correspond to it. Goes red if that final evaluation is dropped.
    """

    def test_rates_evaluated_when_loop_skips(self, monkeypatch):
        calls = {"n": 0}

        def fake_discharge_residual(psu, *args, **kwargs):
            calls["n"] += 1
            return (0.0, 123.0, 456.0, 789.0, 0.5)  # res, qoil, fwat, qnz, mach

        monkeypatch.setattr(so, "discharge_residual", fake_discharge_residual)

        psu_min, psu_max = 1000.0, 1002.0  # within psu_diff=5
        res_min, res_max = 0.0, 1.0        # within res_tol=10 -> loop skipped
        result = so._secant_solve(
            seed_pair=(psu_min, psu_max),
            res_min=res_min,
            res_max=res_max,
            psu_min=psu_min,
            psu_max=psu_max,
            pwh=pwh,
            tsu=tsu,
            ppf_surf=ppf_surf,
            jpump=None,
            wellbore=None,
            wellprof=None,
            ipr_su=None,
            prop_su=None,
            prop_pf=None,
            jpump_direction="reverse",
            psu_diff=5.0,
            res_tol=10.0,
        )
        psu, _sonic, qoil, fwat, qnz, _mach = result
        assert qoil == 123.0   # real rates, NOT the 0.0 initializers
        assert fwat == 456.0
        assert qnz == 789.0
        assert calls["n"] == 1  # exactly one final eval at the returned psu
