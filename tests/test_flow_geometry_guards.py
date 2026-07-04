"""Regression tests for the P1-6 / P1-7 silent-NaN and silent-garbage guards.

P1-7 — ``jetplot.JetBook._dete_zero`` interpolated over a possibly non-contiguous
``dtdp >= 0`` mask: flipping a non-contiguous masked tde_ray hands ``np.interp``
a non-sorted xp and it silently returns garbage pte/vte/rho_te/mach_te. The fix
keeps only the LEADING contiguous run of the mask (the subsonic branch). The same
latent assumption in ``dedi_zero`` now raises the module's typed error instead of
returning garbage.

P1-6 — ``wellprofile.WellProfile._horz_dist`` took ``sqrt(dmd**2 - dvd**2)``;
survey noise with |dvd| > |dmd| produced a silent NaN that cumsum propagated
through the whole hd_ray (and jetpump_hd / incline downstream). The fix clamps
the dvd magnitude to dmd (physically |dvd| <= |dmd| along a wellbore). Same
domain guard on ``forms.horz_angle``'s arcsin.

Both fixes are bit-identical for already-valid inputs — proven here by inlining
the pre-fix formulas and comparing exactly.

These guards are LOCAL LIBRARY PATCHES pending an upstream PR to kwellis/woffl —
see docs/upstream_sync.md. If an upstream sync drops them, these tests go red.
"""

import numpy as np
import pytest

import woffl.flow.jetflow  # noqa: F401  (jetplot<->jetflow circular import: jetflow must load first)
from woffl.flow.errors import ConvergenceError, ThroatEntryNoSolution
from woffl.flow.jetplot import JetBook
from woffl.geometry.forms import horz_angle
from woffl.geometry.wellprofile import WellProfile

# ------------------------------------------------------------------
# P1-7: _dete_zero non-contiguous dtdp >= 0 mask
# ------------------------------------------------------------------

# Throat-entry style sweep: pressure descends from psu. The tde values are
# crafted so np.gradient(tde, prs) oscillates: the dtdp >= 0 mask comes out
# [T, T, T, F, T, T] — a non-contiguous mask like the ones np.gradient
# produces near the tde minimum / bubble-point kinks.
PRS_RAY = np.array([1000.0, 900.0, 800.0, 700.0, 600.0, 500.0])
VEL_RAY = np.array([100.0, 150.0, 200.0, 250.0, 300.0, 350.0])
RHO_RAY = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0])
MACH_RAY = np.array([0.2, 0.4, 0.6, 0.8, 0.9, 1.1])
TDE_NONCONTIG = np.array([50.0, 20.0, -10.0, 3.0, 5.0, 0.0])
TDE_CONTIG = np.array([60.0, 30.0, 5.0, -15.0, -10.0, 20.0])


def _old_dete_zero(prs, vel, rho, tde, mach):
    """The pre-fix _dete_zero interpolation, inlined verbatim."""
    dtdp = np.gradient(tde, prs)
    mask = dtdp >= 0
    tde_flipped = np.flip(tde[mask])
    pte = np.interp(0, tde_flipped, np.flip(prs[mask]))
    vte = np.interp(0, tde_flipped, np.flip(vel[mask]))
    rho_te = np.interp(0, tde_flipped, np.flip(rho[mask]))
    mach_te = np.interp(0, tde_flipped, np.flip(mach[mask]))
    return pte, vte, rho_te, mach_te


def test_dete_zero_noncontiguous_mask_uses_leading_branch():
    """A non-contiguous mask previously fed np.interp a non-sorted xp (silent
    garbage); now only the leading contiguous run — the subsonic branch — is
    interpolated."""
    mask = np.gradient(TDE_NONCONTIG, PRS_RAY) >= 0
    assert mask.tolist() == [True, True, True, False, True, True]  # non-contiguous

    # the old formula's xp is NOT sorted -> np.interp result is meaningless
    old_xp = np.flip(TDE_NONCONTIG[mask])
    assert not np.all(np.diff(old_xp) >= 0)
    old_pte = _old_dete_zero(PRS_RAY, VEL_RAY, RHO_RAY, TDE_NONCONTIG, MACH_RAY)[0]

    pte, vte, rho_te, mach_te = JetBook._dete_zero(
        PRS_RAY, VEL_RAY, RHO_RAY, TDE_NONCONTIG, MACH_RAY
    )

    # expected: interpolation over the leading run only (indices 0..2)
    lead = slice(0, 3)
    tde_lead = np.flip(TDE_NONCONTIG[lead])
    assert pte == np.interp(0, tde_lead, np.flip(PRS_RAY[lead]))
    assert vte == np.interp(0, tde_lead, np.flip(VEL_RAY[lead]))
    assert rho_te == np.interp(0, tde_lead, np.flip(RHO_RAY[lead]))
    assert mach_te == np.interp(0, tde_lead, np.flip(MACH_RAY[lead]))

    # the subsonic zero crossing sits between 800 and 900 psig
    assert 800.0 < pte < 900.0
    # and the old formula was returning something else entirely (garbage)
    assert old_pte != pte


def test_dete_zero_contiguous_mask_bit_identical():
    """A well-behaved (contiguous) mask must return exactly what the pre-fix
    formula returned — the guard is additive only."""
    mask = np.gradient(TDE_CONTIG, PRS_RAY) >= 0
    assert mask.tolist() == [True, True, True, True, False, False]  # contiguous

    old = _old_dete_zero(PRS_RAY, VEL_RAY, RHO_RAY, TDE_CONTIG, MACH_RAY)
    new = JetBook._dete_zero(PRS_RAY, VEL_RAY, RHO_RAY, TDE_CONTIG, MACH_RAY)
    for old_val, new_val in zip(old, new):
        assert old_val == new_val  # bit-identical, not approx


def test_dete_zero_negative_leading_branch_raises_typed():
    """When the leading (subsonic) run is entirely negative and only a later
    spurious run pokes positive, the old code interpolated garbage; now it
    raises the typed no-solution error."""
    tde = np.array([-5.0, -20.0, -40.0, -30.0, -32.0, -35.0])
    mask = np.gradient(tde, PRS_RAY) >= 0
    assert mask[0] and not mask.all()  # leading run exists and mask breaks
    with pytest.raises(ThroatEntryNoSolution):
        JetBook._dete_zero(PRS_RAY, VEL_RAY, RHO_RAY, tde, MACH_RAY)


# ------------------------------------------------------------------
# P1-7 (same latent assumption): dedi_zero ascending-xp guard
# ------------------------------------------------------------------


def _make_book(prs_ray, tde_ray):
    """Build a JetBook and overwrite the arrays dedi_zero reads."""
    book = JetBook(prs_ray[0], 10.0, 50.0, 1000.0, 0.0)
    book.prs_ray = np.asarray(prs_ray, dtype=float)
    book.tde_ray = np.asarray(tde_ray, dtype=float)
    return book


def test_dedi_zero_monotonic_bit_identical():
    prs = np.array([500.0, 600.0, 700.0, 800.0])
    tde = np.array([-30.0, -10.0, 5.0, 25.0])
    book = _make_book(prs, tde)
    assert book.dedi_zero() == np.interp(0, tde, prs)  # bit-identical


def test_dedi_zero_non_monotonic_raises_typed():
    prs = np.array([500.0, 600.0, 700.0, 800.0])
    tde = np.array([-30.0, 5.0, -10.0, 25.0])  # not ascending -> interp garbage
    book = _make_book(prs, tde)
    with pytest.raises(ConvergenceError):
        book.dedi_zero()


# ------------------------------------------------------------------
# P1-6: noisy-survey horizontal distance
# ------------------------------------------------------------------

# clean sample survey (|dvd| <= |dmd| at every station)
CLEAN_MD = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
CLEAN_VD = [0, 95, 180, 270, 350, 375, 400, 410, 415, 420, 425]

# same survey with gauge noise at the first two stations: |dvd| > |dmd|
NOISY_VD = [0, 103, 180, 270, 350, 375, 400, 410, 415, 420, 425]


def _old_horz_dist(md_ray, vd_ray):
    """The pre-fix _horz_dist formula, inlined verbatim."""
    md_diff = np.diff(md_ray, n=1)
    vd_diff = np.diff(vd_ray, n=1)
    with np.errstate(invalid="ignore"):  # the old formula's silent NaN is the point
        hd_diff = np.append(np.zeros(1), np.sqrt(md_diff**2 - vd_diff**2))
    return np.cumsum(hd_diff)


def test_noisy_survey_hd_ray_finite_and_monotonic():
    """|dvd| > |dmd| noise previously put a NaN into hd_diff that cumsum
    smeared over the whole hd_ray; now the profile stays finite/monotonic."""
    md_ray = np.array(CLEAN_MD, dtype=float)
    vd_ray = np.array(NOISY_VD, dtype=float)

    # prove the old formula really went NaN on this input
    old_hd = _old_horz_dist(md_ray, vd_ray)
    assert np.isnan(old_hd[1:]).all()

    wp = WellProfile(CLEAN_MD, NOISY_VD, jetpump_md=450)
    assert np.isfinite(wp.hd_ray).all()
    assert np.all(np.diff(wp.hd_ray) >= 0)  # monotonic non-decreasing
    assert np.isfinite(wp.jetpump_hd)
    # the noisy station clamps to a purely-vertical step (zero hd gained)
    assert wp.hd_ray[1] == 0.0


def test_clean_survey_hd_ray_bit_identical():
    """Already-valid surveys must produce exactly the pre-fix hd_ray."""
    md_ray = np.array(CLEAN_MD, dtype=float)
    vd_ray = np.array(CLEAN_VD, dtype=float)
    assert np.array_equal(
        WellProfile._horz_dist(md_ray, vd_ray), _old_horz_dist(md_ray, vd_ray)
    )


def test_generic_profiles_bit_identical():
    """The packaged Schrader / Kuparuk surveys are clean; their hd_ray must be
    bit-identical to the pre-fix formula."""
    for wp in (WellProfile.schrader(), WellProfile.kuparuk()):
        assert np.array_equal(wp.hd_ray, _old_horz_dist(wp.md_ray, wp.vd_ray))
        assert np.isfinite(wp.hd_ray).all()


def test_nan_survey_rejected():
    """NaN depths are rejected loudly at construction instead of silently
    corrupting every interpolation downstream."""
    bad_vd = list(CLEAN_VD)
    bad_vd[3] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        WellProfile(CLEAN_MD, bad_vd, jetpump_md=450)


# ------------------------------------------------------------------
# P1-6: horz_angle arcsin domain clamp
# ------------------------------------------------------------------


def test_horz_angle_valid_bit_identical():
    assert horz_angle(100.0, 60.0) == np.degrees(np.arcsin(60.0 / 100.0))
    assert horz_angle(100.0, -60.0) == np.degrees(np.arcsin(-60.0 / 100.0))


def test_horz_angle_noise_clamped_not_nan():
    """|ylen| slightly over hlen (survey noise) previously returned NaN."""
    assert horz_angle(100.0, 100.5) == 90.0
    assert horz_angle(100.0, -100.5) == -90.0
