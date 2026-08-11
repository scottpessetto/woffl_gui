"""server/services/evidence.py - field-evidence suction response mining.

All synthetic (no network): well_evidence is PURE, and the pad_evidence
tests monkeypatch the fleet frame / test-BHP / install-date fetchers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import server.services.evidence as evidence
from server.services.evidence import (
    BETA_DEFAULT,
    MIN_PAIRS,
    _apply_pad_fallback,
    pad_evidence,
    well_evidence,
)

_START = pd.Timestamp("2026-01-01")


def _rows(bhp_by_day, ppf_by_day, tubing=250.0):
    """Daily rows in the fleet-query shape (reverse circ: annulus is PF)."""
    return [
        {
            "sample_date": _START + pd.Timedelta(days=i),
            "tubing_prs": tubing,
            "inn_ann_prs": ppf,
            "btmhole_prs": bhp,
        }
        for i, (bhp, ppf) in enumerate(zip(bhp_by_day, ppf_by_day))
    ]


def _stepped_series(levels, days_per_level, bhp_fn, noise=None):
    """Alternating PF levels so 3-30 day pairs see |dPpf| >= 100 psi."""
    ppf = [lvl for lvl in levels for _ in range(days_per_level)]
    bhp = [bhp_fn(p) for p in ppf]
    if noise is not None:
        bhp = [b + n for b, n in zip(bhp, noise)]
    return _rows(bhp, ppf)


# ---------------------------------------------------------------------------
# Floor + filter chain
# ---------------------------------------------------------------------------


def test_floor_is_p5_of_flowing_bhp_minned_with_test_min():
    bhps = list(np.linspace(300.0, 400.0, 40))
    rows = _rows(bhps, [3000.0] * 40)

    ev = well_evidence(rows)
    assert ev is not None
    assert ev["floor"] == pytest.approx(pd.Series(bhps).quantile(0.05))
    assert ev["n_days"] == 40
    assert ev["window"] == ["2026-01-01", "2026-02-09"]
    # psu_ref: median of the LAST 14 flowing days
    assert ev["psu_ref"] == pytest.approx(pd.Series(bhps[-14:]).median())

    # a lower well-test BHP wins the min
    ev_test = well_evidence(rows, min_test_bhp=250.0)
    assert ev_test["floor"] == 250.0
    # a higher one does not
    ev_hi = well_evidence(rows, min_test_bhp=390.0)
    assert ev_hi["floor"] == pytest.approx(pd.Series(bhps).quantile(0.05))


def test_glitch_bhp_row_dropped():
    """A dead-gauge 29.5 psi day must not become the measured floor."""
    bhps = [320.0] * 20 + [29.5]
    rows = _rows(bhps, [3000.0] * 21)
    ev = well_evidence(rows)
    assert ev["n_days"] == 20
    assert ev["floor"] == pytest.approx(320.0)


def test_days_without_valid_ppf_dropped():
    """Shut-in / dead PF gauge days (no reading >= 800 psi) are not flowing."""
    bhps = [310.0] * 10 + [500.0] * 5
    # last 5 days: annulus dead (0) and tubing at production pressure (250)
    ppfs = [3000.0] * 10 + [0.0] * 5
    ev = well_evidence(_rows(bhps, ppfs))
    assert ev["n_days"] == 10
    assert ev["psu_ref"] == pytest.approx(310.0)  # the 500s never entered


def test_bhp_at_or_above_res_pres_dropped():
    """Shut-in buildup days (BHP >= reservoir pressure) are not flowing."""
    bhps = [320.0] * 10 + [1550.0, 1500.0]
    rows = _rows(bhps, [3000.0] * 12)
    ev = well_evidence(rows, res_pres=1500.0)
    assert ev["n_days"] == 10
    assert ev["floor"] == pytest.approx(320.0)
    # without the fit pressure the filter cannot run
    assert well_evidence(rows)["n_days"] == 12


def test_no_surviving_days_returns_none():
    rows = _rows([29.0, 30.0], [3000.0, 3000.0])
    assert well_evidence(rows) is None
    assert well_evidence([]) is None


# ---------------------------------------------------------------------------
# Beta (Theil-Sen median of pairwise slopes)
# ---------------------------------------------------------------------------


def test_theil_sen_recovers_synthetic_beta():
    """bhp = 800 - 0.087 * ppf + noise across PF cuts -> beta ~ 0.087."""
    rng = np.random.default_rng(7)
    levels = [3300.0, 2900.0, 3250.0, 2700.0, 3100.0, 2600.0]
    rows = _stepped_series(
        levels, 6, lambda p: 800.0 - 0.087 * p, noise=rng.normal(0.0, 2.0, 36)
    )
    ev = well_evidence(rows)
    assert ev["n_pairs"] >= MIN_PAIRS
    assert ev["beta_source"] == "well"
    assert ev["beta"] == pytest.approx(0.087, abs=0.01)


def test_pairs_spanning_an_install_date_excluded():
    """The BHP shift across a JPCO is the pump changing, not PF response."""
    # 10 days at (3300 PF, 300 BHP), JPCO, 10 days at (2900 PF, 500 BHP).
    # Every |dPpf| >= 100 pair crosses the install; within a segment PF is
    # flat, so the guard leaves NO qualifying pairs.
    rows = _rows([300.0] * 10 + [500.0] * 10, [3300.0] * 10 + [2900.0] * 10)
    install = _START + pd.Timedelta(days=10)

    unguarded = well_evidence(rows)
    assert unguarded["n_pairs"] >= MIN_PAIRS  # the confounded pairs exist

    ev = well_evidence(rows, install_dates=[install])
    assert ev["n_pairs"] == 0
    assert ev["beta_source"] == "default"
    assert ev["beta"] == BETA_DEFAULT


def test_beta_clamped_at_zero_for_wells_whose_bhp_falls_on_cuts():
    rows = _stepped_series(
        [3300.0, 2900.0, 3300.0, 2900.0], 5, lambda p: 100.0 + 0.1 * p
    )
    ev = well_evidence(rows)
    assert ev["beta_source"] == "well"
    assert ev["beta"] == 0.0


def test_beta_clamped_at_half():
    rows = _stepped_series(
        [3300.0, 2900.0, 3300.0, 2900.0], 5, lambda p: 3400.0 - 0.9 * p
    )
    ev = well_evidence(rows)
    assert ev["beta_source"] == "well"
    assert ev["beta"] == 0.5


# ---------------------------------------------------------------------------
# Fallback chain: well -> pad -> default
# ---------------------------------------------------------------------------


def test_pad_fallback_upgrades_defaults_to_pad_median():
    rows = {
        "MPM-01": {"beta": 0.12, "beta_source": "well", "n_pairs": 9},
        "MPM-02": {"beta": 0.10, "beta_source": "well", "n_pairs": 7},
        "MPM-03": {"beta": BETA_DEFAULT, "beta_source": "default", "n_pairs": 2},
    }
    out = _apply_pad_fallback(rows)
    assert out["MPM-03"]["beta"] == pytest.approx(0.11)
    assert out["MPM-03"]["beta_source"] == "pad"
    # earned betas untouched
    assert out["MPM-01"] == {"beta": 0.12, "beta_source": "well", "n_pairs": 9}


def test_pad_fallback_keeps_default_when_no_well_earns_a_beta():
    rows = {
        "MPM-03": {"beta": BETA_DEFAULT, "beta_source": "default", "n_pairs": 2},
        "MPM-04": {"beta": BETA_DEFAULT, "beta_source": "default", "n_pairs": 0},
    }
    out = _apply_pad_fallback(rows)
    for r in out.values():
        assert r["beta"] == BETA_DEFAULT
        assert r["beta_source"] == "default"


# ---------------------------------------------------------------------------
# pad_evidence assembly (fetchers monkeypatched - still no network)
# ---------------------------------------------------------------------------


def _fleet_frame(per_well: dict[str, list[dict]]) -> pd.DataFrame:
    frames = []
    for well, rows in per_well.items():
        df = pd.DataFrame(rows)
        df["well"] = well
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture()
def offline(monkeypatch):
    monkeypatch.setattr(evidence, "_min_test_bhp", lambda well: None)
    monkeypatch.setattr(evidence, "_install_dates_by_well", lambda: {})
    return monkeypatch


def test_pad_evidence_sparse_well_inherits_pad_median(offline):
    rng = np.random.default_rng(3)
    levels = [3300.0, 2900.0, 3250.0, 2700.0, 3100.0, 2600.0]
    strong = _stepped_series(
        levels, 6, lambda p: 800.0 - 0.11 * p, noise=rng.normal(0.0, 1.0, 36)
    )
    sparse = _rows([320.0] * 8, [3000.0] * 8)  # flat PF -> no pairs
    offline.setattr(
        evidence,
        "_fleet_pressure_daily",
        lambda: _fleet_frame({"MPM-01": strong, "MPM-02": sparse}),
    )

    ev = pad_evidence(["MPM-01", "MPM-02", "MPM-99"])
    assert set(ev) == {"MPM-01", "MPM-02"}  # MPM-99 has no rows -> absent
    assert ev["MPM-01"]["beta_source"] == "well"
    assert ev["MPM-02"]["beta_source"] == "pad"
    assert ev["MPM-02"]["beta"] == pytest.approx(ev["MPM-01"]["beta"])
    assert ev["MPM-02"]["n_pairs"] < MIN_PAIRS


def test_pad_evidence_all_sparse_falls_back_to_default(offline):
    sparse = _rows([320.0] * 8, [3000.0] * 8)
    offline.setattr(
        evidence, "_fleet_pressure_daily", lambda: _fleet_frame({"MPM-02": sparse})
    )
    ev = pad_evidence(["MPM-02"])
    assert ev["MPM-02"]["beta_source"] == "default"
    assert ev["MPM-02"]["beta"] == BETA_DEFAULT


def test_pad_evidence_well_that_raises_is_absent_not_fatal(offline):
    good = _rows([320.0] * 8, [3000.0] * 8)
    offline.setattr(
        evidence,
        "_fleet_pressure_daily",
        lambda: _fleet_frame({"MPM-01": good, "MPM-02": good}),
    )
    real = evidence.well_evidence

    def flaky(daily, **kwargs):
        if set(daily["well"]) == {"MPM-02"}:
            raise RuntimeError("corrupt frame")
        return real(daily, **kwargs)

    offline.setattr(evidence, "well_evidence", flaky)
    ev = pad_evidence(["MPM-01", "MPM-02"])
    assert set(ev) == {"MPM-01"}


def test_pad_evidence_passes_res_pres_through(offline):
    # 10 flowing days + 2 buildup days at 1550; the res_pres map drops them
    rows = _rows([320.0] * 10 + [1550.0] * 2, [3000.0] * 12)
    offline.setattr(
        evidence, "_fleet_pressure_daily", lambda: _fleet_frame({"MPM-01": rows})
    )
    ev = pad_evidence(["MPM-01"], res_pres={"MPM-01": 1500.0})
    assert ev["MPM-01"]["n_days"] == 10
