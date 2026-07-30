"""Regression tests for the pad Configure screen's pre-flight match-check cache key.

The match check models every well at its current pump and reports ✓ / ⚠ / ✗ BUST
against the measured tests — it is the screen that tells the engineer whether the
model can be trusted before they run the optimizer. Its result is memoized in
session state under a signature.

That signature used to be a hand-rolled tuple covering only
``(n_pumps, nozzle, throat, qwf, pwf, res_pres)``. Every OTHER physical and
calibration field that ``wrs.store_to_well_configs`` feeds the model — water cut,
GOR, formation temperature, PF pressure, friction coefficients, tubing/casing
geometry, circulation direction, the PVT overrides — was absent, so editing any
of them in Review and returning to Configure served the STALE verdict computed
under the old values.

These tests pin every one of those fields to the signature. If a future edit
narrows the key back down, they go red.
"""

import pytest

from woffl.gui.pad_page import matchcheck_signature
from woffl.gui.params import SimulationParams
from woffl.gui.workflow_steps import well_review_store as wrs

PUMPS = {"MPS-01": ("12", "B"), "MPS-02": ("13", "C")}


def _params(well="MPS-01", **overrides) -> SimulationParams:
    base = dict(
        selected_well=well,
        qwf=400,
        pwf=900,
        pres=1800,
        form_wc=0.50,
        form_gor=250,
        form_temp=160,
        jpump_tvd=4200,
    )
    base.update(overrides)
    return SimulationParams(**base)


def _store(**overrides) -> dict:
    """Two-well active store; ``overrides`` patch MPS-01's entry."""
    store = {}
    for well, (noz, thr) in PUMPS.items():
        entry = wrs.snapshot_from_params(_params(well), ipr_source="vogel")
        entry["review_nozzle"], entry["review_throat"] = noz, thr
        store[well] = entry
    store["MPS-01"].update(overrides)
    return store


def _sig(store, n_pumps=3, pumps=None):
    pumps = pumps or PUMPS
    current = {w: pumps[w] for w in store}
    return matchcheck_signature(n_pumps, current, store)


def test_identical_state_gives_identical_signature():
    assert _sig(_store()) == _sig(_store())


def test_signature_is_order_independent():
    a = _store()
    b = {w: a[w] for w in reversed(list(a))}
    assert _sig(a) == _sig(b)


def test_pump_count_participates():
    assert _sig(_store(), n_pumps=3) != _sig(_store(), n_pumps=4)


def test_current_pump_participates():
    other = {**PUMPS, "MPS-01": ("14", "D")}
    assert _sig(_store()) != _sig(_store(), pumps=other)


def test_well_set_participates():
    smaller = _store()
    smaller.pop("MPS-02")
    assert _sig(_store()) != _sig(smaller, pumps=PUMPS)


# ── the fields the old hand-rolled tuple silently dropped ────────────────────
#
# Each of these reaches the model through store_to_well_configs(); before the
# fix, changing any of them left the match-check verdict stale.


@pytest.mark.parametrize(
    "field,new_value",
    [
        ("form_wc", 0.85),  # the headline miss — WC drives oil AND water
        ("form_gor", 900.0),
        ("form_temp", 185.0),
        ("ppf_surf_well", 3200.0),  # live PF pressure
        ("ken_well", 0.06),  # friction calibration
        ("kth_well", 0.42),
        ("kdi_well", 0.25),
        ("knz_well", 0.05),
        ("tubing_od", 3.5),
        ("tubing_thickness", 0.35),
        ("casing_od", 9.625),
        ("casing_thickness", 0.45),
        ("jpump_tvd", 5100.0),
        ("jpump_direction", "forward"),  # forward vs reverse circ
        ("field_model", "Kuparuk"),
        ("oil_api", 22.5),
        ("gas_sg", 0.72),
        ("wat_sg", 1.02),
        ("bubble_point", 2100.0),
        ("surf_pres", 260.0),
    ],
)
def test_physical_field_change_invalidates_signature(field, new_value):
    baseline = _sig(_store())
    changed = _sig(_store(**{field: new_value}))
    assert changed != baseline, (
        f"{field!r} does not participate in the match-check signature — "
        "editing it in Review would serve a stale ✓/✗ BUST verdict"
    )


def test_ipr_fields_still_participate():
    """The three fields the old tuple DID cover must keep participating."""
    for field, value in (("qwf", 1234.0), ("pwf", 777.0), ("res_pres", 2100.0)):
        assert _sig(_store(**{field: value})) != _sig(_store()), field


def test_sub_unit_rate_edit_is_not_truncated_away():
    """The old key cast qwf/pwf through int(), so sub-1-BPD edits vanished."""
    base = _store()
    nudged = _store(qwf=float(base["MPS-01"]["qwf"]) + 0.4)
    assert _sig(nudged) != _sig(base)


def test_cosmetic_fields_do_not_invalidate():
    """Notes/provenance don't change the model, so they must not bust the cache
    (otherwise the check re-runs a full batch sweep on every keystroke)."""
    assert _sig(_store(notes="checked with field ops")) == _sig(_store())
