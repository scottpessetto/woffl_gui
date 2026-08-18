"""Tests for the pad-review OFFLINE flag - the "well is online but the old
jet-pump workflow shows it offline" class of bug.

Two independent mechanisms are covered:

1. **Sticky widget state.** The save panel and the Modeling-status expander
   both render a checkbox bound to the SAME ``entry["offline"]`` field, and
   ``_render_modeling_status`` writes its checkbox return straight back into the
   store on every rerun. Streamlit ignores a widget's ``value=`` once the
   widget's key exists in ``session_state``, so a store write that does not drop
   those keys is silently reverted on the next run - the well stays OFFLINE
   forever. ``_drop_offline_widget_state`` is what makes the store authoritative.

2. **Stale force-fit stub.** A well with no cached well tests is auto-saved
   OFFLINE with ``notes == _NO_TESTS_OFFLINE_NOTE``. That flag is a placeholder
   for "no targets to match", not an engineer's decision, so once the well
   matches (it therefore HAS tests) ``_apply_batch_row`` must clear it instead
   of carrying a dead reason forward on every future re-fit.

``srw.st`` is replaced with a fake (never a ``sys.modules`` MagicMock, which
would poison the real streamlit import for later modules in the suite).
"""

from types import SimpleNamespace

import pytest

from woffl.gui.workflow_steps import step_review_wells as srw


class _FakeCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeSt:
    """``st`` stand-in whose ``checkbox`` reproduces Streamlit's key
    precedence: once a widget key exists in session_state, ``value=`` is
    ignored and the persisted widget value is returned."""

    def __init__(self):
        self.session_state = {}
        self.captions: list[str] = []

    def expander(self, *a, **k):
        return _FakeCtx()

    def caption(self, text="", *a, **k):
        self.captions.append(str(text))

    def columns(self, n, **k):
        return [_FakeCtx() for _ in range(n)]

    def checkbox(self, label, value=False, key=None, **k):
        if key is not None and key in self.session_state:
            return self.session_state[key]
        if key is not None:
            self.session_state[key] = value
        return value


@pytest.fixture
def fake_st(monkeypatch):
    fs = _FakeSt()
    monkeypatch.setattr(srw, "st", fs)
    return fs


# ---------------------------------------------------------------------------
# 1. sticky widget state
# ---------------------------------------------------------------------------


def test_stale_offline_widget_key_clobbers_an_online_store_entry(fake_st):
    """The exact mechanism: a leftover checkbox key overwrites the store."""
    fake_st.session_state["sp_well_store_S"] = {"MPS-05": {"offline": False}}
    fake_st.session_state["sp_status_off_S_MPS-05"] = True

    srw._render_modeling_status("S")

    assert srw.store_for("S")["MPS-05"]["offline"] is True


def test_dropping_widget_state_lets_the_store_win(fake_st):
    """After the drop the checkbox re-seeds from the store, so an online save
    survives the next rerun."""
    fake_st.session_state["sp_well_store_S"] = {"MPS-05": {"offline": False}}
    fake_st.session_state["sp_status_off_S_MPS-05"] = True

    srw._drop_offline_widget_state("S", ["MPS-05"])
    srw._render_modeling_status("S")

    assert srw.store_for("S")["MPS-05"]["offline"] is False


def test_an_offline_well_shows_the_reason_it_was_excluded(fake_st):
    """The reported confusion was not knowing WHY a well showed offline."""
    fake_st.session_state["sp_well_store_S"] = {
        "MPS-05": {"offline": True, "notes": srw._NO_TESTS_OFFLINE_NOTE},
        "MPS-07": {"offline": False, "notes": "hand reviewed"},
    }

    srw._render_modeling_status("S")

    reasons = [c for c in fake_st.captions if c.startswith("\u21b3")]
    assert reasons == [f"\u21b3 {srw._NO_TESTS_OFFLINE_NOTE}"]


def test_an_offline_well_with_no_note_still_says_so(fake_st):
    fake_st.session_state["sp_well_store_S"] = {"MPS-05": {"offline": True}}

    srw._render_modeling_status("S")

    assert "\u21b3 no reason recorded" in fake_st.captions


def test_drop_offline_widget_state_pops_both_checkbox_keys(fake_st):
    """Both checkboxes mirror the same field; dropping one is not enough."""
    fake_st.session_state.update(
        {
            "sp_offline_MPS-05": True,
            "sp_status_off_S_MPS-05": True,
            "sp_offline_MPS-07": True,
            "sp_status_off_S_MPS-07": True,
        }
    )

    srw._drop_offline_widget_state("S", ["MPS-05"])

    assert "sp_offline_MPS-05" not in fake_st.session_state
    assert "sp_status_off_S_MPS-05" not in fake_st.session_state
    # Untouched wells keep their state.
    assert fake_st.session_state["sp_offline_MPS-07"] is True
    assert fake_st.session_state["sp_status_off_S_MPS-07"] is True


def test_drop_offline_widget_state_is_pad_scoped(fake_st):
    """The modeling-status key carries the pad; another pad's copy must stay."""
    fake_st.session_state.update(
        {
            "sp_status_off_S_MPS-05": True,
            "sp_status_off_I_MPS-05": True,
        }
    )

    srw._drop_offline_widget_state("S", ["MPS-05"])

    assert "sp_status_off_S_MPS-05" not in fake_st.session_state
    assert fake_st.session_state["sp_status_off_I_MPS-05"] is True


# ---------------------------------------------------------------------------
# 2. stale force-fit no-tests stub
# ---------------------------------------------------------------------------


def _raw() -> dict:
    return {
        "nozzle_no": "12",
        "area_ratio": "B",
        "jpump_direction": "reverse",
        "tubing_od": 4.5,
        "tubing_thickness": 0.5,
        "casing_od": 6.875,
        "casing_thickness": 0.5,
        "form_wc": 0.5,
        "form_gor": 250,
        "form_temp": 160,
        "field_model": "Schrader",
        "oil_api": None,
        "gas_sg": None,
        "wat_sg": None,
        "bubble_point": None,
        "surf_pres": 250,
        "jpump_tvd": 4200,
        "jpump_md": 4753.0,
        "has_bhp": False,
        "ipr_fallback": False,
    }


def _row(well: str = "MPS-05"):
    res = SimpleNamespace(
        ken=0.03,
        kth=0.3,
        kdi=0.4,
        ppf_surf=2600.0,
        qwf_oil=150.0,
        pwf=600.0,
        pres=1700.0,
    )
    return SimpleNamespace(well=well, result=res)


@pytest.fixture
def captured(monkeypatch):
    """Capture the ``offline`` kwarg ``_apply_batch_row`` decides on."""
    seen: dict = {}

    def _snap(params, **kw):
        seen.update(kw)
        return {
            "well_name": params.selected_well,
            "offline": kw["offline"],
            "notes": kw.get("notes", ""),
        }

    monkeypatch.setattr(srw.wrs, "snapshot_from_params", _snap)
    return seen


def test_no_tests_offline_stub_clears_when_the_well_finally_matches(captured):
    store = {"MPS-05": {"offline": True, "notes": srw._NO_TESTS_OFFLINE_NOTE}}

    assert srw._apply_batch_row("S", store, _row(), _raw()) is None

    assert captured["offline"] is False
    assert store["MPS-05"]["offline"] is False


def test_engineer_marked_offline_survives_a_rematch(captured):
    """Only the no-tests placeholder self-heals; a human decision is kept."""
    store = {"MPS-05": {"offline": True, "notes": "pulled 2026-08-01"}}

    assert srw._apply_batch_row("S", store, _row(), _raw()) is None

    assert captured["offline"] is True


def test_unknown_well_falls_back_to_the_pad_default(captured):
    """I-Pad 3/11/15 are the only default-offline wells; S-05 is not one."""
    store: dict = {}

    assert srw._apply_batch_row("S", store, _row("MPS-05"), _raw()) is None
    assert captured["offline"] is False

    assert srw._apply_batch_row("I", store, _row("MPI-03"), _raw()) is None
    assert captured["offline"] is True


def test_stub_entry_stamps_the_note_the_self_heal_matches(monkeypatch):
    """Round-trip invariant: if these two strings ever drift, the stale-offline
    flag becomes permanent again and nothing fails loudly."""
    import woffl.gui.utils as utils_mod

    monkeypatch.setattr(
        utils_mod, "get_well_data", lambda w: {"JP_TVD": 4200, "is_sch": True}
    )

    entry = srw._offline_stub_entry("MPS-05")

    assert entry["offline"] is True
    assert entry["notes"] == srw._NO_TESTS_OFFLINE_NOTE
    # The force-fit "protected, hand-reviewed" check keys on this prefix.
    assert entry["notes"].startswith("force-fit")
