"""The 🔒 WC/GOR/ResP lock checkboxes (`jetpump_solver._render_ipr_pin_controls`).

Scott, 2026-08-04: *"I unchecked the WC box and saved it. On reload it loaded
with WC checked as locked."* B-028 carried a single `form_wc_lock = 1.0` row
from 2026-07-31 and NO unlock row — the toggle had been silently swallowed.

Cause: the handler compared the checkbox against a `_prop_locked_` SESSION FLAG
rather than against what prop_hist actually stored. When the flag drifted False
while the row still said locked, the box rendered checked but `want ==
locked_now`, so unchecking pushed nothing and looked like it worked.

The fix compares intent against the STORED state, with a pushed-marker that
only suppresses a repeat. These pin both halves: a toggle is never swallowed,
and a lagging read never re-pushes (which is what produced hayden's triple
form_wc_lock rows on 2026-07-31).
"""

import sys
from unittest.mock import MagicMock

import pandas as pd
import pytest

_st = MagicMock()
_st.cache_data = lambda *a, **k: (a[0] if a and callable(a[0]) else lambda f: f)
sys.modules.setdefault("streamlit", _st)

from woffl.gui.tabs import jetpump_solver as js  # noqa: E402

WELL = "MPB-28"
FLAG = f"_prop_locked_form_wc_{WELL}"
WIDGET = f"sw_lock_form_wc_{WELL}"
PUSHED = f"_prop_lock_pushed_form_wc_{WELL}"

TESTS = pd.DataFrame(
    {
        "WtDate": [pd.Timestamp("2026-07-25")],
        "BHP": [1211.0],
        "WtTotalFluid": [1649.0],
        "WtOilVol": [283.0],
        "wt_uid": [-3591520.0],
    }
)


class _Col:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Rerun(RuntimeError):
    pass


class FakeSt:
    """Models Streamlit's keyed-widget contract: once the key is in
    session_state, ``value=`` is ignored and the stored state wins."""

    def __init__(self, state):
        self.session_state = dict(state)
        self.toasts = []
        self.warnings = []

    def columns(self, n):
        return [_Col() for _ in range(n if isinstance(n, int) else len(n))]

    def checkbox(self, label, value=False, key=None, help=None, **k):
        if key is not None and key in self.session_state:
            return bool(self.session_state[key])
        v = bool(value)
        if key is not None:
            self.session_state[key] = v
        return v

    def caption(self, *a, **k):
        pass

    def markdown(self, *a, **k):
        pass

    def button(self, *a, **k):
        return False

    def text_input(self, *a, **k):
        return ""

    def toast(self, m, **k):
        self.toasts.append(m)

    def warning(self, m, **k):
        self.warnings.append(m)

    def rerun(self):
        raise _Rerun()


def _render(monkeypatch, state, db_locks):
    """Render the lock row once; return (pushes, session_state).

    The write gate is set per-test, never at import: sibling modules delete
    ``ALLOW_DATABRICKS_WRITES`` in their own fixtures, and the lock UI sits
    behind ``writes_enabled()`` — an import-time setenv made these pass alone
    and fail in the full suite.
    """
    monkeypatch.setenv("ALLOW_DATABRICKS_WRITES", "true")
    pushes = []

    def _set(well, field, locked, value=None):
        pushes.append((field, locked))
        return True, "ok"

    fs = FakeSt(state)
    monkeypatch.setattr(js, "st", fs)
    monkeypatch.setattr(
        "woffl.gui.ipr_anchor.load_saved_ipr",
        lambda w: {
            "values": {}, "friction": {}, "locks": db_locks, "lock_values": {},
            "wc_locked": db_locks.get("form_wc", False), "wc_value": None,
            "saved_at": None, "saved_by": None, "pin_at": None,
            "pin_value": None, "pin_user": None,
        },
    )
    monkeypatch.setattr("woffl.gui.ipr_anchor.set_prop_lock", _set)
    try:
        js._render_ipr_pin_controls(WELL, TESTS, None)
    except _Rerun:
        pass
    return pushes, fs.session_state


class TestUnlockIsNeverSwallowed:
    """Stored state says locked; the engineer unchecks. It MUST push."""

    @pytest.mark.parametrize(
        "state,label",
        [
            ({FLAG: True, WIDGET: False}, "flag agrees with the row"),
            ({WIDGET: False}, "flag never seeded"),
            ({FLAG: False, WIDGET: False}, "flag drifted False — the B-28 bug"),
            ({FLAG: False}, "flag drifted False and widget state was GC'd"),
        ],
    )
    def test_unchecking_pushes_the_unlock(self, monkeypatch, state, label):
        pushes, _ = _render(monkeypatch, state, {"form_wc": True})
        assert pushes == [("form_wc", False)], label


class TestLockStillWorks:
    @pytest.mark.parametrize("state", [{FLAG: True, WIDGET: True}, {WIDGET: True}])
    def test_checking_pushes_the_lock(self, monkeypatch, state):
        pushes, _ = _render(monkeypatch, state, {})
        assert pushes == [("form_wc", True)]


class TestNoSpuriousPushes:
    def test_steady_state_locked_pushes_nothing(self, monkeypatch):
        pushes, _ = _render(monkeypatch, {WIDGET: True}, {"form_wc": True})
        assert pushes == []

    def test_steady_state_unlocked_pushes_nothing(self, monkeypatch):
        pushes, _ = _render(monkeypatch, {WIDGET: False}, {})
        assert pushes == []

    def test_a_lagging_read_does_not_re_push(self, monkeypatch):
        """After an unlock the memoized read can still say locked for a rerun.
        Re-pushing every rerun is what gave hayden three form_wc_lock rows."""
        pushes, _ = _render(
            monkeypatch,
            {FLAG: False, WIDGET: False, PUSHED: False},
            {"form_wc": True},
        )
        assert pushes == []

    def test_marker_clears_once_the_row_catches_up(self, monkeypatch):
        """So a LATER toggle back the other way is free to push again."""
        _, state = _render(monkeypatch, {WIDGET: False, PUSHED: False}, {})
        assert PUSHED not in state
        pushes, _ = _render(monkeypatch, {WIDGET: True}, {})
        assert pushes == [("form_wc", True)]


class TestOtherLocks:
    @pytest.mark.parametrize("field", ["form_gor", "res_pres"])
    def test_every_lockable_field_uses_the_same_path(self, monkeypatch, field):
        pushes, _ = _render(
            monkeypatch,
            {f"sw_lock_{field}_{WELL}": False},
            {field: True},
        )
        assert pushes == [(field, False)]
