"""Tests for woffl.gui.ipr_anchor's IPR-anchor pin push/clear helpers (W3 of
the woffl-prop-hist-persistence plan): ``pin_ipr_anchor``, ``clear_ipr_pin``,
``writes_enabled``.

These are the SHARED push helpers behind both the pad-review save hook
(``workflow_steps/step_review_wells.py::_maybe_pin_saved_ipr``) and the
Solver's "Save IPR as well default" / "Clear saved IPR" buttons
(``tabs/jetpump_solver.py::_render_ipr_pin_controls``). ``push_prop`` /
``resolve_entry_user`` are mocked at the ``woffl.gui.ipr_anchor`` import
boundary — zero live Databricks calls.
"""

import os

import pandas as pd
import pytest

from woffl.gui import ipr_anchor


@pytest.fixture(autouse=True)
def _clean_env():
    prior = os.environ.pop("ALLOW_DATABRICKS_WRITES", None)
    yield
    if prior is not None:
        os.environ["ALLOW_DATABRICKS_WRITES"] = prior
    else:
        os.environ.pop("ALLOW_DATABRICKS_WRITES", None)


class TestWritesEnabled:
    @pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "Yes"])
    def test_truthy_values(self, val):
        os.environ["ALLOW_DATABRICKS_WRITES"] = val
        assert ipr_anchor.writes_enabled() is True

    @pytest.mark.parametrize("val", ["0", "false", "no", "", "maybe"])
    def test_falsy_values(self, val):
        os.environ["ALLOW_DATABRICKS_WRITES"] = val
        assert ipr_anchor.writes_enabled() is False

    def test_unset_is_falsy(self):
        assert ipr_anchor.writes_enabled() is False


def _row(wt_uid, date="2026-05-10"):
    return pd.Series({"WtDate": pd.Timestamp(date), "wt_uid": wt_uid})


class TestPinIprAnchor:
    def test_no_anchor_row_skips(self):
        pushed, message = ipr_anchor.pin_ipr_anchor("MPB-28", None)
        assert pushed is False
        assert message.startswith(ipr_anchor.PIN_SKIP_PREFIX)
        assert "no resolvable test" in message

    def test_manual_test_nan_uid_skips(self):
        pushed, message = ipr_anchor.pin_ipr_anchor("MPB-28", _row(float("nan")))
        assert pushed is False
        assert message.startswith(ipr_anchor.PIN_SKIP_PREFIX)
        assert "manual" in message

    def test_missing_wt_uid_column_skips(self):
        # A Series with no 'wt_uid' key at all (older frame / synthetic data).
        row = pd.Series({"WtDate": pd.Timestamp("2026-05-10")})
        pushed, message = ipr_anchor.pin_ipr_anchor("MPB-28", row)
        assert pushed is False
        assert message.startswith(ipr_anchor.PIN_SKIP_PREFIX)

    def test_successful_push_returns_true_with_date(self, monkeypatch):
        calls = {}

        def _fake_push(well_name, prop_id, value, entry_user):
            calls["well_name"] = well_name
            calls["prop_id"] = prop_id
            calls["value"] = value
            calls["entry_user"] = entry_user
            return 1

        monkeypatch.setattr(ipr_anchor, "push_prop", _fake_push)
        monkeypatch.setattr(ipr_anchor, "resolve_entry_user", lambda: "Scott")

        pushed, message = ipr_anchor.pin_ipr_anchor("MPB-28", _row(302.0))

        assert pushed is True
        assert calls == {
            "well_name": "MPB-28",
            "prop_id": "ipr_wt_uid",
            "value": 302.0,
            "entry_user": "Scott",
        }
        assert "2026-05-10" in message
        assert "📌" in message

    def test_negative_wt_uid_pushes_verbatim_as_a_valid_pin(self, monkeypatch):
        # Real wt_uid values in vw_well_test are signed and span roughly
        # -3.6M to +3.1M -- almost all negative in practice (e.g. C-045's
        # real saved pin, wt_uid=-3576674). pin_ipr_anchor must push it
        # verbatim -- no sign-based special-casing anywhere in this path.
        calls = {}

        def _fake_push(well_name, prop_id, value, entry_user):
            calls["value"] = value
            return 1

        monkeypatch.setattr(ipr_anchor, "push_prop", _fake_push)
        monkeypatch.setattr(ipr_anchor, "resolve_entry_user", lambda: "Scott")

        pushed, message = ipr_anchor.pin_ipr_anchor("MPC-45", _row(-3576674.0))

        assert pushed is True
        assert calls["value"] == -3576674.0
        assert "📌" in message

    def test_push_prop_exception_surfaces_as_failure(self, monkeypatch):
        def _raise(*a, **k):
            raise RuntimeError("No enthid found for well 'MPB-28'.")

        monkeypatch.setattr(ipr_anchor, "push_prop", _raise)
        monkeypatch.setattr(ipr_anchor, "resolve_entry_user", lambda: "Scott")

        pushed, message = ipr_anchor.pin_ipr_anchor("MPB-28", _row(302.0))

        assert pushed is False
        assert message.startswith(ipr_anchor.PIN_FAILURE_PREFIX)
        assert "No enthid found" in message

    def test_resolve_entry_user_exception_surfaces_as_failure(self, monkeypatch):
        def _raise():
            raise RuntimeError("offline")

        monkeypatch.setattr(ipr_anchor, "resolve_entry_user", _raise)
        called = []
        monkeypatch.setattr(ipr_anchor, "push_prop", lambda *a, **k: called.append(1))

        pushed, message = ipr_anchor.pin_ipr_anchor("MPB-28", _row(302.0))

        assert pushed is False
        assert message.startswith(ipr_anchor.PIN_FAILURE_PREFIX)
        assert called == []  # never reached push_prop


class TestClearIprPin:
    def test_pushes_null_value(self, monkeypatch):
        # There is no numeric sentinel that's safe -- real wt_uid values are
        # signed and span both positive and negative ranges (observed
        # roughly -3.6M to +3.1M). clear_ipr_pin must push None so
        # push_prop binds a SQL NULL, never a negative-number sentinel.
        calls = {}

        def _fake_push(well_name, prop_id, value, entry_user):
            calls["value"] = value
            calls["prop_id"] = prop_id
            return 1

        monkeypatch.setattr(ipr_anchor, "push_prop", _fake_push)
        monkeypatch.setattr(ipr_anchor, "resolve_entry_user", lambda: "Scott")

        cleared, message = ipr_anchor.clear_ipr_pin("MPB-28")

        assert cleared is True
        assert calls == {"value": None, "prop_id": "ipr_wt_uid"}
        assert message == "Saved IPR cleared"

    def test_push_failure_surfaces(self, monkeypatch):
        def _raise(*a, **k):
            raise RuntimeError("connection reset")

        monkeypatch.setattr(ipr_anchor, "push_prop", _raise)
        monkeypatch.setattr(ipr_anchor, "resolve_entry_user", lambda: "Scott")

        cleared, message = ipr_anchor.clear_ipr_pin("MPB-28")

        assert cleared is False
        assert message.startswith(ipr_anchor.UNPIN_FAILURE_PREFIX)
        assert "connection reset" in message
