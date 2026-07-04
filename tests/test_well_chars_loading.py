"""Tests for the well-characteristics loading contract (code review P1-12).

Contract under test (woffl/gui/utils.py):

- SUCCESS: `_load_well_characteristics_cached` returns the Databricks frame —
  the only thing st.cache_data is ever allowed to cache (1 h TTL).
- Databricks failure (or empty result): the cached core RAISES
  `WellCharsUnavailableError`. st.cache_data never caches exceptions, so the
  cache stays unfilled and every subsequent call re-probes Databricks.
- CSV fallback: built by the UNCACHED wrapper `load_well_characteristics`,
  which renders the st.warning itself (outside the cached core) and marks
  provenance in session_state["well_chars_source"].
- Double failure (Databricks AND jp_chars.csv): the wrapper raises — never a
  cached empty frame.

No test here touches live Databricks: `fetch_well_props_enriched` is always
monkeypatched.
"""

import sys
from unittest.mock import MagicMock

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Prefer the REAL streamlit (a hard dependency of this app): installing a
# MagicMock into sys.modules here would break test_gui_smoke's page imports
# later in the same session (`streamlit.components` can't resolve from a
# non-package mock). Only mock when streamlit genuinely isn't installed.
# Note: if an earlier-collected module (e.g. tests/test_utils.py) already
# mocked streamlit, `import streamlit` returns that mock — the tests below
# work under either, because they monkeypatch utils.st per-test and clear the
# core's cache around each test.
# ---------------------------------------------------------------------------
try:
    import streamlit  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - CI without streamlit
    _st_mock = MagicMock()
    _st_mock.cache_data = lambda *args, **kwargs: (
        args[0] if args and callable(args[0]) else (lambda fn: fn)
    )
    sys.modules.setdefault("streamlit", _st_mock)

import woffl.assembly.databricks_client as dbc  # noqa: E402
import woffl.gui.utils as utils  # noqa: E402
from woffl.gui.utils import WellCharsUnavailableError  # noqa: E402


# ===================================================================
# Test doubles
# ===================================================================
class _FakeSt:
    """Minimal streamlit stand-in: real-dict session_state + element capture."""

    def __init__(self):
        self.session_state: dict = {}
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(str(msg))

    def error(self, msg, *args, **kwargs):
        self.errors.append(str(msg))


def _clear_core_cache():
    """Drop any process-wide cache entry (real st.cache_data in-suite runs)."""
    clear = getattr(utils._load_well_characteristics_cached, "clear", None)
    if callable(clear):
        try:
            clear()
        except Exception:
            pass


@pytest.fixture
def fake_st(monkeypatch):
    fake = _FakeSt()
    _clear_core_cache()
    monkeypatch.setattr(utils, "st", fake)
    yield fake
    # Don't leak fabricated frames into other test modules' cache hits.
    _clear_core_cache()


def _db_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Well": ["MPX-1", "MPX-2"],
            "out_dia": [4.5, 4.5],
            "JP_MD": [4700.0, 4800.0],
        }
    )


@pytest.fixture
def csv_ok(tmp_path, monkeypatch):
    """Point the fallback at a real (temp) jp_chars-style CSV."""
    csv_path = tmp_path / "jp_chars.csv"
    pd.DataFrame({"Well": ["MPC-99"], "out_dia": [3.5]}).to_csv(csv_path, index=False)
    monkeypatch.setattr(utils, "_jp_chars_csv_path", lambda: str(csv_path))
    return csv_path


@pytest.fixture
def csv_missing(tmp_path, monkeypatch):
    """Point the fallback at a path that does not exist."""
    monkeypatch.setattr(
        utils, "_jp_chars_csv_path", lambda: str(tmp_path / "nope" / "jp_chars.csv")
    )


def _patch_fetch(monkeypatch, fn):
    """Patch the Databricks fetch the cached core imports at call time."""
    monkeypatch.setattr(dbc, "fetch_well_props_enriched", fn)


# ===================================================================
# Success path
# ===================================================================
class TestSuccessPath:
    def test_returns_databricks_frame_and_missing_list(self, fake_st, monkeypatch):
        _patch_fetch(monkeypatch, lambda: (_db_frame(), ["MPX-2"]))

        df = utils.load_well_characteristics()

        assert list(df["Well"]) == ["MPX-1", "MPX-2"]
        assert fake_st.session_state["wells_missing_surveys"] == ["MPX-2"]
        assert fake_st.session_state["well_chars_source"] == "databricks"
        assert fake_st.warnings == []
        assert fake_st.errors == []

    def test_wrapper_exposes_clear(self):
        assert callable(getattr(utils.load_well_characteristics, "clear", None))


# ===================================================================
# The no-caching-of-failures mechanism: the cached core RAISES
# ===================================================================
class TestCachedCoreRaises:
    """st.cache_data never caches exceptions — raising IS the mechanism that
    keeps a Databricks blip from being pinned process-wide for the TTL."""

    def test_core_raises_typed_error_on_databricks_failure(self, fake_st, monkeypatch):
        def _boom():
            raise RuntimeError("connection refused")

        _patch_fetch(monkeypatch, _boom)

        with pytest.raises(WellCharsUnavailableError):
            utils._load_well_characteristics_cached()

    def test_core_raises_on_empty_frame(self, fake_st, monkeypatch):
        _patch_fetch(monkeypatch, lambda: (pd.DataFrame(), []))

        with pytest.raises(WellCharsUnavailableError):
            utils._load_well_characteristics_cached()

    def test_core_emits_no_streamlit_elements(self, fake_st, monkeypatch):
        """The core body can run in a warm background thread — it must never
        render st.warning/st.error (element placement there is undefined)."""

        def _boom():
            raise RuntimeError("connection refused")

        _patch_fetch(monkeypatch, _boom)

        with pytest.raises(WellCharsUnavailableError):
            utils._load_well_characteristics_cached()
        assert fake_st.warnings == []
        assert fake_st.errors == []


# ===================================================================
# CSV fallback: served by the wrapper, marked, warned, never cached
# ===================================================================
class TestCsvFallback:
    def test_fallback_frame_returned_and_marked(self, fake_st, monkeypatch, csv_ok):
        def _boom():
            raise RuntimeError("warehouse offline")

        _patch_fetch(monkeypatch, _boom)

        df = utils.load_well_characteristics()

        assert list(df["Well"]) == ["MPC-99"]
        assert fake_st.session_state["well_chars_source"] == "csv_fallback"
        # CSV has no survey enrichment — banner list must be reset, not stale.
        assert fake_st.session_state["wells_missing_surveys"] == []
        # Warning rendered by the WRAPPER (outside the cached core) and
        # carries the underlying Databricks error.
        assert len(fake_st.warnings) == 1
        assert "warehouse offline" in fake_st.warnings[0]
        assert "jp_chars.csv" in fake_st.warnings[0]

    def test_fallback_is_not_cached_databricks_reprobed_each_call(
        self, fake_st, monkeypatch, csv_ok
    ):
        calls = []

        def _boom():
            calls.append(1)
            raise RuntimeError("warehouse offline")

        _patch_fetch(monkeypatch, _boom)

        utils.load_well_characteristics()
        utils.load_well_characteristics()

        # A cached fallback would swallow the second probe; the contract is
        # one fresh Databricks attempt per call.
        assert len(calls) == 2

    def test_recovers_immediately_after_blip(self, fake_st, monkeypatch, csv_ok):
        """The P1-12 poisoning scenario: a blip at fill time must NOT pin the
        stale CSV for the TTL — the very next call gets Databricks data."""
        state = {"n": 0}

        def _flaky():
            state["n"] += 1
            if state["n"] == 1:
                raise RuntimeError("transient blip")
            return _db_frame(), []

        _patch_fetch(monkeypatch, _flaky)

        df1 = utils.load_well_characteristics()
        assert list(df1["Well"]) == ["MPC-99"]
        assert fake_st.session_state["well_chars_source"] == "csv_fallback"

        df2 = utils.load_well_characteristics()
        assert list(df2["Well"]) == ["MPX-1", "MPX-2"]
        assert fake_st.session_state["well_chars_source"] == "databricks"


# ===================================================================
# Double failure: raise, never a (cacheable) empty frame
# ===================================================================
class TestDoubleFailure:
    def test_raises_with_both_errors_in_message(
        self, fake_st, monkeypatch, csv_missing
    ):
        def _boom():
            raise RuntimeError("warehouse offline")

        _patch_fetch(monkeypatch, _boom)

        with pytest.raises(WellCharsUnavailableError) as exc_info:
            utils.load_well_characteristics()

        msg = str(exc_info.value)
        assert "warehouse offline" in msg
        assert "jp_chars.csv" in msg

    def test_get_available_wells_degrades_to_custom(
        self, fake_st, monkeypatch, csv_missing
    ):
        def _boom():
            raise RuntimeError("warehouse offline")

        _patch_fetch(monkeypatch, _boom)

        assert utils.get_available_wells() == ["Custom"]

    def test_get_well_data_returns_none(self, fake_st, monkeypatch, csv_missing):
        def _boom():
            raise RuntimeError("warehouse offline")

        _patch_fetch(monkeypatch, _boom)

        assert utils.get_well_data("MPB-28") is None
