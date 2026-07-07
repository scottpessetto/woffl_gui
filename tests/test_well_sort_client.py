"""Tests for well_sort_client soft-fail / robustness behavior."""

import pandas as pd
import pytest

import woffl.assembly.well_sort_client as wsc
import woffl.assembly.well_test_client as wtc


def test_fetch_xv_status_softfails_on_query_error(monkeypatch):
    """XV status must degrade to an empty frame, not crash the Well Sort page.

    XV status reads the `reporting` historian catalog, which the hosted
    Databricks App's service principal may lack access to (works locally with
    the engineer's creds). A query error must return an empty DataFrame so the
    _xv_lookup -> {} path keeps Well Sort rendering from the shut-in log alone.
    """

    def _boom(_query):
        raise RuntimeError("[INSUFFICIENT_PERMISSIONS] Catalog 'reporting' ...")

    monkeypatch.setattr(wsc, "execute_query", _boom)
    out = wsc.fetch_xv_status()
    assert isinstance(out, pd.DataFrame)
    assert out.empty
    # And the downstream flattener tolerates it.
    assert wsc._xv_lookup(out) == {}


def test_xv_lookup_handles_empty_and_none():
    assert wsc._xv_lookup(pd.DataFrame()) == {}
    assert wsc._xv_lookup(None) == {}


# ── _normalize_well_name dedup (P2-7 / R-10) ────────────────────────────────


def test_normalize_well_name_is_the_canonical_well_test_client_copy():
    """well_sort_client no longer keeps its own duplicate — it must be the
    exact same function object imported from well_test_client."""
    assert wsc._normalize_well_name is wtc._normalize_well_name


def test_normalize_well_name_still_handles_non_str_gracefully():
    """The well_sort_client copy used to guard non-str input (e.g. a stray
    NaN) by returning it unchanged instead of raising. That guard was folded
    into the canonical well_test_client copy so behavior didn't change when
    the duplicate was deleted."""
    assert pd.isna(wsc._normalize_well_name(float("nan")))
    assert wsc._normalize_well_name(None) is None


def test_normalize_well_name_real_formats():
    assert wsc._normalize_well_name("B-028") == "MPB-28"
    assert wsc._normalize_well_name("S-017") == "MPS-17"
