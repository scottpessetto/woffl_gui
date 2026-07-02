"""Tests for well_sort_client soft-fail / robustness behavior."""

import pandas as pd
import pytest

import woffl.assembly.well_sort_client as wsc


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
