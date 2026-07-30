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


# ── XV tag derivation: 3-digit well numbers must not be truncated ───────────
# Spark's LPAD(str, 2, '0') TRUNCATES anything longer than 2 characters, so the
# original `LPAD(CAST(well_number AS STRING), 2, '0')` collapsed every 3-digit
# well onto its first two characters. Verified live 2026-07-29: 13 producers
# derived only 4 distinct tags, none of which exist in the historian, so their
# ProdXV/PFXV were permanently blank. Fixing it took coverage 182 -> 191 of 245.
#
# Ground truth confirmed against reporting.historian.vw_mpu_measurements — each
# of these returns live data, while its truncated form returns nothing at all:
XV_TAGS_VERIFIED_LIVE = {
    "R-106": "MPU_XZ_462106",   # truncated to MPU_XZ_46210 (shared with 4 others)
    "R-109": "MPU_XZ_462109",
    "R-110": "MPU_XZ_462110",   # truncated to MPU_XZ_46211 (shared with R-111)
    "R-111": "MPU_XZ_462111",
    "R-144": "MPU_XZ_462144",   # truncated to MPU_XZ_46214 (shared with R-142)
    "F-107": "MPU_XZ_242107",   # truncated to MPU_XZ_24210 (shared with F-109)
    "F-109": "MPU_XZ_242109",
}


def test_xv_query_never_truncates_the_well_number():
    """Tripwire: reverting to LPAD-on-well_number silently re-blanks R/F pad.

    The failure is invisible in the UI — the wells just show None forever —
    so this asserts on the query text directly. LPAD on `pad_number` is fine
    (pad codes are always 2 digits); LPAD on `well_number` is the bug.
    """
    sql = wsc.XV_STATUS_QUERY
    assert "LPAD(CAST(well_number" not in sql, (
        "LPAD truncates well numbers longer than 2 chars — R-pad wells are "
        "3-digit (101..145) and would collapse onto duplicate, nonexistent tags."
    )
    # The non-truncating guard for a hypothetical unpadded single digit.
    assert "LENGTH(CAST(well_number AS STRING)) < 2" in sql
    assert "CONCAT('0', CAST(well_number AS STRING))" in sql


def test_xv_query_builds_full_length_tags():
    """The tag is pad + series digit + the FULL well number, in that order."""
    sql = wsc.XV_STATUS_QUERY
    assert "CONCAT('MPU_XZ_', pad_str, '2', well_str) AS prod_tag" in sql
    assert "CONCAT('MPU_XZ_', pad_str, '4', well_str) AS pf_tag" in sql
    # pad_number padding is retained and is NOT the bug.
    assert "LPAD(CAST(pad_number AS STRING), 2, '0')" in sql


@pytest.mark.parametrize("well,tag", sorted(XV_TAGS_VERIFIED_LIVE.items()))
def test_xv_tag_convention_matches_live_historian(well, tag):
    """Documents the convention the SQL implements, against tags confirmed to
    return data in the historian. pad/well come from vw_bhp_tags (pad 46 = R,
    24 = F; well_number is stored zero-padded, e.g. '106', '107')."""
    pad = {"R": "46", "F": "24"}[well[0]]
    wellnum = well.split("-")[1]
    assert f"MPU_XZ_{pad}2{wellnum}" == tag
    # …and the old truncating rule provably did NOT produce it.
    assert f"MPU_XZ_{pad}2{wellnum[:2]}" != tag
