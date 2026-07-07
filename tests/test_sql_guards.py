"""Tests for woffl.assembly.sql_guards: validate-then-interpolate helpers
used before every f-string/str.format() SQL build site in well_test_client,
pad_watercut_client, and well_sort_client (see docs/code_review_2026-07-01.md
finding P2-7)."""

import datetime as dt

import pandas as pd
import pytest

from woffl.assembly.sql_guards import (
    UnsafeSqlValueError,
    validate_iso_date,
    validate_well_name,
)

# ── validate_well_name ──────────────────────────────────────────────────────


class TestValidateWellNameAccepts:
    """Real formats seen in the field (mirrors the fixtures in
    test_well_test_client.py / test_well_sort_client.py / bhp_dict.csv)."""

    @pytest.mark.parametrize(
        "name",
        [
            "B-028",  # raw Databricks vw_well_header format
            "E-041",
            "L-001",
            "S-017",
            "C-041",
            "MPB-28",  # GUI/jp_chars normalized format
            "MPE-41",
            "MPB-03",
            "MPB-100",
            "MPH-08",
            "FAKE-99",  # test-fixture placeholder used in test_databricks_client.py
            "FAKE-1",
            "FAKE-2",
            "B",  # bare pad letter
            "S",
            "I",
            "M",
        ],
    )
    def test_accepts_real_formats(self, name):
        assert validate_well_name(name) == name


class TestValidateWellNameRejects:
    @pytest.mark.parametrize(
        "name",
        [
            "'; DROP TABLE mpu.wells.vw_well_test --",
            "B-028'; DROP TABLE --",
            "B-028 OR 1=1",
            'B-028" --',
            "B-028; SELECT * FROM x",
            "B-028/*",
            "",
            "b-028",  # DB names are always uppercase; lowercase is unexpected/suspect
            "B_028",
            "B 028",
            None,
            123,
        ],
    )
    def test_rejects_unsafe_or_malformed(self, name):
        with pytest.raises(UnsafeSqlValueError):
            validate_well_name(name)


# ── validate_iso_date ───────────────────────────────────────────────────────


class TestValidateIsoDateAccepts:
    def test_accepts_iso_string(self):
        assert validate_iso_date("2024-01-15") == "2024-01-15"

    def test_accepts_date_object(self):
        assert validate_iso_date(dt.date(2024, 1, 15)) == "2024-01-15"

    def test_accepts_datetime_object(self):
        assert validate_iso_date(dt.datetime(2024, 1, 15, 12, 30)) == "2024-01-15"

    def test_accepts_pandas_timestamp(self):
        # pandas.Timestamp is a datetime.datetime subclass.
        assert validate_iso_date(pd.Timestamp("2024-01-15")) == "2024-01-15"


class TestValidateIsoDateRejects:
    @pytest.mark.parametrize(
        "value",
        [
            "2024-01-15'; DROP TABLE --",
            "2024-13-40",  # not a real calendar date
            "2024/01/15",
            "not-a-date",
            "",
            None,
            12345,
        ],
    )
    def test_rejects_unsafe_or_malformed(self, value):
        with pytest.raises(UnsafeSqlValueError):
            validate_iso_date(value)
