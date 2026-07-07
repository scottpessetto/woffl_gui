"""Tests for pad_watercut_client's SQL-guard usage (P2-7).

pad_watercut_client's queries interpolate dates (well/pad identifiers there
are hardcoded module constants, not external input — see PADS/_PAD_LIKE), so
the coverage here focuses on the date guard firing before any query runs.
"""

from unittest.mock import patch

import pytest

from woffl.assembly.pad_watercut_client import (
    _fetch_shut_in,
    _fetch_tests,
    fetch_pad_watercut,
)
from woffl.assembly.sql_guards import UnsafeSqlValueError


class TestFetchTestsDateGuard:
    # NOTE: `_fetch_tests`'s start_date is fed through `pd.Timestamp(...)`
    # before it ever reaches `validate_iso_date` (it derives `tests_start`,
    # the lookback-adjusted date). A malformed/malicious *start_date* is
    # already rejected by pandas' own parser at that point (DateParseError),
    # so these guard tests target `end_date`, which goes straight into
    # `validate_iso_date` unmodified.
    @patch("woffl.assembly.pad_watercut_client.execute_query")
    def test_malicious_end_date_raises_before_query(self, mock_query):
        with pytest.raises(UnsafeSqlValueError):
            _fetch_tests("2024-01-01", "2024-12-31'; DROP TABLE --")
        mock_query.assert_not_called()

    @patch("woffl.assembly.pad_watercut_client.execute_query")
    def test_malformed_end_date_raises_before_query(self, mock_query):
        """Syntactically-shaped but non-existent calendar date (regex passes,
        strptime doesn't)."""
        with pytest.raises(UnsafeSqlValueError):
            _fetch_tests("2024-01-01", "2024-13-40")
        mock_query.assert_not_called()


class TestFetchShutInDateGuard:
    @patch("woffl.assembly.pad_watercut_client.execute_query")
    def test_malicious_start_date_raises_before_query(self, mock_query):
        with pytest.raises(UnsafeSqlValueError):
            _fetch_shut_in("2024-01-01'; DROP TABLE --", "2024-12-31")
        mock_query.assert_not_called()


class TestFetchPadWatercutEndToEnd:
    @patch("woffl.assembly.pad_watercut_client.execute_query")
    def test_malicious_date_raises_before_any_query(self, mock_query):
        """The public entry point must fail closed before either the tests
        or shut-in query executes. `pd.date_range` (upstream of the SQL
        guard) already rejects a date shaped like this with its own
        DateParseError; either way execute_query must never be reached."""
        with pytest.raises(Exception):
            fetch_pad_watercut("2024-01-01", '2024-12-31" --')
        mock_query.assert_not_called()

    @patch("woffl.assembly.pad_watercut_client.execute_query")
    def test_malicious_well_name_style_payload_in_date_raises_before_query(
        self, mock_query
    ):
        """A payload that *would* parse as a date-like string (so it reaches
        the SQL-guard layer rather than being rejected earlier by
        `pd.date_range`) must still be caught by validate_iso_date before
        execute_query runs."""
        with pytest.raises(UnsafeSqlValueError):
            _fetch_shut_in("2024-01-01", "2024-01-02 OR 1=1")
        mock_query.assert_not_called()
