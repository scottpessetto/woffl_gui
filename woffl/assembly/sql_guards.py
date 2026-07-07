"""SQL Guards — validate-then-interpolate for Databricks f-string SQL.

woffl.assembly.well_test_client / pad_watercut_client / well_sort_client build
their queries with plain f-string / str.format() interpolation of well
names, pad letters, and date strings (see docs/code_review_2026-07-01.md
finding P2-7). All of that data is INTERNAL — it originates from Databricks
views or GUI date pickers, never from an end user typing into a text box —
so full parameter binding is overkill. The chosen fix is cheap and
sufficient: validate the shape of every interpolated string immediately
before it's spliced into a query, and raise a typed error instead of ever
letting a stray quote/semicolon/comment reach the SQL text.

Numeric interpolations (e.g. a lookback-days window) are handled separately
at each call site by coercing with int()/float() — that alone makes them
safe and doesn't need a regex gate here.

IMPORTANT: names in these systems are internal Databricks data, not
untrusted end-user input. A false rejection of a real well name is a worse
outcome than the (already low) injection risk, so the patterns below are
deliberately generous — see the real-format inventory in each regex's
comment, drawn from tests/test_well_test_client.py,
tests/test_well_sort_client.py, tests/test_databricks_client.py, and
woffl/jp_data/bhp_dict.csv.
"""

from __future__ import annotations

import datetime as _dt
import re

# Real well-name / pad-letter shapes seen throughout the app:
#   Raw Databricks vw_well_header:  B-028, E-041, L-001, S-017, C-041 (3-digit
#     zero-padded number)
#   GUI/jp_chars normalized:        MPB-28, MPE-41, MPB-03, MPB-100, MPH-08
#   Bare pad letter (pad filters):  B, C, E, F, G, H, I, J, K, L, M, S
#   Test/fixture placeholders:      FAKE-99, FAKE-1, FAKE-2
# Prefix is 1-4 uppercase letters (covers a bare pad letter and the "MP" +
# 1-2 letter pad code); number, when present, is 1-5 digits. Generous on
# purpose — a false rejection in production is worse than the injection risk.
_WELL_NAME_RE = re.compile(r"^[A-Z]{1,4}(-\d{1,5})?$")

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class UnsafeSqlValueError(ValueError):
    """A value destined for f-string/str.format() SQL interpolation failed
    validation (wrong shape, or contains characters that could break out of
    a SQL string literal). Raised instead of silently sanitizing so a bad
    value never reaches a query."""


def validate_well_name(name: str) -> str:
    """Validate a well name / pad letter before splicing it into SQL.

    Accepts the real formats used throughout the app: bare pad letters
    ('S', 'I'), raw Databricks names ('B-028'), and GUI-normalized names
    ('MPB-28', 'MPB-100'). Rejects quotes, semicolons, whitespace, SQL
    comment markers, or any other character outside `[A-Z0-9-]`.

    Returns the name unchanged (so call sites can inline it, e.g.
    ``f"'{validate_well_name(w)}'"``).
    """
    if not isinstance(name, str) or not _WELL_NAME_RE.match(name):
        raise UnsafeSqlValueError(
            f"Unsafe/invalid well name for SQL interpolation: {name!r}"
        )
    return name


def validate_iso_date(value) -> str:
    """Validate (or format) a date destined for SQL interpolation.

    Accepts 'YYYY-MM-DD' strings and validates them are real calendar dates
    (rejects shapes like '2024-13-40'). Also accepts `datetime.date` /
    `datetime.datetime` (and therefore `pandas.Timestamp`, a `datetime`
    subclass) objects and formats them, so callers already holding a real
    date object don't need to pre-format it themselves.

    Returns the 'YYYY-MM-DD' string form.
    """
    if isinstance(value, _dt.datetime):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, _dt.date):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, str) and _ISO_DATE_RE.match(value):
        try:
            _dt.datetime.strptime(value, "%Y-%m-%d")
        except ValueError as exc:
            raise UnsafeSqlValueError(
                f"Invalid calendar date for SQL interpolation: {value!r}"
            ) from exc
        return value
    raise UnsafeSqlValueError(f"Unsafe/invalid date for SQL interpolation: {value!r}")
