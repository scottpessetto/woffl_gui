"""Shared pytest configuration for the tests/ package.

Registers the `live` marker (code review P2-4): a handful of well-data tests
in test_utils.py hit the REAL data source (live Databricks, or the
jp_chars.csv fallback) as an opt-in end-to-end sanity check, rather than
mocking the fetch layer. Those tests use range/shape asserts only -- never a
pinned live value -- but we still don't want them running (and potentially
skipping/failing on network conditions) by default.

Run them explicitly with:

    pytest --run-live

No pyproject.toml changes needed: the marker is registered here via
pytest_configure, and pytest_collection_modifyitems skips `live`-marked
tests unless --run-live is passed.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.live (hit real Databricks / the jp_chars.csv fallback).",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: hits the real Databricks well-properties source (or its CSV "
        "fallback) -- opt-in only, run with `pytest --run-live`.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-live"):
        return
    skip_live = pytest.mark.skip(reason="needs --run-live to hit the real data source")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
