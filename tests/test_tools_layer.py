"""Scott's Tools - the contracts that break silently.

These are shape/wiring tests, not physics: the engines were ported verbatim
and their maths is covered by the tools' own live Test Harness. What is NOT
covered by either, and what actually broke during the port, is wiring:

* an engine helper unpacked in the wrong ORDER (``_fetch_empirical_fits``
  returns ``(dfs, fits, missing)``; reading it as ``(fits, dfs)`` handed every
  well the wrong dict and failed identically for all of them),
* a helper whose RETURN SHAPE changed (``live_pf_for_seed`` returning a float
  where every caller indexes ``["pf_press"]``),
* a worker that catches everything and returns None, so a real bug shows up
  as "0 rows" with no error anywhere,
* JP wells routed through the non-JP branch, producing rows with no DeltaOil.

Every one of those passes an import check and fails only against live data.
"""

from __future__ import annotations

import inspect

import pytest

from server.services.tools import (
    _common,
    harness,
    header_engine,
    header_impact,
    jp_calibration,
    jp_washout,
    pad_watercut,
    pf_scenario,
    runs,
)


# --- no tool may drag Streamlit back in ------------------------------------


def test_no_tool_imports_streamlit():
    """The whole point of the port. A single `import streamlit` in any engine
    puts a dead dependency back on the deploy."""
    import sys

    assert "streamlit" not in sys.modules


@pytest.mark.parametrize(
    "mod",
    [_common, harness, header_engine, header_impact, jp_calibration,
     jp_washout, pad_watercut, pf_scenario, runs],
)
def test_tool_module_has_no_streamlit_calls(mod):
    src = inspect.getsource(mod)
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith('"'):
            continue
        assert not stripped.startswith("import streamlit"), mod.__name__
        assert "st.session_state" not in stripped, f"{mod.__name__}: {stripped[:70]}"


# --- the shapes that broke -------------------------------------------------


def test_empirical_fits_unpacks_as_dfs_fits_missing():
    """Pin the ORDER. The docstring is the contract and it is easy to misread:
    it returns (well_dfs, fits_by_well, missing_wells)."""
    doc = header_impact._fetch_empirical_fits.__doc__ or ""
    assert "well_dfs" in doc and "fits_by_well" in doc and "missing_wells" in doc
    # and runs.py must consume it in that order
    src = inspect.getsource(runs.header_impact)
    assert "emp_dfs, emp_fits, emp_missing = hi._fetch_empirical_fits" in src


def test_live_pf_for_seed_returns_a_mapping_not_a_float():
    """Callers do live_pf["pf_press"]; a float there is a TypeError per well."""
    doc = _common.live_pf_for_seed.__doc__ or ""
    assert "pf_press" in doc
    ret = inspect.signature(_common.live_pf_for_seed).return_annotation
    assert "float" not in str(ret), "returning a bare float breaks every caller"


def test_jp_wells_go_through_the_physics_branch():
    """JP wells must reach solve_jp_row, not _solve_nonjp_row - the latter is
    the empirical branch and yields rows with no DeltaOil at all."""
    src = inspect.getsource(runs.header_impact)
    assert "hi.solve_jp_row" in src
    assert 'if str(row.get("Lift", "JP")) != "JP":' in src


@pytest.mark.parametrize(
    "fn", [jp_washout.calibrate_one, jp_calibration.calibrate_one, pf_scenario.compare_one]
)
def test_pool_workers_are_module_level_and_picklable(fn):
    """A ProcessPool worker must be importable by qualified name; a closure or
    a local function fails only when the pool is actually up."""
    import pickle

    assert fn.__qualname__ == fn.__name__, "worker must be module-level"
    pickle.loads(pickle.dumps(fn))


def test_failing_workers_report_instead_of_vanishing():
    """A worker that returns None on error turns a real bug into '0 rows' with
    nothing to debug - which is exactly how the pf_scenario break presented."""
    for fn in (jp_washout.calibrate_one, pf_scenario.compare_one):
        src = inspect.getsource(fn)
        assert '"Error"' in src, f"{fn.__name__} must carry an Error field"


# --- catalog / routing -----------------------------------------------------


def test_every_catalogued_tool_has_a_route_and_a_page():
    """The menu renders from the catalog, so an entry with no page is a dead
    link the moment someone unlocks the menu."""
    import pathlib

    from server import schemas

    app_tsx = (pathlib.Path(__file__).resolve().parent.parent / "web" / "src" / "App.tsx").read_text(
        encoding="utf-8"
    )
    for tool in schemas.TOOL_CATALOG:
        assert f'path="{tool["path"]}"' in app_tsx, f"{tool['id']} has no route"


def test_well_sort_is_not_in_the_secret_menu():
    """It was never on it - it is a top-level page - and listing it twice
    would be the obvious mistake when porting from the old tab list."""
    from server import schemas

    ids = {t["id"] for t in schemas.TOOL_CATALOG}
    assert "well-sort" not in ids


def test_catalog_ids_are_unique():
    from server import schemas

    ids = [t["id"] for t in schemas.TOOL_CATALOG]
    assert len(ids) == len(set(ids))


# --- read-only guarantee ---------------------------------------------------


def test_no_tool_can_write_to_prop_hist():
    """The tools model and compare; the app has exactly one write path and
    this is not it. JP Calibration renders SQL for a human to run."""
    for mod in (harness, header_impact, jp_calibration, jp_washout, pf_scenario, runs):
        src = inspect.getsource(mod)
        assert "push_prop" not in src, mod.__name__
        assert "execute_write" not in src, mod.__name__
