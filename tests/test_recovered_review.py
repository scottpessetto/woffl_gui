"""September 11 review regressions. Storage and physics boundaries stay offline."""
import asyncio
import threading
from types import SimpleNamespace as S

import httpx
import pandas as pd
import pytest

from server import cache


@pytest.fixture(autouse=True)
def clean_caches():
    cache.clear_all_caches()
    yield
    cache.clear_all_caches()


@pytest.mark.parametrize("pf_at_test,flagged,basis", [
    (2600., True, "vs measured PF"),
    (None, False, "vs limit (no test-day PF)"),
])
def test_washout_scan_preserves_measured_pressure(monkeypatch, pf_at_test, flagged, basis):
    from server.services.tools import jp_washout as svc
    from woffl.assembly import pf_calibration

    row = {"Well": "MPB-1", "Pad": "B", "Pump": "12B", "Nozzle": "12", "Throat": "B",
           "WtDate": pd.Timestamp("2026-08-01"), "Oil": 100., "Water": 500., "Gas": 50.,
           "LiftWat": 2500., "WHP": 210., "BHP": 900., "PfAtTest": pf_at_test, "_chars": {}}
    monkeypatch.setattr(svc, "_build_scan_input", lambda *a: pd.DataFrame([row]))
    monkeypatch.setattr(svc.pool, "submit_all", lambda *a: None)
    monkeypatch.setattr(svc._common, "build_well_config", lambda *a, **kw: S(form_temp=100., jpump_direction="reverse"))
    monkeypatch.setattr(svc._common, "create_well_objects", lambda *a: (None,) * 5)
    monkeypatch.setattr(pf_calibration, "calibrate_pf_for_lift", lambda **kw: S(
        ppf_surf=3200., modeled_qnz=2500., lift_residual=0., converged=True,
        bounded=False, sonic=False, iterations=1))
    out = svc.scan()["rows"][0]
    assert out["PfAtTest"] == pf_at_test
    assert out["Flagged"] is flagged and out["FlagBasis"] == basis
    assert out["PpfRatio"] == (1.231 if pf_at_test else None)


@pytest.mark.parametrize("failed_case", [0, 1])
def test_header_impact_preserves_either_solver_failure(monkeypatch, failed_case):
    from server.services.tools import header_impact as svc

    good = dict(oil=100., psu=900., sonic=False, mach=.3, error="na")
    bad = dict(oil=float("nan"), psu=float("nan"), sonic=False,
               mach=float("nan"), error="ConvergenceError: fixture failure")
    results = [good.copy(), good.copy()]
    results[failed_case] = bad
    it = iter(results)
    monkeypatch.setattr(svc._common, "build_well_config", lambda *a, **kw: S(res_pres=2000.))
    monkeypatch.setattr(svc._common, "create_well_objects", lambda *a: (None,) * 5)
    monkeypatch.setattr(svc, "_solve_at_whp", lambda *a: next(it))
    monkeypatch.setattr(svc, "_empirical_columns", lambda *a: {})
    out = svc.solve_jp_row({"Well": "MPB-1", "WHP now (psi)": 210., "PF held (psi)": 3000.},
                           {"present": True}, {"nozzle_no": "12", "throat_ratio": "B"}, None, None, -50.)
    assert out["Verdict"].startswith("model failed")
    assert "ConvergenceError" in out["Error"]


def test_choke_plan_with_zero_pf_test_only_well(monkeypatch):
    from woffl.gui import pad_optimize as svc

    class Plant:
        coupling = "free_pressure"
        water_key = "lift_wat"
        max_header_psi = 3500.
        infeasible_sweep_msg = "none"
        def pressure_window(self, n=None): return 3000., 3400.
        def budget_at_pressure(self, p, n=None): return 3500.
        def flow_window(self, n=None): return 0., 10000.
        def suction_psi(self): return 200.
        def warm_start_psi(self, n=None): return 3400.
        def flags(self, q, n=None): return dict(in_range=True, recirc=False, over_capacity=False)
        def delivered_header(self, q, setpoint=None, n=None): return 3400., False
        def clamp_window(self, n=None): return 1000., 3500.

    configs = [S(well_name=w, form_wc=.5, res_pres=1700., qwf=1000., pwf=800.)
               for w in ["MODEL", "TESTONLY"]]
    monkeypatch.setattr(svc, "_model_at_forced_header", lambda configs, header, choices: {
        "MODEL": (100. * header / 3400., 4000. * header / 3400., 600., False), "TESTONLY": None})
    rows, meta = svc.run_choke_optimization(configs, Plant(), None, {"MODEL": ("12", "B")},
        {"MODEL": (90., 3800.), "TESTONLY": (50., 0.)}, n_levels=4)
    assert meta["total_pf_bpd"] <= 3500.
    held = next(row for row in rows if row["well"] == "TESTONLY")
    assert held["action"] != "shut"


def test_warm_windows_use_one_new_snapshot_and_correct_slices(monkeypatch):
    from server import config, warmup
    from server.services import tests as svc
    from woffl.assembly import well_test_client

    calls = []
    now = pd.Timestamp.now().normalize()
    def fetch(*args):
        calls.append(args)
        return pd.DataFrame({"WtDate": [now - pd.DateOffset(months=m) for m in (1, 8, 18)],
                             "generation": [len(calls)] * 3}), 0
    monkeypatch.setattr(well_test_client, "fetch_milne_well_tests", fetch)
    warm = dict(warmup.fleet_targets())["well_tests"]
    for generation in (1, 2):
        warm()
        assert len(calls) == generation
        for months, count in ((6, 1), (12, 2), (24, 3)):
            assert months in config.WARM_TEST_MONTHS
            frame = svc.fetch_all_well_tests(months)
            assert len(frame) == count
            assert frame["generation"].tolist() == [generation] * count


def test_failed_warm_window_fetch_preserves_previous_snapshot(monkeypatch):
    from server.services import tests as svc
    from woffl.assembly import well_test_client

    old = pd.DataFrame({"WtDate": [pd.Timestamp.now()], "generation": [1]})
    for months in (6, 12, 24):
        svc.fetch_all_well_tests.cache_prime(old.copy(), months)
    def fail(*a): raise RuntimeError("offline fixture failure")
    monkeypatch.setattr(well_test_client, "fetch_milne_well_tests", fail)
    with pytest.raises(RuntimeError): svc.warm_test_windows()
    for months in (6, 12, 24):
        assert svc.fetch_all_well_tests(months).iloc[0]["generation"] == 1


@pytest.mark.parametrize("day,hours", [("2026-03-08", 23.), ("2026-11-01", 25.), ("2026-08-01", 24.)])
def test_separator_coverage_uses_local_day_length(day, hours):
    from server.services.tools import sep_oil_loss as svc

    start = pd.Timestamp(day, tz="America/Anchorage")
    stamps = pd.date_range(start - pd.Timedelta(hours=1), start + pd.DateOffset(days=1, hours=1), freq="min")
    frame = pd.DataFrame({"t": stamps, "dt_h": 1/60., "wc": 96., "base": 97.,
                          "oil_upper": 240., "oil_lower": 120.})
    row = next(r for r in svc._daily_rows(frame, [], 1000.) if r["date"] == day)
    assert row["hours"] == row["covered_hours"] == hours
    assert row["partial"] is False
    assert row["bbl_upper"] == 240. * hours / 24.


@pytest.mark.parametrize("kind", ["gauge", "samples"])
def test_upload_parser_allows_event_loop_to_progress(monkeypatch, kind):
    from server.main import app
    from server.services.tools import oiw_samples
    from tests.test_oiw_samples import _workbook
    from tests.test_web_gauge import _gauge_xlsx
    from woffl.gui import memory_gauge

    entered, release = threading.Event(), threading.Event()
    if kind == "gauge":
        module, name, path, field = memory_gauge, "parse_xlsx", "/api/gauge/parse", "files"
        blob = _gauge_xlsx(pd.Timestamp("2026-06-01"), 1, 1000.)
    else:
        module, name, path, field = oiw_samples, "oiw_samples", "/api/tools/sep-oil-loss/samples", "file"
        blob = _workbook([["2026-05-03", "08:00", "P-5417C", 1000., "test"]])
    original = getattr(module, name)
    def parse(*a, **kw):
        entered.set()
        assert release.wait(3), "parser blocked the API event loop"
        return original(*a, **kw)
    monkeypatch.setattr(module, name, parse)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            task = asyncio.create_task(client.post(path, files={field: ("test.xlsx", blob)}))
            try:
                assert await asyncio.to_thread(entered.wait, 3)
                release.set()  # must run while parsing is still in progress
                assert (await task).status_code == 200
            finally:
                release.set()
                if not task.done(): await task
    asyncio.run(exercise())


@pytest.mark.parametrize("sql", [
    "INSERT OVERWRITE TABLE mpu.wells.prop_hist SELECT 1",
    "INSERT INTO mpu.wells.prop_hist REPLACE WHERE enthid=1 SELECT 1",
    "INSERT INTO t SELECT * FROM t",
    "INSERT INTO t VALUES (1);;",
    "INSERT INTO t VALUES ((SELECT 1))",
])
def test_write_validator_rejects_non_append_templates(sql):
    from woffl.assembly import databricks_client as dc
    with pytest.raises(dc.UnsafeWriteStatementError): dc._validate_single_insert(sql)


def test_write_validator_accepts_all_application_templates():
    from woffl.assembly import databricks_client as dc, prop_hist_client as history

    dc._validate_single_insert(history.PROP_HIST_INSERT_SQL)
    dc._validate_single_insert(history.ENG_COMMENT_INSERT_SQL)
    rows = [f"(:enthid_{i}, :prop_id_{i}, :prop_value_{i}, :entry_datetime_{i}, :entry_user_{i})"
            for i in range(32)]
    dc._validate_single_insert(history.PROP_HIST_INSERT_HEAD + ", ".join(rows))


def test_dotenv_gate_exclusion_ignores_case(monkeypatch):
    import dotenv
    from databricks import sql
    from woffl.assembly import databricks_client as dc

    # Entire environment and connector are fakes; real process gates stay off.
    env = {"bricks_host": "fixture", "bricks_token": "fixture"}
    monkeypatch.setattr(dc.os, "environ", env)
    monkeypatch.setattr(dc, "_is_deployed", lambda: False)
    monkeypatch.setattr(dotenv, "dotenv_values", lambda: {
        "allow_databricks_writes": "true", "Allow_Prop_Hist_Delete": "yes",
        "ALLOW_DATABRICKS_WRITES": "1", "bricks_http": "/fixture",
    })
    monkeypatch.setattr(sql, "connect", lambda **kw: object())
    dc._new_connection()
    assert env == {"bricks_host": "fixture", "bricks_token": "fixture", "bricks_http": "/fixture"}
    assert not dc._write_gate_enabled()
