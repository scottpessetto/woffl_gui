"""Fixture-only browser QA for history replay. Vite must listen on port 5176.

PYTHONPATH=build/browser-qa;. supplies the optional local Playwright install.
All /api requests are intercepted; no warehouse reads or production writes.
"""
import asyncio
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def fixture(request):
    from server import schemas
    from server.services import pump_match as pm
    from tests.test_pump_match import inputs
    cfg, tracker, frame = inputs()
    req = schemas.PumpMatchRequest(**request)
    cfg.hydraulics_model = req.hydraulics_model
    cfg.qwf, cfg.pwf, cfg.form_wc, cfg.form_gor = 200., 500., .5, 250.
    if req.edited_inputs:
        for key, value in req.edited_inputs.model_dump(exclude_none=True).items():
            setattr(cfg, {"qwf_liq": "qwf", "res_pres": "res_pres"}.get(key, key), value)
    frame.loc[frame["wt_uid"] == "2-20", ["form_wc", "fgor"]] = [.75, 700.]
    eras, rows, work, notes = pm.assemble(cfg, tracker, frame, req, "2026-03-01T00:00:00Z")
    for _cfg, controls, index in work:
        r = rows[index]
        day = int(controls["date"][8:10])
        if r["wt_uid"] == "2-20":
            r.update(status="failed", message="Fixture solver could not lift at this test.")
        else:
            r.update(status=r["phase"], predicted_bhp=480.+day, predicted_oil=85.+day,
                     predicted_pf=1900.+day*4, predicted_liquid=170.+day*2, sonic=False)
    for era in eras:
        own = [r for r in rows if r["installation_id"] == era["installation_id"]]
        era["fit_scores"] = pm.scores([r for r in own if r["phase"] == "fit"])
        era["prediction_scores"] = pm.scores([r for r in own if r["phase"] == "prediction"])
        era["replay_scores"] = pm.scores([r for r in own if r["phase"] == "replay"])
    return schemas.PumpMatchResult(well="MPE-42", request=req, physics_model="fixture",
        snapshot_id="fixture-history-replay", as_of="2026-03-01T00:00:00Z", source="databricks",
        notes=["Synthetic browser fixture, not field validation.", *notes], eras=eras, rows=rows).model_dump()


async def check():
    from playwright.async_api import async_playwright, expect
    from tests.test_pump_match import inputs
    _, tracker, frame = inputs()
    context = dict(well="MPE-42", chars={}, chars_source="databricks",
        seeds=dict(nozzle_no="13", area_ratio="C", form_wc=.5, qwf=200., pwf=500., pres=1500., hydraulics_model="beggs"),
        as_built_locks={}, prop_locks={}, pump=dict(nozzle_no="13", throat_ratio="C", date_set="2026-02-01T18:00:00Z", source="databricks"),
        pump_calibration=dict(status="none", coefficients={}, quality=None), pf=None, clamped=[], jpump_md=None,
        ipr_info=None, ipr_source="manual", ipr_r2=None, test_count=12, saved_ipr_info="Manual IPR")
    history = dict(well="MPE-42", source="databricks", current_pump="13C", installs=[
        dict(date_set=r["Date Set"], date_pulled=None, nozzle=r["Nozzle Number"], throat="C", tubing_od=4.5,
             circulating="reverse", manufacturer="National", raw_pump=None, pump_converted=False) for r in tracker.to_dict("records")],
        tests=[dict(date=r["WtDate"], oil_rate=r["WtOilVol"], fwat_rate=100., lift_wat=r["lift_wat"], bhp=r["BHP"], pf_press=r["pf_press"])
               for r in frame.to_dict("records")],
        bhp_daily=[dict(date=r["WtDate"], bhp=r["BHP"]) for r in frame.to_dict("records")])
    errors, starts, cancels = [], [], []
    job_values = {}
    hold = False
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe")
        page = await browser.new_page(viewport=dict(width=1440, height=1050), timezone_id="America/Anchorage")
        page.on("pageerror", lambda e: errors.append(str(e)))

        async def api(route):
            path = urlparse(route.request.url).path
            method = route.request.method
            if path == "/api/meta":
                data = dict(app="WOFFL", version="fixture", physics_model="fixture", user=None, writes_enabled=False, warehouse_id="", deployed=False)
            elif path == "/api/wells":
                data = dict(wells=[dict(name="MPE-42", pad="E", has_survey=True)], source="databricks")
            elif path.endswith("/context"): data = context
            elif path.endswith("/jp-history"): data = history
            elif path.endswith("/tests"): data = dict(well="MPE-42", tests=[])
            elif path.endswith("/ipr-pin"): data = dict(status="none")
            elif path.endswith("/pump-match") and method == "POST":
                request = route.request.post_data_json
                starts.append(request)
                jid = f"fixture-{len(starts)}"
                job_values[jid] = dict(job_id=jid, kind="pump-match", status="running" if hold else "done",
                    progress="Fixture replay running" if hold else "done", result=None if hold else fixture(request),
                    error=None, started_at="2026-03-01T00:00:00Z", seconds=1.)
                data = dict(job_id=jid)
            elif path.startswith("/api/pump-match/"):
                jid = path.rsplit("/", 1)[-1]
                if method == "DELETE":
                    cancels.append(jid)
                    if jid in job_values: job_values[jid].update(status="cancelled", result=None)
                    data = dict(cancel_requested=True)
                else: data = job_values[jid]
            else:
                await route.fulfill(status=404, json=dict(detail="No browser fixture"))
                return
            await route.fulfill(json=data)

        await page.route(re.compile(r"https?://[^/]+/api/"), api)
        await page.goto("http://127.0.0.1:5176/jp-history?well=MPE-42&match=1")
        await expect(page.get_by_label("Show model match", exact=True)).to_be_checked()
        await expect(page.get_by_label("History comparison", exact=True)).to_have_value("all_tests")
        await expect(page.get_by_label("Training tests", exact=True)).to_have_count(0)
        await expect(page.get_by_role("button", name="Run comparison", exact=True)).to_be_enabled()
        await page.get_by_role("button", name="Run comparison", exact=True).click()
        await expect(page.get_by_text("Historical model results by installation", exact=False)).to_be_visible()
        await expect(page.get_by_text("9/12 tests have predictions", exact=False)).to_be_visible()
        await page.get_by_label("PF rate detail", exact=True).check()
        chart = "window.__ECHARTS__.getInstanceByDom(document.querySelector('[_echarts_instance_]'))"
        await page.wait_for_function(f"{chart}?.getOption().xAxis.length === 4")
        state = await page.evaluate(f"""() => {{ const c = {chart}; const o = c.getOption(); return {{
            names:o.series.map(s=>s.name), svg:!!c.getDom().querySelector('svg'), axes:o.xAxis.length,
            dates:o.series.flatMap(s=>s.markLine?.data ?? []).map(p=>p.xAxis).filter(Boolean) }}; }}""")
        assert state["svg"] and "Oil model" in state["names"] and "BHP model" in state["names"]
        assert "PF rate model" in state["names"]
        assert "Oil fit" not in state["names"] and "Oil prediction" not in state["names"]
        tooltip = await page.evaluate(f"{chart}.getOption().tooltip[0].formatter([{{axisValue:Date.parse('2026-01-05')}}])")
        assert "BHP model" in tooltip and "Oil model" in tooltip and "485" in tooltip
        assert "Model WC / GOR" in tooltip and "50.0% / 250 scf/STB" in tooltip
        tooltip = await page.evaluate(f"{chart}.getOption().tooltip[0].formatter([{{axisValue:Date.parse('2026-02-20')}}])")
        assert "Solve failed" in tooltip and "Fixture solver could not lift" in tooltip
        assert "BHP prediction" not in tooltip and "- psi" not in tooltip
        tooltip = await page.evaluate(f"{chart}.getOption().tooltip[0].formatter([{{axisValue:Date.parse('2026-02-12')}}])")
        assert "No modeled test on this date" in tooltip
        await page.evaluate(f"{chart}.dispatchAction({{type:'dataZoom', dataZoomIndex:0, start:25, end:95}})")
        await page.get_by_role("button", name="13C / 2026-02-01", exact=True).click()
        await expect(page.get_by_text("Fixed oil IPR: 100 BOPD", exact=False)).to_be_visible()
        await page.get_by_text("Inspect a test or model miss", exact=True).click()
        await page.get_by_label("Inspect history test", exact=True).select_option("2-20")
        await expect(page.get_by_text("Fixture solver could not lift", exact=False)).to_be_visible()
        await expect(page.get_by_text("Model inputs: WC 75.0%; GOR 700 scf/STB.", exact=True)).to_be_visible()
        await page.set_viewport_size(dict(width=1440, height=1850))
        await page.get_by_label("Show model match", exact=True).scroll_into_view_if_needed()
        await page.screenshot(path=str(ROOT / "build/pump-match-desktop.png"), full_page=True)
        await page.set_viewport_size(dict(width=960, height=1850))
        await page.get_by_label("Show model match", exact=True).scroll_into_view_if_needed()
        await page.screenshot(path=str(ROOT / "build/pump-match-narrow.png"), full_page=True)

        # Changes hide old curves immediately, before another run completes.
        await page.get_by_label("History comparison", exact=True).select_option("same_pump")
        await expect(page.get_by_text("Historical model results by installation", exact=False)).to_have_count(0)
        await page.get_by_label("Training tests", exact=True).select_option("3")
        await page.get_by_role("button", name="Run comparison", exact=True).click()
        await expect(page.get_by_text("Held-out results by installation", exact=False)).to_be_visible()
        assert "Oil fit" in await page.evaluate(f"{chart}.getOption().series.map(s=>s.name)")
        assert "Oil model" not in await page.evaluate(f"{chart}.getOption().series.map(s=>s.name)")
        await page.get_by_role("button", name="13C / 2026-02-01", exact=True).click()
        await expect(page.get_by_text("Training:", exact=False)).to_be_visible()
        await page.get_by_label("History comparison", exact=True).select_option("previous_pump")
        await expect(page.get_by_text("Held-out results by installation", exact=False)).to_have_count(0)
        await page.get_by_label("Training tests", exact=True).select_option("3")
        await page.get_by_role("button", name="Run comparison", exact=True).click()
        await expect(page.get_by_text("Held-out results by installation", exact=False)).to_be_visible()
        assert "Oil prediction" in await page.evaluate(f"{chart}.getOption().series.map(s=>s.name)")
        tooltip = await page.evaluate(f"{chart}.getOption().tooltip[0].formatter([{{axisValue:Date.parse('2026-01-05')}}])")
        assert "No prediction" in tooltip and "No usable immediately preceding installation" in tooltip
        assert "BHP prediction" not in tooltip
        why = page.get_by_text(re.compile(r"Why \d+ tests have no prediction"))
        if not await why.evaluate("el => el.parentElement.open"):
            await why.click()
        await expect(page.get_by_text("To compare the well fit at every usable test", exact=False)).to_be_visible()
        await page.get_by_label("Show model match", exact=True).uncheck()
        assert "Oil prediction" not in await page.evaluate(f"{chart}.getOption().series.map(s=>s.name)")
        await page.get_by_label("Show model match", exact=True).check()
        hold = True
        await page.get_by_role("button", name="Run again", exact=True).click()
        await expect(page.get_by_role("button", name="Cancel", exact=True)).to_be_visible()
        await page.get_by_role("button", name="Cancel", exact=True).click()
        await expect(page.get_by_text("Comparison cancelled.", exact=True)).to_be_visible()
        assert cancels
        assert not errors, errors
        await browser.close()
    print(json.dumps(dict(status="passed", starts=len(starts), cancellations=len(cancels), browser_errors=errors, chart=state)))


if __name__ == "__main__":
    asyncio.run(check())
