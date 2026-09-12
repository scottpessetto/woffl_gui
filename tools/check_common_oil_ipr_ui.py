"""Explicit common-IPR fit/Apply/Save fixture QA, with no database access."""

import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


async def check(url):
    from playwright.async_api import async_playwright, expect
    from server import schemas
    from server.services import common_ipr, solve
    from tests.test_common_oil_ipr import fixture
    from woffl.assembly import databricks_client

    def forbidden(*args, **kwargs):
        raise AssertionError("Browser fixtures cannot access Databricks")
    databricks_client.execute_query = forbidden
    databricks_client.execute_write = forbidden
    req, tracker, frame = fixture()
    params = req.params.model_dump()
    params.update(nozzle_no="13", area_ratio="C", oil_api=22., gas_sg=.65, wat_sg=1.02)
    context = dict(well="MPE-42", chars={}, chars_source="databricks", seeds=params,
        as_built_locks={}, prop_locks={}, pump=dict(nozzle_no="13", throat_ratio="C", date_set="2026-02-01", source="databricks"),
        pump_calibration=dict(status="none", coefficients={}), pf=None, clamped=[],
        ipr_source="saved", ipr_info=None, ipr_r2=None, test_count=12, saved_ipr_info="Fixture saved IPR")
    history = dict(well="MPE-42", source="databricks", current_pump="13C", bhp_daily=[],
        installs=[dict(date_set=r["Date Set"], date_pulled=None, nozzle=r["Nozzle Number"], throat=r["Throat Ratio"],
                       tubing_od=4.5, circulating="reverse", manufacturer="National", raw_pump=None, pump_converted=False)
                  for r in tracker.to_dict("records")],
        tests=[dict(date=r["WtDate"], oil_rate=r["WtOilVol"], fwat_rate=r["WtTotalFluid"]-r["WtOilVol"],
                    bhp=r["BHP"], lift_wat=3000, pf_press=3168) for r in frame.to_dict("records")])
    tests = [dict(date=r["WtDate"], wt_uid=int(r["wt_uid"]), oil=r["WtOilVol"], bhp=r["BHP"],
                  total_fluid=r["WtTotalFluid"], form_wc=r["form_wc"], fgor=r["fgor"],
                  lift_wat=3000, whp=210, pf_press=3168, pf_source="fixture") for r in frame.to_dict("records")]
    fits, saves, errors, auto_fits = [], [], [], []
    pinned = True
    latest = None
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe")
        page = await browser.new_page(viewport=dict(width=1480, height=1100))
        page.on("pageerror", lambda e: errors.append(str(e)))

        async def api(route):
            nonlocal latest, pinned
            path, method = urlparse(route.request.url).path, route.request.method
            body = route.request.post_data_json if method == "POST" else None
            if path == "/api/meta": result = dict(app="WOFFL", version="fixture", physics_model=schemas.MODEL_VERSION, user=None, writes_enabled=True, deployed=False, warehouse_id="", physics_notice="")
            elif path == "/api/wells": result = dict(wells=[dict(name="MPE-42", pad="E", has_survey=False)], source="fixture")
            elif path.endswith("/context"): result = deepcopy(context)
            elif path.endswith("/jp-history"): result = history
            elif path.endswith("/tests"): result = dict(well="MPE-42", tests=tests)
            elif path.endswith("/ipr-pin"): result = dict(status="applied" if pinned else "none", wt_uid=100 if pinned else None, date_token="2026-01-10" if pinned else None, entry_user=None, entry_datetime=None)
            elif path == "/api/solve": result = solve.solve_single("Custom", schemas.SimParams(**body["params"]))
            elif path == "/api/common-ipr-fit":
                fits.append(body)
                request = schemas.CommonOilIprRequest(**body)
                latest = common_ipr.fit_frame(request, tracker, frame, "2026-04-15T00:00:00Z", "databricks").model_dump()
                result = latest
            elif path.endswith("/save-ipr"):
                saves.append(body)
                assert body["unpin"] is True and body["pin_wt_uid"] is None and body["pin_date"] is None, body
                pinned = False
                context["seeds"]["qwf"] = body["qwf_liq"]
                context["ipr_source"] = "manual"
                result = dict(pinned=False, pin_skipped=True, pin_message="Fixture anchor cleared.", n_values=6, values_message="Saved well inputs.")
            else:
                if path == "/api/ipr-fit": auto_fits.append(body)
                await route.fulfill(status=422, json=dict(detail=dict(error="invalid", message="Unavailable in this fixture")))
                return
            await route.fulfill(json=result)

        await page.route(re.compile(r"https?://[^/]+/api/"), api)
        await page.goto(url + "/jp-history?well=MPE-42")
        await expect(page.get_by_role("button", name="Fit one oil IPR across pump history", exact=True)).to_be_visible()
        assert not fits and not saves
        await page.evaluate("""() => { window.qaStore = async () => {
            const url = performance.getEntriesByType('resource').map(r => r.name).filter(n => n.includes('/src/state/params.ts')).at(-1);
            return (await import(url)).useParamsStore;
        }; }""")
        before = await page.evaluate("async () => (await qaStore()).getState().params")
        await page.get_by_role("button", name="Fit one oil IPR across pump history", exact=True).click()
        fit_button = page.get_by_role("button", name="Fit candidate oil IPR", exact=True)
        apply = page.get_by_role("button", name="Apply candidate IPR", exact=True)
        await fit_button.click()
        await expect(apply).to_be_enabled()
        assert not saves
        assert await page.evaluate("async () => (await qaStore()).getState().params") == before
        await page.evaluate("async () => { const s=(await qaStore()).getState(); s.set('pres', s.params.pres+50); }")
        await expect(apply).to_be_disabled()
        await page.evaluate("async () => { const s=(await qaStore()).getState(); s.set('pres', s.params.pres-50); }")
        await expect(apply).to_be_enabled()
        await page.get_by_text("Review training selections and excluded tests", exact=True).click()
        await page.get_by_label("Include training test 100", exact=True).uncheck()
        await expect(apply).to_be_disabled()
        await fit_button.click()
        await expect(apply).to_be_enabled()
        assert fits[-1]["exclude_tests"] == ["100"]
        candidate_panel = page.get_by_role("button", name="Hide one oil IPR across pump history", exact=True).locator("..")
        await candidate_panel.screenshot(path=str(ROOT / "build/common-oil-ipr-candidate.png"))
        await page.set_viewport_size(dict(width=960, height=1100))
        await candidate_panel.screenshot(path=str(ROOT / "build/common-oil-ipr-narrow.png"))
        assert await candidate_panel.evaluate("el => el.scrollWidth <= el.clientWidth")
        await page.set_viewport_size(dict(width=1480, height=1100))
        await apply.click()
        await expect(page.get_by_text("Candidate applied to the session.", exact=False)).to_be_visible()
        after = await page.evaluate("async () => (await qaStore()).getState().params")
        assert after == {**before, **latest["seeds"]}
        assert not saves
        await expect(page.get_by_label("History well inputs")).to_have_value("edited")
        await expect(page.get_by_text("Saving this common oil curve clears", exact=False)).to_be_visible()
        await page.get_by_role("button", name="Save well inputs", exact=True).click()
        await expect(page.get_by_text("New optimization runs will load", exact=False)).to_be_visible()
        assert saves[-1]["qwf_liq"] == latest["seeds"]["qwf"]
        # Navigation must retain manual/common-curve intent and block auto-fit reseeding.
        await page.get_by_role("link", name="Single Well", exact=True).click()
        await expect(page).to_have_url(re.compile(r"/solver$"))
        await expect(page.get_by_role("button", name="Oil only", exact=True)).to_have_attribute("aria-pressed", "true")
        await page.get_by_role("button", name="Total liquid", exact=True).click()
        await expect(page.get_by_role("button", name="Total liquid", exact=True)).to_have_attribute("aria-pressed", "true")
        await page.get_by_role("button", name="Oil only", exact=True).click()
        await expect(page.get_by_text("Saving this common oil curve clears", exact=False)).to_be_visible()
        await asyncio.sleep(.7)
        assert not auto_fits, auto_fits
        current = await page.evaluate("async () => (await qaStore()).getState()")
        assert current["commonIprIntent"] is True and current["params"]["qwf"] == after["qwf"]
        await page.get_by_role("button", name="Save well inputs", exact=True).click()
        await expect(page.get_by_text("New optimization runs will load", exact=False)).to_be_visible()
        assert len(saves) == 2
        await page.screenshot(path=str(ROOT / "build/common-oil-ipr-save.png"))
        assert not errors, errors
        print(json.dumps(dict(passed=True, fits=len(fits), intercepted_saves=len(saves), browser_errors=errors,
            checks="explicit fit/Apply/Save, fixed Pr, stale candidate, training exclusions, shared manual curve/unpin intent")))
        await browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5177")
    asyncio.run(check(parser.parse_args().url))
