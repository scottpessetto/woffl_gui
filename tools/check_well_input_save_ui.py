"""Fixture-only preview/edit/save QA. Vite on 5176; every API call intercepted."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


async def check():
    from playwright.async_api import async_playwright, expect
    from tests.test_pump_match import inputs
    from tools.check_pump_match_ui import fixture

    _, tracker, frame = inputs()
    context = dict(well="MPE-42", chars={}, chars_source="databricks",
        seeds=dict(nozzle_no="13", area_ratio="C", form_wc=.5, form_gor=250., qwf=200., pwf=500.,
                   pres=1500., form_temp=80., bubble_point=1750., hydraulics_model="beggs"),
        as_built_locks={}, prop_locks={}, pump=dict(nozzle_no="13", throat_ratio="C",
            date_set="2026-02-01T18:00:00Z", source="databricks"),
        pump_calibration=dict(status="none", coefficients={}, quality=None),
        pf=None, clamped=[], jpump_md=None, ipr_info=None, ipr_source="manual", ipr_r2=None,
        test_count=12, saved_ipr_info="Saved manual IPR")
    history = dict(well="MPE-42", source="databricks", current_pump="13C", installs=[
        dict(date_set=r["Date Set"], date_pulled=None, nozzle=r["Nozzle Number"], throat="C",
             tubing_od=4.5, circulating="reverse", manufacturer="National", raw_pump=None, pump_converted=False)
        for r in tracker.to_dict("records")],
        tests=[dict(date=r["WtDate"], oil_rate=r["WtOilVol"], fwat_rate=100., lift_wat=r["lift_wat"],
                    bhp=r["BHP"], pf_press=r["pf_press"]) for r in frame.to_dict("records")],
        bhp_daily=[])
    writes_on, hold_save, fail_save = False, False, False
    stale_pin = False
    release = asyncio.Event()
    clear_release = asyncio.Event()
    saves, starts, errors, job_values = [], [], [], {}
    context_reads = 0
    clears = 0
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe")
        page = await browser.new_page(viewport=dict(width=1440, height=1050))
        page.on("pageerror", lambda e: errors.append(str(e)))

        async def api(route):
            nonlocal context_reads, stale_pin, clears
            path, method = urlparse(route.request.url).path, route.request.method
            if path == "/api/meta":
                data = dict(app="WOFFL", version="fixture", physics_model="fixture", user=None,
                            writes_enabled=writes_on, warehouse_id="", deployed=False)
            elif path == "/api/wells":
                data = dict(wells=[dict(name="MPE-42", pad="E", has_survey=True)], source="databricks")
            elif path.endswith("/context"):
                context_reads += 1
                data = deepcopy(context)
            elif path.endswith("/jp-history"): data = history
            elif path.endswith("/tests"): data = dict(well="MPE-42", tests=[])
            elif path.endswith("/ipr-pin"):
                if method == "DELETE":
                    clears += 1
                    await clear_release.wait()
                    stale_pin = False
                    data = dict(cleared=True, message="Fixture anchor cleared.")
                else:
                    data = dict(status="stale" if stale_pin else "none", wt_uid=None, date_token=None,
                                entry_user=None, entry_datetime=None)
            elif path.endswith("/pump-match") and method == "POST":
                request = route.request.post_data_json
                starts.append(request)
                jid = f"preview-{len(starts)}"
                job_values[jid] = dict(job_id=jid, kind="pump-match", status="done", progress="done",
                    result=fixture(request), error=None, started_at="2026-03-01", seconds=1.)
                data = dict(job_id=jid)
            elif path.startswith("/api/pump-match/"):
                data = dict(cancel_requested=True) if method == "DELETE" else job_values[path.rsplit("/", 1)[-1]]
            elif path.endswith("/save-ipr"):
                payload = route.request.post_data_json
                saves.append(payload)
                assert writes_on and not set(payload) & {"ken", "kth", "kdi", "nozzle_area_factor", "hydraulics_model"}
                if hold_save: await release.wait()
                if fail_save:
                    data = dict(pinned=False, pin_skipped=True, pin_message=None, n_values=0, values_message="Fixture save failed.")
                else:
                    for key, value in payload.items():
                        if key in {"qwf_liq", "pwf", "res_pres", "form_wc", "form_gor", "surf_pres", "bubble_point", "form_temp"} and value is not None:
                            context["seeds"][{"qwf_liq": "qwf", "res_pres": "pres"}.get(key, key)] = value
                    data = dict(pinned=False, pin_skipped=True, pin_message=None, n_values=6, values_message="Saved well inputs.")
            else:
                await route.fulfill(status=422, json=dict(detail=dict(message="Fixture has no single-point solve")))
                return
            await route.fulfill(json=data)

        await page.route(re.compile(r"https?://[^/]+/api/"), api)
        await page.goto("http://127.0.0.1:5176/jp-history?well=MPE-42&match=1")
        button = page.get_by_role("button", name="Save well inputs", exact=True)
        await expect(button).to_be_visible()
        await expect(button).to_be_disabled()
        await expect(page.get_by_text("This app is read-only.", exact=False)).to_be_visible()
        assert not saves

        writes_on = True
        await page.reload()
        await expect(button).to_be_enabled()
        qwf = page.get_by_label("Total Liquid Rate at FBHP (qwf, BLPD)", exact=True)
        await qwf.fill("400.25")
        await qwf.press("Enter")
        await expect(page.get_by_text("1 well input differs", exact=False)).to_be_visible()
        await page.get_by_label("History well inputs").select_option("edited")
        await page.get_by_role("button", name="Run comparison", exact=True).click()
        await expect(page.get_by_text("Historical model results by installation", exact=False)).to_be_visible()
        assert starts[-1]["edited_inputs"]["qwf_liq"] == 400.25
        assert not saves and context["seeds"]["qwf"] == 200.
        # Every new edit hides the earlier preview immediately.
        await qwf.fill("450.75")
        await qwf.press("Enter")
        await expect(page.get_by_text("Historical model results by installation", exact=False)).to_have_count(0)
        await page.get_by_role("button", name="Run comparison", exact=True).click()
        await expect(page.get_by_text("Historical model results by installation", exact=False)).to_be_visible()
        assert starts[-1]["edited_inputs"]["qwf_liq"] == 450.75
        assert not saves
        await page.get_by_text("Review values and add a save note", exact=True).click()
        await page.get_by_label("Well save note").fill("Reviewed across historical pumps")
        hold_save = True
        await button.click()
        await expect(button).to_be_disabled()
        await qwf.fill("500.5")
        await qwf.press("Enter")
        release.set()
        await expect(page.get_by_text("New optimization runs will load these saved well inputs.", exact=False)).to_be_visible()
        assert saves[-1]["qwf_liq"] == 450.75 and saves[-1]["comment"] == "Reviewed across historical pumps"
        assert context["seeds"]["qwf"] == 450.75
        await expect(page.get_by_text("1 well input differs", exact=False)).to_be_visible()
        hold_save = False
        await button.click()
        await expect(page.get_by_text("Well inputs match the loaded database values.", exact=True)).to_be_visible()
        assert saves[-1]["qwf_liq"] == 500.5
        assert context_reads >= 3
        fail_save = True
        await qwf.fill("600")
        await qwf.press("Enter")
        await button.click()
        await expect(page.get_by_text("Fixture save failed.", exact=True)).to_be_visible()
        assert context["seeds"]["qwf"] == 500.5
        await expect(page.get_by_text("1 well input differs", exact=False)).to_be_visible()
        await page.get_by_text("Review values and add a save note", exact=True).click()
        await page.locator("main").evaluate("el => el.scrollTop = el.scrollHeight")
        await expect(button).to_be_in_viewport()
        await page.screenshot(path=str(ROOT / "build/well-input-save-desktop.png"), full_page=True)
        await page.set_viewport_size(dict(width=960, height=1050))
        await expect(button).to_be_in_viewport()
        await page.screenshot(path=str(ROOT / "build/well-input-save-narrow.png"), full_page=True)
        # Client navigation retains the same edited well; Solver has one save action.
        stale_pin = True
        await page.get_by_role("link", name="Solver", exact=True).click()
        await expect(button).to_have_count(1)
        await expect(button).to_be_visible()
        clear = page.get_by_role("button", name="Clear saved IPR", exact=True)
        await expect(clear).to_be_enabled()
        # Save and anchor clearing live in separate components but cannot race.
        hold_save = True
        release.clear()
        await button.click()
        await expect(clear).to_be_disabled()
        release.set()
        await expect(clear).to_be_enabled()
        await clear.click()
        await expect(button).to_be_disabled()
        clear_release.set()
        await expect(page.get_by_text("Fixture anchor cleared.", exact=True)).to_be_visible()
        await expect(button).to_be_enabled()
        assert clears == 1
        assert not errors, errors
        await browser.close()
    print(json.dumps(dict(status="passed", previews=len(starts), intercepted_saves=len(saves),
                         intercepted_clears=clears, context_reads=context_reads, browser_errors=errors)))


if __name__ == "__main__":
    asyncio.run(check())
