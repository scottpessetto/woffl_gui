"""Fixture browser check; real Custom-well physics, no Databricks access.

Run with Vite on --url (default 5177), optional Playwright in build/browser-qa.
"""

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
    from server.services import sensitivity, solve
    from woffl.assembly import databricks_client

    def forbidden(*args, **kwargs):
        raise AssertionError("Browser fixtures cannot access Databricks")
    databricks_client.execute_query = forbidden
    databricks_client.execute_write = forbidden
    params = schemas.SimParams(ken=.2, kth=.7, kdi=.8, nozzle_area_factor=1.15).model_dump()
    params.update(oil_api=22., gas_sg=.65, wat_sg=1.02)
    context = dict(well="MPE-42", chars={}, chars_source="databricks", seeds=params,
        as_built_locks={}, prop_locks={}, pump=dict(nozzle_no="12", throat_ratio="B",
            date_set="2026-07-01T12:00:00Z", source="databricks"),
        pump_calibration=dict(status="none", coefficients={}), pf=None, clamped=[],
        ipr_source="manual", ipr_info=None, ipr_r2=None, test_count=2, saved_ipr_info="Fixture manual IPR")
    tests = [dict(wt_uid=uid, date=date, bhp=bhp, oil=oil, total_fluid=2*oil,
                  form_wc=.5, fgor=250, lift_wat=3500, whp=210, pf_press=3168,
                  water=oil, pf_source="fixture")
             for uid, date, bhp, oil in [(101, "2026-07-15", 900, 280), (102, "2026-08-15", 500, 380)]]
    requests, errors, jobs = [], [], {}
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe")
        page = await browser.new_page(viewport=dict(width=1500, height=1100))
        page.set_default_timeout(30000)
        expect.set_options(timeout=30000)
        page.on("pageerror", lambda e: errors.append(str(e)))

        async def api(route):
            path, method = urlparse(route.request.url).path, route.request.method
            body = route.request.post_data_json if method == "POST" else None
            if path == "/api/meta":
                result = dict(app="WOFFL", version="fixture", physics_model=schemas.MODEL_VERSION,
                              physics_notice="", user=None, writes_enabled=False, deployed=False, warehouse_id="")
            elif path == "/api/wells": result = dict(wells=[dict(name="MPE-42", pad="E", has_survey=False)], source="fixture")
            elif path.endswith("/context"): result = context
            elif path.endswith("/tests"): result = dict(well="MPE-42", tests=tests)
            elif path.endswith("/ipr-pin"): result = dict(status="none", wt_uid=None, date_token=None, entry_user=None, entry_datetime=None)
            elif path.endswith("/jp-history"): result = dict(well="MPE-42", source="fixture", installs=[], tests=[], bhp_daily=[], current_pump=None)
            elif path == "/api/solve":
                result = solve.solve_single("Custom", schemas.SimParams(**body["params"]))
            elif path == "/api/sensitivity":
                req = schemas.SensitivityRequest(**body)
                result = schemas.SensitivityResponse(**sensitivity.run_sensitivity("Custom", req.params,
                    {k: body.get(k) for k in ("target_psu", "target_qoil", "target_qliq", "target_qpf")},
                    req.bounds, req.wc_basis)).model_dump()
            elif path == "/api/sensitivity/combine":
                req = schemas.CombineRequest(**body)
                requests.append(deepcopy(body))
                result = sensitivity.run_combine("Custom", req.params,
                    {k: body.get(k) for k in ("target_psu", "target_qoil", "target_qliq", "target_qpf")},
                    req.knobs, wc_basis=req.wc_basis, test_key=req.test_key, installation_key=req.installation_key)
                result["request"]["well"] = "MPE-42"
                jid = f"sensitivity-fixture-{len(requests)}"
                jobs[jid] = schemas.CombineJobStatus(job_id=jid, kind="sensitivity", status="done", result=result,
                    started_at="2026-09-12T00:00:00Z", seconds=1).model_dump()
                result = dict(job_id=jid)
            elif path.startswith("/api/sensitivity/combine/"):
                result = jobs[path.rsplit("/", 1)[-1]]
            else:
                assert "save" not in path and "prop-lock" not in path, path
                await route.fulfill(status=422, json=dict(detail=dict(error="invalid", message="Unavailable in this fixture")))
                return
            await route.fulfill(json=result)

        await page.route(re.compile(r"https?://[^/]+/api/"), api)
        await page.goto(url + "/sensitivity?well=MPE-42")
        # Use the module instance actually loaded by Vite (including HMR URLs).
        await page.evaluate("""() => { window.qaModule = async (part) => {
            const url = performance.getEntriesByType('resource').map(r => r.name).filter(n => n.includes(part)).at(-1);
            return import(url);
        }; }""")
        await page.evaluate("""async () => { const {useParamsStore:p} = await qaModule('/src/state/params.ts'); p.getState().selectWell('MPE-42'); }""")
        await expect(page.get_by_label("Sensitivity WC basis")).to_be_visible()
        await page.evaluate("""async () => { const {useSensitivityStore:s} = await qaModule('/src/state/sensitivity.ts'); s.getState().setCompareKey('MPE-42', 'uid:101'); }""")
        toggle = page.get_by_role("button", name="Combined Permutations", exact=False)
        await toggle.click()
        panel = toggle.locator("..")
        await panel.locator("label").filter(has_text="Nozzle size").get_by_role("checkbox").check()
        run_button = panel.get_by_role("button", name="Run combination", exact=True)
        await expect(run_button).to_be_enabled()
        await run_button.click()
        await expect(panel.get_by_text("Sampled Scenario Envelope", exact=True)).to_be_visible()
        assert requests[-1]["test_key"] == "uid:101"
        assert requests[-1]["wc_basis"] == "fixed_oil_ipr"
        await expect(panel.get_by_role("button", name="Apply", exact=True).first).to_be_enabled()
        # Changing targets preserves the original study's labels and disables Apply.
        await page.evaluate("""async () => { const {useSensitivityStore:s} = await qaModule('/src/state/sensitivity.ts'); s.getState().setCompareKey('MPE-42', 'uid:102'); }""")
        await expect(panel.get_by_text("This study used different inputs", exact=False)).to_be_visible()
        await expect(panel.get_by_text("Submitted comparison: uid:101", exact=False)).to_be_visible()
        for button in await panel.get_by_role("button", name="Apply", exact=True).all():
            await expect(button).to_be_disabled()
        # Reverting the test restores validity; basis changes independently invalidate it.
        await page.evaluate("""async () => { const {useSensitivityStore:s} = await qaModule('/src/state/sensitivity.ts'); s.getState().setCompareKey('MPE-42', 'uid:101'); }""")
        await page.get_by_label("Sensitivity WC basis").select_option("anchor_measurement")
        await expect(panel.get_by_text("This study used different inputs", exact=False)).to_be_visible()
        await page.get_by_label("Sensitivity WC basis").select_option("fixed_oil_ipr")
        await expect(panel.get_by_role("button", name="Apply", exact=True).first).to_be_enabled()
        await panel.screenshot(path=str(ROOT / "build/sensitivity-study-desktop.png"))
        study = next(iter(jobs.values()))["result"]
        applicable = sorted((r for r in study["runs"] if r["error"] is None and r["applied_inputs"]), key=lambda r: r["score"])
        expected = {**study["request"]["params"], **applicable[0]["applied_inputs"]}
        await panel.get_by_role("button", name="Apply", exact=True).first.click()
        await expect(page).to_have_url(re.compile(r"/solver$"))
        await expect(page.get_by_text("Use a different test for comparison", exact=False)).to_be_visible()
        await asyncio.sleep(.6)
        actual = await page.evaluate("""async () => {
            const {useParamsStore:p} = await qaModule('/src/state/params.ts');
            const {useSensitivityStore:s} = await qaModule('/src/state/sensitivity.ts');
            return {params:p.getState().params, test:s.getState().compareKey['MPE-42'], note:p.getState().matchNote};
        }""")
        assert actual["params"] == expected, (actual["params"], expected)
        assert actual["test"] == "uid:101", actual
        assert "uid:101" in actual["note"]
        assert not errors, errors
        print(json.dumps(dict(passed=True, combination_requests=len(requests), browser_errors=errors,
                             checks="scope/Apply parity, immutable test targets, WC modes, stale Apply guard, comparison handoff")))
        await browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5177")
    asyncio.run(check(parser.parse_args().url))
