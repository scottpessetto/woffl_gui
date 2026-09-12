"""Fixture-only pad accounting/stress QA. Intercepts every API request."""
import asyncio
from copy import deepcopy
from pathlib import Path
import re
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]


async def check():
    from playwright.async_api import async_playwright, expect
    base = dict(current_pump="12B", test_oil=999., test_pf=None, pump=None, pump_state=None,
                oil=None, pf=None, form_water=None, suction=None, sonic=None, marginal_oil=None,
                ipr_source="saved", ipr_r2=None, has_friction=True, current_model_oil=None,
                current_model_pf=None, modeled_hardware_gain=None)
    rows = [dict(base, well="MPM-01", outcome="failed_model", outcome_reason="Fixture no lift solution."),
            dict(base, well="MPM-02", outcome="economic_shut_in", outcome_reason="Fixture viable candidates not allocated.", oil=0., pf=0.)]
    incomplete = dict(pad="M", rows=rows, n_wells=2, notes=[], meta=dict(header_psi=2600., feasible=None,
        recommendation_status="incomplete_exploratory"), coverage=dict(complete=False, expected_online=2,
        accounted_online=1, unaccounted_wells=["MPM-01"], rows=[dict(well=r["well"], pad="M", role="online", outcome=r["outcome"], reason=r["outcome_reason"]) for r in rows]))
    complete = deepcopy(incomplete)
    complete.update(robustness_available=True)
    complete["coverage"].update(complete=True, accounted_online=2, unaccounted_wells=[])
    complete["meta"].update(feasible=True, total_oil_bopd=500., current_model_oil_bopd=500., modeled_hardware_gain_bopd=0.,
                            comparison_basis="Same plan header and saved well inputs; recent tests are separate context.")
    for row in complete["rows"]:
        row.update(outcome="modeled", pump="12B", pump_state="installed", oil=250., pf=3000.,
                   current_model_oil=250., current_model_pf=3000., modeled_hardware_gain=0.)
    requests, errors = [], []
    hold, cancelled = False, False
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe")
        page = await browser.new_page(viewport=dict(width=1440, height=1050))
        page.on("pageerror", lambda e: errors.append(str(e)))
        await page.add_init_script("localStorage.setItem('woffl.optimize', JSON.stringify({pad:'M',offline:{},future:{},lastJob:{M:'pad-fixture'}}));")
        async def api(route):
            nonlocal hold, cancelled
            path, method = urlparse(route.request.url).path, route.request.method
            if path == "/api/meta":
                data = dict(app="WOFFL", version="fixture", physics_model="fixture", user=None, writes_enabled=False, warehouse_id="", deployed=False)
            elif path == "/api/wells":
                data = dict(wells=[dict(name=r["well"], pad="M", has_survey=True) for r in rows], source="fixture")
            elif path == "/api/optimize/pad-status":
                data = dict(pad="M", extras=[], wells=[dict(well=r["well"], pad="M", has_curve=True, has_friction=True,
                    saved_at="2026-09-12", saved_by="fixture", friction_keys=[], locks={}, pin_at=None, pin_user=None,
                    pump_calibration=dict(status="active", quality=None, coefficients={})) for r in rows])
            elif path == "/api/optimize/run" and method == "POST":
                data = dict(job_id="complete-fixture")
            elif path.startswith("/api/optimize/run/"):
                data = dict(job_id=path.rsplit("/", 1)[-1], kind="pad", status="done", error=None, progress="done", seconds=1., started_at="2026-09-12",
                            result=incomplete if path.endswith("pad-fixture") else complete)
            elif path == "/api/optimize/robustness" and method == "POST":
                requests.append(route.request.post_data_json)
                cancelled = False
                data = dict(job_id=f"stress-{len(requests)}")
            elif path.startswith("/api/optimize/robustness/"):
                if method == "DELETE":
                    cancelled = True
                    data = dict(cancelled=True)
                else:
                    scores = [dict(plan="Current", feasible=True, oil=500., machine_water=4000., budget=7000., regret=None, reason="Within modeled plant capacity."),
                              dict(plan="Proposed", feasible=None, oil=None, machine_water=None, budget=None, regret=None, reason="Fixture selected pump does not solve.")]
                    result = dict(request=requests[-1], water_key="totl_wat", assumptions="Fixture engineering cases; no probabilities or saved inputs.", min_comparable_oil_gain=None, max_comparable_oil_gain=None,
                        cases=[dict(case=dict(name="Base"), header_psi=2600., plans=scores, proposed_oil_delta=None, preferred=[])],
                        plans=[dict(plan=s["plan"], feasible_cases=int(s["feasible"] is True), failed_cases=int(s["feasible"] is None), infeasible_cases=0, preferred_when_both_feasible=0, comparable_cases=0, max_regret=None) for s in scores])
                    data = dict(status="cancelled" if cancelled else "running" if hold else "done", progress="Fixture running", result=None if hold else result, error=None)
            else:
                await route.fulfill(status=404, json=dict(detail=dict(message="Fixture data unavailable")))
                return
            await route.fulfill(json=data)
        await page.route(re.compile(r"https?://[^/]+/api/"), api)
        await page.goto((sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5177") + "/optimize")
        await page.get_by_role("button", name="M-Pad", exact=True).click()
        await expect(page.get_by_text("Incomplete exploratory run:", exact=False)).to_be_visible()
        await expect(page.get_by_text("MODEL UNAVAILABLE", exact=True)).to_be_visible()
        await expect(page.get_by_text("SHUT IN (modeled choice)", exact=True)).to_be_visible()
        await expect(page.get_by_text("Inputs and pump fit saved", exact=True).first).to_be_visible()
        await page.get_by_role("button", name="Run M optimization", exact=True).click()
        await expect(page.get_by_text("Modeled hardware gain", exact=True)).to_be_visible()
        await expect(page.get_by_text("Incomplete exploratory run:", exact=False)).to_have_count(0)
        await page.get_by_text("Stress-test current and proposed plans", exact=True).click()
        button = page.get_by_role("button", name="Run plan stress cases", exact=True)
        await button.click()
        await expect(page.get_by_text("Fixture selected pump does not solve.", exact=True)).to_be_visible()
        assert requests[0] == dict(source_job_id="complete-fixture", wc_points=3, gor_percent=20, header_psi=100, joint_cases=False)
        await page.get_by_label("Plan stress WC range").fill("4")
        await expect(page.get_by_text("Ranges changed. Run again to compare these cases.", exact=True)).to_be_visible()
        await expect(page.get_by_text("Fixture selected pump does not solve.", exact=True)).to_have_count(0)
        await page.get_by_label("Add two joint corners").check()
        await button.click()
        await expect(page.get_by_text("Fixture selected pump does not solve.", exact=True)).to_be_visible()
        assert requests[-1]["wc_points"] == 4 and requests[-1]["joint_cases"] is True
        await page.screenshot(path=str(ROOT / "build/pad-decision-desktop.png"), full_page=True)
        await page.set_viewport_size(dict(width=960, height=1050))
        await page.screenshot(path=str(ROOT / "build/pad-decision-narrow.png"), full_page=True)
        hold = True
        await button.click()
        await expect(page.get_by_role("button", name="Cancel", exact=True)).to_be_visible()
        await page.get_by_role("button", name="Cancel", exact=True).click()
        await expect(page.get_by_text("Comparison cancelled.", exact=True)).to_be_visible(timeout=10000)
        assert not errors, errors
        print(dict(status="passed", stress_starts=len(requests), cancelled=cancelled, browser_errors=errors))
        await browser.close()


if __name__ == "__main__":
    asyncio.run(check())
