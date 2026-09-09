"""Browser check against Vite; all API traffic is intercepted locally.

Needs optional Playwright (not an app dependency), installed Chromium/Chrome,
and `npm run dev -- --host 127.0.0.1` in web/. No Databricks connection is used.
Uses the real FastAPI compute routes for successful scenarios, and injected
partial/failed responses to exercise the UI's incomplete-range presentation.
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


async def check(url, executable):
    from fastapi.testclient import TestClient
    from playwright.async_api import async_playwright, expect

    from server.main import app
    from woffl.assembly import databricks_client

    def forbidden(*args, **kwargs):
        raise AssertionError("Browser checks must not access Databricks")

    databricks_client.execute_query = forbidden
    databricks_client.execute_write = forbidden
    client = TestClient(app)  # no lifespan warmup; all well computations are Custom
    calls, errors, unexpected = [], [], []
    mode = "normal"
    last_response = {}

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(executable_path=executable, headless=True)
        page = await browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1)
        page.on("pageerror", lambda error: errors.append(str(error)))

        async def route_api(route):
            nonlocal last_response
            request = route.request
            path = urlparse(request.url).path
            body = request.post_data_json if request.method == "POST" else None
            if path == "/api/meta":
                result = dict(app="WOFFL", version="preview", physics_model="entry-energy-v2",
                              physics_notice="", user=None, writes_enabled=False, warehouse_id="", deployed=False)
            elif path == "/api/wells":
                result = dict(wells=[], source="csv_fallback")
            elif path in {"/api/solve", "/api/solve/wc-uncertainty"}:
                assert body["well"] == "Custom"
                response = client.post(path, json=body)
                result = response.json()
                if response.status_code != 200:
                    await route.fulfill(status=response.status_code, json=result)
                    return
                if path.endswith("wc-uncertainty"):
                    calls.append(body)
                    if mode == "late" and body["uncertainty_points"] == 10:
                        await asyncio.sleep(1.5)
                    if mode in {"partial", "failed"}:
                        failed_points = result["points"][:1] if mode == "partial" else result["points"]
                        for point in failed_points:
                            point.update(oil=None, bhp=None, error="No lift solution for this WC scenario")
                        good = [p for p in result["points"] if p["error"] is None]
                        base = next(p for p in result["points"] if p["wc"] == result["wc_base"])
                        result.update(complete=False, solved_count=len(good), base_solved=base["error"] is None)
                        for metric in ("oil", "bhp"):
                            values = [p[metric] for p in good]
                            result[metric] = dict(low=min(values), base=base[metric], high=max(values)) if values else None
                    last_response = result
            else:
                unexpected.append(path)
                await route.fulfill(status=404, json={"detail": {"error": "http", "message": "Unexpected QA request"}})
                return
            await route.fulfill(json=result)

        await page.route(re.compile(r"https?://[^/]+/api/"), route_api)
        await page.goto(url + "/solver")
        await page.evaluate("""async () => {
            const { useParamsStore } = await import('/src/state/params.ts');
            useParamsStore.getState().set('form_wc', .8);
            useParamsStore.getState().run();
        }""")
        toggle = page.get_by_role("button", name="WC uncertainty")
        await expect(toggle).to_be_visible()
        assert not calls, "collapsed card must not calculate scenarios"
        await toggle.click()
        card = toggle.locator("..")
        width = page.get_by_label("Watercut uncertainty", exact=True)
        await expect(card.get_by_text("WC 75.0% to 85.0%", exact=True)).to_be_visible()
        assert last_response["sample_count"] == 9
        await expect(page.get_by_text("Loading well data", exact=True)).to_have_count(0)
        await expect(card.get_by_text("Lower", exact=True)).to_have_count(2)
        await expect(card.get_by_text("Upper", exact=True)).to_have_count(2)
        await card.screenshot(path=str(ROOT/"build/wc-uncertainty-card.png"))
        await page.screenshot(path=str(ROOT/"build/wc-uncertainty-desktop.png"), full_page=True)

        # A late old response must not restore stale bounds after another edit.
        mode = "late"
        async with page.expect_request(lambda r: r.url.endswith("wc-uncertainty") and r.post_data_json["uncertainty_points"] == 10):
            await width.fill("10")
            await expect(card.get_by_text("Updating oil and BHP ranges", exact=True)).to_be_visible()
            await expect(card.get_by_text("Lower", exact=True)).to_have_count(0)
        await width.fill("2")
        await expect(card.get_by_text("WC 78.0% to 82.0%", exact=True)).to_be_visible()
        await asyncio.sleep(1.7)
        await expect(card.get_by_text("WC 78.0% to 82.0%", exact=True)).to_be_visible()
        mode = "normal"

        # Invalid drafts must not fetch or continue showing the old envelope.
        before = len(calls)
        await width.fill("")
        await expect(card.get_by_role("alert")).to_contain_text("Enter a value")
        await expect(card.get_by_text("Lower", exact=True)).to_have_count(0)
        await asyncio.sleep(.6)
        assert len(calls) == before
        await width.fill("0")
        await expect(card.get_by_text("WC 80.0% to 80.0%", exact=True)).to_be_visible()
        assert last_response["sample_count"] == 1

        # Reopening identical inputs uses the query cache.
        before = len(calls)
        await toggle.click()
        await toggle.click()
        await expect(card.get_by_text("WC 80.0% to 80.0%", exact=True)).to_be_visible()
        await asyncio.sleep(.5)
        assert len(calls) == before

        # Parameter edits while collapsed do no envelope work; reopen with clipping.
        await toggle.click()
        await page.evaluate("""async () => {
            const { useParamsStore } = await import('/src/state/params.ts');
            useParamsStore.getState().set('form_wc', .97);
        }""")
        await asyncio.sleep(.6)
        assert len(calls) == before
        await toggle.click()
        await width.fill("5")
        await expect(card.get_by_text("WC 92.0% to 99.0%", exact=True)).to_be_visible()
        await expect(card.get_by_text("WC range limited to 0% to 99%.", exact=False)).to_be_visible()

        # Partial failure is visible before its bounds and exposes failed samples.
        mode = "partial"
        await width.fill("6")
        await expect(card.get_by_role("alert")).to_contain_text("Incomplete range")
        await card.get_by_text("Unsolved WC scenarios", exact=True).click()
        await expect(card.get_by_text("No lift solution for this WC scenario", exact=False)).to_be_visible()
        await card.screenshot(path=str(ROOT/"build/wc-uncertainty-partial.png"))
        mode = "failed"
        await width.fill("7")
        await expect(card.get_by_role("alert")).to_contain_text("bounds are unavailable")
        await expect(card.get_by_text("Lower", exact=True)).to_have_count(0)

        # Responsive card and keyboard-accessible assumptions.
        mode = "normal"
        await width.fill("4")
        await expect(card.get_by_text("WC 93.0% to 99.0%", exact=True)).to_be_visible()
        await page.set_viewport_size({"width": 600, "height": 1000})
        await card.scroll_into_view_if_needed()
        await card.evaluate("el => { el.style.width = '350px'; el.style.maxWidth = '100%'; }")
        await expect(card.get_by_role("button", name="Assumptions (explainer)")).to_be_visible()
        await card.get_by_role("button", name="Assumptions (explainer)").focus()
        await expect(card.get_by_role("tooltip")).to_be_visible()
        box = await card.get_by_role("tooltip").bounding_box()
        assert box["x"] >= 0 and box["x"] + box["width"] <= 600
        await width.focus()
        await card.screenshot(path=str(ROOT/"build/wc-uncertainty-narrow.png"))
        assert await card.evaluate("el => el.scrollWidth <= el.clientWidth")

        # Dewatering does not request or present oil uncertainty.
        before = len(calls)
        await page.evaluate("""async () => {
            const { useParamsStore } = await import('/src/state/params.ts');
            useParamsStore.getState().set('model_as_water', true);
        }""")
        await expect(card.get_by_text("Available in oil mode", exact=False)).to_be_visible()
        await expect(card.get_by_text("Lower", exact=True)).to_have_count(0)
        await asyncio.sleep(.6)
        assert len(calls) == before
        assert not unexpected, unexpected
        assert not errors, errors
        print(json.dumps(dict(passed=True, wc_requests=len(calls), console_errors=errors,
                              checks="collapsed/cached, numeric bounds, debounce/late response, invalid draft, "
                                     "zero width, clipping, partial/all failure, narrow layout, dewatering")))
        await browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5173")
    parser.add_argument("--executable", default=r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    args = parser.parse_args()
    asyncio.run(check(args.url, args.executable))
