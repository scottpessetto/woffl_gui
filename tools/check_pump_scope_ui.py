"""Optional Playwright QA against Vite; all API reads and saves intercepted.

Run with PYTHONPATH=build/browser-qa;. and Vite on 127.0.0.1:5173.
An optional first argument overrides the base URL.
The save responses below are browser fixtures; no database writes occur.
"""
import asyncio
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


async def check():
    from fastapi.testclient import TestClient
    from playwright.async_api import async_playwright, expect
    from server.main import app

    coefs = dict(ken=.005, kth=.386, kdi=.072, nozzle_area_factor=1.01)
    scope = dict(status="none", coefficients={}, quality=None)
    context = dict(well="MPE-42", chars={}, chars_source="databricks", seeds=dict(
        nozzle_no="13", area_ratio="C", form_wc=.74, qwf=1521, pwf=643, pres=1054),
        as_built_locks={}, prop_locks={}, pump=dict(nozzle_no="13", throat_ratio="C",
        date_set="2026-08-10", source="databricks"), pump_calibration=scope,
        pf=None, clamped=[], jpump_md=None, ipr_info=None,
        ipr_source="manual", ipr_r2=None, test_count=0, saved_ipr_info="Manual IPR point")
    fit = dict(ken=.005, kth=.386, kdi=.072, fnz=1.01, n_used=19, n_dropped=1,
        rms_bhp_psi=76., rms_pf_pct=61.5, rms_dbhp_psi=72., implied_beta=.268,
        railed=["ken"], message="Provisional fit; check field response.")
    result = dict(well="MPE-42", pump="13C", era_start="2026-08-10", n_daily=16, n_test=4,
        ppf_spread=1625., refusal=None, method="event", fallback_reason=None, single=None,
        fit=fit, mined_beta=.062, mined_beta_source="well", current=dict(ken=.03, kth=.3, kdi=.4))
    client = TestClient(app)  # no lifespan, no warehouse warmup
    errors, requests = [], []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True,
            executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe")
        page = await browser.new_page(viewport=dict(width=1440, height=1150))
        page.on("pageerror", lambda e: errors.append(str(e)))

        async def api(route):
            path = urlparse(route.request.url).path
            body = route.request.post_data_json if route.request.method == "POST" else None
            if path == "/api/meta":
                data = dict(app="WOFFL", version="preview", physics_model="entry-energy-v2",
                    user=None, writes_enabled=True, warehouse_id="", deployed=False)
            elif path == "/api/wells": data = dict(wells=[], source="csv_fallback")
            elif path.endswith("/context"): data = context
            elif path.endswith("/tests"): data = dict(well="MPE-42", tests=[])
            elif path.endswith("/ipr-pin"): data = dict(status="none")
            elif path.endswith("/jp-history"):
                data = dict(well="MPE-42", installs=[], tests=[], bhp_daily=[], source="databricks", current_pump="13C")
            elif path == "/api/solve":
                response = client.post(path, json={**body, "well": "Custom"})
                await route.fulfill(status=response.status_code, json=response.json()); return
            elif path == "/api/optimize/run/preview-fit":
                data = dict(job_id="preview-fit", kind="event_cal", status="done", result=result)
            elif path.endswith("/pump-calibration"):
                requests.append((path, body))
                assert body == {"job_id": "preview-fit"}
                scope.update(status="active", coefficients=coefs, quality=dict(n=19, bhp=76., pf=61.5, bounds=["ken"]),
                    message="Saved calibration applies only to this installed pump.")
                data = dict(message="Saved calibration for installed 13C (2026-08-10).")
            elif path.endswith("/save-ipr"):
                requests.append((path, body))
                assert not set(body) & {*coefs, "mach_crit"}
                data = dict(pinned=False, pin_skipped=True, pin_message=None, n_values=6, values_message="Saved well inputs.")
            else:
                await route.fulfill(status=404, json=dict(detail=dict(error="invalid", message="No preview data"))); return
            await route.fulfill(json=data)

        await page.route(re.compile(r"https?://[^/]+/api/"), api)
        base_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5173"
        await page.goto(f"{base_url.rstrip('/')}/solver")
        await page.evaluate("""async (ctx) => {
            const resources = performance.getEntriesByType('resource').map(r => r.name);
            const paramUrl = resources.findLast(u => u.includes('/src/state/params.ts')) ?? '/src/state/params.ts';
            const { useParamsStore } = await import(paramUrl);
            window.qaParams = useParamsStore;
            useParamsStore.getState().selectWell(ctx.well);
            useParamsStore.getState().applyContext(ctx);
            const { useOptimizeStore } = await import(resources.findLast(u => u.includes('/src/state/optimize.ts')) ?? '/src/state/optimize.ts');
            useOptimizeStore.getState().setLastJob('event_cal:MPE-42', 'preview-fit');
        }""", context)
        try:
            await expect(page.get_by_role("button", name="Apply to inputs", exact=True)).to_be_enabled()
        except Exception:
            print(errors, await page.locator("body").inner_text())
            raise
        async with page.expect_response(lambda r: urlparse(r.url).path == "/api/solve" and r.request.post_data_json["params"]["ken"] == .005):
            await page.get_by_role("button", name="Apply to inputs", exact=True).click()
        await expect(page.get_by_text("Session coefficients differ", exact=False)).to_be_visible()
        await page.get_by_role("button", name="Save installed-pump calibration", exact=True).click()
        await expect(page.get_by_text("Saved fit: 19 points", exact=False)).to_be_visible()
        await page.get_by_role("button", name="Save well inputs", exact=True).click()
        await expect(page.get_by_text("New optimization runs will load these saved well inputs.", exact=False)).to_be_visible()
        assert len(requests) == 2
        await page.screenshot(path=str(ROOT / "build/pump-scope-desktop.png"), full_page=True)

        await page.get_by_role("button", name="Try clean replacement", exact=True).click()
        await expect(page.get_by_text("Clean replacement · 13C", exact=True)).to_be_visible()
        p = await page.evaluate("""() => window.qaParams.getState().params""")
        assert (p["ken"], p["kth"], p["kdi"], p["nozzle_area_factor"], p["form_wc"]) == (.03, .3, .4, 1., .74)
        await page.get_by_role("button", name="Restore installed pump", exact=True).click()
        p = await page.evaluate("""() => window.qaParams.getState().params""")
        assert all(p[k] == v for k, v in coefs.items())
        await page.set_viewport_size(dict(width=1080, height=1100))
        await page.screenshot(path=str(ROOT / "build/pump-scope-narrow.png"), full_page=True)
        assert not errors, errors
        await browser.close()
    print(json.dumps(dict(status="passed", browser_errors=errors, mocked_saves=len(requests))))


if __name__ == "__main__":
    asyncio.run(check())
