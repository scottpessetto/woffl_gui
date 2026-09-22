"""Offline browser QA of the built separator page; intercept every API request.

From the repository: PYTHONPATH=build/browser-qa;. venv/Scripts/python.exe
tools/check_separator_samples_ui.py. Build web/dist first. No live API is started.
"""

import asyncio
import csv
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import io
from pathlib import Path
import re
import threading
from urllib.parse import urlparse

import pandas as pd
from fastapi.testclient import TestClient

from server.main import app
from server.services.tools import sep_oil_loss as loss

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "separator-recovery-2026-09-14"


class StaticApp(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "web" / "dist"), **kwargs)

    def do_GET(self):
        if self.path.startswith("/tools/"):
            self.path = "/index.html"
        super().do_GET()

    def log_message(self, *args):
        pass


async def check(base_url):
    from playwright.async_api import async_playwright, expect

    start = pd.Timestamp("2026-08-17T08:00:00-08:00")
    raw = pd.DataFrame(
        [(tag, start + pd.Timedelta(minutes=m), value)
         for m in range(61)
         for tag, value in ((loss.FLOW_TAG, 72000), (loss.WC_TAG, 99),
                            (loss.LEVEL_TAG, 50), (loss.LEVEL_SP_TAG, 50))],
        columns=["tag", "t", "value"],
    )
    loss._raw = lambda days: raw
    client = TestClient(app)  # No lifespan/warmup; the historian seam is replaced.
    errors, unexpected = [], []
    release, waiting = asyncio.Event(), asyncio.Event()
    hold = False
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        )
        page = await browser.new_page(viewport={"width": 1440, "height": 1100})
        await page.add_init_script("localStorage.setItem('woffl.scottsTools', '1')")
        page.on("pageerror", lambda err: errors.append(str(err)))

        async def api(route):
            url = urlparse(route.request.url)
            path = url.path
            if path == "/api/meta":
                await route.fulfill(json=dict(app="WOFFL", version="fixture", physics_model="fixture",
                                             writes_enabled=False, deployed=False, user=None, warehouse_id=""))
            elif path == "/api/wells":
                await route.fulfill(json=dict(wells=[], source="fixture"))
            elif path in ("/api/tools/sep-oil-loss", "/api/tools/sep-oil-loss/day"):
                response = client.get(path + "?" + url.query)
                await route.fulfill(status=response.status_code, json=response.json())
            elif path == "/api/tools/sep-oil-loss/samples":
                response = client.post(path + "?" + url.query, content=route.request.post_data_buffer,
                                       headers={"content-type": route.request.headers["content-type"]})
                if hold:
                    waiting.set()
                    await release.wait()
                await route.fulfill(status=response.status_code, json=response.json())
            else:
                unexpected.append(path)
                await route.fulfill(status=404, json={"detail": "Offline fixture has no such route"})

        await page.route(re.compile(r"https?://[^/]+/api/"), api)
        await page.goto(base_url + "/tools/sep-oil-loss")
        await expect(page.get_by_text("Validate with field samples", exact=True)).to_be_visible()
        await page.get_by_text("Enter a sample", exact=True).click()
        await page.get_by_label("Date", exact=True).fill("2026-08-17")
        await page.get_by_label("Time (Alaska)", exact=True).fill("08:04")
        await page.get_by_label("Lab result (units selected below)").fill("1000")
        await page.get_by_role("button", name="Add to sample log").click()
        await page.get_by_role("button", name="Review sample log").click()
        await expect(page.get_by_role("cell", name="confirm units", exact=True)).to_be_visible()

        await page.get_by_label("Lab units").select_option("ppmv")
        await expect(page.get_by_role("cell", name="confirm units", exact=True)).to_have_count(0)
        await page.get_by_role("button", name="Compare samples", exact=True).click()
        await expect(page.get_by_role("cell", name="matched", exact=True)).to_be_visible()
        await expect(page.get_by_role("cell", name="72.0", exact=True)).to_be_visible()
        await expect(page.get_by_role("cell", name="+0.9000", exact=True)).to_be_visible()
        async with page.expect_download() as download:
            await page.get_by_role("button", name="Download comparisons", exact=True).click()
        rows = list(csv.DictReader(io.StringIO(Path(await (await download.value).path()).read_text(encoding="utf-8-sig"))))
        assert rows[0]["date"] == "2026-08-17"
        assert rows[0]["units"] == "ppmv" and rows[0]["status"] == "matched"

        # A result returned after an input edit must never reappear.
        hold = True
        await page.get_by_role("button", name="Compare samples", exact=True).click()
        await asyncio.wait_for(waiting.wait(), timeout=10)
        await page.get_by_label("Lab units").select_option("mg/L")
        release.set()
        await expect(page.get_by_role("button", name="Compare samples", exact=True)).to_be_disabled()
        await page.wait_for_timeout(200)
        await expect(page.get_by_role("cell", name="matched", exact=True)).to_have_count(0)
        hold = False
        await page.get_by_label("Oil density (kg/m3)").fill("850")
        await page.get_by_role("button", name="Compare samples", exact=True).click()
        await expect(page.get_by_role("cell", name="matched", exact=True)).to_be_visible()
        assert await page.locator("svg").count() > 0
        OUT.mkdir(parents=True, exist_ok=True)
        await page.screenshot(path=str(OUT / "sample-comparison.png"), full_page=True)
        assert not errors, errors
        assert not unexpected, unexpected
        await browser.close()
    print("PASS: manual log, unknown units, matched rates, export, density gate, stale response; zero browser errors")


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 0), StaticApp)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        asyncio.run(check(f"http://127.0.0.1:{server.server_port}"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
