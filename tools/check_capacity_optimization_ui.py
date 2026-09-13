"""Offline browser QA for capacity constraints and CFP plans.

Every /api request is intercepted; this never starts real optimization or
writes field data. Run against the built SPA or Vite, passing its base URL.
"""

import asyncio
from copy import deepcopy
from pathlib import Path
import re
import sys
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
WELLS = [("MPM-01", "M"), ("MPM-02", "M"), ("MPB-01", "B"), ("MPB-02", "B"), ("MPG-01", "G")]
ASSUMPTION = "Fixture: motor-current and recycle limits still require field confirmation."


def pad_result(conditional):
    rows = [dict(well=name, current_pump="12B", test_oil=250., test_pf=3000.,
                 pump="12B", pump_state="installed", oil=250., pf=3000.,
                 form_water=100., suction=900., sonic=False, marginal_oil=None,
                 ipr_source="saved", ipr_r2=None, has_friction=True,
                 current_model_oil=250., current_model_pf=3000.,
                 modeled_hardware_gain=None if conditional else 0.,
                 outcome="modeled", outcome_reason="Fixture saved model.")
            for name, pad in WELLS if pad == "M"]
    return dict(pad="M", rows=rows, n_wells=2, notes=[], robustness_available=False,
                coverage=dict(complete=True, expected_online=2, accounted_online=2,
                              unaccounted_wells=[], rows=[]),
                meta=dict(header_psi=2600., total_oil_bopd=500.,
                          current_model_oil_bopd=500.,
                          modeled_hardware_gain_bopd=None if conditional else 0.,
                          feasible=not conditional, in_range=not conditional,
                          recirc=conditional, over_capacity=False,
                          water_key="totl_wat", lambda_used=0.,
                          recommendation_status="conditional_operating_limits" if conditional else "complete_model_coverage",
                          operating_assumptions=[ASSUMPTION] if conditional else []))


def cfp_result():
    actions = [dict(well="MPB-02", pad="B", type="shut_in", to="SI",
                    own_oil_delta=-100., own_water_delta=-3000., **{"from": "12B"}),
               dict(well="FUTURE-B", pad="B", type="bring_online", to="13C (clean)",
                    own_oil_delta=300., own_water_delta=3000., **{"from": "OFF"})]
    choices = {"MPB-01": "12B", "MPB-02": "SI", "MPG-01": "12B", "FUTURE-B": "13C (clean)"}
    baseline = {"MPB-01": "12B", "MPB-02": "12B", "MPG-01": "12B", "FUTURE-B": "OFF"}
    plan = dict(lam=None, pressure=2800., oil=600., water=6000., at_trip=False,
                actions=actions, n_changes=2, choices=choices, feasible=True,
                converged=True, pressure_residual_psi=0.)
    pair = dict(bring_on=dict(actions[1], standalone_feasible=False,
                             standalone_domain_reason="required_pressure_below_response_grid",
                             fleet_oil_delta=None),
                offset=dict(actions[0], standalone_feasible=True, fleet_oil_delta=-100.),
                fleet_oil_delta=200., own_oil_delta=200., own_water_delta=0.,
                pressure_after=2800., pressure_delta=0., at_trip=False)
    rows = [dict(well=well, pad="G" if well == "MPG-01" else "B", online=well != "FUTURE-B",
                 baseline_label=baseline[well], plan_label=choices[well],
                 baseline_oil=oil0, plan_oil=oil1, baseline_water=water0,
                 plan_water=water1, changed=baseline[well] != choices[well])
            for well, oil0, oil1, water0, water1 in
            [("MPB-01", 100., 100., 1000., 1000.),
             ("MPB-02", 100., 0., 3000., 0.), ("MPG-01", 200., 200., 2000., 2000.),
             ("FUTURE-B", 0., 300., 0., 3000.)]]
    return dict(pads=["B", "G", "C", "J"], notes=[], n_wells=4, p0_psi=2800., wells=rows,
                coverage=dict(complete=True, expected_online=3, accounted_online=3,
                              unaccounted_wells=[], rows=[dict(well=r["well"], pad=r["pad"],
                              role="online" if r["online"] else "future", outcome="modeled",
                              reason="Fixture supported response table.") for r in rows]),
                summary=dict(today=dict(pressure=2800., oil=400., water=6000.,
                                        n_online=3, n_bol_candidates=1),
                             lambda_bopd_per_psi=None, singles=[], n_positive_singles=0,
                             pairs=[pair], frontier=[dict(lam=0., pressure=2800.,
                                                        oil=600., water=6000., at_trip=False)],
                             plan=plan, plan_gain=200., baseline=baseline,
                             plan_status="feasible", required_wells=["MPB-01", "FUTURE-B"],
                             baseline_meets_requirements=False,
                             search_scope=dict(method="lambda_moves_and_bounded_neighborhood",
                                               global_optimum_on_surfaces=False,
                                               direct_solver_validated=False)))


async def check():
    from playwright.async_api import async_playwright, expect

    requests, errors, unexpected_writes, jobs = [], [], [], {}
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5177"
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True, executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe")
        page = await browser.new_page(viewport=dict(width=1440, height=1050))
        page.on("pageerror", lambda error: errors.append(str(error)))
        await page.add_init_script("localStorage.setItem('woffl.optimize', JSON.stringify({"
                                   "pad:'M',offline:{},keepOnline:{},requiredOnline:{},future:{},lastJob:{}}));")

        async def api(route):
            parsed = urlparse(route.request.url)
            path, method = parsed.path, route.request.method
            if path == "/api/meta":
                data = dict(app="WOFFL", version="fixture", physics_model="fixture",
                            user=None, writes_enabled=False, warehouse_id="", deployed=False)
            elif path == "/api/wells":
                data = dict(wells=[dict(name=name, pad=pad, has_survey=True) for name, pad in WELLS],
                            source="fixture")
            elif path == "/api/well-sort/tables":
                data = dict(offline=[], ltsi=[], online=[])
            elif path == "/api/optimize/pad-status":
                query = parse_qs(parsed.query)
                pad = query.get("pad", ["M"])[0]
                rows = [dict(well=name, pad=p, has_curve=True, has_friction=True,
                             saved_at="2026-09-12", saved_by="fixture", friction_keys=[],
                             locks={}, pin_at=None, pin_user=None,
                             pump_calibration=dict(status="active", quality=None, coefficients={}))
                        for name, p in WELLS]
                data = dict(pad=pad, wells=[r for r in rows if r["pad"] == pad],
                            extras=[r for r in rows if r["pad"] != pad])
            elif path == "/api/optimize/run" and method == "POST":
                request = route.request.post_data_json
                requests.append(deepcopy(request))
                job_id = f"capacity-fixture-{len(requests)}"
                if request["kind"] == "cfp":
                    result = cfp_result()
                elif request["strategy"] == "choke":
                    # Request validation is the choke check; empty fixture results
                    # avoid pretending this browser test establishes physics.
                    result = dict(pad="M", plan=[], notes=[], n_wells=0,
                                  meta=dict(header_psi=2600., total_oil_bopd=0.,
                                            n_full=0, n_choked=0, n_shut=0, feasible=False,
                                            projected_d_oil_bopd=None,
                                            operating_assumptions=[ASSUMPTION]))
                else:
                    result = pad_result(conditional=len(requests) == 1)
                jobs[job_id] = dict(job_id=job_id, kind=request["kind"], status="done",
                                    error=None, progress="done", seconds=1.,
                                    started_at="2026-09-12", result=result)
                data = dict(job_id=job_id)
            elif path.startswith("/api/optimize/run/") and method == "GET":
                data = jobs[path.rsplit("/", 1)[-1]]
            else:
                if method not in {"GET", "HEAD", "OPTIONS"}:
                    unexpected_writes.append(dict(path=path, method=method))
                await route.fulfill(status=404, json=dict(detail=dict(message="Fixture data unavailable")))
                return
            await route.fulfill(json=data)

        await page.route(re.compile(r"https?://[^/]+/api/"), api)

        async def run(name):
            async with page.expect_response(lambda response: urlparse(response.url).path == "/api/optimize/run"
                                            and response.request.method == "POST"):
                await page.get_by_role("button", name=name, exact=True).click()
            await expect(page.get_by_role("button", name=name, exact=True)).to_be_enabled()
            return requests[-1]

        async def add_future(name, donor):
            await page.get_by_placeholder("Future well name", exact=True).fill(name)
            await page.get_by_placeholder("existing well (any pad)", exact=True).fill(donor)
            await page.get_by_role("button", name="Add future well", exact=True).click()
            await expect(page.get_by_label(f"Require {name} online", exact=True)).to_be_checked()

        await page.goto(base_url + "/optimize")
        await page.get_by_role("button", name="M-Pad", exact=True).click()
        await expect(page.get_by_label("Maximize oil within capacity", exact=True)).to_be_checked()
        await page.get_by_label("Require MPM-01 online", exact=True).check()
        first = await run("Run M optimization")
        assert first["lambda_bopd_per_bpd"] is None, first
        assert first["required_wells"] == ["MPM-01"], first
        await expect(page.get_by_text("This plan does not meet all modeled operating limits.", exact=False)).to_be_visible()
        await expect(page.get_by_text("Machine flow is outside the modeled operating range", exact=False)).to_be_visible()
        await expect(page.get_by_text(ASSUMPTION, exact=True)).to_be_visible()
        (ROOT / "build").mkdir(exist_ok=True)
        await page.screenshot(path=str(ROOT / "build/capacity-conditional-desktop.png"), full_page=True)

        # The same flags clear on a supported result; warnings must not stick.
        await run("Run M optimization")
        await expect(page.get_by_text(ASSUMPTION, exact=True)).to_have_count(0)
        await expect(page.get_by_text("This plan does not meet all modeled operating limits.", exact=False)).to_have_count(0)

        await add_future("FUTURE-M", "MPM-01")
        await page.get_by_label("Planned nozzle for FUTURE-M", exact=True).select_option("13")
        await page.get_by_label("Planned throat for FUTURE-M", exact=True).select_option("C")
        for duplicate in ("mpm-01", "future-m"):
            await page.get_by_placeholder("Future well name", exact=True).fill(duplicate)
            await page.get_by_placeholder("existing well (any pad)", exact=True).fill("MPM-02")
            await expect(page.get_by_role("button", name="Add future well", exact=True)).to_be_disabled()
            await expect(page.get_by_role("button", name="Add future well", exact=True)).to_have_attribute(
                "title", "Use a name that does not identify an existing or planned well")
        await page.get_by_placeholder("Future well name", exact=True).fill("")
        await page.get_by_label("Strategy", exact=False).select_option("choke")
        choke = await run("Run M optimization")
        assert choke["strategy"] == "choke" and choke["lambda_bopd_per_bpd"] is None, choke
        assert choke["required_wells"] == ["MPM-01"], choke
        assert choke["future"] == [dict(name="FUTURE-M", match="MPM-01", pad="M",
                                        require_online=True, nozzle="13", throat="C")], choke
        await expect(page.get_by_text("This choke plan does not meet all modeled operating limits.", exact=False)).to_be_visible()
        await expect(page.get_by_text(ASSUMPTION, exact=True)).to_be_visible()

        await page.get_by_role("button", name="Pad review", exact=True).click()
        await page.get_by_role("button", name="B", exact=True).click()
        await page.get_by_label("Require MPB-01 online", exact=True).check()
        await add_future("FUTURE-B", "MPB-01")
        # Cross-pad duplicates must also be rejected.
        await page.get_by_placeholder("Future well name", exact=True).fill("future-m")
        await page.get_by_placeholder("existing well (any pad)", exact=True).fill("MPB-01")
        await expect(page.get_by_role("button", name="Add future well", exact=True)).to_be_disabled()

        await page.get_by_role("button", name="CFP run", exact=True).click()
        await page.get_by_label("Reference PW discharge (psi)", exact=True).fill("2800")
        await page.get_by_text("Pad PF at the same reference conditions", exact=True).click()
        await expect(page.get_by_text("The reference discharge is a manual scenario value.", exact=False)).to_be_visible()
        await page.get_by_label("B-Pad PF (psi)", exact=True).fill("2620")
        await page.get_by_label("G-Pad PF (psi)", exact=True).fill("2740")
        cfp = await run("Run CFP optimization")
        assert cfp["kind"] == "cfp" and cfp["p0_psi"] == 2800, cfp
        assert cfp["cfp_pad_pf_psi"] == {"B": 2620, "G": 2740}, cfp
        assert cfp["required_wells"] == ["MPB-01"], cfp
        assert cfp["future"] == [dict(name="FUTURE-B", match="MPB-01", pad="B", require_online=True)], cfp
        assert cfp["lambda_bopd_per_bpd"] is None, cfp
        await expect(page.get_by_text("Bring online with an offset", exact=True)).to_be_visible()
        await expect(page.get_by_role("cell", name="FUTURE-B: 13C (clean)", exact=True)).to_be_visible()
        await expect(page.get_by_role("cell", name="MPB-02: 12B to SI", exact=True)).to_be_visible()
        await expect(page.get_by_text("Required online: MPB-01, FUTURE-B", exact=True)).to_be_visible()
        await expect(page.get_by_text("a direct final-pressure well solve has not been performed.", exact=False)).to_be_visible()
        await expect(page.get_by_text("This older result has no complete well accounting.", exact=False)).to_have_count(0)
        await page.screenshot(path=str(ROOT / "build/capacity-cfp-desktop.png"), full_page=True)
        await page.get_by_text("Bring online with an offset", exact=True).locator("..").screenshot(
            path=str(ROOT / "build/capacity-cfp-pair.png"))
        await page.set_viewport_size(dict(width=960, height=1050))
        await expect(page.get_by_role("cell", name="FUTURE-B: 13C (clean)", exact=True)).to_be_visible()
        await page.screenshot(path=str(ROOT / "build/capacity-cfp-narrow.png"), full_page=True)
        assert not errors, errors
        assert not unexpected_writes, unexpected_writes
        print(dict(status="passed", fixture_runs=len(requests), browser_errors=errors,
                   unexpected_writes=unexpected_writes,
                   checks=["required online", "future default required", "planned choke pump",
                           "duplicate identities", "maximum-oil default", "conditional warnings",
                           "CFP joint pairs", "coherent reference pressures"]))
        await browser.close()


if __name__ == "__main__":
    asyncio.run(check())
