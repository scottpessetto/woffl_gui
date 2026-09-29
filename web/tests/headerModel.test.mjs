import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "typescript";

const dataModule = (source) => `data:text/javascript;base64,${Buffer.from(ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
}).outputText).toString("base64")}`;
const load = (path) => import(dataModule(readFileSync(new URL(path, import.meta.url), "utf8")));

const {
  effective, iprProblem, choicesForRequest, buildRunRequest, runBlockers, savePlan,
  headline, topMovers, statusView, rangeText, readCurve, corrGroupsFor, DEFAULT_HEADER_FORM,
  vogelRate, vogelCurve, wellImpact, reviewState,
} = await load("../src/pages/header/model.ts");

const row = (patch = {}) => ({
  well: "MPF-01", pad: "F", lift: "ESP", reservoir: "kuparuk", pump: null, test_date: "2026-09-15",
  test_age_days: 14, age_ok: true, looks_down: false, online_default: true, down_note: null,
  oil: 76, liquid: 1051, wc: 0.93, gor: 300, whp_test: 406, bhp_test: 510, whp_now: 412, bhp_now: 515,
  header_now: 412, has_gauge: true, gauge_auto_bad: false, gauge_note: null, resvr_press: null,
  pres_well: 1800, pres_basis: "default",
  measured: { slope: 0.62, q25: 0.5, q75: 0.7, r2: 0.65, n_fit: 29, n_days: 60, status: "measured" },
  whp_hdr: { slope: 1.0, r2: 0.9, n_fit: 50, n_days: 60, status: "measured" },
  ipr_fit_stats: { pres: 1365, qmax: 2000, n: 8, spread: 150, rmse: 40, usable: true, why_not: null },
  ipr_fit: { qwf: 1051, pwf: 510, pres: 1365 },
  saved: null, saved_ipr: null,
  corr_options: {
    "ESP kuparuk": { slope: 0.8, lo: 0.62, hi: 0.98, same_lift: true },
    ESP: { slope: 0.6, lo: 0.42, hi: 0.78, same_lift: true },
  },
  corr_group: "ESP kuparuk",
  ipr_options: {
    "F kuparuk": {
      gauge: { qwf: 1051, pwf: 515, pres: 1600, pres_lo: 1400, pres_hi: 2700 },
      nogauge: { qwf: 1051, pwf: 620, pres: 1650, pres_lo: 1450, pres_hi: 2800 },
      same_reservoir: true,
    },
    schrader: { gauge: null, nogauge: { qwf: 1051, pwf: 700, pres: 1800, pres_lo: 1440, pres_hi: 2160 }, same_reservoir: false },
  },
  ipr_group: "F kuparuk",
  ...patch,
});

const board = (rows) => ({
  pads: ["F", "L"], fit_days: 120, built_at: "2026-09-29T08:00:00", rows, correlations: {}, ipr_groups: {},
  header_now: { F: 412, L: 404 }, defaults: { res_pres: {}, online_max_test_age_days: 45 },
});

test("defaults follow the ladder and JP runs the pump model", () => {
  const e = effective(row());
  assert.equal(e.rel.kind, "measured");
  assert.deepEqual([e.rel.lo, e.rel.hi], [0.5, 0.7]);
  // Default IPR: the well's own gauge data (usable fit), else its own ResP.
  assert.equal(e.ipr.kind, "fit");
  assert.equal(effective(row({ ipr_fit: null })).ipr.kind, "correlation");
  assert.equal(effective(row({ lift: "JP" })).rel.kind, "physics");
  const saved = row({ saved: { slope: 0.4, whp_hdr: 1, source: "correlation", r2: 0, days: 0, at: null, by: null } });
  assert.equal(effective(saved).rel.kind, "saved");
  assert.equal(effective(saved).rel.firm, true);   // any saved relation: reviewed, so firm
  // IPR on the default ResP is not firm; on the well's own saved ResP it is.
  assert.equal(effective(row({ ipr_fit: null })).ipr.firm, false);
  assert.equal(effective(row({ ipr_fit: null, pres_basis: "saved" })).ipr.firm, true);
  assert.equal(effective(row()).ipr.firm, true); // usable gauge fit = own data
  const flagged = row({ ipr_fit: null, ipr_fit_any: { qwf: 1051, pwf: 510, pres: 4200 } });
  assert.equal(effective(flagged, { well: "MPF-01", ipr: "fit" }).ipr.firm, false);
});

test("a bad gauge (auto or marked) switches the well to its correlations", () => {
  const marked = effective(row(), { well: "MPF-01", gauge_bad: true });
  assert.equal(marked.gaugeOk, false);
  assert.equal(marked.rel.kind, "correlation");
  assert.equal(marked.rel.group, "ESP kuparuk");
  assert.equal(marked.ipr.kind, "correlation");
  assert.equal(marked.ipr.ipr.pwf, 620); // gaugeless variant: BHP from the group ratio
  assert.equal(effective(row(), { well: "MPF-01", gauge_bad: true, relation: "measured" }).rel.kind, "none");
  const auto = row({ gauge_auto_bad: true, gauge_note: "flat" });
  assert.equal(effective(auto).gaugeOk, false);
  assert.equal(effective(auto, { well: "MPF-01", gauge_bad: false }).gaugeOk, true);
  // A shut-in-looking gauge stops mattering once the gauge is marked bad.
  const down = row({ looks_down: true });
  assert.equal(effective(down).online, false);
  assert.equal(effective(down, { well: "MPF-01", gauge_bad: true }).online, true);
});

test("assigning groups: relation and reservoir IPR", () => {
  const pick = { well: "MPF-01", relation: "correlation", corr_group: "ESP", ipr: "correlation", ipr_group: "schrader" };
  // With a working gauge the schrader group has no variant for this well: no IPR, said plainly.
  assert.equal(effective(row(), pick).ipr.kind, "correlation");
  assert.equal(effective(row(), pick).ipr.ipr, null);
  const e = effective(row(), { ...pick, gauge_bad: true });
  assert.equal(e.rel.slope, 0.6);
  assert.equal(e.ipr.group, "schrader");
  assert.deepEqual([e.ipr.presLo, e.ipr.presHi], [1440, 2160]);
  assert.deepEqual(corrGroupsFor(row()), ["ESP kuparuk", "ESP"]);   // most specific first
  // Same slope in both groups: only the specific one is offered.
  const same = row({ corr_options: { ESP: { slope: 0.23, lo: 0.2, hi: 0.3, same_lift: true },
    "ESP schrader": { slope: 0.23, lo: 0.2, hi: 0.3, same_lift: true }, "JP": { slope: 0.5, lo: 0.4, hi: 0.6, same_lift: false } } });
  assert.deepEqual(corrGroupsFor(same), ["ESP schrader"]);
});

test("IPR problems mirror the server's 300 psi drawdown rule", () => {
  assert.equal(iprProblem({ qwf: 1000, pwf: 600, pres: 1800 }), null);
  assert.match(iprProblem({ qwf: 1000, pwf: 1600, pres: 1800 }), /300 psi/);
  assert.match(iprProblem({ qwf: null, pwf: 600, pres: 1800 }), /incomplete/);
});

test("only choices that differ from the default go into the request, and Estimate needs no board", () => {
  const b = board([row(), row({ well: "MPF-14", online_default: false }), row({ well: "MPF-107", lift: "JP" })]);
  const choices = {
    "MPF-01": { well: "MPF-01", online: true, relation: "auto", ipr: "auto", corr_group: "ESP kuparuk" },
    "MPF-14": { well: "MPF-14", online: true, gauge_bad: true },
    "MPF-107": { well: "MPF-107", relation: "manual", slope: 0.5 },
  };
  assert.deepEqual(choicesForRequest(b, choices), [{ well: "MPF-14", online: true, gauge_bad: true }]);
  const req = buildRunRequest({ ...DEFAULT_HEADER_FORM, pads: ["L", "F"], deltas: { F: 12 } }, b, choices);
  assert.deepEqual(req.pads, ["F", "L"]);
  assert.deepEqual(req.delta_by_pad, { F: 12, L: 10 });
  const noBoard = buildRunRequest({ ...DEFAULT_HEADER_FORM, pads: ["R"] }, null, { "MPR-142": { well: "MPR-142", ipr: "correlation", ipr_group: "schrader" } });
  assert.deepEqual(noBoard.wells, [{ well: "MPR-142", ipr: "correlation", ipr_group: "schrader" }]);
  const ev = buildRunRequest({ ...DEFAULT_HEADER_FORM, pads: ["L"], mode: "event", eventTime: "2026-09-28T12:00", eventWell: "MPL-20" }, null, {});
  assert.equal(ev.event_time, "2026-09-28T12:00");
  assert.equal(ev.pre_hours, 72);
});

test("blockers name the fix", () => {
  assert.match(runBlockers(DEFAULT_HEADER_FORM, null, {})[0], /Pick one or more pads/);
  assert.deepEqual(runBlockers({ ...DEFAULT_HEADER_FORM, pads: ["F"] }, null, {}), []);
  assert.match(runBlockers({ ...DEFAULT_HEADER_FORM, pads: ["F"], deltas: { F: 0 } }, null, {})[0], /0 psi/);
  assert.match(runBlockers({ ...DEFAULT_HEADER_FORM, pads: ["F"], mode: "event" }, null, {})[0], /event time/);
  const bad = runBlockers({ ...DEFAULT_HEADER_FORM, pads: ["F"] }, board([row()]), { "MPF-01": { well: "MPF-01", ipr: "manual", qwf: 900, pwf: 1700, pres: 1800 } });
  assert.match(bad[0], /MPF-01: manual IPR/);
});

test("save plan carries groups and gauge state; never a JP or weak relation", () => {
  assert.deepEqual(savePlan(row()).entry, { well: "MPF-01", relation: "measured", ipr: "fit" });
  assert.deepEqual(savePlan(row({ ipr_fit: null })).entry, { well: "MPF-01", relation: "measured", ipr: "correlation", ipr_group: "F kuparuk" });
  assert.equal(savePlan(row({ lift: "JP" })).entry, null);
  const dead = savePlan(row(), { well: "MPF-01", gauge_bad: true }).entry;
  assert.deepEqual(dead, { well: "MPF-01", relation: "correlation", corr_group: "ESP kuparuk", ipr: "correlation", ipr_group: "F kuparuk", gauge_bad: true });
  const weak = row({ measured: { ...row().measured, status: "weak" }, saved_ipr: { qwf: 1, pwf: 1, pres: 400, source: "fit", at: null, by: null } });
  const plan = savePlan(weak, { well: "MPF-01", relation: "measured", ipr: "saved" });
  assert.equal(plan.entry, null);
  assert.match(plan.why, /weak/);
});

test("headline, range, curve read-off and movers", () => {
  const result = {
    pads: [{ pad: "F", d_header: 15 }, { pad: "L", d_header: 15 }],
    totals: { d_oil: -65.04, d_liq: -254.4, modeled: 44, d_oil_lo: -35.5, d_oil_hi: -92.9 },
    rows: [
      { well: "A", outcome: "modeled", d_oil: -7.2 }, { well: "B", outcome: "modeled", d_oil: -0.02 },
      { well: "C", outcome: "modeled", d_oil: -2.6 }, { well: "D", outcome: "offline", d_oil: -50 },
    ],
  };
  assert.equal(headline(result), "+15.0 psi header on F, L: -65.0 BOPD (-254 BLPD) across 44 wells");
  assert.equal(rangeText(-35.5, -92.9), "-35.5 to -92.9 BOPD");
  assert.equal(rangeText(-3, -3), null);
  assert.deepEqual(topMovers(result.rows).map((r) => r.well), ["A", "C"]);
  const grid = [-10, 0, 10, 20];
  assert.equal(readCurve(grid, [43, 0, -43, -87], 15), -65);
  assert.equal(readCurve(grid, [43, 0, -43, -87], 25), null);
  assert.equal(statusView("conditional").tone, "fair");
});

test("card impact matches the server chain: closed-loop slope once, Vogel at current BHP, WC once", () => {
  // Same case as the Python test: q 1000 at 600 psi, ResP 1800, slope 0.5, WC 0.8, +10 psi.
  const r = row({
    liquid: 1000, wc: 0.8, bhp_now: 600,
    ipr_fit: { qwf: 1000, pwf: 600, pres: 1800 },
    measured: { slope: 0.5, q25: 0.5, q75: 0.5, r2: 0.8, n_fit: 30, n_days: 40, status: "measured" },
  });
  const imp = wellImpact(r, { well: "MPF-01", ipr: "fit" }, 10);
  const expected = (vogelRate({ qwf: 1000, pwf: 600, pres: 1800 }, 605) - 1000) * 0.2;
  assert.ok(Math.abs(imp.dOil - expected) < 1e-9);
  assert.equal(imp.dBhp, 5);
  assert.equal(wellImpact(row({ lift: "JP" }), undefined, 10), null);
  // A flagged fit is offered only when chosen explicitly.
  const flagged = row({ ipr_fit: null, ipr_fit_any: { qwf: 1051, pwf: 510, pres: 1400 } });
  assert.equal(effective(flagged).ipr.kind, "correlation");
  assert.equal(effective(flagged, { well: "MPF-01", ipr: "fit" }).ipr.ipr.pres, 1400);
  const curve = vogelCurve({ qwf: 1000, pwf: 600, pres: 1800 });
  assert.deepEqual(curve[0], [0, 1800]);
  assert.equal(curve[curve.length - 1][1], 0);
});

test("review status: not saved, saved, and drift when the data has moved on", () => {
  assert.equal(reviewState(row()).status, "not_saved");
  const saved = { slope: 0.6, whp_hdr: 1, source: "measured", r2: 0.8, days: 30, at: "2026-09-29", by: "x" };
  assert.equal(reviewState(row({ saved })).status, "saved");
  const moved = reviewState(row({ saved, measured: { ...row().measured, slope: 0.3, q25: 0.28, q75: 0.33 } }));
  assert.equal(moved.status, "drift");
  assert.match(moved.notes[0], /now measures 0.30/);
  const ipr = { qwf: 700, pwf: 510, pres: 1400, source: "fit", at: "2026-09-29", by: "x" };
  assert.equal(reviewState(row({ saved_ipr: ipr })).status, "drift"); // 1051 vs 700 BLPD: +50%
  assert.equal(reviewState(row({ saved_ipr: { ...ipr, qwf: 1000 } })).status, "saved");
  // A bad gauge cannot say the saved slope drifted.
  assert.equal(reviewState(row({ saved, gauge_auto_bad: true, measured: { ...row().measured, slope: 0.1 } })).status, "saved");
});

test("a saved gauge verdict is the default; changing it is a (gauge-only) save", () => {
  const savedBad = row({ gauge_saved: { bad: true, at: "2026-09-29", by: "x" }, gauge_bad_default: true });
  assert.equal(effective(savedBad).gaugeOk, false);
  assert.equal(effective(savedBad, { well: "MPF-01", gauge_bad: false }).gaugeOk, true);
  // Ticking bad on a good gauge: saving writes the flag (with the correlations it implies).
  const plan = savePlan(row(), { well: "MPF-01", gauge_bad: true }).entry;
  assert.equal(plan.gauge_bad, true);
  // Unticking a saved bad flag saves "good".
  assert.equal(savePlan(savedBad, { well: "MPF-01", gauge_bad: false }).entry.gauge_bad, false);
  // Jet pumps can save the gauge verdict alone.
  assert.deepEqual(savePlan(row({ lift: "JP" }), { well: "MPF-01", gauge_bad: true }).entry, { well: "MPF-01", gauge_bad: true });
  // Matching the saved verdict is not a change.
  assert.equal(savePlan(savedBad, { well: "MPF-01", gauge_bad: true }).entry?.gauge_bad, undefined);
});
