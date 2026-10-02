import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "typescript";

const dataModule = (source) => `data:text/javascript;base64,${Buffer.from(ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
}).outputText).toString("base64")}`;
const load = (path) => import(dataModule(readFileSync(new URL(path, import.meta.url), "utf8")));

const { gaugeBhpNear, lrsManualTest } = await load("../src/pages/solver/lrsTest.ts");

const BOUNDS = {
  qwf: [10, 20000], pwf: [100, 2500], form_wc: [0, 1], form_gor: [20, 10000],
  surf_pres: [10, 600], ppf_surf: [800, 5500],
};
const sheet = (patch = {}) => ({
  filename: "E-48.xlsx", well: "MPE-48", well_raw: "MPE-48", test_date: "2026-10-02", hours: 12,
  location: "LRS Unit 6", oil: 1039.2, water: 745.4, total_fluid: 1784.6, form_wc: 0.418, gor: 1.8,
  whp: 363, pf_rate: 4732, pf_press: 3353.8, missing: [], ...patch,
});

test("the sheet becomes a test with its own numbers, the gauge supplying the BHP", () => {
  const { test: t, notes } = lrsManualTest(sheet(), "2026-10-02", BOUNDS, { date: "2026-10-02", bhp: 597 });
  assert.deepEqual(t, {
    date: "2026-10-02", oil: 1039.2, water: 745.4, bhp: 597, gor: null, whp: 363,
    pfRate: 4732, pfPress: 3353.8, source: "LRS Unit 6 (E-48.xlsx)",
  });
  assert.ok(notes.some((n) => n.includes("daily gauge reading of 597 psi")));
});

test("a no-gas GOR below the model floor is dropped, a measured one is kept", () => {
  const none = lrsManualTest(sheet(), "2026-10-02", BOUNDS, null);
  assert.equal(none.test.gor, null);
  assert.ok(none.notes.some((n) => n.includes("GOR 1.8 scf/stb")));
  assert.equal(lrsManualTest(sheet({ gor: 310 }), "2026-10-02", BOUNDS, null).test.gor, 310);
});

test("with no gauge reading the test has no BHP and says it cannot anchor yet", () => {
  const { test: t, notes } = lrsManualTest(sheet(), "2026-10-02", BOUNDS, null);
  assert.equal(t.bhp, null);
  assert.ok(notes.some((n) => n.includes("cannot anchor the IPR without one")));
});

test("an out-of-range value is reported, a missing PF rate is asked for", () => {
  const { test: t, notes } = lrsManualTest(sheet({ whp: 720, pf_rate: null }), "2026-10-02", BOUNDS, null);
  assert.equal(t.whp, 720); // the measurement is kept as measured
  assert.ok(notes.some((n) => n.startsWith("Wellhead pressure 720")));
  assert.ok(notes.some((n) => n.includes("enter it below")));
});

test("the gauge BHP is the test day's, else the latest within three days before", () => {
  const daily = [
    { date: "2026-09-25", bhp: 570 }, { date: "2026-09-30", bhp: 581 },
    { date: "2026-10-01", bhp: 0 }, { date: "2026-10-03", bhp: 590 },
  ];
  assert.deepEqual(gaugeBhpNear(daily, "2026-10-02"), { date: "2026-09-30", bhp: 581 });
  assert.deepEqual(gaugeBhpNear(daily, "2026-09-30"), { date: "2026-09-30", bhp: 581 });
  assert.equal(gaugeBhpNear(daily, "2026-09-29"), null);
  assert.equal(gaugeBhpNear([], "2026-10-02"), null);
});
