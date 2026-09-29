import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "typescript";

const dataModule = (source) => `data:text/javascript;base64,${Buffer.from(ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
}).outputText).toString("base64")}`;
const load = (path) => import(dataModule(readFileSync(new URL(path, import.meta.url), "utf8")));

const {
  hasGaugeBhp, isGaugelessMatchable, earlierTestInPumpLife, nearestGaugeBhp, gaugeGapFlagged,
} = await load("../src/pages/solver/gaugelessMatch.ts");

const row = (date, patch = {}) => ({
  wt_uid: null, date, oil: 300, water: 900, gas: 100, total_fluid: 1200, form_wc: 0.75,
  bhp: null, fgor: 300, lift_wat: 3000, whp: 200, pf_press: 3200, pf_source: null, ...patch,
});

test("a gauge BHP disqualifies a test from the gaugeless match", () => {
  assert.equal(hasGaugeBhp(row("2026-09-01", { bhp: 900 })), true);
  assert.equal(hasGaugeBhp(row("2026-09-01", { bhp: 0 })), false);
  assert.equal(hasGaugeBhp(null), false);
  assert.equal(isGaugelessMatchable(row("2026-09-01")), true);
  assert.equal(isGaugelessMatchable(row("2026-09-01", { bhp: 900 })), false);
  assert.equal(isGaugelessMatchable(row("2026-09-01", { lift_wat: null })), false);
});

test("a late test points at the earliest matchable test on the same pump", () => {
  const tests = [
    row("2026-09-01"),
    row("2026-05-10", { bhp: 1100 }), // gauged: not matchable
    row("2026-03-15"),
    row("2026-02-20", { oil: null }), // no oil: not matchable
    row("2025-12-01"), // previous pump
  ];
  const pick = earlierTestInPumpLife(tests, tests[0], "2026-02-01T00:00:00");
  assert.equal(pick?.date, "2026-03-15");
});

test("an early test or an unknown set date gets no suggestion", () => {
  const tests = [row("2026-03-15"), row("2026-02-10")];
  assert.equal(earlierTestInPumpLife(tests, tests[0], "2026-02-01"), null); // 42 days in
  assert.equal(earlierTestInPumpLife(tests, tests[0], null), null);
});

test("the gauge cross-check stays on the current pump", () => {
  const tests = [
    row("2026-09-01"),
    row("2026-08-01", { bhp: 1200 }),
    row("2026-08-25", { bhp: 1100 }),
    row("2026-01-15", { bhp: 700 }), // previous pump, nearer than nothing but excluded
  ];
  assert.deepEqual(nearestGaugeBhp(tests, tests[0], "2026-02-01"), { bhp: 1100, date: "2026-08-25" });
  assert.equal(nearestGaugeBhp([row("2026-09-01"), row("2026-01-15", { bhp: 700 })], row("2026-09-01"), "2026-02-01"), null);
  // No set date: only readings within 90 days count.
  assert.equal(nearestGaugeBhp([row("2026-09-01"), row("2026-01-15", { bhp: 700 })], row("2026-09-01"), null), null);
});

test("the gap flag respects the match's own BHP resolution", () => {
  assert.equal(gaugeGapFlagged(900, 1100, null), true);
  assert.equal(gaugeGapFlagged(1000, 1100, null), false);
  assert.equal(gaugeGapFlagged(900, 1100, 300), false);
});
