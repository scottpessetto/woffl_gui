import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "typescript";

const source = readFileSync(new URL("../src/components/shutInSeries.ts", import.meta.url), "utf8");
const js = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText;
const { breakAtShutIns, rateSeriesWithShutIns, shutInAt, shutInSpans, shutInWindowsOf } =
  await import(`data:text/javascript;base64,${Buffer.from(js).toString("base64")}`);

const DAY = 86_400_000;
const t = (d) => Date.parse(`${d}T00:00:00Z`);
const rate = (d, oil, water) => ({ x: t(d), oil, water });

test("a long shut-in steps the rates to zero instead of ramping between tests", () => {
  // MPL-06 shape: last test 2023-12-18, casing-leak shut-in 2024-01-10..2026-06-20, restart test 2026-06-24.
  const tests = [rate("2023-12-18", 246, 443), rate("2026-06-24", 1158, 1076)];
  const spans = shutInSpans([{ start: "2024-01-10", end: "2026-06-20", days: 893 }], tests.map((p) => p.x));
  assert.deepEqual(spans.map((s) => [s.start, s.end]), [[t("2024-01-10"), t("2026-06-21")]]);
  const { oil, water } = rateSeriesWithShutIns(tests, spans);
  assert.equal(oil.length, water.length, "one shared x sequence keeps the stack aligned");
  assert.deepEqual(oil.map((p) => p[0]), water.map((p) => p[0]));
  assert.deepEqual(oil.map((p) => p[0]), [t("2023-12-18"), t("2024-01-10"), t("2024-01-10"),
    t("2026-06-21"), t("2026-06-21"), t("2026-06-24")]);
  assert.deepEqual(oil.slice(2, 4).map((p) => p[1]), [0, 0]);
  assert.deepEqual(water.slice(2, 4).map((p) => p[1]), [0, 0]);
  // Edges sit on the old test-to-test line, so producing periods are unchanged.
  const f = (t("2024-01-10") - t("2023-12-18")) / (t("2026-06-24") - t("2023-12-18"));
  assert.ok(Math.abs(oil[1][1] - (246 + (1158 - 246) * f)) < 1e-9);
  assert.equal(oil[5][1], 1158);
});

test("no shut-in data renders exactly the tests", () => {
  const tests = [rate("2026-01-01", 100, 50), rate("2026-02-01", 120, 60)];
  const { oil, water } = rateSeriesWithShutIns(tests, shutInSpans([]));
  assert.deepEqual(oil, [[t("2026-01-01"), 100], [t("2026-02-01"), 120]]);
  assert.deepEqual(water, [[t("2026-01-01"), 50], [t("2026-02-01"), 60]]);
  assert.deepEqual(shutInWindowsOf({ well: "x" }), [], "an older payload without shut_in");
  assert.deepEqual(shutInWindowsOf(null), []);
});

test("shut-ins before the first or after the last test never invent the far side", () => {
  const tests = [rate("2026-03-01", 100, 10)];
  const spans = shutInSpans([{ start: "2026-01-01", end: "2026-01-31" }, { start: "2026-04-01", end: "2026-09-22" }]);
  const { oil } = rateSeriesWithShutIns(tests, spans);
  assert.deepEqual(oil, [
    [t("2026-01-01"), 0], [t("2026-02-01"), 0], [t("2026-02-01"), 100],
    [t("2026-03-01"), 100],
    [t("2026-04-01"), 100], [t("2026-04-01"), 0], [t("2026-09-23"), 0],
  ]);
});

test("a well test on a logged shut-in day wins and splits the zero segment", () => {
  const tests = [rate("2026-01-01", 100, 0), { x: t("2026-01-10") + 14 * 3600_000, oil: 80, water: 0 }, rate("2026-02-01", 90, 0)];
  const spans = shutInSpans([{ start: "2026-01-05", end: "2026-01-20" }], tests.map((p) => p.x));
  assert.deepEqual(spans.map((s) => [s.start, s.end]), [[t("2026-01-05"), t("2026-01-10")], [t("2026-01-11"), t("2026-01-21")]]);
  const { oil } = rateSeriesWithShutIns(tests, spans);
  const testIdx = oil.findIndex((p) => p[1] === 80);
  assert.ok(testIdx > 0 && oil[testIdx - 1][1] > 0 && oil[testIdx + 1][1] > 0, "the test is plotted between two steps");
  assert.equal(shutInAt(spans, tests[1].x), undefined, "hovering the test day shows the test");
  assert.ok(shutInAt(spans, t("2026-01-15")));
});

test("pressure lines break at shut-in edges and keep real in-shut-in samples", () => {
  const spans = shutInSpans([{ start: "2026-01-05", end: "2026-01-06" }]);
  const pts = [[t("2026-01-04"), 900], [t("2026-01-05"), 1500], [t("2026-01-07"), 950], [t("2026-01-08"), 940]];
  assert.deepEqual(breakAtShutIns(pts, spans), [
    [t("2026-01-04"), 900], [t("2026-01-05"), null], [t("2026-01-05"), 1500],
    [t("2026-01-07"), null], [t("2026-01-07"), 950], [t("2026-01-08"), 940],
  ]);
  // Sparse test-date BHP across a long shut-in gets a single null gap per edge.
  const sparse = [[t("2023-12-18"), 800], [t("2026-06-24"), 1900]];
  const long = shutInSpans([{ start: "2024-01-10", end: "2026-06-20" }]);
  assert.deepEqual(breakAtShutIns(sparse, long).map((p) => p[1]), [800, null, null, 1900]);
  assert.equal(shutInAt(long, t("2026-06-20") + DAY - 1).window.start, "2024-01-10");
  assert.equal(shutInAt(long, t("2026-06-21")), undefined, "the restart day is producing");
});
