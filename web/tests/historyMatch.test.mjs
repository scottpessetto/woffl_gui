import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "typescript";

const source = readFileSync(new URL("../src/components/historyMatchSeries.ts", import.meta.url), "utf8");
const js = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText;
const { historyTimestamp, matchLines, matchAt } = await import(`data:text/javascript;base64,${Buffer.from(js).toString("base64")}`);
const theme = readFileSync(new URL("../src/charts/theme.ts", import.meta.url), "utf8");
const themeJs = ts.transpileModule(theme, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText;
const { ttNote } = await import(`data:text/javascript;base64,${Buffer.from(themeJs).toString("base64")}`);
const row = (date, status, installation_id = "a") => ({ date, status, installation_id, wt_uid: `${installation_id}-${date}`,
  predicted_oil: status === "failed" ? null : 100 });

test("model curves break at failed solves, fitting cutoffs, and installations", () => {
  const rows = [row("2026-01-01", "fit"), row("2026-01-05", "prediction"), row("2026-01-10", "failed"),
    row("2026-01-15", "prediction"), row("2026-02-01", "prediction", "b")];
  const series = matchLines({ eras: [{ installation_id: "a" }, { installation_id: "b" }], rows }, "oil", "green", 0, 0);
  assert.equal(series.length, 3);
  assert.deepEqual(series[1].data.map((p) => p.value[1]), [null, 100, null, 100]);
  assert.equal(series[2].data.length, 1);
  assert.ok(series.every((s) => s.connectNulls === false));
  assert.equal(series[0].lineStyle.type, "dotted");
  assert.equal(series[1].lineStyle.type, "dashed");
});

test("long periods without observations stay gaps and era selection is scoped", () => {
  const result = { eras: [{ installation_id: "a" }, { installation_id: "b" }],
    rows: [row("2026-01-01", "prediction"), row("2026-04-01", "prediction"), row("2026-02-01", "prediction", "b")] };
  const series = matchLines(result, "oil", "green", 0, 0, "a");
  assert.equal(series.length, 1);
  assert.deepEqual(series[0].data.map((p) => p.value[1]), [100, null, 100]);
});

test("every-test model curves stay separate from held-out prediction curves", () => {
  const result = { eras: [{ installation_id: "a" }, { installation_id: "b" }],
    rows: [row("2026-01-01", "replay"), row("2026-01-05", "replay"), row("2026-02-01", "replay", "b")] };
  const series = matchLines(result, "oil", "green", 0, 0);
  assert.deepEqual(series.map((s) => s.name), ["Oil model", "Oil model"]);
  assert.deepEqual(series.flatMap((s) => s.data.map((p) => p.value[1])), [100, 100, 100]);
});

test("hover never borrows a prediction from another day or across a failed solve", () => {
  const rows = [row("2026-01-01", "prediction"), row("2026-01-05", "failed")];
  assert.equal(matchAt(rows, Date.parse("2026-01-03")), undefined);
  assert.equal(matchAt(rows, Date.parse("2026-01-05")).status, "failed");
});

test("exact tracker times keep UTC identity in every browser time zone", () => {
  assert.equal(historyTimestamp("2026-01-01 06:00:00"), Date.parse("2026-01-01T06:00:00Z"));
  assert.equal(historyTimestamp("2026-01-01T06:00:00-09:00"), Date.parse("2026-01-01T15:00:00Z"));
  assert.equal(historyTimestamp("2026-01-01"), Date.parse("2026-01-01T00:00:00Z"));
});

test("model absence reasons cannot inject source or solver HTML into tooltips", () => {
  const html = ttNote('No solve: <img src=x onerror="alert(1)"> & invalid input');
  assert.ok(!html.includes("<img"));
  assert.ok(html.includes("&lt;img src=x onerror=&quot;alert(1)&quot;&gt; &amp; invalid input"));
});
