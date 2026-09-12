import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import test from "node:test";
import ts from "typescript";

const require = createRequire(import.meta.url);
const moduleUrl = (source) => `data:text/javascript;base64,${Buffer.from(ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
}).outputText).toString("base64")}`;
const client = moduleUrl(readFileSync(new URL("../src/api/client.ts", import.meta.url), "utf8"));
const study = readFileSync(new URL("../src/pages/sensitivity/study.ts", import.meta.url), "utf8")
  .replace('"../../api/client"', JSON.stringify(client));
const { matchingStudy, appliedStudyParams } = await import(moduleUrl(study));
const stateSource = readFileSync(new URL("../src/state/sensitivity.ts", import.meta.url), "utf8")
  .replace('"zustand"', JSON.stringify(pathToFileURL(require.resolve("zustand")).href));
const { useSensitivityStore: store } = await import(moduleUrl(stateSource));

const request = () => ({ well: "MPE-42", params: { qwf: 750.25, form_wc: .8, form_gor: 420,
  nozzle_no: "12", area_ratio: "C", pump_state: "installed", ken: .2 },
  target_psu: 900, target_qoil: 200, target_qliq: 1000, target_qpf: 3000,
  test_key: "uid:123", installation_key: "2026-07-01T12:00:00Z", wc_basis: "fixed_oil_ipr",
  knobs: [{ id: "form_wc", low: .75, high: .85, levels: 3 }] });

test("study identity rejects changed inputs, targets, comparison identity, hardware era and WC basis", () => {
  const saved = request();
  assert.equal(matchingStudy(structuredClone(saved), saved), true);
  for (const update of [
    { params: { ...saved.params, qwf: 751 } }, { target_qoil: 220 },
    { test_key: "uid:124" }, { installation_key: "2026-08-01T12:00:00Z" },
    { wc_basis: "anchor_measurement" }, { well: "MPE-43" },
  ]) assert.equal(matchingStudy({ ...saved, ...update }, saved), false);
  assert.equal(matchingStudy(saved, null), false);
  // Changing the next grid does not rewrite the immutable submitted case.
  assert.equal(matchingStudy({ ...saved, knobs: [] }, saved), true);
});

test("Apply reconstructs automatic oil-anchor and clean hardware changes from submitted inputs", () => {
  const saved = request();
  const row = { error: null, applied_inputs: { form_wc: .85, qwf: 1000.333333333333,
    nozzle_no: "13", pump_state: "replacement", ken: .03, kth: .3, kdi: .4, nozzle_area_factor: 1 } };
  const applied = appliedStudyParams(saved, row);
  assert.equal(applied.qwf, row.applied_inputs.qwf);
  assert.equal(applied.form_gor, 420);
  assert.equal(applied.ken, .03);
  assert.equal(saved.params.qwf, 750.25);
  assert.equal(appliedStudyParams(saved, { error: null, applied_inputs: null }), null);
});

test("comparison handoff belongs to its well and is consumed once without an anchor pin", () => {
  store.getState().queueComparison("MPE-42", "uid:123");
  assert.equal(store.getState().takeComparison("MPE-43"), null);
  assert.equal(store.getState().takeComparison("MPE-42"), "uid:123");
  assert.equal(store.getState().takeComparison("MPE-42"), null);
});
