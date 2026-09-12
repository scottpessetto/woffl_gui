import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "typescript";

const moduleUrl = (source) => `data:text/javascript;base64,${Buffer.from(ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
}).outputText).toString("base64")}`;
const typesUrl = moduleUrl(readFileSync(new URL("../src/api/types.ts", import.meta.url), "utf8"));
const { DEFAULT_PARAMS } = await import(typesUrl);
const modelUrl = moduleUrl(readFileSync(new URL("../src/lib/wellModel.ts", import.meta.url), "utf8")
  .replaceAll('"../api/types"', JSON.stringify(typesUrl)));
const { calibrationInputBlocker, hasWellModelEdits, sessionOnlyWellModelEdits, savedSessionOnlyModelInputs } = await import(modelUrl);
const saved = { ...DEFAULT_PARAMS, oil_api: 24, form_wc: .7, form_gor: 700, qwf: 1000, pwf: 600, pres: 1700 };
const context = { well: "MPM-01", seeds: saved };

test("unsupported model edits have an actionable reason instead of a futile Save instruction", () => {
  const p = { ...saved, oil_api: 29, gas_sg: .8, rho_pf: 65, jpump_tvd: saved.jpump_tvd + 10 };
  assert.equal(hasWellModelEdits(p, context), true);
  assert.deepEqual(sessionOnlyWellModelEdits(p, context).map(e => e.key).sort(), ["gas_sg", "jpump_tvd", "oil_api", "rho_pf"]);
  assert.match(calibrationInputBlocker(p, context), /cannot persist these settings/);
  assert.match(calibrationInputBlocker(p, context), /Restore session-only settings/);
});

test("restoring unsupported settings preserves savable IPR edits and pump/model choices", () => {
  const p = { ...saved, oil_api: 29, tubing_od: 5, qwf: 1250, form_gor: 999,
    hydraulics_model: "hagedorn_brown", ken: .18, nozzle_area_factor: 1.1 };
  const restore = savedSessionOnlyModelInputs(p, context);
  assert.deepEqual(restore, { tubing_od: saved.tubing_od, oil_api: saved.oil_api });
  const restored = { ...p, ...restore };
  assert.equal(restored.qwf, 1250);
  assert.equal(restored.form_gor, 999);
  assert.equal(restored.hydraulics_model, "hagedorn_brown");
  assert.equal(restored.ken, .18);
  assert.equal(restored.nozzle_area_factor, 1.1);
  assert.deepEqual(sessionOnlyWellModelEdits(restored, context), []);
  assert.match(calibrationInputBlocker(restored, context), /Save the intended IPR/);
});

test("restoring only unsupported edits re-enables the same saved oil model", () => {
  const p = { ...saved, wat_sg: 1.2, jpump_direction: "forward", model_as_water: true };
  const restored = { ...p, ...savedSessionOnlyModelInputs(p, context) };
  assert.equal(hasWellModelEdits(restored, context), false);
  assert.equal(calibrationInputBlocker(restored, context), null);
});

test("composition changes preserving the oil IPR need no model refit or unsupported reset", () => {
  const wc = .8;
  const p = { ...saved, form_wc: wc, form_gor: 900, qwf: saved.qwf * (1 - saved.form_wc) / (1 - wc) };
  assert.equal(hasWellModelEdits(p, context), false);
  assert.equal(calibrationInputBlocker(p, context), null);
  assert.deepEqual(savedSessionOnlyModelInputs(p, context), {});
  assert.match(calibrationInputBlocker(p, null), /Wait for the saved well model/);
});
