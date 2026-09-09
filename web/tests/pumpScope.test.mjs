import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import test from "node:test";
import ts from "typescript";

const require = createRequire(import.meta.url);
const dataModule = (source) => `data:text/javascript;base64,${Buffer.from(ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
}).outputText).toString("base64")}`;
const types = dataModule(readFileSync(new URL("../src/api/types.ts", import.meta.url), "utf8"));
const source = readFileSync(new URL("../src/state/params.ts", import.meta.url), "utf8")
  .replaceAll('"../api/types"', JSON.stringify(types))
  .replace('"zustand"', JSON.stringify(pathToFileURL(require.resolve("zustand")).href));
const { useParamsStore: store, CLEAN_PUMP, effectiveParams } = await import(dataModule(source));
const fit = { ken: .005, kth: .386, kdi: .072, nozzle_area_factor: 1.01 };
const context = {
  well: "MPE-42", seeds: { ...fit, nozzle_no: "13", area_ratio: "C", form_wc: .74, qwf: 1521 },
  pump: { nozzle_no: "13", throat_ratio: "C", date_set: "2026-08-10", source: "databricks" },
  pump_calibration: { status: "active", coefficients: fit }, as_built_locks: {}, prop_locks: {},
};
const reset = () => { store.getState().selectWell(context.well); store.getState().applyContext(context); };

test("same-size clean replacement preserves well inputs and restores installed fit", () => {
  reset();
  store.getState().set("pump_state", "replacement");
  const p = effectiveParams(store.getState().params);
  assert.equal(p.nozzle_no, "13");
  for (const key of Object.keys(CLEAN_PUMP)) assert.equal(p[key], CLEAN_PUMP[key]);
  assert.equal(p.form_wc, .74);
  assert.equal(p.qwf, 1521);
  store.getState().useInstalledPump();
  for (const key of Object.keys(fit)) assert.equal(store.getState().params[key], fit[key]);
});

test("size changes reset every pump coefficient through either edit path", () => {
  for (const edit of [() => store.getState().set("nozzle_no", "14"),
    () => store.getState().setMany({ area_ratio: "B", ...fit })]) {
    reset(); edit();
    const p = store.getState().params;
    assert.equal(p.pump_state, "replacement");
    for (const key of Object.keys(CLEAN_PUMP)) assert.equal(p[key], CLEAN_PUMP[key]);
  }
});

test("an old or foreign fit cannot apply to the installed pump", () => {
  reset();
  for (const result of [{ well: "MPB-28", pump: "13C", era_start: "2026-08-10" },
    { well: "MPE-42", pump: "13C", era_start: "2026-09-01" }]) {
    store.getState().applyPumpFit(result, { ken: .2 });
    assert.equal(store.getState().params.ken, .005);
  }
});

test("a same-size installation change clears the old fit without erasing WC edits", () => {
  reset();
  store.getState().set("form_wc", .70);
  store.getState().refreshPumpContext({ ...context, pump: { ...context.pump, date_set: "2026-09-08" },
    pump_calibration: { status: "stale", coefficients: {} } });
  const p = store.getState().params;
  assert.equal(p.ken, CLEAN_PUMP.ken);
  assert.equal(p.nozzle_area_factor, 1);
  assert.equal(p.form_wc, .70);
});

test("saving updates the fit available to restore without overwriting session edits", () => {
  reset();
  const updated = { ...fit, kth: .35 };
  store.getState().set("form_wc", .70);
  store.getState().refreshPumpContext({ ...context, pump_calibration: { status: "active", coefficients: updated } });
  store.getState().set("pump_state", "replacement");
  store.getState().useInstalledPump();
  assert.equal(store.getState().params.kth, .35);
  assert.equal(store.getState().params.form_wc, .70);
});


test("a no-longer-valid saved fit clears on refresh while explicit session fitting survives", () => {
  reset();
  store.getState().set("form_wc", .70);
  store.getState().refreshPumpContext({ ...context, pump_calibration: { status: "stale", coefficients: {} } });
  assert.equal(store.getState().params.ken, CLEAN_PUMP.ken);
  assert.equal(store.getState().params.form_wc, .70);
  reset();
  store.getState().set("ken", .08);
  store.getState().refreshPumpContext({ ...context, pump_calibration: { status: "active", coefficients: { ...fit, kth: .35 } } });
  assert.equal(store.getState().params.ken, .08);
});
