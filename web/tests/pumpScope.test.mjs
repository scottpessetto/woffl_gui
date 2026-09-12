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
const wellModel = dataModule(readFileSync(new URL("../src/lib/wellModel.ts", import.meta.url), "utf8")
  .replaceAll('"../api/types"', JSON.stringify(types)));
const source = readFileSync(new URL("../src/state/params.ts", import.meta.url), "utf8")
  .replaceAll('"../api/types"', JSON.stringify(types))
  .replace('"../lib/wellModel"', JSON.stringify(wellModel))
  .replace('"zustand"', JSON.stringify(pathToFileURL(require.resolve("zustand")).href));
const { useParamsStore: store, CLEAN_PUMP, effectiveParams } = await import(dataModule(source));
const fit = { ken: .005, kth: .386, kdi: .072, nozzle_area_factor: 1.01 };
const context = {
  well: "MPE-42", seeds: { ...fit, nozzle_no: "13", area_ratio: "C", form_wc: .74, qwf: 1521,
    pump_state: "installed", hydraulics_model: "beggs" },
  pump: { nozzle_no: "13", throat_ratio: "C", date_set: "2026-08-10", source: "databricks" },
  pump_calibration: { status: "active", coefficients: fit }, as_built_locks: {}, prop_locks: {},
  well_model_fingerprint: "baseline", well_model_inputs: { contract: "fixed-oil-ipr-v1", oil_qmax: 500 },
};
const reset = () => { store.getState().selectWell(context.well); store.getState().applyContext(context); };

test("common oil IPR intent survives automatic fits and navigation state, and resets only on explicit reseed", () => {
  reset();
  store.getState().setMany({ qwf: 1234.5 });
  store.getState().setCommonIprIntent(true);
  store.getState().applyIprSeeds({ qwf: 200, pres: 1500, form_wc: .8 });
  assert.equal(store.getState().params.qwf, 1234.5);
  assert.equal(store.getState().params.form_wc, .74);
  assert.equal(store.getState().commonIprIntent, true);
  store.getState().applyIprSeeds({ qwf: 200 }, true);
  assert.equal(store.getState().commonIprIntent, false);
  store.getState().setCommonIprIntent(true);
  reset();
  assert.equal(store.getState().commonIprIntent, false);
});

test("changing hydraulics resets coefficients and cannot restore a foreign-model fit", () => {
  reset();
  store.getState().set("hydraulics_model", "drift_flux");
  let p = store.getState().params;
  assert.equal(p.pump_state, "installed");
  assert.equal(p.form_wc, .74);
  assert.equal(p.qwf, 1521);
  for (const key of Object.keys(CLEAN_PUMP)) assert.equal(p[key], CLEAN_PUMP[key]);
  store.getState().useInstalledPump();
  store.getState().refreshPumpContext(context);
  store.getState().applyPumpFit({ well: context.well, pump: "13C", era_start: "2026-08-10" }, fit);
  p = store.getState().params;
  assert.equal(p.hydraulics_model, "drift_flux");
  for (const key of Object.keys(CLEAN_PUMP)) assert.equal(p[key], CLEAN_PUMP[key]);
});

test("a matching alternative fit applies and restores after a clean-replacement preview", () => {
  reset();
  store.getState().set("hydraulics_model", "hagedorn_brown");
  store.getState().applyPumpFit({ well: context.well, pump: "13C", era_start: "2026-08-10", hydraulics_model: "hagedorn_brown", well_model_inputs: context.well_model_inputs }, fit);
  assert.equal(store.getState().params.ken, fit.ken);
  store.getState().refreshPumpContext({ ...context, pump_calibration: {
    status: "active", coefficients: fit, hydraulics_model: "hagedorn_brown",
  } });
  store.getState().set("pump_state", "replacement");
  store.getState().useInstalledPump();
  assert.equal(store.getState().params.kth, fit.kth);
  store.getState().setMany({ hydraulics_model: "beggs", ...fit });
  assert.equal(store.getState().params.kth, CLEAN_PUMP.kth);
});

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

test("exact API installation stamps apply in Anchorage and UTC and reject same-day replacements", () => {
  const previous = process.env.TZ;
  try {
    for (const tz of ["America/Anchorage", "UTC"]) {
      process.env.TZ = tz;
      for (const time of ["00:00:00", "15:30:00"]) {
        reset();
        const stamp = `2026-08-10T${time}+00:00`;
        store.getState().refreshPumpContext({ ...context, pump: { ...context.pump, date_set: stamp } });
        const result = { well: context.well, pump: "13C", era_start: "2026-08-10", installation_date_set: stamp, well_model_inputs: context.well_model_inputs };
        store.getState().applyPumpFit(result, { ken: .08 });
        assert.equal(store.getState().params.ken, .08);
        store.getState().applyPumpFit({ ...result, installation_date_set: "2026-08-10T10:30:00+00:00" }, { ken: .2 });
        assert.equal(store.getState().params.ken, .08);
      }
    }
  } finally {
    if (previous === undefined) delete process.env.TZ;
    else process.env.TZ = previous;
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

test("saved input refresh updates the baseline while preserving edits made during save", () => {
  reset();
  store.getState().set("qwf", 1600.25);
  const saved = { ...context, seeds: { ...context.seeds, qwf: 1600.25 }, ipr_source: "manual" };
  store.getState().set("qwf", 1700.5); // Another edit after the save was submitted.
  store.getState().refreshPumpContext(saved);
  assert.equal(store.getState().context.seeds.qwf, 1600.25);
  assert.equal(store.getState().context.ipr_source, "manual");
  assert.equal(store.getState().params.qwf, 1700.5);
  assert.ok(store.getState().manualFields.has("qwf"));
  store.getState().refreshPumpContext({ ...saved, well: "MPB-28" });
  assert.equal(store.getState().context.well, "MPE-42");
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

test("a fit cannot apply after its saved or session well curve changes", () => {
  const result = { well: context.well, pump: "13C", era_start: "2026-08-10", well_model_inputs: context.well_model_inputs };
  reset();
  store.getState().set("qwf", 2000);
  store.getState().applyPumpFit(result, { ken: .08 });
  assert.equal(store.getState().params.ken, fit.ken);
  reset();
  store.getState().applyPumpFit(result, { ken: .08 });
  assert.equal(store.getState().params.ken, .08);
  store.getState().refreshPumpContext({ ...context, well_model_fingerprint: "new-curve",
    well_model_inputs: { ...context.well_model_inputs, oil_qmax: 600 },
    pump_calibration: { status: "stale", coefficients: {} } });
  assert.equal(store.getState().params.ken, CLEAN_PUMP.ken);
  store.getState().applyPumpFit(result, { ken: .08 });
  assert.equal(store.getState().params.ken, CLEAN_PUMP.ken);
});

test("saving a new well curve preserves an in-progress replacement preview", () => {
  reset();
  store.getState().set("nozzle_no", "14");
  store.getState().set("qwf", 1800);
  store.getState().refreshPumpContext({ ...context, well_model_fingerprint: "new-curve",
    seeds: { ...context.seeds, qwf: 1800 }, pump_calibration: { status: "stale", coefficients: {} } });
  const p = store.getState().params;
  assert.equal(p.nozzle_no, "14");
  assert.equal(p.pump_state, "replacement");
  assert.equal(p.qwf, 1800);
  assert.equal(p.ken, CLEAN_PUMP.ken);
});
