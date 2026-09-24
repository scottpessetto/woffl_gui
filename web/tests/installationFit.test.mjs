import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "typescript";

const source = readFileSync(new URL("../src/lib/installationFit.ts", import.meta.url), "utf8");
const js = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText;
const { fitAsMatch, paramsFor, shrinkageSigmas, changeouts, directionScore, excludedUids, NOT_HELD_OUT } =
  await import(`data:text/javascript;base64,${Buffer.from(js).toString("base64")}`);

const row = (date, status = "replay", message = null) => ({ date, status, installation_id: "a", wt_uid: date, message,
  bhp: 800, oil: 300, pf: 2000, predicted_bhp: 1, predicted_oil: 1, predicted_pf: 1, predicted_liquid: 1, sonic: false });

const result = {
  well: "MPE-42", physics_model: "entry-energy-v2", snapshot_id: "s", as_of: "2026-09-23", source: "databricks",
  notes: [], request: { hydraulics_model: "beggs", months: 24 },
  eras: [{ installation_id: "a", date_set: "2026-01-01", end: null, pump: "11C", nozzle: "11", throat: "C", flags: [], unavailable: null, n_tests: 3, ppf_span: 40 }],
  rows: [row("2026-01-05"), row("2026-02-05"), row("2026-03-05"), row("2026-03-06", "excluded", "Installation day")],
  models: { M1: { predictions: [
    { index: 0, installation_id: "a", predicted_bhp: 810, predicted_oil: 290, predicted_pf: 1990 },
    { index: 1, installation_id: "a", message: "cannot lift" },
    { index: 2, installation_id: "a", predicted_bhp: 805, predicted_oil: 295, predicted_pf: 2010 }] } },
  cv: { folds: [], scores: {}, embargo_days: 3, held_predictions: { M1: [
    { index: 2, installation_id: "a", origin: "2026-03-01", predicted_bhp: 900, predicted_oil: 250, predicted_pf: 2100 }] } },
};

test("fitted view shows in-sample values, keeps failures as rows and unusable tests as gaps", () => {
  const m = fitAsMatch(result, "M1", "fit");
  assert.deepEqual(m.rows.map((r) => r.status), ["fit", "failed", "fit", "excluded"]);
  assert.equal(m.rows[0].predicted_bhp, 810);
  assert.equal(m.rows[1].predicted_bhp, null);
  assert.equal(m.rows[1].message, "cannot lift");
  assert.equal(m.rows[3].message, "Installation day");
  assert.equal(m.validated_for_sizing, false);
});

test("held-out view never shows an in-sample value as a prediction", () => {
  const m = fitAsMatch(result, "M1", "held_out");
  assert.deepEqual(m.rows.map((r) => r.status), ["excluded", "excluded", "prediction", "excluded"]);
  assert.equal(m.rows[0].predicted_bhp, null);
  assert.equal(m.rows[0].message, NOT_HELD_OUT);
  assert.equal(m.rows[2].predicted_bhp, 900);
});

test("a model that was not fitted draws no values", () => {
  const m = fitAsMatch(result, "M3", "fit");
  assert.ok(m.rows.every((r) => r.predicted_bhp === null));
});

test("parameters split into well level and one installation", () => {
  const params = [{ name: "kth", installation_id: null }, { name: "fnz[0]", installation_id: "a" }, { name: "fnz[1]", installation_id: "b" }];
  assert.deepEqual(paramsFor(params, null).map((p) => p.name), ["kth"]);
  assert.deepEqual(paramsFor(params, "a").map((p) => p.name), ["fnz[0]"]);
});

test("shrinkage is measured in prior sd, in log space for the nozzle factor", () => {
  const fnz = { physical: "fnz", value: Math.exp(0.1), prior_mean: 1, prior_sd: 0.05 };
  assert.ok(Math.abs(shrinkageSigmas(fnz) - 2) < 1e-12);
  assert.equal(shrinkageSigmas({ physical: "kth", value: 0.3, prior_mean: null, prior_sd: null }), null);
});

test("changeout checks count only clear changes", () => {
  const folds = [
    { origin: "2026-02-01", kind: "changeout", models: { M1: { changeout: {
      bhp: { measured: -170, predicted: -30, clear: true, direction_correct: true },
      oil: { measured: 5, predicted: -3, clear: false, direction_correct: null } } } } },
    { origin: "2026-05-01", kind: "later_same_pump", models: { M1: {} } },
  ];
  const rows = changeouts(folds, "M1");
  assert.equal(rows.length, 1);
  assert.deepEqual(directionScore(rows), { correct: 1, clear: 1 });
});

test("Solver exclusion keys become wt_uid strings; date-only keys cannot be matched", () => {
  assert.deepEqual(excludedUids(["uid:-3591520", "date:2026-01-01", "uid:42"]), ["-3591520", "42"]);
});
