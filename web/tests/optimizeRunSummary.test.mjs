import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "typescript";

const dataModule = (source) => `data:text/javascript;base64,${Buffer.from(ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
}).outputText).toString("base64")}`;
const load = (path) => import(dataModule(readFileSync(new URL(path, import.meta.url), "utf8")));

const {
  planChanges, changeText, runStatus, runProgress, requestBlockers, requestChanges, fmtDuration,
} = await load("../src/pages/optimize/runSummary.ts");
const { validationMessage } = await load("../src/api/client.ts");
const { formFor, DEFAULT_RUN_FORM } = await load("../src/state/runForm.ts");

const padReq = (patch = {}) => ({
  kind: "pad", pad: "M", offline: [], future: [], required_wells: [],
  nozzles: ["9", "10"], throats: ["A", "B"], method: "milp", strategy: "jpco",
  lambda_bopd_per_bpd: null, marginal_wc: null, parsimony_bopd: 0, n_pumps: null, n_steps: null,
  setpoint_psi: null, p0_psi: 2792, psi_per_kbpd: 13.69, c_pad_pf_psi: 3400, cfp_pad_pf_psi: {},
  cfp_pads: ["B", "G", "C", "J"], e_pad_build: "SM25000_26STG", e_pad_suction_psi: 2800,
  e_pad_hz_max: 60, e_pad_max_header_psi: 3500, e_pad_amp_limit_a: null, ...patch,
});

test("422 validation lists become one readable sentence instead of 'HTTP 422'", () => {
  const detail = [
    { loc: ["body"], msg: "Value error, A pad request cannot require a well that it also excludes: MPM-12", type: "value_error" },
    { loc: ["body", "lambda_bopd_per_bpd"], msg: "Input should be less than or equal to 10", type: "less_than_equal" },
  ];
  assert.equal(
    validationMessage(detail),
    "A pad request cannot require a well that it also excludes: MPM-12; lambda_bopd_per_bpd: Input should be less than or equal to 10",
  );
  assert.equal(validationMessage({ message: "x" }), null);
  assert.equal(validationMessage([]), null);
});

test("only hardware that differs from today is a change, biggest gain first", () => {
  const rows = [
    { well: "A", current_pump: "12B", pump: "12B", pump_state: "installed", modeled_hardware_gain: 0 },
    { well: "B", current_pump: "12B", pump: "13C", pump_state: "replacement", modeled_hardware_gain: 40 },
    { well: "C", current_pump: "11A", pump: "11A", pump_state: "replacement", modeled_hardware_gain: 12 },
    { well: "D", current_pump: "10B", pump: null, outcome: "economic_shut_in", modeled_hardware_gain: -30 },
    { well: "E", current_pump: "10B", pump: null, outcome: "failed_model", modeled_hardware_gain: null },
    { well: "F", current_pump: null, pump: "9A", pump_state: "replacement", modeled_hardware_gain: null },
  ];
  const changes = planChanges(rows);
  assert.deepEqual(changes.map((c) => c.well), ["B", "C", "D", "F"]);
  assert.equal(changeText(changes[0]), "12B to 13C");
  assert.equal(changeText(changes[1]), "replace 11A with a clean 11A");
  assert.equal(changeText(changes[2]), "shut in (was 10B)");
  assert.equal(changeText(changes[3]), "run 9A");
});

test("an incomplete run that is also infeasible is never shown as qualified", () => {
  const coverage = { complete: false, expected_online: 3, accounted_online: 2, unaccounted_wells: ["X"], rows: [] };
  const s = runStatus({ feasible: null, modeled_subset_feasible: false, recommendation_status: "incomplete_exploratory" }, coverage);
  assert.equal(s.label, "Exploratory");
  assert.match(s.detail, /do not meet the operating limits/);
  const complete = { ...coverage, complete: true, unaccounted_wells: [] };
  assert.equal(runStatus({ feasible: false, recommendation_status: "conditional_operating_limits" }, complete).label, "Conditional");
  assert.equal(runStatus({ feasible: true, converged: true, recommendation_status: "complete_model_coverage" }, complete).label, "Qualified");
  assert.equal(runStatus({ feasible: true }, undefined).label, "Exploratory");
});

test("progress text yields a fraction and a time-left estimate", () => {
  const p = runProgress("trial 5/15 - header 3,100 psi", 50);
  assert.equal(p.fraction, 5 / 15);
  assert.equal(Math.round(p.etaSeconds), 100);
  assert.equal(runProgress("queued - waiting for a job slot (max 1 at once)", 3).queued, true);
  assert.equal(runProgress("reading current pumps + tests...", 3).fraction, null);
  assert.equal(fmtDuration(100), "1 min 40 s");
  assert.equal(fmtDuration(null), "");
});

test("requests the server would reject are named before sending", () => {
  assert.deepEqual(requestBlockers(padReq()), []);
  assert.match(requestBlockers(padReq({ offline: ["MPM-12"], required_wells: ["MPM-12"] }))[0], /both Offline and Required/);
  assert.match(requestBlockers(padReq({ strategy: "choke", future: [{ name: "NEW", match: "MPM-12" }] }))[0], /planned pump for NEW/);
  assert.deepEqual(requestBlockers(padReq({ strategy: "choke", future: [{ name: "NEW", match: "MPM-12", nozzle: "12", throat: "B" }] })), []);
  assert.match(requestBlockers(padReq({ pad: "E", e_pad_suction_psi: 3500 }))[0], /suction must be below/);
  assert.match(requestBlockers(padReq({ lambda_bopd_per_bpd: 20 }))[0], /water price/);
  assert.match(requestBlockers(padReq({ nozzles: [] }))[0], /nozzle/);
  const cfp = padReq({ kind: "cfp", pad: null, cfp_pad_pf_psi: { B: 3000 } });
  assert.match(requestBlockers(cfp)[0], /cannot exceed the reference discharge/);
  assert.match(requestBlockers({ ...cfp, cfp_pad_pf_psi: {}, p0_psi: Number.NaN })[0], /reference discharge/);
});

test("a result is stale only when an input the run uses changed", () => {
  const before = padReq({ offline: ["MPM-01"] });
  assert.deepEqual(requestChanges(before, padReq({ offline: ["MPM-01"] })), []);
  assert.deepEqual(requestChanges(before, padReq({ offline: ["MPM-02"] })), ["offline +MPM-02 -MPM-01"]);
  assert.deepEqual(requestChanges(before, padReq({ offline: ["MPM-01"], n_pumps: 2 })), ["pumps online"]);
  // CFP knobs never make a pad result stale, and vice versa.
  assert.deepEqual(requestChanges(before, padReq({ offline: ["MPM-01"], p0_psi: 2700 })), []);
  const cfpBefore = padReq({ kind: "cfp", pad: null });
  assert.deepEqual(requestChanges(cfpBefore, padReq({ kind: "cfp", pad: null, method: "mckp" })), []);
  assert.deepEqual(requestChanges(cfpBefore, padReq({ kind: "cfp", pad: null, p0_psi: 2700 })), ["reference discharge"]);
  assert.deepEqual(requestChanges(null, padReq()), []);
});

test("stored run forms fall back per field when an old shape is restored", () => {
  const form = formFor({ M: { nPumps: 2, nozzles: "bad", waterPricePerK: "20" } }, "M");
  assert.equal(form.nPumps, 2);
  assert.deepEqual(form.nozzles, DEFAULT_RUN_FORM.nozzles);
  assert.equal(form.waterPricePerK, DEFAULT_RUN_FORM.waterPricePerK);
  assert.equal(formFor({}, "S").nPumps, null);
});
