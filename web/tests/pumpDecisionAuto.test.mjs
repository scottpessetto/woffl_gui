import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "typescript";

const dataModule = (source) => `data:text/javascript;base64,${Buffer.from(ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
}).outputText).toString("base64")}`;
const { shouldAutoRun } = await import(dataModule(
  readFileSync(new URL("../src/pages/optimize/pumpDecisionAuto.ts", import.meta.url), "utf8")));
const { stableStringify } = await import(dataModule(
  readFileSync(new URL("../src/api/client.ts", import.meta.url), "utf8")));

const request = (patch = {}) => ({
  pad: "S", target: null, offline: ["MPS-29"], future: [], n_pumps: 3,
  nozzles: ["9", "10"], throats: ["A", "B"], delta_pf_bpd: 1000, setpoint_psi: null, ...patch,
});
const key = stableStringify(request());
const ready = {
  allWells: true, offlineSettled: true, inputsValid: true, busy: false,
  requestKey: key, debouncedKey: key, lastKey: null, attemptedKey: null,
};

test("prices the pad on open once the offline set is settled", () => {
  assert.equal(shouldAutoRun(ready), true);
  assert.equal(shouldAutoRun({ ...ready, offlineSettled: false }), false);
  assert.equal(shouldAutoRun({ ...ready, inputsValid: false }), false);
});

test("a result or job for the same request is not re-priced on revisit", () => {
  assert.equal(shouldAutoRun({ ...ready, lastKey: key }), false);
  // Key order does not matter: the key is stable for an equal request.
  const reordered = stableStringify(Object.fromEntries(Object.entries(request()).reverse()));
  assert.equal(shouldAutoRun({ ...ready, lastKey: reordered }), false);
});

test("a changed pad-wide input re-prices only after the debounce settles", () => {
  const changed = stableStringify(request({ delta_pf_bpd: 500 }));
  const moved = { ...ready, requestKey: changed, lastKey: key };
  assert.equal(shouldAutoRun({ ...moved, debouncedKey: key }), false);
  assert.equal(shouldAutoRun({ ...moved, debouncedKey: changed }), true);
  const offline = stableStringify(request({ offline: ["MPS-29", "MPS-41"] }));
  assert.equal(shouldAutoRun({ ...ready, requestKey: offline, debouncedKey: offline, lastKey: key }), true);
});

test("never starts beside a running job, for a single well, or twice after a failed start", () => {
  assert.equal(shouldAutoRun({ ...ready, busy: true }), false);
  assert.equal(shouldAutoRun({ ...ready, allWells: false }), false);
  assert.equal(shouldAutoRun({ ...ready, attemptedKey: key }), false);
  // An expired job clears both the stored key and the attempt: it prices again.
  assert.equal(shouldAutoRun({ ...ready, lastKey: null, attemptedKey: null }), true);
});
