import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "typescript";

const source = readFileSync(new URL("../src/pages/optimize/outcomes.ts", import.meta.url), "utf8");
const code = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
}).outputText;
const { unselectedOutcomeLabel } = await import(`data:text/javascript;base64,${Buffer.from(code).toString("base64")}`);

test("only an explicit modeled economic choice can display SHUT IN", () => {
  assert.match(unselectedOutcomeLabel({ outcome: "economic_shut_in" }), /SHUT IN/);
  for (const outcome of ["failed_model", "unsupported_model", "missing_inputs", undefined]) {
    assert.doesNotMatch(unselectedOutcomeLabel({ outcome }), /SHUT IN/);
  }
  assert.equal(unselectedOutcomeLabel({ outcome: "missing_inputs" }), "INPUTS MISSING");
});
