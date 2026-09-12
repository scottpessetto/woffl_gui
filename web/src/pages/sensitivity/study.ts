/** Immutable case identity and reproducible application of a server scenario. */
import type { CombineRequest, CombineRun, SimParams } from "../../api/types";
import { stableStringify } from "../../api/client";

export function studyCaseKey(request: CombineRequest): string {
  const { knobs: _knobs, ...caseInputs } = request;
  return stableStringify({ ...caseInputs, wc_basis: request.wc_basis ?? "fixed_oil_ipr",
    test_key: request.test_key ?? null, installation_key: request.installation_key ?? null });
}

export function matchingStudy(current: CombineRequest, submitted: CombineRequest | null | undefined): boolean {
  return !!submitted && studyCaseKey(current) === studyCaseKey(submitted);
}

export function appliedStudyParams(request: CombineRequest, run: CombineRun): SimParams | null {
  if (run.error || !run.applied_inputs) return null;
  return { ...request.params, ...run.applied_inputs };
}
