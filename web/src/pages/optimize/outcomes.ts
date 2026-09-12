import type { PadRunRow } from "../../api/types";

/** A missing numeric selection alone never authorizes a shut-in label. */
export function unselectedOutcomeLabel(row: Pick<PadRunRow, "outcome">): string {
  if (row.outcome === "economic_shut_in") return "SHUT IN (modeled choice)";
  if (row.outcome === "missing_inputs") return "INPUTS MISSING";
  return "MODEL UNAVAILABLE";
}
