/**
 * Empty-state card shown by pages before a well is selected (simActive off).
 */

import { Card } from "../components/ui";

export function Welcome() {
  return (
    <div className="flex justify-center pt-12">
      <Card className="max-w-xl">
        <h2 className="text-lg font-semibold text-slate-800">Pick a well to get started</h2>
        <ol className="mt-3 list-decimal space-y-2 pl-5 text-sm text-slate-600">
          <li>
            Select a well in the sidebar - inputs auto-populate from Databricks and the solve
            runs.
          </li>
          <li>Adjust anything in the sidebar - results re-solve live.</li>
          <li>Switch views in the Single Well nav to compare pumps, sweeps, and profiles.</li>
        </ol>
        <p className="mt-4 text-xs text-slate-400">
          Custom well: fill the sidebar and press Run simulation.
        </p>
      </Card>
    </div>
  );
}
