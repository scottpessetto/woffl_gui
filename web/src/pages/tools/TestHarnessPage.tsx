/**
 * Test Harness - curated sanity cases against today's live data.
 *
 * Port of woffl/gui/scotts_tools/test_harness.py and the case registry it
 * drove. These are NOT unit tests: pytest covers the code offline and mocked.
 * Each case runs a real pipeline against live Databricks and asserts the
 * answer is physically plausible, which catches the failure pytest
 * structurally cannot - the code is fine, the data moved.
 */

import { useState } from "react";

import { useHarnessCases } from "../../api/hooks";
import { Badge, Button, Card, ErrorNote, Metric, Section, Spinner } from "../../components/ui";
import { fmtNum } from "../../lib/format";
import { RunStatus, useToolRun } from "./ToolRun";

interface HarnessResult {
  name: string;
  description: string;
  passed: boolean;
  summary: string;
  details: Record<string, unknown>;
  error: string | null;
  seconds: number;
}

export default function TestHarnessPage() {
  const cases = useHarnessCases();
  const run = useToolRun<Record<string, never>>("/tools/harness/run");
  const [onlyFailures, setOnlyFailures] = useState(false);
  const [open, setOpen] = useState<Set<string>>(new Set());

  const result = run.result as
    | { results?: HarnessResult[]; passed?: number; failed?: number; total?: number; seconds?: number }
    | null;
  const results = result?.results ?? [];
  const shown = onlyFailures ? results.filter((r) => !r.passed) : results;

  return (
    <div className="space-y-4">
      <Section
        title="Test Harness"
        actions={
          <Button onClick={() => run.run({})} disabled={run.running}>
            {run.running ? "Running..." : "Run all cases"}
          </Button>
        }
      >
        <p className="mb-3 text-sm text-slate-600">
          Each case exercises a real pipeline against today&apos;s data and checks the answer is
          plausible. They catch what pytest cannot: a view dropping a column, an allocation
          going to zero, a pad falling out of the fleet - the maths still runs and returns
          something confidently wrong.
        </p>
        {cases.isLoading && <Spinner label="Loading cases" />}
        {cases.isError && <ErrorNote error={cases.error} />}
        {cases.data && (
          <p className="text-xs text-slate-500">
            {cases.data.cases.length} cases registered. Add more in{" "}
            <code className="rounded bg-slate-100 px-1">server/services/tools/harness.py</code>.
          </p>
        )}
      </Section>

      <RunStatus run={run as never} idle="Press Run all cases to execute against live data." />

      {result && (
        <Card>
          <div className="flex flex-wrap items-center gap-6">
            <Metric label="Passed" value={`${result.passed} / ${result.total}`} />
            <Metric label="Failed" value={fmtNum(result.failed, 0)} />
            <Metric label="Elapsed" value={`${fmtNum(result.seconds, 1)}s`} />
            {result.failed === 0 ? (
              <Badge tone="good">All cases passed</Badge>
            ) : (
              <Badge tone="poor">{result.failed} failed</Badge>
            )}
            <Button size="sm" variant="ghost" onClick={() => setOnlyFailures((v) => !v)}>
              {onlyFailures ? "Show all" : "Only failures"}
            </Button>
          </div>
        </Card>
      )}

      {shown.length > 0 && (
        <div className="space-y-2">
          {shown.map((r) => {
            // Failures open by default - a red row you have to click is a red
            // row people stop clicking.
            const isOpen = open.has(r.name) || !r.passed;
            return (
              <Card key={r.name} padded={false}>
                <button
                  type="button"
                  onClick={() =>
                    setOpen((s) => {
                      const n = new Set(s);
                      if (n.has(r.name)) n.delete(r.name);
                      else n.add(r.name);
                      return n;
                    })
                  }
                  className="flex w-full items-center gap-3 px-4 py-3 text-left hover:bg-slate-50"
                >
                  <Badge tone={r.passed ? "good" : "poor"}>{r.passed ? "PASS" : "FAIL"}</Badge>
                  <span className="font-medium text-slate-800">{r.name}</span>
                  <span className="text-sm text-slate-600">{r.summary}</span>
                  <span className="ml-auto text-xs tabular-nums text-slate-400">
                    {fmtNum(r.seconds, 1)}s
                  </span>
                </button>
                {isOpen && (
                  <div className="border-t border-slate-100 px-4 py-3">
                    {r.description && (
                      <p className="mb-2 text-xs leading-relaxed text-slate-500">{r.description}</p>
                    )}
                    {r.error && <p className="mb-2 text-xs text-red-600">Exception: {r.error}</p>}
                    <pre className="max-h-56 overflow-auto rounded-md bg-slate-50 p-2 text-xs text-slate-700">
                      {JSON.stringify(r.details, null, 2)}
                    </pre>
                  </div>
                )}
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
