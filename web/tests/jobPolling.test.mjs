import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "typescript";
import { QueryClient, QueryObserver } from "@tanstack/react-query";

// Execute the real shared policy without introducing a second TS test stack.
const source = readFileSync(new URL("../src/api/client.ts", import.meta.url), "utf8");
const js = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
}).outputText;
const { ApiError, isMissingJob, retryJobPoll, jobPollDelay } =
  await import(`data:text/javascript;base64,${Buffer.from(js).toString("base64")}`);

test("polling recovers from a lost response without discarding the job", async () => {
  const client = new QueryClient();
  let calls = 0;
  let persistedId = "running-job";
  const observer = new QueryObserver(client, {
    queryKey: ["job", persistedId],
    queryFn: async () => {
      if (++calls === 1) throw new TypeError("network unavailable");
      return { status: "done", result: { oil: 80 } };
    },
    retry: retryJobPoll,
    retryDelay: 0,
  });
  let unsubscribe;
  try {
    const result = await new Promise((resolve) => {
      unsubscribe = observer.subscribe((state) => {
        if (isMissingJob(state.error)) persistedId = null;
        if (state.isSuccess || state.isError) resolve(state);
      });
    });
    assert.equal(result.data.result.oil, 80);
    assert.equal(persistedId, "running-job");
    assert.equal(calls, 2);
  } finally {
    unsubscribe?.();
    client.clear();
  }
});

test("only a confirmed missing job clears its handle; retries are bounded", () => {
  for (const status of [404, 410]) {
    const error = new ApiError(status, { error: "http", message: "missing" });
    assert.equal(isMissingJob(error), true);
    assert.equal(retryJobPoll(0, error), false);
  }
  for (const status of [401, 403, 429, 500, 503]) {
    const error = new ApiError(status, { error: "http", message: "temporary" });
    assert.equal(isMissingJob(error), false);
    assert.equal(retryJobPoll(0, error), true);
    assert.equal(retryJobPoll(3, error), false);
  }
  assert.equal(jobPollDelay(0), 1000);
  assert.equal(jobPollDelay(10), 15000);
});
