/**
 * Which wells a pad optimization run should leave out, and why.
 *
 * Two sources, deliberately weighted differently:
 *
 * 1. The field's own daily downtime log (mpu.wells.vw_shut_in, >= 20 down
 *    hours on the latest date, corrected by live safety-valve state), split
 *    by down code into LONG-TERM shut-in (T01 mech, T02 reservoir, T03
 *    convert, T05 P&A) and ordinary short-term shut-ins. It already reaches
 *    the client through /well-sort/tables.
 * 2. The engineer's own ticks on the readiness board (localStorage).
 *
 * Only LTSI pre-ticks. A T-coded well is mechanically or reservoir-shut for
 * the foreseeable future and has no business in a pad plan, so excluding it
 * by default is safe. An ordinary shut-in is a day-to-day operating state the
 * log can lag by a day either way, so it only gets an advisory badge - the
 * engineer decides. Test recency is deliberately NOT a source: this repo
 * already computes a 60-day StaleTest and already refuses to read it as
 * offline (well_sort_engine.add_online_decision maps stale to "verify_stale",
 * an abstention), because a producing well with an overdue test would be
 * silently dropped from the plan while a well shut in last week with a test
 * from the week before would not be flagged at all.
 *
 * An explicit untick outranks the auto-tick and persists (keepOnline in the
 * optimize store). When /well-sort/tables is unavailable the whole thing
 * degrades to the manual ticks alone - never to a test-recency guess.
 */

import { useMemo } from "react";

import { useWellSortTables } from "../../api/hooks";
import type { WellSortShutRow } from "../../api/types";
import { useOptimizeStore } from "../../state/optimize";
import { useWellSortStore } from "../../state/wellSort";

/** Down codes starting with T are long-term (well_sort_client.LTSI_CODE_PREFIX). */
export interface ShutInfo {
  /** First day of the current consecutive full-day-down streak. */
  since: string | null;
  code: string | null;
  reason: string | null;
  /** True = long-term shut-in, which is what earns the automatic tick. */
  ltsi: boolean;
}

export interface PadOffline {
  /** Wells to exclude from the run: manual ticks plus LTSI, minus keep-online. */
  offline: Set<string>;
  /** Every logged-down well on the requested pads, LTSI or not. */
  shut: Map<string, ShutInfo>;
  /** How many of ``offline`` were pre-ticked from the log rather than by hand. */
  autoCount: number;
  /** False while the downtime log is loading or unavailable. */
  ready: boolean;
}

function row(r: WellSortShutRow, ltsi: boolean): [string, ShutInfo] {
  return [
    r.well,
    { since: r.shut_in_since, code: r.current_code, reason: r.current_reason, ltsi },
  ];
}

/**
 * Merge the downtime log with the board's ticks for one or more pads.
 *
 * Matches the Well Sort page's own query arguments so the two share a single
 * cached fetch rather than each paying for the pipeline.
 */
export function usePadOffline(pads: string[]): PadOffline {
  const popsPads = useWellSortStore((s) => s.popsPads);
  const forceTrue = useWellSortStore((s) => s.forceTrue);
  const tables = useWellSortTables("allocated", 60, popsPads, forceTrue);
  const manualByPad = useOptimizeStore((s) => s.offline);
  const keepOnlineByPad = useOptimizeStore((s) => s.keepOnline);

  const key = pads.join(",");
  return useMemo(() => {
    const on = new Set(pads);
    const shut = new Map<string, ShutInfo>();
    for (const r of tables.data?.offline ?? []) {
      if (r.pad && on.has(r.pad)) shut.set(...row(r, false));
    }
    for (const r of tables.data?.ltsi ?? []) {
      if (r.pad && on.has(r.pad)) shut.set(...row(r, true));
    }

    const keepOnline = new Set(pads.flatMap((p) => keepOnlineByPad[p] ?? []));
    const manual = new Set(pads.flatMap((p) => manualByPad[p] ?? []));
    const offline = new Set<string>();
    let autoCount = 0;
    for (const [well, info] of shut) {
      if (info.ltsi && !keepOnline.has(well) && !manual.has(well)) {
        offline.add(well);
        autoCount += 1;
      }
    }
    for (const well of manual) {
      if (!keepOnline.has(well)) offline.add(well);
    }
    return { offline, shut, autoCount, ready: tables.data !== undefined };
    // `key` stands in for `pads`, which callers rebuild every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, tables.data, manualByPad, keepOnlineByPad]);
}
