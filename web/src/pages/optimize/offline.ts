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
 * Pre-ticked by default: LTSI (a T-coded well is mechanically or reservoir-
 * shut for the foreseeable future), wells currently shut in under the plain
 * SI code (the user's call, 2026-09-22: a well the log shows down under SI
 * today should not enter a plan by default, even though the log can lag a
 * restart by a day - untick it when it is back), and the named non-producers
 * in DEFAULT_OFFLINE (recycle wells). Other down codes only get an advisory
 * badge - the engineer decides. Test recency is deliberately NOT a source: this repo
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

/**
 * Wells that never belong in a producer plan, pre-ticked offline on every
 * pad with the reason shown on the board. Recycle wells take PF-side water,
 * not produce; the user named MPS-29 on 2026-09-22.
 */
export const DEFAULT_OFFLINE: Record<string, string> = {
  "MPS-29": "recycle well",
};

/** The plain shut-in down code that earns the automatic tick. */
export const SI_CODE = "SI";

/** Down codes starting with T are long-term (well_sort_client.LTSI_CODE_PREFIX). */
export interface ShutInfo {
  /** First day of the current consecutive full-day-down streak. */
  since: string | null;
  code: string | null;
  reason: string | null;
  /** True = long-term shut-in. */
  ltsi: boolean;
  /** True = pre-ticked offline (LTSI, a current SI-coded shut-in, or DEFAULT_OFFLINE). */
  auto: boolean;
  /** Why it is pre-ticked when it is not a downtime-log entry (DEFAULT_OFFLINE). */
  note?: string;
}

export interface PadOffline {
  /** Wells to exclude from the run: manual ticks plus LTSI, minus keep-online. */
  offline: Set<string>;
  /** Every logged-down well on the requested pads, LTSI or not. */
  shut: Map<string, ShutInfo>;
  /** How many of ``offline`` were pre-ticked (log or DEFAULT_OFFLINE) rather than by hand. */
  autoCount: number;
  /** False while the downtime log is loading or unavailable. */
  ready: boolean;
  /** The downtime log failed to load: long-term shut-ins are NOT pre-ticked. */
  failed: boolean;
}

function row(r: WellSortShutRow, ltsi: boolean): [string, ShutInfo] {
  const si = (r.current_code ?? "").trim().toUpperCase() === SI_CODE;
  return [
    r.well,
    { since: r.shut_in_since, code: r.current_code, reason: r.current_reason, ltsi, auto: ltsi || si },
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
    // Named non-producers: pad from the MP<pad>-NN name, e.g. MPS-29 -> S.
    for (const [well, note] of Object.entries(DEFAULT_OFFLINE)) {
      const pad = /^MP([A-Z])-/i.exec(well)?.[1]?.toUpperCase();
      if (pad && on.has(pad) && !shut.has(well)) {
        shut.set(well, { since: null, code: null, reason: null, ltsi: false, auto: true, note });
      }
    }

    const keepOnline = new Set(pads.flatMap((p) => keepOnlineByPad[p] ?? []));
    const manual = new Set(pads.flatMap((p) => manualByPad[p] ?? []));
    const offline = new Set<string>();
    let autoCount = 0;
    for (const [well, info] of shut) {
      if (info.auto && !keepOnline.has(well) && !manual.has(well)) {
        offline.add(well);
        autoCount += 1;
      }
    }
    for (const well of manual) {
      if (!keepOnline.has(well)) offline.add(well);
    }
    return { offline, shut, autoCount, ready: tables.data !== undefined, failed: tables.isError && tables.data === undefined };
    // `key` stands in for `pads`, which callers rebuild every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, tables.data, tables.isError, manualByPad, keepOnlineByPad]);
}
