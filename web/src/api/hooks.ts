import { keepPreviousData, useMutation, useQuery, useQueryClient, type QueryClient } from "@tanstack/react-query";

import { api, get, post, stableStringify } from "./client";
import type {
  AgingPumpsResponse,
  BatchResponse,
  CalibrateRequest,
  CalibrateResponse,
  ClearIprPinResponse,
  EquivalentsResponse,
  IprFitRequest,
  IprFitResponse,
  IprPinResponse,
  JpHistoryResponse,
  MarginalWcResponse,
  MetaResponse,
  OptimizeJobStatus,
  OptimizeRunRequest,
  OptimizeRunStarted,
  PadFitStatusResponse,
  PadMarginalWcResponse,
  PfRangeResponse,
  PressureProfileResponse,
  PropHistoryResponse,
  PropLockRequest,
  PropLockResponse,
  SaveIprRequest,
  SaveIprResponse,
  SimParams,
  SolveResult,
  TriageResponse,
  WellContext,
  WellDatabaseResponse,
  WellProfileResponse,
  WellSortEventsResponse,
  WellSortMode,
  WellSortTablesResponse,
  WellTestsResponse,
  WellsResponse,
} from "./types";

const MIN_5 = 5 * 60 * 1000;
const MIN_30 = 30 * 60 * 1000;
const HOUR_1 = 60 * 60 * 1000;

export const useMeta = () =>
  useQuery({
    queryKey: ["meta"],
    queryFn: ({ signal }) => get<MetaResponse>("/meta", signal),
    staleTime: Infinity,
  });

export const useWells = () =>
  useQuery({
    queryKey: ["wells"],
    queryFn: ({ signal }) => get<WellsResponse>("/wells", signal),
    staleTime: HOUR_1,
  });

export const useWellContext = (well: string, months: number, cap: number) =>
  useQuery({
    queryKey: ["well-context", well, months, cap],
    queryFn: ({ signal }) =>
      get<WellContext>(
        `/wells/${encodeURIComponent(well)}/context?months=${months}&cap=${cap}`,
        signal,
      ),
    enabled: well !== "Custom",
    staleTime: MIN_5,
    retry: 1,
  });

export const useWellTests = (well: string, months: number, cap: number) =>
  useQuery({
    queryKey: ["well-tests", well, months, cap],
    queryFn: ({ signal }) =>
      get<WellTestsResponse>(
        `/wells/${encodeURIComponent(well)}/tests?months=${months}&cap=${cap}`,
        signal,
      ),
    enabled: well !== "Custom",
    staleTime: MIN_5,
  });

/**
 * The solve is a pure function of (well, params) - modeled as a QUERY keyed
 * on a stable hash so repeat runs and back-navigation are instant cache hits.
 * `enabled` gates on simActive; params should be the DEBOUNCED object.
 */
export const useSolve = (well: string, params: SimParams, enabled: boolean) =>
  useQuery({
    queryKey: ["solve", well, stableStringify(params)],
    queryFn: ({ signal }) => post<SolveResult>("/solve", { well, params }, signal),
    enabled,
    staleTime: MIN_30,
    placeholderData: keepPreviousData,
    retry: false,
  });

export const useIprFit = (req: IprFitRequest, enabled: boolean) =>
  useQuery({
    queryKey: ["ipr-fit", stableStringify(req)],
    queryFn: ({ signal }) => post<IprFitResponse>("/ipr/fit", req, signal),
    enabled: enabled && req.well !== "Custom",
    staleTime: MIN_5,
    retry: false,
  });

export const useIprPin = (well: string) =>
  useQuery({
    queryKey: ["ipr-pin", well],
    queryFn: ({ signal }) => get<IprPinResponse>(`/wells/${encodeURIComponent(well)}/ipr-pin`, signal),
    enabled: well !== "Custom",
    staleTime: MIN_5,
    retry: false,
  });

/** Queries that reflect prop_hist state - refetched after any save/clear. */
const invalidateSavedIpr = (qc: QueryClient, well: string) => {
  void qc.invalidateQueries({ queryKey: ["ipr-pin", well] });
  void qc.invalidateQueries({ queryKey: ["well-context", well] });
  void qc.invalidateQueries({ queryKey: ["prop-history", well] });
  void qc.invalidateQueries({ queryKey: ["pad-fit"] });
};

export const useSaveIpr = (well: string) => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (req: SaveIprRequest) =>
      post<SaveIprResponse>(`/wells/${encodeURIComponent(well)}/save-ipr`, req),
    onSuccess: () => invalidateSavedIpr(qc, well),
  });
};

export const useClearIprPin = (well: string) => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () =>
      api<ClearIprPinResponse>(`/wells/${encodeURIComponent(well)}/ipr-pin`, { method: "DELETE" }),
    onSuccess: () => invalidateSavedIpr(qc, well),
  });
};

/** Toggle a WC/GOR/ResP field lock. The caller updates the params store's
 * propLocks from the response - a context refetch never reseeds (the
 * seededFor guard), so the store is the only live mirror of lock state. */
export const usePropLock = (well: string) => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (req: PropLockRequest) =>
      post<PropLockResponse>(`/wells/${encodeURIComponent(well)}/prop-lock`, req),
    onSuccess: (r) => {
      if (r.ok) invalidateSavedIpr(qc, well);
    },
  });
};

/** BHP friction calibration - read-only compute, can take tens of seconds
 * (Nelder-Mead multi-start over full solves). The caller applies the
 * returned ken/kth/kdi to the sidebar on success. */
export const useCalibrate = () =>
  useMutation({
    mutationFn: (req: CalibrateRequest) => post<CalibrateResponse>("/calibrate", req),
  });

/**
 * Expensive sweeps run on explicit submit: pages snapshot the params into
 * local state and pass the snapshot here (null = not requested yet).
 */
export const useBatch = (well: string, snapshot: SimParams | null) =>
  useQuery({
    queryKey: ["batch", well, snapshot ? stableStringify(snapshot) : "none"],
    queryFn: ({ signal }) => post<BatchResponse>("/batch", { well, params: snapshot }, signal),
    enabled: snapshot !== null,
    // A snapshot-keyed sweep never goes stale (same inputs = same physics)
    // and must survive page detours - the snapshot store re-attaches to it.
    staleTime: Infinity,
    gcTime: HOUR_1,
    retry: false,
  });

export const usePfRange = (well: string, snapshot: SimParams | null) =>
  useQuery({
    queryKey: ["pf-range", well, snapshot ? stableStringify(snapshot) : "none"],
    queryFn: ({ signal }) => post<PfRangeResponse>("/pf-range", { well, params: snapshot }, signal),
    enabled: snapshot !== null,
    staleTime: MIN_30,
    retry: false,
  });

export const usePressureProfile = (well: string, params: SimParams, enabled: boolean) =>
  useQuery({
    queryKey: ["pressure-profile", well, stableStringify(params)],
    queryFn: ({ signal }) =>
      post<PressureProfileResponse>("/pressure-profile", { well, params }, signal),
    enabled,
    staleTime: MIN_30,
    placeholderData: keepPreviousData,
    retry: false,
  });

export const useWellProfile = (well: string, jpumpTvd: number, fieldModel: string) =>
  useQuery({
    queryKey: ["well-profile", well, jpumpTvd, fieldModel],
    queryFn: ({ signal }) =>
      get<WellProfileResponse>(
        `/wells/${encodeURIComponent(well)}/profile?jpump_tvd=${jpumpTvd}&field_model=${fieldModel}`,
        signal,
      ),
    staleTime: HOUR_1,
  });

export const useEquivalents = (nozzle: string, throat: string) =>
  useQuery({
    queryKey: ["equivalents", nozzle, throat],
    queryFn: ({ signal }) =>
      get<EquivalentsResponse>(`/pumps/equivalents?nozzle=${nozzle}&throat=${throat}`, signal),
    staleTime: Infinity,
  });

export const useJpHistory = (well: string) =>
  useQuery({
    queryKey: ["jp-history", well],
    queryFn: ({ signal }) => get<JpHistoryResponse>(`/wells/${encodeURIComponent(well)}/jp-history`, signal),
    enabled: well !== "Custom",
    staleTime: MIN_30,
  });

export const useWellDatabase = () =>
  useQuery({
    queryKey: ["well-database"],
    queryFn: ({ signal }) => get<WellDatabaseResponse>("/database/wells", signal),
    staleTime: HOUR_1,
  });

export const useAgingPumps = (knownOnly: boolean, onlineOnly: boolean, onlineDays: number, minDays: number) =>
  useQuery({
    queryKey: ["aging-pumps", knownOnly, onlineOnly, onlineDays, minDays],
    queryFn: ({ signal }) =>
      get<AgingPumpsResponse>(
        `/database/aging-pumps?known_only=${knownOnly}&online_only=${onlineOnly}&online_days=${onlineDays}&min_days=${minDays}`,
        signal,
      ),
    staleTime: MIN_30,
  });

export const usePropHistory = (well: string | null) =>
  useQuery({
    queryKey: ["prop-history", well],
    queryFn: ({ signal }) =>
      get<PropHistoryResponse>(`/database/prop-history/${encodeURIComponent(well!)}`, signal),
    enabled: well !== null,
    staleTime: MIN_5,
  });

/** Optimization pad board: saved-fit readiness for a pad's wells + the
 * donor wells of any future entries. Invalidated by every prop_hist write
 * (see invalidateSavedIpr), so a fresh save shows up on the board at once. */
export const usePadFitStatus = (pad: string | null, extras: string[]) =>
  useQuery({
    queryKey: ["pad-fit", pad, [...extras].sort()],
    queryFn: ({ signal }) => {
      const parts = extras.map((w) => `extra=${encodeURIComponent(w)}`).join("&");
      return get<PadFitStatusResponse>(
        `/optimize/pad-status?pad=${encodeURIComponent(pad!)}${parts ? `&${parts}` : ""}`,
        signal,
      );
    },
    enabled: pad !== null,
    staleTime: MIN_5,
    gcTime: HOUR_1,
    placeholderData: keepPreviousData,
  });

/** Start a pad/CFP optimization run (background job server-side). */
export const useStartOptimizeRun = () =>
  useMutation({
    mutationFn: (req: OptimizeRunRequest) =>
      post<OptimizeRunStarted>("/optimize/run", req),
  });

/** Poll one run job every 2.5 s while it's running; stops when settled.
 * A 404 (expired/unknown job after a server restart) surfaces as error -
 * callers clear their stored job id on it. */
export const useOptimizeJob = (jobId: string | null) =>
  useQuery({
    queryKey: ["optimize-job", jobId],
    queryFn: ({ signal }) => get<OptimizeJobStatus>(`/optimize/run/${jobId}`, signal),
    enabled: jobId !== null,
    refetchInterval: (query) => (query.state.data?.status === "running" ? 2500 : false),
    // Keep polling when the window loses focus - an engineer kicks off a
    // multi-minute run and alt-tabs; the monitor must not freeze at 0s.
    refetchIntervalInBackground: true,
    staleTime: Infinity,
    gcTime: HOUR_1,
    retry: false,
  });

// ---------------------------------------------------------------------------
// Well Sort
//
// Cache contract: these pulls are the app's most expensive (~8 s cold) and
// the server holds its own 1 h TTL, so the client mirrors it - staleTime 30
// min (no silent background refetch churn) and gcTime 1 h so the data
// SURVIVES navigating to other pages and back within a session. TanStack's
// default gcTime of 5 min was evicting the cache on any longer detour,
// forcing a from-scratch reload on return. The page's Refresh button
// invalidates the ["well-sort"] prefix for an on-demand refetch.
// ---------------------------------------------------------------------------

/** ?pops_pad=E&pops_pad=S...&force_true=MPS-08... - FastAPI list params.
 * An explicitly empty pads selection is a valid state and must reach the
 * server (else it falls back to the field defaults), hence the marker. */
export function popsQuery(popsPads: string[], forceTrue: string[]): string {
  const parts = popsPads.map((p) => `pops_pad=${encodeURIComponent(p)}`);
  if (popsPads.length === 0) parts.push("pops_pad=");
  for (const w of forceTrue) parts.push(`force_true=${encodeURIComponent(w)}`);
  return parts.join("&");
}

export const useWellSortTables = (
  mode: WellSortMode,
  staleDays: number,
  popsPads: string[],
  forceTrue: string[],
) =>
  useQuery({
    queryKey: ["well-sort", "tables", mode, staleDays, popsPads, forceTrue],
    queryFn: ({ signal }) =>
      get<WellSortTablesResponse>(
        `/well-sort/tables?mode=${mode}&stale_days=${staleDays}&${popsQuery(popsPads, forceTrue)}`,
        signal,
      ),
    staleTime: MIN_30,
    gcTime: HOUR_1,
    placeholderData: keepPreviousData,
  });

export const useWellSortEvents = (windowDays: number, downHours: number) =>
  useQuery({
    queryKey: ["well-sort", "events", windowDays, downHours],
    queryFn: ({ signal }) =>
      get<WellSortEventsResponse>(
        `/well-sort/events?window_days=${windowDays}&down_hours=${downHours}`,
        signal,
      ),
    staleTime: MIN_30,
    gcTime: HOUR_1,
    placeholderData: keepPreviousData,
  });

export const useMarginalWc = (
  thresholdPct: number,
  staleDays: number,
  popsPads: string[],
  forceTrue: string[],
) =>
  useQuery({
    queryKey: ["well-sort", "marginal", thresholdPct, staleDays, popsPads, forceTrue],
    queryFn: ({ signal }) =>
      get<MarginalWcResponse>(
        `/well-sort/marginal-wc?threshold_pct=${thresholdPct}&stale_days=${staleDays}&${popsQuery(popsPads, forceTrue)}`,
        signal,
      ),
    staleTime: MIN_30,
    gcTime: HOUR_1,
    placeholderData: keepPreviousData,
  });

export const usePadMarginalWc = (
  pad: string | null,
  pumpLimit: number,
  staleDays: number,
  popsPads: string[],
  forceTrue: string[],
) =>
  useQuery({
    queryKey: ["well-sort", "pad-marginal", pad, pumpLimit, staleDays, popsPads, forceTrue],
    queryFn: ({ signal }) =>
      get<PadMarginalWcResponse>(
        `/well-sort/pad-marginal-wc?pad=${encodeURIComponent(pad!)}&pump_limit=${pumpLimit}&stale_days=${staleDays}&${popsQuery(popsPads, forceTrue)}`,
        signal,
      ),
    enabled: pad !== null,
    staleTime: MIN_30,
    gcTime: HOUR_1,
    placeholderData: keepPreviousData,
  });

export const useTriage = (
  thresholdPct: number,
  staleDays: number,
  popsPads: string[],
  forceTrue: string[],
) =>
  useQuery({
    queryKey: ["well-sort", "triage", thresholdPct, staleDays, popsPads, forceTrue],
    queryFn: ({ signal }) =>
      get<TriageResponse>(
        `/well-sort/triage?threshold_pct=${thresholdPct}&stale_days=${staleDays}&${popsQuery(popsPads, forceTrue)}`,
        signal,
      ),
    staleTime: MIN_30,
    gcTime: HOUR_1,
    placeholderData: keepPreviousData,
  });
