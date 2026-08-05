import { keepPreviousData, useQuery } from "@tanstack/react-query";

import { get, post, stableStringify } from "./client";
import type {
  AgingPumpsResponse,
  BatchResponse,
  EquivalentsResponse,
  IprFitRequest,
  IprFitResponse,
  IprPinResponse,
  JpHistoryResponse,
  MetaResponse,
  PfRangeResponse,
  PressureProfileResponse,
  PropHistoryResponse,
  SimParams,
  SolveResult,
  WellContext,
  WellDatabaseResponse,
  WellProfileResponse,
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

/**
 * Expensive sweeps run on explicit submit: pages snapshot the params into
 * local state and pass the snapshot here (null = not requested yet).
 */
export const useBatch = (well: string, snapshot: SimParams | null) =>
  useQuery({
    queryKey: ["batch", well, snapshot ? stableStringify(snapshot) : "none"],
    queryFn: ({ signal }) => post<BatchResponse>("/batch", { well, params: snapshot }, signal),
    enabled: snapshot !== null,
    staleTime: MIN_30,
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
