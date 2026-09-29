import type { ApiErrorDetail } from "./types";

export class ApiError extends Error {
  readonly status: number;
  readonly detail: ApiErrorDetail;

  constructor(status: number, detail: ApiErrorDetail) {
    super(detail.message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

/** FastAPI 422 bodies carry ``detail`` as a list of {loc, msg}; pydantic
 *  prefixes our own validators' text with "Value error, ". One sentence an
 *  engineer can act on, or null when ``detail`` is not such a list. */
export function validationMessage(detail: unknown): string | null {
  if (!Array.isArray(detail) || detail.length === 0) return null;
  const parts = detail.map((d) => {
    const item = (d ?? {}) as { msg?: unknown; loc?: unknown };
    const raw = typeof item.msg === "string" ? item.msg : "invalid value";
    // Our validators already name the problem; range errors need the field.
    if (raw.startsWith("Value error, ")) return raw.slice("Value error, ".length);
    const loc = Array.isArray(item.loc) ? item.loc.filter((p) => p !== "body").join(".") : "";
    return loc ? `${loc}: ${raw}` : raw;
  });
  return [...new Set(parts)].join("; ");
}

async function parseError(res: Response): Promise<ApiErrorDetail> {
  try {
    const body = (await res.json()) as Record<string, unknown>;
    // FastAPI wraps HTTPException payloads in {detail: ...}; our handlers
    // return the detail object directly. Accept both.
    const detail = (body.detail ?? body) as Record<string, unknown>;
    if (typeof detail === "string") {
      return { error: "http", message: detail };
    }
    // Request validation (422): FastAPI sends a list of {loc, msg}. Without
    // this every rejected run read "HTTP 422" with no reason.
    const invalid = validationMessage(detail);
    if (invalid !== null) return { error: "invalid", message: invalid };
    return {
      error: (detail.error as ApiErrorDetail["error"]) ?? "http",
      message: (detail.message as string) ?? `HTTP ${res.status}`,
      suggested_gor: (detail.suggested_gor as number | null | undefined) ?? null,
    };
  } catch {
    return { error: "http", message: `HTTP ${res.status} ${res.statusText}` };
  }
}

/** Only a confirmed missing job invalidates its persisted handle. */
export const isMissingJob = (error: unknown): boolean =>
  error instanceof ApiError && (error.status === 404 || error.status === 410);

export const retryJobPoll = (failureCount: number, error: unknown): boolean =>
  !isMissingJob(error) && failureCount < 3;

export const jobPollDelay = (attempt: number): number => Math.min(1000 * 2 ** attempt, 15000);

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!res.ok) {
    throw new ApiError(res.status, await parseError(res));
  }
  return (await res.json()) as T;
}

export const get = <T>(path: string, signal?: AbortSignal): Promise<T> =>
  api<T>(path, { signal });

export const post = <T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> =>
  api<T>(path, { method: "POST", body: JSON.stringify(body), signal });

/** Multipart upload: NO explicit Content-Type so the browser sets the
 * boundary. Same error contract as api(). */
export async function upload<T>(path: string, form: FormData): Promise<T> {
  const res = await fetch(`/api${path}`, { method: "POST", body: form });
  if (!res.ok) {
    throw new ApiError(res.status, await parseError(res));
  }
  return (await res.json()) as T;
}

/** Deterministic JSON for query keys: object keys sorted recursively. */
export function stableStringify(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(",")}]`;
  const entries = Object.entries(value as Record<string, unknown>)
    .filter(([, v]) => v !== undefined)
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
    .map(([k, v]) => `${JSON.stringify(k)}:${stableStringify(v)}`);
  return `{${entries.join(",")}}`;
}
