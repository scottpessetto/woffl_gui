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

async function parseError(res: Response): Promise<ApiErrorDetail> {
  try {
    const body = (await res.json()) as Record<string, unknown>;
    // FastAPI wraps HTTPException payloads in {detail: ...}; our handlers
    // return the detail object directly. Accept both.
    const detail = (body.detail ?? body) as Record<string, unknown>;
    if (typeof detail === "string") {
      return { error: "http", message: detail };
    }
    return {
      error: (detail.error as ApiErrorDetail["error"]) ?? "http",
      message: (detail.message as string) ?? `HTTP ${res.status}`,
      suggested_gor: (detail.suggested_gor as number | null | undefined) ?? null,
    };
  } catch {
    return { error: "http", message: `HTTP ${res.status} ${res.statusText}` };
  }
}

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
