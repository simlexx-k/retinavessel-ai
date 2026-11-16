const DEFAULT_API_URL = "http://127.0.0.1:8000";

export function backendUrl(path: string): string {
  const base = (process.env.RETINAVESSEL_API_URL || DEFAULT_API_URL).replace(/\/$/, "");
  return `${base}${path.startsWith("/") ? path : `/${path}`}`;
}

export function backendHeaders(extra?: HeadersInit): Headers {
  const headers = new Headers(extra);
  const apiKey = process.env.RETINAVESSEL_API_KEY;
  if (apiKey) {
    headers.set("X-API-Key", apiKey);
  }
  return headers;
}

export async function backendFetch(path: string, init: RequestInit = {}): Promise<Response> {
  return fetch(backendUrl(path), {
    ...init,
    cache: "no-store",
    headers: backendHeaders(init.headers),
  });
}

export async function upstreamError(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: unknown };
    if (typeof payload.detail === "string") return payload.detail;
  } catch {
    // Fall through to the stable message below.
  }
  return `Inference service returned ${response.status}`;
}
