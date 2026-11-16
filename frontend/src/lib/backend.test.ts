import { afterEach, describe, expect, it, vi } from "vitest";

import { backendFetch, backendUrl, upstreamError } from "./backend";

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

describe("backend proxy helpers", () => {
  it("builds an internal URL and adds the private API key", async () => {
    vi.stubEnv("RETINAVESSEL_API_URL", "http://api:8000/");
    vi.stubEnv("RETINAVESSEL_API_KEY", "private-fixture-key");
    const fetchMock = vi.fn().mockResolvedValue(new Response("ok"));
    vi.stubGlobal("fetch", fetchMock);

    await backendFetch("/v1/models", { headers: { Accept: "application/json" } });

    expect(backendUrl("/health/ready")).toBe("http://api:8000/health/ready");
    expect(fetchMock).toHaveBeenCalledOnce();
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("http://api:8000/v1/models");
    expect(new Headers(init.headers).get("X-API-Key")).toBe("private-fixture-key");
    expect(init.cache).toBe("no-store");
  });

  it("extracts a FastAPI detail message", async () => {
    const response = Response.json({ detail: "fixture error" }, { status: 422 });
    await expect(upstreamError(response)).resolves.toBe("fixture error");
  });
});
