// @vitest-environment node
import { NextRequest } from "next/server";
import { afterEach, describe, expect, it, vi } from "vitest";

import { POST } from "./route";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe("segment proxy route", () => {
  it("rejects malformed model identifiers before calling the backend", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const request = new NextRequest("http://localhost/api/segment?model_id=BAD!", {
      method: "POST",
      body: new FormData(),
    });
    const response = await POST(request);
    expect(response.status).toBe(422);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("forwards a valid image and preserves inference metadata", async () => {
    vi.stubEnv("RETINAVESSEL_API_URL", "http://api:8000");
    const upstream = new Response(new Uint8Array([137, 80, 78, 71]), {
      status: 200,
      headers: {
        "Content-Type": "image/png",
        "X-Inference-Ms": "123.4",
        "X-Model-ID": "lite",
      },
    });
    const fetchMock = vi.fn().mockResolvedValue(upstream);
    vi.stubGlobal("fetch", fetchMock);
    const form = new FormData();
    form.set("file", new File(["fixture"], "retina.png", { type: "image/png" }));
    const request = new NextRequest(
      "http://localhost/api/segment?model_id=lite&output=overlay&threshold=0.5",
      { method: "POST", body: form },
    );
    const response = await POST(request);
    expect(response.status).toBe(200);
    expect(response.headers.get("x-inference-ms")).toBe("123.4");
    expect(response.headers.get("x-model-id")).toBe("lite");
    expect(fetchMock).toHaveBeenCalledOnce();
  });
});
