import { NextRequest, NextResponse } from "next/server";

import { backendFetch, upstreamError } from "@/lib/backend";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const MODEL_ID = /^[a-z0-9][a-z0-9_-]{0,63}$/;
const ALLOWED_TYPES = new Set([
  "image/png",
  "image/jpeg",
  "image/webp",
  "image/tiff",
  "image/bmp",
]);
const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;
const FORWARDED_HEADERS = [
  "content-disposition",
  "content-type",
  "x-image-height",
  "x-image-width",
  "x-inference-ms",
  "x-model-id",
  "x-research-only",
  "x-threshold",
];

function numberParameter(
  value: string | null,
  fallback: number | null,
  minimum: number,
  maximum: number,
): number | null {
  if (value === null || value === "") return fallback;
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed >= minimum && parsed <= maximum
    ? parsed
    : null;
}

export async function POST(request: NextRequest) {
  const modelId = request.nextUrl.searchParams.get("model_id") || "";
  const output = request.nextUrl.searchParams.get("output") || "overlay";
  const thresholdRaw = request.nextUrl.searchParams.get("threshold");
  const alphaRaw = request.nextUrl.searchParams.get("overlay_alpha");
  const threshold = numberParameter(thresholdRaw, null, 0, 1);
  const overlayAlpha = numberParameter(alphaRaw, 0.45, 0, 1);

  if (!MODEL_ID.test(modelId)) {
    return NextResponse.json({ detail: "Invalid model ID" }, { status: 422 });
  }
  if (!new Set(["mask", "overlay"]).has(output)) {
    return NextResponse.json({ detail: "Invalid output type" }, { status: 422 });
  }
  if ((thresholdRaw !== null && threshold === null) || overlayAlpha === null) {
    return NextResponse.json(
      { detail: "Threshold and overlay alpha must be between zero and one" },
      { status: 422 },
    );
  }

  let formData: FormData;
  try {
    formData = await request.formData();
  } catch {
    return NextResponse.json({ detail: "Invalid multipart request" }, { status: 400 });
  }
  const file = formData.get("file");
  if (!(file instanceof File) || file.size === 0) {
    return NextResponse.json({ detail: "Select a fundus image" }, { status: 422 });
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return NextResponse.json({ detail: "Image exceeds the 25 MB limit" }, { status: 413 });
  }
  if (file.type && !ALLOWED_TYPES.has(file.type)) {
    return NextResponse.json({ detail: "Unsupported image format" }, { status: 415 });
  }

  const upstreamForm = new FormData();
  upstreamForm.set("file", file, file.name);
  const query = new URLSearchParams({ output, overlay_alpha: String(overlayAlpha) });
  if (threshold !== null) query.set("threshold", String(threshold));

  try {
    const response = await backendFetch(
      `/v1/models/${encodeURIComponent(modelId)}/segment?${query}`,
      { method: "POST", body: upstreamForm },
    );
    if (!response.ok) {
      return NextResponse.json(
        { detail: await upstreamError(response) },
        { status: response.status },
      );
    }
    const headers = new Headers({ "Cache-Control": "no-store" });
    for (const name of FORWARDED_HEADERS) {
      const value = response.headers.get(name);
      if (value) headers.set(name, value);
    }
    return new Response(response.body, { status: 200, headers });
  } catch {
    return NextResponse.json(
      { detail: "Inference service is unavailable" },
      { status: 503 },
    );
  }
}
