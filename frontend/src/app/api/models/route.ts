import { NextResponse } from "next/server";

import { backendFetch, upstreamError } from "@/lib/backend";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const response = await backendFetch("/v1/models");
    if (!response.ok) {
      return NextResponse.json(
        { detail: await upstreamError(response) },
        { status: response.status },
      );
    }
    return NextResponse.json(await response.json(), {
      headers: { "Cache-Control": "no-store" },
    });
  } catch {
    return NextResponse.json(
      { detail: "Could not load the model registry" },
      { status: 503 },
    );
  }
}
