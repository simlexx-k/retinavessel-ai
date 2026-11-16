import { NextResponse } from "next/server";

import { backendFetch, upstreamError } from "@/lib/backend";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const response = await backendFetch("/health/ready");
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
      { detail: "Inference service is unavailable" },
      { status: 503 },
    );
  }
}
