// @vitest-environment jsdom
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { SegmentationWorkspace } from "./segmentation-workspace";

const health = {
  status: "ready",
  model_count: 1,
  default_model: "lite",
  device: "cpu",
  research_only: true,
};
const registry = {
  default_model: "lite",
  models: [{
    id: "lite",
    description: "Lightweight research candidate",
    architecture: "lite_unet",
    parameters: 236156,
    tile: 384,
    overlap: 64,
    threshold: 0.5,
    checkpoint_sha256: "a".repeat(64),
    validation_dice: 0.8452,
    validation_epoch: 10,
  }],
};

beforeEach(() => {
  const NativeURL = URL;
  class MockURL extends NativeURL {
    static createObjectURL = vi.fn(() => "blob:fixture");
    static revokeObjectURL = vi.fn();
  }
  vi.stubGlobal("URL", MockURL);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("SegmentationWorkspace", () => {
  it("loads the registry, uploads an image, and renders inference metadata", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path === "/api/health") return Response.json(health);
      if (path === "/api/models") return Response.json(registry);
      if (path.startsWith("/api/segment?")) {
        return new Response(new Blob(["png"], { type: "image/png" }), {
          headers: {
            "Content-Type": "image/png",
            "X-Inference-Ms": "1250",
            "X-Image-Width": "2048",
            "X-Image-Height": "2048",
            "X-Model-ID": "lite",
            "X-Threshold": "0.5",
          },
        });
      }
      return Response.json({ detail: "not found" }, { status: 404 });
    });
    vi.stubGlobal("fetch", fetchMock);
    const { container } = render(<SegmentationWorkspace />);

    expect(await screen.findByText("CPU inference ready")).toBeInTheDocument();
    expect(screen.getByText("Val Dice 0.8452")).toBeInTheDocument();

    const input = container.querySelector<HTMLInputElement>('input[type="file"]');
    expect(input).not.toBeNull();
    const file = new File(["fixture"], "retina.png", { type: "image/png" });
    fireEvent.change(input!, { target: { files: [file] } });
    fireEvent.click(screen.getByRole("button", { name: /run vessel segmentation/i }));

    expect(await screen.findByText("1.25 s")).toBeInTheDocument();
    expect(screen.getByText("2048 × 2048")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /download png/i })).toHaveAttribute(
      "download",
      "retina_overlay.png",
    );
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(3));
  }, 15_000);
});
