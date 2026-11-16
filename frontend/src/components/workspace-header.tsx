import { Activity, CircleHelp, ShieldAlert } from "lucide-react";
import { Tooltip } from "radix-ui";

import { RetinaMark } from "@/components/retina-mark";
import type { HealthResponse } from "@/lib/types";

export function WorkspaceHeader({ health, pending }: { health?: HealthResponse; pending: boolean }) {
  const ready = health?.status === "ready";

  return (
    <header className="workspace-header">
      <div className="product-identity">
        <span className="product-mark"><RetinaMark /></span>
        <span className="product-copy">
          <strong>RetinaVessel</strong>
          <small>Segmentation workspace</small>
        </span>
        <span className="version-chip">RESEARCH</span>
      </div>

      <div className="header-actions">
        <div className="research-flag">
          <ShieldAlert size={15} aria-hidden="true" />
          <span>Not for clinical use</span>
        </div>
        <Tooltip.Root>
          <Tooltip.Trigger asChild>
            <button className="icon-button subtle" type="button" aria-label="About this workspace">
              <CircleHelp size={17} />
            </button>
          </Tooltip.Trigger>
          <Tooltip.Portal>
            <Tooltip.Content className="tooltip-content" sideOffset={8}>
              This prototype segments vessels only. It does not diagnose eye disease.
              <Tooltip.Arrow className="tooltip-arrow" />
            </Tooltip.Content>
          </Tooltip.Portal>
        </Tooltip.Root>
        <div className={`backend-status ${ready ? "is-ready" : ""}`} role="status">
          <span className="status-pulse" aria-hidden="true" />
          <Activity size={15} aria-hidden="true" />
          <span>
            <strong>{ready ? "Inference ready" : pending ? "Connecting" : "Unavailable"}</strong>
            <small>{ready ? `${health.device.toUpperCase()} · ${health.model_count} model${health.model_count === 1 ? "" : "s"}` : "FastAPI service"}</small>
          </span>
        </div>
      </div>
    </header>
  );
}
