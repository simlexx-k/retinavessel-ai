"use client";
/* eslint-disable @next/next/no-img-element -- local object URLs cannot use next/image */

import {
  Columns2,
  Download,
  Eye,
  ImageIcon,
  Maximize2,
  RotateCcw,
  ScanLine,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { ReactCompareSlider, ReactCompareSliderHandle, ReactCompareSliderImage } from "react-compare-slider";
import { Tabs, Tooltip } from "radix-ui";
import { useEffect, useState } from "react";

import { RetinaMark } from "@/components/retina-mark";
import type { OutputMode } from "@/components/analysis-controls";
import type { InferenceMetadata, ModelMetadata } from "@/lib/types";

type ViewerMode = "compare" | "source" | "result";

type InferenceViewerProps = {
  sourceUrl: string | null;
  resultUrl: string | null;
  sourceName: string | null;
  resultMode: OutputMode | null;
  metadata: InferenceMetadata | null;
  model: ModelMetadata | null;
  running: boolean;
};

export function InferenceViewer(props: InferenceViewerProps) {
  const [view, setView] = useState<ViewerMode>("source");
  const [zoom, setZoom] = useState(1);

  useEffect(() => {
    setView(props.resultUrl ? "compare" : "source");
    setZoom(1);
  }, [props.resultUrl, props.sourceUrl]);

  const resultLabel = props.resultMode === "mask" ? "Binary mask" : "Vessel overlay";
  const downloadName = `${stripExtension(props.sourceName || "retina")}_${props.resultMode || "segmentation"}.png`;

  return (
    <section className="viewer-shell" aria-label="Image inspection workspace">
      <div className="viewer-toolbar">
        <div className="viewer-title">
          <p className="overline">Inspection viewport</p>
          <h2>{props.sourceName || "No study loaded"}</h2>
        </div>
        <Tabs.Root value={view} onValueChange={(value) => setView(value as ViewerMode)}>
          <Tabs.List className="viewer-tabs" aria-label="Viewport display">
            <Tabs.Trigger value="source"><ImageIcon size={14} /> Original</Tabs.Trigger>
            <Tabs.Trigger value="compare" disabled={!props.resultUrl}><Columns2 size={14} /> Compare</Tabs.Trigger>
            <Tabs.Trigger value="result" disabled={!props.resultUrl}><Eye size={14} /> Result</Tabs.Trigger>
          </Tabs.List>
        </Tabs.Root>
        <div className="viewer-tools" aria-label="Viewport controls">
          <ToolButton label="Zoom out" disabled={zoom <= 0.75} onClick={() => setZoom((value) => Math.max(0.75, value - 0.25))}><ZoomOut size={16} /></ToolButton>
          <span className="zoom-value">{Math.round(zoom * 100)}%</span>
          <ToolButton label="Zoom in" disabled={zoom >= 2} onClick={() => setZoom((value) => Math.min(2, value + 0.25))}><ZoomIn size={16} /></ToolButton>
          <ToolButton label="Fit image" onClick={() => setZoom(1)}><Maximize2 size={16} /></ToolButton>
          <ToolButton label="Reset viewport" onClick={() => { setZoom(1); setView(props.resultUrl ? "compare" : "source"); }}><RotateCcw size={16} /></ToolButton>
        </div>
      </div>

      <div className="viewport-frame">
        <div className="viewport-corner top-left" aria-hidden="true" />
        <div className="viewport-corner top-right" aria-hidden="true" />
        <div className="viewport-corner bottom-left" aria-hidden="true" />
        <div className="viewport-corner bottom-right" aria-hidden="true" />

        {!props.sourceUrl ? (
          <EmptyViewport />
        ) : (
          <motion.div
            className="viewport-content"
            animate={{ scale: zoom }}
            transition={{ type: "spring", stiffness: 260, damping: 28 }}
          >
            {view === "source" && <img src={props.sourceUrl} alt="Original fundus photograph" />}
            {view === "result" && props.resultUrl && <img src={props.resultUrl} alt={`${resultLabel} result`} />}
            {view === "compare" && props.resultUrl && (
              <ReactCompareSlider
                className="compare-slider"
                itemOne={<ReactCompareSliderImage src={props.sourceUrl} alt="Original fundus photograph" />}
                itemTwo={<ReactCompareSliderImage src={props.resultUrl} alt={`${resultLabel} result`} />}
                handle={<ReactCompareSliderHandle buttonStyle={{ backdropFilter: "blur(6px)", background: "rgba(8, 21, 27, .82)", border: "1px solid rgba(255,255,255,.35)" }} />}
                defaultPosition={50}
              />
            )}
          </motion.div>
        )}

        <AnimatePresence>
          {props.running && (
            <motion.div className="processing-layer" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <span className="processing-reticle"><ScanLine size={25} /></span>
              <strong>Tracing vascular network</strong>
              <small>Processing overlapping native-resolution tiles</small>
              <span className="processing-track"><motion.i initial={{ x: "-100%" }} animate={{ x: "260%" }} transition={{ duration: 1.6, repeat: Infinity, ease: "easeInOut" }} /></span>
            </motion.div>
          )}
        </AnimatePresence>

        {props.sourceUrl && view !== "result" && <div className="viewport-label source-label">SOURCE</div>}
        {props.resultUrl && view !== "source" && <div className="viewport-label result-label">{resultLabel.toUpperCase()}</div>}
      </div>

      <div className="viewer-statusbar">
        <div className="status-metrics">
          <Metric label="Resolution" value={formatResolution(props.metadata)} />
          <Metric label="Inference" value={formatInference(props.metadata)} />
          <Metric label="Threshold" value={props.metadata?.threshold?.toFixed(2) ?? "—"} />
          <Metric label="Model" value={props.metadata?.modelId || props.model?.id || "—"} wide />
        </div>
        {props.resultUrl ? (
          <a className="download-result" href={props.resultUrl} download={downloadName}>
            <Download size={16} /> Download PNG
          </a>
        ) : (
          <span className="awaiting-result">Awaiting segmentation result</span>
        )}
      </div>
    </section>
  );
}

function EmptyViewport() {
  return (
    <div className="empty-viewport">
      <span className="empty-retina"><RetinaMark /></span>
      <div><strong>Load a fundus photograph</strong><p>The source image and segmentation result will remain aligned here for close inspection.</p></div>
      <span className="supported-formats">PNG · JPEG · WEBP · TIFF · BMP</span>
    </div>
  );
}

function ToolButton({ label, children, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement> & { label: string }) {
  return (
    <Tooltip.Root>
      <Tooltip.Trigger asChild><button className="viewport-tool" type="button" aria-label={label} {...props}>{children}</button></Tooltip.Trigger>
      <Tooltip.Portal><Tooltip.Content className="tooltip-content" sideOffset={7}>{label}<Tooltip.Arrow className="tooltip-arrow" /></Tooltip.Content></Tooltip.Portal>
    </Tooltip.Root>
  );
}

function Metric({ label, value, wide = false }: { label: string; value: string; wide?: boolean }) {
  return <div className={`status-metric ${wide ? "is-wide" : ""}`}><span>{label}</span><strong title={value}>{value}</strong></div>;
}

function formatResolution(metadata: InferenceMetadata | null): string {
  return metadata?.width && metadata.height ? `${metadata.width} × ${metadata.height}` : "—";
}

function formatInference(metadata: InferenceMetadata | null): string {
  return metadata?.inferenceMs === null || metadata?.inferenceMs === undefined ? "—" : `${(metadata.inferenceMs / 1000).toFixed(2)} s`;
}

function stripExtension(filename: string): string {
  return filename.replace(/\.[^.]+$/, "");
}
