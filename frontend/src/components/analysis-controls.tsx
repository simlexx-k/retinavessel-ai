"use client";

import {
  Check,
  ChevronDown,
  FileImage,
  Layers3,
  ScanLine,
  Sparkles,
  UploadCloud,
} from "lucide-react";
import { useDropzone } from "react-dropzone";
import { Select, Slider, ToggleGroup } from "radix-ui";

import type { ModelMetadata, ModelsResponse } from "@/lib/types";

export type OutputMode = "overlay" | "mask";

type AnalysisControlsProps = {
  file: File | null;
  models?: ModelsResponse;
  modelId: string;
  outputMode: OutputMode;
  threshold: number;
  overlayAlpha: number;
  running: boolean;
  serviceReady: boolean;
  error: string | null;
  onFile: (file: File | null, validationError?: string) => void;
  onModel: (modelId: string) => void;
  onOutputMode: (mode: OutputMode) => void;
  onThreshold: (value: number) => void;
  onOverlayAlpha: (value: number) => void;
  onSubmit: () => void;
};

const ACCEPTED_IMAGES = {
  "image/png": [".png"],
  "image/jpeg": [".jpg", ".jpeg"],
  "image/webp": [".webp"],
  "image/tiff": [".tif", ".tiff"],
  "image/bmp": [".bmp"],
};

export function AnalysisControls(props: AnalysisControlsProps) {
  const selectedModel = props.models?.models.find((model) => model.id === props.modelId);
  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    accept: ACCEPTED_IMAGES,
    maxFiles: 1,
    maxSize: 25 * 1024 * 1024,
    multiple: false,
    noClick: true,
    onDropAccepted: ([file]) => props.onFile(file ?? null),
    onDropRejected: ([rejection]) => {
      const code = rejection?.errors[0]?.code;
      props.onFile(
        null,
        code === "file-too-large"
          ? "Image exceeds the 25 MB upload limit."
          : "Use a PNG, JPEG, WebP, TIFF, or BMP fundus image.",
      );
    },
  });

  return (
    <aside className="control-rail" aria-label="Analysis controls">
      <div className="rail-heading">
        <div>
          <p className="overline">Analysis setup</p>
          <h1>Vessel segmentation</h1>
        </div>
        <ScanLine size={20} aria-hidden="true" />
      </div>

      <section className="control-section upload-section" aria-labelledby="image-input-heading">
        <div className="section-label">
          <span>01</span>
          <div><h2 id="image-input-heading">Fundus image</h2><p>Source photograph</p></div>
        </div>
        <div
          {...getRootProps({
            className: `file-dropzone ${isDragActive ? "is-dragging" : ""} ${props.file ? "has-file" : ""}`,
          })}
        >
          <input {...getInputProps()} />
          {props.file ? (
            <>
              <span className="file-icon"><FileImage size={21} /></span>
              <span className="file-details">
                <strong>{props.file.name}</strong>
                <small>{formatBytes(props.file.size)} · local image</small>
              </span>
              <button type="button" className="replace-file" onClick={open}>Replace</button>
            </>
          ) : (
            <>
              <span className="upload-glyph"><UploadCloud size={23} /></span>
              <span className="drop-copy"><strong>{isDragActive ? "Release to load image" : "Drop fundus image"}</strong><small>PNG, JPEG, WebP, TIFF or BMP · 25 MB max</small></span>
              <button type="button" className="browse-button" onClick={open}>Browse files</button>
            </>
          )}
        </div>
      </section>

      <section className="control-section" aria-labelledby="model-heading">
        <div className="section-label">
          <span>02</span>
          <div><h2 id="model-heading">Model checkpoint</h2><p>Inference candidate</p></div>
        </div>
        <Select.Root value={props.modelId} onValueChange={props.onModel} disabled={!props.models}>
          <Select.Trigger className="select-trigger" aria-label="Model checkpoint">
            <Select.Value placeholder="Loading model registry…" />
            <Select.Icon><ChevronDown size={16} /></Select.Icon>
          </Select.Trigger>
          <Select.Portal>
            <Select.Content className="select-content" position="popper" sideOffset={6}>
              <Select.Viewport>
                {props.models?.models.map((model) => (
                  <Select.Item className="select-item" value={model.id} key={model.id}>
                    <Select.ItemText>{model.description || model.id}</Select.ItemText>
                    <Select.ItemIndicator><Check size={15} /></Select.ItemIndicator>
                  </Select.Item>
                ))}
              </Select.Viewport>
            </Select.Content>
          </Select.Portal>
        </Select.Root>
        {selectedModel && <ModelSummary model={selectedModel} />}
      </section>

      <section className="control-section rendering-section" aria-labelledby="rendering-heading">
        <div className="section-label">
          <span>03</span>
          <div><h2 id="rendering-heading">Output rendering</h2><p>Mask presentation</p></div>
        </div>
        <ToggleGroup.Root
          className="mode-toggle"
          type="single"
          value={props.outputMode}
          onValueChange={(value) => value && props.onOutputMode(value as OutputMode)}
          aria-label="Output rendering"
        >
          <ToggleGroup.Item value="overlay"><Layers3 size={15} /> Overlay</ToggleGroup.Item>
          <ToggleGroup.Item value="mask"><ScanLine size={15} /> Binary mask</ToggleGroup.Item>
        </ToggleGroup.Root>

        <RangeControl
          label="Decision threshold"
          value={props.threshold}
          display={props.threshold.toFixed(2)}
          step={0.01}
          onChange={props.onThreshold}
        />
        {props.outputMode === "overlay" && (
          <RangeControl
            label="Overlay opacity"
            value={props.overlayAlpha}
            display={`${Math.round(props.overlayAlpha * 100)}%`}
            step={0.05}
            onChange={props.onOverlayAlpha}
          />
        )}
      </section>

      <div className="rail-footer">
        {props.error && <div className="inline-error" role="alert">{props.error}</div>}
        <button
          className="run-analysis"
          type="button"
          disabled={props.running || !props.file || !props.modelId || !props.serviceReady}
          onClick={props.onSubmit}
        >
          {props.running ? <><span className="button-spinner" /> Processing native resolution</> : <><Sparkles size={17} /> Run segmentation <span className="shortcut">⌘↵</span></>}
        </button>
        <p className="privacy-note">Images are sent only to your configured inference server.</p>
      </div>
    </aside>
  );
}

function ModelSummary({ model }: { model: ModelMetadata }) {
  return (
    <div className="model-summary">
      <div className="model-summary-top">
        <span className="verified-mark"><Check size={12} /> Registered</span>
        <span className="architecture-chip">{model.architecture.replace("_", "-")}</span>
      </div>
      <dl>
        <div><dt>Parameters</dt><dd>{formatParameters(model.parameters)}</dd></div>
        <div><dt>Val. Dice</dt><dd>{model.validation_dice?.toFixed(4) ?? "—"}</dd></div>
        <div><dt>Tile / overlap</dt><dd>{model.tile} / {model.overlap}px</dd></div>
        <div><dt>Checkpoint</dt><dd title={model.checkpoint_sha256}>{model.checkpoint_sha256.slice(0, 8)}…</dd></div>
      </dl>
    </div>
  );
}

function RangeControl({ label, value, display, step, onChange }: { label: string; value: number; display: string; step: number; onChange: (value: number) => void }) {
  return (
    <div className="range-control">
      <div className="range-heading"><span>{label}</span><output>{display}</output></div>
      <Slider.Root className="slider-root" min={0} max={1} step={step} value={[value]} onValueChange={([next]) => onChange(next)} aria-label={label}>
        <Slider.Track className="slider-track"><Slider.Range className="slider-range" /></Slider.Track>
        <Slider.Thumb className="slider-thumb" />
      </Slider.Root>
    </div>
  );
}

function formatParameters(value: number): string {
  return new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 2 }).format(value);
}

function formatBytes(value: number): string {
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 / 1024).toFixed(2)} MB`;
}
