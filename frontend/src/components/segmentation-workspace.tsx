"use client";
/* eslint-disable @next/next/no-img-element -- blob URLs are local inference previews */

import { ChangeEvent, DragEvent, FormEvent, useEffect, useMemo, useRef, useState } from "react";

import { RetinaMark } from "@/components/retina-mark";
import type { HealthResponse, InferenceMetadata, ModelsResponse } from "@/lib/types";

const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;
const ACCEPTED_TYPES = new Set(["image/png", "image/jpeg", "image/webp", "image/tiff", "image/bmp"]);

type OutputMode = "overlay" | "mask";

function formatParameters(value: number): string {
  return new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 2 }).format(value);
}

function errorDetail(payload: unknown, fallback: string): string {
  if (payload && typeof payload === "object" && "detail" in payload) {
    const detail = (payload as { detail?: unknown }).detail;
    if (typeof detail === "string") return detail;
  }
  return fallback;
}

export function SegmentationWorkspace() {
  const [models, setModels] = useState<ModelsResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [modelId, setModelId] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [outputMode, setOutputMode] = useState<OutputMode>("overlay");
  const [threshold, setThreshold] = useState(0.5);
  const [overlayAlpha, setOverlayAlpha] = useState(0.45);
  const [metadata, setMetadata] = useState<InferenceMetadata | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let cancelled = false;
    async function loadService() {
      try {
        const [healthResponse, modelsResponse] = await Promise.all([
          fetch("/api/health", { cache: "no-store" }),
          fetch("/api/models", { cache: "no-store" }),
        ]);
        const healthPayload = (await healthResponse.json()) as unknown;
        const modelsPayload = (await modelsResponse.json()) as unknown;
        if (!healthResponse.ok || !modelsResponse.ok) {
          throw new Error(errorDetail(modelsPayload, errorDetail(healthPayload, "Service unavailable")));
        }
        if (!cancelled) {
          const registry = modelsPayload as ModelsResponse;
          setHealth(healthPayload as HealthResponse);
          setModels(registry);
          setModelId(registry.default_model);
        }
      } catch (caught) {
        if (!cancelled) setError(caught instanceof Error ? caught.message : "Service unavailable");
      }
    }
    void loadService();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      if (resultUrl) URL.revokeObjectURL(resultUrl);
    };
  }, [previewUrl, resultUrl]);

  const selectedModel = useMemo(
    () => models?.models.find((model) => model.id === modelId) ?? null,
    [modelId, models],
  );

  function chooseFile(candidate: File | null) {
    setError(null);
    setMetadata(null);
    if (resultUrl) URL.revokeObjectURL(resultUrl);
    setResultUrl(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setFile(null);
    if (!candidate) return;
    if (candidate.size > MAX_UPLOAD_BYTES) {
      setError("Image exceeds the 25 MB upload limit.");
      return;
    }
    if (candidate.type && !ACCEPTED_TYPES.has(candidate.type)) {
      setError("Use PNG, JPEG, WebP, TIFF, or BMP fundus imagery.");
      return;
    }
    setFile(candidate);
    setPreviewUrl(URL.createObjectURL(candidate));
  }

  function handleFileInput(event: ChangeEvent<HTMLInputElement>) {
    chooseFile(event.target.files?.[0] ?? null);
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragActive(false);
    chooseFile(event.dataTransfer.files?.[0] ?? null);
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!file || !modelId) {
      setError("Select a fundus image and model first.");
      return;
    }
    setLoading(true);
    setError(null);
    setMetadata(null);
    if (resultUrl) URL.revokeObjectURL(resultUrl);
    setResultUrl(null);
    const body = new FormData();
    body.set("file", file, file.name);
    const query = new URLSearchParams({
      model_id: modelId,
      output: outputMode,
      threshold: threshold.toFixed(2),
      overlay_alpha: overlayAlpha.toFixed(2),
    });
    try {
      const response = await fetch(`/api/segment?${query}`, { method: "POST", body });
      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as unknown;
        throw new Error(errorDetail(payload, `Segmentation failed (${response.status})`));
      }
      const blob = await response.blob();
      setResultUrl(URL.createObjectURL(blob));
      setMetadata({
        modelId: response.headers.get("x-model-id") || modelId,
        inferenceMs: numberHeader(response.headers.get("x-inference-ms")),
        width: numberHeader(response.headers.get("x-image-width")),
        height: numberHeader(response.headers.get("x-image-height")),
        threshold: numberHeader(response.headers.get("x-threshold")),
      });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Segmentation failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#workspace" aria-label="RetinaVessel AI home">
          <RetinaMark />
          <span>
            <strong>RetinaVessel</strong>
            <small>AI research workspace</small>
          </span>
        </a>
        <div className={`service-status ${health ? "online" : "offline"}`} role="status">
          <span aria-hidden="true" />
          {health ? `${health.device.toUpperCase()} inference ready` : "Connecting to inference"}
        </div>
      </header>

      <section className="hero">
        <div>
          <p className="eyebrow">Vessel segmentation · native resolution</p>
          <h1>Trace the retinal vascular network.</h1>
          <p className="hero-copy">
            Upload a fundus photograph to generate a research-only vessel mask or transparent overlay using a verified lightweight U-Net checkpoint.
          </p>
        </div>
        <aside className="research-notice">
          <span>Research use only</span>
          <p>Not for diagnosis, triage, treatment, or clinical decision-making.</p>
        </aside>
      </section>

      <section className="workspace" id="workspace">
        <form className="control-panel" onSubmit={submit}>
          <div className="section-heading">
            <span>01</span>
            <div><h2>Configure analysis</h2><p>Choose an image, model, and rendering mode.</p></div>
          </div>

          <div
            className={`dropzone ${dragActive ? "drag-active" : ""} ${file ? "has-file" : ""}`}
            onClick={() => inputRef.current?.click()}
            onDragEnter={(event) => { event.preventDefault(); setDragActive(true); }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={() => setDragActive(false)}
            onDrop={handleDrop}
            role="button"
            tabIndex={0}
            onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") inputRef.current?.click(); }}
          >
            <input ref={inputRef} hidden type="file" accept="image/png,image/jpeg,image/webp,image/tiff,image/bmp" onChange={handleFileInput} />
            <div className="upload-icon" aria-hidden="true">↥</div>
            {file ? <><strong>{file.name}</strong><span>{(file.size / 1024 / 1024).toFixed(2)} MB · click to replace</span></> : <><strong>Drop a fundus image here</strong><span>or browse · PNG, JPEG, WebP, TIFF, BMP · max 25 MB</span></>}
          </div>

          <label className="field-label" htmlFor="model">Model checkpoint</label>
          <select id="model" value={modelId} onChange={(event) => setModelId(event.target.value)} disabled={!models}>
            {!models && <option>Loading registry…</option>}
            {models?.models.map((model) => <option key={model.id} value={model.id}>{model.description || model.id}</option>)}
          </select>
          {selectedModel && (
            <div className="model-facts">
              <span>{selectedModel.architecture.replace("_", " ")}</span>
              <span>{formatParameters(selectedModel.parameters)} params</span>
              <span>Val Dice {selectedModel.validation_dice?.toFixed(4) ?? "—"}</span>
            </div>
          )}

          <fieldset className="mode-toggle">
            <legend>Result rendering</legend>
            <button type="button" className={outputMode === "overlay" ? "selected" : ""} onClick={() => setOutputMode("overlay")}>Overlay</button>
            <button type="button" className={outputMode === "mask" ? "selected" : ""} onClick={() => setOutputMode("mask")}>Binary mask</button>
          </fieldset>

          <div className="range-field">
            <label htmlFor="threshold"><span>Decision threshold</span><output>{threshold.toFixed(2)}</output></label>
            <input id="threshold" type="range" min="0" max="1" step="0.01" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} />
          </div>
          {outputMode === "overlay" && (
            <div className="range-field">
              <label htmlFor="alpha"><span>Overlay strength</span><output>{Math.round(overlayAlpha * 100)}%</output></label>
              <input id="alpha" type="range" min="0" max="1" step="0.05" value={overlayAlpha} onChange={(event) => setOverlayAlpha(Number(event.target.value))} />
            </div>
          )}

          {error && <div className="error-message" role="alert">{error}</div>}
          <button className="run-button" type="submit" disabled={loading || !file || !modelId}>
            {loading ? <><span className="spinner" /> Segmenting at native resolution…</> : <>Run vessel segmentation <span>→</span></>}
          </button>
        </form>

        <section className="viewer-panel">
          <div className="section-heading">
            <span>02</span>
            <div><h2>Inspect output</h2><p>Compare the source photograph with the generated vessel result.</p></div>
          </div>
          <div className="image-grid">
            <Figure title="Original" src={previewUrl} empty="Your source image will appear here." />
            <Figure title={outputMode === "overlay" ? "Vessel overlay" : "Vessel mask"} src={resultUrl} empty={loading ? "The model is processing overlapping image tiles…" : "Run segmentation to display the result."} loading={loading} />
          </div>
          {metadata && resultUrl && (
            <div className="result-footer">
              <div className="metric-strip">
                <Metric label="Inference" value={metadata.inferenceMs === null ? "—" : `${(metadata.inferenceMs / 1000).toFixed(2)} s`} />
                <Metric label="Resolution" value={metadata.width && metadata.height ? `${metadata.width} × ${metadata.height}` : "—"} />
                <Metric label="Threshold" value={metadata.threshold?.toFixed(2) ?? "—"} />
              </div>
              <a className="download-button" href={resultUrl} download={`${file?.name.replace(/\.[^.]+$/, "") || "retina"}_${outputMode}.png`}>Download PNG ↓</a>
            </div>
          )}
        </section>
      </section>

      <footer>
        <p>RetinaVessel AI · reproducible retinal vessel segmentation research prototype</p>
        <p>Model outputs require expert interpretation and independent validation.</p>
      </footer>
    </main>
  );
}

function numberHeader(value: string | null): number | null {
  if (value === null) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function Figure({ title, src, empty, loading = false }: { title: string; src: string | null; empty: string; loading?: boolean }) {
  return (
    <figure className={`image-card ${src ? "has-image" : ""}`}>
      <figcaption>{title}</figcaption>
      <div className="image-stage">
        {src ? <img src={src} alt={`${title} fundus visualization`} /> : <div className="empty-stage"><RetinaMark />{loading && <span className="scan-line" />}<p>{empty}</p></div>}
      </div>
    </figure>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return <div><span>{label}</span><strong>{value}</strong></div>;
}
