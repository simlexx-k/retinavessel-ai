"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import { motion } from "motion/react";
import { Tooltip } from "radix-ui";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import { AnalysisControls, type OutputMode } from "@/components/analysis-controls";
import { InferenceViewer } from "@/components/inference-viewer";
import { WorkspaceHeader } from "@/components/workspace-header";
import type { HealthResponse, InferenceMetadata, ModelsResponse } from "@/lib/types";

type InferenceRequest = {
  file: File;
  modelId: string;
  outputMode: OutputMode;
  threshold: number;
  overlayAlpha: number;
};

type InferenceResult = {
  blob: Blob;
  metadata: InferenceMetadata;
  mode: OutputMode;
};

export function SegmentationWorkspace() {
  const [modelId, setModelId] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [sourceUrl, setSourceUrl] = useState<string | null>(null);
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [resultMode, setResultMode] = useState<OutputMode | null>(null);
  const [outputMode, setOutputMode] = useState<OutputMode>("overlay");
  const [threshold, setThreshold] = useState(0.5);
  const [overlayAlpha, setOverlayAlpha] = useState(0.45);
  const [metadata, setMetadata] = useState<InferenceMetadata | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);

  const healthQuery = useQuery({
    queryKey: ["service-health"],
    queryFn: () => fetchJson<HealthResponse>("/api/health"),
    refetchInterval: 30_000,
  });
  const modelsQuery = useQuery({
    queryKey: ["model-registry"],
    queryFn: () => fetchJson<ModelsResponse>("/api/models"),
  });

  useEffect(() => {
    if (!modelId && modelsQuery.data?.default_model) setModelId(modelsQuery.data.default_model);
  }, [modelId, modelsQuery.data]);

  useEffect(() => () => { if (sourceUrl) URL.revokeObjectURL(sourceUrl); }, [sourceUrl]);
  useEffect(() => () => { if (resultUrl) URL.revokeObjectURL(resultUrl); }, [resultUrl]);

  const inference = useMutation({
    mutationFn: runInference,
    onSuccess: (result) => {
      setResultUrl(URL.createObjectURL(result.blob));
      setResultMode(result.mode);
      setMetadata(result.metadata);
      toast.success("Segmentation complete", {
        description: result.metadata.inferenceMs === null
          ? "The result is ready for inspection."
          : `Native-resolution inference completed in ${(result.metadata.inferenceMs / 1000).toFixed(2)} seconds.`,
      });
    },
    onError: (caught) => {
      const message = caught instanceof Error ? caught.message : "Segmentation failed";
      setLocalError(message);
      toast.error("Segmentation failed", { description: message });
    },
  });

  const selectedModel = useMemo(
    () => modelsQuery.data?.models.find((model) => model.id === modelId) ?? null,
    [modelId, modelsQuery.data],
  );
  const serviceReady = healthQuery.data?.status === "ready";
  const serviceError = healthQuery.error || modelsQuery.error;
  const displayedError = localError || (serviceError instanceof Error ? serviceError.message : null);

  function chooseFile(candidate: File | null, validationError?: string) {
    setLocalError(validationError ?? null);
    setFile(candidate);
    setSourceUrl(candidate ? URL.createObjectURL(candidate) : null);
    setResultUrl(null);
    setResultMode(null);
    setMetadata(null);
    inference.reset();
  }

  function submit() {
    if (!file || !modelId) {
      setLocalError("Select a fundus image and model first.");
      return;
    }
    setLocalError(null);
    setResultUrl(null);
    setResultMode(null);
    setMetadata(null);
    inference.mutate({ file, modelId, outputMode, threshold, overlayAlpha });
  }

  return (
    <Tooltip.Provider delayDuration={350}>
      <main className="application-shell">
        <WorkspaceHeader health={healthQuery.data} pending={healthQuery.isPending} />
        <div className="workspace-body">
          <motion.div className="controls-column" initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.35 }}>
            <AnalysisControls
              file={file}
              models={modelsQuery.data}
              modelId={modelId}
              outputMode={outputMode}
              threshold={threshold}
              overlayAlpha={overlayAlpha}
              running={inference.isPending}
              serviceReady={serviceReady}
              error={displayedError}
              onFile={chooseFile}
              onModel={setModelId}
              onOutputMode={setOutputMode}
              onThreshold={setThreshold}
              onOverlayAlpha={setOverlayAlpha}
              onSubmit={submit}
            />
          </motion.div>
          <motion.div className="viewer-column" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.05 }}>
            <InferenceViewer
              sourceUrl={sourceUrl}
              resultUrl={resultUrl}
              sourceName={file?.name ?? null}
              resultMode={resultMode}
              metadata={metadata}
              model={selectedModel}
              running={inference.isPending}
            />
          </motion.div>
        </div>
        <footer className="application-footer">
          <span>RetinaVessel AI · reproducible vessel segmentation prototype</span>
          <span>Outputs require expert interpretation and independent validation.</span>
        </footer>
      </main>
    </Tooltip.Provider>
  );
}

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(path, { cache: "no-store" });
  const payload = (await response.json().catch(() => null)) as unknown;
  if (!response.ok) throw new Error(errorDetail(payload, `Service request failed (${response.status})`));
  return payload as T;
}

async function runInference(request: InferenceRequest): Promise<InferenceResult> {
  const body = new FormData();
  body.set("file", request.file, request.file.name);
  const query = new URLSearchParams({
    model_id: request.modelId,
    output: request.outputMode,
    threshold: request.threshold.toFixed(2),
    overlay_alpha: request.overlayAlpha.toFixed(2),
  });
  const response = await fetch(`/api/segment?${query}`, { method: "POST", body });
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as unknown;
    throw new Error(errorDetail(payload, `Segmentation failed (${response.status})`));
  }
  return {
    blob: await response.blob(),
    mode: request.outputMode,
    metadata: {
      modelId: response.headers.get("x-model-id") || request.modelId,
      inferenceMs: numberHeader(response.headers.get("x-inference-ms")),
      width: numberHeader(response.headers.get("x-image-width")),
      height: numberHeader(response.headers.get("x-image-height")),
      threshold: numberHeader(response.headers.get("x-threshold")),
    },
  };
}

function errorDetail(payload: unknown, fallback: string): string {
  if (payload && typeof payload === "object" && "detail" in payload) {
    const detail = (payload as { detail?: unknown }).detail;
    if (typeof detail === "string") return detail;
  }
  return fallback;
}

function numberHeader(value: string | null): number | null {
  if (value === null) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}
