export type ModelMetadata = {
  id: string;
  description: string;
  architecture: "unet" | "lite_unet";
  parameters: number;
  tile: number;
  overlap: number;
  threshold: number;
  checkpoint_sha256: string;
  validation_dice: number | null;
  validation_epoch: number | null;
};

export type ModelsResponse = {
  default_model: string;
  models: ModelMetadata[];
};

export type HealthResponse = {
  status: string;
  model_count: number;
  default_model: string;
  device: string;
  research_only: boolean;
};

export type InferenceMetadata = {
  modelId: string;
  inferenceMs: number | null;
  width: number | null;
  height: number | null;
  threshold: number | null;
};
