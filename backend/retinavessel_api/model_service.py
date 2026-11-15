"""Checkpoint registry and native-resolution tiled model inference."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from retinavessel.inference import choose_device, predict_probability
from retinavessel.model import UNet, model_params

from .config import Settings


@dataclass(frozen=True)
class SegmentationResult:
    probability: np.ndarray
    elapsed_ms: float
    threshold: float


class LoadedModel:
    def __init__(self, model_id, description, checkpoint_path, device):
        self.model_id = model_id
        self.description = description
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {self.checkpoint_path}")
        checkpoint = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=True
        )
        required = {"model_state", "architecture", "base", "tile", "overlap", "threshold"}
        missing = required - set(checkpoint)
        if missing:
            raise ValueError(
                f"Checkpoint {self.checkpoint_path} is missing {sorted(missing)}"
            )
        architecture = checkpoint["architecture"]
        if architecture not in {"unet", "lite_unet"}:
            raise ValueError(f"Unsupported checkpoint architecture: {architecture}")
        self.model = UNet(
            base=int(checkpoint["base"]), lite=architecture == "lite_unet"
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(device).eval()
        self.device = device
        self.architecture = architecture
        self.tile = int(checkpoint["tile"])
        self.overlap = int(checkpoint["overlap"])
        self.threshold = float(checkpoint["threshold"])
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Checkpoint threshold must be between zero and one")
        self.metadata = {
            "id": model_id,
            "description": description,
            "architecture": architecture,
            "parameters": model_params(self.model),
            "tile": self.tile,
            "overlap": self.overlap,
            "threshold": self.threshold,
            "checkpoint_sha256": _sha256(self.checkpoint_path),
            "validation_dice": checkpoint.get("validation_dice"),
            "validation_epoch": checkpoint.get("epoch"),
        }

    def segment(self, image: Image.Image, threshold=None):
        selected_threshold = self.threshold if threshold is None else float(threshold)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        probability = predict_probability(
            self.model,
            image,
            self.device,
            tile=self.tile,
            overlap=self.overlap,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return SegmentationResult(probability, elapsed_ms, selected_threshold)


class ModelRegistry:
    def __init__(self, settings):
        self.device = choose_device(settings.device)
        self.default_model = settings.default_model
        self.models = {
            spec.model_id: LoadedModel(
                spec.model_id, spec.description, spec.checkpoint, self.device
            )
            for spec in settings.models
        }

    def get(self, model_id=None):
        selected = model_id or self.default_model
        try:
            return self.models[selected]
        except KeyError as error:
            raise KeyError(f"Unknown model: {selected}") from error

    def public_metadata(self):
        return [self.models[key].metadata for key in sorted(self.models)]

    def close(self):
        self.models.clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
