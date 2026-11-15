"""Environment and model-registry configuration for the inference API."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from urllib.parse import urlsplit


MODEL_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    checkpoint: Path
    description: str = ""
    expected_sha256: str | None = None


@dataclass(frozen=True)
class Settings:
    models: tuple[ModelSpec, ...]
    default_model: str
    device: str = "auto"
    max_upload_bytes: int = 25 * 1024 * 1024
    max_image_pixels: int = 30_000_000
    api_key: str | None = None
    cors_origins: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.models:
            raise ValueError("At least one model must be configured")
        ids = [model.model_id for model in self.models]
        if len(ids) != len(set(ids)):
            raise ValueError("Model IDs must be unique")
        if any(not MODEL_ID_PATTERN.fullmatch(model_id) for model_id in ids):
            raise ValueError("Model IDs must match [a-z0-9][a-z0-9_-]{0,63}")
        if self.default_model not in ids:
            raise ValueError("Default model is not present in the registry")
        if self.max_upload_bytes <= 0 or self.max_image_pixels <= 0:
            raise ValueError("Upload byte and image pixel limits must be positive")
        if self.api_key is not None and len(self.api_key) < 16:
            raise ValueError("RETINAVESSEL_API_KEY must contain at least 16 characters")
        for origin in self.cors_origins:
            _validate_origin(origin)

    @classmethod
    def from_env(cls):
        registry_file = os.getenv("RETINAVESSEL_MODEL_REGISTRY")
        checkpoint = os.getenv("RETINAVESSEL_CHECKPOINT")
        if registry_file and checkpoint:
            raise ValueError(
                "Set RETINAVESSEL_MODEL_REGISTRY or RETINAVESSEL_CHECKPOINT, not both"
            )
        if registry_file:
            models, default_model = _read_registry(Path(registry_file))
        elif checkpoint:
            model_id = os.getenv("RETINAVESSEL_MODEL_ID", "default")
            models = (ModelSpec(model_id, Path(checkpoint).expanduser()),)
            default_model = model_id
        else:
            raise ValueError(
                "Configure RETINAVESSEL_MODEL_REGISTRY or RETINAVESSEL_CHECKPOINT"
            )
        return cls(
            models=models,
            default_model=default_model,
            device=os.getenv("RETINAVESSEL_DEVICE", "auto"),
            max_upload_bytes=_positive_env(
                "RETINAVESSEL_MAX_UPLOAD_BYTES", 25 * 1024 * 1024
            ),
            max_image_pixels=_positive_env(
                "RETINAVESSEL_MAX_IMAGE_PIXELS", 30_000_000
            ),
            api_key=os.getenv("RETINAVESSEL_API_KEY") or None,
            cors_origins=cors_origins_from_env(),
        )


def cors_origins_from_env():
    raw = os.getenv("RETINAVESSEL_CORS_ORIGINS", "")
    origins = tuple(origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip())
    for origin in origins:
        _validate_origin(origin)
    return origins


def _positive_env(name, default):
    raw = os.getenv(name)
    value = default if raw is None else int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _read_registry(path):
    payload = json.loads(path.expanduser().read_text())
    raw_models = payload.get("models")
    if not isinstance(raw_models, dict) or not raw_models:
        raise ValueError("Model registry must contain a non-empty 'models' object")
    models = []
    for model_id, entry in raw_models.items():
        expected_sha256 = None
        if isinstance(entry, str):
            checkpoint = entry
            description = ""
        elif isinstance(entry, dict):
            checkpoint = entry.get("checkpoint")
            description = entry.get("description", "")
            expected_sha256 = entry.get("sha256")
        else:
            raise ValueError(f"Invalid registry entry for {model_id}")
        if not checkpoint:
            raise ValueError(f"Registry entry {model_id} has no checkpoint")
        checkpoint_path = Path(checkpoint).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = path.expanduser().parent / checkpoint_path
        if expected_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", str(expected_sha256)
        ):
            raise ValueError(f"Registry entry {model_id} has an invalid SHA-256")
        models.append(
            ModelSpec(
                model_id,
                checkpoint_path,
                str(description),
                None if expected_sha256 is None else str(expected_sha256),
            )
        )
    default_model = payload.get("default_model")
    if not isinstance(default_model, str):
        raise ValueError("Model registry must declare 'default_model'")
    return tuple(models), default_model


def _validate_origin(origin):
    if origin == "*":
        raise ValueError("CORS wildcard is not allowed; configure explicit origins")
    parsed = urlsplit(origin)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid CORS origin: {origin}")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError(f"CORS origins cannot contain paths, queries or fragments: {origin}")
