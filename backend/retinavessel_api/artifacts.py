"""Validate and promote trusted training checkpoints into the model registry."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

import torch

from retinavessel.model import UNet, model_params

from .config import MODEL_ID_PATTERN


REQUIRED_CHECKPOINT_FIELDS = {
    "model_state",
    "architecture",
    "base",
    "tile",
    "overlap",
    "threshold",
    "epoch",
    "validation_dice",
}


def inspect_checkpoint(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    missing = REQUIRED_CHECKPOINT_FIELDS - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing {sorted(missing)}")
    architecture = checkpoint["architecture"]
    if architecture not in {"unet", "lite_unet"}:
        raise ValueError(f"Unsupported architecture: {architecture}")
    base = int(checkpoint["base"])
    tile = int(checkpoint["tile"])
    overlap = int(checkpoint["overlap"])
    threshold = float(checkpoint["threshold"])
    if base <= 0 or tile <= 0 or not 0 <= overlap < tile:
        raise ValueError("Checkpoint contains invalid architecture or tiling values")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Checkpoint threshold must be between zero and one")
    model = UNet(base=base, lite=architecture == "lite_unet")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return {
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "architecture": architecture,
        "base": base,
        "parameters": model_params(model),
        "tile": tile,
        "overlap": overlap,
        "threshold": threshold,
        "seed": checkpoint.get("seed"),
        "validation_epoch": int(checkpoint["epoch"]),
        "validation_dice": float(checkpoint["validation_dice"]),
    }


def promote_checkpoint(
    checkpoint,
    model_id,
    models_dir,
    registry_path,
    description="",
    make_default=False,
    run_manifest=None,
    experiment=None,
):
    if not MODEL_ID_PATTERN.fullmatch(model_id):
        raise ValueError("Invalid model ID")
    checkpoint = Path(checkpoint)
    models_dir = Path(models_dir)
    registry_path = Path(registry_path)
    metadata = inspect_checkpoint(checkpoint)
    provenance = _read_provenance(
        run_manifest, experiment, metadata
    ) if run_manifest else None

    models_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{model_id}-{metadata['sha256'][:12]}.pt"
    destination = models_dir / filename
    if destination.exists():
        if _sha256(destination) != metadata["sha256"]:
            raise ValueError(f"Existing destination has the wrong hash: {destination}")
    else:
        temporary = destination.with_suffix(".pt.tmp")
        shutil.copyfile(checkpoint, temporary)
        if _sha256(temporary) != metadata["sha256"]:
            temporary.unlink(missing_ok=True)
            raise ValueError("Copied checkpoint hash does not match the source")
        temporary.replace(destination)

    artifact = {
        "model_id": model_id,
        "description": description,
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_filename": checkpoint.name,
        "checkpoint": metadata,
        "provenance": provenance,
    }
    _atomic_json(destination.with_suffix(".json"), artifact)

    if registry_path.exists() and registry_path.is_dir():
        raise ValueError(f"Registry path is a directory: {registry_path}")
    if registry_path.exists():
        registry = json.loads(registry_path.read_text())
    else:
        registry = {"default_model": model_id, "models": {}}
    if not isinstance(registry.get("models"), dict):
        raise ValueError("Registry has no valid models object")
    relative = os.path.relpath(destination, registry_path.parent)
    registry["models"][model_id] = {
        "checkpoint": relative,
        "description": description,
        "sha256": metadata["sha256"],
        "architecture": metadata["architecture"],
        "parameters": metadata["parameters"],
        "validation_dice": metadata["validation_dice"],
        "validation_epoch": metadata["validation_epoch"],
        "candidate": True,
    }
    if make_default or not registry.get("default_model"):
        registry["default_model"] = model_id
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(registry_path, registry)
    return destination, artifact, registry


def _read_provenance(path, experiment, metadata):
    if not experiment:
        raise ValueError("Experiment name is required with a run manifest")
    manifest = json.loads(Path(path).read_text())
    matches = [row for row in manifest.get("results", [])
               if row.get("experiment") == experiment]
    if len(matches) != 1:
        raise ValueError(f"Expected one manifest result for {experiment}")
    result = matches[0]
    checks = {
        "model": metadata["architecture"],
        "parameters": metadata["parameters"],
        "best_epoch": metadata["validation_epoch"],
    }
    for key, expected in checks.items():
        if result.get(key) != expected:
            raise ValueError(f"Run manifest {key} does not match checkpoint")
    if abs(float(result.get("best_validation_dice")) - metadata["validation_dice"]) > 1e-9:
        raise ValueError("Run manifest validation Dice does not match checkpoint")
    return {
        "experiment": experiment,
        "training_code_commit": manifest.get("training_code_commit"),
        "split_manifest_sha256": manifest.get("split_manifest_sha256"),
        "pytorch_version": manifest.get("pytorch_version"),
        "cuda_device": manifest.get("cuda_device"),
        "official_test_evaluated": manifest.get("official_test_evaluated"),
        "maximum_epochs": manifest.get("maximum_epochs"),
        "early_stopping_patience": manifest.get("early_stopping_patience"),
        "completed_epochs": result.get("completed_epochs"),
        "degradation_training": result.get("degradation_training"),
    }


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--registry", default="backend/model_registry.json")
    parser.add_argument("--description", default="")
    parser.add_argument("--run-manifest")
    parser.add_argument("--experiment")
    parser.add_argument("--default", action="store_true")
    args = parser.parse_args()
    destination, artifact, registry = promote_checkpoint(
        checkpoint=args.checkpoint,
        model_id=args.model_id,
        models_dir=args.models_dir,
        registry_path=args.registry,
        description=args.description,
        make_default=args.default,
        run_manifest=args.run_manifest,
        experiment=args.experiment,
    )
    print(json.dumps({
        "destination": str(destination),
        "checkpoint": artifact["checkpoint"],
        "default_model": registry["default_model"],
    }, indent=2))


if __name__ == "__main__":
    main()
