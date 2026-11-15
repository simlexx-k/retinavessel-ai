"""Validation-only robustness, topology and efficiency evaluation.

This module deliberately resolves samples only from the frozen development
manifest. It never opens the official FIVES test partition.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from .data import find_fives_root, load_manifest, read_pair, samples_from_frozen_split
from .degradations import (
    CONDITIONS,
    apply_degradation,
    condition_manifest,
    stable_degradation_seed,
)
from .inference import choose_device, predict_probability
from .metrics import approx_fov, compute_metrics
from .model import UNet, model_params
from .quality import read_quality_workbook


PIXEL_METRICS = ("dice", "iou", "sensitivity", "specificity", "precision")
TOPOLOGY_METRICS = (
    "cldice",
    "thin_centerline_recall",
    "skeleton_component_ratio",
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_models(values):
    models = []
    seen = set()
    for value in values:
        if "=" not in value:
            raise ValueError("Each --model must use MODEL_ID=/path/to/best.pt")
        model_id, checkpoint = value.split("=", 1)
        if not model_id or not checkpoint or model_id in seen:
            raise ValueError(f"Invalid or duplicate model declaration: {value}")
        if not all(character.isalnum() or character in "_-" for character in model_id):
            raise ValueError(f"Invalid model ID: {model_id}")
        seen.add(model_id)
        models.append((model_id, Path(checkpoint)))
    if not models:
        raise ValueError("At least one model is required")
    return models


def _load_checkpoint(path, device):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "model_state",
        "architecture",
        "base",
        "tile",
        "overlap",
        "threshold",
        "epoch",
        "validation_dice",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint {path} is missing {sorted(missing)}")
    architecture = checkpoint["architecture"]
    if architecture not in {"unet", "lite_unet"}:
        raise ValueError(f"Unsupported architecture: {architecture}")
    model = UNet(
        base=int(checkpoint["base"]), lite=architecture == "lite_unet"
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    tile = int(checkpoint["tile"])
    overlap = int(checkpoint["overlap"])
    threshold = float(checkpoint["threshold"])
    if tile <= 0 or not 0 <= overlap < tile:
        raise ValueError(f"Invalid tiling metadata in {path}")
    if not 0 <= threshold <= 1:
        raise ValueError(f"Invalid threshold in {path}")
    metadata = {
        "checkpoint": str(path),
        "checkpoint_sha256": _sha256(path),
        "architecture": architecture,
        "base": int(checkpoint["base"]),
        "parameters": model_params(model),
        "parameter_bytes": sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        ),
        "tile": tile,
        "overlap": overlap,
        "threshold": threshold,
        "training_seed": checkpoint.get("seed"),
        "validation_epoch": int(checkpoint["epoch"]),
        "checkpoint_validation_dice": float(checkpoint["validation_dice"]),
    }
    return model, metadata


def _quality_lookup(root):
    workbooks = sorted(root.glob("*.xlsx"))
    if len(workbooks) != 1:
        return {}, None
    return read_quality_workbook(workbooks[0])["train"], workbooks[0]


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _warm_up(model, image, device, tile, overlap, repetitions):
    if repetitions <= 0:
        return
    width, height = image.size
    left = max(0, (width - tile) // 2)
    top = max(0, (height - tile) // 2)
    crop = image.crop((left, top, min(left + tile, width), min(top + tile, height)))
    for _ in range(repetitions):
        predict_probability(model, crop, device, tile=tile, overlap=overlap)
    _synchronize(device)


def _bootstrap_mean(values, repetitions, seed):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    if len(values) == 1 or repetitions <= 0:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return mean, float(lower), float(upper)


def _summaries(records, topology, bootstrap_repetitions, bootstrap_seed):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["model_id"], record["condition"])].append(record)
    metric_names = PIXEL_METRICS + (TOPOLOGY_METRICS if topology else ())
    rows = []
    for (model_id, condition), members in grouped.items():
        first = members[0]
        row = {
            "model_id": model_id,
            "condition": condition,
            "category": first["category"],
            "severity": first["severity"],
            "image_count": len(members),
            "parameters": first["parameters"],
            "parameter_bytes": first["parameter_bytes"],
            "mean_latency_ms": float(np.mean([item["latency_ms"] for item in members])),
            "median_latency_ms": float(np.median([item["latency_ms"] for item in members])),
            "p95_latency_ms": float(np.quantile(
                [item["latency_ms"] for item in members], 0.95
            )),
            "maximum_peak_accelerator_bytes": max(
                item["peak_accelerator_bytes"] for item in members
            ),
        }
        for metric in metric_names:
            values = [item[metric] for item in members]
            seed = stable_degradation_seed(
                bootstrap_seed, f"{model_id}:{metric}", condition
            )
            mean, lower, upper = _bootstrap_mean(
                values, bootstrap_repetitions, seed
            )
            row[f"mean_{metric}"] = mean
            row[f"median_{metric}"] = float(np.median(values))
            row[f"{metric}_ci95_lower"] = lower
            row[f"{metric}_ci95_upper"] = upper
        rows.append(row)
    return rows


def _paired_drops(records, topology, bootstrap_repetitions, bootstrap_seed):
    lookup = {
        (record["model_id"], record["condition"], record["image_id"]): record
        for record in records
    }
    models = sorted({record["model_id"] for record in records})
    conditions = [name for name in CONDITIONS if name != "clean"]
    evaluated = {record["condition"] for record in records}
    metric_names = ("dice",) + (
        ("cldice", "thin_centerline_recall") if topology else ()
    )
    rows = []
    for model_id in models:
        clean_ids = sorted(
            record["image_id"]
            for record in records
            if record["model_id"] == model_id and record["condition"] == "clean"
        )
        for condition in conditions:
            if condition not in evaluated or not clean_ids:
                continue
            row = {
                "model_id": model_id,
                "condition": condition,
                "category": CONDITIONS[condition].category,
                "severity": CONDITIONS[condition].severity,
                "image_count": len(clean_ids),
            }
            for metric in metric_names:
                drops = [
                    lookup[(model_id, "clean", image_id)][metric]
                    - lookup[(model_id, condition, image_id)][metric]
                    for image_id in clean_ids
                ]
                seed = stable_degradation_seed(
                    bootstrap_seed, f"{model_id}:{metric}:drop", condition
                )
                mean, lower, upper = _bootstrap_mean(
                    drops, bootstrap_repetitions, seed
                )
                row[f"mean_{metric}_drop"] = mean
                row[f"{metric}_drop_ci95_lower"] = lower
                row[f"{metric}_drop_ci95_upper"] = upper
            rows.append(row)
    return rows


def _write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_validation(args):
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be non-negative")
    if args.bootstrap_repetitions < 0:
        raise ValueError("--bootstrap-repetitions must be non-negative")
    if args.thin_max_radius <= 0:
        raise ValueError("--thin-max-radius must be positive")
    if args.thin_tolerance < 0:
        raise ValueError("--thin-tolerance must be non-negative")
    split_path = Path(args.split_manifest)
    split_sha256 = _sha256(split_path)
    if args.expected_split_sha256 and split_sha256 != args.expected_split_sha256:
        raise ValueError("Frozen split SHA-256 does not match the declared value")
    split_manifest = json.loads(split_path.read_text())
    root = find_fives_root(args.data)
    official_train = load_manifest(root, "train")
    _, validation_samples = samples_from_frozen_split(
        official_train, split_manifest
    )
    if args.validation_limit is not None:
        if args.validation_limit <= 0:
            raise ValueError("--validation-limit must be positive")
        validation_samples = validation_samples[: args.validation_limit]
    if not validation_samples:
        raise ValueError("Frozen split contains no validation samples")

    selected_conditions = list(args.conditions)
    condition_manifest(selected_conditions)
    models = _parse_models(args.model)
    device = choose_device(args.device)
    quality_scores, quality_workbook = _quality_lookup(root)
    records = []
    model_metadata = []
    first_image, _ = read_pair(validation_samples[0])

    for model_id, checkpoint_path in models:
        model, metadata = _load_checkpoint(checkpoint_path, device)
        metadata["model_id"] = model_id
        model_metadata.append(metadata)
        _warm_up(
            model,
            first_image,
            device,
            metadata["tile"],
            metadata["overlap"],
            args.warmup_runs,
        )
        for condition in selected_conditions:
            specification = CONDITIONS[condition]
            print(
                f"model={model_id}, condition={condition}, "
                f"images={len(validation_samples)}"
            )
            for index, sample in enumerate(validation_samples, 1):
                clean_image, mask = read_pair(sample)
                image = apply_degradation(
                    clean_image,
                    condition,
                    stable_degradation_seed(
                        args.degradation_seed, sample.id, condition
                    ),
                )
                fov = approx_fov(clean_image)
                truth = np.asarray(mask) > 127
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                _synchronize(device)
                started = time.perf_counter()
                probability = predict_probability(
                    model,
                    image,
                    device,
                    tile=metadata["tile"],
                    overlap=metadata["overlap"],
                )
                _synchronize(device)
                latency_ms = (time.perf_counter() - started) * 1000.0
                peak_bytes = (
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda"
                    else 0
                )
                metrics = compute_metrics(
                    probability,
                    truth,
                    fov,
                    threshold=metadata["threshold"],
                    include_topology=args.topology,
                    thin_max_radius=args.thin_max_radius,
                    thin_tolerance=args.thin_tolerance,
                )
                quality = quality_scores.get(sample.id)
                record = {
                    "model_id": model_id,
                    "architecture": metadata["architecture"],
                    "parameters": metadata["parameters"],
                    "parameter_bytes": metadata["parameter_bytes"],
                    "condition": condition,
                    "category": specification.category,
                    "severity": specification.severity,
                    "image_id": sample.id,
                    "disease_group": sample.condition,
                    "official_quality_class": (
                        "unknown"
                        if quality is None
                        else "all_good" if quality.all_good else "degraded"
                    ),
                    "quality_illumination_color": (
                        "" if quality is None else quality.illumination_color
                    ),
                    "quality_blur": "" if quality is None else quality.blur,
                    "quality_low_contrast": (
                        "" if quality is None else quality.low_contrast
                    ),
                    "latency_ms": latency_ms,
                    "peak_accelerator_bytes": peak_bytes,
                    **metrics,
                }
                records.append(record)
                if index % 10 == 0 or index == len(validation_samples):
                    print(f"  processed {index}/{len(validation_samples)}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summaries = _summaries(
        records,
        args.topology,
        args.bootstrap_repetitions,
        args.bootstrap_seed,
    )
    paired_drops = _paired_drops(
        records,
        args.topology,
        args.bootstrap_repetitions,
        args.bootstrap_seed,
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "validation_per_image.csv", records)
    _write_csv(output / "validation_summary.csv", summaries)
    _write_csv(output / "paired_degradation_drops.csv", paired_drops)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "FIVES frozen development validation partition",
        "official_test_evaluated": False,
        "validation_image_count": len(validation_samples),
        "debug_validation_limit": args.validation_limit,
        "split_manifest": str(split_path),
        "split_manifest_sha256": split_sha256,
        "quality_workbook": (
            None if quality_workbook is None else str(quality_workbook)
        ),
        "device": str(device),
        "pytorch_version": torch.__version__,
        "degradation_seed": args.degradation_seed,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "topology_enabled": args.topology,
        "thin_vessel_definition": {
            "ground_truth_centerline_max_radius_pixels": args.thin_max_radius,
            "prediction_tolerance_pixels": args.thin_tolerance,
        },
        "conditions": condition_manifest(selected_conditions),
        "models": model_metadata,
        "summary": summaries,
        "paired_degradation_drops": paired_drops,
        "caveats": [
            "Confidence intervals resample images because FIVES public patient IDs are unavailable.",
            "Synthetic degradations are controlled stress tests, not claims about all clinical acquisition failures.",
            "The FIVES field of view is image-derived and held fixed from the clean image for every degradation.",
            "Latency is hardware- and software-specific; compare models only within the same run environment.",
        ],
    }
    (output / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps({
        "output": str(output),
        "models": len(models),
        "conditions": len(selected_conditions),
        "images": len(validation_samples),
        "official_test_evaluated": False,
    }, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Repeat MODEL_ID=/path/to/best.pt for each checkpoint",
    )
    parser.add_argument("--out", default="outputs/validation_v3")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--conditions", nargs="+", choices=list(CONDITIONS), default=list(CONDITIONS)
    )
    parser.add_argument("--topology", action="store_true")
    parser.add_argument("--thin-max-radius", type=float, default=2.0)
    parser.add_argument("--thin-tolerance", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--degradation-seed", type=int, default=2026)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--expected-split-sha256")
    parser.add_argument(
        "--validation-limit",
        type=int,
        help="Debug only; omit for scientific evaluation",
    )
    run_validation(parser.parse_args())


if __name__ == "__main__":
    main()
