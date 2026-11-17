"""One-time, plan-locked evaluation on the official FIVES test partition."""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, stdev
import time

import numpy as np
import torch

from .data import find_fives_root, load_manifest, read_pair
from .degradations import stable_degradation_seed
from .inference import choose_device, predict_probability
from .metrics import approx_fov, compute_metrics
from .validation import _load_checkpoint, _synchronize, _warm_up


METRICS = (
    "dice",
    "iou",
    "sensitivity",
    "specificity",
    "precision",
    "cldice",
    "thin_centerline_recall",
    "skeleton_component_ratio",
)
PIXEL_METRICS = ("dice", "iou", "sensitivity", "specificity", "precision")
EXPECTED_CONFIGURATIONS = ("unet", "lite_unet", "lite_robust")
EXPECTED_SEEDS = (7, 42, 123)


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty result table: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    numeric = {
        "seed": int,
        "parameters": int,
        "parameter_bytes": int,
        "latency_ms": float,
        "peak_accelerator_bytes": int,
        "tp": int,
        "fp": int,
        "fn": int,
        "tn": int,
        "thin_centerline_pixels": int,
        "predicted_skeleton_components": int,
        "truth_skeleton_components": int,
        **{metric: float for metric in METRICS},
    }
    for row in rows:
        for field, converter in numeric.items():
            row[field] = converter(row[field])
    return rows


def _validate_plan(plan: dict[str, object]) -> list[dict[str, object]]:
    if plan.get("schema_version") != 1:
        raise ValueError("Unsupported official-test plan schema")
    if plan.get("official_test_image_count") != 200:
        raise ValueError("The frozen plan must require exactly 200 official-test images")
    if tuple(plan.get("configurations", [])) != EXPECTED_CONFIGURATIONS:
        raise ValueError("Frozen configuration order or membership changed")
    if tuple(plan.get("seeds", [])) != EXPECTED_SEEDS:
        raise ValueError("Frozen seed order or membership changed")
    if plan.get("selected_configuration") != "lite_robust":
        raise ValueError("Frozen configuration decision changed")
    if plan.get("individual_seed_selected") is not False:
        raise ValueError("The official test must not select an individual seed")

    inference = plan.get("inference", {})
    if inference != {"tile": 384, "overlap": 64, "threshold": 0.5, "warmup_runs": 2}:
        raise ValueError("Frozen inference policy changed")
    uncertainty = plan.get("uncertainty", {})
    if uncertainty != {
        "bootstrap_repetitions": 2000,
        "bootstrap_seed": 2026,
        "unit": "image",
    }:
        raise ValueError("Frozen uncertainty policy changed")
    topology = plan.get("topology", {})
    if topology != {
        "enabled": True,
        "thin_centerline_max_radius_pixels": 2.0,
        "prediction_tolerance_pixels": 1,
    }:
        raise ValueError("Frozen topology policy changed")

    checkpoints = plan.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 9:
        raise ValueError("The frozen plan must contain exactly nine checkpoints")
    expected_pairs = {
        (configuration, seed)
        for configuration in EXPECTED_CONFIGURATIONS
        for seed in EXPECTED_SEEDS
    }
    actual_pairs = {(item.get("configuration"), item.get("seed")) for item in checkpoints}
    if actual_pairs != expected_pairs:
        raise ValueError("The plan does not contain one checkpoint per configuration and seed")
    model_ids = [item.get("model_id") for item in checkpoints]
    if len(set(model_ids)) != 9 or any(not isinstance(value, str) for value in model_ids):
        raise ValueError("Checkpoint model IDs must be nine unique strings")
    for item in checkpoints:
        configuration = item["configuration"]
        expected_architecture = "unet" if configuration == "unet" else "lite_unet"
        expected_parameters = 1964097 if configuration == "unet" else 236156
        if item.get("architecture") != expected_architecture:
            raise ValueError(f"Architecture mismatch in plan for {item['model_id']}")
        if item.get("parameters") != expected_parameters:
            raise ValueError(f"Parameter-count mismatch in plan for {item['model_id']}")
        digest = item.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"Invalid checkpoint hash for {item['model_id']}")
        relative = Path(str(item.get("relative_path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe checkpoint path for {item['model_id']}")
    return checkpoints


def _bootstrap_mean(values: list[float], repetitions: int, seed: int) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(repetitions, len(array)))
    means = array[indices].mean(axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return float(array.mean()), float(lower), float(upper)


def _checkpoint_summaries(
    records: list[dict[str, object]], repetitions: int, bootstrap_seed: int
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["model_id"])].append(record)
    rows: list[dict[str, object]] = []
    for model_id in sorted(grouped):
        members = grouped[model_id]
        first = members[0]
        row: dict[str, object] = {
            "model_id": model_id,
            "configuration": first["configuration"],
            "seed": first["seed"],
            "architecture": first["architecture"],
            "image_count": len(members),
            "parameters": first["parameters"],
            "parameter_bytes": first["parameter_bytes"],
            "mean_latency_ms": float(mean(float(item["latency_ms"]) for item in members)),
            "median_latency_ms": float(np.median([item["latency_ms"] for item in members])),
            "p95_latency_ms": float(np.quantile([item["latency_ms"] for item in members], 0.95)),
            "maximum_peak_accelerator_bytes": max(
                int(item["peak_accelerator_bytes"]) for item in members
            ),
        }
        for metric in METRICS:
            values = [float(item[metric]) for item in members]
            seed = stable_degradation_seed(bootstrap_seed, model_id, metric)
            metric_mean, lower, upper = _bootstrap_mean(values, repetitions, seed)
            row[f"mean_{metric}"] = metric_mean
            row[f"median_{metric}"] = float(np.median(values))
            row[f"{metric}_ci95_lower"] = lower
            row[f"{metric}_ci95_upper"] = upper
        totals = {
            field: sum(int(item[field]) for item in members)
            for field in ("tp", "fp", "fn", "tn")
        }
        tp, fp, fn, tn = (totals[field] for field in ("tp", "fp", "fn", "tn"))
        denominators = {
            "dice": (2 * tp, 2 * tp + fp + fn),
            "iou": (tp, tp + fp + fn),
            "sensitivity": (tp, tp + fn),
            "specificity": (tn, tn + fp),
            "precision": (tp, tp + fp),
        }
        for metric in PIXEL_METRICS:
            numerator, denominator = denominators[metric]
            row[f"micro_{metric}"] = float(numerator / denominator) if denominator else 0.0
        rows.append(row)
    return rows


def _configuration_summaries(
    records: list[dict[str, object]], repetitions: int, bootstrap_seed: int
) -> list[dict[str, object]]:
    lookup = {
        (str(record["configuration"]), int(record["seed"]), str(record["image_id"])): record
        for record in records
    }
    rows: list[dict[str, object]] = []
    for configuration in EXPECTED_CONFIGURATIONS:
        image_sets = [
            {str(record["image_id"]) for record in records
             if record["configuration"] == configuration and int(record["seed"]) == seed}
            for seed in EXPECTED_SEEDS
        ]
        if not image_sets or any(values != image_sets[0] for values in image_sets[1:]):
            raise ValueError(f"Image IDs differ between seeds for {configuration}")
        image_ids = sorted(image_sets[0])
        row: dict[str, object] = {
            "configuration": configuration,
            "seed_count": len(EXPECTED_SEEDS),
            "image_count_per_seed": len(image_ids),
        }
        for metric in METRICS:
            matrix = np.asarray([
                [float(lookup[(configuration, seed, image_id)][metric]) for image_id in image_ids]
                for seed in EXPECTED_SEEDS
            ])
            seed_means = matrix.mean(axis=1)
            rng = np.random.default_rng(
                stable_degradation_seed(bootstrap_seed, configuration, metric)
            )
            indices = rng.integers(0, len(image_ids), size=(repetitions, len(image_ids)))
            bootstrap = matrix[:, indices].mean(axis=2).mean(axis=0)
            lower, upper = np.quantile(bootstrap, (0.025, 0.975))
            row[f"mean_{metric}_across_seeds"] = float(seed_means.mean())
            row[f"{metric}_across_seed_sample_sd"] = float(stdev(seed_means))
            row[f"{metric}_image_bootstrap_ci95_lower"] = float(lower)
            row[f"{metric}_image_bootstrap_ci95_upper"] = float(upper)
        rows.append(row)
    return rows


def _disease_summaries(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["model_id"]), str(record["disease_group"]))].append(record)
    rows: list[dict[str, object]] = []
    for (model_id, disease_group), members in sorted(grouped.items()):
        first = members[0]
        rows.append({
            "model_id": model_id,
            "configuration": first["configuration"],
            "seed": first["seed"],
            "disease_group": disease_group,
            "image_count": len(members),
            **{
                f"mean_{metric}": float(mean(float(item[metric]) for item in members))
                for metric in METRICS
            },
        })
    return rows


def run_official_test(args) -> dict[str, object]:
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text())
    checkpoint_specs = _validate_plan(plan)
    checkpoint_root = Path(args.checkpoint_root).resolve()
    output = Path(args.out).resolve()
    partial_path = output / "official_test_per_image.partial.csv"
    state_path = output / "official_test_state.json"
    output.mkdir(parents=True, exist_ok=True)

    if (output / "official_test_report.json").exists():
        raise FileExistsError("This output directory already contains a completed official-test run")
    if (partial_path.exists() or state_path.exists()) and not args.resume:
        raise FileExistsError("Partial official-test results exist; pass --resume for the same frozen run")
    if args.resume and partial_path.exists() != state_path.exists():
        raise ValueError("A resumable run requires both the partial table and state record")

    root = find_fives_root(args.data)
    samples = load_manifest(root, "test")
    if len(samples) != plan["official_test_image_count"]:
        raise ValueError(
            f"Expected {plan['official_test_image_count']} official-test images, found {len(samples)}"
        )
    if len({sample.id for sample in samples}) != len(samples):
        raise ValueError("Official-test image IDs are not unique")

    plan_sha256 = _digest(plan_path)
    if args.resume and state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("status") != "partial" or state.get("plan_sha256") != plan_sha256:
            raise ValueError("Partial-run state does not match the frozen plan")
    records = _read_csv(partial_path) if args.resume and partial_path.exists() else []
    expected_ids = {sample.id for sample in samples}
    plan_by_model = {str(item["model_id"]): item for item in checkpoint_specs}
    completed: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        model_id = str(record["model_id"])
        if model_id not in plan_by_model:
            raise ValueError(f"Unknown model in partial results: {model_id}")
        specification = plan_by_model[model_id]
        if (
            record["checkpoint_sha256"] != specification["sha256"]
            or record["configuration"] != specification["configuration"]
            or int(record["seed"]) != specification["seed"]
        ):
            raise ValueError(f"Partial result identity mismatch: {model_id}")
        completed[model_id].append(record)
    for model_id, model_records in completed.items():
        image_ids = {str(record["image_id"]) for record in model_records}
        if image_ids != expected_ids or len(model_records) != len(samples):
            raise ValueError(f"Partial results for {model_id} are not one complete checkpoint")

    device = choose_device(args.device)
    preflight: list[tuple[dict[str, object], Path, dict[str, object]]] = []
    for specification in checkpoint_specs:
        model_id = str(specification["model_id"])
        checkpoint_path = (checkpoint_root / str(specification["relative_path"])).resolve()
        try:
            checkpoint_path.relative_to(checkpoint_root)
        except ValueError as error:
            raise ValueError(f"Checkpoint path escapes root: {checkpoint_path}") from error
        if _digest(checkpoint_path) != specification["sha256"]:
            raise ValueError(f"Checkpoint SHA-256 mismatch: {model_id}")
        model, metadata = _load_checkpoint(checkpoint_path, device)
        if metadata["architecture"] != specification["architecture"]:
            raise ValueError(f"Checkpoint architecture mismatch: {model_id}")
        if metadata["parameters"] != specification["parameters"]:
            raise ValueError(f"Checkpoint parameter-count mismatch: {model_id}")
        if metadata["training_seed"] != specification["seed"]:
            raise ValueError(f"Checkpoint training-seed mismatch: {model_id}")
        frozen_inference = plan["inference"]
        for field in ("tile", "overlap", "threshold"):
            if metadata[field] != frozen_inference[field]:
                raise ValueError(f"Checkpoint {field} mismatch: {model_id}")
        metadata.update({
            "model_id": model_id,
            "configuration": specification["configuration"],
            "seed": specification["seed"],
        })
        preflight.append((specification, checkpoint_path, metadata))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    model_metadata = [metadata for _, _, metadata in preflight]
    first_image, _ = read_pair(samples[0])
    for specification, checkpoint_path, metadata in preflight:
        model_id = str(specification["model_id"])
        model, loaded_metadata = _load_checkpoint(checkpoint_path, device)
        if loaded_metadata["checkpoint_sha256"] != metadata["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed after preflight: {model_id}")

        if model_id in completed:
            print(f"model={model_id}: already complete; verified and skipped")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        _warm_up(
            model,
            first_image,
            device,
            metadata["tile"],
            metadata["overlap"],
            int(frozen_inference["warmup_runs"]),
        )
        print(f"model={model_id}, official-test images={len(samples)}")
        model_records: list[dict[str, object]] = []
        for index, sample in enumerate(samples, 1):
            image, mask = read_pair(sample)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            _synchronize(device)
            started = time.perf_counter()
            probability = predict_probability(
                model,
                image,
                device,
                tile=int(metadata["tile"]),
                overlap=int(metadata["overlap"]),
            )
            _synchronize(device)
            latency_ms = (time.perf_counter() - started) * 1000.0
            peak_accelerator_bytes = (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            )
            metrics = compute_metrics(
                probability,
                np.asarray(mask) > 127,
                approx_fov(image),
                threshold=float(metadata["threshold"]),
                include_topology=True,
                thin_max_radius=float(plan["topology"]["thin_centerline_max_radius_pixels"]),
                thin_tolerance=int(plan["topology"]["prediction_tolerance_pixels"]),
            )
            model_records.append({
                "model_id": model_id,
                "configuration": specification["configuration"],
                "seed": specification["seed"],
                "architecture": metadata["architecture"],
                "parameters": metadata["parameters"],
                "parameter_bytes": metadata["parameter_bytes"],
                "checkpoint_sha256": metadata["checkpoint_sha256"],
                "image_id": sample.id,
                "disease_group": sample.condition,
                "latency_ms": latency_ms,
                "peak_accelerator_bytes": peak_accelerator_bytes,
                **metrics,
            })
            if index % 10 == 0 or index == len(samples):
                print(f"  processed {index}/{len(samples)}")
        records.extend(model_records)
        _write_csv(partial_path, records)
        state_path.write_text(json.dumps({
            "schema_version": 1,
            "status": "partial",
            "official_test_evaluated": True,
            "plan_sha256": plan_sha256,
            "completed_model_ids": sorted({str(item["model_id"]) for item in records}),
            "completed_record_count": len(records),
        }, indent=2) + "\n")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if len(records) != 9 * len(samples):
        raise ValueError(f"Expected {9 * len(samples)} per-image records, found {len(records)}")
    repetitions = int(plan["uncertainty"]["bootstrap_repetitions"])
    bootstrap_seed = int(plan["uncertainty"]["bootstrap_seed"])
    checkpoint_summary = _checkpoint_summaries(records, repetitions, bootstrap_seed)
    configuration_summary = _configuration_summaries(records, repetitions, bootstrap_seed)
    disease_summary = _disease_summaries(records)
    _write_csv(output / "official_test_per_image.csv", records)
    _write_csv(output / "official_test_checkpoint_summary.csv", checkpoint_summary)
    _write_csv(output / "official_test_configuration_summary.csv", configuration_summary)
    _write_csv(output / "official_test_disease_summary.csv", disease_summary)

    report: dict[str, object] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": plan["dataset"],
        "official_test_evaluated": True,
        "official_test_image_count": len(samples),
        "plan": str(plan_path),
        "plan_sha256": plan_sha256,
        "device": str(device),
        "pytorch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "models": model_metadata,
        "checkpoint_summary": checkpoint_summary,
        "configuration_summary": configuration_summary,
        "selected_configuration_remains": plan["selected_configuration"],
        "individual_seed_selected": False,
        "caveats": [
            "Confidence intervals resample images because public FIVES patient identifiers are unavailable.",
            "The retinal field of view is estimated from each clean image.",
            "Official-test results cannot revise the frozen model or inference policy.",
            "This is research-only vessel segmentation, not a diagnostic evaluation.",
        ],
    }
    (output / "official_test_report.json").write_text(json.dumps(report, indent=2) + "\n")
    state_path.write_text(json.dumps({
        "schema_version": 1,
        "status": "complete",
        "official_test_evaluated": True,
        "plan_sha256": plan_sha256,
        "completed_model_ids": sorted({str(item["model_id"]) for item in records}),
        "completed_record_count": len(records),
    }, indent=2) + "\n")
    partial_path.unlink()
    print(json.dumps({
        "output": str(output),
        "models": 9,
        "images": len(samples),
        "per_image_records": len(records),
        "official_test_evaluated": True,
    }, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--out", default="outputs/fives_official_test_v6")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only a partial run produced from the same frozen plan",
    )
    run_official_test(parser.parse_args())


if __name__ == "__main__":
    main()
