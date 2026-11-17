"""Aggregate the three predeclared validation seeds without test-set access."""

from collections import defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = Path(__file__).resolve().parent
OUTPUT = PROVENANCE / "aggregation"
SEEDS = (7, 42, 123)
CONFIGURATIONS = ("unet", "lite_unet", "lite_robust")
METRICS = ("mean_dice", "mean_cldice", "mean_thin_centerline_recall")


def canonical_model(model_id: str) -> str:
    for configuration in ("lite_robust", "lite_unet", "unet"):
        if model_id.startswith(f"{configuration}_seed"):
            return configuration
    raise ValueError(f"Unrecognized model identifier: {model_id}")


def read_seed(seed: int) -> list[dict[str, str]]:
    path = PROVENANCE / "runs" / f"seed{seed}" / "validation_summary.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 27:
        raise ValueError(f"Expected 27 validation rows for seed {seed}, found {len(rows)}")
    if {row["condition"] for row in rows} != {
        "clean", "blur_mild", "blur_severe", "contrast_mild", "contrast_severe",
        "illumination_mild", "illumination_severe", "noise_mild", "noise_severe",
    }:
        raise ValueError(f"Condition set mismatch for seed {seed}")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def average_and_sd(values: list[float]) -> tuple[float, float]:
    if len(values) != len(SEEDS):
        raise ValueError(f"Expected {len(SEEDS)} seed values, found {len(values)}")
    return mean(values), stdev(values)


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    seed_rows = {seed: read_seed(seed) for seed in SEEDS}
    lookup: dict[tuple[int, str, str], dict[str, str]] = {}
    clean_by_seed: list[dict[str, object]] = []

    for seed, rows in seed_rows.items():
        for row in rows:
            configuration = canonical_model(row["model_id"])
            lookup[(seed, configuration, row["condition"])] = row
            if row["condition"] == "clean":
                clean_by_seed.append({
                    "seed": seed,
                    "configuration": configuration,
                    "model_id": row["model_id"],
                    "parameters": int(row["parameters"]),
                    "parameter_bytes": int(row["parameter_bytes"]),
                    "mean_latency_ms": float(row["mean_latency_ms"]),
                    "maximum_peak_accelerator_bytes": int(row["maximum_peak_accelerator_bytes"]),
                    **{metric: float(row[metric]) for metric in METRICS},
                })

    clean_by_seed.sort(key=lambda row: (CONFIGURATIONS.index(str(row["configuration"])), int(row["seed"])))
    clean_fields = [
        "seed", "configuration", "model_id", "parameters", "parameter_bytes",
        "mean_latency_ms", "maximum_peak_accelerator_bytes", *METRICS,
    ]
    write_csv(OUTPUT / "clean_metrics_by_seed.csv", clean_by_seed, clean_fields)

    clean_aggregate: list[dict[str, object]] = []
    for configuration in CONFIGURATIONS:
        rows = [row for row in clean_by_seed if row["configuration"] == configuration]
        result: dict[str, object] = {
            "configuration": configuration,
            "seed_count": len(rows),
            "parameters": rows[0]["parameters"],
            "parameter_bytes": rows[0]["parameter_bytes"],
        }
        for metric in (*METRICS, "mean_latency_ms", "maximum_peak_accelerator_bytes"):
            aggregate_mean, aggregate_sd = average_and_sd([float(row[metric]) for row in rows])
            result[f"{metric}_across_seed_mean"] = aggregate_mean
            result[f"{metric}_across_seed_sample_sd"] = aggregate_sd
        clean_aggregate.append(result)
    aggregate_fields = list(clean_aggregate[0])
    write_csv(OUTPUT / "clean_metrics_across_seeds.csv", clean_aggregate, aggregate_fields)

    conditions = sorted({row["condition"] for rows in seed_rows.values() for row in rows} - {"clean"})
    effects_by_seed: list[dict[str, object]] = []
    for condition in conditions:
        for seed in SEEDS:
            robust = lookup[(seed, "lite_robust", condition)]
            clean_lite = lookup[(seed, "lite_unet", condition)]
            effects_by_seed.append({
                "condition": condition,
                "category": robust["category"],
                "severity": robust["severity"],
                "seed": seed,
                "dice_difference": float(robust["mean_dice"]) - float(clean_lite["mean_dice"]),
                "cldice_difference": float(robust["mean_cldice"]) - float(clean_lite["mean_cldice"]),
                "thin_centerline_recall_difference": (
                    float(robust["mean_thin_centerline_recall"])
                    - float(clean_lite["mean_thin_centerline_recall"])
                ),
            })
    effect_fields = list(effects_by_seed[0])
    write_csv(OUTPUT / "robust_minus_clean_by_seed_condition.csv", effects_by_seed, effect_fields)

    effects_aggregate: list[dict[str, object]] = []
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in effects_by_seed:
        grouped[str(row["condition"])].append(row)
    for condition in conditions:
        rows = grouped[condition]
        result: dict[str, object] = {
            "condition": condition,
            "category": rows[0]["category"],
            "severity": rows[0]["severity"],
            "seed_count": len(rows),
        }
        for metric in ("dice_difference", "cldice_difference", "thin_centerline_recall_difference"):
            aggregate_mean, aggregate_sd = average_and_sd([float(row[metric]) for row in rows])
            result[f"{metric}_across_seed_mean"] = aggregate_mean
            result[f"{metric}_across_seed_sample_sd"] = aggregate_sd
        effects_aggregate.append(result)
    effect_aggregate_fields = list(effects_aggregate[0])
    write_csv(OUTPUT / "robust_minus_clean_across_seeds.csv", effects_aggregate, effect_aggregate_fields)

    clean_lookup = {
        (int(row["seed"]), str(row["configuration"])): row
        for row in clean_by_seed
    }
    robust_wins = {
        metric: all(
            float(clean_lookup[(seed, "lite_robust")][metric])
            > max(
                float(clean_lookup[(seed, "lite_unet")][metric]),
                float(clean_lookup[(seed, "unet")][metric]),
            )
            for seed in SEEDS
        )
        for metric in METRICS
    }
    if not all(robust_wins.values()):
        raise ValueError(f"Frozen decision invariant failed: {robust_wins}")

    unet_parameters = int(clean_lookup[(SEEDS[0], "unet")]["parameters"])
    robust_parameters = int(clean_lookup[(SEEDS[0], "lite_robust")]["parameters"])
    source_files = [
        PROVENANCE / "runs" / f"seed{seed}" / "validation_summary.csv"
        for seed in SEEDS
    ]
    summary = {
        "schema_version": 1,
        "seeds": list(SEEDS),
        "official_test_evaluated": False,
        "method": {
            "clean_metrics": "Across-seed arithmetic mean and sample standard deviation.",
            "robustness_effect": (
                "Within each seed and degradation, robust-lite minus clean-lite mean metric; "
                "then across-seed arithmetic mean and sample standard deviation. Because both "
                "models use the same 89 images, the difference of means equals the mean paired difference."
            ),
        },
        "source_validation_summaries": [
            {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
            for path in source_files
        ],
        "clean_configuration_summary": clean_aggregate,
        "robust_minus_clean_summary": effects_aggregate,
        "decision": {
            "selected_configuration": "lite_robust",
            "selected_architecture": "lite_unet",
            "training_degradation": True,
            "individual_seed_selected": False,
            "robust_wins_every_seed": robust_wins,
            "parameter_reduction_vs_unet_fraction": 1 - robust_parameters / unet_parameters,
            "rationale": (
                "The degradation-trained lightweight U-Net has the highest clean Dice, clDice "
                "and thin-centerline recall in every predeclared seed while using fewer parameters."
            ),
        },
    }
    (OUTPUT / "multiseed_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
