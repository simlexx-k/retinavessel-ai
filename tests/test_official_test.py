import json
from pathlib import Path

import pytest

from retinavessel.official_test import (
    METRICS,
    _checkpoint_summaries,
    _configuration_summaries,
    _validate_plan,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs" / "fives_official_test_plan.json"


def test_frozen_official_test_plan_has_all_nine_declared_checkpoints():
    plan = json.loads(PLAN.read_text())
    checkpoints = _validate_plan(plan)

    assert len(checkpoints) == 9
    assert {(item["configuration"], item["seed"]) for item in checkpoints} == {
        (configuration, seed)
        for configuration in ("unet", "lite_unet", "lite_robust")
        for seed in (7, 42, 123)
    }
    assert {item["sha256"] for item in checkpoints} == {
        item["sha256"]
        for item in json.loads((ROOT / "provenance" / "manifest.json").read_text())["checkpoints"]
    }


def test_frozen_plan_rejects_changed_inference_policy():
    plan = json.loads(PLAN.read_text())
    plan["inference"]["threshold"] = 0.51

    with pytest.raises(ValueError, match="inference policy"):
        _validate_plan(plan)


def test_official_test_summaries_preserve_seed_and_configuration_structure():
    records = []
    for configuration_index, configuration in enumerate(("unet", "lite_unet", "lite_robust")):
        architecture = "unet" if configuration == "unet" else "lite_unet"
        parameters = 1964097 if configuration == "unet" else 236156
        for seed_index, seed in enumerate((7, 42, 123)):
            for image_index, image_id in enumerate(("1_A", "2_D")):
                value = 0.5 + configuration_index * 0.1 + seed_index * 0.01 + image_index * 0.001
                records.append({
                    "model_id": f"{configuration}_seed{seed}",
                    "configuration": configuration,
                    "seed": seed,
                    "architecture": architecture,
                    "image_id": image_id,
                    "parameters": parameters,
                    "parameter_bytes": parameters * 4,
                    "latency_ms": 10.0,
                    "peak_accelerator_bytes": 100,
                    "tp": 50,
                    "fp": 5,
                    "fn": 5,
                    "tn": 100,
                    **{metric: value for metric in METRICS},
                })

    checkpoint_rows = _checkpoint_summaries(records, repetitions=20, bootstrap_seed=2026)
    configuration_rows = _configuration_summaries(records, repetitions=20, bootstrap_seed=2026)

    assert len(checkpoint_rows) == 9
    assert all(row["image_count"] == 2 for row in checkpoint_rows)
    assert len(configuration_rows) == 3
    robust = next(row for row in configuration_rows if row["configuration"] == "lite_robust")
    assert robust["seed_count"] == 3
    assert robust["image_count_per_seed"] == 2
    assert robust["mean_dice_across_seeds"] == pytest.approx(0.7105)
