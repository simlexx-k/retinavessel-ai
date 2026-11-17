import csv
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "provenance"


def test_all_predeclared_runs_are_recorded_without_official_test_access():
    manifest = json.loads((PROVENANCE / "manifest.json").read_text())

    assert manifest["dataset"]["official_test_evaluated"] is False
    assert manifest["planned_notebooks"] == []
    assert {run["seed"] for run in manifest["external_artifacts"] if "seed" in run} == {7, 42, 123}
    assert all(run["official_test_evaluated"] is False for run in manifest["external_artifacts"])
    assert len(manifest["checkpoints"]) == 9

    seed123 = next(
        notebook
        for notebook in manifest["executed_notebooks"]
        if notebook["path"].endswith("retinavessel-ai-v5-seed123.ipynb")
    )
    assert seed123["original_path"] == "retinavessel-ai-v5-seed123.ipynb"
    assert "source_template" not in seed123


def test_frozen_multiseed_decision_matches_aggregation():
    summary = json.loads((PROVENANCE / "aggregation" / "multiseed_summary.json").read_text())

    assert summary["seeds"] == [7, 42, 123]
    assert summary["official_test_evaluated"] is False
    assert summary["decision"]["selected_configuration"] == "lite_robust"
    assert summary["decision"]["individual_seed_selected"] is False
    assert all(summary["decision"]["robust_wins_every_seed"].values())
    assert summary["decision"]["parameter_reduction_vs_unet_fraction"] == pytest.approx(
        0.8797635758315398
    )

    rows = list(csv.DictReader(
        (PROVENANCE / "aggregation" / "clean_metrics_across_seeds.csv").open()
    ))
    by_configuration = {row["configuration"]: row for row in rows}
    assert float(by_configuration["lite_robust"]["mean_dice_across_seed_mean"]) == pytest.approx(
        0.8466228599142087
    )
    assert float(by_configuration["lite_robust"]["mean_cldice_across_seed_mean"]) == pytest.approx(
        0.8571723294678723
    )
    assert float(
        by_configuration["lite_robust"]["mean_thin_centerline_recall_across_seed_mean"]
    ) == pytest.approx(0.35462186218037356)

    effects = list(csv.DictReader(
        (PROVENANCE / "aggregation" / "robust_minus_clean_across_seeds.csv").open()
    ))
    assert len(effects) == 8
    for row in effects:
        assert float(row["dice_difference_across_seed_mean"]) > 0
        assert float(row["cldice_difference_across_seed_mean"]) > 0
        assert float(row["thin_centerline_recall_difference_across_seed_mean"]) > 0
