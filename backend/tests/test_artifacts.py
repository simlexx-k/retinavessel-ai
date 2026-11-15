import json

import torch

from retinavessel.model import UNet
from retinavessel_api.artifacts import promote_checkpoint


def test_promote_checkpoint_validates_and_updates_registry(tmp_path):
    checkpoint = tmp_path / "source.pt"
    model = UNet(base=2, lite=True)
    torch.save({
        "model_state": model.state_dict(),
        "architecture": "lite_unet",
        "base": 2,
        "tile": 64,
        "overlap": 16,
        "threshold": 0.5,
        "seed": 42,
        "validation_dice": 0.8,
        "epoch": 4,
    }, checkpoint)
    run_manifest = tmp_path / "run_manifest.json"
    run_manifest.write_text(json.dumps({
        "training_code_commit": "abc123",
        "split_manifest_sha256": "split123",
        "pytorch_version": "test",
        "cuda_device": "test-gpu",
        "maximum_epochs": 30,
        "early_stopping_patience": 5,
        "official_test_evaluated": False,
        "results": [{
            "experiment": "fixture",
            "model": "lite_unet",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "best_epoch": 4,
            "best_validation_dice": 0.8,
            "completed_epochs": 9,
            "degradation_training": True,
        }],
    }))
    models_dir = tmp_path / "models"
    registry_path = tmp_path / "backend" / "model_registry.json"
    destination, artifact, registry = promote_checkpoint(
        checkpoint=checkpoint,
        model_id="lite_candidate",
        models_dir=models_dir,
        registry_path=registry_path,
        description="fixture candidate",
        make_default=True,
        run_manifest=run_manifest,
        experiment="fixture",
    )
    assert destination.is_file()
    assert destination.name.startswith("lite_candidate-")
    assert destination.with_suffix(".json").is_file()
    assert artifact["provenance"]["official_test_evaluated"] is False
    assert registry["default_model"] == "lite_candidate"
    assert registry["models"]["lite_candidate"]["checkpoint"].startswith("../models/")
    assert json.loads(registry_path.read_text()) == registry
