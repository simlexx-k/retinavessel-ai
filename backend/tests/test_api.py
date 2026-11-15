from io import BytesIO
import json

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest
import torch

from retinavessel.model import UNet
from retinavessel_api.config import ModelSpec, Settings
from retinavessel_api.main import create_app


@pytest.fixture(scope="module", autouse=True)
def limit_torch_threads_for_tiny_test_model():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_checkpoint(path):
    model = UNet(base=2, lite=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "architecture": "lite_unet",
            "base": 2,
            "tile": 64,
            "overlap": 16,
            "threshold": 0.5,
            "seed": 42,
            "validation_dice": 0.75,
            "epoch": 3,
        },
        path,
    )


def png_bytes(size=(64, 64)):
    array = np.full((size[1], size[0], 3), 100, dtype=np.uint8)
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_registry_and_segmentation(tmp_path):
    checkpoint = tmp_path / "tiny.pt"
    make_checkpoint(checkpoint)
    settings = Settings(
        models=(ModelSpec("tiny", checkpoint, "test model"),),
        default_model="tiny",
        device="cpu",
        max_upload_bytes=1024 * 1024,
        max_image_pixels=4096,
    )
    with TestClient(create_app(settings)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {
            "status": "ready",
            "model_count": 1,
            "default_model": "tiny",
            "device": "cpu",
            "research_only": True,
        }
        models = client.get("/v1/models").json()
        assert models["models"][0]["id"] == "tiny"
        assert models["models"][0]["parameters"] > 0
        assert len(models["models"][0]["checkpoint_sha256"]) == 64

        response = client.post(
            "/v1/models/tiny/segment?output=mask",
            files={"file": ("retina.png", png_bytes(), "image/png")},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.headers["x-model-id"] == "tiny"
        assert response.headers["x-research-only"] == "true"
        with Image.open(BytesIO(response.content)) as result:
            assert result.mode == "L"
            assert result.size == (64, 64)

        overlay = client.post(
            "/v1/models/tiny/segment?output=overlay&overlay_alpha=0.25",
            files={"file": ("retina.png", png_bytes(), "image/png")},
        )
        assert overlay.status_code == 200
        with Image.open(BytesIO(overlay.content)) as result:
            assert result.mode == "RGB"
            assert result.size == (64, 64)


def test_upload_and_model_errors(tmp_path):
    checkpoint = tmp_path / "tiny.pt"
    make_checkpoint(checkpoint)
    settings = Settings(
        models=(ModelSpec("tiny", checkpoint),),
        default_model="tiny",
        device="cpu",
        max_upload_bytes=128,
        max_image_pixels=4096,
    )
    with TestClient(create_app(settings)) as client:
        unknown = client.post(
            "/v1/models/missing/segment",
            files={"file": ("retina.png", png_bytes(), "image/png")},
        )
        assert unknown.status_code == 404

        unsupported = client.post(
            "/v1/models/tiny/segment",
            files={"file": ("notes.txt", b"not an image", "text/plain")},
        )
        assert unsupported.status_code == 415

        oversized = client.post(
            "/v1/models/tiny/segment",
            files={"file": ("large.png", b"x" * 129, "image/png")},
        )
        assert oversized.status_code == 413


def test_registry_configuration_resolves_relative_checkpoints(tmp_path, monkeypatch):
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({
        "default_model": "lite",
        "models": {
            "lite": {
                "checkpoint": "models/lite.pt",
                "description": "fixture",
            }
        },
    }))
    monkeypatch.setenv("RETINAVESSEL_MODEL_REGISTRY", str(registry))
    monkeypatch.delenv("RETINAVESSEL_CHECKPOINT", raising=False)
    settings = Settings.from_env()
    assert settings.default_model == "lite"
    assert settings.models[0].checkpoint == tmp_path / "models/lite.pt"
