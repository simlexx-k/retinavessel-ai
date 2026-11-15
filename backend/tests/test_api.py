from io import BytesIO
import hashlib
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
        live = client.get("/health/live", headers={"X-Request-ID": "fixture-request"})
        assert live.status_code == 200
        assert live.json() == {"status": "alive"}
        assert live.headers["x-request-id"] == "fixture-request"
        assert float(live.headers["x-process-time-ms"]) >= 0
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
                "sha256": "a" * 64,
            }
        },
    }))
    monkeypatch.setenv("RETINAVESSEL_MODEL_REGISTRY", str(registry))
    monkeypatch.delenv("RETINAVESSEL_CHECKPOINT", raising=False)
    settings = Settings.from_env()
    assert settings.default_model == "lite"
    assert settings.models[0].checkpoint == tmp_path / "models/lite.pt"
    assert settings.models[0].expected_sha256 == "a" * 64


def test_registry_checkpoint_hash_is_enforced(tmp_path):
    checkpoint = tmp_path / "tiny.pt"
    make_checkpoint(checkpoint)
    actual_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    settings = Settings(
        models=(ModelSpec("tiny", checkpoint, expected_sha256=actual_hash),),
        default_model="tiny",
        device="cpu",
    )
    with TestClient(create_app(settings)) as client:
        assert client.get("/health/ready").status_code == 200

    invalid = Settings(
        models=(ModelSpec("tiny", checkpoint, expected_sha256="0" * 64),),
        default_model="tiny",
        device="cpu",
    )
    with pytest.raises(ValueError, match="does not match registry"):
        with TestClient(create_app(invalid)):
            pass


def test_api_key_and_explicit_cors_policy(tmp_path):
    checkpoint = tmp_path / "tiny.pt"
    make_checkpoint(checkpoint)
    settings = Settings(
        models=(ModelSpec("tiny", checkpoint),),
        default_model="tiny",
        device="cpu",
        api_key="fixture-secret-key",
        cors_origins=("http://localhost:3000",),
    )
    with TestClient(create_app(settings)) as client:
        assert client.get("/health/live").status_code == 200

        missing = client.get("/v1/models")
        assert missing.status_code == 401
        assert missing.headers["www-authenticate"] == "APIKey"

        wrong = client.get("/v1/models", headers={"X-API-Key": "wrong"})
        assert wrong.status_code == 401

        allowed = client.get(
            "/v1/models",
            headers={
                "Origin": "http://localhost:3000",
                "X-API-Key": "fixture-secret-key",
            },
        )
        assert allowed.status_code == 200
        assert allowed.headers["access-control-allow-origin"] == "http://localhost:3000"
        assert "X-Request-ID" in allowed.headers["access-control-expose-headers"]

        preflight = client.options(
            "/v1/models",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "X-API-Key",
            },
        )
        assert preflight.status_code == 200


def test_cors_wildcard_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="wildcard"):
        Settings(
            models=(ModelSpec("tiny", tmp_path / "unused.pt"),),
            default_model="tiny",
            cors_origins=("*",),
        )
