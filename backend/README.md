# RetinaVessel inference backend

This directory is an independent FastAPI application. It imports the reusable
model and tiled-inference code from `src/retinavessel`; the research/training
package does not import or depend on FastAPI.

The service is a research visualization backend, not a diagnostic or clinical
system. It loads one or more trusted `best.pt` checkpoints once at application
startup and serves native-resolution vessel masks or red overlays.

## Structure

```text
backend/
├── retinavessel_api/
│   ├── config.py          # environment and model-registry parsing
│   ├── imaging.py         # upload validation and PNG rendering
│   ├── middleware.py      # request IDs, timing and request logs
│   ├── model_service.py   # checkpoint loading and tiled inference
│   ├── routes.py          # public health and protected inference routes
│   ├── security.py        # optional API-key authentication
│   └── main.py            # FastAPI assembly and process lifecycle
├── tests/
├── model_registry.example.json
├── requirements.txt
└── requirements-dev.txt
```

## Local setup

From the repository root:

```bash
python -m venv .venv-backend
source .venv-backend/bin/activate
pip install -r backend/requirements.txt

export PYTHONPATH="$PWD/src:$PWD/backend"
export RETINAVESSEL_CHECKPOINT="$PWD/models/lite_robust_best.pt"
export RETINAVESSEL_MODEL_ID="lite_robust"
export RETINAVESSEL_DEVICE="auto"
export RETINAVESSEL_API_KEY="$(openssl rand -hex 32)"  # optional locally
export RETINAVESSEL_CORS_ORIGINS="http://localhost:3000"

uvicorn retinavessel_api.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the generated API documentation.

For several models, copy `model_registry.example.json`, set real checkpoint
paths, and configure only the registry variable:

```bash
export RETINAVESSEL_MODEL_REGISTRY="$PWD/backend/model_registry.json"
unset RETINAVESSEL_CHECKPOINT
```

Relative checkpoint paths are resolved relative to the registry JSON file.

## API

- `GET /health/live` — process liveness; never requires an API key.
- `GET /health/ready` — loaded model readiness, device and model count.
- `GET /health` — compatibility alias for readiness.
- `GET /v1/models` — model metadata and checkpoint hashes.
- `POST /v1/models/{model_id}/segment` — multipart upload returning a PNG.

Example overlay request:

```bash
curl -f -X POST \
  "http://127.0.0.1:8000/v1/models/lite_robust/segment?output=overlay" \
  -H "X-API-Key: $RETINAVESSEL_API_KEY" \
  -F "file=@fundus.png" \
  --output overlay.png
```

Use `output=mask` for the binary vessel mask. Optional query parameters are
`threshold` (0–1) and `overlay_alpha` (0–1). Responses include model,
threshold, image-size and inference-time headers and are marked
`Cache-Control: no-store` and `X-Research-Only: true`.

If `RETINAVESSEL_API_KEY` is configured, all `/v1` routes require it through
the `X-API-Key` header. Health routes remain public. API-key comparison is
constant-time, but this is only one deployment layer: use HTTPS at the reverse
proxy and do not put the key in browser code intended for untrusted users.

Every response includes `X-Request-ID` and `X-Process-Time-Ms`. A caller may
supply an `X-Request-ID` containing 1–128 letters, digits, dots, underscores or
hyphens; otherwise the server generates one.

## Configuration

| Variable | Purpose | Default |
|---|---|---:|
| `RETINAVESSEL_MODEL_REGISTRY` | JSON registry for multiple checkpoints | unset |
| `RETINAVESSEL_CHECKPOINT` | Single-checkpoint alternative | unset |
| `RETINAVESSEL_MODEL_ID` | ID for single-checkpoint mode | `default` |
| `RETINAVESSEL_DEVICE` | `auto`, `cpu`, `cuda`, or `mps` | `auto` |
| `RETINAVESSEL_MAX_UPLOAD_BYTES` | Maximum compressed upload size | 25 MiB |
| `RETINAVESSEL_MAX_IMAGE_PIXELS` | Maximum decoded width × height | 30 million |
| `RETINAVESSEL_API_KEY` | Optional key required by all `/v1` routes | unset |
| `RETINAVESSEL_CORS_ORIGINS` | Comma-separated explicit browser origins | unset |

The registry and single-checkpoint variables are mutually exclusive. The
application intentionally fails during startup if neither is configured or if
a checkpoint is missing or incompatible.

CORS wildcards are rejected. Configure exact frontend origins including
scheme and port, for example `http://localhost:3000,https://research.example`.

## Container

The reference container runs one CPU inference worker as an unprivileged user.
One worker is intentional: every worker would load its own copy of every model.
Copy the example registry, place checkpoints under the repository-level
`models/` directory, then run:

```bash
cp backend/model_registry.example.json backend/model_registry.json
mkdir -p models
export RETINAVESSEL_API_KEY="$(openssl rand -hex 32)"
docker compose -f backend/compose.yaml up --build
```

The compose service binds only to `127.0.0.1`, mounts registry/checkpoints
read-only, drops Linux capabilities and uses a read-only root filesystem.
Select a CUDA-compatible PyTorch base image and NVIDIA container runtime before
using this container definition for GPU serving.

## Tests

```bash
pip install -r backend/requirements-dev.txt
PYTHONPATH=src:backend pytest -q backend/tests
```

Tests generate a tiny random checkpoint and never require medical data.
