"""Public health and protected inference API routes."""
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, HTTPException, Path, Query, Request, UploadFile
from fastapi.responses import Response

from . import __version__
from .imaging import ImageInputError, read_image, render_png, safe_download_stem
from .security import require_api_key


public_router = APIRouter()
v1_router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])


@public_router.get("/")
def root():
    return {
        "service": "retinavessel-ai",
        "version": __version__,
        "research_only": True,
        "docs": "/docs",
        "liveness": "/health/live",
        "readiness": "/health/ready",
    }


@public_router.get("/health/live")
def liveness():
    return {"status": "alive"}


def _readiness(request):
    registry = request.app.state.registry
    return {
        "status": "ready",
        "model_count": len(registry.models),
        "default_model": registry.default_model,
        "device": str(registry.device),
        "research_only": True,
    }


@public_router.get("/health")
def health_compatibility_alias(request: Request):
    return _readiness(request)


@public_router.get("/health/ready")
def readiness(request: Request):
    return _readiness(request)


@v1_router.get("/models")
def models(request: Request):
    registry = request.app.state.registry
    return {
        "default_model": registry.default_model,
        "models": registry.public_metadata(),
    }


@v1_router.post(
    "/models/{model_id}/segment",
    response_class=Response,
    responses={
        200: {"content": {"image/png": {}}},
        401: {"description": "Invalid or missing API key"},
        404: {"description": "Unknown model"},
        413: {"description": "Upload exceeds configured limits"},
        415: {"description": "Unsupported media type"},
        422: {"description": "Invalid image or request parameter"},
    },
)
def segment(
    request: Request,
    model_id: Annotated[str, Path(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")],
    file: Annotated[UploadFile, File(description="Fundus image")],
    output: Annotated[Literal["mask", "overlay"], Query()] = "overlay",
    threshold: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    overlay_alpha: Annotated[float, Query(ge=0.0, le=1.0)] = 0.45,
):
    registry = request.app.state.registry
    try:
        model = registry.get(model_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    settings = request.app.state.settings
    try:
        image = read_image(
            file.file,
            file.content_type,
            settings.max_upload_bytes,
            settings.max_image_pixels,
        )
    except ImageInputError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error
    with request.app.state.inference_lock:
        result = model.segment(image, threshold=threshold)
    payload, suffix = render_png(
        image,
        result.probability,
        output,
        result.threshold,
        overlay_alpha,
    )
    stem = safe_download_stem(file.filename)
    headers = {
        "Cache-Control": "no-store",
        "Content-Disposition": f'inline; filename="{stem}_{suffix}.png"',
        "X-Image-Height": str(image.height),
        "X-Image-Width": str(image.width),
        "X-Inference-Ms": f"{result.elapsed_ms:.3f}",
        "X-Model-ID": model.model_id,
        "X-Research-Only": "true",
        "X-Threshold": f"{result.threshold:.6f}",
    }
    return Response(content=payload, media_type="image/png", headers=headers)
