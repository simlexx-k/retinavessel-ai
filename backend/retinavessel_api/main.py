"""FastAPI application assembly and process lifecycle."""
from contextlib import asynccontextmanager
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .config import Settings, cors_origins_from_env
from .middleware import request_context
from .model_service import ModelRegistry
from .routes import public_router, v1_router


EXPOSED_HEADERS = [
    "Content-Disposition",
    "X-Image-Height",
    "X-Image-Width",
    "X-Inference-Ms",
    "X-Model-ID",
    "X-Process-Time-Ms",
    "X-Request-ID",
    "X-Research-Only",
    "X-Threshold",
]


def create_app(settings=None, registry_factory=ModelRegistry):
    @asynccontextmanager
    async def lifespan(application):
        resolved = settings or Settings.from_env()
        registry = registry_factory(resolved)
        application.state.settings = resolved
        application.state.registry = registry
        application.state.inference_lock = threading.Lock()
        yield
        registry.close()

    application = FastAPI(
        title="RetinaVessel AI Inference API",
        version=__version__,
        description=(
            "Research-only retinal blood-vessel segmentation. "
            "Not for diagnosis, treatment, triage, or clinical decision-making."
        ),
        lifespan=lifespan,
    )
    origins = settings.cors_origins if settings else cors_origins_from_env()
    if origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
            expose_headers=EXPOSED_HEADERS,
        )
    application.middleware("http")(request_context)
    application.include_router(public_router)
    application.include_router(v1_router)
    return application


app = create_app()
