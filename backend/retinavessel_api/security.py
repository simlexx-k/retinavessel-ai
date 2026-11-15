"""Optional API-key authentication for inference routes."""
import secrets

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader


api_key_header = APIKeyHeader(
    name="X-API-Key",
    scheme_name="RetinaVesselApiKey",
    description="Optional deployment API key configured by the server operator.",
    auto_error=False,
)


def require_api_key(
    request: Request,
    supplied: str | None = Security(api_key_header),
):
    expected = request.app.state.settings.api_key
    if expected is None:
        return
    if supplied is None or not secrets.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "APIKey"},
        )
