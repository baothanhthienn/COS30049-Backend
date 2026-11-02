import secrets
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from api.core.config import settings

_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: str | None = Security(_header_scheme)) -> None:
   
    if not settings.api_key:
        return  # dev mode: no key configured, skip check
    if not api_key or not secrets.compare_digest(api_key, settings.api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
