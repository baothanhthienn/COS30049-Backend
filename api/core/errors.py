import logging
import uuid

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.core.config import settings

_logger = logging.getLogger(__name__)

_ALLOWED_ORIGINS = {
    "https://cos-30049-frontend.vercel.app",
    "http://localhost:5173",
    "http://localhost:3000",
} | {o.strip() for o in settings.cors_origin.split(",") if o.strip()}


def _envelope(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message, "reference_id": str(uuid.uuid4())}}


def _cors_headers(request: Request) -> dict:
    """Add CORS headers to error responses so they are not blocked by the browser."""
    origin = request.headers.get("origin", "")
    if origin in _ALLOWED_ORIGINS:
        return {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Credentials": "true",
        }
    return {}


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=_envelope("HTTP_ERROR", exc.detail or "An error occurred"),
        headers=_cors_headers(request),
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    messages = "; ".join(f"{'.'.join(str(l) for l in e['loc'])}: {e['msg']}" for e in exc.errors())
    return JSONResponse(
        status_code=422,
        content=_envelope("VALIDATION_ERROR", messages),
        headers=_cors_headers(request),
    )


async def unhandled_exception_handler(request: Request, exc: Exception):
    _logger.exception("Unhandled error: %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content=_envelope("INTERNAL_ERROR", "An unexpected error occurred"),
        headers=_cors_headers(request),
    )
