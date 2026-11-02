import logging
import uuid

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

_logger = logging.getLogger(__name__)


def _envelope(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message, "reference_id": str(uuid.uuid4())}}


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(status_code=exc.status_code, content=_envelope("HTTP_ERROR", exc.detail or "An error occurred"))


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    messages = "; ".join(f"{'.'.join(str(l) for l in e['loc'])}: {e['msg']}" for e in exc.errors())
    return JSONResponse(status_code=422, content=_envelope("VALIDATION_ERROR", messages))


async def unhandled_exception_handler(request: Request, exc: Exception):
    _logger.exception("Unhandled error: %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content=_envelope("INTERNAL_ERROR", "An unexpected error occurred"))
