import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api import model_loader
from api.core.config import settings
from api.core.errors import (
    http_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from api.db.session import dispose_engine
from api.routers import classify, health, sessions

_logger = logging.getLogger(__name__)

_rate_buckets: dict[str, list[float]] = defaultdict(list)

_origins = [o.strip() for o in settings.cors_origin.split(",") if o.strip()]


@asynccontextmanager
async def lifespan(_: FastAPI):
    _logger.info("Allowed origins: %s", ", ".join(_origins))
    _logger.info("Hosted: %s | SSL: %s", settings.is_hosted, settings.database_ssl)
    try:
        model_loader.ensure_loaded()
        _logger.info("Model loaded: %s", model_loader.get_best_model_name())
    except Exception as exc:
        # Log clearly but don't crash — health endpoint will report model status
        _logger.error("Model load failed (classify will be unavailable): %s", exc)
    try:
        yield
    finally:
        await dispose_engine()


app = FastAPI(
    title="Prompt Injection Guardrail",
    version=settings.model_version,
    description="ML-powered prompt injection detection API.",
    lifespan=lifespan,
)

# ProxyHeadersMiddleware — real client IP behind Railway's load balancer
if settings.is_hosted:
    try:
        from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
        app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")
    except ImportError:
        _logger.warning("ProxyHeadersMiddleware not available — client IPs may show as proxy IP")

# CORS lockdown — origins from env only, no wildcard
# Re-read at app creation time (not just import time) so Railway env vars are picked up
_cors_origins = [o.strip() for o in settings.cors_origin.split(",") if o.strip()]
_logger.info("CORS origins: %s", _cors_origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)


@app.middleware("http")
async def remove_server_header(request: Request, call_next):
    response = await call_next(request)
    if "server" in response.headers:
        del response.headers["server"]
    return response


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > settings.max_body_bytes:
        return JSONResponse(
            status_code=413,
            content={"error": {"code": "PAYLOAD_TOO_LARGE", "message": f"Body exceeds {settings.max_body_bytes} bytes", "reference_id": ""}},
        )
    return await call_next(request)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.method == "POST" and request.url.path == "/api/classify":
        ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window = _rate_buckets[ip] = [t for t in _rate_buckets[ip] if now - t < 60]
        if len(window) >= settings.rate_limit_per_minute:
            return JSONResponse(
                status_code=429,
                content={"error": {"code": "RATE_LIMITED", "message": "Too many requests. Retry after 60 seconds.", "reference_id": ""}},
            )
        _rate_buckets[ip].append(now)
    return await call_next(request)


app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

app.include_router(classify.router)
app.include_router(sessions.router)
app.include_router(health.router)
