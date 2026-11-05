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
from api.db.session import dispose_engine, engine
from api.db import models as _models  # noqa: F401 — registers tables on Base.metadata
from api.db.session import Base
from api.routers import classify, health, sessions

_logger = logging.getLogger(__name__)

_rate_buckets: dict[str, list[float]] = defaultdict(list)

_BUILTIN_ORIGINS = {
    "https://cos-30049-frontend.vercel.app",
    "http://localhost:5173",
    "http://localhost:3000",
}
_origins = list(_BUILTIN_ORIGINS | {o.strip() for o in settings.cors_origin.split(",") if o.strip()})


@asynccontextmanager
async def lifespan(_: FastAPI):
    _logger.info("CORS origins: %s", _origins)
    _logger.info("Hosted: %s | SSL: %s", settings.is_hosted, settings.database_ssl)
    try:
        model_loader.ensure_loaded()
        _logger.info("Model loaded: %s", model_loader.get_best_model_name())
    except Exception as exc:
        _logger.error("Model load failed (classify will be unavailable): %s", exc)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        _logger.info("DB tables verified/created")
    except Exception as exc:
        _logger.error("DB table creation failed (will rely on Alembic): %s", exc)
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

if settings.is_hosted:
    try:
        from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
        app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")
    except ImportError:
        _logger.warning("ProxyHeadersMiddleware not available")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    if "server" in response.headers:
        del response.headers["server"]
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "0"       
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
    if settings.is_hosted:
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
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
        api_key_val = request.headers.get("x-api-key") or request.headers.get("X-API-Key")
        bucket_key = f"key:{api_key_val}" if api_key_val else f"ip:{request.client.host if request.client else 'unknown'}"
        now = time.monotonic()
        window = _rate_buckets[bucket_key] = [t for t in _rate_buckets[bucket_key] if now - t < 60]
        if len(window) >= settings.rate_limit_per_minute:
            return JSONResponse(
                status_code=429,
                content={"error": {"code": "RATE_LIMITED", "message": "Too many requests. Retry after 60 seconds.", "reference_id": ""}},
            )
        _rate_buckets[bucket_key].append(now)
    return await call_next(request)


app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

app.include_router(classify.router)
app.include_router(sessions.router)
app.include_router(health.router)
