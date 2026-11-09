import logging
import time
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
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
from api.routers import chat, classify, health, sessions

_logger = logging.getLogger(__name__)

_redis: aioredis.Redis | None = None


def log_response(
    method: str,
    path: str,
    status: int,
    duration_ms: float,
    extra: dict | None = None,
) -> None:
    msg = "%s %s → %d (%.1fms)"
    args: list = [method, path, status, duration_ms]
    if extra:
        msg += " | %s"
        args.append(" ".join(f"{k}={v}" for k, v in extra.items()))
    if status >= 500:
        _logger.error(msg, *args)
    elif status >= 400:
        _logger.warning(msg, *args)
    else:
        _logger.info(msg, *args)

import re as _re

_BUILTIN_ORIGINS = {
    "https://cos-30049-frontend.vercel.app",
    "http://localhost:5173",
    "http://localhost:3000",
}
_env_origins = {o.strip() for o in settings.cors_origin.split(",") if o.strip()}
_origins = list(_BUILTIN_ORIGINS | _env_origins)

_VERCEL_PREVIEW_RE = _re.compile(
    r'^https://cos-30049-frontend-[a-z0-9]+-[a-z0-9-]+\.vercel\.app$'
)


def _is_allowed_origin(origin: str) -> bool:
    return origin in _BUILTIN_ORIGINS or origin in _env_origins or bool(_VERCEL_PREVIEW_RE.match(origin))


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _redis
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
            await conn.execute(
                __import__("sqlalchemy").text(
                    "ALTER TABLE classifications ADD COLUMN IF NOT EXISTS reply TEXT"
                )
            )
        _logger.info("DB tables verified/created")
    except Exception as exc:
        _logger.error("DB table creation failed: %s", exc)
    if settings.redis_url:
        try:
            _redis = aioredis.from_url(settings.redis_url, decode_responses=True)
            await _redis.ping()
            _logger.info("Redis connected")
        except Exception as exc:
            _logger.warning("Redis unavailable, falling back to in-memory rate limiting: %s", exc)
            _redis = None
    try:
        yield
    finally:
        if _redis:
            await _redis.aclose()
        await dispose_engine()


app = FastAPI(
    title="Prompt Injection Guardrail",
    version=settings.model_version,
    description="ML-powered prompt injection detection API.",
    lifespan=lifespan,
    swagger_ui_parameters={"tryItOutEnabled": True},
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
    allow_origin_regex=r'^https://cos-30049-frontend-[a-z0-9]+-[a-z0-9-]+\.vercel\.app$',
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    log_response(
        request.method,
        request.url.path,
        response.status_code,
        (time.monotonic() - start) * 1000,
    )
    return response


_DOCS_PATHS = {"/docs", "/redoc", "/openapi.json"}


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
    if request.url.path not in _DOCS_PATHS:
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


_fallback_buckets: dict[str, list[float]] = {}


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if (request.method == "POST" and request.url.path in ("/api/classify", "/api/chat")) or \
       (request.method == "GET" and request.url.path == "/api/health/db"):
        api_key_val = request.headers.get("x-api-key") or request.headers.get("X-API-Key")
        bucket_key = f"rl:{api_key_val if api_key_val else (request.client.host if request.client else 'unknown')}"

        if _redis is not None:
            try:
                now_ms = int(time.time() * 1000)
                window_ms = 60_000
                pipe = _redis.pipeline()
                pipe.zremrangebyscore(bucket_key, 0, now_ms - window_ms)
                pipe.zcard(bucket_key)
                pipe.zadd(bucket_key, {str(now_ms): now_ms})
                pipe.expire(bucket_key, 61)
                _, count, *_ = await pipe.execute()
                if count >= settings.rate_limit_per_minute:
                    return JSONResponse(
                        status_code=429,
                        content={"error": {"code": "RATE_LIMITED", "message": "Too many requests. Retry after 60 seconds.", "reference_id": ""}},
                    )
            except Exception as exc:
                _logger.warning("Redis rate-limit error, skipping: %s", exc)
        else:
            now = time.monotonic()
            window = _fallback_buckets.get(bucket_key, [])
            window = [t for t in window if now - t < 60]
            if len(window) >= settings.rate_limit_per_minute:
                return JSONResponse(
                    status_code=429,
                    content={"error": {"code": "RATE_LIMITED", "message": "Too many requests. Retry after 60 seconds.", "reference_id": ""}},
                )
            window.append(now)
            _fallback_buckets[bucket_key] = window

    return await call_next(request)


app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

app.include_router(chat.router)
app.include_router(classify.router)
app.include_router(sessions.router)
app.include_router(health.router)
