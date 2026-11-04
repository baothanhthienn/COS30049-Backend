import logging

from fastapi import APIRouter
from sqlalchemy import text

from api import model_loader
from api.core.config import settings
from api.db.session import AsyncSessionLocal, _normalise_url

router = APIRouter()
_logger = logging.getLogger(__name__)


@router.get(
    "/api/health",
    summary="Liveness check",
    description="Returns the loaded model name, model status, and DB connection scheme.",
    openapi_extra={
        "responses": {
            "200": {
                "description": "Service is alive",
                "content": {
                    "application/json": {
                        "example": {
                            "status": "ok",
                            "model": "RF",
                            "model_status": "loaded",
                            "version": "1.0.0",
                            "hosted": True,
                            "db_url_scheme": "postgresql+asyncpg",
                            "db_ssl": True,
                        }
                    }
                },
            }
        }
    },
)
def health():
    try:
        model_name = model_loader.get_best_model_name()
        model_status = "loaded"
    except Exception as exc:
        _logger.warning("Model status check failed: %s", exc)
        model_name = "unknown"
        model_status = "unavailable"

    return {
        "status": "ok",
        "model": model_name,
        "model_status": model_status,
        "version": settings.model_version,
        "hosted": settings.is_hosted,
        "db_url_scheme": _normalise_url(settings.database_url).split("://")[0],
        "db_ssl": settings.database_ssl,
    }


@router.get(
    "/api/health/db",
    summary="Database connectivity probe",
    description="Executes `SELECT 1` against the configured database and returns ok/error.",
    openapi_extra={
        "responses": {
            "200": {
                "description": "DB reachable or unreachable",
                "content": {
                    "application/json": {
                        "examples": {
                            "ok": {"value": {"db": "ok"}},
                            "error": {"value": {"db": "error"}},
                        }
                    }
                },
            }
        }
    },
)
async def health_db():
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        return {"db": "ok"}
    except Exception as exc:
        _logger.error("DB health check failed: %s", exc)
        return {"db": "error"}
