import logging

from fastapi import APIRouter
from sqlalchemy import text

from api import model_loader
from api.core.config import settings
from api.db.session import AsyncSessionLocal, _normalise_url

router = APIRouter()
_logger = logging.getLogger(__name__)


@router.get("/api/health")
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


@router.get("/api/health/db")
async def health_db():
    """Temporary: test DB connectivity and return the raw error if it fails."""
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        return {"db": "ok"}
    except Exception as exc:
        return {"db": "error", "detail": str(exc)}
