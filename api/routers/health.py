import logging

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api import model_loader
from api.core.config import settings
from api.db.session import get_db

router = APIRouter()
_logger = logging.getLogger(__name__)


@router.get(
    "/api/health",
    summary="Liveness check",
    description="Returns service status, DB connectivity, model name, and version. No auth required.",
)
async def health(db: AsyncSession = Depends(get_db)):
    db_status = "unknown"
    db_name = None

    try:
        result = await db.execute(
            text("SELECT current_database() AS db_name, version() AS pg_version")
        )
        row = result.mappings().one()
        db_status = "connected"
        db_name = row["db_name"]
    except Exception as exc:
        _logger.warning("Health DB check failed: %s", exc)
        db_status = "unreachable"

    return {
        "status": "ok",
        "database": db_status,
        "database_name": db_name,
        "model": model_loader.get_best_model_name(),
        "version": settings.model_version,
        "hosted": settings.is_hosted,
    }
