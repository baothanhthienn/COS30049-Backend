from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api import model_loader
from api.core.config import settings
from api.db.session import get_db

router = APIRouter()


@router.get(
    "/api/health",
    summary="Liveness check",
    description="Returns service status, DB connectivity, model name, and version. No auth required.",
)
async def health(db: AsyncSession = Depends(get_db)):
    # Run a real query so Railway health checks verify DB connection + permissions, not just process liveness
    result = await db.execute(text("SELECT current_database() AS db_name, version() AS pg_version"))
    row = result.mappings().one()
    return {
        "status": "ok",
        "database": "connected",
        "database_name": row["db_name"],
        "model": model_loader.get_best_model_name(),
        "version": settings.model_version,
        "hosted": settings.is_hosted,
    }
