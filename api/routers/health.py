from fastapi import APIRouter
from api import model_loader
from api.core.config import settings

router = APIRouter()


@router.get(
    "/api/health",
    summary="Liveness check",
    description="Returns service status, active model name, and version.",
)
def health():
    return {
        "status": "ok",
        "model": model_loader.get_best_model_name(),
        "version": settings.model_version,
    }
