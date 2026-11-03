import logging

from fastapi import APIRouter

from api import model_loader
from api.core.config import settings

router = APIRouter()
_logger = logging.getLogger(__name__)


@router.get(
    "/api/health",
    summary="Liveness check",
    description="Returns service status, model name, and version. No auth required. Always returns 200.",
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
    }
