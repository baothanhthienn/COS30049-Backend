"""
Endpoints:
  POST /predict   — classify a single text
  GET  /stats     — running session statistics
  GET  /health    — liveness check
"""

import logging
import threading
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

_logger = logging.getLogger(__name__)

from api.schemas import (
    PredictRequest, PredictResponse, StatsResponse,
    HealthResponse, SpanOut,
)
from api import model_loader

# Lock guards _stats across concurrent requests; FastAPI runs handlers in a
# thread pool, so unsynchronised increments would produce silent count drift.
_stats_lock = threading.Lock()
_stats = {
    "total_requests": 0,
    "total_blocked":  0,
    "total_allowed":  0,
    "cluster_counts": defaultdict(int),
}


@asynccontextmanager
async def lifespan(application: FastAPI):
    # Lifespan loads the model once at startup; avoids a cold-load penalty on
    # the first request and surfaces missing-pkl errors before traffic arrives.
    model_loader.ensure_loaded()
    yield


app = FastAPI(
    title="Prompt Injection Guardrail",
    version="1.0.0",
    description="ML-powered prompt injection detection API.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        result = model_loader.predict(req.text)
    except Exception as exc:
        _logger.exception("predict failed")
        msg = str(exc)
        # Surface missing-pkl errors directly so the caller knows to run train.py.
        detail = msg if ("not found" in msg.lower() or "run `python" in msg.lower()) \
            else "Prediction service unavailable."
        raise HTTPException(status_code=500, detail=detail)

    with _stats_lock:
        _stats["total_requests"] += 1
        if result["verdict"] == "BLOCK":
            _stats["total_blocked"] += 1
            if result["cluster_label"]:
                _stats["cluster_counts"][result["cluster_label"]] += 1
        else:
            _stats["total_allowed"] += 1

    return PredictResponse(
        verdict=result["verdict"],
        label=result["label"],
        confidence=result["confidence"],
        cluster_id=result["cluster_id"],
        cluster_label=result["cluster_label"],
        spans=[SpanOut(**s) for s in result["spans"]],
        decoded_text=result["decoded_text"],
    )


@app.get("/stats", response_model=StatsResponse)
def stats():
    with _stats_lock:
        total = _stats["total_requests"]
        blocked = _stats["total_blocked"]
        allowed = _stats["total_allowed"]
        clusters = dict(_stats["cluster_counts"])
    block_rate = blocked / total if total > 0 else 0.0
    return StatsResponse(
        total_requests=total,
        total_blocked=blocked,
        total_allowed=allowed,
        block_rate=round(block_rate, 4),
        cluster_counts=clusters,
    )


@app.get("/health", response_model=HealthResponse)
def health():
    # Reflect the actual best model so the health check stays accurate after retraining.
    model_name = model_loader.get_best_model_name()
    return HealthResponse(
        status="ok",
        model=model_name,
        version="1.0.0",
    )