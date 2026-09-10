"""
FastAPI application — Prompt Injection Guardrail API.

Endpoints:
  POST /predict   — classify a single text
  GET  /stats     — running session statistics
  GET  /health    — liveness check

Run: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    PredictRequest, PredictResponse, StatsResponse,
    HealthResponse, SpanOut,
)
from api import model_loader

# In-memory session stats (resets on server restart)
_stats = {
    "total_requests": 0,
    "total_blocked":  0,
    "total_allowed":  0,
    "cluster_counts": defaultdict(int),
}


@asynccontextmanager
async def lifespan(application: FastAPI):
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
        raise HTTPException(status_code=500, detail=str(exc))

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
    total = _stats["total_requests"]
    block_rate = _stats["total_blocked"] / total if total > 0 else 0.0
    return StatsResponse(
        total_requests=total,
        total_blocked=_stats["total_blocked"],
        total_allowed=_stats["total_allowed"],
        block_rate=round(block_rate, 4),
        cluster_counts=dict(_stats["cluster_counts"]),
    )


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model="RandomForest",
        version="1.0.0",
    )
