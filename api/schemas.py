"""Pydantic request/response models for the FastAPI endpoints."""

from typing import Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)


class SpanOut(BaseModel):
    start: int
    end:   int
    label: str


class PredictResponse(BaseModel):
    verdict:        str            # "BLOCK" | "ALLOW"
    label:          int            # 1 = injection, 0 = benign
    confidence:     float          # P(injection) from RF
    cluster_id:     Optional[int]  # only when verdict == "BLOCK"
    cluster_label:  Optional[str]  # human-readable attack family
    spans:          list[SpanOut]
    decoded_text:   str            # normalised text used for feature extraction


class StatsResponse(BaseModel):
    total_requests:    int
    total_blocked:     int
    total_allowed:     int
    block_rate:        float
    cluster_counts:    dict[str, int]


class HealthResponse(BaseModel):
    status:  str
    model:   str
    version: str
