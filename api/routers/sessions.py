import json
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import case, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.auth import require_api_key
from api.db.models import Classification
from api.db.session import get_db

router = APIRouter(dependencies=[Depends(require_api_key)])


class HistoryItem(BaseModel):
    id: uuid.UUID
    created_at: datetime
    verdict: str
    confidence: float
    attack_family: str | None
    family_scores: dict[str, float]


class HistoryResponse(BaseModel):
    session_id: uuid.UUID
    total: int
    items: list[HistoryItem]


class TrendPoint(BaseModel):
    hour: str
    total: int
    blocked: int
    block_rate: float


class TrendResponse(BaseModel):
    session_id: uuid.UUID
    points: list[TrendPoint]


@router.get(
    "/api/sessions/{session_id}/history",
    response_model=HistoryResponse,
    summary="Paginated classification history for a session",
    description="Returns classifications for the given session, ordered newest first.",
    openapi_extra={"responses": {"200": {"description": "Paginated history"}}}
)
async def history(
    session_id: uuid.UUID,
    limit: int = 20,
    offset: int = 0,
    q: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    base_filter = Classification.session_id == session_id
    if q:
        base_filter = base_filter & Classification.text.ilike(f"%{q}%")

    total_q = await db.execute(select(func.count()).where(base_filter))
    total = total_q.scalar_one()

    rows_q = await db.execute(
        select(Classification)
        .where(base_filter)
        .order_by(Classification.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    rows = rows_q.scalars().all()

    items = [
        HistoryItem(
            id=r.id,
            created_at=r.created_at,
            verdict=r.verdict,
            confidence=r.confidence,
            attack_family=r.attack_family,
            family_scores=json.loads(r.family_scores) if r.family_scores else {},
        )
        for r in rows
    ]
    return HistoryResponse(session_id=session_id, total=total, items=items)


@router.get(
    "/api/sessions/{session_id}/trend",
    response_model=TrendResponse,
    summary="Hourly block-rate trend for a session",
    description="Groups classifications by hour and returns block rate per bucket.",
    openapi_extra={"responses": {"200": {"description": "Trend data points"}}}
)
async def trend(session_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(
            func.date_trunc("hour", Classification.created_at).label("hour"),
            func.count().label("total"),
            func.sum(
                case((Classification.verdict == "BLOCK", 1), else_=0)
            ).label("blocked"),
        )
        .where(Classification.session_id == session_id)
        .group_by("hour")
        .order_by("hour")
    )
    rows = result.all()
    points = [
        TrendPoint(
            hour=row.hour.strftime("%Y-%m-%dT%H:00:00Z"),
            total=row.total,
            blocked=row.blocked or 0,
            block_rate=round((row.blocked or 0) / row.total, 4),
        )
        for row in rows
    ]
    return TrendResponse(session_id=session_id, points=points)


@router.delete(
    "/api/sessions/{session_id}",
    status_code=204,
    summary="Delete all records for a session",
    description="Permanently removes all classifications associated with the session.",
    openapi_extra={"responses": {"204": {"description": "Deleted"}}}
)
async def delete_session(session_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        delete(Classification).where(Classification.session_id == session_id)
    )
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    await db.commit()
