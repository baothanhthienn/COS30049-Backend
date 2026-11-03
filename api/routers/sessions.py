import json
import uuid
from collections import Counter
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.auth import require_api_key
from api.db.models import Classification
from api.db.session import get_db

router = APIRouter(dependencies=[Depends(require_api_key)])

class HistoryItem(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    text: str
    created_at: datetime
    verdict: str
    confidence: float
    attack_family: str | None
    family_scores: dict[str, float]
    model_version: str


class HistoryResponse(BaseModel):
    session_id: uuid.UUID
    total: int
    limit: int
    offset: int
    items: list[HistoryItem]


@router.get(
    "/api/sessions/{session_id}/history",
    response_model=HistoryResponse,
    summary="Paginated classification history for a session",
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
        # Parameterised ilike — safe against SQL injection
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
            session_id=r.session_id,
            text=r.text,
            created_at=r.created_at,
            verdict=r.verdict,
            confidence=r.confidence,
            attack_family=r.attack_family,
            family_scores=json.loads(r.family_scores) if r.family_scores else {},
            model_version=r.model_version,
        )
        for r in rows
    ]
    return HistoryResponse(
        session_id=session_id,
        total=total,
        limit=limit,
        offset=offset,
        items=items,
    )


class TrendPoint(BaseModel):
    id: uuid.UUID
    created_at: datetime
    verdict: str
    confidence: float
    attack_family: str | None


class FamilyCount(BaseModel):
    family: str
    count: int


class TrendSummary(BaseModel):
    total: int
    blocked: int
    benign: int
    top_family: str | None
    block_rate_rolling: float          # block rate over the last 10 requests
    family_breakdown: list[FamilyCount]  # ranked attack families


class TrendResponse(BaseModel):
    session_id: uuid.UUID
    points: list[TrendPoint]
    summary: TrendSummary


@router.get(
    "/api/sessions/{session_id}/trend",
    response_model=TrendResponse,
    summary="Trend analysis for a session",
    description=(
        "Returns all classification points for the session plus a summary: "
        "total/blocked/benign counts, top attack family, rolling block rate "
        "(last 10 requests), and a full family breakdown ranked by frequency."
    ),
)
async def trend(session_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    # Fetch all points ordered oldest-first so rolling window is easy to compute
    rows_q = await db.execute(
        select(Classification)
        .where(Classification.session_id == session_id)
        .order_by(Classification.created_at.asc())
    )
    rows = rows_q.scalars().all()

    if not rows:
        return TrendResponse(
            session_id=session_id,
            points=[],
            summary=TrendSummary(
                total=0,
                blocked=0,
                benign=0,
                top_family=None,
                block_rate_rolling=0.0,
                family_breakdown=[],
            ),
        )

    points = [
        TrendPoint(
            id=r.id,
            created_at=r.created_at,
            verdict=r.verdict,
            confidence=r.confidence,
            attack_family=r.attack_family,
        )
        for r in rows
    ]

    total = len(rows)
    blocked = sum(1 for r in rows if r.verdict == "BLOCK")
    benign = total - blocked

    # Rolling block rate: last 10 requests
    window = rows[-10:]
    window_blocked = sum(1 for r in window if r.verdict == "BLOCK")
    block_rate_rolling = round(window_blocked / len(window), 4)

    # Attack family breakdown (blocked rows only)
    family_counts: Counter[str] = Counter(
        r.attack_family for r in rows if r.attack_family and r.verdict == "BLOCK"
    )
    family_breakdown = [
        FamilyCount(family=fam, count=cnt)
        for fam, cnt in family_counts.most_common()
    ]
    top_family = family_breakdown[0].family if family_breakdown else None

    return TrendResponse(
        session_id=session_id,
        points=points,
        summary=TrendSummary(
            total=total,
            blocked=blocked,
            benign=benign,
            top_family=top_family,
            block_rate_rolling=block_rate_rolling,
            family_breakdown=family_breakdown,
        ),
    )

@router.delete(
    "/api/sessions/{session_id}",
    status_code=204,
    summary="Delete all records for a session",
)
async def delete_session(session_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        delete(Classification).where(Classification.session_id == session_id)
    )
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    await db.commit()
