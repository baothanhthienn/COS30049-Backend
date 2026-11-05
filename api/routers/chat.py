import asyncio
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from groq import Groq
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.auth import require_api_key
from api.core.config import settings
from api.db.models import Classification
from api.db.session import get_db
from api.ml import pipeline

router = APIRouter(dependencies=[Depends(require_api_key)])

_executor = ThreadPoolExecutor(max_workers=settings.inference_concurrency)
_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(settings.inference_concurrency)
    return _semaphore


def _get_groq_client() -> Groq:
    if not settings.groq_api_key:
        raise HTTPException(status_code=503, detail="Chat service unavailable — GROQ_API_KEY not configured")
    return Groq(api_key=settings.groq_api_key)


_ATTACK_LABELS = {
    "instruction_override": "Instruction Override",
    "role_play_jailbreak": "Role-Play Jailbreak",
    "context_exfiltration": "Context Exfiltration",
    "encoding_obfuscation": "Encoding / Obfuscation",
    "delimiter_escape": "Delimiter Escape",
    "indirect_injection": "Indirect Injection",
}


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=10_000)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10_000)
    session_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    history: list[HistoryTurn] = Field(default_factory=list, max_length=20)


class ChatResponse(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    verdict: str
    confidence: float
    attack_family: str | None
    family_scores: dict[str, float]
    reply: str
    blocked_reason: str | None


def _groq_chat(client: Groq, messages: list[dict]) -> str:
    completion = client.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        temperature=0.7,
        max_completion_tokens=1024,
        stream=False,
    )
    return completion.choices[0].message.content or ""


def _build_block_explanation(attack_family: str | None, family_scores: dict, confidence: float) -> str:
    label = _ATTACK_LABELS.get(attack_family or "", attack_family or "unknown")
    top_scores = sorted(family_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    scores_text = ", ".join(f"{_ATTACK_LABELS.get(k, k)} ({v:.0%})" for k, v in top_scores)
    return (
        f"You are a security assistant explaining why a prompt was blocked by an AI guardrail.\n\n"
        f"The classifier detected a **{label}** attack with {confidence:.0%} confidence.\n"
        f"Top attack signals: {scores_text}.\n\n"
        f"In 2-3 sentences, explain clearly to the user why this type of prompt is flagged, "
        f"what the specific threat pattern is, and what a safe alternative would look like."
    )


@router.post(
    "/api/chat",
    response_model=ChatResponse,
    summary="Guardrailed chat",
    description=(
        "Run the two-stage ML guardrail on the message first. "
        "ALLOW → pass to the LLM and return its reply. "
        "BLOCK → ask the LLM to explain the attack; return the explanation with attack details."
    ),
)
async def chat(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    client = _get_groq_client()

    sem = _get_semaphore()
    if sem._value <= 0:
        raise HTTPException(status_code=503, detail="Server busy — too many concurrent requests")

    loop = asyncio.get_running_loop()
    async with sem:
        result = await loop.run_in_executor(_executor, pipeline.run, req.message)

    record = Classification(
        session_id=req.session_id,
        text=req.message,
        verdict=result["verdict"],
        confidence=result["confidence"],
        attack_family=result["attack_family"],
        family_scores=json.dumps(result["family_scores"]),
        model_version=settings.model_version,
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)
    blocked_reason: str | None = None

    try:
        if result["verdict"] == "ALLOW":
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                *[{"role": t.role, "content": t.content} for t in req.history[-10:]],
                {"role": "user", "content": req.message},
            ]
            reply = await loop.run_in_executor(_executor, _groq_chat, client, messages)
        else:
            system_prompt = _build_block_explanation(
                result["attack_family"],
                result["family_scores"],
                result["confidence"],
            )
            messages = [{"role": "system", "content": system_prompt}]
            blocked_reason = await loop.run_in_executor(_executor, _groq_chat, client, messages)
            reply = f"Your message was blocked. {blocked_reason}"
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM service error: {type(exc).__name__}") from exc

    return ChatResponse(
        id=record.id,
        session_id=record.session_id,
        verdict=result["verdict"],
        confidence=result["confidence"],
        attack_family=result["attack_family"],
        family_scores=result["family_scores"],
        reply=reply,
        blocked_reason=blocked_reason,
    )
