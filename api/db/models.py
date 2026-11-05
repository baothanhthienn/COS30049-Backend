import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Index, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from api.db.session import Base


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    preview: Mapped[str] = mapped_column(String(80), nullable=False, default="")
    last_verdict: Mapped[str | None] = mapped_column(String(8), nullable=True)

    __table_args__ = (
        Index("ix_sessions_created_at", "created_at"),
    )


class Classification(Base):
    __tablename__ = "classifications"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    verdict: Mapped[str] = mapped_column(String(8), nullable=False)   # "BLOCK" | "ALLOW"
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    attack_family: Mapped[str | None] = mapped_column(String(64), nullable=True)
    family_scores: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON blob
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    reply: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_classifications_session_created", "session_id", "created_at"),
        Index("ix_classifications_attack_family", "attack_family"),
    )
