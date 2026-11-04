from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("preview", sa.String(80), nullable=False, server_default=""),
        sa.Column("last_verdict", sa.String(8), nullable=True),
    )
    op.create_index("ix_sessions_created_at", "sessions", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_sessions_created_at", table_name="sessions")
    op.drop_table("sessions")
