from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "classifications",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("verdict", sa.String(8), nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("attack_family", sa.String(64), nullable=True),
        sa.Column("family_scores", sa.Text, nullable=True),
        sa.Column("model_version", sa.String(32), nullable=False),
    )
    op.create_index("ix_classifications_session_id", "classifications", ["session_id"])
    op.create_index(
        "ix_classifications_session_created",
        "classifications",
        ["session_id", "created_at"],
    )
    op.create_index(
        "ix_classifications_attack_family",
        "classifications",
        ["attack_family"],
    )


def downgrade() -> None:
    op.drop_index("ix_classifications_attack_family", table_name="classifications")
    op.drop_index("ix_classifications_session_created", table_name="classifications")
    op.drop_index("ix_classifications_session_id", table_name="classifications")
    op.drop_table("classifications")
