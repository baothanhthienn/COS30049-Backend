from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index("ix_classifications_attack_family", table_name="classifications")
    op.create_index(
        "ix_classifications_session_verdict_family",
        "classifications",
        ["session_id", "verdict", "attack_family"],
        postgresql_where="attack_family IS NOT NULL",
    )

    op.create_index(
        "ix_classifications_session_verdict_time",
        "classifications",
        ["session_id", "verdict", "created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_classifications_session_verdict_time", table_name="classifications"
    )
    op.drop_index(
        "ix_classifications_session_verdict_family", table_name="classifications"
    )
    op.create_index(
        "ix_classifications_attack_family", "classifications", ["attack_family"]
    )
