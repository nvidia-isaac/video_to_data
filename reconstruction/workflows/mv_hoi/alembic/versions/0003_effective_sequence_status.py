"""Expose the latest effective sequence outcome."""

from alembic import op

from orchestration.schema import sequence_status_view_sql


revision = "0003_effective_sequence_status"
down_revision = "0002_campaign_requests"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    bind.exec_driver_sql("DROP VIEW IF EXISTS sequence_status")
    bind.exec_driver_sql(
        sequence_status_view_sql(
            include_requests=True, include_effective=True,
            dialect_name=bind.dialect.name,
        )
    )


def downgrade() -> None:
    bind = op.get_bind()
    bind.exec_driver_sql("DROP VIEW IF EXISTS sequence_status")
    bind.exec_driver_sql(
        sequence_status_view_sql(
            include_requests=True, include_effective=False,
            dialect_name=bind.dialect.name,
        )
    )
