"""Prefer a successful current export as the effective sequence status."""

from alembic import op

from orchestration.schema import sequence_status_view_sql


revision = "0007_export_effective_status"
down_revision = "0006_cross_campaign_fulfillment"
branch_labels = None
depends_on = None


def _replace_view(*, prefer_successful_export: bool) -> None:
    bind = op.get_bind()
    bind.exec_driver_sql("DROP VIEW IF EXISTS sequence_status")
    bind.exec_driver_sql(
        sequence_status_view_sql(
            include_requests=True,
            include_effective=True,
            prefer_current_export=True,
            prefer_successful_export_effective=prefer_successful_export,
            dialect_name=bind.dialect.name,
        )
    )


def upgrade() -> None:
    _replace_view(prefer_successful_export=True)


def downgrade() -> None:
    _replace_view(prefer_successful_export=False)
