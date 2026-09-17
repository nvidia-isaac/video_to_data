"""Prefer an existing current export over unmet historical QC dependencies."""

from alembic import op

from orchestration.schema import sequence_status_view_sql


revision = "0005_export_status_precedence"
down_revision = "0004_campaign_inventory_count"
branch_labels = None
depends_on = None


def _replace_view(*, prefer_current_export: bool) -> None:
    bind = op.get_bind()
    bind.exec_driver_sql("DROP VIEW IF EXISTS sequence_status")
    bind.exec_driver_sql(
        sequence_status_view_sql(
            include_requests=True,
            include_effective=True,
            prefer_current_export=prefer_current_export,
            dialect_name=bind.dialect.name,
        )
    )


def upgrade() -> None:
    _replace_view(prefer_current_export=True)


def downgrade() -> None:
    _replace_view(prefer_current_export=False)
