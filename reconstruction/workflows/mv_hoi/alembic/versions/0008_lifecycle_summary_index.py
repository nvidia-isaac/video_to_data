"""Index active-campaign authoritative request selection."""

from alembic import op


revision = "0008_lifecycle_summary_index"
down_revision = "0007_export_effective_status"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.get_bind().exec_driver_sql(
        "CREATE INDEX IF NOT EXISTS "
        "idx_stage_requests_campaign_sequence_recency "
        "ON stage_requests(campaign_id, sequence_id, created_at DESC, id DESC)"
    )


def downgrade() -> None:
    op.get_bind().exec_driver_sql(
        "DROP INDEX IF EXISTS idx_stage_requests_campaign_sequence_recency"
    )
