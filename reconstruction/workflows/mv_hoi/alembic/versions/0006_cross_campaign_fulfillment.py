"""Track explicit cross-campaign request fulfillment."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

from orchestration.schema import sequence_status_view_sql


revision = "0006_cross_campaign_fulfillment"
down_revision = "0005_export_status_precedence"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns = {
        item["name"] for item in inspect(op.get_bind()).get_columns("stage_requests")
    }
    if "fulfilled_by_request_id" in columns:
        return
    bind = op.get_bind()
    bind.exec_driver_sql("DROP VIEW IF EXISTS sequence_status")
    with op.batch_alter_table("stage_requests") as batch:
        batch.add_column(
            sa.Column("fulfilled_by_request_id", sa.Integer(), nullable=True)
        )
        batch.create_foreign_key(
            "fk_stage_requests_fulfilled_by",
            "stage_requests",
            ["fulfilled_by_request_id"],
            ["id"],
        )
    op.create_index(
        "idx_stage_requests_fulfilled_by",
        "stage_requests",
        ["fulfilled_by_request_id"],
    )
    bind.exec_driver_sql(
        sequence_status_view_sql(
            include_requests=True,
            include_effective=True,
            prefer_current_export=True,
            dialect_name=bind.dialect.name,
        )
    )


def downgrade() -> None:
    columns = {
        item["name"] for item in inspect(op.get_bind()).get_columns("stage_requests")
    }
    if "fulfilled_by_request_id" not in columns:
        return
    bind = op.get_bind()
    bind.exec_driver_sql("DROP VIEW IF EXISTS sequence_status")
    op.drop_index(
        "idx_stage_requests_fulfilled_by",
        table_name="stage_requests",
    )
    with op.batch_alter_table("stage_requests") as batch:
        # SQLite does not preserve names for reflected self-referential foreign
        # keys. Batch recreation drops the key together with its column.
        batch.drop_column("fulfilled_by_request_id")
    bind.exec_driver_sql(
        sequence_status_view_sql(
            include_requests=True,
            include_effective=True,
            prefer_current_export=True,
            dialect_name=bind.dialect.name,
        )
    )
