"""Store the independently frozen campaign inventory count."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

from orchestration.schema import sequence_status_view_sql


revision = "0004_campaign_inventory_count"
down_revision = "0003_effective_sequence_status"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if "inventory_sequence_count" in {
        item["name"] for item in inspect(op.get_bind()).get_columns("processing_campaigns")
    }:
        return
    bind = op.get_bind()
    bind.exec_driver_sql("DROP VIEW IF EXISTS sequence_status")
    with op.batch_alter_table("processing_campaigns") as batch:
        batch.add_column(sa.Column("inventory_sequence_count", sa.Integer(), nullable=True))
        batch.create_check_constraint(
            "ck_campaigns_inventory_count",
            "inventory_sequence_count IS NULL OR inventory_sequence_count >= 0",
        )
    bind.exec_driver_sql(
        sequence_status_view_sql(
            include_requests=True, include_effective=True,
            dialect_name=bind.dialect.name,
        )
    )


def downgrade() -> None:
    if "inventory_sequence_count" not in {
        item["name"] for item in inspect(op.get_bind()).get_columns("processing_campaigns")
    }:
        return
    bind = op.get_bind()
    bind.exec_driver_sql("DROP VIEW IF EXISTS sequence_status")
    with op.batch_alter_table("processing_campaigns") as batch:
        batch.drop_constraint("ck_campaigns_inventory_count", type_="check")
        batch.drop_column("inventory_sequence_count")
    bind.exec_driver_sql(
        sequence_status_view_sql(
            include_requests=True, include_effective=True,
            dialect_name=bind.dialect.name,
        )
    )
