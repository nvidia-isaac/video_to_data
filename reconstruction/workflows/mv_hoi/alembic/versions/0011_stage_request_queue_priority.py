"""Add persistent per-stage campaign request priority."""

from alembic import op
import sqlalchemy as sa


revision = "0011_request_queue_priority"
down_revision = "0010_cleanup_bytes_bigint"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("stage_requests")}
    if "queue_priority" not in columns:
        op.add_column("stage_requests", sa.Column(
            "queue_priority", sa.Integer(), nullable=False, server_default="0",
        ))
    checks = {
        constraint.get("name")
        for constraint in inspector.get_check_constraints("stage_requests")
    }
    if bind.dialect.name != "sqlite" and "ck_requests_queue_priority" not in checks:
        op.create_check_constraint(
            "ck_requests_queue_priority", "stage_requests", "queue_priority >= 0",
        )
    indexes = {index["name"] for index in inspector.get_indexes("stage_requests")}
    if "idx_stage_requests_dispatch_priority" not in indexes:
        op.create_index(
            "idx_stage_requests_dispatch_priority",
            "stage_requests",
            ["campaign_id", "stage", "status", "queue_priority", "created_at", "id"],
        )


def downgrade() -> None:
    # This additive column is deliberately retained on downgrade.  The project
    # builds early migration tables from the current SQLAlchemy metadata, so a
    # fresh database can already contain this column before revision 0011.  A
    # destructive downgrade could not distinguish that case from a deployed
    # database upgraded from 0010, and would also break the current table check.
    pass
