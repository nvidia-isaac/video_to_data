"""Add durable post-export intermediate cleanup jobs."""

from alembic import op

from orchestration.schema import METADATA


revision = "0009_intermediate_cleanup_jobs"
down_revision = "0008_lifecycle_summary_index"
branch_labels = None
depends_on = None


def upgrade() -> None:
    METADATA.tables["intermediate_cleanup_jobs"].create(bind=op.get_bind())


def downgrade() -> None:
    METADATA.tables["intermediate_cleanup_jobs"].drop(bind=op.get_bind())
