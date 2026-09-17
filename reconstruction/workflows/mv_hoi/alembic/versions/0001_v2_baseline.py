"""Create the deployed v2 orchestration schema."""

from alembic import op

from orchestration.schema import BASELINE_METADATA, sequence_status_view_sql


revision = "0001_v2_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    BASELINE_METADATA.create_all(bind=bind)
    bind.exec_driver_sql(
        "INSERT INTO schema_metadata(key, value) VALUES ('schema_version', '2') "
        "ON CONFLICT(key) DO NOTHING"
    )
    bind.exec_driver_sql("DROP VIEW IF EXISTS sequence_status")
    bind.exec_driver_sql(sequence_status_view_sql(
        include_requests=False, dialect_name=bind.dialect.name,
    ))


def downgrade() -> None:
    bind = op.get_bind()
    bind.exec_driver_sql("DROP VIEW IF EXISTS sequence_status")
    BASELINE_METADATA.drop_all(bind=bind)
