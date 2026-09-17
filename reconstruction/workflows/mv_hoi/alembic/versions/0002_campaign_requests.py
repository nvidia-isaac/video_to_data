"""Add durable campaigns, requests, and revalidation provenance."""

from alembic import op
import sqlalchemy as sa

from orchestration.schema import BASELINE_METADATA, METADATA, sequence_status_view_sql


revision = "0002_campaign_requests"
down_revision = "0001_v2_baseline"
branch_labels = None
depends_on = None


def _rebuild_execution_stage(*, include_revalidation: bool) -> None:
    old = BASELINE_METADATA.tables["workflow_executions"]
    values = "'calibration','preprocess','reconstruction','export'"
    if include_revalidation:
        values += ",'revalidation'"
    if op.get_bind().dialect.name != "sqlite":
        op.drop_constraint(
            "ck_executions_stage", "workflow_executions", type_="check",
        )
        op.create_check_constraint(
            "ck_executions_stage",
            "workflow_executions",
            f"pipeline_stage IN ({values})",
        )
        return
    with op.batch_alter_table(
        "workflow_executions", recreate="always", copy_from=old,
    ) as batch:
        batch.drop_constraint("ck_executions_stage", type_="check")
        batch.create_check_constraint(
            "ck_executions_stage", f"pipeline_stage IN ({values})"
        )


def _add_request_provenance(table_name: str) -> None:
    with op.batch_alter_table(table_name) as batch:
        batch.add_column(sa.Column("request_id", sa.Integer(), nullable=True))
        batch.create_foreign_key(
            f"fk_{table_name}_request", "stage_requests", ["request_id"], ["id"]
        )
    op.create_index(
        f"uq_{table_name}_request_id", table_name, ["request_id"], unique=True
    )


def upgrade() -> None:
    bind = op.get_bind()
    bind.exec_driver_sql("DROP VIEW IF EXISTS sequence_status")
    _rebuild_execution_stage(include_revalidation=True)

    METADATA.tables["processing_campaigns"].create(bind=bind)
    METADATA.tables["stage_requests"].create(bind=bind)

    for table_name in (
        "calibration_runs", "preprocess_runs", "reconstruction_runs",
    ):
        _add_request_provenance(table_name)

    old_export = BASELINE_METADATA.tables["export_runs"]
    with op.batch_alter_table(
        "export_runs", recreate="always", copy_from=old_export,
    ) as batch:
        batch.add_column(sa.Column("request_id", sa.Integer(), nullable=True))
        batch.create_foreign_key(
            "fk_export_runs_request", "stage_requests", ["request_id"], ["id"]
        )
        batch.drop_constraint("ck_export_authorization_type", type_="check")
        batch.drop_constraint("ck_export_authorization_semantics", type_="check")
        batch.create_check_constraint(
            "ck_export_authorization_type",
            "authorization_type IN ('QC','MANUAL_OVERRIDE','LEGACY','REVALIDATION')",
        )
        batch.create_check_constraint(
            "ck_export_authorization_semantics",
            "(authorization_type='QC' AND qc_review_id IS NOT NULL "
            "AND override_reason IS NULL) OR "
            "(authorization_type='MANUAL_OVERRIDE' AND requested_by IS NOT NULL "
            "AND trim(requested_by) != '' AND override_reason IS NOT NULL "
            "AND trim(override_reason) != '') OR "
            "(authorization_type='LEGACY' AND trigger='MIGRATION') OR "
            "authorization_type='REVALIDATION'",
        )
    op.create_index(
        "uq_export_runs_request_id", "export_runs", ["request_id"], unique=True
    )
    bind.exec_driver_sql(sequence_status_view_sql(
        include_requests=True, dialect_name=bind.dialect.name,
    ))


def downgrade() -> None:
    bind = op.get_bind()
    if bind.exec_driver_sql(
        "SELECT EXISTS(SELECT 1 FROM stage_requests) OR "
        "EXISTS(SELECT 1 FROM workflow_executions WHERE pipeline_stage='revalidation') OR "
        "EXISTS(SELECT 1 FROM export_runs WHERE authorization_type='REVALIDATION')"
    ).scalar():
        raise RuntimeError(
            "Cannot remove campaign/request schema after scheduling data exists"
        )
    bind.exec_driver_sql("DROP VIEW IF EXISTS sequence_status")

    op.drop_index("uq_export_runs_request_id", table_name="export_runs")
    head_export = METADATA.tables["export_runs"]
    with op.batch_alter_table(
        "export_runs", recreate="always", copy_from=head_export,
    ) as batch:
        batch.drop_constraint("ck_export_authorization_type", type_="check")
        batch.drop_constraint("ck_export_authorization_semantics", type_="check")
        batch.create_check_constraint(
            "ck_export_authorization_type",
            "authorization_type IN ('QC','MANUAL_OVERRIDE','LEGACY')",
        )
        batch.create_check_constraint(
            "ck_export_authorization_semantics",
            "(authorization_type='QC' AND qc_review_id IS NOT NULL "
            "AND override_reason IS NULL) OR "
            "(authorization_type='MANUAL_OVERRIDE' AND requested_by IS NOT NULL "
            "AND trim(requested_by) != '' AND override_reason IS NOT NULL "
            "AND trim(override_reason) != '') OR "
            "(authorization_type='LEGACY' AND trigger='MIGRATION')",
        )
        batch.drop_constraint("fk_export_runs_request", type_="foreignkey")
        batch.drop_column("request_id")

    for table_name in (
        "reconstruction_runs", "preprocess_runs", "calibration_runs",
    ):
        op.drop_index(f"uq_{table_name}_request_id", table_name=table_name)
        with op.batch_alter_table(table_name) as batch:
            batch.drop_constraint(f"fk_{table_name}_request", type_="foreignkey")
            batch.drop_column("request_id")

    METADATA.tables["stage_requests"].drop(bind=bind)
    METADATA.tables["processing_campaigns"].drop(bind=bind)
    _rebuild_execution_stage(include_revalidation=False)
    bind.exec_driver_sql(sequence_status_view_sql(
        include_requests=False, dialect_name=bind.dialect.name,
    ))
