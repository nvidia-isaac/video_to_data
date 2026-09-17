"""SQLAlchemy Core metadata for the MV-HOI orchestration database."""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.functions import FunctionElement


RUN_STATUSES = "'SUBMITTING','RUNNING','SUCCEEDED','FAILED','CANCELED','UNKNOWN','SKIPPED'"
EXECUTION_STATUSES = "'SUBMITTING','RUNNING','SUCCEEDED','FAILED','CANCELED','UNKNOWN'"
REQUEST_STATUSES = (
    "'PENDING','BLOCKED','RESERVED','SUBMITTED','RUNNING','SUCCEEDED',"
    "'FAILED','CANCELED','UNKNOWN'"
)
ACTIVE_REQUEST_STATUSES = (
    "'PENDING','BLOCKED','RESERVED','SUBMITTED','RUNNING','UNKNOWN'"
)

class UtcNowText(FunctionElement):
    """Backend-specific UTC timestamp formatted as the existing text schema."""

    type = Text()
    inherit_cache = True


@compiles(UtcNowText, "sqlite")
def _compile_sqlite_now(_element, _compiler, **_kw):
    return "(strftime('%Y-%m-%dT%H:%M:%fZ','now'))"


@compiles(UtcNowText, "postgresql")
def _compile_postgresql_now(_element, _compiler, **_kw):
    return (
        "to_char(clock_timestamp() AT TIME ZONE 'UTC', "
        """'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"')"""
    )


UTC_NOW_TEXT = UtcNowText()


def build_metadata(*, include_scheduling: bool) -> MetaData:
    """Build baseline-v2 or current-head metadata.

    ``include_scheduling=False`` exactly describes the deployed v2 schema.
    The head schema adds campaigns, durable requests, and request provenance.
    """
    metadata = MetaData()

    Table(
        "schema_metadata", metadata,
        Column("key", Text, primary_key=True),
        Column("value", Text, nullable=False),
    )
    Table(
        "pipeline_versions", metadata,
        Column("version", Text, primary_key=True),
        Column("message", Text),
        Column("registry_metadata_json", Text),
        Column("created_at", Text, nullable=False, server_default=UTC_NOW_TEXT),
    )
    sequences = Table(
        "sequences", metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("dataset", Text, nullable=False),
        Column("sequence_name", Text, nullable=False),
        Column("sequence_kind", Text, nullable=False, server_default="hoi"),
        Column("source_uri", Text),
        Column("hoi_metadata_uri", Text),
        Column("calibration_sequence_id", Integer, ForeignKey("sequences.id")),
        Column("object_id", Text),
        Column("discovered_at", Text, nullable=False, server_default=UTC_NOW_TEXT),
        Column("updated_at", Text, nullable=False, server_default=UTC_NOW_TEXT),
        CheckConstraint("sequence_kind IN ('hoi','calibration')", name="ck_sequences_kind"),
        UniqueConstraint("dataset", "sequence_name", name="uq_sequences_identity"),
    )
    Table(
        "blacklisted_sequences", metadata,
        Column(
            "sequence_id", Integer, ForeignKey("sequences.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        Column("reason", Text),
        Column("created_by", Text),
        Column("created_at", Text, nullable=False, server_default=UTC_NOW_TEXT),
    )

    execution_stages = "'calibration','preprocess','reconstruction','export'"
    if include_scheduling:
        execution_stages += ",'revalidation'"
    executions = Table(
        "workflow_executions", metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("pipeline_stage", Text, nullable=False),
        Column("pipeline_version", Text, ForeignKey("pipeline_versions.version")),
        Column("backend", Text, nullable=False, server_default="osmo"),
        Column("workflow_name", Text, nullable=False, unique=True),
        Column("osmo_workflow_id", Text),
        Column("workflow_spec_path", Text),
        Column("pool", Text),
        Column("status", Text, nullable=False, server_default="SUBMITTING"),
        Column("details", Text),
        Column("last_query_payload_json", Text),
        Column("submitted_at", Text, nullable=False, server_default=UTC_NOW_TEXT),
        Column("started_at", Text),
        Column("completed_at", Text),
        Column("last_refreshed_at", Text),
        CheckConstraint(f"pipeline_stage IN ({execution_stages})", name="ck_executions_stage"),
        CheckConstraint("backend IN ('osmo','local')", name="ck_executions_backend"),
        CheckConstraint(f"status IN ({EXECUTION_STATUSES})", name="ck_executions_status"),
    )
    Index(
        "idx_workflow_executions_osmo_id", executions.c.osmo_workflow_id,
        unique=True,
        sqlite_where=(executions.c.osmo_workflow_id.is_not(None)
                      & (executions.c.osmo_workflow_id != "")),
        postgresql_where=(executions.c.osmo_workflow_id.is_not(None)
                          & (executions.c.osmo_workflow_id != "")),
    )
    Index("idx_workflow_executions_status", executions.c.status, executions.c.pipeline_stage)

    if include_scheduling:
        campaigns = Table(
            "processing_campaigns", metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("name", Text, nullable=False, unique=True),
            Column("campaign_type", Text, nullable=False),
            Column("dataset", Text, nullable=False),
            Column("status", Text, nullable=False),
            Column("phase", Text, nullable=False),
            Column("pipeline_version", Text, ForeignKey("pipeline_versions.version"), nullable=False),
            Column("inventory_uri", Text),
            Column("inventory_sha256", Text),
            Column("inventory_sequence_count", Integer),
            Column("configuration_uri", Text),
            Column("configuration_sha256", Text),
            Column("output_uri", Text, nullable=False),
            Column("canary_report_uri", Text),
            Column("canary_report_sha256", Text),
            Column("canary_approved_by", Text),
            Column("canary_approved_at", Text),
            Column("created_by", Text, nullable=False),
            Column("created_at", Text, nullable=False),
            Column("updated_at", Text, nullable=False),
            Column("frozen_at", Text),
            Column("started_at", Text),
            Column("completed_at", Text),
            CheckConstraint(
                "campaign_type IN ('LEGACY_REVALIDATION','BACKLOG_REPROCESSING','REMEDIATION')",
                name="ck_campaigns_type",
            ),
            CheckConstraint(
                "status IN ('DRAFT','FROZEN','RUNNING','SUCCEEDED',"
                "'COMPLETED_WITH_FAILURES','CANCELED')",
                name="ck_campaigns_status",
            ),
            CheckConstraint("phase IN ('CANARY','BULK','COMPLETE')", name="ck_campaigns_phase"),
            CheckConstraint(
                "status='DRAFT' OR (inventory_uri IS NOT NULL AND inventory_sha256 IS NOT NULL "
                "AND configuration_uri IS NOT NULL AND configuration_sha256 IS NOT NULL)",
                name="ck_campaigns_frozen_identity",
            ),
            CheckConstraint(
                "inventory_sequence_count IS NULL OR inventory_sequence_count >= 0",
                name="ck_campaigns_inventory_count",
            ),
        )
        requests = Table(
            "stage_requests", metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("sequence_id", Integer, ForeignKey("sequences.id"), nullable=False),
            Column("campaign_id", Integer, ForeignKey("processing_campaigns.id")),
            Column(
                "fulfilled_by_request_id",
                Integer,
                ForeignKey("stage_requests.id"),
            ),
            Column("cohort", Text),
            Column("queue_priority", Integer, nullable=False, server_default="0"),
            Column("stage", Text, nullable=False),
            Column("status", Text, nullable=False),
            Column("trigger", Text, nullable=False),
            Column("pipeline_version", Text, ForeignKey("pipeline_versions.version"), nullable=False),
            Column("requested_by", Text),
            Column("reason", Text),
            Column("parameters_json", Text),
            Column("source_manifest_json", Text),
            Column("source_manifest_sha256", Text),
            Column("workflow_execution_id", Integer, ForeignKey("workflow_executions.id")),
            Column("reserved_by", Text),
            Column("lease_expires_at", Text),
            Column("blocked_reason", Text),
            Column("details", Text),
            Column("result_manifest_uri", Text),
            Column("result_manifest_sha256", Text),
            Column("result_summary_json", Text),
            Column("created_at", Text, nullable=False),
            Column("updated_at", Text, nullable=False),
            Column("reserved_at", Text),
            Column("submitted_at", Text),
            Column("completed_at", Text),
            CheckConstraint("cohort IS NULL OR cohort IN ('CANARY','BULK')", name="ck_requests_cohort"),
            CheckConstraint("queue_priority >= 0", name="ck_requests_queue_priority"),
            CheckConstraint(
                "stage IN ('calibration','preprocess','reconstruction','export','revalidation')",
                name="ck_requests_stage",
            ),
            CheckConstraint(f"status IN ({REQUEST_STATUSES})", name="ck_requests_status"),
            CheckConstraint("trigger IN ('AUTO','MANUAL','MIGRATION')", name="ck_requests_trigger"),
            CheckConstraint(
                "(campaign_id IS NULL AND cohort IS NULL) OR campaign_id IS NOT NULL",
                name="ck_requests_campaign_cohort",
            ),
        )
        Index("idx_stage_requests_campaign", requests.c.campaign_id, requests.c.cohort)
        Index(
            "idx_stage_requests_dispatch_priority",
            requests.c.campaign_id,
            requests.c.stage,
            requests.c.status,
            requests.c.queue_priority,
            requests.c.created_at,
            requests.c.id,
        )
        Index(
            "idx_stage_requests_campaign_sequence_recency",
            requests.c.campaign_id,
            requests.c.sequence_id,
            requests.c.created_at.desc(),
            requests.c.id.desc(),
        )
        Index(
            "idx_stage_requests_fulfilled_by",
            requests.c.fulfilled_by_request_id,
        )
        Index("idx_stage_requests_status", requests.c.status, requests.c.stage)
        Index(
            "idx_stage_requests_active", requests.c.sequence_id, requests.c.stage,
            unique=True,
            sqlite_where=requests.c.status.in_([
                "PENDING", "BLOCKED", "RESERVED", "SUBMITTED", "RUNNING", "UNKNOWN",
            ]),
            postgresql_where=requests.c.status.in_([
                "PENDING", "BLOCKED", "RESERVED", "SUBMITTED", "RUNNING", "UNKNOWN",
            ]),
        )

    def common_run_columns() -> list[Column]:
        columns = [
            Column("id", Integer, primary_key=True),
            Column("sequence_id", Integer, ForeignKey("sequences.id"), nullable=False),
            Column("workflow_execution_id", Integer, ForeignKey("workflow_executions.id")),
            Column("pipeline_version", Text, ForeignKey("pipeline_versions.version")),
            Column("status", Text, nullable=False),
            Column("details", Text),
            Column("trigger", Text, nullable=False),
            Column("requested_by", Text),
            Column("output_uri", Text),
            Column("is_current", Integer, nullable=False, server_default="0"),
            Column("legacy_source_table", Text),
            Column("legacy_source_id", Integer),
            Column("created_at", Text, nullable=False, server_default=UTC_NOW_TEXT),
            Column("started_at", Text),
            Column("completed_at", Text),
            Column("updated_at", Text, nullable=False, server_default=UTC_NOW_TEXT),
            Column("superseded_at", Text),
        ]
        if include_scheduling:
            columns.append(Column("request_id", Integer, ForeignKey("stage_requests.id"), unique=True))
        return columns

    def run_constraints(name: str) -> list:
        return [
            CheckConstraint(f"status IN ({RUN_STATUSES})", name=f"ck_{name}_status"),
            CheckConstraint(
                "trigger IN ('AUTO','MANUAL','MIGRATION')", name=f"ck_{name}_trigger"
            ),
            CheckConstraint("is_current IN (0,1)", name=f"ck_{name}_current"),
            UniqueConstraint(
                "legacy_source_table", "legacy_source_id", name=f"uq_{name}_legacy"
            ),
            CheckConstraint(
                "workflow_execution_id IS NOT NULL OR "
                "(trigger='MIGRATION' AND status='SKIPPED')",
                name=f"ck_{name}_execution",
            ),
        ]

    calibration = Table(
        "calibration_runs", metadata, *common_run_columns(),
        Column("calibration_setup", Text), Column("source_uri", Text),
        *run_constraints("calibration_runs"),
    )
    preprocess = Table(
        "preprocess_runs", metadata, *common_run_columns(),
        Column("calibration_run_id", Integer, ForeignKey("calibration_runs.id")),
        Column("source_uri", Text), Column("mesh_uri", Text), Column("metadata_uri", Text),
        *run_constraints("preprocess_runs"),
    )
    reconstruction = Table(
        "reconstruction_runs", metadata, *common_run_columns(),
        Column("preprocess_run_id", Integer, ForeignKey("preprocess_runs.id")),
        Column("labeled_bboxes_uri", Text),
        Column("labeled_bboxes_manifest_json", Text),
        Column("labeled_bboxes_sha256", Text),
        Column("bbox_source", Text), Column("hitl_item_id", Text),
        *run_constraints("reconstruction_runs"),
    )
    qc = Table(
        "qc_reviews", metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("reconstruction_run_id", Integer, ForeignKey("reconstruction_runs.id"), nullable=False),
        Column("provider", Text, nullable=False), Column("external_item_id", Text, nullable=False),
        Column("external_revision", Text), Column("payload_sha256", Text, nullable=False),
        Column("external_status", Text), Column("decision", Text, nullable=False),
        Column("details", Text), Column("failure_annotation_count", Integer),
        Column("failure_coverage", Float), Column("failure_segments_json", Text),
        Column("thresholds_json", Text), Column("raw_payload_json", Text),
        Column("reviewer", Text),
        Column("reviewed_at", Text, nullable=False, server_default=UTC_NOW_TEXT),
        Column("observed_at", Text, nullable=False, server_default=UTC_NOW_TEXT),
        CheckConstraint("provider IN ('kratos','legacy')"),
        CheckConstraint("decision IN ('PASS','FAIL')"),
        UniqueConstraint("provider", "external_item_id", "payload_sha256"),
    )
    export_auth = "'QC','MANUAL_OVERRIDE','LEGACY'"
    if include_scheduling:
        export_auth += ",'REVALIDATION'"
    export_checks = run_constraints("export_runs")
    export_checks.extend([
        CheckConstraint(
            f"authorization_type IN ({export_auth})", name="ck_export_authorization_type"
        ),
        CheckConstraint(
            "(authorization_type='QC' AND qc_review_id IS NOT NULL AND override_reason IS NULL) "
            "OR (authorization_type='MANUAL_OVERRIDE' AND requested_by IS NOT NULL "
            "AND trim(requested_by) != '' AND override_reason IS NOT NULL "
            "AND trim(override_reason) != '') "
            "OR (authorization_type='LEGACY' AND trigger='MIGRATION')"
            + (" OR authorization_type='REVALIDATION'" if include_scheduling else ""),
            name="ck_export_authorization_semantics",
        ),
    ])
    export = Table(
        "export_runs", metadata, *common_run_columns(),
        Column("reconstruction_run_id", Integer, ForeignKey("reconstruction_runs.id")),
        Column("qc_review_id", Integer, ForeignKey("qc_reviews.id")),
        Column("authorization_type", Text, nullable=False), Column("override_reason", Text),
        Column("export_task_name", Text), Column("copy_task_name", Text),
        Column("source_uri", Text), *export_checks,
    )

    if include_scheduling:
        cleanup = Table(
            "intermediate_cleanup_jobs", metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column(
                "export_run_id", Integer, ForeignKey("export_runs.id"),
                nullable=False, unique=True,
            ),
            Column("sequence_id", Integer, ForeignKey("sequences.id"), nullable=False),
            Column(
                "campaign_id", Integer, ForeignKey("processing_campaigns.id"),
                nullable=False,
            ),
            Column("status", Text, nullable=False, server_default="PENDING"),
            Column("source", Text, nullable=False),
            Column("source_audit_sha256", Text),
            Column("attempt_count", Integer, nullable=False, server_default="0"),
            Column("lease_owner", Text),
            Column("lease_expires_at", Text),
            Column("details", Text),
            Column("manifest_uri", Text),
            Column("manifest_sha256", Text),
            Column("protected_manifest_sha256", Text),
            Column("deleted_object_count", Integer),
            Column("reclaimed_bytes", BigInteger),
            Column("created_at", Text, nullable=False, server_default=UTC_NOW_TEXT),
            Column("updated_at", Text, nullable=False, server_default=UTC_NOW_TEXT),
            Column("started_at", Text),
            Column("completed_at", Text),
            CheckConstraint(
                "status IN ('PENDING','RUNNING','BLOCKED','FAILED','SUCCEEDED')",
                name="ck_intermediate_cleanup_jobs_status",
            ),
            CheckConstraint(
                "source IN ('AUTOMATIC','BACKFILL','MANUAL')",
                name="ck_intermediate_cleanup_jobs_source",
            ),
            CheckConstraint(
                "attempt_count >= 0", name="ck_intermediate_cleanup_jobs_attempts",
            ),
        )
        Index(
            "idx_intermediate_cleanup_jobs_admission",
            cleanup.c.status, cleanup.c.source, cleanup.c.created_at, cleanup.c.id,
        )
        Index(
            "idx_intermediate_cleanup_jobs_campaign",
            cleanup.c.campaign_id, cleanup.c.status,
        )

    for name, table in (
        ("calibration", calibration), ("preprocess", preprocess),
        ("reconstruction", reconstruction), ("export", export),
    ):
        Index(f"idx_{name}_current", table.c.sequence_id, unique=True,
              sqlite_where=table.c.is_current == 1,
              postgresql_where=table.c.is_current == 1)
        Index(f"idx_{name}_active_reservation", table.c.sequence_id, unique=True,
              sqlite_where=table.c.status.in_(["SUBMITTING", "RUNNING", "UNKNOWN"]),
              postgresql_where=table.c.status.in_(
                  ["SUBMITTING", "RUNNING", "UNKNOWN"]
              ))
        Index(f"idx_{name}_status", table.c.status)
    Index("idx_qc_reviews_run", qc.c.reconstruction_run_id, qc.c.reviewed_at.desc(), qc.c.id.desc())
    return metadata


BASELINE_METADATA = build_metadata(include_scheduling=False)
METADATA = build_metadata(include_scheduling=True)


def sequence_status_view_sql(
    *,
    include_requests: bool,
    include_effective: bool = False,
    prefer_current_export: bool = True,
    prefer_successful_export_effective: bool = True,
    dialect_name: str = "sqlite",
) -> str:
    """Return the current-status view for SQLite or PostgreSQL."""
    aggregate = "GREATEST" if dialect_name == "postgresql" else "max"
    request_columns = ""
    request_joins = ""
    request_activity = ""
    effective_columns = ""
    current_export_case = (
        "WHEN x.id IS NOT NULL THEN x.status" if prefer_current_export else ""
    )
    effective_export_status_case = (
        "WHEN x.id IS NOT NULL AND x.status='SUCCEEDED' THEN x.status"
        if prefer_successful_export_effective else ""
    )
    effective_export_stage_case = (
        "WHEN x.id IS NOT NULL AND x.status='SUCCEEDED' THEN 'export'"
        if prefer_successful_export_effective else ""
    )
    effective_export_details_case = (
        "WHEN x.id IS NOT NULL AND x.status='SUCCEEDED' THEN x.details"
        if prefer_successful_export_effective else ""
    )
    effective_export_updated_case = (
        "WHEN x.id IS NOT NULL AND x.status='SUCCEEDED' THEN x.updated_at"
        if prefer_successful_export_effective else ""
    )
    if include_requests:
        request_columns = """
                lr.id AS latest_request_id,
                lr.stage AS latest_request_stage,
                lr.status AS latest_request_status,
                lr.blocked_reason AS latest_request_blocked_reason,
                lr.details AS latest_request_details,
                lr.updated_at AS latest_request_updated_at,
                cr.id AS calibration_request_id,
                cr.status AS calibration_request_status,
                cr.blocked_reason AS calibration_request_blocked_reason,
                pr.id AS preprocess_request_id,
                pr.status AS preprocess_request_status,
                pr.blocked_reason AS preprocess_request_blocked_reason,
                rr.id AS reconstruction_request_id,
                rr.status AS reconstruction_request_status,
                rr.blocked_reason AS reconstruction_request_blocked_reason,
                er.id AS export_request_id,
                er.status AS export_request_status,
                er.blocked_reason AS export_request_blocked_reason,
                vr.id AS revalidation_request_id,
                vr.status AS revalidation_request_status,
                vr.blocked_reason AS revalidation_request_blocked_reason,
                vr.result_manifest_uri AS revalidation_result_manifest_uri,
                campaign.name AS campaign_name,
                campaign.campaign_type AS campaign_type,
                lr.cohort AS campaign_cohort,
        """
        request_joins = """
            LEFT JOIN stage_requests lr ON lr.id=(
                SELECT sr.id FROM stage_requests sr WHERE sr.sequence_id=s.id
                ORDER BY sr.updated_at DESC, sr.id DESC LIMIT 1)
            LEFT JOIN stage_requests cr ON cr.id=(
                SELECT sr.id FROM stage_requests sr
                WHERE sr.sequence_id=COALESCE(
                    s.calibration_sequence_id,
                    CASE WHEN s.sequence_kind='calibration' THEN s.id END)
                  AND sr.stage='calibration'
                  AND sr.status IN ('PENDING','BLOCKED','RESERVED','SUBMITTED','RUNNING','UNKNOWN')
                ORDER BY sr.created_at DESC, sr.id DESC LIMIT 1)
            LEFT JOIN stage_requests pr ON pr.id=(
                SELECT sr.id FROM stage_requests sr WHERE sr.sequence_id=s.id
                  AND sr.stage='preprocess'
                  AND sr.status IN ('PENDING','BLOCKED','RESERVED','SUBMITTED','RUNNING','UNKNOWN')
                ORDER BY sr.created_at DESC, sr.id DESC LIMIT 1)
            LEFT JOIN stage_requests rr ON rr.id=(
                SELECT sr.id FROM stage_requests sr WHERE sr.sequence_id=s.id
                  AND sr.stage='reconstruction'
                  AND sr.status IN ('PENDING','BLOCKED','RESERVED','SUBMITTED','RUNNING','UNKNOWN')
                ORDER BY sr.created_at DESC, sr.id DESC LIMIT 1)
            LEFT JOIN stage_requests er ON er.id=(
                SELECT sr.id FROM stage_requests sr WHERE sr.sequence_id=s.id
                  AND sr.stage='export'
                  AND sr.status IN ('PENDING','BLOCKED','RESERVED','SUBMITTED','RUNNING','UNKNOWN')
                ORDER BY sr.created_at DESC, sr.id DESC LIMIT 1)
            LEFT JOIN stage_requests vr ON vr.id=(
                SELECT sr.id FROM stage_requests sr WHERE sr.sequence_id=s.id
                  AND sr.stage='revalidation'
                ORDER BY sr.updated_at DESC, sr.id DESC LIMIT 1)
            LEFT JOIN processing_campaigns campaign ON campaign.id=lr.campaign_id
        """
        request_activity = """, COALESCE(cr.updated_at, ''), COALESCE(pr.updated_at, ''),
                    COALESCE(rr.updated_at, ''), COALESCE(er.updated_at, ''),
                    COALESCE(vr.updated_at, ''), COALESCE(lr.updated_at, '')"""

    if include_requests and include_effective:
        effective_columns = f"""
            CASE
                WHEN b.sequence_id IS NOT NULL THEN 'BLACKLISTED'
                {effective_export_status_case}
                WHEN lr.id IS NOT NULL THEN lr.status
                WHEN s.sequence_kind='calibration' THEN COALESCE(c.status, 'NOT_STARTED')
                WHEN x.id IS NOT NULL THEN x.status
                WHEN q.decision='FAIL' THEN 'QC_FAILED'
                WHEN q.decision='PASS' THEN 'READY'
                WHEN r.id IS NOT NULL AND r.status!='SUCCEEDED' THEN r.status
                WHEN r.id IS NOT NULL THEN 'WAITING_QC'
                WHEN p.id IS NOT NULL AND p.status!='SUCCEEDED' THEN p.status
                WHEN p.id IS NOT NULL THEN 'WAITING_LABELS'
                WHEN s.calibration_sequence_id IS NULL OR c.id IS NULL
                     OR c.status!='SUCCEEDED' THEN 'WAITING_CALIBRATION'
                ELSE 'NOT_STARTED'
            END AS effective_status,
            CASE
                WHEN b.sequence_id IS NOT NULL THEN 'blacklist'
                {effective_export_stage_case}
                WHEN lr.id IS NOT NULL THEN lr.stage
                WHEN s.sequence_kind='calibration' THEN 'calibration'
                WHEN x.id IS NOT NULL THEN 'export'
                WHEN q.id IS NOT NULL THEN 'qc'
                WHEN r.id IS NOT NULL THEN 'reconstruction'
                WHEN p.id IS NOT NULL THEN 'preprocess'
                WHEN s.calibration_sequence_id IS NULL OR c.id IS NULL
                     OR c.status!='SUCCEEDED' THEN 'calibration'
                ELSE 'preprocess'
            END AS effective_stage,
            CASE
                WHEN b.sequence_id IS NOT NULL THEN b.reason
                {effective_export_details_case}
                WHEN lr.id IS NOT NULL THEN COALESCE(lr.blocked_reason, lr.details)
                WHEN x.id IS NOT NULL THEN x.details
                WHEN q.id IS NOT NULL THEN q.details
                WHEN r.id IS NOT NULL THEN r.details
                WHEN p.id IS NOT NULL THEN p.details
                WHEN c.id IS NOT NULL THEN c.details
                ELSE NULL
            END AS effective_details,
            CASE
                WHEN b.sequence_id IS NOT NULL THEN b.created_at
                {effective_export_updated_case}
                WHEN lr.id IS NOT NULL THEN lr.updated_at
                WHEN x.id IS NOT NULL THEN x.updated_at
                WHEN q.id IS NOT NULL THEN q.observed_at
                WHEN r.id IS NOT NULL THEN r.updated_at
                WHEN p.id IS NOT NULL THEN p.updated_at
                WHEN c.id IS NOT NULL THEN c.updated_at
                ELSE s.updated_at
            END AS effective_updated_at,
        """

    calibration_request_case = "WHEN cr.id IS NOT NULL THEN cr.status" if include_requests else ""
    preprocess_request_case = "WHEN pr.id IS NOT NULL THEN pr.status" if include_requests else ""
    reconstruction_request_case = "WHEN rr.id IS NOT NULL THEN rr.status" if include_requests else ""
    export_request_case = "WHEN er.id IS NOT NULL THEN er.status" if include_requests else ""
    return f"""
        CREATE VIEW sequence_status AS
        SELECT
            s.id AS sequence_id,
            s.dataset,
            s.sequence_name,
            s.sequence_kind,
            cs.sequence_name AS calibration_sequence_name,
            b.reason AS blacklist_reason,
            b.created_by AS blacklisted_by,
            b.created_at AS blacklisted_at,
            {request_columns}
            c.id AS calibration_run_id,
            c.status AS calibration_run_status,
            c.pipeline_version AS calibration_version,
            c.details AS calibration_details,
            ce.workflow_name AS calibration_workflow,
            p.id AS preprocess_run_id,
            p.status AS preprocess_run_status,
            p.pipeline_version AS preprocess_version,
            p.details AS preprocess_details,
            pe.workflow_name AS preprocess_workflow,
            r.id AS reconstruction_run_id,
            r.status AS reconstruction_run_status,
            r.pipeline_version AS reconstruction_version,
            r.details AS reconstruction_details,
            re.workflow_name AS reconstruction_workflow,
            q.id AS qc_review_id,
            q.decision AS qc_decision,
            x.id AS export_run_id,
            x.status AS export_run_status,
            x.authorization_type AS export_authorization,
            x.details AS export_details,
            xe.workflow_name AS export_workflow,
            {effective_columns}
            CASE
                WHEN b.sequence_id IS NOT NULL THEN 'BLACKLISTED'
                {calibration_request_case}
                WHEN c.id IS NULL THEN 'NOT_STARTED'
                ELSE c.status
            END AS calibration_status,
            CASE
                WHEN b.sequence_id IS NOT NULL THEN 'BLACKLISTED'
                {preprocess_request_case}
                WHEN p.id IS NOT NULL THEN p.status
                WHEN s.sequence_kind='hoi' AND
                     (s.calibration_sequence_id IS NULL OR c.id IS NULL
                      OR c.status != 'SUCCEEDED') THEN 'WAITING_CALIBRATION'
                ELSE 'NOT_STARTED'
            END AS preprocess_status,
            CASE
                WHEN b.sequence_id IS NOT NULL THEN 'BLACKLISTED'
                {reconstruction_request_case}
                WHEN p.id IS NULL OR p.status != 'SUCCEEDED' THEN 'WAITING_PREPROCESS'
                WHEN r.id IS NULL THEN 'WAITING_LABELS'
                ELSE r.status
            END AS reconstruction_status,
            CASE
                WHEN b.sequence_id IS NOT NULL THEN 'BLACKLISTED'
                {export_request_case}
                {current_export_case}
                WHEN r.id IS NULL OR r.status != 'SUCCEEDED' THEN 'WAITING_RECONSTRUCTION'
                WHEN q.id IS NULL THEN 'WAITING_QC'
                WHEN q.decision='FAIL' THEN 'QC_FAILED'
                WHEN x.id IS NULL THEN 'READY'
                ELSE x.status
            END AS export_status,
            {aggregate}(COALESCE(c.updated_at, ''), COALESCE(p.updated_at, ''),
                COALESCE(r.updated_at, ''), COALESCE(q.observed_at, ''),
                COALESCE(x.updated_at, ''){request_activity}) AS last_activity_at
        FROM sequences s
        LEFT JOIN blacklisted_sequences b ON b.sequence_id=s.id
        LEFT JOIN sequences cs ON cs.id=s.calibration_sequence_id
        LEFT JOIN calibration_runs c ON c.sequence_id=COALESCE(
            s.calibration_sequence_id,
            CASE WHEN s.sequence_kind='calibration' THEN s.id END)
            AND c.is_current=1
        LEFT JOIN workflow_executions ce ON ce.id=c.workflow_execution_id
        LEFT JOIN preprocess_runs p ON p.sequence_id=s.id AND p.is_current=1
        LEFT JOIN workflow_executions pe ON pe.id=p.workflow_execution_id
        LEFT JOIN reconstruction_runs r ON r.sequence_id=s.id AND r.is_current=1
        LEFT JOIN workflow_executions re ON re.id=r.workflow_execution_id
        LEFT JOIN qc_reviews q ON q.id=(
            SELECT q2.id FROM qc_reviews q2
            WHERE q2.reconstruction_run_id=r.id
            ORDER BY q2.reviewed_at DESC, q2.id DESC LIMIT 1)
        LEFT JOIN export_runs x ON x.sequence_id=s.id AND x.is_current=1
        LEFT JOIN workflow_executions xe ON xe.id=x.workflow_execution_id
        {request_joins}
    """
