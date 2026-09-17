"""Persistent state for MV-HOI orchestration.

Per-sequence stage history is stored in explicit stage tables.  OSMO
executions are separate because one export workflow can contain many sequence
exports.  The public helpers keep the legacy flattened row shape while the
database stores normalized execution states and direct foreign keys.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Iterable

try:
    from .database import (
        connect as _connect,
        database_target,
        is_postgresql_connection,
        upgrade_database,
    )
except ImportError:  # Direct script execution.
    from database import (
        connect as _connect,
        database_target,
        is_postgresql_connection,
        upgrade_database,
    )


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MV_HOI_DIR = os.path.dirname(SCRIPT_DIR)
DB_PATH = database_target(test=False)
TEST_DB_PATH = database_target(test=True)

# Compatibility selectors used by the existing CLIs.  Test mode now selects a
# separate database instead of a second set of tables.
PIPELINES_TABLE = "pipelines"
PIPELINES_TEST_TABLE = "pipelines_test"
PROD_ENVIRONMENT = "prod"

CALIBRATION_STAGE = "mv_calibration"
PREPROCESS_STAGE = "mv_preprocess"
RECONSTRUCTION_STAGE = "mv_hoi_reconstruction"
EXPORT_STAGE = "mv_export"
STAGES = (CALIBRATION_STAGE, PREPROCESS_STAGE, RECONSTRUCTION_STAGE, EXPORT_STAGE)
REVALIDATION_STAGE = "revalidation"
EXECUTION_STAGE_NAMES = {
    CALIBRATION_STAGE: "calibration",
    PREPROCESS_STAGE: "preprocess",
    RECONSTRUCTION_STAGE: "reconstruction",
    EXPORT_STAGE: "export",
    REVALIDATION_STAGE: "revalidation",
}
RUN_STAGE_NAMES = {value: key for key, value in EXECUTION_STAGE_NAMES.items()}

RUN_TABLES = {
    CALIBRATION_STAGE: "calibration_runs",
    PREPROCESS_STAGE: "preprocess_runs",
    RECONSTRUCTION_STAGE: "reconstruction_runs",
    EXPORT_STAGE: "export_runs",
}
TABLE_STAGES = {table: stage for stage, table in RUN_TABLES.items()}

_RUN_STATUSES = (
    "SUBMITTING", "RUNNING", "SUCCEEDED", "FAILED", "CANCELED", "UNKNOWN", "SKIPPED",
)
_EXECUTION_STATUSES = (
    "SUBMITTING", "RUNNING", "SUCCEEDED", "FAILED", "CANCELED", "UNKNOWN",
)
_REQUEST_STATUSES = (
    "PENDING", "BLOCKED", "RESERVED", "SUBMITTED", "RUNNING", "SUCCEEDED",
    "FAILED", "CANCELED", "UNKNOWN",
)
_ACTIVE_REQUEST_STATUSES = (
    "PENDING", "BLOCKED", "RESERVED", "SUBMITTED", "RUNNING", "UNKNOWN",
)
_CAMPAIGN_STATUSES = (
    "DRAFT", "FROZEN", "RUNNING", "SUCCEEDED", "COMPLETED_WITH_FAILURES",
    "CANCELED",
)
_CAMPAIGN_TYPES = (
    "LEGACY_REVALIDATION", "BACKLOG_REPROCESSING", "REMEDIATION",
)
_REQUEST_STAGES = (
    "calibration", "preprocess", "reconstruction", "export", "revalidation",
)
_TERMINAL_REQUEST_STATUSES = frozenset(("SUCCEEDED", "FAILED", "CANCELED"))
_TERMINAL_EXECUTION_STATUSES = frozenset(("SUCCEEDED", "FAILED", "CANCELED"))
_TERMINAL_RUN_STATUSES = frozenset(("SUCCEEDED", "FAILED", "CANCELED", "SKIPPED"))
_CLEANUP_JOB_STATUSES = frozenset((
    "PENDING", "RUNNING", "BLOCKED", "FAILED", "SUCCEEDED",
))
_CLEANUP_JOB_SOURCES = frozenset(("AUTOMATIC", "BACKFILL", "MANUAL"))
_REQUEST_TRANSITIONS = {
    "PENDING": {"PENDING", "BLOCKED", "RESERVED", "SUCCEEDED", "FAILED", "CANCELED"},
    "BLOCKED": {"BLOCKED", "PENDING", "RESERVED", "FAILED", "CANCELED"},
    "RESERVED": {"RESERVED", "PENDING", "BLOCKED", "SUBMITTED", "FAILED", "CANCELED"},
    "SUBMITTED": {"SUBMITTED", "RUNNING", "UNKNOWN", "SUCCEEDED", "FAILED", "CANCELED"},
    "RUNNING": {"RUNNING", "UNKNOWN", "SUCCEEDED", "FAILED", "CANCELED"},
    "UNKNOWN": {"UNKNOWN", "RUNNING", "SUCCEEDED", "FAILED", "CANCELED"},
    **{status: {status} for status in _TERMINAL_REQUEST_STATUSES},
}
_EXECUTION_TRANSITIONS = {
    "SUBMITTING": set(_EXECUTION_STATUSES),
    "RUNNING": {"RUNNING", "UNKNOWN", "SUCCEEDED", "FAILED", "CANCELED"},
    "UNKNOWN": {"UNKNOWN", "RUNNING", "SUCCEEDED", "FAILED", "CANCELED"},
    **{status: {status} for status in _TERMINAL_EXECUTION_STATUSES},
}
_RUN_TRANSITIONS = {
    "SUBMITTING": set(_RUN_STATUSES),
    "RUNNING": {"RUNNING", "UNKNOWN", "SUCCEEDED", "FAILED", "CANCELED", "SKIPPED"},
    "UNKNOWN": {"UNKNOWN", "RUNNING", "SUCCEEDED", "FAILED", "CANCELED", "SKIPPED"},
    **{status: {status} for status in _TERMINAL_RUN_STATUSES},
}


def _guard_status_transition(
    entity: str, current: str, requested: str,
    transitions: dict[str, set[str]],
) -> None:
    """Enforce the state machine and preserve immutable terminal history."""
    if requested not in transitions.get(current, set()):
        raise ValueError(
            f"Invalid {entity} status transition: {current} -> {requested}; "
            "create a new request/attempt instead"
        )


def get_connection(db_path: str = DB_PATH):
    return _connect(db_path)


def init_db(db_path: str = DB_PATH) -> None:
    """Validate/bootstrap schema v2 and upgrade it to the current revision."""
    upgrade_database(db_path)


# -- Semver/version cache -------------------------------------------------

_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def parse_semver(version: str) -> tuple[int, int, int]:
    match = _SEMVER_RE.match(version)
    if not match:
        raise ValueError(f"Invalid semver: {version!r} (expected X.Y.Z)")
    return tuple(int(value) for value in match.groups())


def validate_semver_gt(new: str, latest: str | None) -> None:
    if latest is not None and parse_semver(new) <= parse_semver(latest):
        raise ValueError(f"Version {new} must be greater than current latest {latest}")


def get_latest_version(db_path: str = DB_PATH) -> str | None:
    conn = get_connection(db_path)
    try:
        versions = [row[0] for row in conn.execute("SELECT version FROM pipeline_versions")]
    finally:
        conn.close()
    return max(versions, key=parse_semver) if versions else None


def insert_version(version: str, message: str = "", db_path: str = DB_PATH) -> str:
    validate_semver_gt(version, get_latest_version(db_path))
    conn = get_connection(db_path)
    try:
        conn.execute(
            "INSERT INTO pipeline_versions(version, message) VALUES (?, ?)",
            (version, message),
        )
        conn.commit()
    finally:
        conn.close()
    return version


def ensure_version_cached(version: str, message: str = "", db_path: str = DB_PATH) -> str:
    parse_semver(version)
    conn = get_connection(db_path)
    try:
        conn.execute(
            "INSERT OR IGNORE INTO pipeline_versions(version, message) VALUES (?, ?)",
            (version, message),
        )
        conn.commit()
    finally:
        conn.close()
    return version


# -- Sequences and blacklist ----------------------------------------------

def _ensure_sequence(
    conn: sqlite3.Connection,
    dataset: str,
    sequence_name: str,
    *,
    sequence_kind: str = "hoi",
    source_uri: str | None = None,
    hoi_metadata_uri: str | None = None,
    object_id: str | None = None,
) -> int:
    conn.execute(
        """INSERT INTO sequences
           (dataset, sequence_name, sequence_kind, source_uri, hoi_metadata_uri, object_id)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(dataset, sequence_name) DO UPDATE SET
             source_uri=COALESCE(excluded.source_uri, sequences.source_uri),
             hoi_metadata_uri=COALESCE(excluded.hoi_metadata_uri, sequences.hoi_metadata_uri),
             object_id=COALESCE(excluded.object_id, sequences.object_id),
             updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')""",
        (dataset, sequence_name, sequence_kind, source_uri, hoi_metadata_uri, object_id),
    )
    return conn.execute(
        "SELECT id FROM sequences WHERE dataset=? AND sequence_name=?",
        (dataset, sequence_name),
    ).fetchone()[0]


def upsert_sequence(
    dataset: str,
    sequence_name: str,
    *,
    sequence_kind: str = "hoi",
    source_uri: str | None = None,
    hoi_metadata_uri: str | None = None,
    calibration_sequence_name: str | None = None,
    object_id: str | None = None,
    db_path: str = DB_PATH,
) -> int:
    conn = get_connection(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        sequence_id = _ensure_sequence(
            conn, dataset, sequence_name, sequence_kind=sequence_kind,
            source_uri=source_uri, hoi_metadata_uri=hoi_metadata_uri, object_id=object_id,
        )
        if calibration_sequence_name:
            calibration_id = _ensure_sequence(
                conn, dataset, calibration_sequence_name, sequence_kind="calibration"
            )
            conn.execute(
                "UPDATE sequences SET calibration_sequence_id=?, "
                "updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') "
                "WHERE id=?",
                (calibration_id, sequence_id),
            )
        conn.commit()
        return sequence_id
    finally:
        conn.close()


def upsert_blacklisted_sequence(
    dataset: str,
    sequence_name: str,
    reason: str | None = None,
    *,
    created_by: str | None = None,
    db_path: str = DB_PATH,
) -> None:
    conn = get_connection(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        sequence_id = _ensure_sequence(conn, dataset, sequence_name)
        conn.execute(
            """INSERT INTO blacklisted_sequences(sequence_id, reason, created_by)
               VALUES (?, ?, ?) ON CONFLICT(sequence_id) DO UPDATE SET
               reason=excluded.reason,
               created_by=COALESCE(
                   excluded.created_by, blacklisted_sequences.created_by
               )""",
            (sequence_id, reason, created_by),
        )
        conn.commit()
    finally:
        conn.close()


def remove_blacklisted_sequence(
    dataset: str, sequence_name: str, db_path: str = DB_PATH
) -> bool:
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """DELETE FROM blacklisted_sequences WHERE sequence_id=(
               SELECT id FROM sequences WHERE dataset=? AND sequence_name=?)""",
            (dataset, sequence_name),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def get_blacklisted_sequence(
    dataset: str, sequence_name: str, db_path: str = DB_PATH
) -> dict | None:
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            """SELECT s.dataset, s.sequence_name, b.reason, b.created_by,
                      b.created_at, b.created_at AS blacklisted_at
               FROM blacklisted_sequences b JOIN sequences s ON s.id=b.sequence_id
               WHERE s.dataset=? AND s.sequence_name=?""",
            (dataset, sequence_name),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def is_sequence_blacklisted(
    dataset: str, sequence_name: str, db_path: str = DB_PATH
) -> bool:
    return get_blacklisted_sequence(dataset, sequence_name, db_path) is not None


def get_blacklisted_sequences(dataset: str, db_path: str = DB_PATH) -> list[dict]:
    conn = get_connection(db_path)
    try:
        return [dict(row) for row in conn.execute(
            """SELECT s.dataset, s.sequence_name, b.reason, b.created_by,
                      b.created_at, b.created_at AS blacklisted_at
               FROM blacklisted_sequences b JOIN sequences s ON s.id=b.sequence_id
               WHERE s.dataset=? ORDER BY s.sequence_name""",
            (dataset,),
        )]
    finally:
        conn.close()


def normalize_failure_details(details: str | None) -> str:
    """Normalize failure text for the repeated-failure circuit breaker."""
    return re.sub(r"\s+", " ", (details or "").strip())


# -- Durable campaigns and scheduling requests ----------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"), default=str)


def _json_sha256(payload: str | None) -> str | None:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest() if payload is not None else None


def create_campaign(
    *,
    name: str,
    campaign_type: str,
    dataset: str,
    pipeline_version: str,
    output_uri: str,
    created_by: str,
    canary: bool | None = None,
    db_path: str = DB_PATH,
) -> dict:
    if campaign_type not in _CAMPAIGN_TYPES:
        raise ValueError(f"Unsupported campaign type: {campaign_type}")
    if not name.strip() or not dataset.strip() or not created_by.strip():
        raise ValueError("Campaign name, dataset, and creator are required")
    ensure_version_cached(pipeline_version, db_path=db_path)
    now = _utc_now()
    if canary is None:
        canary = campaign_type == "LEGACY_REVALIDATION"
    phase = "CANARY" if canary else "BULK"
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """INSERT INTO processing_campaigns
               (name, campaign_type, dataset, status, phase, pipeline_version,
                output_uri, created_by, created_at, updated_at)
               VALUES (?, ?, ?, 'DRAFT', ?, ?, ?, ?, ?, ?)""",
            (name, campaign_type, dataset, phase, pipeline_version, output_uri,
             created_by, now, now),
        )
        conn.commit()
        campaign_id = cursor.lastrowid
    finally:
        conn.close()
    return get_campaign(campaign_id, db_path=db_path)


def get_campaign(
    campaign: int | str, *, db_path: str = DB_PATH,
) -> dict | None:
    conn = get_connection(db_path)
    try:
        column = "id" if isinstance(campaign, int) else "name"
        row = conn.execute(
            f"SELECT * FROM processing_campaigns WHERE {column}=?", (campaign,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_campaigns(
    *, dataset: str | None = None, status: str | None = None,
    db_path: str = DB_PATH,
) -> list[dict]:
    clauses: list[str] = []
    params: list = []
    if dataset:
        clauses.append("dataset=?")
        params.append(dataset)
    if status:
        if status not in _CAMPAIGN_STATUSES:
            raise ValueError(f"Unsupported campaign status: {status}")
        clauses.append("status=?")
        params.append(status)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    conn = get_connection(db_path)
    try:
        return [dict(row) for row in conn.execute(
            f"SELECT * FROM processing_campaigns{where} ORDER BY created_at, id", params
        )]
    finally:
        conn.close()


def update_campaign_pipeline_version(
    campaign: int | str,
    pipeline_version: str,
    *,
    message: str = "",
    db_path: str = DB_PATH,
) -> dict:
    """Advance an active campaign without rewriting submitted history.

    Only undispatched PENDING/BLOCKED requests move to the new release.
    Workflow-owning and terminal requests remain immutable and are reported so
    an in-flight attempt can finish without being canceled or reprocessed.
    """
    ensure_version_cached(pipeline_version, message, db_path=db_path)
    now = _utc_now()
    conn = get_connection(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        column = "id" if isinstance(campaign, int) else "name"
        row = conn.execute(
            f"SELECT * FROM processing_campaigns WHERE {column}=?"
            + (" FOR UPDATE" if is_postgresql_connection(conn) else ""),
            (campaign,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown campaign: {campaign}")
        if row["status"] not in ("FROZEN", "RUNNING"):
            raise ValueError("Campaign must be FROZEN or RUNNING")

        active_rows = conn.execute(
            """SELECT status, COUNT(*) AS count FROM stage_requests
               WHERE campaign_id=? AND status IN
                   ('RESERVED', 'SUBMITTED', 'RUNNING', 'UNKNOWN')
               GROUP BY status ORDER BY status""",
            (row["id"],),
        ).fetchall()
        request_rows = conn.execute(
            """SELECT status, COUNT(*) AS count FROM stage_requests
               WHERE campaign_id=? GROUP BY status ORDER BY status""",
            (row["id"],),
        ).fetchall()
        previous_version = row["pipeline_version"]
        updated_requests = 0
        if previous_version != pipeline_version:
            cursor = conn.execute(
                """UPDATE stage_requests SET pipeline_version=?, updated_at=?
                   WHERE campaign_id=? AND status IN ('PENDING', 'BLOCKED')""",
                (pipeline_version, now, row["id"]),
            )
            updated_requests = cursor.rowcount
            conn.execute(
                """UPDATE processing_campaigns
                   SET pipeline_version=?, updated_at=? WHERE id=?""",
                (pipeline_version, now, row["id"]),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    result = get_campaign(int(row["id"]), db_path=db_path)
    result["previous_pipeline_version"] = previous_version
    result["updated_request_count"] = updated_requests
    result["request_status_counts"] = {
        request_row["status"]: request_row["count"]
        for request_row in request_rows
    }
    result["unchanged_active_request_counts"] = {
        active_row["status"]: active_row["count"] for active_row in active_rows
    }
    return result


def get_campaign_lifecycle_snapshot(
    dataset: str,
    *,
    campaign: int | str | None = None,
    db_path: str = DB_PATH,
) -> dict:
    """Read the authoritative request lineage for campaign-member sequences.

    The active scope is the union of FROZEN and RUNNING campaigns. An explicit
    campaign remains queryable after it closes. This deliberately reads the
    normalized campaign tables instead of the legacy ``sequence_status`` view.
    """
    conn = get_connection(db_path)
    try:
        observed = conn.execute(
            "SELECT strftime('%Y-%m-%dT%H:%M:%fZ','now') AS observed_at"
        ).fetchone()["observed_at"]
        params: list = [dataset]
        if campaign is None:
            campaign_where = "dataset=? AND status IN ('FROZEN','RUNNING')"
        else:
            column = "id" if isinstance(campaign, int) else "name"
            campaign_where = f"dataset=? AND {column}=?"
            params.append(campaign)
        campaigns = [
            dict(row) for row in conn.execute(
                f"""SELECT * FROM processing_campaigns
                    WHERE {campaign_where}
                    ORDER BY created_at, id""",
                params,
            )
        ]
        if campaign is not None and not campaigns:
            raise ValueError(
                f"Campaign {campaign!r} does not exist in dataset {dataset!r}"
            )
        if not campaigns:
            return {
                "observed_at": observed,
                "campaigns": [],
                "campaign_memberships": [],
                "sequences": [],
            }

        campaign_ids = [row["id"] for row in campaigns]
        placeholders = ",".join("?" for _ in campaign_ids)
        memberships = [
            dict(row) for row in conn.execute(
                f"""SELECT campaign_id, COUNT(DISTINCT sequence_id) AS member_count
                    FROM stage_requests
                    WHERE campaign_id IN ({placeholders})
                    GROUP BY campaign_id""",
                campaign_ids,
            )
        ]
        rows = [
            dict(row) for row in conn.execute(
                f"""
                WITH ranked_requests AS (
                    SELECT
                        sr.id,
                        sr.sequence_id,
                        sr.campaign_id,
                        sr.fulfilled_by_request_id,
                        sr.cohort,
                        sr.stage,
                        sr.status,
                        sr.pipeline_version,
                        sr.blocked_reason,
                        sr.details,
                        sr.result_summary_json,
                        sr.created_at,
                        sr.updated_at,
                        s.dataset,
                        s.sequence_name,
                        s.sequence_kind,
                        cs.sequence_name AS calibration_sequence_name,
                        b.reason AS blacklist_reason,
                        b.created_by AS blacklisted_by,
                        b.created_at AS blacklisted_at,
                        c.name AS campaign_name,
                        c.campaign_type,
                        c.status AS campaign_status,
                        c.phase AS campaign_phase,
                        ROW_NUMBER() OVER (
                            PARTITION BY sr.sequence_id
                            ORDER BY sr.created_at DESC, sr.id DESC
                        ) AS lineage_rank
                    FROM stage_requests sr
                    JOIN sequences s ON s.id=sr.sequence_id
                    LEFT JOIN sequences cs ON cs.id=s.calibration_sequence_id
                    LEFT JOIN blacklisted_sequences b ON b.sequence_id=s.id
                    JOIN processing_campaigns c ON c.id=sr.campaign_id
                    WHERE sr.campaign_id IN ({placeholders})
                ),
                authoritative AS (
                    SELECT * FROM ranked_requests WHERE lineage_rank=1
                ),
                latest_qc AS (
                    SELECT
                        q.*,
                        ROW_NUMBER() OVER (
                            PARTITION BY q.reconstruction_run_id
                            ORDER BY q.reviewed_at DESC, q.id DESC
                        ) AS review_rank
                    FROM qc_reviews q
                )
                SELECT
                    a.*,
                    a.fulfilled_by_request_id AS fulfillment_request_id,
                    COALESCE(a.fulfilled_by_request_id, a.id)
                        AS effective_request_id,
                    cal.id AS calibration_run_id,
                    cal.status AS calibration_run_status,
                    cal.updated_at AS calibration_run_updated_at,
                    cal_exec.workflow_name AS calibration_workflow,
                    pre.id AS preprocess_run_id,
                    pre.status AS preprocess_run_status,
                    pre.details AS preprocess_run_details,
                    pre.updated_at AS preprocess_run_updated_at,
                    pre_exec.workflow_name AS preprocess_workflow,
                    rec.id AS reconstruction_run_id,
                    rec.status AS reconstruction_run_status,
                    rec.details AS reconstruction_run_details,
                    rec.updated_at AS reconstruction_run_updated_at,
                    rec_exec.workflow_name AS reconstruction_workflow,
                    exp.id AS export_run_id,
                    exp.status AS export_run_status,
                    exp.authorization_type AS export_authorization_type,
                    exp.details AS export_run_details,
                    exp.updated_at AS export_run_updated_at,
                    exp_exec.workflow_name AS export_workflow,
                    qc.id AS qc_review_id,
                    qc.decision AS qc_decision,
                    qc.details AS qc_details,
                    qc.observed_at AS qc_observed_at
                FROM authoritative a
                LEFT JOIN calibration_runs direct_cal
                    ON direct_cal.request_id=COALESCE(
                        a.fulfilled_by_request_id, a.id
                    )
                LEFT JOIN preprocess_runs direct_pre
                    ON direct_pre.request_id=COALESCE(
                        a.fulfilled_by_request_id, a.id
                    )
                LEFT JOIN preprocess_runs current_pre
                    ON current_pre.sequence_id=a.sequence_id
                   AND current_pre.is_current=1
                   AND a.stage='reconstruction'
                LEFT JOIN reconstruction_runs direct_rec
                    ON direct_rec.request_id=COALESCE(
                        a.fulfilled_by_request_id, a.id
                    )
                LEFT JOIN export_runs exp
                    ON exp.request_id=COALESCE(
                        a.fulfilled_by_request_id, a.id
                    )
                LEFT JOIN reconstruction_runs rec
                    ON rec.id=COALESCE(
                        direct_rec.id, exp.reconstruction_run_id
                    )
                LEFT JOIN preprocess_runs pre
                    ON pre.id=COALESCE(
                        direct_pre.id, rec.preprocess_run_id, current_pre.id
                    )
                LEFT JOIN calibration_runs cal
                    ON cal.id=COALESCE(
                        direct_cal.id, pre.calibration_run_id
                    )
                LEFT JOIN latest_qc qc
                    ON qc.reconstruction_run_id=rec.id
                   AND qc.review_rank=1
                LEFT JOIN workflow_executions cal_exec
                    ON cal_exec.id=cal.workflow_execution_id
                LEFT JOIN workflow_executions pre_exec
                    ON pre_exec.id=pre.workflow_execution_id
                LEFT JOIN workflow_executions rec_exec
                    ON rec_exec.id=rec.workflow_execution_id
                LEFT JOIN workflow_executions exp_exec
                    ON exp_exec.id=exp.workflow_execution_id
                ORDER BY a.sequence_name
                """,
                campaign_ids,
            )
        ]
        return {
            "observed_at": observed,
            "campaigns": campaigns,
            "campaign_memberships": memberships,
            "sequences": rows,
        }
    finally:
        conn.close()


def freeze_campaign(
    campaign: int | str,
    *,
    inventory_uri: str,
    inventory_sha256: str,
    configuration_uri: str,
    configuration_sha256: str,
    inventory_sequence_count: int | None = None,
    db_path: str = DB_PATH,
) -> dict:
    values = (inventory_uri, inventory_sha256, configuration_uri, configuration_sha256)
    if any(not value or not value.strip() for value in values):
        raise ValueError("Frozen campaign inventory/configuration URIs and hashes are required")
    if inventory_sequence_count is not None and inventory_sequence_count < 0:
        raise ValueError("Inventory sequence count cannot be negative")
    conn = get_connection(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        column = "id" if isinstance(campaign, int) else "name"
        row = conn.execute(
            f"SELECT * FROM processing_campaigns WHERE {column}=?", (campaign,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown campaign: {campaign}")
        existing = (
            row["inventory_uri"], row["inventory_sha256"],
            row["configuration_uri"], row["configuration_sha256"],
        )
        if row["status"] != "DRAFT":
            if existing != values:
                raise ValueError("Frozen campaign identity is immutable")
            if (
                inventory_sequence_count is not None
                and row["inventory_sequence_count"] != inventory_sequence_count
            ):
                raise ValueError("Frozen campaign inventory count is immutable")
            conn.rollback()
            return dict(row)
        now = _utc_now()
        conn.execute(
            """UPDATE processing_campaigns SET status='FROZEN',
               inventory_uri=?, inventory_sha256=?, configuration_uri=?,
               configuration_sha256=?, inventory_sequence_count=?,
               frozen_at=?, updated_at=? WHERE id=?""",
            (*values, inventory_sequence_count, now, now, row["id"]),
        )
        conn.commit()
        campaign_id = row["id"]
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return get_campaign(campaign_id, db_path=db_path)


def set_campaign_inventory_count(
    campaign: int | str, count: int, *, db_path: str = DB_PATH,
) -> dict:
    """Set the frozen count once, or verify an already recorded count."""
    if count < 0:
        raise ValueError("Inventory sequence count cannot be negative")
    conn = get_connection(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        column = "id" if isinstance(campaign, int) else "name"
        row = conn.execute(
            f"SELECT * FROM processing_campaigns WHERE {column}=?", (campaign,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown campaign: {campaign}")
        existing = row["inventory_sequence_count"]
        if existing is not None and int(existing) != count:
            raise ValueError(
                f"Frozen inventory count mismatch: campaign={existing}, inventory={count}"
            )
        if existing is None:
            conn.execute(
                "UPDATE processing_campaigns SET inventory_sequence_count=?, updated_at=? "
                "WHERE id=?", (count, _utc_now(), row["id"]),
            )
        conn.commit()
        campaign_id = row["id"]
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return get_campaign(campaign_id, db_path=db_path)


def _validate_campaign_canary_for_approval(conn, row) -> None:
    if row["status"] not in ("FROZEN", "RUNNING") or row["phase"] != "CANARY":
        raise ValueError("Campaign is not awaiting canary approval")
    canary = conn.execute(
        """SELECT sr.status, sr.stage, sr.result_manifest_uri,
                  sr.result_manifest_sha256,
                  EXISTS (
                    SELECT 1 FROM export_runs er
                    WHERE er.request_id=sr.id
                      AND er.status='RUNNING'
                      AND er.details='candidate_export_verified'
                  ) AS staged_export_verified
           FROM stage_requests sr
           WHERE sr.campaign_id=? AND sr.cohort='CANARY'""",
        (row["id"],),
    ).fetchall()
    if not canary:
        raise ValueError("Campaign has no canary requests")
    successful = []
    for item in canary:
        staged = bool(item["staged_export_verified"])
        if item["status"] not in _TERMINAL_REQUEST_STATUSES and not staged:
            raise ValueError(
                "Every canary request must be terminal before approval or have a "
                "verified staged export"
            )
        if item["status"] == "SUCCEEDED" or staged:
            successful.append(item)
    if not successful:
        raise ValueError("Canary approval requires at least one successful result")
    if row["campaign_type"] == "LEGACY_REVALIDATION" and any(
        item["status"] == "SUCCEEDED"
        and (not item["result_manifest_uri"] or not item["result_manifest_sha256"])
        for item in successful
    ):
        raise ValueError("Every successful canary request requires a verified result")


def _approve_campaign_canary_tx(
    conn,
    row,
    *,
    report_uri: str,
    report_sha256: str,
    approved_by: str,
    now: str,
) -> None:
    _validate_campaign_canary_for_approval(conn, row)
    conn.execute(
        """UPDATE processing_campaigns SET phase='BULK', status='RUNNING',
           canary_report_uri=?, canary_report_sha256=?, canary_approved_by=?,
           canary_approved_at=?, started_at=COALESCE(started_at, ?), updated_at=?
           WHERE id=?""",
        (report_uri, report_sha256, approved_by, now, now, now, row["id"]),
    )
    conn.execute(
        """UPDATE stage_requests SET status='PENDING', blocked_reason=NULL,
           details='canary_approved', updated_at=?
           WHERE campaign_id=? AND cohort='BULK' AND status='BLOCKED'
             AND blocked_reason='WAITING_CANARY_APPROVAL'""",
        (now, row["id"]),
    )


def approve_campaign_canary(
    campaign: int | str,
    *,
    report_uri: str,
    report_sha256: str,
    approved_by: str,
    db_path: str = DB_PATH,
) -> dict:
    if not report_uri or not report_sha256 or not approved_by.strip():
        raise ValueError("Canary report identity and approving operator are required")
    conn = get_connection(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        column = "id" if isinstance(campaign, int) else "name"
        row = conn.execute(
            f"SELECT * FROM processing_campaigns WHERE {column}=?", (campaign,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown campaign: {campaign}")
        now = _utc_now()
        _approve_campaign_canary_tx(
            conn, row, report_uri=report_uri, report_sha256=report_sha256,
            approved_by=approved_by, now=now,
        )
        conn.commit()
        campaign_id = row["id"]
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return get_campaign(campaign_id, db_path=db_path)


def approve_paired_campaign_canaries(
    approvals: Iterable[dict], *, db_path: str = DB_PATH,
) -> list[dict]:
    """Promote multiple canaries atomically, or promote none of them."""
    approvals = list(approvals)
    if len(approvals) < 2:
        raise ValueError("Paired canary approval requires at least two campaigns")
    names = [item.get("campaign") for item in approvals]
    if len(names) != len(set(names)):
        raise ValueError("Paired canary campaigns must be distinct")
    conn = get_connection(db_path)
    campaign_ids: list[int] = []
    try:
        conn.execute("BEGIN IMMEDIATE")
        rows = []
        for item in approvals:
            values = (
                item.get("report_uri"), item.get("report_sha256"),
                item.get("approved_by"),
            )
            if any(not value or not str(value).strip() for value in values):
                raise ValueError("Every paired approval requires report identity and approver")
            campaign = item["campaign"]
            column = "id" if isinstance(campaign, int) else "name"
            row = conn.execute(
                f"SELECT * FROM processing_campaigns WHERE {column}=?",
                (campaign,),
            ).fetchone()
            if row is None:
                raise ValueError(f"Unknown campaign: {campaign}")
            _validate_campaign_canary_for_approval(conn, row)
            rows.append((row, item))
        datasets = {row["dataset"] for row, _ in rows}
        if len(datasets) != 1:
            raise ValueError("Paired canaries must target the same dataset")
        now = _utc_now()
        for row, item in rows:
            _approve_campaign_canary_tx(
                conn, row, report_uri=item["report_uri"],
                report_sha256=item["report_sha256"],
                approved_by=item["approved_by"], now=now,
            )
            campaign_ids.append(int(row["id"]))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return [get_campaign(value, db_path=db_path) for value in campaign_ids]


def create_stage_request(
    *,
    sequence_name: str,
    dataset: str,
    stage: str,
    pipeline_version: str,
    trigger: str = "AUTO",
    requested_by: str | None = None,
    reason: str | None = None,
    campaign: int | str | None = None,
    cohort: str | None = None,
    queue_priority: int = 0,
    parameters: object | None = None,
    source_manifest: object | None = None,
    source_manifest_sha256: str | None = None,
    status: str = "PENDING",
    blocked_reason: str | None = None,
    db_path: str = DB_PATH,
) -> dict:
    if stage not in _REQUEST_STAGES:
        raise ValueError(f"Unsupported request stage: {stage}")
    normalized_trigger = _normalize_trigger(trigger)
    if status not in ("PENDING", "BLOCKED"):
        raise ValueError("New requests must be PENDING or BLOCKED")
    if status == "BLOCKED" and not blocked_reason:
        raise ValueError("Blocked requests require a reason")
    if cohort not in (None, "CANARY", "BULK"):
        raise ValueError(f"Unsupported request cohort: {cohort}")
    if isinstance(queue_priority, bool) or not isinstance(queue_priority, int) or queue_priority < 0:
        raise ValueError("Request queue priority must be a nonnegative integer")
    parameters_json = _canonical_json(parameters)
    manifest_json = _canonical_json(source_manifest)
    computed_manifest_hash = _json_sha256(manifest_json)
    if source_manifest_sha256 and source_manifest_sha256 != computed_manifest_hash:
        raise ValueError("Source manifest SHA-256 does not match its canonical JSON")
    source_manifest_sha256 = source_manifest_sha256 or computed_manifest_hash
    ensure_version_cached(pipeline_version, db_path=db_path)
    now = _utc_now()
    conn = get_connection(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        sequence_id = _ensure_sequence(conn, dataset, sequence_name)
        campaign_id = None
        if campaign is not None:
            column = "id" if isinstance(campaign, int) else "name"
            campaign_row = conn.execute(
                f"SELECT * FROM processing_campaigns WHERE {column}=?", (campaign,)
            ).fetchone()
            if campaign_row is None:
                raise ValueError(f"Unknown campaign: {campaign}")
            if campaign_row["status"] not in ("FROZEN", "RUNNING"):
                raise ValueError("Campaign must be frozen before requests are created")
            if campaign_row["dataset"] != dataset:
                raise ValueError("Request dataset does not match campaign")
            if campaign_row["pipeline_version"] != pipeline_version:
                raise ValueError("Request pipeline version does not match campaign")
            if cohort is None:
                raise ValueError("Campaign requests require CANARY or BULK cohort")
            if cohort == "BULK" and campaign_row["phase"] == "CANARY":
                status = "BLOCKED"
                blocked_reason = "WAITING_CANARY_APPROVAL"
            campaign_id = campaign_row["id"]
        elif cohort is not None:
            raise ValueError("Mainline requests cannot have a campaign cohort")
        cursor = conn.execute(
            """INSERT INTO stage_requests
               (sequence_id, campaign_id, cohort, queue_priority, stage, status, trigger,
                pipeline_version, requested_by, reason, parameters_json,
                source_manifest_json, source_manifest_sha256, blocked_reason,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (sequence_id, campaign_id, cohort, queue_priority, stage, status, normalized_trigger,
             pipeline_version, requested_by, reason, parameters_json, manifest_json,
             source_manifest_sha256, blocked_reason, now, now),
        )
        conn.commit()
        request_id = cursor.lastrowid
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return get_stage_request(request_id, db_path=db_path)


def create_campaign_stage_requests_batch(
    campaign: int | str, requests: Iterable[dict], *, db_path: str = DB_PATH,
) -> int:
    """Insert a validated frozen campaign request set in one transaction."""
    requests = [dict(item) for item in requests]
    if not requests:
        return 0
    now = _utc_now()
    conn = get_connection(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        column = "id" if isinstance(campaign, int) else "name"
        campaign_row = conn.execute(
            f"SELECT * FROM processing_campaigns WHERE {column}=?"
            + (" FOR UPDATE" if is_postgresql_connection(conn) else ""),
            (campaign,),
        ).fetchone()
        if campaign_row is None:
            raise ValueError(f"Unknown campaign: {campaign}")
        if campaign_row["status"] not in ("FROZEN", "RUNNING"):
            raise ValueError("Campaign must be frozen before requests are created")
        for item in requests:
            stage = item["stage"]
            cohort = item["cohort"]
            queue_priority = item.get("queue_priority", 0)
            status = item.get("status", "PENDING")
            blocker = item.get("blocked_reason")
            if stage not in _REQUEST_STAGES:
                raise ValueError(f"Unsupported request stage: {stage}")
            if cohort not in ("CANARY", "BULK"):
                raise ValueError(f"Unsupported request cohort: {cohort}")
            if (
                isinstance(queue_priority, bool)
                or not isinstance(queue_priority, int)
                or queue_priority < 0
            ):
                raise ValueError("Request queue priority must be a nonnegative integer")
            if item["dataset"] != campaign_row["dataset"]:
                raise ValueError("Request dataset does not match campaign")
            if item["pipeline_version"] != campaign_row["pipeline_version"]:
                raise ValueError("Request pipeline version does not match campaign")
            if cohort == "BULK" and campaign_row["phase"] == "CANARY":
                status, blocker = "BLOCKED", "WAITING_CANARY_APPROVAL"
            if status not in ("PENDING", "BLOCKED"):
                raise ValueError("New requests must be PENDING or BLOCKED")
            if status == "BLOCKED" and not blocker:
                raise ValueError("Blocked requests require a reason")
            sequence_id = _ensure_sequence(
                conn, item["dataset"], item["sequence_name"],
            )
            manifest_json = _canonical_json(item.get("source_manifest", {}))
            conn.execute(
                """INSERT INTO stage_requests
                   (sequence_id, campaign_id, cohort, queue_priority, stage, status, trigger,
                    pipeline_version, requested_by, reason, parameters_json,
                    source_manifest_json, source_manifest_sha256, blocked_reason,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    sequence_id, campaign_row["id"], cohort, queue_priority, stage, status,
                    _normalize_trigger(item.get("trigger", "MIGRATION")),
                    item["pipeline_version"], item["requested_by"],
                    item.get("reason"), _canonical_json(item.get("parameters", {})),
                    manifest_json, _json_sha256(manifest_json), blocker, now, now,
                ),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return len(requests)


def get_stage_request(request_id: int, *, db_path: str = DB_PATH) -> dict | None:
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            """SELECT sr.*, s.dataset, s.sequence_name, c.name AS campaign_name,
                      c.campaign_type, c.phase AS campaign_phase
               FROM stage_requests sr JOIN sequences s ON s.id=sr.sequence_id
               LEFT JOIN processing_campaigns c ON c.id=sr.campaign_id
               WHERE sr.id=?""",
            (request_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_stage_requests(
    *,
    dataset: str | None = None,
    campaign: int | str | None = None,
    stage: str | None = None,
    status: str | Iterable[str] | None = None,
    db_path: str = DB_PATH,
) -> list[dict]:
    clauses: list[str] = []
    params: list = []
    if dataset:
        clauses.append("s.dataset=?")
        params.append(dataset)
    if campaign is not None:
        column = "c.id" if isinstance(campaign, int) else "c.name"
        clauses.append(f"{column}=?")
        params.append(campaign)
    if stage:
        clauses.append("sr.stage=?")
        params.append(stage)
    if status:
        statuses = [status] if isinstance(status, str) else list(status)
        if any(value not in _REQUEST_STATUSES for value in statuses):
            raise ValueError("Unsupported request status")
        clauses.append("sr.status IN (" + ",".join("?" for _ in statuses) + ")")
        params.extend(statuses)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    conn = get_connection(db_path)
    try:
        return [dict(row) for row in conn.execute(
            """SELECT sr.*, s.dataset, s.sequence_name, c.name AS campaign_name,
                      c.campaign_type, c.phase AS campaign_phase
               FROM stage_requests sr JOIN sequences s ON s.id=sr.sequence_id
               LEFT JOIN processing_campaigns c ON c.id=sr.campaign_id"""
            + where + " ORDER BY sr.created_at, sr.id",
            params,
        )]
    finally:
        conn.close()


def list_campaign_advancement_requests(
    campaign: int | str, *, db_path: str = DB_PATH,
) -> list[dict]:
    """Return campaign request fields needed to advance preprocessing lineage.

    Frozen preprocess manifests can be large and are not part of ordinary label
    refresh.  Omit them from this bulk scheduler read; the exceptional marker
    restoration path loads its one request explicitly when needed.
    """
    column = "c.id" if isinstance(campaign, int) else "c.name"
    conn = get_connection(db_path)
    try:
        return [dict(row) for row in conn.execute(
            f"""SELECT sr.id, sr.sequence_id, sr.campaign_id, sr.cohort,
                       sr.stage, sr.status, sr.pipeline_version, sr.requested_by,
                       sr.parameters_json,
                       CASE WHEN sr.stage='reconstruction'
                            THEN sr.source_manifest_json END AS source_manifest_json,
                       sr.workflow_execution_id, sr.blocked_reason, sr.details,
                       sr.created_at, sr.updated_at, sr.queue_priority,
                       s.dataset, s.sequence_name
                FROM stage_requests sr
                JOIN sequences s ON s.id=sr.sequence_id
                JOIN processing_campaigns c ON c.id=sr.campaign_id
                WHERE {column}=?
                ORDER BY sr.created_at, sr.id""",
            (campaign,),
        )]
    finally:
        conn.close()


def count_active_stage_executions(
    stage: str,
    *,
    campaign: int | str | None = None,
    db_path: str = DB_PATH,
) -> int:
    """Count compute-active requests, not terminal work awaiting publication.

    An active request with missing execution lineage remains capacity-consuming.
    This makes admission conservative for corrupt or ambiguous state while a
    remotely completed execution can release its compute slot before export
    publication finishes.
    """
    if stage not in _REQUEST_STAGES:
        raise ValueError(f"Unsupported request stage: {stage}")
    clauses = [
        "sr.stage=?",
        "sr.status IN ('RESERVED','SUBMITTED','RUNNING','UNKNOWN')",
        "(sr.workflow_execution_id IS NULL "
        "OR we.status IN ('SUBMITTING','RUNNING','UNKNOWN'))",
    ]
    params: list = [stage]
    join = ""
    if campaign is not None:
        join = " JOIN processing_campaigns c ON c.id=sr.campaign_id"
        clauses.append("c.id=?" if isinstance(campaign, int) else "c.name=?")
        params.append(campaign)
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS count FROM stage_requests sr "
            "LEFT JOIN workflow_executions we ON we.id=sr.workflow_execution_id"
            + join
            + " WHERE "
            + " AND ".join(clauses),
            params,
        ).fetchone()
        return int(row["count"])
    finally:
        conn.close()


def get_active_stage_request(
    dataset: str, sequence_name: str, stage: str, *, db_path: str = DB_PATH,
) -> dict | None:
    rows = list_stage_requests(
        dataset=dataset, stage=stage, status=_ACTIVE_REQUEST_STATUSES,
        db_path=db_path,
    )
    return next((row for row in rows if row["sequence_name"] == sequence_name), None)


def reserve_stage_request(
    request_id: int,
    *,
    reserved_by: str,
    lease_seconds: int = 900,
    force_blacklist: bool = False,
    db_path: str = DB_PATH,
) -> dict | None:
    if lease_seconds < 1:
        raise ValueError("Reservation lease must be positive")
    now_dt = datetime.now(timezone.utc)
    now = now_dt.isoformat()
    expires = (now_dt + timedelta(seconds=lease_seconds)).isoformat()
    conn = get_connection(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """SELECT sr.*, b.reason AS blacklist_reason,
                      c.status AS campaign_status, c.phase AS campaign_phase,
                      c.campaign_type
               FROM stage_requests sr
               LEFT JOIN blacklisted_sequences b ON b.sequence_id=sr.sequence_id
               LEFT JOIN processing_campaigns c ON c.id=sr.campaign_id
               WHERE sr.id=?"""
            + (" FOR UPDATE OF sr" if is_postgresql_connection(conn) else ""),
            (request_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown stage request: {request_id}")
        if row["status"] not in ("PENDING", "BLOCKED", "RESERVED"):
            return None
        if row["status"] == "RESERVED" and (
            row["workflow_execution_id"] is not None
            or not row["lease_expires_at"] or row["lease_expires_at"] >= now
        ):
            return None
        blocked_reason = None
        # Frozen campaign membership is an explicit, audited instruction to
        # process the sequence even when a legacy timeout-box entry exists.
        # A new campaign failure writes the blacklist again, and successful
        # publication is the only operation that clears it.
        campaign_bypasses_blacklist = row["campaign_type"] in _CAMPAIGN_TYPES
        if (
            row["blacklist_reason"]
            and not force_blacklist
            and not campaign_bypasses_blacklist
        ):
            blocked_reason = "BLACKLISTED"
        elif row["campaign_id"] and row["campaign_status"] not in ("FROZEN", "RUNNING"):
            blocked_reason = "CAMPAIGN_NOT_ACTIVE"
        elif row["cohort"] == "BULK" and row["campaign_phase"] == "CANARY":
            blocked_reason = "WAITING_CANARY_APPROVAL"
        if blocked_reason:
            conn.execute(
                """UPDATE stage_requests SET status='BLOCKED', blocked_reason=?,
                   reserved_by=NULL, reserved_at=NULL, lease_expires_at=NULL,
                   updated_at=? WHERE id=?""",
                (blocked_reason, now, request_id),
            )
            conn.commit()
            return None
        conn.execute(
            """UPDATE stage_requests SET status='RESERVED', blocked_reason=NULL,
               reserved_by=?, reserved_at=?, lease_expires_at=?, updated_at=?
               WHERE id=?""",
            (reserved_by, now, expires, now, request_id),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return get_stage_request(request_id, db_path=db_path)


def attach_request_execution(
    request_id: int,
    execution_id: int,
    *,
    status: str = "SUBMITTED",
    db_path: str = DB_PATH,
) -> dict:
    if status not in ("SUBMITTED", "RUNNING", "UNKNOWN"):
        raise ValueError("Execution attachment requires SUBMITTED, RUNNING, or UNKNOWN")
    now = _utc_now()
    conn = get_connection(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        request = conn.execute(
            "SELECT * FROM stage_requests WHERE id=?"
            + (" FOR UPDATE" if is_postgresql_connection(conn) else ""),
            (request_id,),
        ).fetchone()
        execution = conn.execute(
            "SELECT * FROM workflow_executions WHERE id=?", (execution_id,)
        ).fetchone()
        if request is None or execution is None:
            raise ValueError("Unknown request or workflow execution")
        if request["status"] != "RESERVED":
            raise ValueError("Only a reserved request can attach an execution")
        expected = request["stage"]
        if execution["pipeline_stage"] != expected:
            raise ValueError("Workflow execution stage does not match request")
        conn.execute(
            """UPDATE stage_requests SET workflow_execution_id=?, status=?,
               submitted_at=?, updated_at=?, lease_expires_at=NULL WHERE id=?""",
            (execution_id, status, now, now, request_id),
        )
        if request["campaign_id"]:
            conn.execute(
                """UPDATE processing_campaigns SET status='RUNNING',
                   started_at=COALESCE(started_at, ?), updated_at=?
                   WHERE id=? AND status='FROZEN'""",
                (now, now, request["campaign_id"]),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return get_stage_request(request_id, db_path=db_path)


def update_stage_request(
    request_id: int,
    *,
    status: str | None = None,
    blocked_reason: str | None = None,
    details: str | None = None,
    result_manifest_uri: str | None = None,
    result_manifest_sha256: str | None = None,
    result_summary: object | None = None,
    source_manifest: object | None = None,
    source_manifest_sha256: str | None = None,
    parameters: object | None = None,
    db_path: str = DB_PATH,
) -> dict:
    if status is not None and status not in _REQUEST_STATUSES:
        raise ValueError(f"Unsupported request status: {status}")
    if status == "BLOCKED" and not blocked_reason:
        raise ValueError("Blocked requests require a reason")
    now = _utc_now()
    updates = ["updated_at=?"]
    params: list = [now]
    if status is not None:
        updates.append("status=?")
        params.append(status)
        if status in ("SUCCEEDED", "FAILED", "CANCELED"):
            updates.append("completed_at=?")
            params.append(now)
    if blocked_reason is not None or status == "PENDING":
        updates.append("blocked_reason=?")
        params.append(blocked_reason)
    if details is not None:
        updates.append("details=?")
        params.append(details)
    if result_manifest_uri is not None:
        updates.append("result_manifest_uri=?")
        params.append(result_manifest_uri)
    if result_manifest_sha256 is not None:
        updates.append("result_manifest_sha256=?")
        params.append(result_manifest_sha256)
    if result_summary is not None:
        updates.append("result_summary_json=?")
        params.append(_canonical_json(result_summary))
    if source_manifest is not None:
        manifest_json = _canonical_json(source_manifest)
        computed = _json_sha256(manifest_json)
        if source_manifest_sha256 and source_manifest_sha256 != computed:
            raise ValueError("Source manifest SHA-256 does not match its canonical JSON")
        updates.extend(["source_manifest_json=?", "source_manifest_sha256=?"])
        params.extend([manifest_json, source_manifest_sha256 or computed])
    if parameters is not None:
        updates.append("parameters_json=?")
        params.append(_canonical_json(parameters))
    params.append(request_id)
    conn = get_connection(db_path)
    try:
        existing = conn.execute(
            "SELECT stage, status, result_manifest_uri, result_manifest_sha256 "
            "FROM stage_requests WHERE id=?", (request_id,)
        ).fetchone()
        if existing is None:
            raise ValueError(f"Unknown stage request: {request_id}")
        if status is not None:
            _guard_status_transition(
                "stage request", existing["status"], status,
                _REQUEST_TRANSITIONS,
            )
        if status == "SUCCEEDED" and existing["stage"] == "revalidation":
            final_uri = result_manifest_uri or existing["result_manifest_uri"]
            final_sha = result_manifest_sha256 or existing["result_manifest_sha256"]
            if not final_uri or not final_sha:
                raise ValueError(
                    "Successful revalidation requests require a verified result manifest"
                )
        if source_manifest is not None and existing["status"] not in ("PENDING", "BLOCKED"):
            raise ValueError(
                "A request source manifest is immutable after reservation"
            )
        if parameters is not None and existing["status"] not in ("PENDING", "BLOCKED"):
            raise ValueError(
                "Request parameters are immutable after reservation"
            )
        cursor = conn.execute(
            f"UPDATE stage_requests SET {', '.join(updates)} WHERE id=?", params
        )
        conn.commit()
    finally:
        conn.close()
    return get_stage_request(request_id, db_path=db_path)


def campaign_progress(
    campaign: int | str, *, db_path: str = DB_PATH,
) -> dict:
    row = get_campaign(campaign, db_path=db_path)
    if row is None:
        raise ValueError(f"Unknown campaign: {campaign}")
    conn = get_connection(db_path)
    try:
        counts = {
            item["status"]: item["count"] for item in conn.execute(
                """SELECT status, COUNT(*) AS count FROM stage_requests
                   WHERE campaign_id=? GROUP BY status""",
                (row["id"],),
            )
        }
        sequence_count = conn.execute(
            "SELECT COUNT(DISTINCT sequence_id) FROM stage_requests WHERE campaign_id=?",
            (row["id"],),
        ).fetchone()[0]
        committed_exports = conn.execute(
            """SELECT COUNT(DISTINCT sr.sequence_id)
               FROM stage_requests sr
               JOIN export_runs er
                 ON er.request_id=COALESCE(sr.fulfilled_by_request_id, sr.id)
               WHERE sr.campaign_id=? AND er.status='SUCCEEDED'""",
            (row["id"],),
        ).fetchone()[0]
    finally:
        conn.close()
    return {
        **row, "request_counts": counts, "sequence_count": sequence_count,
        "expected_sequence_count": row["inventory_sequence_count"],
        "committed_export_count": committed_exports,
    }


# -- Post-export intermediate cleanup ------------------------------------

def get_intermediate_cleanup_job(
    *, job_id: int | None = None, export_run_id: int | None = None,
    db_path: str = DB_PATH,
) -> dict | None:
    if (job_id is None) == (export_run_id is None):
        raise ValueError("Specify exactly one cleanup job or export run ID")
    column, value = (
        ("j.id", job_id) if job_id is not None
        else ("j.export_run_id", export_run_id)
    )
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            f"""SELECT j.*, s.dataset, s.sequence_name, pc.name AS campaign_name,
                       er.output_uri AS export_uri, er.completed_at AS export_completed_at
                FROM intermediate_cleanup_jobs j
                JOIN sequences s ON s.id=j.sequence_id
                JOIN processing_campaigns pc ON pc.id=j.campaign_id
                JOIN export_runs er ON er.id=j.export_run_id
                WHERE {column}=?""",
            (value,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def enqueue_intermediate_cleanup(
    export_run_id: int,
    *,
    source: str,
    source_audit_sha256: str | None = None,
    db_path: str = DB_PATH,
) -> dict:
    """Idempotently enqueue one exact committed campaign export."""
    normalized_source = source.upper()
    if normalized_source not in _CLEANUP_JOB_SOURCES:
        raise ValueError(f"Unsupported cleanup source: {source}")
    now = _utc_now()
    conn = get_connection(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        export = conn.execute(
            """SELECT er.id, er.sequence_id, er.status, er.request_id,
                      sr.campaign_id
               FROM export_runs er
               JOIN stage_requests sr ON sr.id=er.request_id
               WHERE er.id=?"""
            + (" FOR UPDATE OF er" if is_postgresql_connection(conn) else ""),
            (export_run_id,),
        ).fetchone()
        if export is None or export["status"] != "SUCCEEDED":
            raise ValueError("Cleanup requires a successful campaign export run")
        if export["campaign_id"] is None:
            raise ValueError("Cleanup requires campaign-owned export lineage")
        conn.execute(
            """INSERT OR IGNORE INTO intermediate_cleanup_jobs
               (export_run_id, sequence_id, campaign_id, status, source,
                source_audit_sha256, attempt_count, created_at, updated_at)
               VALUES (?, ?, ?, 'PENDING', ?, ?, 0, ?, ?)""",
            (
                export["id"], export["sequence_id"], export["campaign_id"],
                normalized_source, source_audit_sha256, now, now,
            ),
        )
        existing = conn.execute(
            "SELECT source, source_audit_sha256 FROM intermediate_cleanup_jobs "
            "WHERE export_run_id=?",
            (export_run_id,),
        ).fetchone()
        if (
            source_audit_sha256
            and existing["source_audit_sha256"] not in (None, source_audit_sha256)
        ):
            raise ValueError("Cleanup job belongs to a different backfill audit")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return get_intermediate_cleanup_job(
        export_run_id=export_run_id, db_path=db_path,
    )


def enqueue_missing_intermediate_cleanups(
    campaign: int | str,
    *,
    completed_after: str,
    source: str = "AUTOMATIC",
    db_path: str = DB_PATH,
) -> list[dict]:
    """Recover successful post-cutoff exports that missed direct enqueueing."""
    campaign_row = get_campaign(campaign, db_path=db_path)
    if campaign_row is None:
        raise ValueError(f"Unknown campaign: {campaign}")
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """SELECT er.id
               FROM export_runs er
               JOIN stage_requests sr ON sr.id=er.request_id
               LEFT JOIN intermediate_cleanup_jobs j ON j.export_run_id=er.id
               WHERE sr.campaign_id=? AND er.status='SUCCEEDED'
                 AND er.completed_at>=? AND j.id IS NULL
               ORDER BY er.completed_at, er.id""",
            (campaign_row["id"], completed_after),
        ).fetchall()
    finally:
        conn.close()
    return [
        enqueue_intermediate_cleanup(
            int(row["id"]), source=source, db_path=db_path,
        )
        for row in rows
    ]


def list_intermediate_cleanup_jobs(
    *,
    campaign: int | str | None = None,
    status: str | Iterable[str] | None = None,
    source: str | None = None,
    db_path: str = DB_PATH,
) -> list[dict]:
    clauses: list[str] = []
    params: list = []
    if campaign is not None:
        clauses.append("pc.id=?" if isinstance(campaign, int) else "pc.name=?")
        params.append(campaign)
    if status is not None:
        statuses = [status] if isinstance(status, str) else list(status)
        unknown = set(statuses) - _CLEANUP_JOB_STATUSES
        if unknown:
            raise ValueError(f"Unsupported cleanup statuses: {sorted(unknown)}")
        clauses.append("j.status IN (" + ",".join("?" for _ in statuses) + ")")
        params.extend(statuses)
    if source is not None:
        normalized_source = source.upper()
        if normalized_source not in _CLEANUP_JOB_SOURCES:
            raise ValueError(f"Unsupported cleanup source: {source}")
        clauses.append("j.source=?")
        params.append(normalized_source)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """SELECT j.*, s.dataset, s.sequence_name, pc.name AS campaign_name,
                      er.output_uri AS export_uri, er.completed_at AS export_completed_at
               FROM intermediate_cleanup_jobs j
               JOIN sequences s ON s.id=j.sequence_id
               JOIN processing_campaigns pc ON pc.id=j.campaign_id
               JOIN export_runs er ON er.id=j.export_run_id"""
            + where
            + " ORDER BY CASE j.source WHEN 'AUTOMATIC' THEN 0 "
              "WHEN 'MANUAL' THEN 1 ELSE 2 END, j.created_at, j.id",
            params,
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def reserve_intermediate_cleanup_jobs(
    *,
    reserved_by: str,
    limit: int,
    campaign: int | str | None = None,
    export_run_ids: Iterable[int] | None = None,
    lease_seconds: int = 1800,
    db_path: str = DB_PATH,
) -> list[dict]:
    if not reserved_by or limit < 1 or lease_seconds < 1:
        raise ValueError("Invalid cleanup reservation parameters")
    requested_ids = list(dict.fromkeys(int(value) for value in (export_run_ids or ())))
    now_dt = datetime.now(timezone.utc)
    now = now_dt.isoformat()
    lease_expires = (now_dt + timedelta(seconds=lease_seconds)).isoformat()
    conn = get_connection(db_path)
    reserved_rows = []
    try:
        conn.execute("BEGIN IMMEDIATE")
        clauses = [
            "(j.status='PENDING' OR "
            "(j.status IN ('RUNNING','BLOCKED') AND j.lease_expires_at<?))",
        ]
        params: list = [now]
        if campaign is not None:
            clauses.append("pc.id=?" if isinstance(campaign, int) else "pc.name=?")
            params.append(campaign)
        if requested_ids:
            clauses.append(
                "j.export_run_id IN (" + ",".join("?" for _ in requested_ids) + ")"
            )
            params.extend(requested_ids)
        params.append(limit)
        rows = conn.execute(
            """SELECT j.id FROM intermediate_cleanup_jobs j
               JOIN processing_campaigns pc ON pc.id=j.campaign_id
               WHERE """ + " AND ".join(clauses)
            + " ORDER BY CASE j.source WHEN 'AUTOMATIC' THEN 0 "
              "WHEN 'MANUAL' THEN 1 ELSE 2 END, j.created_at, j.id LIMIT ?"
            + (
                " FOR UPDATE OF j SKIP LOCKED"
                if is_postgresql_connection(conn) else ""
            ),
            params,
        ).fetchall()
        ids = [int(row["id"]) for row in rows]
        if ids:
            placeholders = ",".join("?" for _ in ids)
            conn.execute(
                f"""UPDATE intermediate_cleanup_jobs
                    SET status='RUNNING', lease_owner=?, lease_expires_at=?,
                        attempt_count=attempt_count+1,
                        started_at=COALESCE(started_at, ?), updated_at=?
                    WHERE id IN ({placeholders})""",
                (reserved_by, lease_expires, now, now, *ids),
            )
            reserved_rows = conn.execute(
                f"""SELECT j.*, s.dataset, s.sequence_name,
                           pc.name AS campaign_name, er.output_uri AS export_uri,
                           er.completed_at AS export_completed_at
                    FROM intermediate_cleanup_jobs j
                    JOIN sequences s ON s.id=j.sequence_id
                    JOIN processing_campaigns pc ON pc.id=j.campaign_id
                    JOIN export_runs er ON er.id=j.export_run_id
                    WHERE j.id IN ({placeholders})
                    ORDER BY CASE j.source WHEN 'AUTOMATIC' THEN 0
                                  WHEN 'MANUAL' THEN 1 ELSE 2 END,
                             j.created_at, j.id""",
                ids,
            ).fetchall()
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return [dict(row) for row in reserved_rows]


def finish_intermediate_cleanup_job(
    job_id: int,
    *,
    status: str,
    details: str | None = None,
    manifest_uri: str | None = None,
    manifest_sha256: str | None = None,
    protected_manifest_sha256: str | None = None,
    deleted_object_count: int | None = None,
    reclaimed_bytes: int | None = None,
    retry_after_seconds: int | None = None,
    db_path: str = DB_PATH,
) -> dict:
    if status not in _CLEANUP_JOB_STATUSES - {"PENDING", "RUNNING"}:
        raise ValueError(f"Unsupported cleanup completion status: {status}")
    now_dt = datetime.now(timezone.utc)
    now = now_dt.isoformat()
    retry_at = (
        (now_dt + timedelta(seconds=retry_after_seconds)).isoformat()
        if status == "BLOCKED" and retry_after_seconds else None
    )
    completed_at = now if status in ("FAILED", "SUCCEEDED") else None
    conn = get_connection(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT status FROM intermediate_cleanup_jobs WHERE id=?"
            + (" FOR UPDATE" if is_postgresql_connection(conn) else ""),
            (job_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown cleanup job: {job_id}")
        if row["status"] == "SUCCEEDED" and status != "SUCCEEDED":
            raise ValueError("A completed cleanup job is immutable")
        conn.execute(
            """UPDATE intermediate_cleanup_jobs
               SET status=?, details=?, manifest_uri=COALESCE(?, manifest_uri),
                   manifest_sha256=COALESCE(?, manifest_sha256),
                   protected_manifest_sha256=COALESCE(?, protected_manifest_sha256),
                   deleted_object_count=COALESCE(?, deleted_object_count),
                   reclaimed_bytes=COALESCE(?, reclaimed_bytes),
                   lease_owner=NULL, lease_expires_at=?, completed_at=?, updated_at=?
               WHERE id=?""",
            (
                status, details, manifest_uri, manifest_sha256,
                protected_manifest_sha256, deleted_object_count, reclaimed_bytes,
                retry_at, completed_at, now, job_id,
            ),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return get_intermediate_cleanup_job(job_id=job_id, db_path=db_path)


def update_intermediate_cleanup_job_progress(
    job_id: int, *, details: dict, db_path: str = DB_PATH,
) -> dict:
    """Persist idempotent phase progress while a cleanup lease is active."""
    now = _utc_now()
    conn = get_connection(db_path)
    try:
        conn.execute(
            "UPDATE intermediate_cleanup_jobs SET details=?, updated_at=? "
            "WHERE id=? AND status='RUNNING'",
            (json.dumps(details, sort_keys=True), now, job_id),
        )
        conn.commit()
    finally:
        conn.close()
    return get_intermediate_cleanup_job(job_id=job_id, db_path=db_path)


def adopt_request_fulfillment(
    target_request_id: int,
    source_request_id: int,
    *,
    db_path: str = DB_PATH,
) -> dict:
    """Fulfill one frozen campaign entry with a newer published request.

    The target request remains immutable history (normally a canceled request
    that was superseded for a priority retry). Campaign accounting follows the
    explicit link to the newer request's committed REVALIDATION export.
    """
    if target_request_id == source_request_id:
        raise ValueError("A request cannot fulfill itself")
    conn = get_connection(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        target = conn.execute(
            """SELECT sr.*, pc.dataset AS campaign_dataset,
                      pc.output_uri AS campaign_output_uri
               FROM stage_requests sr
               JOIN processing_campaigns pc ON pc.id=sr.campaign_id
               WHERE sr.id=?""",
            (target_request_id,),
        ).fetchone()
        source = conn.execute(
            """SELECT sr.*, s.dataset AS sequence_dataset
               FROM stage_requests sr
               JOIN sequences s ON s.id=sr.sequence_id
               WHERE sr.id=?""",
            (source_request_id,),
        ).fetchone()
        if target is None:
            raise ValueError(
                f"Target request {target_request_id} is not a campaign request"
            )
        if source is None:
            raise ValueError(f"Unknown source request: {source_request_id}")
        if target["fulfilled_by_request_id"] not in (None, source_request_id):
            raise ValueError(
                f"Target request {target_request_id} is already fulfilled by "
                f"request {target['fulfilled_by_request_id']}"
            )
        if target["sequence_id"] != source["sequence_id"]:
            raise ValueError("Fulfillment requests must belong to the same sequence")
        if target["stage"] != "revalidation" or source["stage"] != "revalidation":
            raise ValueError("Cross-campaign fulfillment is limited to revalidation")
        if target["campaign_dataset"] != source["sequence_dataset"]:
            raise ValueError("Fulfillment requests must belong to the same dataset")
        if source["status"] != "SUCCEEDED":
            raise ValueError("The fulfilling request must have succeeded")
        export = conn.execute(
            """SELECT * FROM export_runs
               WHERE request_id=? AND status='SUCCEEDED'
                 AND authorization_type='REVALIDATION'""",
            (source_request_id,),
        ).fetchone()
        if export is None:
            raise ValueError(
                "The fulfilling request must have a committed REVALIDATION export"
            )
        expected_prefix = (
            str(target["campaign_output_uri"]).rstrip("/") + "/"
        )
        output_uri = str(export["output_uri"] or "").rstrip("/") + "/"
        if not output_uri.startswith(expected_prefix):
            raise ValueError(
                "The fulfilling export is outside the target campaign output prefix"
            )
        if target["fulfilled_by_request_id"] is None:
            conn.execute(
                """UPDATE stage_requests
                   SET fulfilled_by_request_id=?, updated_at=?
                   WHERE id=?""",
                (source_request_id, _utc_now(), target_request_id),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return get_stage_request(target_request_id, db_path=db_path)


def finish_campaign(
    campaign: int | str, *, db_path: str = DB_PATH,
) -> dict:
    progress = campaign_progress(campaign, db_path=db_path)
    counts = progress["request_counts"]
    active = sum(
        counts.get(status, 0)
        for status in ("PENDING", "RESERVED", "SUBMITTED", "RUNNING", "UNKNOWN")
    )
    if active:
        raise ValueError(f"Campaign still has {active} active request(s)")
    if progress["sequence_count"] == 0:
        raise ValueError("Cannot complete an empty campaign")
    if (
        progress["expected_sequence_count"] is not None
        and progress["sequence_count"] != progress["expected_sequence_count"]
    ):
        raise ValueError(
            "Campaign request membership does not match its frozen inventory count"
        )
    conn = get_connection(db_path)
    try:
        outcomes = conn.execute(
            """SELECT sr.sequence_id,
                      MAX(CASE WHEN er.status='SUCCEEDED' THEN 1 ELSE 0 END) AS exported,
                      MAX(CASE WHEN sr.status IN ('FAILED','CANCELED','BLOCKED')
                               THEN 1 ELSE 0 END) AS failed
               FROM stage_requests sr
               LEFT JOIN export_runs er
                 ON er.request_id=COALESCE(sr.fulfilled_by_request_id, sr.id)
               WHERE sr.campaign_id=? GROUP BY sr.sequence_id""",
            (progress["id"],),
        ).fetchall()
    finally:
        conn.close()
    unresolved = [item["sequence_id"] for item in outcomes if not item["exported"] and not item["failed"]]
    if unresolved:
        raise ValueError(
            f"Campaign has {len(unresolved)} sequence(s) without a committed export or terminal failure"
        )
    failed = sum(bool(item["failed"]) and not bool(item["exported"]) for item in outcomes)
    if len(outcomes) != progress["sequence_count"]:
        raise RuntimeError("Campaign sequence outcome count did not reconcile")
    status = "COMPLETED_WITH_FAILURES" if failed else "SUCCEEDED"
    now = _utc_now()
    conn = get_connection(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        # BLOCKED is deliberately active while a campaign is open. Closing the
        # campaign preserves its blocker but terminalizes the request so a
        # later frozen REMEDIATION campaign can create fresh intent.
        conn.execute(
            """UPDATE stage_requests SET status='FAILED',
               details=COALESCE(details, blocked_reason, 'campaign_closed_blocked'),
               completed_at=COALESCE(completed_at, ?), updated_at=?
               WHERE campaign_id=? AND status='BLOCKED'""",
            (now, now, progress["id"]),
        )
        conn.execute(
            """UPDATE processing_campaigns SET status=?, phase='COMPLETE',
               completed_at=?, updated_at=? WHERE id=?""",
            (status, now, now, progress["id"]),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return get_campaign(progress["id"], db_path=db_path)


def cancel_campaign(
    campaign: int | str, *, db_path: str = DB_PATH,
) -> dict:
    progress = campaign_progress(campaign, db_path=db_path)
    counts = progress["request_counts"]
    remote_active = sum(
        counts.get(status, 0) for status in ("SUBMITTED", "RUNNING", "UNKNOWN")
    )
    if remote_active:
        raise ValueError(
            f"Campaign has {remote_active} remote/ambiguous request(s); cancel those "
            "executions before closing the campaign"
        )
    now = _utc_now()
    conn = get_connection(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """UPDATE stage_requests SET status='CANCELED', completed_at=?,
               updated_at=?, details='campaign_canceled'
               WHERE campaign_id=? AND status IN ('PENDING','BLOCKED','RESERVED')""",
            (now, now, progress["id"]),
        )
        conn.execute(
            """UPDATE processing_campaigns SET status='CANCELED', phase='COMPLETE',
               completed_at=?, updated_at=? WHERE id=?""",
            (now, now, progress["id"]),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return get_campaign(progress["id"], db_path=db_path)


def adopt_valid_replacement_preprocess_results(
    source_campaign: int | str,
    replacement_campaign: int | str,
    *,
    adopted_by: str,
    db_path: str = DB_PATH,
) -> dict:
    """Adopt successful transferred canary preprocessing after a patch retry.

    This is deliberately narrower than ordinary preprocessing reuse.  It only
    accepts priority-100 transferred members whose frozen source manifest is
    byte-identical across two immutable campaign inventories.  The source
    campaign must already be canceled and the replacement must still be a
    frozen canary.  No workflow or stage run is cloned: the replacement request
    records the current successful source run as provenance, allowing downstream
    reconstruction to depend on that run without duplicating preprocessing.
    """
    if not adopted_by.strip():
        raise ValueError("Preprocess adoption requires an owner")
    now = _utc_now()
    conn = get_connection(db_path)
    adopted: list[dict] = []
    existing: list[dict] = []
    try:
        conn.execute("BEGIN IMMEDIATE")

        def campaign_row(value: int | str):
            column = "id" if isinstance(value, int) else "name"
            return conn.execute(
                f"SELECT * FROM processing_campaigns WHERE {column}=?"
                + (" FOR UPDATE" if is_postgresql_connection(conn) else ""),
                (value,),
            ).fetchone()

        source = campaign_row(source_campaign)
        replacement = campaign_row(replacement_campaign)
        if source is None or replacement is None:
            raise ValueError("Unknown source or replacement campaign")
        if (
            source["campaign_type"] != "BACKLOG_REPROCESSING"
            or replacement["campaign_type"] != "BACKLOG_REPROCESSING"
        ):
            raise ValueError("Preprocess adoption is limited to backlog campaigns")
        if source["status"] != "CANCELED":
            raise ValueError("Source campaign must be canceled before adoption")
        if replacement["status"] != "FROZEN" or replacement["phase"] != "CANARY":
            raise ValueError("Replacement must be a frozen canary before adoption")
        if source["pipeline_version"] == replacement["pipeline_version"]:
            raise ValueError("Adoption requires a patch-version replacement")
        for field in ("dataset", "inventory_sha256", "inventory_sequence_count", "output_uri"):
            if source[field] != replacement[field]:
                raise ValueError(f"Replacement differs from source {field}")

        rows = conn.execute(
            """SELECT tr.id AS target_request_id, tr.status AS target_status,
                      tr.queue_priority AS target_priority,
                      tr.cohort AS target_cohort,
                      tr.source_manifest_json AS target_manifest_json,
                      tr.source_manifest_sha256 AS target_manifest_sha256,
                      tr.parameters_json AS target_parameters_json,
                      tr.workflow_execution_id AS target_execution_id,
                      tr.result_summary_json AS target_result_summary_json,
                      sr.id AS source_request_id,
                      sr.status AS source_status,
                      sr.queue_priority AS source_priority,
                      sr.source_manifest_json AS source_manifest_json,
                      sr.source_manifest_sha256 AS source_manifest_sha256,
                      sr.result_summary_json AS source_result_summary_json,
                      pr.id AS preprocess_run_id,
                      pr.request_id AS preprocess_run_request_id,
                      pr.status AS preprocess_run_status,
                      pr.pipeline_version AS preprocess_pipeline_version,
                      pr.output_uri AS preprocess_output_uri,
                      pr.is_current AS preprocess_is_current,
                      s.sequence_name
               FROM stage_requests tr
               JOIN sequences s ON s.id=tr.sequence_id
               JOIN stage_requests sr ON sr.sequence_id=tr.sequence_id
                    AND sr.campaign_id=? AND sr.stage='preprocess'
               JOIN preprocess_runs pr ON pr.sequence_id=tr.sequence_id
                    AND pr.is_current=1
               WHERE tr.campaign_id=? AND tr.stage='preprocess'
                 AND tr.cohort='CANARY' AND tr.queue_priority=100
                 AND tr.status IN ('PENDING','SUCCEEDED')
                 AND sr.status='SUCCEEDED'
                 AND pr.status='SUCCEEDED' AND pr.is_current=1
               ORDER BY tr.created_at, tr.id""",
            (source["id"], replacement["id"]),
        ).fetchall()
        for row in rows:
            item = dict(row)
            try:
                source_manifest = json.loads(item["source_manifest_json"] or "{}")
                target_manifest = json.loads(item["target_manifest_json"] or "{}")
            except (TypeError, ValueError) as exc:
                raise ValueError("Invalid transferred source manifest JSON") from exc
            transfer = target_manifest.get("membership_transfer")
            lineage = target_manifest.get("historical_preprocess_lineage")
            source_summary = json.loads(item["source_result_summary_json"] or "{}")
            prior_adoption = source_summary.get("preprocess_adoption") or {}
            run_bound_to_source = (
                int(item["preprocess_run_request_id"] or -1)
                == int(item["source_request_id"])
                or (
                    int(prior_adoption.get("preprocess_run_id", -1))
                    == int(item["preprocess_run_id"])
                    and prior_adoption.get("preprocess_output_uri")
                    == item["preprocess_output_uri"]
                )
            )
            if not (
                item["source_status"] == "SUCCEEDED"
                and item["source_priority"] == 100
                and item["source_manifest_sha256"] == item["target_manifest_sha256"]
                and source_manifest == target_manifest
                and target_manifest.get("route")
                == "reprocessing_missing_revalidation_input"
                and isinstance(transfer, dict)
                and transfer.get("source_campaign") == "legacy_revalidation"
                and isinstance(lineage, dict)
                and lineage.get("reuse_permitted") is False
                and item["preprocess_run_status"] == "SUCCEEDED"
                and bool(item["preprocess_is_current"])
                and run_bound_to_source
                and parse_semver(item["preprocess_pipeline_version"])
                <= parse_semver(replacement["pipeline_version"])
                and bool(item["preprocess_output_uri"])
            ):
                raise ValueError(
                    f"Source preprocessing is not adoptable for {item['sequence_name']}"
                )
            evidence = {
                "schema": "v2d.mv_hoi.preprocess_patch_adoption.v1",
                "adopted_by": adopted_by,
                "source_campaign_id": source["id"],
                "source_request_id": item["source_request_id"],
                "preprocess_run_id": item["preprocess_run_id"],
                "preprocess_pipeline_version": item["preprocess_pipeline_version"],
                "preprocess_output_uri": item["preprocess_output_uri"],
                "target_pipeline_version": replacement["pipeline_version"],
                "source_manifest_sha256": item["source_manifest_sha256"],
                "reason": "implementation_patch_does_not_affect_preprocessing",
            }
            if item["target_status"] == "SUCCEEDED":
                summary = json.loads(item["target_result_summary_json"] or "{}")
                if summary.get("preprocess_adoption") != evidence:
                    raise ValueError(
                        f"Conflicting adoption already exists for {item['sequence_name']}"
                    )
                existing.append({"sequence": item["sequence_name"], **evidence})
                continue
            if item["target_execution_id"] is not None:
                raise ValueError("An executed replacement preprocess cannot be adopted")
            parameters = json.loads(item["target_parameters_json"] or "{}")
            parameters["preprocess_adoption"] = evidence
            conn.execute(
                """UPDATE stage_requests SET status='SUCCEEDED', details=?,
                          result_summary_json=?, parameters_json=?, completed_at=?,
                          updated_at=? WHERE id=? AND status='PENDING'
                          AND workflow_execution_id IS NULL""",
                (
                    f"adopted_valid_preprocess_from_request_"
                    f"{item['source_request_id']}_run_{item['preprocess_run_id']}",
                    _canonical_json({"preprocess_adoption": evidence}),
                    _canonical_json(parameters), now, now,
                    item["target_request_id"],
                ),
            )
            adopted.append({"sequence": item["sequence_name"], **evidence})
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return {
        "source_campaign_id": int(source["id"]),
        "replacement_campaign_id": int(replacement["id"]),
        "adopted_count": len(adopted),
        "already_adopted_count": len(existing),
        "adopted": adopted,
        "already_adopted": existing,
    }


def adopt_valid_replacement_revalidation_results(
    source_campaign: int | str,
    replacement_campaign: int | str,
    *,
    adopted_by: str,
    verified_work_outputs: dict[int, dict] | None = None,
    db_path: str = DB_PATH,
) -> dict:
    """Reuse terminal revalidation evidence while rerunning only export work."""
    if not adopted_by.strip():
        raise ValueError("Revalidation adoption requires an owner")
    now = _utc_now()
    verified_work_outputs = verified_work_outputs or {}
    conn = get_connection(db_path)
    export_retries: list[dict] = []
    carried_failures: list[dict] = []
    try:
        conn.execute("BEGIN IMMEDIATE")

        def campaign_row(value: int | str):
            column = "id" if isinstance(value, int) else "name"
            return conn.execute(
                f"SELECT * FROM processing_campaigns WHERE {column}=?"
                + (" FOR UPDATE" if is_postgresql_connection(conn) else ""),
                (value,),
            ).fetchone()

        source = campaign_row(source_campaign)
        replacement = campaign_row(replacement_campaign)
        if source is None or replacement is None:
            raise ValueError("Unknown source or replacement campaign")
        if (
            source["campaign_type"] != "LEGACY_REVALIDATION"
            or replacement["campaign_type"] != "LEGACY_REVALIDATION"
        ):
            raise ValueError("Revalidation adoption requires revalidation campaigns")
        if source["status"] != "CANCELED":
            raise ValueError("Source campaign must be canceled before adoption")
        if replacement["status"] != "FROZEN" or replacement["phase"] != "CANARY":
            raise ValueError("Replacement must be a frozen canary before adoption")
        for field in ("dataset", "inventory_sha256", "inventory_sequence_count", "output_uri"):
            if source[field] != replacement[field]:
                raise ValueError(f"Replacement differs from source {field}")

        rows = conn.execute(
            """SELECT tr.*, sr.id AS source_request_id,
                      sr.status AS source_status,
                      sr.details AS source_details,
                      sr.result_manifest_uri AS source_result_manifest_uri,
                      sr.result_manifest_sha256 AS source_result_manifest_sha256,
                      sr.result_summary_json AS source_result_summary_json,
                      sr.source_manifest_sha256 AS source_source_manifest_sha256,
                      sr.workflow_execution_id AS source_execution_id,
                      we.status AS source_execution_status,
                      we.osmo_workflow_id AS source_workflow_id,
                      we.last_query_payload_json AS source_query_payload_json,
                      s.sequence_name
               FROM stage_requests tr
               JOIN sequences s ON s.id=tr.sequence_id
               JOIN stage_requests sr ON sr.sequence_id=tr.sequence_id
                    AND sr.campaign_id=? AND sr.stage='revalidation'
                    AND sr.cohort='CANARY'
               LEFT JOIN workflow_executions we ON we.id=sr.workflow_execution_id
               WHERE tr.campaign_id=? AND tr.stage='revalidation'
                 AND tr.cohort='CANARY'
               ORDER BY tr.created_at, tr.id""",
            (source["id"], replacement["id"]),
        ).fetchall()
        for row in rows:
            item = dict(row)
            if item["source_manifest_sha256"] != item["source_source_manifest_sha256"]:
                raise ValueError(
                    f"Frozen source identity differs for {item['sequence_name']}"
                )
            parameters = json.loads(item["parameters_json"] or "{}")
            summary = json.loads(item["source_result_summary_json"] or "{}")
            remote_evidence = verified_work_outputs.get(int(item["source_request_id"]))
            if remote_evidence is not None:
                query_payload = json.loads(
                    item["source_query_payload_json"] or "{}"
                )
                valid_remote_identity = (
                    item["source_status"] == "CANCELED"
                    and str(item.get("source_details") or "").startswith(
                        "invalid_generation_retired:remote_completed"
                    )
                    and item["source_execution_status"] == "SUCCEEDED"
                    and query_payload.get("status") == "COMPLETED"
                    and query_payload.get("tasks")
                    and all(
                        status == "COMPLETED"
                        for status in query_payload["tasks"].values()
                    )
                    and remote_evidence.get("workflow_execution_id")
                    == item["source_execution_id"]
                    and remote_evidence.get("workflow_id")
                    == item["source_workflow_id"]
                )
                if not valid_remote_identity:
                    raise ValueError(
                        f"Remote completion evidence is invalid for "
                        f"{item['sequence_name']}"
                    )
            if item["source_status"] == "SUCCEEDED" or (
                item["source_status"] == "CANCELED" and remote_evidence is not None
            ):
                uri = (
                    item["source_result_manifest_uri"]
                    if item["source_status"] == "SUCCEEDED"
                    else remote_evidence.get("candidate_commit_uri")
                )
                digest = (
                    item["source_result_manifest_sha256"]
                    if item["source_status"] == "SUCCEEDED"
                    else remote_evidence.get("candidate_commit_sha256")
                )
                if not uri or not digest:
                    raise ValueError(
                        f"Successful source lacks verified result: {item['sequence_name']}"
                    )
                parts = str(uri).rsplit("/", 2)
                if len(parts) != 3 or parts[1] != "candidate_export":
                    raise ValueError(
                        f"Unsupported source candidate URI: {item['sequence_name']}"
                    )
                evidence = {
                    "schema": "v2d.mv_hoi.revalidation_export_policy_adoption.v1",
                    "adopted_by": adopted_by,
                    "source_campaign_id": int(source["id"]),
                    "source_request_id": int(item["source_request_id"]),
                    "source_pipeline_version": source["pipeline_version"],
                    "target_pipeline_version": replacement["pipeline_version"],
                    "source_manifest_sha256": item["source_manifest_sha256"],
                    "reason": "symmetric_trim_changes_export_only",
                    "remote_completion_reconciled": (
                        item["source_status"] == "CANCELED"
                    ),
                }
                if remote_evidence is not None:
                    for key in (
                        "workflow_execution_id", "workflow_id", "commit_schema",
                        "commit_file_count",
                    ):
                        evidence[key] = remote_evidence.get(key)
                parameters["retry_work_output_url"] = parts[0]
                parameters["revalidation_adoption"] = evidence
                conn.execute(
                    """UPDATE stage_requests SET status='PENDING', blocked_reason=NULL,
                              details=?, parameters_json=?, result_manifest_uri=NULL,
                              result_manifest_sha256=NULL, result_summary_json=NULL,
                              completed_at=NULL, updated_at=?
                       WHERE id=? AND workflow_execution_id IS NULL""",
                    (
                        f"adopted_request_{item['source_request_id']}:export_only_retry",
                        _canonical_json(parameters), now, item["id"],
                    ),
                )
                export_retries.append({"sequence": item["sequence_name"], **evidence})
            elif (
                item["source_status"] == "FAILED"
                and summary.get("canary_outcome") == "EXPECTED_QC_DATA"
                and summary.get("evidence_valid") is True
                and summary.get("failure_category") in {
                    "accuracy_limit", "source_data", "compare_foundation_pose",
                }
            ):
                conn.execute(
                    """UPDATE stage_requests SET status='FAILED', blocked_reason=NULL,
                              details=?, result_summary_json=?, completed_at=?, updated_at=?
                       WHERE id=? AND workflow_execution_id IS NULL""",
                    (
                        f"carried_expected_outcome_from_request_{item['source_request_id']}",
                        _canonical_json(summary), now, now, item["id"],
                    ),
                )
                carried_failures.append({
                    "sequence": item["sequence_name"],
                    "source_request_id": int(item["source_request_id"]),
                    "failure_category": summary["failure_category"],
                })
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return {
        "source_campaign_id": int(source["id"]),
        "replacement_campaign_id": int(replacement["id"]),
        "export_retry_count": len(export_retries),
        "carried_expected_failure_count": len(carried_failures),
        "export_retries": export_retries,
        "carried_expected_failures": carried_failures,
    }


def adopt_valid_replacement_reconstruction_results(
    source_campaign: int | str,
    replacement_campaign: int | str,
    *,
    adopted_by: str,
    db_path: str = DB_PATH,
) -> dict:
    """Clone verified reconstruction lineage for an export-only policy patch."""
    if not adopted_by.strip():
        raise ValueError("Reconstruction adoption requires an owner")
    now = _utc_now()
    conn = get_connection(db_path)
    adopted: list[dict] = []
    carried_failures: list[dict] = []
    try:
        conn.execute("BEGIN IMMEDIATE")

        def campaign_row(value: int | str):
            column = "id" if isinstance(value, int) else "name"
            return conn.execute(
                f"SELECT * FROM processing_campaigns WHERE {column}=?"
                + (" FOR UPDATE" if is_postgresql_connection(conn) else ""),
                (value,),
            ).fetchone()

        source = campaign_row(source_campaign)
        replacement = campaign_row(replacement_campaign)
        if source is None or replacement is None:
            raise ValueError("Unknown source or replacement campaign")
        if (
            source["campaign_type"] != "BACKLOG_REPROCESSING"
            or replacement["campaign_type"] != "BACKLOG_REPROCESSING"
        ):
            raise ValueError("Reconstruction adoption requires backlog campaigns")
        if source["status"] != "CANCELED":
            raise ValueError("Source campaign must be canceled before adoption")
        if replacement["status"] != "FROZEN" or replacement["phase"] != "CANARY":
            raise ValueError("Replacement must be a frozen canary before adoption")
        for field in ("dataset", "inventory_sha256", "inventory_sequence_count", "output_uri"):
            if source[field] != replacement[field]:
                raise ValueError(f"Replacement differs from source {field}")

        rows = conn.execute(
            """SELECT tr.*, sr.id AS source_request_id,
                      sr.status AS source_status,
                      sr.details AS source_details,
                      sr.result_summary_json AS source_result_summary_json,
                      sr.source_manifest_sha256 AS source_source_manifest_sha256,
                      rr.id AS source_run_id,
                      rr.workflow_execution_id AS source_execution_id,
                      rr.pipeline_version AS source_run_pipeline_version,
                      rr.status AS source_run_status,
                      rr.output_uri AS source_output_uri,
                      rr.preprocess_run_id, rr.labeled_bboxes_uri,
                      rr.labeled_bboxes_manifest_json, rr.labeled_bboxes_sha256,
                      rr.bbox_source, rr.hitl_item_id, rr.started_at,
                      rr.completed_at AS source_run_completed_at,
                      we.status AS source_execution_status,
                      we.last_query_payload_json AS source_query_payload_json,
                      s.sequence_name
               FROM stage_requests tr
               JOIN sequences s ON s.id=tr.sequence_id
               JOIN stage_requests sr ON sr.sequence_id=tr.sequence_id
                    AND sr.campaign_id=? AND sr.stage='reconstruction'
                    AND sr.cohort='CANARY'
               LEFT JOIN reconstruction_runs rr ON rr.request_id=sr.id
               LEFT JOIN workflow_executions we ON we.id=rr.workflow_execution_id
               WHERE tr.campaign_id=? AND tr.stage='reconstruction'
                 AND tr.cohort='CANARY'
               ORDER BY tr.created_at, tr.id""",
            (source["id"], replacement["id"]),
        ).fetchall()
        for row in rows:
            item = dict(row)
            summary = json.loads(item["source_result_summary_json"] or "{}")
            query_payload = json.loads(item["source_query_payload_json"] or "{}")
            remotely_completed_retired = (
                item["source_status"] == "CANCELED"
                and item["source_run_status"] == "CANCELED"
                and str(item.get("source_details") or "").startswith(
                    "invalid_generation_retired:remote_completed"
                )
                and item["source_execution_status"] == "SUCCEEDED"
                and query_payload.get("status") == "COMPLETED"
                and query_payload.get("tasks")
                and all(
                    status == "COMPLETED"
                    for status in query_payload["tasks"].values()
                )
            )
            successful_source = (
                item["source_status"] == "SUCCEEDED"
                and item["source_run_status"] == "SUCCEEDED"
            ) or remotely_completed_retired
            expected_failure = (
                item["source_status"] == "FAILED"
                and summary.get("canary_outcome") == "EXPECTED_QC_DATA"
                and summary.get("evidence_valid") is True
                and summary.get("failure_category") in {"accuracy_limit", "source_data"}
            )
            # A sequence can have several source requests due to infrastructure
            # relocation or retry.  Non-adoptable attempts carry no reusable
            # lineage and must not veto a later verified completion for the same
            # target request.
            if not (successful_source or expected_failure):
                continue

            target_manifest = json.loads(item["source_manifest_json"] or "{}")
            exact_manifest_identity = (
                item["source_manifest_sha256"]
                == item["source_source_manifest_sha256"]
            )
            stage_manifest_identity = (
                item.get("labeled_bboxes_sha256")
                and item["source_source_manifest_sha256"]
                == item["labeled_bboxes_sha256"]
                and int(target_manifest.get("preprocess_run_id") or -1)
                == int(item.get("preprocess_run_id") or -2)
                and target_manifest.get("sequence") == item["sequence_name"]
            )
            if not (exact_manifest_identity or stage_manifest_identity):
                raise ValueError(
                    f"Frozen source identity differs for {item['sequence_name']}"
                )

            if successful_source:
                if not item["source_execution_id"] or not item["source_output_uri"]:
                    raise ValueError(
                        f"Successful reconstruction lacks lineage: {item['sequence_name']}"
                    )
                existing_run = conn.execute(
                    "SELECT * FROM reconstruction_runs WHERE request_id=?",
                    (item["id"],),
                ).fetchone()
                if existing_run is None:
                    run_id = _next_run_id(conn)
                    conn.execute(
                        """INSERT INTO reconstruction_runs
                           (id, sequence_id, workflow_execution_id, request_id,
                            pipeline_version, status, details, trigger, requested_by,
                            output_uri, is_current, created_at, started_at, completed_at,
                            updated_at, preprocess_run_id, labeled_bboxes_uri,
                            labeled_bboxes_manifest_json, labeled_bboxes_sha256,
                           bbox_source, hitl_item_id)
                           VALUES (?, ?, ?, ?, ?, 'SUCCEEDED', ?, 'MIGRATION', ?, ?,
                                   0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            run_id, item["sequence_id"], item["source_execution_id"],
                            item["id"], item["source_run_pipeline_version"],
                            f"adopted_reconstruction_run_{item['source_run_id']}",
                            adopted_by, item["source_output_uri"], now,
                            item["started_at"], item["source_run_completed_at"], now,
                            item["preprocess_run_id"], item["labeled_bboxes_uri"],
                            item["labeled_bboxes_manifest_json"],
                            item["labeled_bboxes_sha256"], item["bbox_source"],
                            item["hitl_item_id"],
                        ),
                    )
                    _set_current(conn, RECONSTRUCTION_STAGE, item["sequence_id"], run_id)
                else:
                    run_id = int(existing_run["id"])
                    if existing_run["output_uri"] != item["source_output_uri"]:
                        raise ValueError("Conflicting reconstruction adoption exists")
                evidence = {
                    "schema": "v2d.mv_hoi.reconstruction_export_policy_adoption.v1",
                    "adopted_by": adopted_by,
                    "source_campaign_id": int(source["id"]),
                    "source_request_id": int(item["source_request_id"]),
                    "source_reconstruction_run_id": int(item["source_run_id"]),
                    "adopted_reconstruction_run_id": run_id,
                    "source_output_uri": item["source_output_uri"],
                    "target_source_manifest_sha256": item["source_manifest_sha256"],
                    "source_stage_manifest_sha256": item[
                        "source_source_manifest_sha256"
                    ],
                    "source_identity_mode": (
                        "exact_frozen_manifest"
                        if exact_manifest_identity else
                        "campaign_inventory_and_preprocess_bound_stage_manifest"
                    ),
                    "remote_completion_reconciled": remotely_completed_retired,
                    "reason": "symmetric_trim_changes_export_only",
                }
                conn.execute(
                    """UPDATE stage_requests SET status='SUCCEEDED', blocked_reason=NULL,
                              details=?, result_summary_json=?, completed_at=?, updated_at=?
                       WHERE id=? AND workflow_execution_id IS NULL""",
                    (
                        f"adopted_reconstruction_from_request_{item['source_request_id']}",
                        _canonical_json({"reconstruction_adoption": evidence}),
                        now, now, item["id"],
                    ),
                )
                adopted.append({"sequence": item["sequence_name"], **evidence})
            elif expected_failure:
                conn.execute(
                    """UPDATE stage_requests SET status='FAILED', blocked_reason=NULL,
                              details=?, result_summary_json=?, completed_at=?, updated_at=?
                       WHERE id=? AND workflow_execution_id IS NULL""",
                    (
                        f"carried_expected_outcome_from_request_{item['source_request_id']}",
                        _canonical_json(summary), now, now, item["id"],
                    ),
                )
                carried_failures.append({
                    "sequence": item["sequence_name"],
                    "source_request_id": int(item["source_request_id"]),
                    "failure_category": summary["failure_category"],
                })
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return {
        "source_campaign_id": int(source["id"]),
        "replacement_campaign_id": int(replacement["id"]),
        "adopted_count": len(adopted),
        "carried_expected_failure_count": len(carried_failures),
        "adopted": adopted,
        "carried_expected_failures": carried_failures,
    }


def create_revalidation_successor(
    source_campaign: int | str,
    *,
    name: str,
    pipeline_version: str,
    created_by: str,
    db_path: str = DB_PATH,
) -> dict:
    """Replace a terminal canary campaign without rebuilding its inventory.

    Successful canaries are converted into export-only retries against their
    verified request-scoped work outputs. Failed canaries remain failed, and
    frozen bulk members remain blocked until the successor canary is approved.
    The source inventory and configuration identities are reused byte-for-byte.
    """
    if not name.strip() or not created_by.strip():
        raise ValueError("Successor campaign name and creator are required")
    ensure_version_cached(pipeline_version, db_path=db_path)
    now = _utc_now()
    conn = get_connection(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        column = "id" if isinstance(source_campaign, int) else "name"
        source = conn.execute(
            f"SELECT * FROM processing_campaigns WHERE {column}=?",
            (source_campaign,),
        ).fetchone()
        if source is None:
            raise ValueError(f"Unknown campaign: {source_campaign}")
        if source["campaign_type"] != "LEGACY_REVALIDATION":
            raise ValueError("Only legacy revalidation campaigns have successors")
        if source["status"] not in ("FROZEN", "RUNNING"):
            raise ValueError("Successor source campaign must still be open")
        if source["phase"] != "CANARY":
            raise ValueError("Successor source must still be in its canary phase")
        if pipeline_version == source["pipeline_version"]:
            raise ValueError("Successor must use a different pipeline version")
        if not all(
            source[key]
            for key in (
                "inventory_uri", "inventory_sha256",
                "configuration_uri", "configuration_sha256",
            )
        ):
            raise ValueError("Successor source lacks frozen inventory/configuration")

        requests = conn.execute(
            """SELECT * FROM stage_requests
               WHERE campaign_id=? ORDER BY id""",
            (source["id"],),
        ).fetchall()
        sequence_count = len({row["sequence_id"] for row in requests})
        frozen_count = source["inventory_sequence_count"]
        if frozen_count is not None and sequence_count != frozen_count:
            raise ValueError(
                "Source request membership does not match its frozen inventory"
            )
        frozen_count = sequence_count if frozen_count is None else frozen_count
        remote_active = [
            row for row in requests
            if row["status"] in ("RESERVED", "SUBMITTED", "RUNNING", "UNKNOWN")
        ]
        if remote_active:
            raise ValueError(
                f"Source campaign has {len(remote_active)} active or ambiguous request(s)"
            )
        canaries = [row for row in requests if row["cohort"] == "CANARY"]
        if not canaries:
            raise ValueError("Source campaign has no canary requests")
        if any(
            row["status"] not in ("SUCCEEDED", "FAILED", "CANCELED")
            for row in canaries
        ):
            raise ValueError("Every source canary must be terminal")

        cursor = conn.execute(
            """INSERT INTO processing_campaigns
               (name, campaign_type, dataset, status, phase, pipeline_version,
                inventory_uri, inventory_sha256, inventory_sequence_count,
                configuration_uri, configuration_sha256, output_uri,
                created_by, created_at, updated_at, frozen_at)
               VALUES (?, 'LEGACY_REVALIDATION', ?, 'FROZEN', 'CANARY', ?,
                       ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                name, source["dataset"], pipeline_version,
                source["inventory_uri"], source["inventory_sha256"],
                frozen_count,
                source["configuration_uri"], source["configuration_sha256"],
                source["output_uri"], created_by, now, now, now,
            ),
        )
        successor_id = cursor.lastrowid

        # Release the partial unique active-request index before inserting the
        # successor's bulk intents. Remote or ambiguous requests were rejected
        # above, so only local pending/blocking state is terminalized here.
        conn.execute(
            """UPDATE stage_requests SET status='CANCELED',
               details='superseded_by_revalidation_successor',
               completed_at=COALESCE(completed_at, ?), updated_at=?
               WHERE campaign_id=? AND status IN ('PENDING','BLOCKED')""",
            (now, now, source["id"]),
        )
        conn.execute(
            """UPDATE processing_campaigns SET status='CANCELED',
               phase='COMPLETE', completed_at=?, updated_at=? WHERE id=?""",
            (now, now, source["id"]),
        )

        values = []
        for row in requests:
            parameters = json.loads(row["parameters_json"] or "{}")
            parameters["successor_source_request_id"] = row["id"]
            parameters["successor_source_pipeline_version"] = row["pipeline_version"]
            details = f"successor_of_request_{row['id']}"
            completed_at = None
            if row["cohort"] == "BULK":
                status = "BLOCKED"
                blocked_reason = "WAITING_CANARY_APPROVAL"
            elif row["status"] == "SUCCEEDED":
                if (
                    not row["result_manifest_uri"]
                    or not row["result_manifest_sha256"]
                ):
                    raise ValueError(
                        f"Successful canary request {row['id']} lacks a verified result"
                    )
                parts = str(row["result_manifest_uri"]).rsplit("/", 2)
                if len(parts) != 3 or parts[1] != "candidate_export":
                    raise ValueError(
                        f"Canary request {row['id']} has an unsupported result URI"
                    )
                parameters["retry_work_output_url"] = parts[0]
                status = "PENDING"
                blocked_reason = None
                details += ":export_only_retry"
            elif row["status"] == "FAILED":
                status = "FAILED"
                blocked_reason = None
                completed_at = now
                details += f":carried_failure:{row['details'] or 'unknown'}"
            else:
                status = "PENDING"
                blocked_reason = None
                details += ":full_canary_retry"
            values.append((
                row["sequence_id"], successor_id, row["cohort"],
                row["queue_priority"], row["stage"],
                status, "MIGRATION", pipeline_version, created_by,
                f"successor_of:{source['name']}", _canonical_json(parameters),
                row["source_manifest_json"], row["source_manifest_sha256"],
                blocked_reason, details, now, completed_at, now,
            ))
        conn.executemany(
            """INSERT INTO stage_requests
               (sequence_id, campaign_id, cohort, queue_priority, stage, status, trigger,
                pipeline_version, requested_by, reason, parameters_json,
                source_manifest_json, source_manifest_sha256, blocked_reason,
                details, created_at, completed_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            values,
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return campaign_progress(successor_id, db_path=db_path)


def maybe_blacklist_repeated_failure(
    run_id: int,
    *,
    stage: str | None = None,
    created_by: str = "automatic_failure_circuit_breaker",
    db_path: str = DB_PATH,
) -> bool:
    """Blacklist after the two newest same-stage runs fail identically."""
    conn = get_connection(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        located_stage, table = _run_location(conn, run_id, stage)
        del located_stage
        run = conn.execute(
            f"SELECT sequence_id FROM {table} WHERE id=?", (run_id,)
        ).fetchone()
        recent = conn.execute(
            f"""SELECT status, details FROM {table}
                WHERE sequence_id=? ORDER BY created_at DESC, id DESC LIMIT 2""",
            (run["sequence_id"],),
        ).fetchall()
        if len(recent) != 2 or any(row["status"] != "FAILED" for row in recent):
            conn.rollback()
            return False
        normalized = [normalize_failure_details(row["details"]) for row in recent]
        if not normalized[0] or normalized[0] != normalized[1]:
            conn.rollback()
            return False
        if conn.execute(
            "SELECT 1 FROM blacklisted_sequences WHERE sequence_id=?",
            (run["sequence_id"],),
        ).fetchone():
            conn.rollback()
            return False
        conn.execute(
            """INSERT INTO blacklisted_sequences
               (sequence_id, reason, created_by) VALUES (?, ?, ?)""",
            (run["sequence_id"], normalized[0], created_by),
        )
        conn.commit()
        return True
    finally:
        conn.close()


# -- Status normalization --------------------------------------------------

def _normalize_status(status: str, *, execution: bool = False) -> str:
    aliases = {
        "WAITING_WF": "RUNNING",
        "WAITING_EXPORT": "RUNNING",
        "WAITING_QC": "SUCCEEDED",
        "PASS": "SUCCEEDED",
        "FAIL": "FAILED",
        "COMPLETED": "SUCCEEDED",
    }
    normalized = aliases.get(status, status)
    accepted = _EXECUTION_STATUSES if execution else _RUN_STATUSES
    if normalized not in accepted:
        if normalized.startswith("FAILED"):
            return "FAILED"
        raise ValueError(f"Unsupported status: {status}")
    return normalized


def _legacy_status(status: str) -> str:
    return {
        "SUBMITTING": "WAITING_WF",
        "RUNNING": "WAITING_WF",
        "SUCCEEDED": "PASS",
        "FAILED": "FAIL",
        "CANCELED": "FAIL",
        "UNKNOWN": "UNKNOWN",
        "SKIPPED": "SKIPPED",
    }[status]


def _normalize_trigger(trigger: str) -> str:
    value = trigger.upper()
    aliases = {"AUTOMATIC": "AUTO", "RECOVERY": "MANUAL"}
    value = aliases.get(value, value)
    if value not in ("AUTO", "MANUAL", "MIGRATION"):
        raise ValueError(f"Unsupported trigger: {trigger}")
    return value


# -- Executions and stage runs --------------------------------------------

def create_workflow_execution(
    *,
    workflow_name: str,
    osmo_workflow_id: str | None = None,
    pipeline_type: str | None = None,
    pipeline_version: str | None = None,
    status: str = "SUBMITTING",
    details: str | None = None,
    backend: str = "osmo",
    workflow_spec_path: str | None = None,
    pool: str | None = None,
    created_at: str | None = None,
    updated_at: str | None = None,
    db_path: str = DB_PATH,
) -> int:
    stage = pipeline_type or RECONSTRUCTION_STAGE
    if stage not in EXECUTION_STAGE_NAMES:
        raise ValueError(f"Unsupported stage: {stage}")
    normalized = _normalize_status(status, execution=True)
    conn = get_connection(db_path)
    try:
        cursor_id = _create_workflow_execution(
            conn,
            workflow_name=workflow_name,
            osmo_workflow_id=osmo_workflow_id,
            stage=stage,
            pipeline_version=pipeline_version,
            status=normalized,
            details=details,
            backend=backend,
            workflow_spec_path=workflow_spec_path,
            pool=pool,
            created_at=created_at,
            updated_at=updated_at,
        )
        conn.commit()
        return cursor_id
    finally:
        conn.close()


def reserve_request_with_execution(
    request_id: int,
    *,
    reserved_by: str,
    workflow_name: str,
    pipeline_version: str,
    workflow_spec_path: str,
    pool: str,
    details: str = "campaign_submit_reserved",
    force_blacklist: bool = False,
    db_path: str = DB_PATH,
) -> dict | None:
    """Atomically reserve one request and create/attach its OSMO execution."""
    now = _utc_now()
    conn = get_connection(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        request = conn.execute(
            """SELECT sr.*, b.reason AS blacklist_reason,
                      c.status AS campaign_status, c.phase AS campaign_phase,
                      c.campaign_type
               FROM stage_requests sr
               LEFT JOIN blacklisted_sequences b ON b.sequence_id=sr.sequence_id
               LEFT JOIN processing_campaigns c ON c.id=sr.campaign_id
               WHERE sr.id=?"""
            + (" FOR UPDATE OF sr" if is_postgresql_connection(conn) else ""),
            (request_id,),
        ).fetchone()
        if request is None:
            raise ValueError(f"Unknown stage request: {request_id}")
        if request["status"] not in ("PENDING", "BLOCKED", "RESERVED"):
            conn.rollback()
            return None
        blocked_reason = None
        campaign_bypasses_blacklist = request["campaign_type"] in _CAMPAIGN_TYPES
        if (
            request["blacklist_reason"]
            and not force_blacklist
            and not campaign_bypasses_blacklist
        ):
            blocked_reason = "BLACKLISTED"
        elif request["campaign_id"] and request["campaign_status"] not in ("FROZEN", "RUNNING"):
            blocked_reason = "CAMPAIGN_NOT_ACTIVE"
        elif request["cohort"] == "BULK" and request["campaign_phase"] == "CANARY":
            blocked_reason = "WAITING_CANARY_APPROVAL"
        if blocked_reason:
            conn.execute(
                """UPDATE stage_requests SET status='BLOCKED', blocked_reason=?,
                   details=?, updated_at=? WHERE id=?""",
                (blocked_reason, blocked_reason, now, request_id),
            )
            conn.commit()
            return None
        execution_id = _create_workflow_execution(
            conn, workflow_name=workflow_name, osmo_workflow_id=None,
            stage=RUN_STAGE_NAMES.get(request["stage"], request["stage"]),
            pipeline_version=pipeline_version,
            status="SUBMITTING", details=details, backend="osmo",
            workflow_spec_path=workflow_spec_path, pool=pool,
        )
        conn.execute(
            """UPDATE stage_requests SET workflow_execution_id=?, status='SUBMITTED',
               reserved_by=?, reserved_at=?, submitted_at=?, lease_expires_at=NULL,
               blocked_reason=NULL, details=?, updated_at=?
               WHERE id=?""",
            (execution_id, reserved_by, now, now, details, now, request_id),
        )
        if request["campaign_id"]:
            conn.execute(
                """UPDATE processing_campaigns SET status='RUNNING',
                   started_at=COALESCE(started_at, ?), updated_at=?
                   WHERE id=? AND status='FROZEN'""",
                (now, now, request["campaign_id"]),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return {
        "request": get_stage_request(request_id, db_path=db_path),
        "execution_id": execution_id,
    }


def _create_workflow_execution(
    conn: sqlite3.Connection,
    *,
    workflow_name: str,
    osmo_workflow_id: str | None,
    stage: str,
    pipeline_version: str | None,
    status: str,
    details: str | None,
    backend: str = "osmo",
    workflow_spec_path: str | None = None,
    pool: str | None = None,
    created_at: str | None = None,
    updated_at: str | None = None,
) -> int:
    if osmo_workflow_id:
        row = conn.execute(
            "SELECT id FROM workflow_executions WHERE osmo_workflow_id=?",
            (osmo_workflow_id,),
        ).fetchone()
        if row:
            return row[0]
    row = conn.execute(
        "SELECT id FROM workflow_executions WHERE workflow_name=?", (workflow_name,)
    ).fetchone()
    if row:
        return row[0]
    timestamp = created_at or updated_at
    cursor = conn.execute(
        """INSERT INTO workflow_executions
           (pipeline_stage, pipeline_version, backend, workflow_name,
            osmo_workflow_id, workflow_spec_path, pool, status, details,
            submitted_at, started_at, completed_at, last_refreshed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?,
                   COALESCE(?, strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                   CASE WHEN ? IN ('RUNNING','SUCCEEDED','FAILED','CANCELED')
                        THEN COALESCE(?, strftime('%Y-%m-%dT%H:%M:%fZ','now')) END,
                   CASE WHEN ? IN ('SUCCEEDED','FAILED','CANCELED')
                        THEN COALESCE(?, strftime('%Y-%m-%dT%H:%M:%fZ','now')) END,
                   COALESCE(?, strftime('%Y-%m-%dT%H:%M:%fZ','now')))""",
        (EXECUTION_STAGE_NAMES[stage], pipeline_version, backend, workflow_name, osmo_workflow_id,
         workflow_spec_path, pool, status, details, timestamp,
         status, timestamp, status, updated_at or timestamp,
         updated_at or timestamp),
    )
    return cursor.lastrowid


def _preserve_pool_selection_details(
    existing: str | None, replacement: str,
) -> str:
    """Keep the immutable submission-time pool decision during reconciliation."""
    marker = "pool_selection="
    existing = str(existing or "")
    if marker not in existing or marker in replacement:
        return replacement
    selection = existing[existing.index(marker):]
    return f"{replacement}; {selection}" if replacement else selection


def update_workflow_execution(
    execution_id: int,
    *,
    status: str | None = None,
    details: str | None = None,
    osmo_workflow_id: str | None = None,
    query_payload: dict | None = None,
    db_path: str = DB_PATH,
) -> None:
    updates = ["last_refreshed_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')"]
    params: list = []
    normalized = None
    if status is not None:
        normalized = _normalize_status(status, execution=True)
        updates.append("status=?")
        params.append(normalized)
        if normalized == "RUNNING":
            updates.append(
                "started_at=COALESCE(started_at, "
                "strftime('%Y-%m-%dT%H:%M:%fZ','now'))"
            )
        if normalized in ("SUCCEEDED", "FAILED", "CANCELED"):
            updates.append(
                "completed_at=COALESCE(completed_at, "
                "strftime('%Y-%m-%dT%H:%M:%fZ','now'))"
            )
    if osmo_workflow_id is not None:
        updates.append("osmo_workflow_id=?")
        params.append(osmo_workflow_id)
    if query_payload is not None:
        updates.append("last_query_payload_json=?")
        params.append(json.dumps(query_payload, sort_keys=True))
    conn = get_connection(db_path)
    try:
        existing = conn.execute(
            "SELECT status, details FROM workflow_executions WHERE id=?", (execution_id,)
        ).fetchone()
        if existing is None:
            raise ValueError(f"Unknown workflow execution: {execution_id}")
        if normalized is not None:
            _guard_status_transition(
                "workflow execution", existing["status"], normalized,
                _EXECUTION_TRANSITIONS,
            )
        if details is not None:
            updates.append("details=?")
            params.append(
                _preserve_pool_selection_details(existing["details"], details)
            )
        params.append(execution_id)
        conn.execute(
            f"UPDATE workflow_executions SET {', '.join(updates)} WHERE id=?", params
        )
        conn.commit()
    finally:
        conn.close()


def get_workflow_execution(execution_id: int, *, db_path: str = DB_PATH) -> dict | None:
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM workflow_executions WHERE id=?", (execution_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_workflow_executions(
    *, pipeline_stage: str | None = None,
    status: str | Iterable[str] | None = None,
    db_path: str = DB_PATH,
) -> list[dict]:
    clauses: list[str] = []
    params: list = []
    if pipeline_stage is not None:
        if pipeline_stage not in EXECUTION_STAGE_NAMES.values():
            raise ValueError(f"Unsupported execution stage: {pipeline_stage}")
        clauses.append("pipeline_stage=?")
        params.append(pipeline_stage)
    if status:
        statuses = [status] if isinstance(status, str) else list(status)
        normalized = [_normalize_status(value, execution=True) for value in statuses]
        clauses.append("status IN (" + ",".join("?" for _ in normalized) + ")")
        params.extend(normalized)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    conn = get_connection(db_path)
    try:
        return [dict(row) for row in conn.execute(
            "SELECT * FROM workflow_executions" + where + " ORDER BY submitted_at, id",
            params,
        )]
    finally:
        conn.close()


def _next_run_id(conn) -> int:
    if is_postgresql_connection(conn):
        # Run IDs are globally unique across four stage tables. Serialize only
        # this short allocation point rather than all PostgreSQL writers.
        conn.execute("SELECT pg_advisory_xact_lock(1448542026)")
    union = " UNION ALL ".join(f"SELECT id FROM {table}" for table in RUN_TABLES.values())
    return conn.execute(
        f"SELECT COALESCE(MAX(id), 0) + 1 FROM ({union}) AS stage_run_ids"
    ).fetchone()[0]


def _set_current(
    conn: sqlite3.Connection,
    stage: str,
    sequence_id: int,
    run_id: int,
) -> None:
    table = RUN_TABLES[stage]
    conn.execute(
        f"""UPDATE {table} SET is_current=0,
             superseded_at=COALESCE(
                 superseded_at, strftime('%Y-%m-%dT%H:%M:%fZ','now')),
             updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')
             WHERE sequence_id=? AND is_current=1 AND id!=?""",
        (sequence_id, run_id),
    )
    conn.execute(
        f"UPDATE {table} SET is_current=1, superseded_at=NULL WHERE id=?", (run_id,)
    )
    descendants: tuple[str, ...] = ()
    if stage == PREPROCESS_STAGE:
        descendants = ("reconstruction_runs", "export_runs")
    elif stage == RECONSTRUCTION_STAGE:
        descendants = ("export_runs",)
    for descendant in descendants:
        conn.execute(
            f"""UPDATE {descendant} SET is_current=0,
                 superseded_at=COALESCE(
                     superseded_at, strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                 updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')
                 WHERE sequence_id=? AND is_current=1""",
            (sequence_id,),
        )


def _require_current_successful_run(
    conn: sqlite3.Connection,
    table: str,
    run_id: int | None,
    *,
    expected_sequence_id: int | None = None,
) -> sqlite3.Row:
    if run_id is None:
        raise ValueError(f"A current successful {table} dependency is required")
    row = conn.execute(
        f"SELECT * FROM {table} WHERE id=?", (run_id,)
    ).fetchone()
    if row is None or row["status"] != "SUCCEEDED" or not row["is_current"]:
        raise ValueError(f"{table} run {run_id} is not current and successful")
    if expected_sequence_id is not None and row["sequence_id"] != expected_sequence_id:
        raise ValueError(f"{table} run {run_id} belongs to a different sequence")
    return row


def create_stage_run(
    *,
    sequence_name: str,
    dataset: str,
    stage: str,
    pipeline_version: str | None = None,
    status: str = "SUBMITTING",
    details: str | None = None,
    trigger: str = "automatic",
    requested_by: str | None = None,
    workflow_name: str | None = None,
    osmo_workflow_id: str | None = None,
    backend: str = "osmo",
    workflow_spec_path: str | None = None,
    pool: str | None = None,
    execution_id: int | None = None,
    workflow_task_name: str | None = None,
    auxiliary_task_name: str | None = None,
    calibration_run_id: int | None = None,
    preprocess_run_id: int | None = None,
    reconstruction_run_id: int | None = None,
    qc_review_id: int | None = None,
    authorization_type: str | None = None,
    override_reason: str | None = None,
    source_uri: str | None = None,
    output_uri: str | None = None,
    mesh_uri: str | None = None,
    metadata_uri: str | None = None,
    calibration_setup: str | None = None,
    labeled_bboxes_uri: str | None = None,
    labeled_bboxes_manifest_json: str | None = None,
    labeled_bboxes_sha256: str | None = None,
    bbox_source: str | None = None,
    hitl_item_id: str | None = None,
    set_current: bool = True,
    created_at: str | None = None,
    updated_at: str | None = None,
    legacy_source_table: str | None = None,
    legacy_source_id: int | None = None,
    request_id: int | None = None,
    environment: str = PROD_ENVIRONMENT,
    db_path: str = DB_PATH,
) -> dict:
    del environment  # Environments use separate database files in schema v2.
    if stage not in STAGES:
        raise ValueError(f"Unsupported stage: {stage}")
    normalized_status = _normalize_status(status)

    if stage == EXPORT_STAGE and authorization_type is None:
        authorization_type = "QC" if qc_review_id else "LEGACY"

    normalized_trigger = _normalize_trigger(trigger)
    sequence_kind = "calibration" if stage == CALIBRATION_STAGE else "hoi"
    table = RUN_TABLES[stage]
    timestamp = created_at or updated_at or datetime.now(timezone.utc).isoformat()

    conn = get_connection(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        if execution_id is None and workflow_name and normalized_status != "SKIPPED":
            execution_id = _create_workflow_execution(
                conn,
                workflow_name=workflow_name,
                osmo_workflow_id=osmo_workflow_id,
                stage=stage,
                pipeline_version=pipeline_version,
                status=_normalize_status(status, execution=True),
                details=details,
                backend=backend,
                workflow_spec_path=workflow_spec_path,
                pool=pool,
                created_at=created_at,
                updated_at=updated_at,
            )
        if (
            normalized_trigger != "MIGRATION"
            and normalized_status != "SKIPPED"
            and execution_id is None
        ):
            raise ValueError("Runtime stage runs require a workflow execution")
        sequence_id = _ensure_sequence(
            conn, dataset, sequence_name, sequence_kind=sequence_kind, source_uri=source_uri
        )
        if normalized_trigger != "MIGRATION" and normalized_status != "SKIPPED":
            if stage == PREPROCESS_STAGE:
                calibration = _require_current_successful_run(
                    conn, "calibration_runs", calibration_run_id
                )
                expected_calibration_id = conn.execute(
                    "SELECT calibration_sequence_id FROM sequences WHERE id=?",
                    (sequence_id,),
                ).fetchone()[0]
                if (
                    expected_calibration_id is not None
                    and calibration["sequence_id"] != expected_calibration_id
                ):
                    raise ValueError(
                        f"calibration run {calibration_run_id} does not match the "
                        "sequence calibration reference"
                    )
            elif stage == RECONSTRUCTION_STAGE:
                _require_current_successful_run(
                    conn, "preprocess_runs", preprocess_run_id,
                    expected_sequence_id=sequence_id,
                )
            elif stage == EXPORT_STAGE:
                reconstruction = _require_current_successful_run(
                    conn, "reconstruction_runs", reconstruction_run_id,
                    expected_sequence_id=sequence_id,
                )
                del reconstruction
                if authorization_type == "QC":
                    review = conn.execute(
                        "SELECT reconstruction_run_id, decision FROM qc_reviews WHERE id=?",
                        (qc_review_id,),
                    ).fetchone()
                    if (
                        review is None
                        or review["decision"] != "PASS"
                        or review["reconstruction_run_id"] != reconstruction_run_id
                    ):
                        raise ValueError(
                            "QC export requires a passing review for the selected "
                            "current reconstruction"
                        )
                elif authorization_type == "LEGACY":
                    raise ValueError("LEGACY export authorization is migration-only")
        run_id = _next_run_id(conn)
        request = None
        if request_id is not None:
            request = conn.execute(
                "SELECT * FROM stage_requests WHERE id=?", (request_id,)
            ).fetchone()
            if request is None or request["sequence_id"] != sequence_id:
                raise ValueError("Stage request belongs to a different sequence")
            expected_request_stage = EXECUTION_STAGE_NAMES[stage]
            if not (
                request["stage"] == expected_request_stage
                or (stage == EXPORT_STAGE and request["stage"] == "revalidation")
            ):
                raise ValueError(
                    f"Stage request {request_id} is for {request['stage']}, not "
                    f"{expected_request_stage}"
                )
            if request["workflow_execution_id"] not in (None, execution_id):
                raise ValueError("Stage request references a different execution")
        if authorization_type == "REVALIDATION":
            if request is None or request["stage"] != "revalidation":
                raise ValueError("REVALIDATION export requires a revalidation request")
            if request["status"] != "SUCCEEDED":
                raise ValueError("REVALIDATION request must be successful before export commit")
            if not request["result_manifest_uri"] or not request["result_manifest_sha256"]:
                raise ValueError("REVALIDATION request requires a verified result manifest")

        common_columns = (
            "id, sequence_id, workflow_execution_id, pipeline_version, status, details, "
            "trigger, requested_by, output_uri, is_current, legacy_source_table, "
            "legacy_source_id, created_at, started_at, completed_at, updated_at, request_id"
        )
        started_at = (
            timestamp
            if normalized_status in ("RUNNING", "SUCCEEDED", "FAILED", "CANCELED")
            else None
        )
        completed_at = (
            timestamp
            if normalized_status in ("SUCCEEDED", "FAILED", "CANCELED", "SKIPPED")
            else None
        )
        common_values = [
            run_id, sequence_id, execution_id, pipeline_version, normalized_status,
            details, normalized_trigger, requested_by, output_uri, 0,
            legacy_source_table, legacy_source_id, timestamp, started_at,
            completed_at,
            updated_at or timestamp,
            request_id,
        ]
        if stage == CALIBRATION_STAGE:
            columns = common_columns + ", calibration_setup, source_uri"
            values = common_values + [calibration_setup, source_uri]
        elif stage == PREPROCESS_STAGE:
            columns = common_columns + ", calibration_run_id, source_uri, mesh_uri, metadata_uri"
            values = common_values + [calibration_run_id, source_uri, mesh_uri, metadata_uri]
        elif stage == RECONSTRUCTION_STAGE:
            columns = common_columns + (
                ", preprocess_run_id, labeled_bboxes_uri, labeled_bboxes_manifest_json, "
                "labeled_bboxes_sha256, bbox_source, hitl_item_id"
            )
            values = common_values + [
                preprocess_run_id, labeled_bboxes_uri, labeled_bboxes_manifest_json,
                labeled_bboxes_sha256, bbox_source, hitl_item_id,
            ]
        else:
            columns = common_columns + (
                ", reconstruction_run_id, qc_review_id, authorization_type, "
                "override_reason, export_task_name, copy_task_name, source_uri"
            )
            values = common_values + [
                reconstruction_run_id, qc_review_id, authorization_type,
                override_reason, workflow_task_name, auxiliary_task_name, source_uri,
            ]
        placeholders = ", ".join("?" for _ in values)
        conn.execute(f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", values)
        if set_current:
            _set_current(conn, stage, sequence_id, run_id)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return get_stage_run(run_id, stage=stage, db_path=db_path)


def commit_revalidation_export(
    request_id: int,
    *,
    reconstruction_run_id: int,
    result_manifest_uri: str,
    result_manifest_sha256: str,
    result_summary: object,
    source_uri: str,
    output_uri: str,
    details: str = "revalidation_committed",
    query_payload: object | None = None,
    recover_publication_failure: bool = False,
    db_path: str = DB_PATH,
) -> dict:
    """Atomically commit a verified revalidation request and its export run."""
    if not result_manifest_uri or not result_manifest_sha256:
        raise ValueError("A verified result manifest URI and SHA-256 are required")
    now = _utc_now()
    conn = get_connection(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        request = conn.execute(
            """SELECT sr.*, s.dataset, s.sequence_name, c.output_uri AS campaign_output_uri
               FROM stage_requests sr
               JOIN sequences s ON s.id=sr.sequence_id
               JOIN processing_campaigns c ON c.id=sr.campaign_id
               WHERE sr.id=?""",
            (request_id,),
        ).fetchone()
        if request is None or request["stage"] != "revalidation":
            raise ValueError("REVALIDATION export requires a campaign revalidation request")
        existing = conn.execute(
            "SELECT id FROM export_runs WHERE request_id=?", (request_id,)
        ).fetchone()
        if existing:
            conn.commit()
            return get_stage_run(existing["id"], stage=EXPORT_STAGE, db_path=db_path)
        recoverable_failed_request = (
            recover_publication_failure
            and request["status"] == "FAILED"
            and str(request["details"] or "").startswith("export_run_commit_failed:")
        )
        if (
            request["status"] not in ("SUBMITTED", "RUNNING", "UNKNOWN", "SUCCEEDED")
            and not recoverable_failed_request
        ):
            raise ValueError(
                f"Revalidation request {request_id} cannot commit from {request['status']}"
            )
        if request["workflow_execution_id"] is None:
            raise ValueError("Revalidation request has no workflow execution")
        if request["pipeline_version"] is None:
            raise ValueError("Revalidation request has no pipeline version")
        destination = output_uri.rstrip("/") + "/"
        campaign_destination = request["campaign_output_uri"].rstrip("/") + "/"
        if not destination.startswith(campaign_destination):
            raise ValueError("Export destination is outside the frozen campaign output prefix")
        _require_current_successful_run(
            conn, "reconstruction_runs", reconstruction_run_id,
            expected_sequence_id=request["sequence_id"],
        )

        run_id = _next_run_id(conn)
        conn.execute(
            """INSERT INTO export_runs
               (id, sequence_id, workflow_execution_id, pipeline_version, status,
                details, trigger, requested_by, output_uri, is_current,
                created_at, started_at, completed_at, updated_at, request_id,
                reconstruction_run_id, authorization_type, source_uri)
               VALUES (?, ?, ?, ?, 'SUCCEEDED', ?, 'MIGRATION', ?, ?, 0,
                       ?, ?, ?, ?, ?, ?, 'REVALIDATION', ?)""",
            (
                run_id, request["sequence_id"], request["workflow_execution_id"],
                request["pipeline_version"], details, request["requested_by"],
                output_uri, now, now, now, now, request_id,
                reconstruction_run_id, source_uri,
            ),
        )
        _set_current(conn, EXPORT_STAGE, request["sequence_id"], run_id)
        conn.execute(
            """UPDATE stage_requests
               SET status='SUCCEEDED', details=?, result_manifest_uri=?,
                   result_manifest_sha256=?, result_summary_json=?,
                   completed_at=?, updated_at=?, blocked_reason=NULL
               WHERE id=?""",
            (
                details, result_manifest_uri, result_manifest_sha256,
                _canonical_json(result_summary), now, now, request_id,
            ),
        )
        conn.execute(
            """UPDATE workflow_executions
               SET status='SUCCEEDED', details=?, last_query_payload_json=?,
                   completed_at=COALESCE(completed_at, ?), last_refreshed_at=?
               WHERE id=?""",
            (
                details,
                _canonical_json(query_payload) if query_payload is not None else None,
                now, now, request["workflow_execution_id"],
            ),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return get_stage_run(run_id, stage=EXPORT_STAGE, db_path=db_path)


def reconcile_failed_export_success(
    request_id: int,
    *,
    result_manifest_uri: str,
    result_manifest_sha256: str,
    result_summary: object,
    details: str,
    db_path: str = DB_PATH,
) -> dict:
    """Reconcile a failed export request from a verified durable commit."""

    now = _utc_now()
    conn = get_connection(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        request = conn.execute(
            """SELECT sr.*, pc.output_uri AS campaign_output_uri
               FROM stage_requests sr
               JOIN processing_campaigns pc ON pc.id=sr.campaign_id
               WHERE sr.id=?"""
            + (" FOR UPDATE OF sr" if is_postgresql_connection(conn) else ""),
            (request_id,),
        ).fetchone()
        run = conn.execute(
            "SELECT * FROM export_runs WHERE request_id=?"
            + (" FOR UPDATE" if is_postgresql_connection(conn) else ""),
            (request_id,),
        ).fetchone()
        if request is None or request["stage"] != "export" or run is None:
            raise ValueError("Export reconciliation requires an export request and run")
        if request["status"] == "SUCCEEDED" and run["status"] == "SUCCEEDED":
            conn.rollback()
            return get_stage_run(run["id"], stage=EXPORT_STAGE, db_path=db_path)
        if request["status"] != "FAILED" or run["status"] != "FAILED":
            raise ValueError("Only matching failed export lineage can be reconciled")
        expected_prefix = str(request["campaign_output_uri"]).rstrip("/") + "/"
        if not result_manifest_uri.startswith(expected_prefix):
            raise ValueError("Reconciled export is outside the campaign output prefix")
        conn.execute(
            """UPDATE export_runs
               SET status='SUCCEEDED', details=?, completed_at=?, updated_at=?
               WHERE id=?""",
            (details, now, now, run["id"]),
        )
        _set_current(conn, EXPORT_STAGE, request["sequence_id"], run["id"])
        conn.execute(
            """UPDATE stage_requests
               SET status='SUCCEEDED', details=?, result_manifest_uri=?,
                   result_manifest_sha256=?, result_summary_json=?,
                   completed_at=?, updated_at=?, blocked_reason=NULL
               WHERE id=?""",
            (
                details, result_manifest_uri, result_manifest_sha256,
                _canonical_json(result_summary), now, now, request_id,
            ),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return get_stage_run(run["id"], stage=EXPORT_STAGE, db_path=db_path)


def reopen_failed_export_for_candidate_cleanup(
    request_id: int,
    *,
    details: str,
    db_path: str = DB_PATH,
) -> dict:
    """Resume cleanup after a failed export was durably promoted."""

    if not details.startswith("candidate_cleanup_pending:"):
        raise ValueError("Export cleanup recovery requires cleanup-pending details")
    now = _utc_now()
    conn = get_connection(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        request = conn.execute(
            "SELECT * FROM stage_requests WHERE id=?"
            + (" FOR UPDATE" if is_postgresql_connection(conn) else ""),
            (request_id,),
        ).fetchone()
        run = conn.execute(
            "SELECT * FROM export_runs WHERE request_id=?"
            + (" FOR UPDATE" if is_postgresql_connection(conn) else ""),
            (request_id,),
        ).fetchone()
        if request is None or request["stage"] != "export" or run is None:
            raise ValueError("Export cleanup recovery requires an export request and run")
        if request["status"] == "RUNNING" and run["status"] == "RUNNING":
            conn.rollback()
            return get_stage_run(run["id"], stage=EXPORT_STAGE, db_path=db_path)
        if request["status"] != "FAILED" or run["status"] != "FAILED":
            raise ValueError("Only matching failed export lineage can resume cleanup")
        conn.execute(
            """UPDATE export_runs
               SET status='RUNNING', details=?, completed_at=NULL, updated_at=?
               WHERE id=?""",
            (details, now, run["id"]),
        )
        conn.execute(
            """UPDATE stage_requests
               SET status='RUNNING', details=?, completed_at=NULL, updated_at=?,
                   blocked_reason=NULL
               WHERE id=?""",
            (details, now, request_id),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return get_stage_run(run["id"], stage=EXPORT_STAGE, db_path=db_path)


def _run_location(
    conn: sqlite3.Connection, run_id: int, stage: str | None = None
) -> tuple[str, str]:
    if stage:
        table = RUN_TABLES[stage]
        if conn.execute(f"SELECT 1 FROM {table} WHERE id=?", (run_id,)).fetchone():
            return stage, table
        raise KeyError(run_id)
    matches = [
        (candidate_stage, table)
        for candidate_stage, table in RUN_TABLES.items()
        if conn.execute(f"SELECT 1 FROM {table} WHERE id=?", (run_id,)).fetchone()
    ]
    if len(matches) != 1:
        raise KeyError(run_id) if not matches else RuntimeError(f"Duplicate global run id {run_id}")
    return matches[0]


def _latest_qc_join() -> str:
    return """LEFT JOIN qc_reviews q ON q.id=(
        SELECT q2.id FROM qc_reviews q2 WHERE q2.reconstruction_run_id=r.id
        ORDER BY q2.reviewed_at DESC, q2.id DESC LIMIT 1)"""


def _select_for_stage(stage: str) -> str:
    table = RUN_TABLES[stage]
    qc_join = (
        _latest_qc_join()
        if stage == RECONSTRUCTION_STAGE
        else "LEFT JOIN qc_reviews q ON FALSE"
    )
    if stage == PREPROCESS_STAGE:
        upstream = "r.calibration_run_id"
    elif stage == RECONSTRUCTION_STAGE:
        upstream = "r.preprocess_run_id"
    elif stage == EXPORT_STAGE:
        upstream = "r.reconstruction_run_id"
    else:
        upstream = "NULL"
    task = "r.export_task_name" if stage == EXPORT_STAGE else "NULL"
    auxiliary = "r.copy_task_name" if stage == EXPORT_STAGE else "NULL"
    return f"""
        SELECT r.*, r.id AS stage_run_id, s.dataset, s.sequence_name,
               s.sequence_kind AS target_kind, e.workflow_name, e.osmo_workflow_id,
               e.status AS remote_run_status, e.id AS workflow_execution_id,
               e.pool AS pool, e.details AS execution_details,
               e.submitted_at AS execution_submitted_at,
               sr.campaign_id AS request_campaign_id,
               sr.cohort AS request_cohort,
               sr.queue_priority AS request_queue_priority,
               sr.created_at AS request_created_at,
               sr.parameters_json AS request_parameters_json,
               c.name AS request_campaign_name,
               c.campaign_type AS request_campaign_type,
               q.decision AS qc_status, q.details AS qc_details,
               {upstream} AS upstream_run_id,
               {task} AS workflow_task_name,
               {auxiliary} AS auxiliary_task_name
        FROM {table} r
        JOIN sequences s ON s.id=r.sequence_id
        LEFT JOIN workflow_executions e ON e.id=r.workflow_execution_id
        LEFT JOIN stage_requests sr ON sr.id=r.request_id
        LEFT JOIN processing_campaigns c ON c.id=sr.campaign_id
        {qc_join}
    """


def _flatten(row: sqlite3.Row | None, stage: str) -> dict | None:
    if row is None:
        return None
    result = dict(row)
    raw_status = result["status"]
    result["run_status"] = raw_status
    result["execution_status"] = _legacy_status(raw_status)
    result["pipeline_type"] = stage
    result["stage"] = stage
    result["execution_id"] = result.get("workflow_execution_id")
    result["remote_status"] = (
        _legacy_status(result["remote_run_status"])
        if result.get("remote_run_status") else None
    )
    upstream = result.pop("upstream_run_id", None)
    result["upstream_run_ids"] = [upstream] if upstream is not None else []
    public_status = _legacy_status(raw_status)
    if stage == RECONSTRUCTION_STAGE and raw_status == "SUCCEEDED":
        if result.get("qc_status") is None:
            public_status = "WAITING_QC"
        elif result["qc_status"] == "FAIL":
            public_status = "FAIL"
        else:
            public_status = "PASS"
    result["status"] = public_status
    result["osmo_export_workflow_id"] = (
        result.get("osmo_workflow_id") if stage == EXPORT_STAGE else None
    )
    result["environment"] = PROD_ENVIRONMENT
    result["attempt"] = result["id"]
    result["run_name"] = result.get("workflow_name") or (
        f"{stage}:{result['dataset']}:{result['sequence_name']}:{result['id']}"
    )
    return result


def get_stage_run(
    run_id: int, *, stage: str | None = None, db_path: str = DB_PATH
) -> dict | None:
    conn = get_connection(db_path)
    try:
        try:
            located_stage, _ = _run_location(conn, run_id, stage)
        except KeyError:
            return None
        row = conn.execute(_select_for_stage(located_stage) + " WHERE r.id=?", (run_id,)).fetchone()
        return _flatten(row, located_stage)
    finally:
        conn.close()


def get_stage_run_by_request(
    request_id: int, *, stage: str, db_path: str = DB_PATH,
) -> dict | None:
    if stage not in STAGES:
        raise ValueError(f"Unsupported stage: {stage}")
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            f"SELECT id FROM {RUN_TABLES[stage]} WHERE request_id=?", (request_id,)
        ).fetchone()
    finally:
        conn.close()
    return get_stage_run(row["id"], stage=stage, db_path=db_path) if row else None


def update_stage_run(
    run_id: int,
    status: str | None = None,
    details: str | None = None,
    *,
    stage: str | None = None,
    db_path: str = DB_PATH,
) -> None:
    conn = get_connection(db_path)
    try:
        located_stage, table = _run_location(conn, run_id, stage)
        del located_stage
        updates = ["updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')"]
        params: list = []
        normalized = None
        if status is not None:
            normalized = _normalize_status(status)
            current = conn.execute(
                f"SELECT status FROM {table} WHERE id=?", (run_id,)
            ).fetchone()
            if current is None:
                raise ValueError(f"Unknown stage run: {run_id}")
            _guard_status_transition(
                "stage run", current["status"], normalized,
                _RUN_TRANSITIONS,
            )
            updates.append("status=?")
            params.append(normalized)
            if normalized == "RUNNING":
                updates.append(
                    "started_at=COALESCE(started_at, "
                    "strftime('%Y-%m-%dT%H:%M:%fZ','now'))"
                )
            if normalized in ("SUCCEEDED", "FAILED", "CANCELED", "SKIPPED"):
                updates.append(
                    "completed_at=COALESCE(completed_at, "
                    "strftime('%Y-%m-%dT%H:%M:%fZ','now'))"
                )
        if details is not None:
            updates.append("details=?")
            params.append(details)
        params.append(run_id)
        conn.execute(f"UPDATE {table} SET {', '.join(updates)} WHERE id=?", params)
        conn.commit()
    finally:
        conn.close()


def apply_execution_observation(
    execution_id: int,
    *,
    execution_status: str,
    details: str | None = None,
    query_payload: object | None = None,
    osmo_workflow_id: str | None = None,
    run_outcomes: Iterable[dict] = (),
    request_outcomes: Iterable[dict] = (),
    db_path: str = DB_PATH,
) -> None:
    """Atomically apply one backend observation to its execution and children.

    A run outcome automatically updates its linked request unless an explicit
    ``request_status`` of ``None`` is supplied. Revalidation has no stage run,
    so its request result is supplied through ``request_outcomes``.
    """
    normalized_execution = _normalize_status(execution_status, execution=True)
    run_outcomes = [dict(item) for item in run_outcomes]
    explicit_requests = [dict(item) for item in request_outcomes]
    now = _utc_now()
    conn = get_connection(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        execution = conn.execute(
            "SELECT * FROM workflow_executions WHERE id=?", (execution_id,)
        ).fetchone()
        if execution is None:
            raise ValueError(f"Unknown workflow execution: {execution_id}")
        _guard_status_transition(
            "workflow execution", execution["status"], normalized_execution,
            _EXECUTION_TRANSITIONS,
        )
        execution_updates = ["status=?", "last_refreshed_at=?"]
        execution_params: list = [normalized_execution, now]
        if details is not None:
            execution_updates.append("details=?")
            execution_params.append(
                _preserve_pool_selection_details(execution["details"], details)
            )
        if query_payload is not None:
            execution_updates.append("last_query_payload_json=?")
            execution_params.append(_canonical_json(query_payload))
        if osmo_workflow_id is not None:
            execution_updates.append("osmo_workflow_id=?")
            execution_params.append(osmo_workflow_id)
        if normalized_execution == "RUNNING":
            execution_updates.append("started_at=COALESCE(started_at, ?)")
            execution_params.append(now)
        if normalized_execution in _TERMINAL_EXECUTION_STATUSES:
            execution_updates.append("completed_at=COALESCE(completed_at, ?)")
            execution_params.append(now)
        execution_params.append(execution_id)
        conn.execute(
            f"UPDATE workflow_executions SET {', '.join(execution_updates)} WHERE id=?",
            execution_params,
        )

        linked_request_outcomes: list[dict] = []
        for outcome in run_outcomes:
            run_id = int(outcome["run_id"])
            located_stage, table = _run_location(conn, run_id, outcome.get("stage"))
            run = conn.execute(
                f"SELECT * FROM {table} WHERE id=?", (run_id,)
            ).fetchone()
            if run["workflow_execution_id"] != execution_id:
                raise ValueError(
                    f"{located_stage} run {run_id} does not belong to execution {execution_id}"
                )
            run_status = _normalize_status(outcome["status"])
            _guard_status_transition(
                "stage run", run["status"], run_status, _RUN_TRANSITIONS,
            )
            run_details = outcome.get("details", details)
            run_updates = ["status=?", "updated_at=?"]
            run_params: list = [run_status, now]
            if run_details is not None:
                run_updates.append("details=?")
                run_params.append(run_details)
            if run_status == "RUNNING":
                run_updates.append("started_at=COALESCE(started_at, ?)")
                run_params.append(now)
            if run_status in _TERMINAL_RUN_STATUSES:
                run_updates.append("completed_at=COALESCE(completed_at, ?)")
                run_params.append(now)
            run_params.append(run_id)
            conn.execute(
                f"UPDATE {table} SET {', '.join(run_updates)} WHERE id=?", run_params,
            )
            if outcome.get("set_current"):
                _set_current(conn, located_stage, run["sequence_id"], run_id)
            if run["request_id"] is not None and outcome.get("request_status", True) is not None:
                request_status = outcome.get("request_status")
                if request_status is True or request_status is None:
                    request_status = run_status
                if request_status == "SKIPPED":
                    request_status = "CANCELED"
                linked_request_outcomes.append({
                    "request_id": run["request_id"], "status": request_status,
                    "details": run_details,
                })

        by_request: dict[int, dict] = {
            int(item["request_id"]): item
            for item in (*linked_request_outcomes, *explicit_requests)
        }
        for request_id, outcome in by_request.items():
            request = conn.execute(
                "SELECT * FROM stage_requests WHERE id=?", (request_id,)
            ).fetchone()
            if request is None:
                raise ValueError(f"Unknown stage request: {request_id}")
            if request["workflow_execution_id"] != execution_id:
                raise ValueError(
                    f"Stage request {request_id} does not belong to execution {execution_id}"
                )
            request_status = str(outcome["status"])
            if request_status not in _REQUEST_STATUSES:
                raise ValueError(f"Unsupported request status: {request_status}")
            _guard_status_transition(
                "stage request", request["status"], request_status,
                _REQUEST_TRANSITIONS,
            )
            manifest_uri = outcome.get("result_manifest_uri")
            manifest_sha = outcome.get("result_manifest_sha256")
            if request_status == "SUCCEEDED" and request["stage"] == "revalidation":
                manifest_uri = manifest_uri or request["result_manifest_uri"]
                manifest_sha = manifest_sha or request["result_manifest_sha256"]
                if not manifest_uri or not manifest_sha:
                    raise ValueError(
                        "Successful revalidation requests require a verified result manifest"
                    )
            request_updates = ["status=?", "updated_at=?"]
            request_params: list = [request_status, now]
            request_details = outcome.get("details", details)
            if request_details is not None:
                request_updates.append("details=?")
                request_params.append(request_details)
            if request_status != "BLOCKED":
                request_updates.append("blocked_reason=NULL")
            if request_status in _TERMINAL_REQUEST_STATUSES:
                request_updates.append("completed_at=COALESCE(completed_at, ?)")
                request_params.append(now)
            if manifest_uri is not None:
                request_updates.append("result_manifest_uri=?")
                request_params.append(manifest_uri)
            if manifest_sha is not None:
                request_updates.append("result_manifest_sha256=?")
                request_params.append(manifest_sha)
            if "result_summary" in outcome:
                request_updates.append("result_summary_json=?")
                request_params.append(_canonical_json(outcome["result_summary"]))
            request_params.append(request_id)
            conn.execute(
                f"UPDATE stage_requests SET {', '.join(request_updates)} WHERE id=?",
                request_params,
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def set_campaign_execution_retry_state(
    execution_id: int,
    *,
    status: str,
    details: str,
    query_payload: object,
    osmo_workflow_id: str | None = None,
    db_path: str = DB_PATH,
) -> None:
    """Move one campaign execution and its unfinished children into retry state.

    This is the only path allowed to reopen a terminal campaign execution.
    OSMO restarts create a new workflow ID. The execution row follows that
    replacement while the request, execution, and stage-run lineage remains
    unchanged.
    """

    if status not in ("RUNNING", "UNKNOWN"):
        raise ValueError("Campaign retry state must be RUNNING or UNKNOWN")
    now = _utc_now()
    conn = get_connection(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        execution = conn.execute(
            "SELECT * FROM workflow_executions WHERE id=?"
            + (" FOR UPDATE" if is_postgresql_connection(conn) else ""),
            (execution_id,),
        ).fetchone()
        if execution is None:
            raise ValueError(f"Unknown workflow execution: {execution_id}")
        requests = conn.execute(
            """SELECT sr.id, sr.status, sr.campaign_id
               FROM stage_requests sr
               WHERE sr.workflow_execution_id=?""",
            (execution_id,),
        ).fetchall()
        if not requests or any(row["campaign_id"] is None for row in requests):
            raise ValueError(
                "Infrastructure retry is limited to campaign-owned executions"
            )

        execution_updates = [
            "status=?", "details=?", "last_query_payload_json=?",
            "completed_at=NULL", "last_refreshed_at=?",
            """started_at=CASE WHEN ?='RUNNING'
                               THEN COALESCE(started_at, ?)
                               ELSE started_at END""",
        ]
        execution_params: list = [
            status, details, _canonical_json(query_payload), now, status, now,
        ]
        if osmo_workflow_id is not None:
            execution_updates.append("osmo_workflow_id=?")
            execution_params.append(osmo_workflow_id)
        execution_params.append(execution_id)
        conn.execute(
            f"""UPDATE workflow_executions
                SET {', '.join(execution_updates)}
                WHERE id=?""",
            execution_params,
        )
        conn.execute(
            """UPDATE stage_requests
               SET status=?, details=?, blocked_reason=NULL,
                   completed_at=NULL, updated_at=?
               WHERE workflow_execution_id=?
                 AND status NOT IN ('SUCCEEDED','CANCELED')""",
            (status, details, now, execution_id),
        )
        for table in RUN_TABLES.values():
            conn.execute(
                f"""UPDATE {table}
                    SET status=?, details=?, completed_at=NULL, updated_at=?,
                        started_at=CASE WHEN ?='RUNNING'
                                        THEN COALESCE(started_at, ?)
                                        ELSE started_at END
                    WHERE workflow_execution_id=?
                      AND status NOT IN ('SUCCEEDED','CANCELED','SKIPPED')""",
                (status, details, now, status, now, execution_id),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def replace_campaign_infrastructure_attempt(
    execution_id: int,
    *,
    details: str,
    query_payload: object,
    max_attempts: int,
    backoff_seconds: Iterable[int] = (300, 900, 1800),
    db_path: str = DB_PATH,
) -> list[dict] | None:
    """Terminalize a failed attempt and create fresh request intent.

    Returns the replacement requests, an empty list when an export batch was
    released for the export scheduler to rebuild, or ``None`` when the retry
    limit is exhausted. A replacement request receives a new ID, workflow, and
    work prefix; the failed request, run, and execution remain immutable
    history.
    """
    now = _utc_now()
    backoff_seconds = tuple(int(value) for value in backoff_seconds)
    if len(backoff_seconds) < max_attempts or any(value < 0 for value in backoff_seconds):
        raise ValueError("Infrastructure retry backoff must cover every attempt")
    conn = get_connection(db_path)
    replacement_ids: list[int] = []
    try:
        conn.execute("BEGIN IMMEDIATE")
        execution = conn.execute(
            "SELECT * FROM workflow_executions WHERE id=?"
            + (" FOR UPDATE" if is_postgresql_connection(conn) else ""),
            (execution_id,),
        ).fetchone()
        if execution is None:
            raise ValueError(f"Unknown workflow execution: {execution_id}")
        requests = conn.execute(
            """SELECT sr.* FROM stage_requests sr
               WHERE sr.workflow_execution_id=? ORDER BY sr.id"""
            + (" FOR UPDATE" if is_postgresql_connection(conn) else ""),
            (execution_id,),
        ).fetchall()
        if not requests or any(row["campaign_id"] is None for row in requests):
            raise ValueError(
                "Infrastructure replacement is limited to campaign requests"
            )
        campaign_versions: dict[int, str] = {}
        for campaign_id in {int(row["campaign_id"]) for row in requests}:
            campaign = conn.execute(
                "SELECT status, pipeline_version FROM processing_campaigns WHERE id=?",
                (campaign_id,),
            ).fetchone()
            if campaign is None or campaign["status"] not in ("FROZEN", "RUNNING"):
                raise ValueError(
                    "Infrastructure replacement requires an active campaign"
                )
            campaign_versions[campaign_id] = campaign["pipeline_version"]

        existing_replacements: list[int] = []
        for request in requests:
            replacement = conn.execute(
                """SELECT id FROM stage_requests
                   WHERE campaign_id=? AND sequence_id=? AND stage=? AND reason=?
                   ORDER BY id DESC LIMIT 1""",
                (
                    request["campaign_id"], request["sequence_id"], request["stage"],
                    f"infrastructure_retry_of_request_{request['id']}",
                ),
            ).fetchone()
            if replacement is not None:
                existing_replacements.append(replacement["id"])
        if existing_replacements:
            if len(existing_replacements) != len(requests):
                raise RuntimeError(
                    "Infrastructure replacement exists for only part of an execution"
                )
            conn.rollback()
            return [
                get_stage_request(request_id, db_path=db_path)
                for request_id in existing_replacements
            ]

        attempts: list[int] = []
        for request in requests:
            parameters = json.loads(request["parameters_json"] or "{}")
            attempts.append(int(parameters.get("infrastructure_retry_attempt", 0)) + 1)
        if any(attempt > max_attempts for attempt in attempts):
            conn.rollback()
            return None

        conn.execute(
            """UPDATE workflow_executions
               SET status='FAILED', details=?, last_query_payload_json=?,
                   completed_at=COALESCE(completed_at, ?), last_refreshed_at=?
               WHERE id=?""",
            (details, _canonical_json(query_payload), now, now, execution_id),
        )
        for table in RUN_TABLES.values():
            conn.execute(
                f"""UPDATE {table}
                    SET status='FAILED', details=?,
                        completed_at=COALESCE(completed_at, ?), updated_at=?
                    WHERE workflow_execution_id=?
                      AND status NOT IN ('SUCCEEDED','CANCELED','SKIPPED')""",
                (details, now, now, execution_id),
            )
        conn.execute(
            """UPDATE stage_requests
               SET status='FAILED', details=?, blocked_reason=NULL,
                   completed_at=COALESCE(completed_at, ?), updated_at=?
               WHERE workflow_execution_id=?
                 AND status NOT IN ('SUCCEEDED','CANCELED')""",
            (details, now, now, execution_id),
        )

        for request, attempt in zip(requests, attempts):
            # Export batches are reconstructed from current successful
            # reconstruction/QC state by the export scheduler. Creating a
            # detached export request here would bypass that preflight.
            if request["stage"] == "export":
                continue
            parameters = json.loads(request["parameters_json"] or "{}")
            parameters.update({
                "infrastructure_retry_attempt": attempt,
                "prior_request_id": request["id"],
                "prior_execution_id": execution_id,
                "retry_not_before": (
                    datetime.fromisoformat(now).astimezone(timezone.utc)
                    + timedelta(seconds=backoff_seconds[attempt - 1])
                ).isoformat(),
                "retry_backoff_seconds": backoff_seconds[attempt - 1],
            })
            if request["stage"] == "revalidation":
                parameters["work_output_layout"] = "request_scoped_v1"
            cursor = conn.execute(
                """INSERT INTO stage_requests
                   (sequence_id, campaign_id, cohort, queue_priority, stage, status, trigger,
                    pipeline_version, requested_by, reason, parameters_json,
                    source_manifest_json, source_manifest_sha256, details,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, 'PENDING', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    request["sequence_id"], request["campaign_id"], request["cohort"],
                    request["queue_priority"], request["stage"], request["trigger"],
                    campaign_versions[int(request["campaign_id"])],
                    request["requested_by"],
                    f"infrastructure_retry_of_request_{request['id']}",
                    _canonical_json(parameters), request["source_manifest_json"],
                    request["source_manifest_sha256"],
                    f"infrastructure_retry_attempt={attempt}", now, now,
                ),
            )
            replacement_ids.append(cursor.lastrowid)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return [
        get_stage_request(request_id, db_path=db_path)
        for request_id in replacement_ids
    ]


def replace_campaign_request_infrastructure_attempt(
    request_id: int,
    execution_id: int,
    *,
    details: str,
    query_payload: object,
    failure_evidence: object | None = None,
    max_attempts: int,
    backoff_seconds: Iterable[int] = (300, 900, 1800),
    db_path: str = DB_PATH,
) -> dict | None:
    """Replace one failed request from a shared campaign execution.

    Batched export executions contain independent per-sequence requests.  A
    failure in one task must not terminalize or replace its siblings, so this
    helper locks and updates only the selected request and its run.  ``None``
    means the request exhausted its retry budget.
    """

    now = _utc_now()
    backoff_seconds = tuple(int(value) for value in backoff_seconds)
    if len(backoff_seconds) < max_attempts or any(value < 0 for value in backoff_seconds):
        raise ValueError("Infrastructure retry backoff must cover every attempt")
    conn = get_connection(db_path)
    replacement_id: int | None = None
    existing_id: int | None = None
    try:
        conn.execute("BEGIN IMMEDIATE")
        request = conn.execute(
            """SELECT sr.*, pc.status AS campaign_status,
                      pc.pipeline_version AS campaign_pipeline_version
               FROM stage_requests sr
               JOIN processing_campaigns pc ON pc.id=sr.campaign_id
               WHERE sr.id=?"""
            + (" FOR UPDATE OF sr" if is_postgresql_connection(conn) else ""),
            (request_id,),
        ).fetchone()
        if request is None or request["campaign_id"] is None:
            raise ValueError("Infrastructure replacement requires a campaign request")
        if int(request["workflow_execution_id"] or -1) != int(execution_id):
            raise ValueError("Request does not belong to the selected execution")
        if request["campaign_status"] not in ("FROZEN", "RUNNING"):
            raise ValueError("Infrastructure replacement requires an active campaign")

        reason = f"infrastructure_retry_of_request_{request_id}"
        existing = conn.execute(
            """SELECT id FROM stage_requests
               WHERE campaign_id=? AND sequence_id=? AND stage=? AND reason=?
               ORDER BY id DESC LIMIT 1""",
            (
                request["campaign_id"], request["sequence_id"],
                request["stage"], reason,
            ),
        ).fetchone()
        if existing is not None:
            existing_id = int(existing["id"])
            conn.rollback()
        else:
            parameters = json.loads(request["parameters_json"] or "{}")
            attempt = int(parameters.get("infrastructure_retry_attempt", 0)) + 1
            if attempt > max_attempts:
                conn.rollback()
                return None

            active = conn.execute(
                """SELECT id FROM stage_requests
                   WHERE campaign_id=? AND sequence_id=? AND stage=?
                     AND id<>?
                     AND status IN ('PENDING','BLOCKED','RESERVED','SUBMITTED',
                                    'RUNNING','UNKNOWN')
                   ORDER BY id DESC LIMIT 1""",
                (
                    request["campaign_id"], request["sequence_id"],
                    request["stage"], request_id,
                ),
            ).fetchone()
            if active is not None:
                raise RuntimeError(
                    "An active sequence/stage request already exists: "
                    f"{active['id']}"
                )

            remote_status = (
                str(query_payload.get("status") or "UNKNOWN")
                if isinstance(query_payload, dict) else "UNKNOWN"
            )
            execution_status = (
                "SUCCEEDED" if remote_status == "COMPLETED"
                else "FAILED" if remote_status.startswith("FAILED")
                else "RUNNING" if remote_status == "RUNNING"
                else "UNKNOWN"
            )
            conn.execute(
                """UPDATE workflow_executions
                   SET status=?, details=?, last_query_payload_json=?,
                       completed_at=CASE
                           WHEN ? IN ('SUCCEEDED','FAILED')
                           THEN COALESCE(completed_at, ?)
                           ELSE completed_at END,
                       last_refreshed_at=?
                   WHERE id=?""",
                (
                    execution_status, details, _canonical_json(query_payload),
                    execution_status, now, now, execution_id,
                ),
            )
            run_table = {
                "calibration": "calibration_runs",
                "preprocess": "preprocess_runs",
                "reconstruction": "reconstruction_runs",
                "export": "export_runs",
            }.get(request["stage"])
            if run_table:
                conn.execute(
                    f"""UPDATE {run_table}
                        SET status='FAILED', details=?,
                            completed_at=COALESCE(completed_at, ?), updated_at=?
                        WHERE request_id=?
                          AND status NOT IN ('SUCCEEDED','CANCELED','SKIPPED')""",
                    (details, now, now, request_id),
                )
            conn.execute(
                """UPDATE stage_requests
                   SET status='FAILED', details=?, blocked_reason=NULL,
                       completed_at=COALESCE(completed_at, ?), updated_at=?
                   WHERE id=? AND status NOT IN ('SUCCEEDED','CANCELED')""",
                (details, now, now, request_id),
            )

            parameters.update({
                "infrastructure_retry_attempt": attempt,
                "prior_request_id": request_id,
                "prior_execution_id": execution_id,
                "retry_not_before": (
                    datetime.fromisoformat(now).astimezone(timezone.utc)
                    + timedelta(seconds=backoff_seconds[attempt - 1])
                ).isoformat(),
                "retry_backoff_seconds": backoff_seconds[attempt - 1],
            })
            if failure_evidence is not None:
                parameters["infrastructure_failure_evidence"] = failure_evidence
            cursor = conn.execute(
                """INSERT INTO stage_requests
                   (sequence_id, campaign_id, cohort, queue_priority, stage, status,
                    trigger, pipeline_version, requested_by, reason, parameters_json,
                    source_manifest_json, source_manifest_sha256, details,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, 'PENDING', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    request["sequence_id"], request["campaign_id"], request["cohort"],
                    request["queue_priority"], request["stage"], request["trigger"],
                    request["campaign_pipeline_version"], request["requested_by"],
                    reason, _canonical_json(parameters), request["source_manifest_json"],
                    request["source_manifest_sha256"],
                    f"infrastructure_retry_attempt={attempt}", now, now,
                ),
            )
            replacement_id = int(cursor.lastrowid)
            if request["stage"] == "export" and parameters.get("candidate_uri"):
                candidate = str(parameters["candidate_uri"]).rstrip("/")
                parent = re.sub(r"/request_\d+$", "", candidate)
                parameters["candidate_uri"] = f"{parent}/request_{replacement_id}"
                conn.execute(
                    "UPDATE stage_requests SET parameters_json=? WHERE id=?",
                    (_canonical_json(parameters), replacement_id),
                )
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return get_stage_request(
        existing_id if existing_id is not None else int(replacement_id),
        db_path=db_path,
    )


def _stage_rows(
    conn: sqlite3.Connection,
    stage: str,
    where: str,
    params: Iterable,
    *,
    order: str = "r.created_at DESC, r.id DESC",
) -> list[dict]:
    return [
        _flatten(row, stage)
        for row in conn.execute(
            _select_for_stage(stage) + f" WHERE {where} ORDER BY {order}", tuple(params)
        )
    ]


def get_current_stage_run(
    sequence_name: str,
    dataset: str,
    stage: str,
    db_path: str = DB_PATH,
    table: str = PIPELINES_TABLE,
    environment: str | None = None,
) -> dict | None:
    del table, environment
    conn = get_connection(db_path)
    try:
        rows = _stage_rows(
            conn, stage,
            "s.dataset=? AND s.sequence_name=? AND r.is_current=1",
            (dataset, sequence_name),
        )
        return rows[0] if rows else None
    finally:
        conn.close()


def get_latest_successful_stage_run(
    sequence_name: str,
    dataset: str,
    stage: str,
    db_path: str = DB_PATH,
    table: str = PIPELINES_TABLE,
) -> dict | None:
    del table
    conn = get_connection(db_path)
    try:
        rows = _stage_rows(
            conn, stage,
            "s.dataset=? AND s.sequence_name=? AND r.is_current=1 AND r.status='SUCCEEDED'",
            (dataset, sequence_name),
        )
        return rows[0] if rows else None
    finally:
        conn.close()


def list_current_successful_preprocess_lineage(
    dataset: str, *, db_path: str = DB_PATH,
) -> list[dict]:
    """Return the minimal current preprocess identity needed by schedulers."""
    conn = get_connection(db_path)
    try:
        return [dict(row) for row in conn.execute(
            """SELECT s.sequence_name, r.id AS stage_run_id, r.request_id,
                      r.output_uri
               FROM preprocess_runs r
               JOIN sequences s ON s.id=r.sequence_id
               WHERE s.dataset=? AND r.is_current=1 AND r.status='SUCCEEDED'""",
            (dataset,),
        )]
    finally:
        conn.close()


def list_current_stage_runs(
    dataset: str,
    stage: str | None = None,
    status: str | list[str] | None = None,
    db_path: str = DB_PATH,
    table: str = PIPELINES_TABLE,
    environment: str | None = None,
) -> list[dict]:
    del table, environment
    stages = (stage,) if stage else STAGES
    conn = get_connection(db_path)
    try:
        rows = []
        for candidate in stages:
            rows.extend(_stage_rows(
                conn, candidate, "s.dataset=? AND r.is_current=1", (dataset,)
            ))
    finally:
        conn.close()
    rows.sort(key=lambda row: (row.get("created_at") or "", row["id"]), reverse=True)
    if status:
        accepted = set(status if isinstance(status, list) else [status])
        rows = [row for row in rows if row["status"] in accepted]
    return rows


def list_stage_run_history(
    dataset: str,
    stage: str | None = None,
    sequence_name: str | None = None,
    db_path: str = DB_PATH,
    table: str = PIPELINES_TABLE,
) -> list[dict]:
    del table
    stages = (stage,) if stage else STAGES
    conn = get_connection(db_path)
    try:
        rows = []
        for candidate in stages:
            where = "s.dataset=?"
            params: list = [dataset]
            if sequence_name:
                where += " AND s.sequence_name=?"
                params.append(sequence_name)
            rows.extend(_stage_rows(conn, candidate, where, params))
    finally:
        conn.close()
    return sorted(
        rows, key=lambda row: (row.get("created_at") or "", row["id"]), reverse=True
    )


# -- QC and export authorization ------------------------------------------

def record_qc_review(
    reconstruction_run_id: int,
    status: str,
    details: str | None = None,
    source: str | None = None,
    external_review_id: str | None = None,
    reviewed_at: str | None = None,
    *,
    external_revision: str | None = None,
    payload_sha256: str | None = None,
    external_status: str | None = None,
    failure_annotation_count: int | None = None,
    failure_coverage: float | None = None,
    failure_segments: list[dict] | None = None,
    thresholds: dict | None = None,
    raw_payload: object | None = None,
    reviewer: str | None = None,
    db_path: str = DB_PATH,
) -> int:
    if status == "PENDING":
        return 0
    decision = "FAIL" if status in ("FAIL", "INVALID") else "PASS"
    provider = "legacy" if source and source.startswith("legacy") else "kratos"
    external_item_id = external_review_id or f"reconstruction:{reconstruction_run_id}"
    normalized_payload = raw_payload if raw_payload is not None else {
        "decision": decision,
        "details": details,
        "failure_segments": failure_segments or [],
    }
    canonical = json.dumps(normalized_payload, sort_keys=True, separators=(",", ":"), default=str)
    if payload_sha256 is None:
        import hashlib
        payload_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """INSERT OR IGNORE INTO qc_reviews
               (reconstruction_run_id, provider, external_item_id, external_revision,
                payload_sha256, external_status, decision, details,
                failure_annotation_count, failure_coverage, failure_segments_json,
                thresholds_json, raw_payload_json, reviewer, reviewed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       COALESCE(?, strftime('%Y-%m-%dT%H:%M:%fZ','now')))""",
            (reconstruction_run_id, provider, external_item_id, external_revision,
             payload_sha256, external_status, decision, details,
             failure_annotation_count, failure_coverage,
             json.dumps(failure_segments, sort_keys=True) if failure_segments is not None else None,
             json.dumps(thresholds, sort_keys=True) if thresholds is not None else None,
             canonical, reviewer, reviewed_at),
        )
        if cursor.lastrowid:
            review_id = cursor.lastrowid
        else:
            review_id = conn.execute(
                """SELECT id FROM qc_reviews
                   WHERE provider=? AND external_item_id=? AND payload_sha256=?""",
                (provider, external_item_id, payload_sha256),
            ).fetchone()[0]
        if status == "FAIL":
            sequence_id = conn.execute(
                "SELECT sequence_id FROM reconstruction_runs WHERE id=?",
                (reconstruction_run_id,),
            ).fetchone()[0]
            conn.execute(
                """INSERT INTO blacklisted_sequences
                   (sequence_id, reason, created_by) VALUES (?, ?, 'automatic_qc')
                   ON CONFLICT(sequence_id) DO NOTHING""",
                (sequence_id, details or "human QC failure"),
            )
        conn.commit()
        return review_id
    finally:
        conn.close()


def get_latest_qc_review(
    reconstruction_run_id: int, db_path: str = DB_PATH
) -> dict | None:
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            """SELECT *, decision AS status, provider AS source,
                      external_item_id AS external_review_id
               FROM qc_reviews WHERE reconstruction_run_id=?
               ORDER BY reviewed_at DESC, id DESC LIMIT 1""",
            (reconstruction_run_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_qc_review(review_id: int, db_path: str = DB_PATH) -> dict | None:
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            """SELECT *, decision AS status, provider AS source,
                      external_item_id AS external_review_id
               FROM qc_reviews WHERE id=?""",
            (review_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_sequence_status(
    dataset: str,
    sequence_name: str | None = None,
    db_path: str = DB_PATH,
    *,
    sequence_kind: str | None = None,
) -> list[dict]:
    if sequence_kind not in (None, "hoi", "calibration"):
        raise ValueError(f"Unsupported sequence kind: {sequence_kind}")
    conn = get_connection(db_path)
    try:
        query = "SELECT * FROM sequence_status WHERE dataset=?"
        params: list = [dataset]
        if sequence_name:
            query += " AND sequence_name=?"
            params.append(sequence_name)
        if sequence_kind:
            query += " AND sequence_kind=?"
            params.append(sequence_kind)
        query += " ORDER BY sequence_name"
        return [dict(row) for row in conn.execute(query, params)]
    finally:
        conn.close()


def get_sequence_history(
    dataset: str, sequence_name: str, db_path: str = DB_PATH
) -> list[dict]:
    """Return requests, stage runs, QC observations, and executions chronologically."""
    conn = get_connection(db_path)
    try:
        sequence = conn.execute(
            "SELECT id FROM sequences WHERE dataset=? AND sequence_name=?",
            (dataset, sequence_name),
        ).fetchone()
        if not sequence:
            return []
        sequence_id = sequence["id"]
        records: list[dict] = []
        execution_ids: set[int] = set()
        reconstruction_ids: list[int] = []
        for row in conn.execute(
            """SELECT sr.*, c.name AS campaign_name, c.campaign_type
               FROM stage_requests sr
               LEFT JOIN processing_campaigns c ON c.id=sr.campaign_id
               WHERE sr.sequence_id=?""",
            (sequence_id,),
        ):
            item = dict(row)
            item["record_type"] = "stage_request"
            records.append(item)
            if row["workflow_execution_id"] is not None:
                execution_ids.add(row["workflow_execution_id"])
        for stage, table in RUN_TABLES.items():
            for row in conn.execute(
                f"SELECT * FROM {table} WHERE sequence_id=?", (sequence_id,)
            ):
                item = dict(row)
                item["record_type"] = "stage_run"
                item["stage"] = stage
                records.append(item)
                if row["workflow_execution_id"] is not None:
                    execution_ids.add(row["workflow_execution_id"])
                if stage == RECONSTRUCTION_STAGE:
                    reconstruction_ids.append(row["id"])
        if reconstruction_ids:
            placeholders = ",".join("?" for _ in reconstruction_ids)
            for row in conn.execute(
                f"SELECT * FROM qc_reviews WHERE reconstruction_run_id IN ({placeholders})",
                reconstruction_ids,
            ):
                item = dict(row)
                item["record_type"] = "qc_review"
                records.append(item)
        if execution_ids:
            placeholders = ",".join("?" for _ in execution_ids)
            for row in conn.execute(
                f"SELECT * FROM workflow_executions WHERE id IN ({placeholders})",
                sorted(execution_ids),
            ):
                item = dict(row)
                item["record_type"] = "workflow_execution"
                records.append(item)

        def event_time(item: dict) -> str:
            return (
                item.get("created_at")
                or item.get("observed_at")
                or item.get("submitted_at")
                or ""
            )

        return sorted(records, key=lambda item: (event_time(item), item["record_type"], item["id"]))
    finally:
        conn.close()


# -- Compatibility wrappers ----------------------------------------------

def insert_workflow(
    sequence_name: str,
    dataset: str,
    pipeline_type: str,
    pipeline_version: str,
    workflow_name: str,
    osmo_workflow_id: str = "",
    status: str = "WAITING_WF",
    details: str = "workflow_running",
    db_path: str = DB_PATH,
    table: str = PIPELINES_TABLE,
    trigger: str = "automatic",
    created_at: str | None = None,
    updated_at: str | None = None,
    **stage_fields,
) -> dict:
    del table
    return create_stage_run(
        sequence_name=sequence_name, dataset=dataset, stage=pipeline_type,
        pipeline_version=pipeline_version, status=status, details=details,
        trigger=trigger, workflow_name=workflow_name,
        osmo_workflow_id=osmo_workflow_id or None,
        created_at=created_at, updated_at=updated_at, db_path=db_path,
        **stage_fields,
    )


def update_workflow(
    workflow_name: str,
    status: str | None = None,
    details: str | None = None,
    osmo_export_workflow_id: str | None = None,
    db_path: str = DB_PATH,
    table: str = PIPELINES_TABLE,
) -> None:
    del table
    if osmo_export_workflow_id is not None:
        raise ValueError("Export is a separate mv_export stage; create an export run")
    conn = get_connection(db_path)
    try:
        execution = conn.execute(
            "SELECT id, pipeline_stage FROM workflow_executions WHERE workflow_name=?",
            (workflow_name,),
        ).fetchone()
        if not execution:
            return
        run_stage = RUN_STAGE_NAMES[execution["pipeline_stage"]]
        run = conn.execute(
            f"SELECT id FROM {RUN_TABLES[run_stage]} "
            "WHERE workflow_execution_id=? ORDER BY id DESC LIMIT 1",
            (execution["id"],),
        ).fetchone()
    finally:
        conn.close()
    if run:
        update_stage_run(
            run[0], status=status, details=details,
            stage=run_stage, db_path=db_path,
        )


def get_workflow(
    workflow_name: str, db_path: str = DB_PATH, table: str = PIPELINES_TABLE
) -> dict | None:
    del table
    conn = get_connection(db_path)
    try:
        execution = conn.execute(
            "SELECT id, pipeline_stage FROM workflow_executions WHERE workflow_name=?",
            (workflow_name,),
        ).fetchone()
        if not execution:
            return None
        run_stage = RUN_STAGE_NAMES[execution["pipeline_stage"]]
        row = conn.execute(
            _select_for_stage(run_stage)
            + " WHERE r.workflow_execution_id=? ORDER BY r.id DESC LIMIT 1",
            (execution["id"],),
        ).fetchone()
        return _flatten(row, run_stage)
    finally:
        conn.close()


def get_latest_workflow(
    sequence_name: str,
    dataset: str,
    pipeline_type: str,
    db_path: str = DB_PATH,
    table: str = PIPELINES_TABLE,
) -> dict | None:
    return get_current_stage_run(
        sequence_name, dataset, pipeline_type, db_path=db_path, table=table
    )


def get_workflows_by_dataset(
    dataset: str,
    pipeline_type: str | None = None,
    status: str | list[str] | None = None,
    db_path: str = DB_PATH,
    table: str = PIPELINES_TABLE,
) -> list[dict]:
    rows = list_stage_run_history(
        dataset, stage=pipeline_type, db_path=db_path, table=table
    )
    if status:
        accepted = set(status if isinstance(status, list) else [status])
        rows = [row for row in rows if row["status"] in accepted]
    return rows


def get_workflows_by_export_id(
    dataset: str,
    osmo_export_workflow_id: str,
    pipeline_type: str | None = None,
    status: str | list[str] | None = None,
    db_path: str = DB_PATH,
    table: str = PIPELINES_TABLE,
) -> list[dict]:
    rows = get_workflows_by_dataset(
        dataset, pipeline_type=pipeline_type or EXPORT_STAGE,
        status=status, db_path=db_path, table=table,
    )
    return [row for row in rows if row.get("osmo_workflow_id") == osmo_export_workflow_id]


def get_summary(
    dataset: str,
    pipeline_type: str | None = None,
    db_path: str = DB_PATH,
    table: str = PIPELINES_TABLE,
) -> dict:
    rows = get_workflows_by_dataset(
        dataset, pipeline_type, db_path=db_path, table=table
    )
    counts: dict[str, int] = {}
    failures: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        if row["status"] == "FAIL":
            detail = row.get("details") or ""
            failures[detail] = failures.get(detail, 0) + 1
    return {"counts": counts, "failure_reasons": failures}
