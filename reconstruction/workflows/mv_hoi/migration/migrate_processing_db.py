"""Migrate the legacy MV-HOI processing database to explicit schema v2.

The legacy database is always opened read-only.  Production ``pipelines`` and
test ``pipelines_test`` rows are written to separate destination databases.
The command defaults to a dry run: both databases are built and validated in
temporary files, but neither requested destination is written unless
``--apply`` is supplied.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
from urllib.parse import quote

SCRIPT_DIR = Path(__file__).resolve().parent
MV_HOI_DIR = SCRIPT_DIR.parent

try:
    from ..orchestration import db
    from ..orchestration.runtime import require_submit_authority
except ImportError:
    # Support direct execution and top-level ``migration`` package imports.
    sys.path.insert(0, str(MV_HOI_DIR))
    from orchestration import db
    from orchestration.runtime import require_submit_authority


DEFAULT_SOURCE = MV_HOI_DIR / "processing.db"
DEFAULT_PRODUCTION_DESTINATION = MV_HOI_DIR / "processing_v2.db"
DEFAULT_TEST_DESTINATION = MV_HOI_DIR / "processing_test_v2.db"

DESTINATIONS = (
    ("production", "pipelines"),
    ("test", "pipelines_test"),
)
REPORT_TABLES = (
    "pipeline_versions",
    "sequences",
    "blacklisted_sequences",
    "workflow_executions",
    "calibration_runs",
    "preprocess_runs",
    "reconstruction_runs",
    "qc_reviews",
    "export_runs",
)
TERMINAL_RUN_STATUSES = {"SUCCEEDED", "FAILED", "CANCELED", "SKIPPED"}
TERMINAL_EXECUTION_STATUSES = {"SUCCEEDED", "FAILED", "CANCELED"}


def _open_legacy_read_only(path: Path) -> sqlite3.Connection:
    resolved = path.resolve(strict=True)
    uri = f"file:{quote(str(resolved), safe='/')}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _value(row: dict, key: str) -> str | None:
    value = row.get(key)
    return str(value) if value is not None else None


def _normalize_status(status: str, *, execution: bool = False) -> str:
    aliases = {
        "WAITING_WF": "RUNNING",
        "WAITING_EXPORT": "RUNNING",
        "WAITING_QC": "SUCCEEDED",
        "IN_PROGRESS": "RUNNING",
        "PASS": "SUCCEEDED",
        "COMPLETED": "SUCCEEDED",
        "FAIL": "FAILED",
        "CANCELLED": "CANCELED",
    }
    normalized = aliases.get(status.upper(), status.upper())
    accepted = {
        "SUBMITTING", "RUNNING", "SUCCEEDED", "FAILED", "CANCELED", "UNKNOWN"
    }
    if not execution:
        accepted.add("SKIPPED")
    if normalized.startswith("FAILED"):
        normalized = "FAILED"
    if normalized not in accepted:
        raise ValueError(f"Unsupported legacy status: {status!r}")
    return normalized


def _is_qc_failure(row: dict) -> bool:
    return (
        str(row.get("status", "")).upper() in {"FAIL", "FAILED"}
        and str(row.get("details") or "").lower().startswith("qc_fail:")
    )


def _main_run_status(row: dict) -> str:
    """Normalize the legacy overloaded row to its actual pipeline stage."""
    raw = str(row["status"])
    if row["pipeline_type"] == db.RECONSTRUCTION_STAGE:
        # These states prove reconstruction completed.  FAIL with an export ID
        # is an overloaded export failure, while qc_fail is a completed QC
        # observation against a successful reconstruction.
        if raw.upper() in {"WAITING_QC", "WAITING_EXPORT", "PASS", "SUCCEEDED"}:
            return "SUCCEEDED"
        if _is_qc_failure(row) or row.get("_export_id"):
            return "SUCCEEDED"
    return _normalize_status(raw)


def _export_run_status(legacy_status: str) -> str:
    normalized = legacy_status.upper()
    if normalized == "WAITING_EXPORT":
        return "RUNNING"
    if normalized in {"PASS", "SUCCEEDED", "COMPLETED"}:
        return "SUCCEEDED"
    if normalized in {"FAIL", "FAILED"}:
        return "FAILED"
    if normalized in {"CANCELED", "CANCELLED"}:
        return "CANCELED"
    # An export ID proves an attempt existed, but other overloaded legacy
    # states do not identify the export task's state reliably.
    return "UNKNOWN"


def _export_task_names(sequence_name: str) -> tuple[str, str]:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", sequence_name).strip("_") or "sequence"
    digest = hashlib.sha1(sequence_name.encode("utf-8")).hexdigest()[:8]
    suffix = f"{safe[:50]}_{digest}"
    return f"export_{suffix}", f"copy_failure_segments_{suffix}"


def _ensure_version(
    conn: sqlite3.Connection, version: str | None, *, message: str | None = None,
    registry_metadata_json: str | None = None, created_at: str | None = None,
) -> None:
    if not version:
        return
    conn.execute(
        """INSERT OR IGNORE INTO pipeline_versions
           (version, message, registry_metadata_json, created_at)
           VALUES (?, ?, ?, ?)""",
        (
            version,
            message if message is not None else "created by legacy migration",
            registry_metadata_json,
            created_at or _now(),
        ),
    )


def _copy_pipeline_versions(src: sqlite3.Connection, dst: sqlite3.Connection) -> None:
    if not _table_exists(src, "pipeline_versions"):
        return
    columns = _columns(src, "pipeline_versions")
    registry_expr = (
        "registry_metadata_json" if "registry_metadata_json" in columns else "NULL"
    )
    created_expr = "created_at" if "created_at" in columns else "NULL"
    for row in src.execute(
        f"""SELECT version, message, {registry_expr} AS registry_metadata_json,
                   {created_expr} AS created_at
            FROM pipeline_versions"""
    ):
        _ensure_version(
            dst,
            row["version"],
            message=row["message"],
            registry_metadata_json=row["registry_metadata_json"],
            created_at=_value(dict(row), "created_at"),
        )


def _ensure_sequence(
    conn: sqlite3.Connection,
    *,
    dataset: str,
    sequence_name: str,
    sequence_kind: str,
    discovered_at: str | None,
    updated_at: str | None,
) -> int:
    discovered = discovered_at or updated_at or _now()
    updated = updated_at or discovered
    conn.execute(
        """INSERT INTO sequences
           (dataset, sequence_name, sequence_kind, discovered_at, updated_at)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(dataset, sequence_name) DO UPDATE SET
             sequence_kind=CASE
               WHEN excluded.sequence_kind='calibration' THEN 'calibration'
               ELSE sequences.sequence_kind END,
             discovered_at=min(sequences.discovered_at, excluded.discovered_at),
             updated_at=max(sequences.updated_at, excluded.updated_at)""",
        (dataset, sequence_name, sequence_kind, discovered, updated),
    )
    return dst_scalar(
        conn,
        "SELECT id FROM sequences WHERE dataset=? AND sequence_name=?",
        (dataset, sequence_name),
    )


def dst_scalar(conn: sqlite3.Connection, sql: str, params: tuple = ()):
    row = conn.execute(sql, params).fetchone()
    if row is None:
        raise RuntimeError(f"Expected a row for query: {sql}")
    return row[0]


def _copy_blacklists(src: sqlite3.Connection, dst: sqlite3.Connection) -> None:
    if not _table_exists(src, "blacklisted_sequences"):
        return
    columns = _columns(src, "blacklisted_sequences")
    time_expr = (
        "blacklisted_at" if "blacklisted_at" in columns
        else "created_at" if "created_at" in columns else "NULL"
    )
    for legacy in src.execute(
        f"""SELECT dataset, sequence_name, reason, {time_expr} AS created_at
            FROM blacklisted_sequences"""
    ):
        row = dict(legacy)
        sequence_id = _ensure_sequence(
            dst,
            dataset=row["dataset"],
            sequence_name=row["sequence_name"],
            sequence_kind="hoi",
            discovered_at=_value(row, "created_at"),
            updated_at=_value(row, "created_at"),
        )
        dst.execute(
            """INSERT INTO blacklisted_sequences
               (sequence_id, reason, created_by, created_at)
               VALUES (?, ?, 'legacy_migration', ?)
               ON CONFLICT(sequence_id) DO NOTHING""",
            (sequence_id, row.get("reason"), _value(row, "created_at") or _now()),
        )


def _unique_workflow_name(
    conn: sqlite3.Connection, preferred: str, source_table: str, legacy_id: int
) -> str:
    candidate = preferred
    suffix = 0
    while conn.execute(
        "SELECT 1 FROM workflow_executions WHERE workflow_name=?", (candidate,)
    ).fetchone():
        suffix += 1
        candidate = f"{preferred}:legacy-{source_table}-{legacy_id}-{suffix}"
    return candidate


def _insert_execution(
    conn: sqlite3.Connection,
    *,
    stage: str,
    pipeline_version: str | None,
    workflow_name: str,
    osmo_workflow_id: str | None,
    status: str,
    details: str | None,
    created_at: str | None,
    updated_at: str | None,
    source_table: str,
    legacy_id: int,
) -> int:
    execution_stage = db.EXECUTION_STAGE_NAMES[stage]
    if osmo_workflow_id:
        existing = conn.execute(
            """SELECT id, pipeline_stage, pipeline_version FROM workflow_executions
               WHERE osmo_workflow_id=?""",
            (osmo_workflow_id,),
        ).fetchone()
        if existing:
            if existing["pipeline_stage"] != execution_stage:
                raise ValueError(
                    f"OSMO workflow {osmo_workflow_id!r} occurs in both "
                    f"{existing['pipeline_stage']!r} and {execution_stage!r}"
                )
            # Legacy per-sequence rows carry the reconstruction version, not a
            # trustworthy batch-export version.  Do not claim a single version
            # when one shared export contains mixed reconstruction versions.
            if existing["pipeline_version"] != pipeline_version:
                conn.execute(
                    "UPDATE workflow_executions SET pipeline_version=NULL WHERE id=?",
                    (existing["id"],),
                )
            return existing["id"]

    name = _unique_workflow_name(
        conn, workflow_name or f"legacy:{source_table}:{legacy_id}", source_table, legacy_id
    )
    timestamp = created_at or updated_at or _now()
    cursor = conn.execute(
        """INSERT INTO workflow_executions
           (pipeline_stage, pipeline_version, backend, workflow_name,
            osmo_workflow_id, status, details, submitted_at, started_at,
            completed_at, last_refreshed_at)
           VALUES (?, ?, 'osmo', ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            execution_stage,
            pipeline_version,
            name,
            osmo_workflow_id,
            status,
            details,
            timestamp,
            timestamp if status != "SUBMITTING" else None,
            (updated_at or timestamp) if status in TERMINAL_EXECUTION_STATUSES else None,
            updated_at or timestamp,
        ),
    )
    return cursor.lastrowid


def _insert_stage_run(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    stage: str,
    sequence_id: int,
    execution_id: int | None,
    pipeline_version: str | None,
    status: str,
    details: str | None,
    source_table: str,
    legacy_id: int,
    created_at: str | None,
    updated_at: str | None,
    preprocess_run_id: int | None = None,
    reconstruction_run_id: int | None = None,
    qc_review_id: int | None = None,
    workflow_name: str | None = None,
    authorization_type: str | None = None,
) -> None:
    timestamp = created_at or updated_at or _now()
    common_columns = (
        "id, sequence_id, workflow_execution_id, pipeline_version, status, details, "
        "trigger, requested_by, is_current, legacy_source_table, legacy_source_id, "
        "created_at, started_at, completed_at, updated_at"
    )
    common_values: list[object] = [
        run_id,
        sequence_id,
        execution_id,
        pipeline_version,
        status,
        details,
        "MIGRATION",
        "legacy_migration",
        0,
        source_table,
        legacy_id,
        timestamp,
        timestamp if status not in {"SUBMITTING", "SKIPPED"} else None,
        (updated_at or timestamp) if status in TERMINAL_RUN_STATUSES else None,
        updated_at or timestamp,
    ]

    if stage == db.CALIBRATION_STAGE:
        table = "calibration_runs"
        columns = common_columns
        values = common_values
    elif stage == db.PREPROCESS_STAGE:
        table = "preprocess_runs"
        columns = common_columns + ", calibration_run_id"
        values = common_values + [None]
    elif stage == db.RECONSTRUCTION_STAGE:
        table = "reconstruction_runs"
        columns = common_columns + ", preprocess_run_id, hitl_item_id"
        values = common_values + [preprocess_run_id, workflow_name]
    elif stage == db.EXPORT_STAGE:
        table = "export_runs"
        task_name, copy_name = _export_task_names(
            dst_scalar(conn, "SELECT sequence_name FROM sequences WHERE id=?", (sequence_id,))
        )
        columns = common_columns + (
            ", reconstruction_run_id, qc_review_id, authorization_type, "
            "export_task_name, copy_task_name"
        )
        values = common_values + [
            reconstruction_run_id,
            qc_review_id,
            authorization_type or "LEGACY",
            task_name,
            copy_name,
        ]
    else:
        raise ValueError(f"Unsupported pipeline_type: {stage!r}")

    placeholders = ", ".join("?" for _ in values)
    conn.execute(f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", values)


def _insert_failed_qc_review(
    conn: sqlite3.Connection,
    *,
    reconstruction_run_id: int,
    sequence_id: int,
    row: dict,
    source_table: str,
) -> int:
    payload = {
        "legacy_source_table": source_table,
        "legacy_source_id": row["id"],
        "status": row["status"],
        "details": row.get("details"),
    }
    raw_payload = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    payload_sha256 = hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()
    reviewed_at = _value(row, "updated_at") or _value(row, "created_at") or _now()
    cursor = conn.execute(
        """INSERT INTO qc_reviews
           (reconstruction_run_id, provider, external_item_id, external_revision,
            payload_sha256, external_status, decision, details, thresholds_json,
            raw_payload_json, reviewer, reviewed_at, observed_at)
           VALUES (?, 'legacy', ?, ?, ?, 'FAIL', 'FAIL', ?, '{}', ?,
                   'legacy_migration', ?, ?)""",
        (
            reconstruction_run_id,
            row.get("workflow_name") or f"legacy:{source_table}:{row['id']}",
            str(row["id"]),
            payload_sha256,
            row.get("details"),
            raw_payload,
            reviewed_at,
            reviewed_at,
        ),
    )
    conn.execute(
        """INSERT INTO blacklisted_sequences
           (sequence_id, reason, created_by, created_at)
           VALUES (?, ?, 'legacy_qc_migration', ?)
           ON CONFLICT(sequence_id) DO NOTHING""",
        (sequence_id, row.get("details") or "legacy QC failure", reviewed_at),
    )
    return cursor.lastrowid


def _mark_current_runs(conn: sqlite3.Connection) -> None:
    for table in db.RUN_TABLES.values():
        sequence_ids = [
            row[0] for row in conn.execute(f"SELECT DISTINCT sequence_id FROM {table}")
        ]
        for sequence_id in sequence_ids:
            eligible = conn.execute(
                f"""SELECT id, created_at FROM {table}
                    WHERE sequence_id=? AND status!='SKIPPED'
                    ORDER BY created_at, id""",
                (sequence_id,),
            ).fetchall()
            if not eligible:
                continue
            for index, run in enumerate(eligible[:-1]):
                conn.execute(
                    f"UPDATE {table} SET superseded_at=? WHERE id=?",
                    (eligible[index + 1]["created_at"], run["id"]),
                )
            conn.execute(
                f"UPDATE {table} SET is_current=1, superseded_at=NULL WHERE id=?",
                (eligible[-1]["id"],),
            )


def _aggregate_execution_status(statuses: list[str]) -> str:
    for candidate in (
        "RUNNING", "SUBMITTING", "FAILED", "CANCELED", "UNKNOWN", "SUCCEEDED"
    ):
        if candidate in statuses:
            return candidate
    raise ValueError("Cannot aggregate an empty execution-status list")


def _validate_legacy_table(src: sqlite3.Connection, table: str) -> None:
    required = {
        "id", "sequence_name", "dataset", "pipeline_type", "pipeline_version",
        "workflow_name", "status", "details", "created_at", "updated_at",
    }
    missing = required - _columns(src, table)
    if missing:
        raise ValueError(f"Legacy table {table} is missing columns: {sorted(missing)}")


def _migrate_table(source: Path, source_table: str, destination: Path) -> dict[str, int]:
    src = _open_legacy_read_only(source)
    db.init_db(str(destination))
    dst = db.get_connection(str(destination))
    run_id = 0
    latest_non_skipped: dict[tuple[str, str, str], int] = {}
    export_execution_statuses: defaultdict[int, list[str]] = defaultdict(list)
    try:
        dst.execute("BEGIN IMMEDIATE")
        _copy_pipeline_versions(src, dst)
        _copy_blacklists(src, dst)

        if _table_exists(src, source_table):
            _validate_legacy_table(src, source_table)
            columns = _columns(src, source_table)
            export_expr = (
                "osmo_export_workflow_id"
                if "osmo_export_workflow_id" in columns else "NULL"
            )
            osmo_expr = "osmo_workflow_id" if "osmo_workflow_id" in columns else "NULL"
            rows = src.execute(
                f"""SELECT *, {export_expr} AS _export_id,
                           {osmo_expr} AS _osmo_workflow_id
                    FROM {source_table}
                    ORDER BY created_at, id"""
            )
            for legacy in rows:
                row = dict(legacy)
                stage = row["pipeline_type"]
                if stage not in (
                    db.CALIBRATION_STAGE,
                    db.PREPROCESS_STAGE,
                    db.RECONSTRUCTION_STAGE,
                ):
                    raise ValueError(
                        f"Unsupported pipeline_type {stage!r} in "
                        f"{source_table} id={row['id']}"
                    )
                _ensure_version(dst, row.get("pipeline_version"))
                sequence_id = _ensure_sequence(
                    dst,
                    dataset=row["dataset"],
                    sequence_name=row["sequence_name"],
                    sequence_kind=(
                        "calibration" if stage == db.CALIBRATION_STAGE else "hoi"
                    ),
                    discovered_at=_value(row, "created_at"),
                    updated_at=_value(row, "updated_at"),
                )
                status = _main_run_status(row)
                execution_id = None
                if status != "SKIPPED":
                    execution_id = _insert_execution(
                        dst,
                        stage=stage,
                        pipeline_version=row.get("pipeline_version"),
                        workflow_name=row.get("workflow_name")
                        or f"legacy:{source_table}:{row['id']}",
                        osmo_workflow_id=row.get("_osmo_workflow_id") or None,
                        status=_normalize_status(status, execution=True),
                        details=row.get("details"),
                        created_at=_value(row, "created_at"),
                        updated_at=_value(row, "updated_at"),
                        source_table=source_table,
                        legacy_id=row["id"],
                    )

                key = (row["dataset"], row["sequence_name"], stage)
                preprocess_run_id = latest_non_skipped.get(
                    (row["dataset"], row["sequence_name"], db.PREPROCESS_STAGE)
                )
                run_id += 1
                _insert_stage_run(
                    dst,
                    run_id=run_id,
                    stage=stage,
                    sequence_id=sequence_id,
                    execution_id=execution_id,
                    pipeline_version=row.get("pipeline_version"),
                    status=status,
                    details=row.get("details"),
                    source_table=source_table,
                    legacy_id=row["id"],
                    created_at=_value(row, "created_at"),
                    updated_at=_value(row, "updated_at"),
                    preprocess_run_id=preprocess_run_id,
                    workflow_name=row.get("workflow_name"),
                )
                main_run_id = run_id
                if status != "SKIPPED":
                    latest_non_skipped[key] = main_run_id

                if stage == db.RECONSTRUCTION_STAGE and _is_qc_failure(row):
                    _insert_failed_qc_review(
                        dst,
                        reconstruction_run_id=main_run_id,
                        sequence_id=sequence_id,
                        row=row,
                        source_table=source_table,
                    )

                export_id = row.get("_export_id") or None
                if stage == db.RECONSTRUCTION_STAGE and export_id:
                    export_status = _export_run_status(str(row["status"]))
                    export_execution_id = _insert_execution(
                        dst,
                        stage=db.EXPORT_STAGE,
                        pipeline_version=row.get("pipeline_version"),
                        workflow_name=str(export_id),
                        osmo_workflow_id=str(export_id),
                        status=export_status,
                        details=row.get("details"),
                        created_at=_value(row, "created_at"),
                        updated_at=_value(row, "updated_at"),
                        source_table=source_table,
                        legacy_id=row["id"],
                    )
                    export_execution_statuses[export_execution_id].append(export_status)
                    run_id += 1
                    _insert_stage_run(
                        dst,
                        run_id=run_id,
                        stage=db.EXPORT_STAGE,
                        sequence_id=sequence_id,
                        execution_id=export_execution_id,
                        pipeline_version=row.get("pipeline_version"),
                        status=export_status,
                        details=row.get("details"),
                        source_table=source_table,
                        legacy_id=row["id"],
                        created_at=_value(row, "created_at"),
                        updated_at=_value(row, "updated_at"),
                        reconstruction_run_id=main_run_id,
                        authorization_type="LEGACY",
                    )
                    latest_non_skipped[
                        (row["dataset"], row["sequence_name"], db.EXPORT_STAGE)
                    ] = run_id

        for execution_id, statuses in export_execution_statuses.items():
            dst.execute(
                "UPDATE workflow_executions SET status=? WHERE id=?",
                (_aggregate_execution_status(statuses), execution_id),
            )
        _mark_current_runs(dst)
        dst.commit()

        integrity = dst.execute("PRAGMA integrity_check").fetchone()[0]
        foreign_keys = dst.execute("PRAGMA foreign_key_check").fetchall()
        if integrity != "ok" or foreign_keys:
            raise RuntimeError(
                f"Destination validation failed: integrity={integrity!r}, "
                f"foreign_key_errors={len(foreign_keys)}"
            )
        return {
            table: dst.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in REPORT_TABLES
        }
    except Exception:
        dst.rollback()
        raise
    finally:
        dst.close()
        src.close()


def _remove_sqlite_file(path: Path) -> None:
    path.unlink(missing_ok=True)
    Path(f"{path}-wal").unlink(missing_ok=True)
    Path(f"{path}-shm").unlink(missing_ok=True)


def _temporary_sibling(destination: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    path = Path(name)
    path.unlink()
    return path


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directories(paths: list[Path]) -> None:
    for directory in sorted({path.parent for path in paths}):
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _install_destinations(
    temporary_paths: list[Path], destinations: list[Path], *, force: bool
) -> None:
    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    try:
        if not force:
            existing = [destination for destination in destinations if destination.exists()]
            if existing:
                raise FileExistsError(
                    f"Destination appeared during migration: {existing[0]}"
                )
        if force:
            for destination in destinations:
                if destination.exists():
                    backup = _temporary_sibling(destination)
                    os.replace(destination, backup)
                    backups[destination] = backup
        for temporary, destination in zip(temporary_paths, destinations):
            os.replace(temporary, destination)
            installed.append(destination)
        _fsync_directories(destinations)
    except Exception:
        for destination in installed:
            _remove_sqlite_file(destination)
        for destination, backup in backups.items():
            os.replace(backup, destination)
        _fsync_directories(destinations)
        raise
    else:
        for backup in backups.values():
            _remove_sqlite_file(backup)


def migrate(
    source: str | os.PathLike,
    production_destination: str | os.PathLike,
    test_destination: str | os.PathLike,
    *,
    apply: bool = False,
    force: bool = False,
) -> dict[str, dict[str, int]]:
    """Build, validate, and optionally install both schema-v2 databases."""
    source_path = Path(source).resolve(strict=True)
    destinations = [
        Path(production_destination).resolve(),
        Path(test_destination).resolve(),
    ]
    if source_path in destinations:
        raise ValueError("Source and destination must be different files")
    if destinations[0] == destinations[1]:
        raise ValueError("Production and test destinations must be different files")

    with _open_legacy_read_only(source_path) as src:
        if not any(_table_exists(src, table) for _, table in DESTINATIONS):
            raise ValueError("Source database has no pipelines or pipelines_test table")

    if apply and not force:
        existing = [path for path in destinations if path.exists()]
        if existing:
            raise FileExistsError(
                f"Destination exists: {existing[0]}; use --force to replace both outputs"
            )

    source_sha256 = _sha256_file(source_path)
    reports: dict[str, dict[str, int]] = {}

    if not apply:
        with tempfile.TemporaryDirectory(prefix="mv-hoi-migration-") as directory:
            temporary_paths = [
                Path(directory) / "processing_v2.db",
                Path(directory) / "processing_test_v2.db",
            ]
            for (label, table), temporary in zip(DESTINATIONS, temporary_paths):
                reports[label] = _migrate_table(source_path, table, temporary)
        if _sha256_file(source_path) != source_sha256:
            raise RuntimeError("Legacy source changed while migration was running")
        return reports

    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths = [_temporary_sibling(destination) for destination in destinations]
    try:
        for (label, table), temporary in zip(DESTINATIONS, temporary_paths):
            reports[label] = _migrate_table(source_path, table, temporary)
            _fsync_file(temporary)
        if _sha256_file(source_path) != source_sha256:
            raise RuntimeError("Legacy source changed while migration was running")
        _install_destinations(temporary_paths, destinations, force=force)
        return reports
    finally:
        for temporary in temporary_paths:
            _remove_sqlite_file(temporary)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="legacy database (default: mv_hoi/processing.db)",
    )
    parser.add_argument(
        "--production-destination",
        type=Path,
        default=DEFAULT_PRODUCTION_DESTINATION,
        help="schema-v2 production database",
    )
    parser.add_argument(
        "--test-destination",
        type=Path,
        default=DEFAULT_TEST_DESTINATION,
        help="schema-v2 test database",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="atomically replace the destination files (default: validate only)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="replace existing destinations; only valid with --apply",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.force and not args.apply:
        raise SystemExit("--force requires --apply")
    if args.apply:
        require_submit_authority("write migrated orchestration databases")
    report = migrate(
        args.source,
        args.production_destination,
        args.test_destination,
        apply=args.apply,
        force=args.force,
    )
    mode = "applied" if args.apply else "dry-run validated"
    print(f"Migration {mode}: {json.dumps(report, sort_keys=True)}")
    if not args.apply:
        print("No destination was written. Re-run with --apply to commit it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
