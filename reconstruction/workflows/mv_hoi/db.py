# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SQLite database for tracking pipeline versions and workflow status.

Tables:
  pipeline_versions — semver version string with message (shared)
  workflows         — per-sequence workflow submissions and their status
  workflows_test    — same schema, used for test-mode submissions
  blacklisted_sequences — dataset-scoped sequence submission blacklist
"""

import os
import re
import sqlite3

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processing.db")


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    conn = get_connection(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS pipeline_versions (
            version    TEXT PRIMARY KEY,
            message    TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS workflows (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            sequence_name    TEXT NOT NULL,
            dataset          TEXT NOT NULL,
            pipeline_type    TEXT NOT NULL,
            pipeline_version TEXT,
            workflow_name    TEXT UNIQUE NOT NULL,
            osmo_workflow_id TEXT,
            status           TEXT NOT NULL DEFAULT 'WAITING_WF',
            details          TEXT,
            created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (pipeline_version) REFERENCES pipeline_versions(version)
        );

        CREATE INDEX IF NOT EXISTS idx_workflows_sequence
            ON workflows(dataset, sequence_name);
        CREATE INDEX IF NOT EXISTS idx_workflows_status
            ON workflows(status);

        CREATE TABLE IF NOT EXISTS workflows_test (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            sequence_name    TEXT NOT NULL,
            dataset          TEXT NOT NULL,
            pipeline_type    TEXT NOT NULL,
            pipeline_version TEXT,
            workflow_name    TEXT UNIQUE NOT NULL,
            osmo_workflow_id TEXT,
            status           TEXT NOT NULL DEFAULT 'WAITING_WF',
            details          TEXT,
            created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (pipeline_version) REFERENCES pipeline_versions(version)
        );

        CREATE INDEX IF NOT EXISTS idx_workflows_test_sequence
            ON workflows_test(dataset, sequence_name);
        CREATE INDEX IF NOT EXISTS idx_workflows_test_status
            ON workflows_test(status);

        CREATE TABLE IF NOT EXISTS blacklisted_sequences (
            dataset        TEXT NOT NULL,
            sequence_name  TEXT NOT NULL,
            reason         TEXT,
            blacklisted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (dataset, sequence_name)
        );
    """)
    columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(blacklisted_sequences)")
    }
    if "blacklisted_at" not in columns:
        conn.execute(
            "ALTER TABLE blacklisted_sequences ADD COLUMN blacklisted_at TIMESTAMP"
        )
        conn.execute(
            """UPDATE blacklisted_sequences
               SET blacklisted_at = CURRENT_TIMESTAMP
               WHERE blacklisted_at IS NULL"""
        )
    conn.commit()
    conn.close()


# ── Semver helpers ────────────────────────────────────────────────────

_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def parse_semver(version: str) -> tuple[int, int, int]:
    """Parse 'major.minor.patch' into a tuple. Raises ValueError if invalid."""
    m = _SEMVER_RE.match(version)
    if not m:
        raise ValueError(f"Invalid semver: {version!r} (expected X.Y.Z)")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def validate_semver_gt(new: str, latest: str | None) -> None:
    """Raise ValueError if *new* is not strictly greater than *latest*."""
    new_t = parse_semver(new)
    if latest is None:
        return
    latest_t = parse_semver(latest)
    if new_t <= latest_t:
        raise ValueError(
            f"Version {new} must be greater than current latest {latest}"
        )


# ── Pipeline versions ──────────────────────────────────────────────────


def get_latest_version(db_path: str = DB_PATH) -> str | None:
    """Return the latest semver string, or None if no versions exist."""
    conn = get_connection(db_path)
    rows = conn.execute("SELECT version FROM pipeline_versions").fetchall()
    conn.close()
    if not rows:
        return None
    versions = [r["version"] for r in rows]
    versions.sort(key=lambda v: parse_semver(v))
    return versions[-1]


def insert_version(version: str, message: str = "", db_path: str = DB_PATH) -> str:
    """Insert a new semver version. Validates it's greater than the latest."""
    latest = get_latest_version(db_path)
    validate_semver_gt(version, latest)
    conn = get_connection(db_path)
    conn.execute(
        "INSERT INTO pipeline_versions (version, message) VALUES (?, ?)",
        (version, message),
    )
    conn.commit()
    conn.close()
    return version


# ── Sequence blacklist ────────────────────────────────────────────────


def upsert_blacklisted_sequence(
    dataset: str,
    sequence_name: str,
    reason: str | None = None,
    db_path: str = DB_PATH,
) -> None:
    """Add a dataset/sequence blacklist entry, or update its reason."""
    conn = get_connection(db_path)
    conn.execute(
        """INSERT INTO blacklisted_sequences
           (dataset, sequence_name, reason, blacklisted_at)
           VALUES (?, ?, ?, CURRENT_TIMESTAMP)
           ON CONFLICT(dataset, sequence_name) DO UPDATE SET
               reason = excluded.reason""",
        (dataset, sequence_name, reason),
    )
    conn.commit()
    conn.close()


def remove_blacklisted_sequence(
    dataset: str,
    sequence_name: str,
    db_path: str = DB_PATH,
) -> bool:
    """Remove a dataset/sequence blacklist entry. Return True if removed."""
    conn = get_connection(db_path)
    cur = conn.execute(
        """DELETE FROM blacklisted_sequences
           WHERE dataset = ? AND sequence_name = ?""",
        (dataset, sequence_name),
    )
    conn.commit()
    removed = cur.rowcount > 0
    conn.close()
    return removed


def get_blacklisted_sequence(
    dataset: str,
    sequence_name: str,
    db_path: str = DB_PATH,
) -> dict | None:
    conn = get_connection(db_path)
    row = conn.execute(
        """SELECT dataset, sequence_name, reason, blacklisted_at
           FROM blacklisted_sequences
           WHERE dataset = ? AND sequence_name = ?""",
        (dataset, sequence_name),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def is_sequence_blacklisted(
    dataset: str,
    sequence_name: str,
    db_path: str = DB_PATH,
) -> bool:
    """Return True if the sequence is blacklisted for the dataset."""
    return get_blacklisted_sequence(dataset, sequence_name, db_path=db_path) is not None


def get_blacklisted_sequences(
    dataset: str,
    db_path: str = DB_PATH,
) -> list[dict]:
    conn = get_connection(db_path)
    rows = conn.execute(
        """SELECT dataset, sequence_name, reason, blacklisted_at
           FROM blacklisted_sequences
           WHERE dataset = ?
           ORDER BY sequence_name""",
        (dataset,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Workflow CRUD ──────────────────────────────────────────────────────


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
    table: str = "workflows",
) -> None:
    conn = get_connection(db_path)
    conn.execute(
        f"""INSERT INTO {table}
           (sequence_name, dataset, pipeline_type, pipeline_version,
            workflow_name, osmo_workflow_id, status, details)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (sequence_name, dataset, pipeline_type, pipeline_version,
         workflow_name, osmo_workflow_id, status, details),
    )
    conn.commit()
    conn.close()


def update_workflow(
    workflow_name: str,
    status: str | None = None,
    details: str | None = None,
    db_path: str = DB_PATH,
    table: str = "workflows",
) -> None:
    conn = get_connection(db_path)
    updates = ["updated_at = CURRENT_TIMESTAMP"]
    params: list = []
    if status is not None:
        updates.append("status = ?")
        params.append(status)
    if details is not None:
        updates.append("details = ?")
        params.append(details)
    params.append(workflow_name)
    conn.execute(
        f"UPDATE {table} SET {', '.join(updates)} WHERE workflow_name = ?",
        params,
    )
    conn.commit()
    conn.close()


def get_workflow(
    workflow_name: str, db_path: str = DB_PATH, table: str = "workflows",
) -> dict | None:
    conn = get_connection(db_path)
    row = conn.execute(
        f"SELECT * FROM {table} WHERE workflow_name = ?", (workflow_name,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_latest_workflow(
    sequence_name: str,
    dataset: str,
    pipeline_type: str,
    db_path: str = DB_PATH,
    table: str = "workflows",
) -> dict | None:
    conn = get_connection(db_path)
    row = conn.execute(
        f"""SELECT * FROM {table}
           WHERE sequence_name = ? AND dataset = ? AND pipeline_type = ?
           ORDER BY created_at DESC LIMIT 1""",
        (sequence_name, dataset, pipeline_type),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_recent_workflows_for_sequence(
    sequence_name: str,
    dataset: str,
    pipeline_type: str | None = None,
    limit: int = 2,
    db_path: str = DB_PATH,
    table: str = "workflows",
) -> list[dict]:
    """Return recent workflow rows for a dataset/sequence, newest first."""
    conn = get_connection(db_path)
    query = f"""SELECT * FROM {table}
       WHERE sequence_name = ? AND dataset = ?"""
    params: list = [sequence_name, dataset]
    if pipeline_type:
        query += " AND pipeline_type = ?"
        params.append(pipeline_type)
    query += " ORDER BY created_at DESC, id DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_workflows_by_dataset(
    dataset: str,
    pipeline_type: str | None = None,
    status: str | list[str] | None = None,
    db_path: str = DB_PATH,
    table: str = "workflows",
) -> list[dict]:
    conn = get_connection(db_path)
    query = f"SELECT * FROM {table} WHERE dataset = ?"
    params: list = [dataset]
    if pipeline_type:
        query += " AND pipeline_type = ?"
        params.append(pipeline_type)
    if status:
        if isinstance(status, list):
            placeholders = ", ".join("?" for _ in status)
            query += f" AND status IN ({placeholders})"
            params.extend(status)
        else:
            query += " AND status = ?"
            params.append(status)
    query += " ORDER BY created_at DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_summary(
    dataset: str,
    pipeline_type: str | None = None,
    db_path: str = DB_PATH,
    table: str = "workflows",
) -> dict:
    conn = get_connection(db_path)

    count_q = f"SELECT status, COUNT(*) as count FROM {table} WHERE dataset = ?"
    params: list = [dataset]
    if pipeline_type:
        count_q += " AND pipeline_type = ?"
        params.append(pipeline_type)
    count_q += " GROUP BY status"
    counts = {row["status"]: row["count"] for row in conn.execute(count_q, params)}

    fail_q = f"""
        SELECT details, COUNT(*) as count FROM {table}
        WHERE dataset = ? AND status = 'FAIL'
    """
    fail_params: list = [dataset]
    if pipeline_type:
        fail_q += " AND pipeline_type = ?"
        fail_params.append(pipeline_type)
    fail_q += " GROUP BY details ORDER BY count DESC"
    failures = {
        row["details"]: row["count"]
        for row in conn.execute(fail_q, fail_params)
    }

    conn.close()
    return {"counts": counts, "failure_reasons": failures}
