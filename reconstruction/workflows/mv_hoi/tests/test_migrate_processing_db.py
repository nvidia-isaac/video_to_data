import hashlib
from pathlib import Path
import sqlite3
import sys

import pytest


WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

from migration import migrate_processing_db as migration
from orchestration import db


LEGACY_SCHEMA = """
CREATE TABLE pipeline_versions (
    version TEXT PRIMARY KEY, message TEXT, created_at TIMESTAMP
);
CREATE TABLE blacklisted_sequences (
    dataset TEXT NOT NULL, sequence_name TEXT NOT NULL, reason TEXT,
    blacklisted_at TIMESTAMP, PRIMARY KEY(dataset, sequence_name)
);
CREATE TABLE {table} (
    id INTEGER PRIMARY KEY,
    sequence_name TEXT NOT NULL,
    dataset TEXT NOT NULL,
    pipeline_type TEXT NOT NULL,
    pipeline_version TEXT,
    workflow_name TEXT UNIQUE NOT NULL,
    osmo_workflow_id TEXT,
    osmo_export_workflow_id TEXT,
    status TEXT NOT NULL,
    details TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
"""


def _legacy_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(LEGACY_SCHEMA.format(table="pipelines"))
    conn.execute(
        """CREATE TABLE pipelines_test (
           id INTEGER PRIMARY KEY, sequence_name TEXT NOT NULL, dataset TEXT NOT NULL,
           pipeline_type TEXT NOT NULL, pipeline_version TEXT,
           workflow_name TEXT UNIQUE NOT NULL, osmo_workflow_id TEXT,
           osmo_export_workflow_id TEXT, status TEXT NOT NULL, details TEXT,
           created_at TIMESTAMP, updated_at TIMESTAMP)"""
    )
    conn.execute(
        "INSERT INTO pipeline_versions VALUES ('1.2.3', 'legacy', '2026-01-01 01:00:00')"
    )
    conn.execute(
        """INSERT INTO blacklisted_sequences VALUES
           ('dataset', 'blocked', 'bad input', '2026-01-02 02:00:00')"""
    )

    rows = [
        (1, "calibration-1", "dataset", db.CALIBRATION_STAGE, "1.2.3",
         "calibration", "osmo-calibration", None, "PASS", "completed",
         "2026-01-03 01:00:00", "2026-01-03 02:00:00"),
        (2, "seq-a", "dataset", db.PREPROCESS_STAGE, "1.2.3",
         "preprocess-a", "osmo-preprocess-a", None, "PASS", "completed",
         "2026-02-01 00:00:00", "2026-02-01 00:30:00"),
        (3, "seq-a", "dataset", db.RECONSTRUCTION_STAGE, "1.2.3",
         "recon-a-1", "osmo-a-1", None, "WAITING_QC", "workflow_completed",
         "2026-02-01 01:00:00", "2026-02-01 02:00:00"),
        (4, "seq-a", "dataset", db.RECONSTRUCTION_STAGE, "1.2.3",
         "recon-a-2", "osmo-a-2", None, "FAIL",
         "qc_fail: failure_coverage>50%", "2026-02-02 01:00:00",
         "2026-02-02 02:00:00"),
        (5, "seq-b", "dataset", db.RECONSTRUCTION_STAGE, "1.2.3",
         "recon-b", "osmo-b", "export-batch", "PASS", "export_completed",
         "2026-02-03 01:00:00", "2026-02-03 02:00:00"),
        (6, "seq-c", "dataset", db.RECONSTRUCTION_STAGE, "1.2.3",
         "recon-c", "osmo-c", "export-batch", "PASS", "export_completed",
         "2026-02-04 01:00:00", "2026-02-04 02:00:00"),
        (7, "seq-c", "dataset", db.RECONSTRUCTION_STAGE, "1.2.3",
         "recon-c-skipped", None, None, "SKIPPED", "manual skip",
         "2026-02-05 01:00:00", "2026-02-05 02:00:00"),
        (8, "seq-d", "dataset", db.RECONSTRUCTION_STAGE, "1.2.3",
         "recon-d", "osmo-d", "export-running", "WAITING_EXPORT",
         "export_running", "2026-02-06 01:00:00", "2026-02-06 02:00:00"),
        (9, "seq-e", "dataset", db.RECONSTRUCTION_STAGE, "1.2.3",
         "recon-e", "osmo-e", "export-failed", "FAIL",
         "task_failed: export_seq", "2026-02-07 01:00:00",
         "2026-02-07 02:00:00"),
    ]
    conn.executemany(
        "INSERT INTO pipelines VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
    )
    conn.execute(
        """INSERT INTO pipelines_test VALUES
           (1, 'seq-test', 'dataset', ?, '1.2.3', 'recon-test', 'osmo-test',
            'export-test', 'PASS', 'completed', '2026-03-01 01:00:00',
            '2026-03-01 02:00:00')""",
        (db.RECONSTRUCTION_STAGE,),
    )
    conn.commit()
    conn.close()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    return (
        tmp_path / "processing.db",
        tmp_path / "processing_v2.db",
        tmp_path / "processing_test_v2.db",
    )


def test_dry_run_validates_both_without_writing_or_mutating_source(tmp_path):
    source, production, test = _paths(tmp_path)
    _legacy_db(source)
    source_hash = _sha256(source)

    report = migration.migrate(source, production, test)

    assert report == {
        "production": {
            "pipeline_versions": 1,
            "sequences": 7,
            "blacklisted_sequences": 2,
            "workflow_executions": 11,
            "calibration_runs": 1,
            "preprocess_runs": 1,
            "reconstruction_runs": 7,
            "qc_reviews": 1,
            "export_runs": 4,
        },
        "test": {
            "pipeline_versions": 1,
            "sequences": 2,
            "blacklisted_sequences": 1,
            "workflow_executions": 2,
            "calibration_runs": 0,
            "preprocess_runs": 0,
            "reconstruction_runs": 1,
            "qc_reviews": 0,
            "export_runs": 1,
        },
    }
    assert not production.exists()
    assert not test.exists()
    assert _sha256(source) == source_hash


def test_apply_maps_explicit_runs_qc_blacklist_and_shared_exports(tmp_path):
    source, production, test = _paths(tmp_path)
    _legacy_db(source)
    source_hash = _sha256(source)

    migration.migrate(source, production, test, apply=True)

    conn = db.get_connection(str(production))
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert not conn.execute("PRAGMA foreign_key_check").fetchall()
    assert conn.execute(
        "SELECT value FROM schema_metadata WHERE key='schema_version'"
    ).fetchone()[0] == "2"

    # WAITING_QC is a successful reconstruction without an invented pending
    # review.  A completed qc_fail is a successful reconstruction plus an
    # immutable failed legacy review and a persistent blacklist entry.
    waiting_qc = conn.execute(
        """SELECT r.status, r.is_current, r.preprocess_run_id,
                  (SELECT COUNT(*) FROM qc_reviews q
                   WHERE q.reconstruction_run_id=r.id) AS reviews
           FROM reconstruction_runs r
           WHERE r.legacy_source_table='pipelines' AND r.legacy_source_id=3"""
    ).fetchone()
    assert tuple(waiting_qc) == ("SUCCEEDED", 0, 2, 0)

    failed_qc = conn.execute(
        """SELECT r.status, r.is_current, q.provider, q.decision, q.details
           FROM reconstruction_runs r JOIN qc_reviews q
             ON q.reconstruction_run_id=r.id
           WHERE r.legacy_source_table='pipelines' AND r.legacy_source_id=4"""
    ).fetchone()
    assert tuple(failed_qc) == (
        "SUCCEEDED", 1, "legacy", "FAIL", "qc_fail: failure_coverage>50%"
    )
    blacklist = conn.execute(
        """SELECT b.reason, b.created_by FROM blacklisted_sequences b
           JOIN sequences s ON s.id=b.sequence_id
           WHERE s.dataset='dataset' AND s.sequence_name='seq-a'"""
    ).fetchone()
    assert tuple(blacklist) == (
        "qc_fail: failure_coverage>50%", "legacy_qc_migration"
    )

    # One export execution represents the shared batch, with one explicit
    # export run for each sequence and direct reconstruction dependencies.
    shared = conn.execute(
        """SELECT e.pipeline_stage, e.status, COUNT(x.id)
           FROM workflow_executions e JOIN export_runs x
             ON x.workflow_execution_id=e.id
           WHERE e.osmo_workflow_id='export-batch'
           GROUP BY e.id"""
    ).fetchone()
    assert tuple(shared) == ("export", "SUCCEEDED", 2)
    exports = conn.execute(
        """SELECT x.status, x.authorization_type, x.qc_review_id,
                  r.legacy_source_id
           FROM export_runs x JOIN reconstruction_runs r
             ON r.id=x.reconstruction_run_id
           ORDER BY x.legacy_source_id"""
    ).fetchall()
    assert [tuple(row) for row in exports] == [
        ("SUCCEEDED", "LEGACY", None, 5),
        ("SUCCEEDED", "LEGACY", None, 6),
        ("RUNNING", "LEGACY", None, 8),
        ("FAILED", "LEGACY", None, 9),
    ]

    # A newer SKIPPED record is history only; it does not take ownership of
    # canonical artifacts or create a workflow execution.
    seq_c = conn.execute(
        """SELECT legacy_source_id, status, is_current, workflow_execution_id
           FROM reconstruction_runs r JOIN sequences s ON s.id=r.sequence_id
           WHERE s.sequence_name='seq-c' ORDER BY legacy_source_id"""
    ).fetchall()
    assert [tuple(row) for row in seq_c] == [
        (6, "SUCCEEDED", 1, seq_c[0][3]),
        (7, "SKIPPED", 0, None),
    ]
    assert seq_c[0][3] is not None

    # Generic v2 tables are deliberately absent.
    table_names = {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert not {
        "stage_runs", "run_dependencies", "external_inputs",
        "export_approvals", "legacy_mappings",
    } & table_names
    assert tuple(conn.execute(
        "SELECT message, created_at FROM pipeline_versions WHERE version='1.2.3'"
    ).fetchone()) == ("legacy", "2026-01-01 01:00:00")
    assert conn.execute(
        """SELECT b.created_at FROM blacklisted_sequences b JOIN sequences s
           ON s.id=b.sequence_id WHERE s.sequence_name='blocked'"""
    ).fetchone()[0] == "2026-01-02 02:00:00"
    conn.close()

    test_conn = db.get_connection(str(test))
    assert test_conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert not test_conn.execute("PRAGMA foreign_key_check").fetchall()
    assert test_conn.execute(
        "SELECT COUNT(*) FROM sequences WHERE sequence_name='seq-test'"
    ).fetchone()[0] == 1
    assert test_conn.execute(
        "SELECT COUNT(*) FROM sequences WHERE sequence_name='seq-a'"
    ).fetchone()[0] == 0
    test_export = test_conn.execute(
        "SELECT authorization_type, legacy_source_table FROM export_runs"
    ).fetchone()
    assert tuple(test_export) == ("LEGACY", "pipelines_test")
    assert test_conn.execute(
        """SELECT COUNT(*) FROM blacklisted_sequences b JOIN sequences s
           ON s.id=b.sequence_id WHERE s.sequence_name='blocked'"""
    ).fetchone()[0] == 1
    test_conn.close()
    assert _sha256(source) == source_hash


def test_apply_requires_force_and_replaces_both_destinations(tmp_path):
    source, production, test = _paths(tmp_path)
    _legacy_db(source)
    production.write_bytes(b"keep-production")
    test.write_bytes(b"keep-test")

    with pytest.raises(FileExistsError):
        migration.migrate(source, production, test, apply=True)
    assert production.read_bytes() == b"keep-production"
    assert test.read_bytes() == b"keep-test"

    migration.migrate(source, production, test, apply=True, force=True)
    for destination in (production, test):
        conn = sqlite3.connect(destination)
        assert conn.execute(
            "SELECT value FROM schema_metadata WHERE key='schema_version'"
        ).fetchone()[0] == "2"
        conn.close()


def test_pair_install_rolls_back_if_second_replace_fails(tmp_path, monkeypatch):
    source, production, test = _paths(tmp_path)
    _legacy_db(source)
    production.write_bytes(b"old-production")
    test.write_bytes(b"old-test")
    real_replace = migration.os.replace
    calls = 0

    def fail_fourth_replace(source_path, destination_path):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("simulated second install failure")
        return real_replace(source_path, destination_path)

    monkeypatch.setattr(migration.os, "replace", fail_fourth_replace)
    with pytest.raises(OSError, match="second install failure"):
        migration.migrate(source, production, test, apply=True, force=True)

    assert production.read_bytes() == b"old-production"
    assert test.read_bytes() == b"old-test"


def test_rejects_aliases_and_missing_legacy_tables(tmp_path):
    source, production, test = _paths(tmp_path)
    sqlite3.connect(source).close()
    with pytest.raises(ValueError, match="different"):
        migration.migrate(source, source, test)
    with pytest.raises(ValueError, match="different"):
        migration.migrate(source, production, production)
    with pytest.raises(ValueError, match="no pipelines"):
        migration.migrate(source, production, test)


def test_export_failure_preserves_reconstruction_success():
    row = {
        "pipeline_type": db.RECONSTRUCTION_STAGE,
        "status": "FAIL",
        "details": "task_failed: export_seq",
        "_export_id": "export-batch-1",
    }

    assert migration._main_run_status(row) == "SUCCEEDED"
    assert migration._export_run_status(row["status"]) == "FAILED"
