from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import pytest


WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

from orchestration import db, lifecycle, query


def _database(tmp_path: Path) -> str:
    path = str(tmp_path / "orchestration.db")
    db.init_db(path)
    db.ensure_version_cached("1.6.0", db_path=path)
    return path


def _campaign(
    path: str,
    name: str,
    *,
    campaign_type: str = "LEGACY_REVALIDATION",
    count: int = 1,
) -> dict:
    campaign = db.create_campaign(
        name=name,
        campaign_type=campaign_type,
        dataset="dataset",
        pipeline_version="1.6.0",
        output_uri="swift://example/data_export_2",
        created_by="test",
        db_path=path,
    )
    return db.freeze_campaign(
        campaign["id"],
        inventory_uri=f"swift://example/{name}/inventory.json",
        inventory_sha256="a" * 64,
        inventory_sequence_count=count,
        configuration_uri=f"swift://example/{name}/configuration.json",
        configuration_sha256="b" * 64,
        db_path=path,
    )


def _request(
    path: str,
    campaign: dict,
    sequence: str,
    *,
    stage: str = "revalidation",
    status: str = "PENDING",
    blocked_reason: str | None = None,
) -> dict:
    return db.create_stage_request(
        sequence_name=sequence,
        dataset="dataset",
        stage=stage,
        pipeline_version="1.6.0",
        campaign=campaign["id"],
        cohort="CANARY" if campaign["campaign_type"] == "LEGACY_REVALIDATION" else "BULK",
        status=status,
        blocked_reason=blocked_reason,
        db_path=path,
    )


def _set_request(
    path: str,
    request: dict,
    status: str,
    *,
    result_summary: dict | None = None,
) -> None:
    kwargs = {}
    if request["stage"] == "revalidation" and status == "SUCCEEDED":
        kwargs = {
            "result_manifest_uri": "swift://example/result/commit.json",
            "result_manifest_sha256": "c" * 64,
        }
    db.update_stage_request(
        request["id"],
        status=status,
        result_summary=result_summary,
        db_path=path,
        **kwargs,
    )


def _insert_export(
    path: str,
    request: dict | None,
    *,
    sequence: str,
    authorization: str = "REVALIDATION",
    status: str = "SUCCEEDED",
    output_uri: str | None = None,
) -> int:
    execution_id = db.create_workflow_execution(
        workflow_name=f"export-{sequence}-{request['id'] if request else 'legacy'}",
        pipeline_type=db.EXPORT_STAGE,
        pipeline_version="1.6.0",
        status=status,
        db_path=path,
    )
    conn = db.get_connection(path)
    try:
        sequence_id = conn.execute(
            "SELECT id FROM sequences WHERE dataset=? AND sequence_name=?",
            ("dataset", sequence),
        ).fetchone()[0]
        run_id = conn.execute(
            """SELECT COALESCE(MAX(id), 0) + 1 FROM (
                   SELECT id FROM calibration_runs
                   UNION ALL SELECT id FROM preprocess_runs
                   UNION ALL SELECT id FROM reconstruction_runs
                   UNION ALL SELECT id FROM export_runs
               ) all_runs"""
        ).fetchone()[0]
        conn.execute(
            """INSERT INTO export_runs
               (id, sequence_id, workflow_execution_id, pipeline_version,
                status, trigger, output_uri, is_current, authorization_type,
                request_id)
               VALUES (?, ?, ?, '1.6.0', ?, ?, ?, 0, ?, ?)""",
            (
                run_id,
                sequence_id,
                execution_id,
                status,
                "MIGRATION" if authorization == "LEGACY" else "MANUAL",
                output_uri or f"swift://example/data_export_2/{sequence}",
                authorization,
                request["id"] if request else None,
            ),
        )
        conn.commit()
        return run_id
    finally:
        conn.close()


def _snapshot(rows: list[dict]) -> dict:
    return {
        "observed_at": "2026-07-29T00:00:00.000Z",
        "campaigns": [{
            "id": 1,
            "name": "campaign",
            "campaign_type": "BACKLOG_REPROCESSING",
            "status": "RUNNING",
            "phase": "BULK",
            "inventory_sequence_count": len(rows),
        }],
        "campaign_memberships": [{
            "campaign_id": 1,
            "member_count": len(rows),
        }],
        "sequences": rows,
    }


def test_revalidation_result_pending_is_waiting_export():
    resolved = lifecycle.resolve_lifecycle_row({
        "id": 7,
        "stage": "revalidation",
        "status": "RUNNING",
        "details": "result_reconciliation_pending",
        "updated_at": "2026-08-02T00:00:00Z",
        "export_run_status": None,
    })

    assert resolved["lifecycle_status"] == "WAITING_EXPORT"
    assert resolved["lifecycle_stage"] == "export"
    assert resolved["lifecycle_details"] == "result_reconciliation_pending"


def test_newest_active_campaign_request_wins_and_closed_campaign_is_ignored(
    tmp_path: Path,
):
    path = _database(tmp_path)
    primary = _campaign(path, "primary")
    older = _request(path, primary, "shared")
    _set_request(path, older, "SUCCEEDED")
    _insert_export(path, older, sequence="shared")

    remediation = _campaign(
        path, "remediation", campaign_type="REMEDIATION",
    )
    newer = _request(path, remediation, "shared")
    _insert_export(
        path,
        None,
        sequence="shared",
        authorization="LEGACY",
        output_uri="swift://example/data_export/shared",
    )

    closed = _campaign(
        path, "closed", campaign_type="REMEDIATION",
    )
    failed = _request(path, closed, "closed-only", stage="preprocess")
    _set_request(path, failed, "FAILED")
    conn = db.get_connection(path)
    try:
        conn.execute(
            """UPDATE processing_campaigns
               SET status='COMPLETED_WITH_FAILURES', phase='COMPLETE'
               WHERE id=?""",
            (closed["id"],),
        )
        conn.commit()
    finally:
        conn.close()

    snapshot = db.get_campaign_lifecycle_snapshot("dataset", db_path=path)
    assert [row["sequence_name"] for row in snapshot["sequences"]] == ["shared"]
    assert snapshot["sequences"][0]["id"] == newer["id"]
    summary = query.build_campaign_lifecycle_summary(snapshot)
    assert summary["buckets"] == {"REVALIDATION_PENDING": 1}
    assert summary["distinct_membership"] == 1
    assert summary["actual_membership_references"] == 2
    assert summary["overlap"] == 1
    assert summary["membership_reconciles"]


def test_committed_revalidation_and_adopted_fulfillment_are_exported(
    tmp_path: Path,
):
    path = _database(tmp_path)
    committed_campaign = _campaign(path, "committed")
    committed = _request(path, committed_campaign, "committed-sequence")
    _set_request(path, committed, "SUCCEEDED")
    _insert_export(path, committed, sequence="committed-sequence")
    committed_summary = query.build_campaign_lifecycle_summary(
        db.get_campaign_lifecycle_snapshot(
            "dataset", campaign="committed", db_path=path,
        )
    )
    assert committed_summary["buckets"] == {"EXPORTED": 1}
    assert committed_summary["export_authorizations"] == {"REVALIDATION": 1}

    target_campaign = _campaign(path, "target")
    target = _request(path, target_campaign, "adopted-sequence")
    _set_request(path, target, "CANCELED")
    source = db.create_stage_request(
        sequence_name="adopted-sequence",
        dataset="dataset",
        stage="revalidation",
        pipeline_version="1.6.0",
        db_path=path,
    )
    _set_request(path, source, "SUCCEEDED")
    _insert_export(
        path,
        source,
        sequence="adopted-sequence",
        output_uri="swift://example/data_export_2/adopted-sequence",
    )
    db.adopt_request_fulfillment(target["id"], source["id"], db_path=path)

    adopted_summary = query.build_campaign_lifecycle_summary(
        db.get_campaign_lifecycle_snapshot(
            "dataset", campaign="target", db_path=path,
        )
    )
    assert adopted_summary["buckets"] == {"EXPORTED": 1}
    assert adopted_summary["export_authorizations"] == {"REVALIDATION": 1}


def test_lifecycle_transitions_are_mutually_exclusive_and_retain_categories():
    rows = [
        {"stage": "preprocess", "status": "PENDING"},
        {
            "stage": "preprocess",
            "status": "BLOCKED",
            "blocked_reason": "MISSING_RAW_INPUT",
        },
        {
            "stage": "reconstruction",
            "status": "BLOCKED",
            "blocked_reason": "WAITING_LABELS",
        },
        {"stage": "preprocess", "status": "SUBMITTED"},
        {
            "stage": "preprocess",
            "status": "FAILED",
            "result_summary_json": json.dumps({
                "failure_category": "accuracy_check_failed",
                "failed_accuracy_checks": ["object_silhouette_alignment"],
            }),
        },
        {
            "stage": "reconstruction",
            "status": "SUCCEEDED",
            "reconstruction_run_status": "SUCCEEDED",
        },
        {
            "stage": "reconstruction",
            "status": "SUCCEEDED",
            "reconstruction_run_status": "SUCCEEDED",
            "qc_decision": "PASS",
        },
        {
            "stage": "reconstruction",
            "status": "SUCCEEDED",
            "reconstruction_run_status": "SUCCEEDED",
            "qc_decision": "FAIL",
        },
        {"stage": "export", "status": "UNKNOWN"},
        {"stage": "revalidation", "status": "PENDING"},
        {
            "stage": "revalidation",
            "status": "SUCCEEDED",
            "export_run_status": "SUCCEEDED",
            "export_authorization_type": "REVALIDATION",
        },
    ]
    summary = query.build_campaign_lifecycle_summary(_snapshot(rows))

    assert summary["reconciled_total"] == len(rows)
    assert sum(summary["buckets"].values()) == len(rows)
    assert summary["buckets"] == {
        "PREPROCESS_PENDING": 1,
        "MISSING_RAW_INPUT": 1,
        "WAITING_LABELS": 1,
        "PREPROCESS_RUNNING": 1,
        "PREPROCESS_FAILED": 1,
        "WAITING_QC": 1,
        "WAITING_EXPORT": 1,
        "QC_FAILED": 1,
        "EXPORT_UNKNOWN": 1,
        "REVALIDATION_PENDING": 1,
        "EXPORTED": 1,
    }
    assert summary["blocked_reasons"] == {
        "MISSING_RAW_INPUT": 1,
        "WAITING_LABELS": 1,
    }
    assert summary["failure_categories"] == {
        "accuracy_check_failed:object_silhouette_alignment": 1,
        "qc_failed": 1,
    }


def test_success_without_required_run_or_export_is_inconsistent():
    rows = [
        {"stage": "preprocess", "status": "SUCCEEDED"},
        {"stage": "reconstruction", "status": "SUCCEEDED"},
        {"stage": "revalidation", "status": "SUCCEEDED"},
        {"stage": "export", "status": "SUCCEEDED"},
    ]
    summary = query.build_campaign_lifecycle_summary(_snapshot(rows))
    assert summary["buckets"] == {"INCONSISTENT": 4}
    assert summary["has_inconsistent"]


def test_snapshot_exposes_exact_export_lineage_and_workflows(tmp_path: Path):
    path = _database(tmp_path)
    calibration = db.create_stage_run(
        sequence_name="calibration",
        dataset="dataset",
        stage=db.CALIBRATION_STAGE,
        pipeline_version="1.6.0",
        workflow_name="calibration-workflow",
        status="SUCCEEDED",
        source_uri="swift://recordings/calibration",
        output_uri="swift://output/calibration",
        db_path=path,
    )
    db.upsert_sequence(
        "dataset",
        "sequence",
        calibration_sequence_name="calibration",
        db_path=path,
    )
    preprocess = db.create_stage_run(
        sequence_name="sequence",
        dataset="dataset",
        stage=db.PREPROCESS_STAGE,
        pipeline_version="1.6.0",
        workflow_name="preprocess-workflow",
        status="SUCCEEDED",
        calibration_run_id=calibration["stage_run_id"],
        source_uri="swift://recordings/sequence",
        output_uri="swift://output/sequence",
        db_path=path,
    )
    reconstruction = db.create_stage_run(
        sequence_name="sequence",
        dataset="dataset",
        stage=db.RECONSTRUCTION_STAGE,
        pipeline_version="1.6.0",
        workflow_name="reconstruction-workflow",
        status="SUCCEEDED",
        preprocess_run_id=preprocess["stage_run_id"],
        output_uri="swift://output/sequence/reconstruction_1",
        db_path=path,
    )
    review_id = db.record_qc_review(
        reconstruction["stage_run_id"],
        "PASS",
        source="kratos",
        external_review_id="sequence.json",
        raw_payload={"annotations": []},
        db_path=path,
    )
    campaign = _campaign(
        path, "export-campaign", campaign_type="REMEDIATION", count=1,
    )
    request = _request(
        path, campaign, "sequence", stage="export",
    )
    exported = db.create_stage_run(
        sequence_name="sequence",
        dataset="dataset",
        stage=db.EXPORT_STAGE,
        pipeline_version="1.6.0",
        workflow_name="export-workflow",
        status="SUCCEEDED",
        reconstruction_run_id=reconstruction["stage_run_id"],
        qc_review_id=review_id,
        authorization_type="QC",
        source_uri=reconstruction["output_uri"],
        output_uri="swift://example/data_export_2/sequence",
        request_id=request["id"],
        db_path=path,
    )
    db.update_stage_request(request["id"], status="SUCCEEDED", db_path=path)

    row = db.get_campaign_lifecycle_snapshot(
        "dataset", campaign=campaign["name"], db_path=path,
    )["sequences"][0]

    assert row["calibration_sequence_name"] == "calibration"
    assert row["calibration_run_id"] == calibration["stage_run_id"]
    assert row["calibration_workflow"] == "calibration-workflow"
    assert row["preprocess_run_id"] == preprocess["stage_run_id"]
    assert row["preprocess_workflow"] == "preprocess-workflow"
    assert row["reconstruction_run_id"] == reconstruction["stage_run_id"]
    assert row["reconstruction_workflow"] == "reconstruction-workflow"
    assert row["qc_review_id"] == review_id
    assert row["qc_decision"] == "PASS"
    assert row["export_run_id"] == exported["stage_run_id"]
    assert row["export_workflow"] == "export-workflow"
    assert row["export_authorization_type"] == "QC"


@pytest.mark.parametrize("reuse_prior_campaign", [False, True])
def test_waiting_labels_snapshot_exposes_current_preprocess_lineage(
    tmp_path: Path, reuse_prior_campaign: bool,
):
    path = _database(tmp_path)
    target_campaign = _campaign(
        path, "reconstruction-target", campaign_type="BACKLOG_REPROCESSING",
    )
    source_campaign = (
        _campaign(
            path, "preprocess-source", campaign_type="BACKLOG_REPROCESSING",
        )
        if reuse_prior_campaign
        else target_campaign
    )
    preprocess_request = _request(
        path, source_campaign, "sequence", stage="preprocess",
    )
    preprocess = db.create_stage_run(
        sequence_name="sequence",
        dataset="dataset",
        stage=db.PREPROCESS_STAGE,
        pipeline_version="1.6.0",
        workflow_name="preprocess-workflow",
        status="SUCCEEDED",
        trigger="MIGRATION",
        output_uri="swift://output/sequence",
        request_id=preprocess_request["id"],
        db_path=path,
    )
    _set_request(path, preprocess_request, "SUCCEEDED")
    reconstruction_request = _request(
        path,
        target_campaign,
        "sequence",
        stage="reconstruction",
        status="BLOCKED",
        blocked_reason="WAITING_LABELS",
    )

    row = db.get_campaign_lifecycle_snapshot(
        "dataset", campaign=target_campaign["name"], db_path=path,
    )["sequences"][0]
    resolved = lifecycle.resolve_lifecycle_row(row)

    assert row["id"] == reconstruction_request["id"]
    assert resolved["lifecycle_status"] == "WAITING_LABELS"
    assert row["preprocess_run_id"] == preprocess["stage_run_id"]
    assert row["preprocess_run_status"] == "SUCCEEDED"
    assert row["preprocess_workflow"] == "preprocess-workflow"


def test_lifecycle_cli_is_read_only_and_pipeline_summary_path_is_preserved(
    monkeypatch,
):
    monkeypatch.setattr(
        query,
        "load_config",
        lambda: {"datasets": {"dataset": {}}},
    )
    monkeypatch.setattr(
        query,
        "init_db",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("lifecycle summary must not initialize or upgrade the DB")
        ),
    )
    monkeypatch.setattr(
        query,
        "refresh_workflow_states",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("lifecycle summary must not refresh OSMO")
        ),
    )
    seen = {}
    monkeypatch.setattr(
        query,
        "show_campaign_lifecycle_summary",
        lambda dataset, campaign=None, db_path=None: seen.update(
            dataset=dataset, campaign=campaign,
        ) or True,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["query.py", "--dataset", "dataset", "--campaign", "closed", "--summary"],
    )
    assert query.main() == 0
    assert seen == {"dataset": "dataset", "campaign": "closed"}

    called = []
    monkeypatch.setattr(query, "init_db", lambda *_args, **_kwargs: called.append("init"))
    monkeypatch.setattr(
        query,
        "refresh_workflow_states",
        lambda *_args, **_kwargs: called.append("refresh"),
    )
    monkeypatch.setattr(
        query,
        "show_summary",
        lambda *_args, **_kwargs: called.append("summary"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "query.py",
            "--dataset",
            "dataset",
            "--pipeline",
            "mv_hoi_reconstruction",
            "--summary",
        ],
    )
    assert query.main() == 0
    assert called == ["init", "refresh", "summary"]


def test_production_scale_snapshot_avoids_sequence_status_and_finishes_quickly(
    tmp_path: Path,
    monkeypatch,
):
    path = _database(tmp_path)
    size = 3_200
    campaign = _campaign(
        path,
        "large",
        campaign_type="BACKLOG_REPROCESSING",
        count=size,
    )
    conn = db.get_connection(path)
    try:
        conn.executemany(
            """INSERT INTO sequences(dataset, sequence_name, sequence_kind)
               VALUES ('dataset', ?, 'hoi')""",
            [(f"sequence-{index:04d}",) for index in range(size)],
        )
        sequence_ids = conn.execute(
            """SELECT id FROM sequences WHERE dataset='dataset'
               ORDER BY sequence_name"""
        ).fetchall()
        now = "2026-07-29T00:00:00.000Z"
        conn.executemany(
            """INSERT INTO stage_requests
               (sequence_id, campaign_id, cohort, stage, status, trigger,
                pipeline_version, created_at, updated_at)
               VALUES (?, ?, 'BULK', 'preprocess', 'PENDING', 'AUTO',
                       '1.6.0', ?, ?)""",
            [(row[0], campaign["id"], now, now) for row in sequence_ids],
        )
        conn.execute(
            """UPDATE stage_requests
               SET source_manifest_json=?, parameters_json=?
               WHERE id=(SELECT MIN(id) FROM stage_requests WHERE campaign_id=?)""",
            ("x" * 1_000_000, "y" * 100_000, campaign["id"]),
        )
        conn.commit()
    finally:
        conn.close()

    statements = []
    real_get_connection = db.get_connection

    class RecordingConnection:
        def __init__(self):
            self.connection = real_get_connection(path)

        def execute(self, statement, parameters=()):
            statements.append(statement)
            return self.connection.execute(statement, parameters)

        def close(self):
            self.connection.close()

    monkeypatch.setattr(
        db,
        "get_connection",
        lambda _db_path=path: RecordingConnection(),
    )
    started = time.monotonic()
    snapshot = db.get_campaign_lifecycle_snapshot("dataset", db_path=path)
    elapsed = time.monotonic() - started

    assert len(snapshot["sequences"]) == size
    assert elapsed < 5
    assert all("sequence_status" not in statement.lower() for statement in statements)
    lifecycle_sql = next(
        statement for statement in statements if "WITH ranked_requests AS" in statement
    )
    assert "sr.*" not in lifecycle_sql
    assert "source_manifest_json" not in lifecycle_sql
    assert "parameters_json" not in lifecycle_sql
    assert "source_manifest_json" not in snapshot["sequences"][0]
    assert "parameters_json" not in snapshot["sequences"][0]


def test_legacy_accuracy_limit_detail_does_not_require_failed_checks():
    detail = query._categorized_failure_detail(
        "task_failed: check_accuracy",
        {
            "failure_category": "accuracy_limit",
            "reason": "accuracy failure segments 12 > 10",
        },
    )
    assert detail == (
        "task_failed: check_accuracy; accuracy_failure: "
        "accuracy failure segments 12 > 10"
    )
