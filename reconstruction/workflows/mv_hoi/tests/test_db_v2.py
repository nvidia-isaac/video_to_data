import json
import sqlite3
import sys
from pathlib import Path

import pytest


WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

from orchestration import db


def _database(tmp_path: Path, name: str = "processing_v2.db") -> str:
    path = str(tmp_path / name)
    db.init_db(path)
    db.ensure_version_cached("1.0.0", db_path=path)
    return path


def _successful_calibration(path: str, name: str = "calib") -> dict:
    return db.create_stage_run(
        sequence_name=name,
        dataset="dataset",
        stage=db.CALIBRATION_STAGE,
        pipeline_version="1.0.0",
        workflow_name=f"calibration-{name}",
        status="SUCCEEDED",
        source_uri=f"swift://recordings/{name}",
        output_uri=f"swift://output/{name}",
        db_path=path,
    )


def _successful_preprocess(path: str, sequence: str = "seq") -> dict:
    calibration = _successful_calibration(path, f"calib-{sequence}")
    db.upsert_sequence(
        "dataset",
        sequence,
        calibration_sequence_name=f"calib-{sequence}",
        db_path=path,
    )
    return db.create_stage_run(
        sequence_name=sequence,
        dataset="dataset",
        stage=db.PREPROCESS_STAGE,
        pipeline_version="1.0.0",
        workflow_name=f"preprocess-{sequence}",
        status="SUCCEEDED",
        calibration_run_id=calibration["stage_run_id"],
        source_uri=f"swift://recordings/{sequence}",
        output_uri=f"swift://output/{sequence}/mv_preprocess",
        db_path=path,
    )


def _successful_reconstruction(path: str, sequence: str = "seq") -> dict:
    preprocess = _successful_preprocess(path, sequence)
    manifest = json.dumps([{"name": "left.json", "sha256": "abc"}])
    return db.create_stage_run(
        sequence_name=sequence,
        dataset="dataset",
        stage=db.RECONSTRUCTION_STAGE,
        pipeline_version="1.0.0",
        workflow_name=f"reconstruction-{sequence}",
        status="SUCCEEDED",
        preprocess_run_id=preprocess["stage_run_id"],
        labeled_bboxes_uri=f"swift://output/{sequence}/labeled_bboxes/",
        labeled_bboxes_manifest_json=manifest,
        labeled_bboxes_sha256="bbox-sha",
        hitl_item_id=f"reconstruction-{sequence}.json",
        output_uri=f"swift://output/{sequence}",
        db_path=path,
    )


def test_schema_contains_only_explicit_model_tables(tmp_path):
    path = _database(tmp_path)
    conn = db.get_connection(path)
    names = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        )
    }
    conn.close()

    assert {
        "sequences",
        "blacklisted_sequences",
        "pipeline_versions",
        "workflow_executions",
        "calibration_runs",
        "preprocess_runs",
        "reconstruction_runs",
        "qc_reviews",
        "export_runs",
        "sequence_status",
    } <= names
    assert {
        "pipelines",
        "stage_runs",
        "run_dependencies",
        "external_inputs",
        "export_approvals",
        "legacy_mappings",
    }.isdisjoint(names)


def test_current_head_and_downstream_supersession(tmp_path):
    path = _database(tmp_path)
    first_preprocess = _successful_preprocess(path)
    reconstruction = db.create_stage_run(
        sequence_name="seq",
        dataset="dataset",
        stage=db.RECONSTRUCTION_STAGE,
        pipeline_version="1.0.0",
        workflow_name="reconstruction-1",
        status="SUCCEEDED",
        preprocess_run_id=first_preprocess["stage_run_id"],
        db_path=path,
    )
    calibration = db.get_current_stage_run(
        "calib-seq", "dataset", db.CALIBRATION_STAGE, db_path=path
    )
    second_preprocess = db.create_stage_run(
        sequence_name="seq",
        dataset="dataset",
        stage=db.PREPROCESS_STAGE,
        pipeline_version="1.0.0",
        workflow_name="preprocess-2",
        status="SUCCEEDED",
        calibration_run_id=calibration["stage_run_id"],
        db_path=path,
    )

    assert db.get_current_stage_run(
        "seq", "dataset", db.PREPROCESS_STAGE, db_path=path
    )["stage_run_id"] == second_preprocess["stage_run_id"]
    assert db.get_current_stage_run(
        "seq", "dataset", db.RECONSTRUCTION_STAGE, db_path=path
    ) is None
    history = db.list_stage_run_history(
        "dataset", stage=db.RECONSTRUCTION_STAGE, db_path=path
    )
    assert history[0]["stage_run_id"] == reconstruction["stage_run_id"]
    assert history[0]["is_current"] == 0


def test_direct_dependency_and_export_authorization_validation(tmp_path):
    path = _database(tmp_path)
    reconstruction = _successful_reconstruction(path)
    review_id = db.record_qc_review(
        reconstruction["stage_run_id"],
        "PASS",
        source="kratos",
        external_review_id="item.json",
        raw_payload={"annotations": []},
        db_path=path,
    )
    export = db.create_stage_run(
        sequence_name="seq",
        dataset="dataset",
        stage=db.EXPORT_STAGE,
        pipeline_version="1.0.0",
        workflow_name="export-batch",
        status="SUCCEEDED",
        reconstruction_run_id=reconstruction["stage_run_id"],
        qc_review_id=review_id,
        authorization_type="QC",
        source_uri="swift://output/seq",
        output_uri="swift://export/seq",
        db_path=path,
    )
    conn = db.get_connection(path)
    stored = conn.execute(
        "SELECT reconstruction_run_id, qc_review_id, authorization_type "
        "FROM export_runs WHERE id=?",
        (export["stage_run_id"],),
    ).fetchone()
    conn.close()
    assert tuple(stored) == (reconstruction["stage_run_id"], review_id, "QC")

    with pytest.raises(ValueError, match="passing review"):
        db.create_stage_run(
            sequence_name="seq",
            dataset="dataset",
            stage=db.EXPORT_STAGE,
            pipeline_version="1.0.0",
            workflow_name="bad-export",
            status="SUCCEEDED",
            reconstruction_run_id=reconstruction["stage_run_id"],
            authorization_type="QC",
            db_path=path,
        )


def test_manual_override_requires_requester_and_reason(tmp_path):
    path = _database(tmp_path)
    reconstruction = _successful_reconstruction(path)
    with pytest.raises(sqlite3.IntegrityError):
        db.create_stage_run(
            sequence_name="seq",
            dataset="dataset",
            stage=db.EXPORT_STAGE,
            pipeline_version="1.0.0",
            workflow_name="manual-bad",
            status="SUCCEEDED",
            reconstruction_run_id=reconstruction["stage_run_id"],
            authorization_type="MANUAL_OVERRIDE",
            requested_by="operator",
            override_reason="",
            db_path=path,
        )

    run = db.create_stage_run(
        sequence_name="seq",
        dataset="dataset",
        stage=db.EXPORT_STAGE,
        pipeline_version="1.0.0",
        workflow_name="manual-good",
        status="SUCCEEDED",
        reconstruction_run_id=reconstruction["stage_run_id"],
        authorization_type="MANUAL_OVERRIDE",
        requested_by="operator",
        override_reason="approved one-off export",
        db_path=path,
    )
    assert run["authorization_type"] == "MANUAL_OVERRIDE"


def test_export_batch_execution_is_shared(tmp_path):
    path = _database(tmp_path)
    reconstructions = [_successful_reconstruction(path, sequence) for sequence in ("a", "b")]
    execution = db.create_workflow_execution(
        workflow_name="batch",
        osmo_workflow_id="batch-1",
        pipeline_type=db.EXPORT_STAGE,
        pipeline_version="1.0.0",
        db_path=path,
    )
    runs = [
        db.create_stage_run(
            sequence_name=sequence,
            dataset="dataset",
            stage=db.EXPORT_STAGE,
            pipeline_version="1.0.0",
            execution_id=execution,
            status="RUNNING",
            reconstruction_run_id=reconstruction["stage_run_id"],
            authorization_type="MANUAL_OVERRIDE",
            requested_by="operator",
            override_reason="batch export",
            db_path=path,
        )
        for sequence, reconstruction in zip(("a", "b"), reconstructions)
    ]
    assert {run["execution_id"] for run in runs} == {execution}
    assert len(db.get_workflows_by_export_id("dataset", "batch-1", db_path=path)) == 2


def test_ambiguous_reservation_blocks_duplicate_run(tmp_path):
    path = _database(tmp_path)
    first = db.create_stage_run(
        sequence_name="calib",
        dataset="dataset",
        stage=db.CALIBRATION_STAGE,
        pipeline_version="1.0.0",
        workflow_name="ambiguous-1",
        status="UNKNOWN",
        db_path=path,
    )
    with pytest.raises(sqlite3.IntegrityError):
        db.create_stage_run(
            sequence_name="calib",
            dataset="dataset",
            stage=db.CALIBRATION_STAGE,
            pipeline_version="1.0.0",
            workflow_name="ambiguous-2",
            status="SUBMITTING",
            db_path=path,
        )
    assert db.get_current_stage_run(
        "calib", "dataset", db.CALIBRATION_STAGE, db_path=path
    )["stage_run_id"] == first["stage_run_id"]


def test_failure_circuit_breaker_is_stage_scoped_and_persistent(tmp_path):
    path = _database(tmp_path)
    first = db.create_stage_run(
        sequence_name="calib",
        dataset="dataset",
        stage=db.CALIBRATION_STAGE,
        pipeline_version="1.0.0",
        workflow_name="failure-1",
        status="FAILED",
        details=" task_failed:   calibrate ",
        db_path=path,
    )
    assert not db.maybe_blacklist_repeated_failure(
        first["stage_run_id"], stage=db.CALIBRATION_STAGE, db_path=path
    )
    second = db.create_stage_run(
        sequence_name="calib",
        dataset="dataset",
        stage=db.CALIBRATION_STAGE,
        pipeline_version="1.0.0",
        workflow_name="failure-2",
        status="FAILED",
        details="task_failed: calibrate",
        db_path=path,
    )
    assert db.maybe_blacklist_repeated_failure(
        second["stage_run_id"], stage=db.CALIBRATION_STAGE, db_path=path
    )
    blacklist = db.get_blacklisted_sequence("dataset", "calib", db_path=path)
    assert blacklist["reason"] == "task_failed: calibrate"
    assert blacklist["created_by"] == "automatic_failure_circuit_breaker"


def test_sequence_status_and_history_are_wide_and_immutable(tmp_path):
    path = _database(tmp_path)
    reconstruction = _successful_reconstruction(path)
    db.record_qc_review(
        reconstruction["stage_run_id"], "PASS", source="kratos", db_path=path
    )
    status = db.get_sequence_status("dataset", "seq", db_path=path)[0]
    assert status["preprocess_status"] == "SUCCEEDED"
    assert status["reconstruction_status"] == "SUCCEEDED"
    assert status["qc_decision"] == "PASS"
    assert status["export_status"] == "READY"
    history = db.get_sequence_history("dataset", "seq", db_path=path)
    assert {item["record_type"] for item in history} == {
        "stage_run", "workflow_execution", "qc_review"
    }


def test_successful_current_export_remains_effective_after_failed_revalidation(tmp_path):
    path = _database(tmp_path)
    reconstruction = _successful_reconstruction(path)
    db.create_stage_run(
        sequence_name="seq", dataset="dataset", stage=db.EXPORT_STAGE,
        pipeline_version="1.0.0", workflow_name="legacy-export",
        status="SUCCEEDED", trigger="MIGRATION",
        reconstruction_run_id=reconstruction["stage_run_id"],
        authorization_type="LEGACY", output_uri="swift://legacy/data_export/seq",
        db_path=path,
    )
    legacy_status = db.get_sequence_status("dataset", "seq", db_path=path)[0]
    assert legacy_status["export_status"] == "SUCCEEDED"
    campaign = db.create_campaign(
        name="revalidation", campaign_type="LEGACY_REVALIDATION",
        dataset="dataset", pipeline_version="1.0.0",
        output_uri="swift://new/data_export_2", created_by="operator",
        db_path=path,
    )
    campaign = db.freeze_campaign(
        campaign["id"], inventory_uri="swift://inventory.json",
        inventory_sha256="a" * 64, configuration_uri="swift://config.json",
        configuration_sha256="b" * 64, db_path=path,
    )
    request = db.create_stage_request(
        sequence_name="seq", dataset="dataset", stage="revalidation",
        pipeline_version="1.0.0", campaign=campaign["id"], cohort="CANARY",
        db_path=path,
    )
    db.update_stage_request(
        request["id"], status="FAILED", details="comparison gate failed",
        db_path=path,
    )

    status = db.get_sequence_status("dataset", "seq", db_path=path)[0]

    assert status["export_run_status"] == "SUCCEEDED"
    assert status["revalidation_request_status"] == "FAILED"
    assert status["latest_request_id"] == request["id"]
    assert status["effective_status"] == "SUCCEEDED"
    assert status["effective_stage"] == "export"
    assert status["effective_details"] is None


def test_failed_qc_immediately_blacklists_and_later_pass_does_not_clear(tmp_path):
    path = _database(tmp_path)
    reconstruction = _successful_reconstruction(path)
    db.record_qc_review(
        reconstruction["stage_run_id"],
        "FAIL",
        details="qc_fail: failure_coverage>50%",
        source="kratos",
        external_review_id="item.json",
        raw_payload={"revision": 1},
        db_path=path,
    )
    assert db.get_blacklisted_sequence("dataset", "seq", db_path=path)["created_by"] == "automatic_qc"
    db.record_qc_review(
        reconstruction["stage_run_id"],
        "PASS",
        source="kratos",
        external_review_id="item.json",
        raw_payload={"revision": 2},
        db_path=path,
    )
    assert db.get_blacklisted_sequence("dataset", "seq", db_path=path)


def test_late_qc_for_superseded_reconstruction_cannot_authorize_export(tmp_path):
    path = _database(tmp_path)
    first = _successful_reconstruction(path)
    review_id = db.record_qc_review(
        first["stage_run_id"], "PASS", source="kratos", db_path=path
    )
    preprocess = db.get_current_stage_run(
        "seq", "dataset", db.PREPROCESS_STAGE, db_path=path
    )
    db.create_stage_run(
        sequence_name="seq",
        dataset="dataset",
        stage=db.RECONSTRUCTION_STAGE,
        pipeline_version="1.0.0",
        workflow_name="reconstruction-new",
        status="SUCCEEDED",
        preprocess_run_id=preprocess["stage_run_id"],
        db_path=path,
    )
    with pytest.raises(ValueError, match="not current and successful"):
        db.create_stage_run(
            sequence_name="seq",
            dataset="dataset",
            stage=db.EXPORT_STAGE,
            pipeline_version="1.0.0",
            workflow_name="late-export",
            status="SUCCEEDED",
            reconstruction_run_id=first["stage_run_id"],
            qc_review_id=review_id,
            authorization_type="QC",
            db_path=path,
        )


def test_production_and_test_databases_are_isolated(tmp_path):
    production = _database(tmp_path, "production.db")
    test = _database(tmp_path, "test.db")
    _successful_calibration(production, "production-only")
    assert db.get_sequence_status("dataset", db_path=production)
    assert db.get_sequence_status("dataset", db_path=test) == []


def test_local_execution_records_backend_and_has_no_osmo_id(tmp_path):
    path = _database(tmp_path)
    run = db.create_stage_run(
        sequence_name="calib-local",
        dataset="dataset",
        stage=db.CALIBRATION_STAGE,
        pipeline_version="1.0.0",
        workflow_name="local-calibration",
        backend="local",
        workflow_spec_path="local/calibration.yaml",
        status="SUCCEEDED",
        db_path=path,
    )
    conn = db.get_connection(path)
    execution = conn.execute(
        "SELECT pipeline_stage, backend, osmo_workflow_id, workflow_spec_path "
        "FROM workflow_executions WHERE id=?",
        (run["execution_id"],),
    ).fetchone()
    conn.close()
    assert tuple(execution) == (
        "calibration", "local", None, "local/calibration.yaml"
    )


def test_init_refuses_in_place_legacy_mutation(tmp_path):
    path = tmp_path / "processing.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE pipelines(id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError, match="migrate_processing_db.py"):
        db.init_db(str(path))
