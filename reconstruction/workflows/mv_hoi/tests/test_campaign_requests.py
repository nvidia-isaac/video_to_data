from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

from orchestration import campaign as campaign_cli
from orchestration import campaign_report, db, database


def _database(tmp_path: Path) -> str:
    path = str(tmp_path / "orchestration.db")
    db.init_db(path)
    return path


def _legacy_v2_database(path: str) -> None:
    """Create the pre-Alembic v2 baseline from canonical migration metadata."""
    from alembic import command

    command.upgrade(database.alembic_config(path), "0001_v2_baseline")
    conn = db.get_connection(path)
    try:
        conn.execute("DROP TABLE alembic_version")
        conn.commit()
    finally:
        conn.close()


def _campaign(path: str, *, campaign_type: str = "LEGACY_REVALIDATION") -> dict:
    campaign = db.create_campaign(
        name="campaign-1",
        campaign_type=campaign_type,
        dataset="dataset",
        pipeline_version="1.6.0",
        output_uri="swift://example/data_export_2/",
        created_by="operator",
        db_path=path,
    )
    return db.freeze_campaign(
        campaign["id"],
        inventory_uri="swift://example/manifests/inventory.json",
        inventory_sha256="a" * 64,
        configuration_uri="swift://example/manifests/config.json",
        configuration_sha256="b" * 64,
        db_path=path,
    )


def test_existing_v2_is_stamped_and_upgraded_without_row_changes(tmp_path: Path):
    path = str(tmp_path / "existing.db")
    _legacy_v2_database(path)
    db.ensure_version_cached("1.6.0", db_path=path)
    sequence_id = db.upsert_sequence("dataset", "sequence", db_path=path)

    db.init_db(path)

    conn = db.get_connection(path)
    try:
        assert conn.execute("SELECT version_num FROM alembic_version").fetchone()[0] == (
            database.ALEMBIC_HEAD
        )
        assert len(database.ALEMBIC_HEAD) <= 32
        assert conn.execute("SELECT id FROM sequences").fetchone()[0] == sequence_id
        assert not conn.execute("PRAGMA foreign_key_check").fetchall()
        assert "request_id" in {
            row[1] for row in conn.execute("PRAGMA table_info(reconstruction_runs)")
        }
        assert "idx_stage_requests_campaign_sequence_recency" in {
            row[1] for row in conn.execute("PRAGMA index_list(stage_requests)")
        }
        assert "intermediate_cleanup_jobs" in {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    finally:
        conn.close()


def test_existing_campaign_database_upgrades_inventory_count_in_place(tmp_path: Path):
    from alembic import command

    path = _database(tmp_path)
    command.downgrade(database.alembic_config(path), "0003_effective_sequence_status")
    conn = db.get_connection(path)
    try:
        assert "inventory_sequence_count" not in {
            row[1] for row in conn.execute("PRAGMA table_info(processing_campaigns)")
        }
    finally:
        conn.close()

    db.init_db(path)

    conn = db.get_connection(path)
    try:
        assert conn.execute("SELECT version_num FROM alembic_version").fetchone()[0] == (
            database.ALEMBIC_HEAD
        )
        assert "inventory_sequence_count" in {
            row[1] for row in conn.execute("PRAGMA table_info(processing_campaigns)")
        }
    finally:
        conn.close()


def test_sqlite_batch_upgrade_preserves_runs_referencing_workflow_executions(
    tmp_path: Path,
):
    path = str(tmp_path / "referenced.db")
    _legacy_v2_database(path)
    conn = db.get_connection(path)
    try:
        conn.execute(
            "INSERT INTO pipeline_versions(version, message) VALUES ('1.6.1', 'release')"
        )
        sequence_id = conn.execute(
            "INSERT INTO sequences(dataset, sequence_name, sequence_kind) "
            "VALUES ('dataset', 'sequence', 'calibration')"
        ).lastrowid
        execution_id = conn.execute(
            "INSERT INTO workflow_executions(pipeline_stage, pipeline_version, "
            "workflow_name, status) VALUES "
            "('calibration', '1.6.1', 'referenced-workflow', 'SUCCEEDED')"
        ).lastrowid
        run_id = conn.execute(
            "INSERT INTO calibration_runs(sequence_id, workflow_execution_id, "
            "pipeline_version, status, trigger, is_current) VALUES "
            "(?, ?, '1.6.1', 'SUCCEEDED', 'MIGRATION', 1)",
            (sequence_id, execution_id),
        ).lastrowid
        conn.commit()
    finally:
        conn.close()

    db.init_db(path)

    conn = db.get_connection(path)
    try:
        assert conn.execute("SELECT version_num FROM alembic_version").fetchone()[0] == (
            database.ALEMBIC_HEAD
        )
        assert conn.execute(
            "SELECT workflow_execution_id FROM calibration_runs WHERE id=?",
            (run_id,),
        ).fetchone()[0] == execution_id
        assert not conn.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        conn.close()


def test_database_url_and_path_resolve_to_same_sqlite_backend(tmp_path: Path, monkeypatch):
    path = tmp_path / "url.db"
    monkeypatch.setenv("MV_HOI_DATABASE_URL", f"sqlite+pysqlite:///{path}")
    assert database.database_target() == f"sqlite+pysqlite:///{path}"
    assert database.sqlite_path(database.database_target()) == str(path)
    db.init_db(database.database_target())
    db.ensure_version_cached("1.2.3", db_path=database.database_target())
    assert db.get_latest_version(db_path=str(path)) == "1.2.3"


def test_production_url_does_not_replace_separate_test_database(tmp_path: Path, monkeypatch):
    production = tmp_path / "production.db"
    testing = tmp_path / "testing.db"
    monkeypatch.setenv("MV_HOI_DATABASE_URL", f"sqlite+pysqlite:///{production}")
    monkeypatch.setenv("MV_HOI_TEST_DB_PATH", str(testing))
    assert database.database_target(test=False).endswith("production.db")
    assert database.database_target(test=True) == str(testing)


def test_postgresql_database_url_uses_psycopg():
    assert database.normalize_database_url(
        "postgresql://operator@example/db"
    ) == "postgresql+psycopg://operator@example/db"


def test_unsupported_database_url_is_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        database.normalize_database_url("mysql://example/db")


def test_campaign_freeze_is_immutable(tmp_path: Path):
    path = _database(tmp_path)
    campaign = _campaign(path)
    assert campaign["status"] == "FROZEN"
    assert campaign["phase"] == "CANARY"
    same = db.freeze_campaign(
        campaign["id"],
        inventory_uri=campaign["inventory_uri"],
        inventory_sha256=campaign["inventory_sha256"],
        configuration_uri=campaign["configuration_uri"],
        configuration_sha256=campaign["configuration_sha256"],
        db_path=path,
    )
    assert same["id"] == campaign["id"]
    with pytest.raises(ValueError, match="immutable"):
        db.freeze_campaign(
            campaign["id"],
            inventory_uri="swift://example/other.json",
            inventory_sha256="c" * 64,
            configuration_uri=campaign["configuration_uri"],
            configuration_sha256=campaign["configuration_sha256"],
            db_path=path,
        )


def test_campaign_completion_reconciles_frozen_inventory_count(tmp_path: Path):
    path = _database(tmp_path)
    campaign = _campaign(path, campaign_type="BACKLOG_REPROCESSING")
    db.set_campaign_inventory_count(campaign["id"], 2, db_path=path)
    request = db.create_stage_request(
        sequence_name="only-one", dataset="dataset", stage="preprocess",
        pipeline_version="1.6.0", campaign=campaign["id"], cohort="BULK",
        db_path=path,
    )
    db.update_stage_request(request["id"], status="FAILED", db_path=path)

    with pytest.raises(ValueError, match="frozen inventory count"):
        db.finish_campaign(campaign["id"], db_path=path)


def test_request_manifest_reservation_and_ambiguous_duplicate_guard(tmp_path: Path):
    path = _database(tmp_path)
    campaign = _campaign(path)
    manifest = {"objects": [{"key": "b", "etag": "2"}, {"key": "a", "etag": "1"}]}
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    request = db.create_stage_request(
        sequence_name="sequence",
        dataset="dataset",
        stage="revalidation",
        pipeline_version="1.6.0",
        campaign=campaign["id"],
        cohort="CANARY",
        source_manifest=manifest,
        db_path=path,
    )
    assert request["source_manifest_sha256"] == hashlib.sha256(
        canonical.encode()
    ).hexdigest()
    with pytest.raises(Exception, match="UNIQUE constraint failed"):
        db.create_stage_request(
            sequence_name="sequence",
            dataset="dataset",
            stage="revalidation",
            pipeline_version="1.6.0",
            campaign=campaign["id"],
            cohort="CANARY",
            db_path=path,
        )

    reserved = db.reserve_stage_request(
        request["id"], reserved_by="host", db_path=path,
    )
    assert reserved["id"] == request["id"]
    execution_id = db.create_workflow_execution(
        workflow_name="workflow", pipeline_type=db.REVALIDATION_STAGE,
        pipeline_version="1.6.0", db_path=path,
    )
    attached = db.attach_request_execution(
        request["id"], execution_id, status="UNKNOWN", db_path=path,
    )
    assert attached["status"] == "UNKNOWN"
    assert db.reserve_stage_request(
        request["id"], reserved_by="host", db_path=path,
    ) is None


def test_execution_observation_updates_execution_run_and_request_atomically(
    tmp_path: Path,
):
    path = _database(tmp_path)
    request = db.create_stage_request(
        sequence_name="calibration", dataset="dataset", stage="calibration",
        pipeline_version="1.6.0", db_path=path,
    )
    reservation = db.reserve_request_with_execution(
        request["id"], reserved_by="dispatcher", workflow_name="calibration-wf",
        pipeline_version="1.6.0", workflow_spec_path="calibration.yaml",
        pool="pool",
        details='submit_reserved; pool_selection={"selected":"pool"}',
        db_path=path,
    )
    run = db.create_stage_run(
        sequence_name="calibration", dataset="dataset", stage=db.CALIBRATION_STAGE,
        pipeline_version="1.6.0", status="SUBMITTING",
        execution_id=reservation["execution_id"], request_id=request["id"],
        set_current=False, db_path=path,
    )

    db.apply_execution_observation(
        reservation["execution_id"], execution_status="RUNNING",
        run_outcomes=[{
            "run_id": run["stage_run_id"], "stage": db.CALIBRATION_STAGE,
            "status": "RUNNING", "set_current": True,
        }], db_path=path,
    )
    assert db.get_workflow_execution(
        reservation["execution_id"], db_path=path,
    )["status"] == "RUNNING"
    assert "pool_selection=" in db.get_workflow_execution(
        reservation["execution_id"], db_path=path,
    )["details"]
    db.update_workflow_execution(
        reservation["execution_id"], details="backend poll",
        osmo_workflow_id="calibration-wf-1", query_payload={"status": "RUNNING"},
        db_path=path,
    )
    polled = db.get_workflow_execution(
        reservation["execution_id"], db_path=path,
    )
    assert polled["details"].startswith("backend poll; ")
    assert 'pool_selection={"selected":"pool"}' in polled["details"]
    assert polled["osmo_workflow_id"] == "calibration-wf-1"
    assert db.get_stage_run(
        run["stage_run_id"], stage=db.CALIBRATION_STAGE, db_path=path,
    )["run_status"] == "RUNNING"
    assert db.get_stage_request(request["id"], db_path=path)["status"] == "RUNNING"

    with pytest.raises(KeyError):
        db.apply_execution_observation(
            reservation["execution_id"], execution_status="FAILED",
            run_outcomes=[{"run_id": 999999, "status": "FAILED"}], db_path=path,
        )
    assert db.get_workflow_execution(
        reservation["execution_id"], db_path=path,
    )["status"] == "RUNNING"

    db.apply_execution_observation(
        reservation["execution_id"], execution_status="FAILED",
        details="calibration failed", run_outcomes=[{
            "run_id": run["stage_run_id"], "stage": db.CALIBRATION_STAGE,
            "status": "FAILED",
        }], db_path=path,
    )
    execution = db.get_workflow_execution(
        reservation["execution_id"], db_path=path,
    )
    assert execution["details"].startswith("calibration failed; ")
    assert 'pool_selection={"selected":"pool"}' in execution["details"]
    with pytest.raises(ValueError, match="create a new request/attempt"):
        db.apply_execution_observation(
            reservation["execution_id"], execution_status="RUNNING",
            run_outcomes=[{
                "run_id": run["stage_run_id"], "stage": db.CALIBRATION_STAGE,
                "status": "RUNNING",
            }], db_path=path,
        )


def test_blacklist_blocks_automatic_request_reservation(tmp_path: Path):
    path = _database(tmp_path)
    request = db.create_stage_request(
        sequence_name="sequence", dataset="dataset", stage="preprocess",
        pipeline_version="1.6.0", db_path=path,
    )
    db.upsert_blacklisted_sequence(
        "dataset", "sequence", reason="timeout box", db_path=path,
    )
    assert db.reserve_stage_request(
        request["id"], reserved_by="host", db_path=path,
    ) is None


@pytest.mark.parametrize(
    ("campaign_type", "stage", "cohort"),
    [
        ("LEGACY_REVALIDATION", "revalidation", "CANARY"),
        ("BACKLOG_REPROCESSING", "preprocess", "BULK"),
        ("REMEDIATION", "preprocess", "BULK"),
    ],
)
def test_frozen_campaign_membership_bypasses_existing_blacklist(
    tmp_path: Path, campaign_type: str, stage: str, cohort: str,
):
    path = _database(tmp_path)
    campaign = _campaign(path, campaign_type=campaign_type)
    request = db.create_stage_request(
        sequence_name="sequence", dataset="dataset", stage=stage,
        pipeline_version="1.6.0", campaign=campaign["id"], cohort=cohort,
        db_path=path,
    )
    db.upsert_blacklisted_sequence(
        "dataset", "sequence", reason="legacy timeout box", db_path=path,
    )

    reserved = db.reserve_stage_request(
        request["id"], reserved_by="campaign-dispatcher", db_path=path,
    )

    assert reserved is not None
    assert reserved["status"] == "RESERVED"
    assert db.get_blacklisted_sequence(
        "dataset", "sequence", db_path=path,
    )["reason"] == "legacy timeout box"


def test_canary_approval_requires_verified_success(tmp_path: Path):
    path = _database(tmp_path)
    campaign = _campaign(path)
    request = db.create_stage_request(
        sequence_name="sequence", dataset="dataset", stage="revalidation",
        pipeline_version="1.6.0", campaign=campaign["id"], cohort="CANARY",
        db_path=path,
    )
    with pytest.raises(ValueError, match="terminal before approval"):
        db.approve_campaign_canary(
            campaign["id"], report_uri="swift://report", report_sha256="c" * 64,
            approved_by="operator", db_path=path,
        )
    db.update_stage_request(
        request["id"], status="SUCCEEDED",
        result_manifest_uri="swift://result", result_manifest_sha256="d" * 64,
        db_path=path,
    )
    approved = db.approve_campaign_canary(
        campaign["id"], report_uri="swift://report", report_sha256="c" * 64,
        approved_by="operator", db_path=path,
    )
    assert approved["phase"] == "BULK"
    assert approved["canary_approved_by"] == "operator"


def test_paired_canary_approval_is_atomic(tmp_path: Path):
    path = _database(tmp_path)
    campaigns = []
    requests = []
    for index in range(2):
        campaign = db.create_campaign(
            name=f"campaign-{index}", campaign_type="LEGACY_REVALIDATION",
            dataset="dataset", pipeline_version="1.6.0",
            output_uri="swift://example/data_export_3/", created_by="rollout",
            db_path=path,
        )
        campaign = db.freeze_campaign(
            campaign["id"], inventory_uri=f"swift://inventory-{index}",
            inventory_sha256=("a" if index == 0 else "b") * 64,
            configuration_uri=f"swift://configuration-{index}",
            configuration_sha256=("c" if index == 0 else "d") * 64,
            db_path=path,
        )
        request = db.create_stage_request(
            sequence_name=f"sequence-{index}", dataset="dataset",
            stage="revalidation", pipeline_version="1.6.0",
            campaign=campaign["id"], cohort="CANARY", db_path=path,
        )
        campaigns.append(campaign)
        requests.append(request)
    db.update_stage_request(
        requests[0]["id"], status="SUCCEEDED",
        result_manifest_uri="swift://result-0", result_manifest_sha256="e" * 64,
        db_path=path,
    )
    approvals = [{
        "campaign": campaign["id"], "report_uri": f"swift://report-{index}",
        "report_sha256": "f" * 64, "approved_by": "accuracy-segment-rollout",
    } for index, campaign in enumerate(campaigns)]

    with pytest.raises(ValueError, match="terminal before approval"):
        db.approve_paired_campaign_canaries(approvals, db_path=path)
    assert all(
        db.get_campaign(campaign["id"], db_path=path)["phase"] == "CANARY"
        for campaign in campaigns
    )

    db.update_stage_request(
        requests[1]["id"], status="SUCCEEDED",
        result_manifest_uri="swift://result-1", result_manifest_sha256="1" * 64,
        db_path=path,
    )
    promoted = db.approve_paired_campaign_canaries(approvals, db_path=path)
    assert [item["phase"] for item in promoted] == ["BULK", "BULK"]


def test_patch_replacement_adopts_only_identical_successful_transferred_preprocess(
    tmp_path: Path,
):
    path = _database(tmp_path)
    db.ensure_version_cached("1.6.35", db_path=path)
    db.ensure_version_cached("1.6.36", db_path=path)
    manifest = {
        "sequence": "transferred",
        "route": "reprocessing_missing_revalidation_input",
        "membership_transfer": {"source_campaign": "legacy_revalidation"},
        "historical_preprocess_lineage": {"reuse_permitted": False},
    }

    source = db.create_campaign(
        name="source", campaign_type="BACKLOG_REPROCESSING", dataset="dataset",
        pipeline_version="1.6.35", output_uri="swift://example/data_export_3",
        created_by="accuracy-segment-rollout", canary=True, db_path=path,
    )
    source = db.freeze_campaign(
        source["id"], inventory_uri="swift://inventory-v1635",
        inventory_sha256="a" * 64, inventory_sequence_count=3456,
        configuration_uri="swift://config-v1635",
        configuration_sha256="b" * 64, db_path=path,
    )
    source_request = db.create_stage_request(
        sequence_name="transferred", dataset="dataset", stage="preprocess",
        pipeline_version="1.6.35", campaign=source["id"], cohort="CANARY",
        queue_priority=100, source_manifest=manifest, db_path=path,
    )
    db.update_stage_request(source_request["id"], status="SUCCEEDED", db_path=path)
    source_run = db.create_stage_run(
        sequence_name="transferred", dataset="dataset", stage=db.PREPROCESS_STAGE,
        pipeline_version="1.6.35", status="SUCCEEDED",
        output_uri="swift://example/data_output/transferred",
        request_id=source_request["id"], trigger="MIGRATION",
        workflow_name="completed-preprocess", db_path=path,
    )
    db.cancel_campaign(source["id"], db_path=path)

    replacement = db.create_campaign(
        name="replacement", campaign_type="BACKLOG_REPROCESSING",
        dataset="dataset", pipeline_version="1.6.36",
        output_uri="swift://example/data_export_3",
        created_by="accuracy-segment-rollout", canary=True, db_path=path,
    )
    replacement = db.freeze_campaign(
        replacement["id"], inventory_uri="swift://inventory-v1636",
        inventory_sha256="a" * 64, inventory_sequence_count=3456,
        configuration_uri="swift://config-v1636",
        configuration_sha256="c" * 64, db_path=path,
    )
    target = db.create_stage_request(
        sequence_name="transferred", dataset="dataset", stage="preprocess",
        pipeline_version="1.6.36", campaign=replacement["id"], cohort="CANARY",
        queue_priority=100, source_manifest=manifest,
        parameters={"route": manifest["route"]}, db_path=path,
    )

    adopted = db.adopt_valid_replacement_preprocess_results(
        source["id"], replacement["id"],
        adopted_by="accuracy-segment-rollout", db_path=path,
    )

    assert adopted["adopted_count"] == 1
    assert adopted["adopted"][0]["preprocess_run_id"] == source_run["id"]
    target = db.get_stage_request(target["id"], db_path=path)
    assert target["status"] == "SUCCEEDED"
    assert target["workflow_execution_id"] is None
    evidence = json.loads(target["result_summary_json"])["preprocess_adoption"]
    assert evidence["source_request_id"] == source_request["id"]
    assert json.loads(target["parameters_json"])["preprocess_adoption"] == evidence

    repeated = db.adopt_valid_replacement_preprocess_results(
        source["id"], replacement["id"],
        adopted_by="accuracy-segment-rollout", db_path=path,
    )
    assert repeated["adopted_count"] == 0
    assert repeated["already_adopted_count"] == 1


def test_patch_replacement_refuses_nonidentical_transferred_manifest(tmp_path: Path):
    path = _database(tmp_path)
    db.ensure_version_cached("1.6.35", db_path=path)
    db.ensure_version_cached("1.6.36", db_path=path)
    source = db.create_campaign(
        name="source", campaign_type="BACKLOG_REPROCESSING", dataset="dataset",
        pipeline_version="1.6.35", output_uri="swift://example/data_export_3",
        created_by="rollout", canary=True, db_path=path,
    )
    source = db.freeze_campaign(
        source["id"], inventory_uri="swift://source", inventory_sha256="a" * 64,
        inventory_sequence_count=1, configuration_uri="swift://source-config",
        configuration_sha256="b" * 64, db_path=path,
    )
    source_manifest = {
        "route": "reprocessing_missing_revalidation_input",
        "membership_transfer": {"source_campaign": "legacy_revalidation"},
        "historical_preprocess_lineage": {"reuse_permitted": False},
    }
    source_request = db.create_stage_request(
        sequence_name="sequence", dataset="dataset", stage="preprocess",
        pipeline_version="1.6.35", campaign=source["id"], cohort="CANARY",
        queue_priority=100, source_manifest=source_manifest, db_path=path,
    )
    db.update_stage_request(source_request["id"], status="SUCCEEDED", db_path=path)
    db.create_stage_run(
        sequence_name="sequence", dataset="dataset", stage=db.PREPROCESS_STAGE,
        pipeline_version="1.6.35", status="SUCCEEDED",
        output_uri="swift://example/data_output/sequence",
        request_id=source_request["id"], trigger="MIGRATION",
        workflow_name="completed-preprocess", db_path=path,
    )
    db.cancel_campaign(source["id"], db_path=path)
    replacement = db.create_campaign(
        name="replacement", campaign_type="BACKLOG_REPROCESSING",
        dataset="dataset", pipeline_version="1.6.36",
        output_uri="swift://example/data_export_3", created_by="rollout",
        canary=True, db_path=path,
    )
    replacement = db.freeze_campaign(
        replacement["id"], inventory_uri="swift://replacement",
        inventory_sha256="a" * 64, inventory_sequence_count=1,
        configuration_uri="swift://replacement-config",
        configuration_sha256="c" * 64, db_path=path,
    )
    db.create_stage_request(
        sequence_name="sequence", dataset="dataset", stage="preprocess",
        pipeline_version="1.6.36", campaign=replacement["id"], cohort="CANARY",
        queue_priority=100, source_manifest={**source_manifest, "changed": True},
        db_path=path,
    )

    with pytest.raises(ValueError, match="not adoptable"):
        db.adopt_valid_replacement_preprocess_results(
            source["id"], replacement["id"], adopted_by="rollout", db_path=path,
        )


def test_patch_replacement_reuses_preprocess_already_adopted_by_source(tmp_path: Path):
    path = _database(tmp_path)
    for version in ("1.6.40", "1.6.41", "1.6.43"):
        db.ensure_version_cached(version, db_path=path)
    manifest = {
        "sequence": "transferred",
        "route": "reprocessing_missing_revalidation_input",
        "membership_transfer": {"source_campaign": "legacy_revalidation"},
        "historical_preprocess_lineage": {"reuse_permitted": False},
    }
    earlier_request = db.create_stage_request(
        sequence_name="transferred", dataset="dataset", stage="preprocess",
        pipeline_version="1.6.40", source_manifest=manifest,
        db_path=path,
    )
    run = db.create_stage_run(
        sequence_name="transferred", dataset="dataset", stage=db.PREPROCESS_STAGE,
        pipeline_version="1.6.40", status="SUCCEEDED", trigger="MIGRATION",
        request_id=earlier_request["id"], output_uri="swift://output/transferred",
        workflow_name="preprocess", db_path=path,
    )
    db.update_stage_request(earlier_request["id"], status="SUCCEEDED", db_path=path)
    source = db.freeze_campaign(
        db.create_campaign(
            name="source", campaign_type="BACKLOG_REPROCESSING", dataset="dataset",
            pipeline_version="1.6.41", output_uri="swift://export", created_by="test",
            canary=True, db_path=path,
        )["id"], inventory_uri="swift://inventory", inventory_sha256="a" * 64,
        inventory_sequence_count=1, configuration_uri="swift://config-41",
        configuration_sha256="b" * 64, db_path=path,
    )
    source_request = db.create_stage_request(
        sequence_name="transferred", dataset="dataset", stage="preprocess",
        pipeline_version="1.6.41", campaign=source["id"], cohort="CANARY",
        queue_priority=100, source_manifest=manifest, db_path=path,
    )
    db.update_stage_request(
        source_request["id"], status="SUCCEEDED",
        result_summary={"preprocess_adoption": {
            "preprocess_run_id": run["id"],
            "preprocess_output_uri": run["output_uri"],
        }}, db_path=path,
    )
    db.cancel_campaign(source["id"], db_path=path)
    replacement = db.freeze_campaign(
        db.create_campaign(
            name="replacement", campaign_type="BACKLOG_REPROCESSING",
            dataset="dataset", pipeline_version="1.6.43",
            output_uri="swift://export", created_by="test", canary=True,
            db_path=path,
        )["id"], inventory_uri="swift://inventory", inventory_sha256="a" * 64,
        inventory_sequence_count=1, configuration_uri="swift://config-43",
        configuration_sha256="c" * 64, db_path=path,
    )
    target = db.create_stage_request(
        sequence_name="transferred", dataset="dataset", stage="preprocess",
        pipeline_version="1.6.43", campaign=replacement["id"], cohort="CANARY",
        queue_priority=100, source_manifest=manifest, db_path=path,
    )

    result = db.adopt_valid_replacement_preprocess_results(
        source["id"], replacement["id"], adopted_by="test", db_path=path,
    )

    assert result["adopted_count"] == 1
    assert result["adopted"][0]["preprocess_run_id"] == run["id"]
    assert db.get_stage_request(target["id"], db_path=path)["status"] == "SUCCEEDED"


def test_export_policy_replacement_adopts_revalidation_for_export_only(tmp_path: Path):
    path = _database(tmp_path)
    db.ensure_version_cached("1.6.41", db_path=path)
    db.ensure_version_cached("1.6.42", db_path=path)
    manifest = {"sequence": "sequence", "route": "revalidation"}
    source = db.freeze_campaign(
        db.create_campaign(
            name="source-revalidation", campaign_type="LEGACY_REVALIDATION",
            dataset="dataset", pipeline_version="1.6.41",
            output_uri="swift://example/data_export_3", created_by="rollout",
            canary=True, db_path=path,
        )["id"],
        inventory_uri="swift://inventory", inventory_sha256="a" * 64,
        inventory_sequence_count=1, configuration_uri="swift://config-1641",
        configuration_sha256="b" * 64, db_path=path,
    )
    source_request = db.create_stage_request(
        sequence_name="sequence", dataset="dataset", stage="revalidation",
        pipeline_version="1.6.41", campaign=source["id"], cohort="CANARY",
        source_manifest=manifest, db_path=path,
    )
    result_uri = (
        "swift://host/account/bucket/_work/source/sequence/request_1/"
        "candidate_export/commit.json"
    )
    db.update_stage_request(
        source_request["id"], status="SUCCEEDED",
        result_manifest_uri=result_uri, result_manifest_sha256="c" * 64,
        db_path=path,
    )
    db.cancel_campaign(source["id"], db_path=path)
    replacement = db.freeze_campaign(
        db.create_campaign(
            name="replacement-revalidation", campaign_type="LEGACY_REVALIDATION",
            dataset="dataset", pipeline_version="1.6.42",
            output_uri="swift://example/data_export_3", created_by="rollout",
            canary=True, db_path=path,
        )["id"],
        inventory_uri="swift://inventory", inventory_sha256="a" * 64,
        inventory_sequence_count=1, configuration_uri="swift://config-1642",
        configuration_sha256="d" * 64, db_path=path,
    )
    target = db.create_stage_request(
        sequence_name="sequence", dataset="dataset", stage="revalidation",
        pipeline_version="1.6.42", campaign=replacement["id"], cohort="CANARY",
        source_manifest=manifest, db_path=path,
    )

    result = db.adopt_valid_replacement_revalidation_results(
        source["id"], replacement["id"], adopted_by="rollout", db_path=path,
    )

    assert result["export_retry_count"] == 1
    target = db.get_stage_request(target["id"], db_path=path)
    assert target["status"] == "PENDING"
    parameters = json.loads(target["parameters_json"])
    assert parameters["retry_work_output_url"] == result_uri.rsplit("/", 2)[0]
    assert parameters["revalidation_adoption"]["source_request_id"] == source_request["id"]


def test_export_policy_adopts_verified_reconciled_revalidation_completion(tmp_path: Path):
    path = _database(tmp_path)
    for version in ("1.6.41", "1.6.43"):
        db.ensure_version_cached(version, db_path=path)
    manifest = {"sequence": "sequence", "route": "revalidation"}
    source = db.freeze_campaign(
        db.create_campaign(
            name="source", campaign_type="LEGACY_REVALIDATION", dataset="dataset",
            pipeline_version="1.6.41", output_uri="swift://export", created_by="test",
            canary=True, db_path=path,
        )["id"], inventory_uri="swift://inventory", inventory_sha256="a" * 64,
        inventory_sequence_count=1, configuration_uri="swift://config-41",
        configuration_sha256="b" * 64, db_path=path,
    )
    request = db.create_stage_request(
        sequence_name="sequence", dataset="dataset", stage="revalidation",
        pipeline_version="1.6.41", campaign=source["id"], cohort="CANARY",
        source_manifest=manifest, db_path=path,
    )
    reserved = db.reserve_request_with_execution(
        request["id"], reserved_by="test", workflow_name="remote",
        pipeline_version="1.6.41", workflow_spec_path="revalidation.yaml",
        pool="pool", db_path=path,
    )
    execution = reserved["execution_id"]
    db.update_workflow_execution(
        execution, status="SUCCEEDED", osmo_workflow_id="remote-1",
        query_payload={"status": "COMPLETED", "tasks": {
            "check_accuracy": "COMPLETED", "export_revalidated": "COMPLETED",
        }}, db_path=path,
    )
    db.update_stage_request(
        request["id"], status="CANCELED",
        details="invalid_generation_retired:remote_completed", db_path=path,
    )
    db.cancel_campaign(source["id"], db_path=path)
    replacement = db.freeze_campaign(
        db.create_campaign(
            name="replacement", campaign_type="LEGACY_REVALIDATION",
            dataset="dataset", pipeline_version="1.6.43",
            output_uri="swift://export", created_by="test", canary=True,
            db_path=path,
        )["id"], inventory_uri="swift://inventory", inventory_sha256="a" * 64,
        inventory_sequence_count=1, configuration_uri="swift://config-43",
        configuration_sha256="c" * 64, db_path=path,
    )
    target = db.create_stage_request(
        sequence_name="sequence", dataset="dataset", stage="revalidation",
        pipeline_version="1.6.43", campaign=replacement["id"], cohort="CANARY",
        source_manifest=manifest, db_path=path,
    )
    commit_uri = "swift://work/source/sequence/request_1/candidate_export/commit.json"
    result = db.adopt_valid_replacement_revalidation_results(
        source["id"], replacement["id"], adopted_by="test",
        verified_work_outputs={request["id"]: {
            "workflow_execution_id": execution, "workflow_id": "remote-1",
            "candidate_commit_uri": commit_uri,
            "candidate_commit_sha256": "f" * 64,
            "commit_schema": "v2d.mv_hoi.revalidation_export_commit.v3",
            "commit_file_count": 12,
        }}, db_path=path,
    )

    assert result["export_retry_count"] == 1
    updated = db.get_stage_request(target["id"], db_path=path)
    evidence = json.loads(updated["parameters_json"])["revalidation_adoption"]
    assert evidence["remote_completion_reconciled"] is True
    assert evidence["workflow_id"] == "remote-1"


def test_export_policy_replacement_adopts_reconstruction_without_rerun(tmp_path: Path):
    path = _database(tmp_path)
    db.ensure_version_cached("1.6.41", db_path=path)
    db.ensure_version_cached("1.6.42", db_path=path)
    manifest = {"sequence": "sequence", "route": "backlog"}
    source = db.freeze_campaign(
        db.create_campaign(
            name="source-backlog", campaign_type="BACKLOG_REPROCESSING",
            dataset="dataset", pipeline_version="1.6.41",
            output_uri="swift://example/data_export_3", created_by="rollout",
            canary=True, db_path=path,
        )["id"],
        inventory_uri="swift://inventory", inventory_sha256="a" * 64,
        inventory_sequence_count=1, configuration_uri="swift://config-1641",
        configuration_sha256="b" * 64, db_path=path,
    )
    source_request = db.create_stage_request(
        sequence_name="sequence", dataset="dataset", stage="reconstruction",
        pipeline_version="1.6.41", campaign=source["id"], cohort="CANARY",
        source_manifest=manifest, db_path=path,
    )
    execution = db.create_workflow_execution(
        workflow_name="source-reconstruction", pipeline_type=db.RECONSTRUCTION_STAGE,
        pipeline_version="1.6.41", status="SUCCEEDED", db_path=path,
    )
    preprocess_execution = db.create_workflow_execution(
        workflow_name="source-preprocess", pipeline_type=db.PREPROCESS_STAGE,
        pipeline_version="1.6.41", status="SUCCEEDED", db_path=path,
    )
    calibration_execution = db.create_workflow_execution(
        workflow_name="source-calibration", pipeline_type=db.CALIBRATION_STAGE,
        pipeline_version="1.6.41", status="SUCCEEDED", db_path=path,
    )
    calibration_run = db.create_stage_run(
        sequence_name="calibration", dataset="dataset", stage=db.CALIBRATION_STAGE,
        pipeline_version="1.6.41", status="SUCCEEDED", trigger="AUTO",
        execution_id=calibration_execution, db_path=path,
    )
    db.upsert_sequence(
        "dataset", "sequence", calibration_sequence_name="calibration", db_path=path,
    )
    preprocess_run = db.create_stage_run(
        sequence_name="sequence", dataset="dataset", stage=db.PREPROCESS_STAGE,
        pipeline_version="1.6.41", status="SUCCEEDED", trigger="AUTO",
        execution_id=preprocess_execution,
        calibration_run_id=calibration_run["stage_run_id"],
        output_uri="swift://example/data_output/sequence", db_path=path,
    )
    source_run = db.create_stage_run(
        sequence_name="sequence", dataset="dataset", stage=db.RECONSTRUCTION_STAGE,
        pipeline_version="1.6.41", status="SUCCEEDED", trigger="AUTO",
        execution_id=execution, request_id=source_request["id"],
        preprocess_run_id=preprocess_run["id"],
        output_uri="swift://example/data_output/sequence/reconstruction_1",
        db_path=path,
    )
    db.update_stage_request(source_request["id"], status="SUCCEEDED", db_path=path)
    db.cancel_campaign(source["id"], db_path=path)
    replacement = db.freeze_campaign(
        db.create_campaign(
            name="replacement-backlog", campaign_type="BACKLOG_REPROCESSING",
            dataset="dataset", pipeline_version="1.6.42",
            output_uri="swift://example/data_export_3", created_by="rollout",
            canary=True, db_path=path,
        )["id"],
        inventory_uri="swift://inventory", inventory_sha256="a" * 64,
        inventory_sequence_count=1, configuration_uri="swift://config-1642",
        configuration_sha256="d" * 64, db_path=path,
    )
    target = db.create_stage_request(
        sequence_name="sequence", dataset="dataset", stage="reconstruction",
        pipeline_version="1.6.42", campaign=replacement["id"], cohort="CANARY",
        source_manifest=manifest, db_path=path,
    )

    result = db.adopt_valid_replacement_reconstruction_results(
        source["id"], replacement["id"], adopted_by="rollout", db_path=path,
    )

    assert result["adopted_count"] == 1
    target = db.get_stage_request(target["id"], db_path=path)
    assert target["status"] == "SUCCEEDED"
    adopted_run = db.get_stage_run_by_request(
        target["id"], stage=db.RECONSTRUCTION_STAGE, db_path=path,
    )
    assert adopted_run["output_uri"] == source_run["output_uri"]
    assert adopted_run["pipeline_version"] == "1.6.41"
    assert adopted_run["is_current"] == 1


def test_reconstruction_adoption_accepts_reconciled_remote_completion_and_stage_manifest(
    tmp_path: Path,
):
    path = _database(tmp_path)
    for version in ("1.6.41", "1.6.43"):
        db.ensure_version_cached(version, db_path=path)
    target_manifest = {
        "sequence": "sequence", "route": "backlog", "preprocess_run_id": 1,
    }
    stage_manifest = [{"name": "front.json", "sha256": "f" * 64}]
    source = db.freeze_campaign(
        db.create_campaign(
            name="source", campaign_type="BACKLOG_REPROCESSING", dataset="dataset",
            pipeline_version="1.6.41", output_uri="swift://export", created_by="test",
            canary=True, db_path=path,
        )["id"], inventory_uri="swift://inventory", inventory_sha256="a" * 64,
        inventory_sequence_count=1, configuration_uri="swift://config-41",
        configuration_sha256="b" * 64, db_path=path,
    )
    failed_attempt = db.create_stage_request(
        sequence_name="sequence", dataset="dataset", stage="reconstruction",
        pipeline_version="1.6.41", campaign=source["id"], cohort="CANARY",
        source_manifest={"unrelated": "infrastructure attempt"}, db_path=path,
    )
    db.update_stage_request(
        failed_attempt["id"], status="FAILED",
        details="operator_infrastructure_relocation", db_path=path,
    )
    source_request = db.create_stage_request(
        sequence_name="sequence", dataset="dataset", stage="reconstruction",
        pipeline_version="1.6.41", campaign=source["id"], cohort="CANARY",
        source_manifest=stage_manifest, db_path=path,
    )
    execution = db.create_workflow_execution(
        workflow_name="remote-completed", pipeline_type=db.RECONSTRUCTION_STAGE,
        pipeline_version="1.6.41", status="SUCCEEDED", db_path=path,
    )
    db.update_workflow_execution(
        execution, query_payload={"status": "COMPLETED", "tasks": {
            "foundation_pose": "COMPLETED", "upload_hitl": "COMPLETED",
        }}, db_path=path,
    )
    preprocess = db.create_stage_run(
        sequence_name="sequence", dataset="dataset", stage=db.PREPROCESS_STAGE,
        pipeline_version="1.6.41", status="SUCCEEDED", trigger="MIGRATION",
        output_uri="swift://preprocess", workflow_name="preprocess", db_path=path,
    )
    target_manifest["preprocess_run_id"] = preprocess["id"]
    run = db.create_stage_run(
        sequence_name="sequence", dataset="dataset", stage=db.RECONSTRUCTION_STAGE,
        pipeline_version="1.6.41", status="CANCELED", trigger="MIGRATION",
        execution_id=execution, request_id=source_request["id"],
        preprocess_run_id=preprocess["id"], output_uri="swift://reconstruction",
        labeled_bboxes_manifest_json=json.dumps(
            stage_manifest, sort_keys=True, separators=(",", ":")
        ),
        labeled_bboxes_sha256=source_request["source_manifest_sha256"],
        db_path=path,
    )
    db.update_stage_request(
        source_request["id"], status="CANCELED",
        details="invalid_generation_retired:remote_completed", db_path=path,
    )
    db.cancel_campaign(source["id"], db_path=path)
    replacement = db.freeze_campaign(
        db.create_campaign(
            name="replacement", campaign_type="BACKLOG_REPROCESSING",
            dataset="dataset", pipeline_version="1.6.43",
            output_uri="swift://export", created_by="test", canary=True,
            db_path=path,
        )["id"], inventory_uri="swift://inventory", inventory_sha256="a" * 64,
        inventory_sequence_count=1, configuration_uri="swift://config-43",
        configuration_sha256="c" * 64, db_path=path,
    )
    target = db.create_stage_request(
        sequence_name="sequence", dataset="dataset", stage="reconstruction",
        pipeline_version="1.6.43", campaign=replacement["id"], cohort="CANARY",
        source_manifest=target_manifest, db_path=path,
    )

    result = db.adopt_valid_replacement_reconstruction_results(
        source["id"], replacement["id"], adopted_by="test", db_path=path,
    )

    assert result["adopted_count"] == 1
    evidence = result["adopted"][0]
    assert evidence["remote_completion_reconciled"] is True
    assert evidence["source_identity_mode"].startswith("campaign_inventory")
    adopted = db.get_stage_run_by_request(
        target["id"], stage=db.RECONSTRUCTION_STAGE, db_path=path,
    )
    assert adopted["output_uri"] == run["output_uri"]


def test_canary_approval_keeps_failed_sequences_in_failed_pool(tmp_path: Path):
    path = _database(tmp_path)
    campaign = _campaign(path)
    succeeded = db.create_stage_request(
        sequence_name="succeeded", dataset="dataset", stage="revalidation",
        pipeline_version="1.6.0", campaign=campaign["id"], cohort="CANARY",
        db_path=path,
    )
    failed = db.create_stage_request(
        sequence_name="failed", dataset="dataset", stage="revalidation",
        pipeline_version="1.6.0", campaign=campaign["id"], cohort="CANARY",
        db_path=path,
    )
    db.update_stage_request(
        succeeded["id"], status="SUCCEEDED",
        result_manifest_uri="swift://result/commit.json",
        result_manifest_sha256="d" * 64, db_path=path,
    )
    db.update_stage_request(
        failed["id"], status="FAILED", details="pose_comparison_failed",
        db_path=path,
    )

    approved = db.approve_campaign_canary(
        campaign["id"], report_uri="swift://report", report_sha256="c" * 64,
        approved_by="operator", db_path=path,
    )

    assert approved["phase"] == "BULK"
    assert db.get_stage_request(failed["id"], db_path=path)["status"] == "FAILED"


def test_canary_report_requires_manual_review_only_for_successful_outputs(
    tmp_path: Path,
):
    path = _database(tmp_path)
    campaign = _campaign(path)
    succeeded = db.create_stage_request(
        sequence_name="succeeded", dataset="dataset", stage="revalidation",
        pipeline_version="1.6.0", campaign=campaign["id"], cohort="CANARY",
        db_path=path,
    )
    failed = db.create_stage_request(
        sequence_name="failed", dataset="dataset", stage="revalidation",
        pipeline_version="1.6.0", campaign=campaign["id"], cohort="CANARY",
        db_path=path,
    )
    db.update_stage_request(
        succeeded["id"], status="SUCCEEDED",
        result_manifest_uri="swift://result/commit.json",
        result_manifest_sha256="d" * 64, db_path=path,
    )
    db.update_stage_request(
        failed["id"], status="FAILED", details="accuracy gate failed",
        result_summary={
            "failure_category": "accuracy_check_failed",
            "failed_accuracy_checks": ["object_silhouette_alignment"],
        },
        db_path=path,
    )

    report, rows = campaign_report.build_report(campaign["name"], db_path=path)
    assert set(report["manual_reviews"]) == {"succeeded"}
    assert report["accuracy_failure_counts"] == {
        "object_silhouette_alignment": 1,
    }
    failed_row = next(row for row in rows if row["sequence"] == "failed")
    assert failed_row["failure_category"] == "accuracy_check_failed"
    assert failed_row["failed_accuracy_checks"] == [
        "object_silhouette_alignment",
    ]
    report["manual_reviews"]["succeeded"] = {
        "anonymized_videos": "PASS", "pose_overlay": "PASS",
        "reviewer": "operator", "reviewed_at": "2026-07-20T00:00:00Z",
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    campaign_cli._validate_manual_canary_reviews(report_path, campaign, path)


def test_revalidation_successor_reuses_frozen_identity_and_only_retries_export(
    tmp_path: Path,
):
    path = _database(tmp_path)
    campaign = _campaign(path)
    succeeded = db.create_stage_request(
        sequence_name="succeeded", dataset="dataset", stage="revalidation",
        pipeline_version="1.6.0", campaign=campaign["id"], cohort="CANARY",
        parameters={"work_output_layout": "request_scoped_v1"},
        db_path=path,
    )
    failed = db.create_stage_request(
        sequence_name="failed", dataset="dataset", stage="revalidation",
        pipeline_version="1.6.0", campaign=campaign["id"], cohort="CANARY",
        db_path=path,
    )
    bulk = db.create_stage_request(
        sequence_name="bulk", dataset="dataset", stage="revalidation",
        pipeline_version="1.6.0", campaign=campaign["id"], cohort="BULK",
        db_path=path,
    )
    db.set_campaign_inventory_count(campaign["id"], 3, db_path=path)
    result_uri = (
        "swift://host/account/bucket/_work/campaign/succeeded/"
        f"request_{succeeded['id']}/candidate_export/commit.json"
    )
    db.update_stage_request(
        succeeded["id"], status="SUCCEEDED",
        result_manifest_uri=result_uri,
        result_manifest_sha256="d" * 64, db_path=path,
    )
    db.update_stage_request(
        failed["id"], status="FAILED", details="quality gate failed",
        db_path=path,
    )
    db.ensure_version_cached("1.6.1", db_path=path)

    successor = db.create_revalidation_successor(
        campaign["id"], name="campaign-2", pipeline_version="1.6.1",
        created_by="operator", db_path=path,
    )

    assert successor["inventory_uri"] == campaign["inventory_uri"]
    assert successor["inventory_sha256"] == campaign["inventory_sha256"]
    assert successor["configuration_uri"] == campaign["configuration_uri"]
    assert successor["configuration_sha256"] == campaign["configuration_sha256"]
    assert successor["request_counts"] == {"BLOCKED": 1, "FAILED": 1, "PENDING": 1}
    assert db.get_campaign(campaign["id"], db_path=path)["status"] == "CANCELED"
    assert db.get_stage_request(bulk["id"], db_path=path)["status"] == "CANCELED"
    requests = {
        row["sequence_name"]: row
        for row in db.list_stage_requests(campaign="campaign-2", db_path=path)
    }
    retry_parameters = json.loads(requests["succeeded"]["parameters_json"])
    assert retry_parameters["successor_source_request_id"] == succeeded["id"]
    assert retry_parameters["retry_work_output_url"] == result_uri.rsplit("/", 2)[0]
    assert requests["succeeded"]["pipeline_version"] == "1.6.1"
    assert requests["failed"]["status"] == "FAILED"
    assert requests["bulk"]["blocked_reason"] == "WAITING_CANARY_APPROVAL"


def test_revalidation_export_requires_successful_request_manifest(tmp_path: Path):
    path = _database(tmp_path)
    db.ensure_version_cached("1.6.0", db_path=path)
    preprocess_execution = db.create_workflow_execution(
        workflow_name="preprocess", pipeline_type=db.PREPROCESS_STAGE,
        pipeline_version="1.6.0", status="SUCCEEDED", db_path=path,
    )
    calibration_execution = db.create_workflow_execution(
        workflow_name="calibration", pipeline_type=db.CALIBRATION_STAGE,
        pipeline_version="1.6.0", status="SUCCEEDED", db_path=path,
    )
    calibration = db.create_stage_run(
        sequence_name="calibration", dataset="dataset", stage=db.CALIBRATION_STAGE,
        pipeline_version="1.6.0", status="SUCCEEDED", trigger="MANUAL",
        execution_id=calibration_execution, db_path=path,
    )
    db.upsert_sequence(
        "dataset", "sequence", calibration_sequence_name="calibration", db_path=path,
    )
    preprocess = db.create_stage_run(
        sequence_name="sequence", dataset="dataset", stage=db.PREPROCESS_STAGE,
        pipeline_version="1.6.0", status="SUCCEEDED", trigger="MANUAL",
        execution_id=preprocess_execution,
        calibration_run_id=calibration["stage_run_id"], db_path=path,
    )
    reconstruction_execution = db.create_workflow_execution(
        workflow_name="reconstruction", pipeline_type=db.RECONSTRUCTION_STAGE,
        pipeline_version="1.6.0", status="SUCCEEDED", db_path=path,
    )
    reconstruction = db.create_stage_run(
        sequence_name="sequence", dataset="dataset", stage=db.RECONSTRUCTION_STAGE,
        pipeline_version="1.6.0", status="SUCCEEDED", trigger="MANUAL",
        execution_id=reconstruction_execution,
        preprocess_run_id=preprocess["stage_run_id"], db_path=path,
    )
    campaign = _campaign(path)
    request = db.create_stage_request(
        sequence_name="sequence", dataset="dataset", stage="revalidation",
        pipeline_version="1.6.0", campaign=campaign["id"], cohort="CANARY",
        db_path=path,
    )
    reservation = db.reserve_request_with_execution(
        request["id"], reserved_by="dispatcher", workflow_name="revalidation",
        pipeline_version="1.6.0", workflow_spec_path="revalidation.yaml",
        pool="pool", db_path=path,
    )
    assert reservation is not None
    exported = db.commit_revalidation_export(
        request["id"], reconstruction_run_id=reconstruction["stage_run_id"],
        result_manifest_uri=(
            "swift://example/data_export_2/sequence/commit.json"
        ), result_manifest_sha256="f" * 64,
        result_summary={"all_gates": "PASS"},
        source_uri="swift://example/data_output/sequence",
        output_uri="swift://example/data_export_2/sequence", db_path=path,
    )
    assert exported["authorization_type"] == "REVALIDATION"
    assert exported["request_id"] == request["id"]
    assert db.get_stage_request(request["id"], db_path=path)["status"] == "SUCCEEDED"
    assert db.get_workflow_execution(
        reservation["execution_id"], db_path=path,
    )["status"] == "SUCCEEDED"
    assert db.commit_revalidation_export(
        request["id"], reconstruction_run_id=reconstruction["stage_run_id"],
        result_manifest_uri=(
            "swift://example/data_export_2/sequence/commit.json"
        ), result_manifest_sha256="f" * 64,
        result_summary={"all_gates": "PASS"},
        source_uri="swift://example/data_output/sequence",
        output_uri="swift://example/data_export_2/sequence", db_path=path,
    )["stage_run_id"] == exported["stage_run_id"]

def test_revalidation_export_recovers_only_publication_commit_failures(tmp_path: Path):
    path = _database(tmp_path)
    db.ensure_version_cached("1.6.0", db_path=path)
    preprocess_execution = db.create_workflow_execution(
        workflow_name="preprocess", pipeline_type=db.PREPROCESS_STAGE,
        pipeline_version="1.6.0", status="SUCCEEDED", db_path=path,
    )
    calibration_execution = db.create_workflow_execution(
        workflow_name="calibration", pipeline_type=db.CALIBRATION_STAGE,
        pipeline_version="1.6.0", status="SUCCEEDED", db_path=path,
    )
    calibration = db.create_stage_run(
        sequence_name="calibration", dataset="dataset", stage=db.CALIBRATION_STAGE,
        pipeline_version="1.6.0", status="SUCCEEDED", trigger="MANUAL",
        execution_id=calibration_execution, db_path=path,
    )
    db.upsert_sequence(
        "dataset", "sequence", calibration_sequence_name="calibration", db_path=path,
    )
    preprocess = db.create_stage_run(
        sequence_name="sequence", dataset="dataset", stage=db.PREPROCESS_STAGE,
        pipeline_version="1.6.0", status="SUCCEEDED", trigger="MANUAL",
        execution_id=preprocess_execution,
        calibration_run_id=calibration["stage_run_id"], db_path=path,
    )
    reconstruction_execution = db.create_workflow_execution(
        workflow_name="reconstruction", pipeline_type=db.RECONSTRUCTION_STAGE,
        pipeline_version="1.6.0", status="SUCCEEDED", db_path=path,
    )
    reconstruction = db.create_stage_run(
        sequence_name="sequence", dataset="dataset", stage=db.RECONSTRUCTION_STAGE,
        pipeline_version="1.6.0", status="SUCCEEDED", trigger="MANUAL",
        execution_id=reconstruction_execution,
        preprocess_run_id=preprocess["stage_run_id"], db_path=path,
    )
    campaign = _campaign(path)
    request = db.create_stage_request(
        sequence_name="sequence", dataset="dataset", stage="revalidation",
        pipeline_version="1.6.0", campaign=campaign["id"], cohort="CANARY",
        db_path=path,
    )
    reservation = db.reserve_request_with_execution(
        request["id"], reserved_by="dispatcher", workflow_name="revalidation",
        pipeline_version="1.6.0", workflow_spec_path="revalidation.yaml",
        pool="pool", db_path=path,
    )
    db.apply_execution_observation(
        reservation["execution_id"], execution_status="FAILED",
        details="export_run_commit_failed: transient NoSuchKey",
        request_outcomes=[{"request_id": request["id"], "status": "FAILED"}],
        db_path=path,
    )

    with pytest.raises(ValueError, match="cannot commit from FAILED"):
        db.commit_revalidation_export(
            request["id"], reconstruction_run_id=reconstruction["stage_run_id"],
            result_manifest_uri="swift://result", result_manifest_sha256="f" * 64,
            result_summary={"all_gates": "PASS"},
            source_uri="swift://example/data_output/sequence",
            output_uri="swift://example/data_export_2/sequence", db_path=path,
        )

    exported = db.commit_revalidation_export(
        request["id"], reconstruction_run_id=reconstruction["stage_run_id"],
        result_manifest_uri="swift://result", result_manifest_sha256="f" * 64,
        result_summary={"all_gates": "PASS"},
        source_uri="swift://example/data_output/sequence",
        output_uri="swift://example/data_export_2/sequence",
        recover_publication_failure=True, db_path=path,
    )
    assert exported["authorization_type"] == "REVALIDATION"
    assert db.get_stage_request(request["id"], db_path=path)["status"] == "SUCCEEDED"

    other = db.create_stage_request(
        sequence_name="sequence", dataset="dataset", stage="revalidation",
        pipeline_version="1.6.0", campaign=campaign["id"], cohort="CANARY",
        db_path=path,
    )
    other_reservation = db.reserve_request_with_execution(
        other["id"], reserved_by="dispatcher", workflow_name="revalidation-2",
        pipeline_version="1.6.0", workflow_spec_path="revalidation.yaml",
        pool="pool", db_path=path,
    )
    db.apply_execution_observation(
        other_reservation["execution_id"], execution_status="FAILED",
        details="result_validation_failed: accuracy",
        request_outcomes=[{"request_id": other["id"], "status": "FAILED"}],
        db_path=path,
    )
    with pytest.raises(ValueError, match="cannot commit from FAILED"):
        db.commit_revalidation_export(
            other["id"], reconstruction_run_id=reconstruction["stage_run_id"],
            result_manifest_uri="swift://result-2",
            result_manifest_sha256="e" * 64,
            result_summary={"all_gates": "PASS"},
            source_uri="swift://example/data_output/sequence",
            output_uri="swift://example/data_export_2/sequence-2",
            recover_publication_failure=True, db_path=path,
        )


def test_newer_priority_export_can_fulfill_frozen_campaign_request(tmp_path: Path):
    path = _database(tmp_path)
    campaign = _campaign(path)
    db.set_campaign_inventory_count(campaign["id"], 1, db_path=path)
    original = db.create_stage_request(
        sequence_name="sequence", dataset="dataset", stage="revalidation",
        pipeline_version="1.6.0", campaign=campaign["id"], cohort="BULK",
        db_path=path,
    )
    db.update_stage_request(
        original["id"], status="CANCELED",
        details="superseded_by_priority_retry", db_path=path,
    )

    priority = db.create_campaign(
        name="priority", campaign_type="LEGACY_REVALIDATION", dataset="dataset",
        pipeline_version="1.6.0", output_uri="swift://example/data_export_2/",
        created_by="operator", db_path=path,
    )
    priority = db.freeze_campaign(
        priority["id"], inventory_uri="swift://example/priority.json",
        inventory_sha256="c" * 64,
        configuration_uri="swift://example/priority-config.json",
        configuration_sha256="d" * 64, inventory_sequence_count=1,
        db_path=path,
    )
    retry = db.create_stage_request(
        sequence_name="sequence", dataset="dataset", stage="revalidation",
        pipeline_version="1.6.0", campaign=priority["id"], cohort="CANARY",
        db_path=path,
    )
    db.update_stage_request(
        retry["id"], status="SUCCEEDED",
        result_manifest_uri="swift://example/data_export_2/sequence/commit.json",
        result_manifest_sha256="e" * 64, db_path=path,
    )
    execution_id = db.create_workflow_execution(
        workflow_name="priority-export", pipeline_type=db.REVALIDATION_STAGE,
        pipeline_version="1.6.0", status="SUCCEEDED", db_path=path,
    )
    conn = db.get_connection(path)
    try:
        sequence_id = conn.execute(
            "SELECT id FROM sequences WHERE dataset='dataset' AND sequence_name='sequence'"
        ).fetchone()[0]
        conn.execute(
            """INSERT INTO export_runs
               (sequence_id, workflow_execution_id, pipeline_version,
                status, trigger, requested_by,
                output_uri, is_current, authorization_type, request_id)
               VALUES (?, ?, '1.6.0', 'SUCCEEDED', 'MANUAL', 'operator',
                       'swift://example/data_export_2/sequence', 1,
                       'REVALIDATION', ?)""",
            (sequence_id, execution_id, retry["id"]),
        )
        conn.commit()
    finally:
        conn.close()

    adopted = db.adopt_request_fulfillment(
        original["id"], retry["id"], db_path=path,
    )
    assert adopted["status"] == "CANCELED"
    assert adopted["fulfilled_by_request_id"] == retry["id"]
    assert db.adopt_request_fulfillment(
        original["id"], retry["id"], db_path=path,
    )["fulfilled_by_request_id"] == retry["id"]
    assert db.campaign_progress(
        campaign["id"], db_path=path,
    )["committed_export_count"] == 1
    status = db.get_sequence_status("dataset", db_path=path)[0]
    assert status["effective_status"] == "SUCCEEDED"
    assert status["effective_stage"] == "export"
    assert db.finish_campaign(
        campaign["id"], db_path=path,
    )["status"] == "SUCCEEDED"


def test_completing_failed_campaign_releases_blocked_request_for_remediation(tmp_path: Path):
    path = _database(tmp_path)
    backlog = _campaign(path, campaign_type="BACKLOG_REPROCESSING")
    blocked = db.create_stage_request(
        sequence_name="sequence", dataset="dataset", stage="reconstruction",
        pipeline_version="1.6.0", campaign=backlog["id"], cohort="BULK",
        status="BLOCKED", blocked_reason="WAITING_RELABEL", db_path=path,
    )
    completed = db.finish_campaign(backlog["id"], db_path=path)
    assert completed["status"] == "COMPLETED_WITH_FAILURES"
    terminal = db.get_stage_request(blocked["id"], db_path=path)
    assert terminal["status"] == "FAILED"
    assert terminal["blocked_reason"] == "WAITING_RELABEL"

    remediation = db.create_campaign(
        name="remediation", campaign_type="REMEDIATION", dataset="dataset",
        pipeline_version="1.6.0", output_uri="swift://example/data_export_2/",
        created_by="operator", db_path=path,
    )
    remediation = db.freeze_campaign(
        remediation["id"], inventory_uri="swift://example/remediation.json",
        inventory_sha256="c" * 64, configuration_uri="swift://example/config.json",
        configuration_sha256="d" * 64, db_path=path,
    )
    followup = db.create_stage_request(
        sequence_name="sequence", dataset="dataset", stage="reconstruction",
        pipeline_version="1.6.0", campaign=remediation["id"], cohort="BULK",
        db_path=path,
    )
    assert followup["status"] == "PENDING"


def test_campaign_version_update_changes_only_undispatched_requests(tmp_path: Path):
    path = _database(tmp_path)
    campaign = _campaign(path)
    pending = db.create_stage_request(
        sequence_name="pending", dataset="dataset", stage="revalidation",
        pipeline_version="1.6.0", campaign=campaign["id"], cohort="CANARY",
        db_path=path,
    )
    completed = db.create_stage_request(
        sequence_name="completed", dataset="dataset", stage="revalidation",
        pipeline_version="1.6.0", campaign=campaign["id"], cohort="CANARY",
        db_path=path,
    )
    db.update_stage_request(
        completed["id"], status="SUCCEEDED",
        result_manifest_uri="swift://example/data_export_2/completed/commit.json",
        result_manifest_sha256="c" * 64, db_path=path,
    )

    result = db.update_campaign_pipeline_version(
        campaign["id"], "1.6.1", message="new export image", db_path=path,
    )

    assert result["previous_pipeline_version"] == "1.6.0"
    assert result["pipeline_version"] == "1.6.1"
    assert result["updated_request_count"] == 1
    assert db.get_stage_request(pending["id"], db_path=path)["pipeline_version"] == "1.6.1"
    assert db.get_stage_request(completed["id"], db_path=path)["pipeline_version"] == "1.6.0"


def test_campaign_version_update_preserves_workflow_owning_requests(tmp_path: Path):
    path = _database(tmp_path)
    campaign = _campaign(path)
    request = db.create_stage_request(
        sequence_name="active", dataset="dataset", stage="revalidation",
        pipeline_version="1.6.0", campaign=campaign["id"], cohort="CANARY",
        db_path=path,
    )
    assert db.reserve_stage_request(
        request["id"], reserved_by="test", db_path=path,
    )["status"] == "RESERVED"

    result = db.update_campaign_pipeline_version(
        campaign["id"], "1.6.1", db_path=path,
    )

    assert result["pipeline_version"] == "1.6.1"
    assert result["updated_request_count"] == 0
    assert result["unchanged_active_request_counts"] == {"RESERVED": 1}
    assert db.get_stage_request(request["id"], db_path=path)["pipeline_version"] == "1.6.0"


def test_active_execution_admission_excludes_publication_pending_work(tmp_path: Path):
    path = _database(tmp_path)
    campaign = _campaign(path)

    def request_with_execution(sequence: str) -> tuple[dict, dict]:
        request = db.create_stage_request(
            sequence_name=sequence, dataset="dataset", stage="revalidation",
            pipeline_version="1.6.0", campaign=campaign["id"], cohort="CANARY",
            db_path=path,
        )
        reservation = db.reserve_request_with_execution(
            request["id"], reserved_by="test", workflow_name=f"wf-{sequence}",
            pipeline_version="1.6.0", workflow_spec_path="workflow.yaml",
            pool="pool", db_path=path,
        )
        return request, reservation

    active, active_reservation = request_with_execution("active")
    db.apply_execution_observation(
        active_reservation["execution_id"], execution_status="RUNNING",
        request_outcomes=[{
            "request_id": active["id"], "status": "RUNNING",
        }], db_path=path,
    )
    pending, pending_reservation = request_with_execution("pending-publication")
    db.apply_execution_observation(
        pending_reservation["execution_id"], execution_status="SUCCEEDED",
        details="result_reconciliation_pending",
        request_outcomes=[{
            "request_id": pending["id"], "status": "RUNNING",
            "details": "result_reconciliation_pending",
        }], db_path=path,
    )
    missing = db.create_stage_request(
        sequence_name="missing-execution", dataset="dataset",
        stage="revalidation", pipeline_version="1.6.0",
        campaign=campaign["id"], cohort="CANARY", db_path=path,
    )
    db.reserve_stage_request(missing["id"], reserved_by="test", db_path=path)

    assert db.count_active_stage_executions(
        "revalidation", campaign=campaign["id"], db_path=path,
    ) == 2
