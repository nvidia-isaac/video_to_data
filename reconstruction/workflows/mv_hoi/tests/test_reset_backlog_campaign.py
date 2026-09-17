import sys
from pathlib import Path


WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

from migration import reset_backlog_campaign as reset  # noqa: E402
from orchestration import db  # noqa: E402


def test_reset_retires_campaign_lineage_and_only_its_blacklists(tmp_path):
    path = str(tmp_path / "db.sqlite")
    db.init_db(path)
    db.ensure_version_cached("1.6.28", db_path=path)
    campaign = db.create_campaign(
        name="old-backlog", campaign_type="BACKLOG_REPROCESSING",
        dataset="dataset", pipeline_version="1.6.28",
        output_uri="swift://host/account/bucket/data_export_2",
        created_by="operator", db_path=path,
    )
    campaign = db.freeze_campaign(
        campaign["id"], inventory_uri="swift://inventory.json",
        inventory_sha256="a" * 64, configuration_uri="swift://config.json",
        configuration_sha256="b" * 64, inventory_sequence_count=2,
        db_path=path,
    )
    preprocess_request = db.create_stage_request(
        sequence_name="sequence", dataset="dataset", stage="preprocess",
        pipeline_version="1.6.28", campaign=campaign["id"], cohort="BULK",
        db_path=path,
    )
    reconstruction_request = db.create_stage_request(
        sequence_name="sequence", dataset="dataset", stage="reconstruction",
        pipeline_version="1.6.28", campaign=campaign["id"], cohort="BULK",
        db_path=path,
    )
    db.create_stage_request(
        sequence_name="older-blacklist", dataset="dataset", stage="preprocess",
        pipeline_version="1.6.28", campaign=campaign["id"], cohort="BULK",
        db_path=path,
    )
    preprocess_run = db.create_stage_run(
        sequence_name="sequence", dataset="dataset", stage=db.PREPROCESS_STAGE,
        pipeline_version="1.6.28", status="SUCCEEDED", trigger="MIGRATION",
        workflow_name="old-preprocess",
        request_id=preprocess_request["id"],
        output_uri="swift://host/account/bucket/data_output_2/sequence/request_1",
        db_path=path,
    )
    reconstruction_run = db.create_stage_run(
        sequence_name="sequence", dataset="dataset",
        stage=db.RECONSTRUCTION_STAGE, pipeline_version="1.6.28",
        status="SUCCEEDED", trigger="MIGRATION",
        workflow_name="old-reconstruction",
        request_id=reconstruction_request["id"],
        preprocess_run_id=preprocess_run["stage_run_id"],
        output_uri="swift://host/account/bucket/data_output_2/sequence/request_2",
        db_path=path,
    )
    db.update_stage_request(
        preprocess_request["id"], status="SUCCEEDED", db_path=path,
    )
    db.update_stage_request(
        reconstruction_request["id"], status="SUCCEEDED", db_path=path,
    )
    db.upsert_blacklisted_sequence(
        "dataset", "sequence", reason="campaign failure",
        created_by="campaign:old-backlog", db_path=path,
    )
    db.upsert_blacklisted_sequence(
        "dataset", "older-blacklist", reason="older safeguard",
        created_by="legacy_migration", db_path=path,
    )
    db.cancel_campaign(campaign["id"], db_path=path)

    report = reset.audit(
        campaign["id"], expected_campaign_id=campaign["id"], db_path=path,
    )

    assert report["membership_count"] == 2
    assert len(report["current_runs_to_retire"]) == 2
    assert len(report["campaign_blacklists_to_remove"]) == 1
    assert not report["remote_active_requests"]

    result = reset.apply(report, db_path=path)

    assert result == {
        "retired_current_runs": {"preprocess": 1, "reconstruction": 1},
        "removed_campaign_blacklists": 1,
    }
    assert db.get_stage_run(
        preprocess_run["stage_run_id"], stage=db.PREPROCESS_STAGE, db_path=path,
    )["is_current"] == 0
    assert db.get_stage_run(
        reconstruction_run["stage_run_id"],
        stage=db.RECONSTRUCTION_STAGE, db_path=path,
    )["is_current"] == 0
    assert db.get_blacklisted_sequence("dataset", "sequence", db_path=path) is None
    assert db.get_blacklisted_sequence(
        "dataset", "older-blacklist", db_path=path,
    )["created_by"] == "legacy_migration"
