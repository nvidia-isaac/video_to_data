from pathlib import Path
import subprocess
import sys

import pytest


WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

from orchestration import db, mark_ready


def test_uploaded_batch_items_requires_nonempty_balanced_artifacts():
    listing = "\n".join([
        "2026-07-20 12:00:00 10 prefix/batch_20260720/dataset/item-a.mp4",
        "2026-07-20 12:00:01 20 prefix/batch_20260720/jsons/item-a.json",
    ])

    videos, jsons = mark_ready.uploaded_batch_items(listing, "batch_20260720")

    assert videos == {"item-a"}
    assert jsons == {"item-a"}
    with pytest.raises(ValueError, match="empty"):
        mark_ready.uploaded_batch_items(
            "2026-07-20 12:00:00 0 prefix/batch_20260720/dataset/item.mp4",
            "batch_20260720",
        )


def test_expected_batch_items_uses_successful_runs_and_reports_active(tmp_path):
    path = str(tmp_path / "processing.db")
    db.init_db(path)
    db.ensure_version_cached("1.6.2", db_path=path)
    for sequence, workflow, status in (
        ("done", "v2d_reconstruction_20260720_done", "PASS"),
        ("active", "v2d_reconstruction_20260720_active", "WAITING_WF"),
    ):
        db.insert_workflow(
            sequence_name=sequence, dataset="dataset",
            pipeline_type=db.RECONSTRUCTION_STAGE, pipeline_version="1.6.2",
            workflow_name=workflow, status=status, trigger="migration",
            hitl_item_id=f"{sequence}.json", db_path=path,
        )

    expected, active = mark_ready.expected_batch_items(
        "dataset", "batch_20260720", db_path=path,
    )

    assert expected == {"done"}
    assert active == ["v2d_reconstruction_20260720_active"]


def test_expected_batch_items_excludes_canceled_campaign_lineage(tmp_path):
    path = str(tmp_path / "processing.db")
    db.init_db(path)
    db.ensure_version_cached("1.6.28", db_path=path)

    def add_campaign_run(name: str, *, canceled: bool) -> None:
        campaign = db.create_campaign(
            name=f"{name}-campaign",
            campaign_type="BACKLOG_REPROCESSING",
            dataset="dataset",
            pipeline_version="1.6.28",
            output_uri="swift://example/data_output",
            created_by="test",
            db_path=path,
        )
        campaign = db.freeze_campaign(
            campaign["id"],
            inventory_uri=f"swift://example/{name}-inventory.json",
            inventory_sha256="a" * 64,
            inventory_sequence_count=1,
            configuration_uri=f"swift://example/{name}-configuration.json",
            configuration_sha256="b" * 64,
            db_path=path,
        )
        request = db.create_stage_request(
            sequence_name=name,
            dataset="dataset",
            stage="reconstruction",
            pipeline_version="1.6.28",
            campaign=campaign["id"],
            cohort="BULK",
            db_path=path,
        )
        db.insert_workflow(
            sequence_name=name,
            dataset="dataset",
            pipeline_type=db.RECONSTRUCTION_STAGE,
            pipeline_version="1.6.28",
            workflow_name=f"v2d_reconstruction_20260720_{name}",
            status="PASS",
            trigger="migration",
            request_id=request["id"],
            hitl_item_id=f"{name}.json",
            db_path=path,
        )
        db.update_stage_request(
            request["id"], status="SUCCEEDED", db_path=path,
        )
        if canceled:
            db.cancel_campaign(campaign["id"], db_path=path)

    add_campaign_run("current", canceled=False)
    add_campaign_run("retired", canceled=True)

    expected, active = mark_ready.expected_batch_items(
        "dataset", "batch_20260720", db_path=path,
    )

    assert expected == {"current"}
    assert active == []


def test_put_ready_marker_is_create_once(monkeypatch):
    commands = []
    responses = iter([
        subprocess.CompletedProcess([], 0, stdout="{}", stderr=""),
        subprocess.CompletedProcess(
            [], 254, stdout="",
            stderr="An error occurred (PreconditionFailed) with status 412",
        ),
    ])

    def run(command, *, capture_output, text):
        commands.append(command)
        assert capture_output is True
        assert text is True
        return next(responses)

    monkeypatch.setattr(mark_ready.subprocess, "run", run)
    marker = "s3://bucket/batch/markers/ready_for_processing"

    assert mark_ready.put_ready_marker(marker, '{"created_at":"first"}') is True
    assert mark_ready.put_ready_marker(marker, '{"created_at":"second"}') is False
    assert len(commands) == 2
    for command in commands:
        assert command[:3] == ["aws", "s3api", "put-object"]
        assert command[command.index("--bucket") + 1] == "bucket"
        assert (
            command[command.index("--key") + 1]
            == "batch/markers/ready_for_processing"
        )
        assert command[command.index("--if-none-match") + 1] == "*"
