# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import base64
import json
import subprocess
import sys
from pathlib import Path

import pytest


WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

from orchestration import db, query, submit


def _db_path(tmp_path: Path) -> str:
    path = str(tmp_path / "processing_v2.db")
    db.init_db(path)
    db.ensure_version_cached("1.0.0", db_path=path)
    return path


def _dataset_cfg() -> dict:
    return {
        "swift_base": "swift://host/AUTH_bucket/root",
        "mesh_base": "swift://host/AUTH_mesh/meshes",
        "weights_base_url": "swift://host/AUTH_weights/releases/test",
        "osmo_pool": "pool",
        "pipelines": {
            "mv_calibration": {
                "input_path": "data",
                "output_path": "data_output",
                "max_concurrent": 10,
                "workflows": {
                    "calibration": {
                        "workflow_yaml": "osmo/mv_calibration.yaml",
                        "calibration_setup": "setup",
                    }
                },
            },
            "mv_preprocess": {
                "input_path": "data",
                "output_path": "data_output",
                "campaign_output_path": "data_output",
                "max_concurrent": 10,
                "workflows": {
                    "preprocess": {"workflow_yaml": "osmo/mv_preprocess.yaml"}
                },
            },
            "mv_hoi_reconstruction": {
                "input_path": "data_output",
                "output_path": "data_output",
                "campaign_output_path": "data_output",
                "max_concurrent": 10,
                "workflows": {
                    "reconstruction": {
                        "workflow_yaml": "osmo/mv_hoi_reconstruction.yaml",
                        "hitl_s3_base": "s3://hitl/items",
                    }
                },
            },
        },
    }


def test_face_detector_video_gate_requires_all_eight_cameras(monkeypatch):
    seen = []

    def fake_object_exists(_client, bucket, key):
        seen.append((bucket, key))
        return not key.endswith("/left_stereo_camera_right.mp4")

    monkeypatch.setattr(submit, "object_exists", fake_object_exists)

    missing = submit.missing_face_detector_videos(
        object(), "AUTH_bucket", "root/data_output/sequence/face_detector/",
    )

    assert missing == ["left_stereo_camera_right"]
    assert len(seen) == 8
    assert {key.rsplit("/", 1)[-1] for _, key in seen} == {
        f"{camera}.mp4" for camera in submit.FACE_DETECTOR_CAMERAS
    }


def test_campaign_processing_uses_canonical_preprocess_and_isolated_reconstruction():
    cfg = _dataset_cfg()
    cfg["pipelines"][db.PREPROCESS_STAGE]["campaign_output_path"] = "data_output"
    cfg["pipelines"][db.RECONSTRUCTION_STAGE]["campaign_output_path"] = "data_output"
    assert submit.campaign_processing_output_path(
        cfg, db.PREPROCESS_STAGE, None,
    ) == "data_output"
    request = {
        "id": 42,
        "sequence_name": "sequence_a",
        "campaign_type": "BACKLOG_REPROCESSING",
        "parameters_json": "{}",
    }
    assert submit.campaign_processing_output_path(
        cfg, db.PREPROCESS_STAGE, request,
    ) == "data_output"
    assert submit.processing_sequence_output_url(
        cfg["swift_base"], "data_output", "sequence_a",
        db.PREPROCESS_STAGE, request,
    ).endswith("/data_output/sequence_a")
    assert submit.processing_sequence_output_url(
        cfg["swift_base"], "data_output", "sequence_a",
        db.RECONSTRUCTION_STAGE, request,
    ).endswith("/data_output/sequence_a/reconstruction_42")
    cfg["pipelines"][db.PREPROCESS_STAGE]["campaign_output_path"] = "data_output_2"
    with pytest.raises(ValueError, match="must match"):
        submit.campaign_processing_output_path(
            cfg, db.PREPROCESS_STAGE, request,
        )


def test_adopted_preprocess_lineage_requires_complete_identity_evidence(monkeypatch):
    manifest_sha = "a" * 64
    target = {
        "id": 20, "status": "SUCCEEDED", "stage": "preprocess",
        "workflow_execution_id": None, "queue_priority": 100,
        "campaign_id": 7, "sequence_name": "sequence_a",
        "pipeline_version": "1.6.36", "source_manifest_sha256": manifest_sha,
        "result_summary_json": json.dumps({
            "preprocess_adoption": {
                "schema": "v2d.mv_hoi.preprocess_patch_adoption.v1",
                "adopted_by": "accuracy-segment-rollout",
                "reason": "implementation_patch_does_not_affect_preprocessing",
                "source_request_id": 10,
                "source_campaign_id": 6,
                "preprocess_run_id": 30,
                "preprocess_pipeline_version": "1.6.35",
                "target_pipeline_version": "1.6.36",
                "preprocess_output_uri": (
                    "swift://host/AUTH_bucket/root/data_output/sequence_a"
                ),
                "source_manifest_sha256": manifest_sha,
            },
        }),
    }
    source = {
        "id": 10, "status": "SUCCEEDED", "stage": "preprocess",
        "campaign_id": 6, "sequence_name": "sequence_a",
        "source_manifest_sha256": manifest_sha,
    }
    monkeypatch.setattr(
        submit, "get_stage_request",
        lambda request_id, **_kwargs: {20: target, 10: source}.get(request_id),
    )
    scheduled = {
        "campaign_id": 7, "sequence_name": "sequence_a",
        "pipeline_version": "1.6.36",
    }
    run = {
        "stage_run_id": 30, "request_id": 10, "status": "PASS",
        "run_status": "SUCCEEDED",
        "is_current": 1, "pipeline_version": "1.6.35",
        "output_uri": "swift://host/AUTH_bucket/root/data_output/sequence_a",
    }

    assert submit._matches_adopted_preprocess_lineage(scheduled, 20, run)
    assert not submit._matches_adopted_preprocess_lineage(
        scheduled, 20, {**run, "stage_run_id": 31},
    )
    assert not submit._matches_adopted_preprocess_lineage(
        scheduled, 20, {**run, "run_status": "FAILED"},
    )
    target["result_summary_json"] = json.dumps({
        "preprocess_adoption": {
            **json.loads(target["result_summary_json"])["preprocess_adoption"],
            "source_manifest_sha256": "b" * 64,
        },
    })
    assert not submit._matches_adopted_preprocess_lineage(scheduled, 20, run)


def test_adopted_preprocess_lineage_follows_valid_replacement_chain(monkeypatch):
    manifest_sha = "a" * 64
    output_uri = "swift://host/AUTH_bucket/root/data_output/sequence_a"

    def adopted(request_id, campaign_id, version, source_id, source_campaign):
        return {
            "id": request_id,
            "status": "SUCCEEDED",
            "stage": "preprocess",
            "workflow_execution_id": None,
            "queue_priority": 100,
            "campaign_id": campaign_id,
            "sequence_name": "sequence_a",
            "pipeline_version": version,
            "source_manifest_sha256": manifest_sha,
            "result_summary_json": json.dumps({
                "preprocess_adoption": {
                    "schema": "v2d.mv_hoi.preprocess_patch_adoption.v1",
                    "adopted_by": "accuracy-segment-rollout",
                    "reason": "implementation_patch_does_not_affect_preprocessing",
                    "source_request_id": source_id,
                    "source_campaign_id": source_campaign,
                    "preprocess_run_id": 30,
                    "preprocess_pipeline_version": "1.6.35",
                    "preprocess_output_uri": output_uri,
                    "source_manifest_sha256": manifest_sha,
                    "target_pipeline_version": version,
                },
            }),
        }

    target = adopted(30, 8, "1.6.37", 20, 7)
    intermediate = adopted(20, 7, "1.6.36", 10, 6)
    source = {
        "id": 10,
        "status": "SUCCEEDED",
        "stage": "preprocess",
        "campaign_id": 6,
        "sequence_name": "sequence_a",
        "source_manifest_sha256": manifest_sha,
    }
    requests = {30: target, 20: intermediate, 10: source}
    monkeypatch.setattr(
        submit, "get_stage_request",
        lambda request_id, **_kwargs: requests.get(request_id),
    )
    scheduled = {
        "campaign_id": 8,
        "sequence_name": "sequence_a",
        "pipeline_version": "1.6.37",
    }
    run = {
        "stage_run_id": 30,
        "request_id": 10,
        "status": "PASS",
        "run_status": "SUCCEEDED",
        "is_current": 1,
        "pipeline_version": "1.6.35",
        "output_uri": output_uri,
    }

    assert submit._matches_adopted_preprocess_lineage(scheduled, 30, run)
    intermediate["result_summary_json"] = target["result_summary_json"]
    assert not submit._matches_adopted_preprocess_lineage(scheduled, 30, run)


def test_mainline_processing_uses_stage_specific_configured_output():
    cfg = _dataset_cfg()
    cfg["pipelines"][db.PREPROCESS_STAGE]["output_path"] = "data_output_2"
    request = {
        "id": 43,
        "sequence_name": "sequence_b",
        "campaign_type": None,
        "parameters_json": json.dumps({
            "processing_output_layout": "request_scoped_v1",
        }),
    }

    output_path = submit.campaign_processing_output_path(
        cfg, db.PREPROCESS_STAGE, request,
    )

    assert output_path == "data_output_2"
    assert submit.processing_sequence_output_url(
        cfg["swift_base"], output_path, "sequence_b",
        db.PREPROCESS_STAGE, request,
    ).endswith("/data_output_2/sequence_b")
    assert submit.processing_sequence_output_url(
        cfg["swift_base"], output_path, "sequence_b",
        db.RECONSTRUCTION_STAGE, request,
    ).endswith("/data_output_2/sequence_b/reconstruction_43")


def _successful_calibration(path: str, name: str = "calib") -> dict:
    return db.create_stage_run(
        sequence_name=name,
        dataset="dataset_a",
        stage=db.CALIBRATION_STAGE,
        pipeline_version="1.0.0",
        workflow_name=f"cal-{name}",
        status="SUCCEEDED",
        db_path=path,
    )


def _successful_preprocess(path: str, sequence: str = "sequence_a") -> dict:
    calibration = _successful_calibration(path)
    db.upsert_sequence(
        "dataset_a", sequence, calibration_sequence_name="calib", db_path=path
    )
    return db.create_stage_run(
        sequence_name=sequence,
        dataset="dataset_a",
        stage=db.PREPROCESS_STAGE,
        pipeline_version="1.0.0",
        workflow_name=f"pre-{sequence}",
        status="SUCCEEDED",
        calibration_run_id=calibration["stage_run_id"],
        output_uri=f"swift://host/AUTH_bucket/root/data_output/{sequence}",
        db_path=path,
    )


def test_blacklist_is_dataset_scoped_persistent_and_audited(tmp_path):
    path = _db_path(tmp_path)
    db.upsert_blacklisted_sequence(
        "dataset_a",
        "shared",
        reason="operator hold",
        created_by="alice",
        db_path=path,
    )
    assert db.is_sequence_blacklisted("dataset_a", "shared", db_path=path)
    assert not db.is_sequence_blacklisted("dataset_b", "shared", db_path=path)
    entry = db.get_blacklisted_sequences("dataset_a", db_path=path)[0]
    assert entry["reason"] == "operator hold"
    assert entry["created_by"] == "alice"
    assert entry["blacklisted_at"]
    assert db.remove_blacklisted_sequence("dataset_a", "shared", db_path=path)
    assert not db.is_sequence_blacklisted("dataset_a", "shared", db_path=path)


def test_submit_skips_blacklist_without_force(tmp_path, monkeypatch):
    path = _db_path(tmp_path)
    db.upsert_blacklisted_sequence(
        "dataset_a", "blocked", reason="timeout box", db_path=path
    )
    monkeypatch.setattr(submit, "DB_PATH", path)
    monkeypatch.setattr(
        submit, "osmo_submit", lambda *_args, **_kwargs: pytest.fail("must not submit")
    )
    assert submit.submit_sequence(
        "blocked", "dataset_a", _dataset_cfg(), db.CALIBRATION_STAGE
    ) is None


def test_force_is_one_time_bypass_and_does_not_clear_blacklist(tmp_path, monkeypatch):
    path = _db_path(tmp_path)
    db.upsert_blacklisted_sequence(
        "dataset_a", "blocked", reason="timeout box", db_path=path
    )
    monkeypatch.setattr(submit, "DB_PATH", path)
    monkeypatch.setattr(submit, "_generate_workflow_name", lambda *_: "forced-run")
    monkeypatch.setattr(
        submit, "get_s3_client", lambda *_args, **_kwargs: (object(), "bucket", "root")
    )
    monkeypatch.setattr(submit, "osmo_submit", lambda *_args, **_kwargs: "forced-run-1")

    result = submit.submit_sequence(
        "blocked",
        "dataset_a",
        _dataset_cfg(),
        db.CALIBRATION_STAGE,
        force=True,
        pipeline_version="1.0.0",
    )
    assert result and not result.ambiguous
    assert db.get_blacklisted_sequence("dataset_a", "blocked", db_path=path)
    assert db.get_workflow("forced-run", db_path=path)["status"] == "WAITING_WF"


@pytest.mark.parametrize(
    "metadata",
    [None, {}, {"calib_seq_name": "calib"}],
)
def test_unmet_preprocess_prerequisite_creates_no_execution_attempt(
    tmp_path, monkeypatch, metadata
):
    path = _db_path(tmp_path)
    monkeypatch.setattr(submit, "DB_PATH", path)
    monkeypatch.setattr(submit, "_generate_workflow_name", lambda *_: "not-an-attempt")
    monkeypatch.setattr(
        submit, "get_s3_client", lambda *_args, **_kwargs: (object(), "bucket", "root")
    )
    monkeypatch.setattr(submit, "get_hoi_metadata", lambda *_args, **_kwargs: metadata)
    monkeypatch.setattr(submit, "path_exists", lambda *_args, **_kwargs: False)
    result = submit.submit_sequence(
        "sequence_a", "dataset_a", _dataset_cfg(), db.PREPROCESS_STAGE,
        pipeline_version="1.0.0",
    )
    assert result.prereq_skipped
    assert db.get_workflow("not-an-attempt", db_path=path) is None
    assert db.list_stage_run_history(
        "dataset_a", stage=db.PREPROCESS_STAGE, db_path=path
    ) == []


def test_preprocess_submission_passes_pinned_weights_base_url(tmp_path, monkeypatch):
    path = _db_path(tmp_path)
    _successful_calibration(path, "calib")
    captured = {}
    monkeypatch.setattr(submit, "DB_PATH", path)
    monkeypatch.setattr(submit, "_generate_workflow_name", lambda *_: "preprocess-run")
    monkeypatch.setattr(
        submit, "get_s3_client", lambda *_args, **_kwargs: (object(), "bucket", "root")
    )
    monkeypatch.setattr(
        submit,
        "get_hoi_metadata",
        lambda *_args, **_kwargs: {
            "calib_seq_name": "calib",
            "object": {"id": "box"},
        },
    )
    monkeypatch.setattr(submit, "path_exists", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        submit,
        "resolve_mesh_url",
        lambda *_args, **_kwargs: "swift://host/AUTH_mesh/meshes/box/model",
    )

    def accept(_yaml, _pool, set_vars, **_kwargs):
        captured.update(set_vars)
        return "preprocess-run-1"

    monkeypatch.setattr(submit, "osmo_submit", accept)
    result = submit.submit_sequence(
        "sequence_a",
        "dataset_a",
        _dataset_cfg(),
        db.PREPROCESS_STAGE,
        pipeline_version="1.0.0",
    )
    assert result
    assert captured["weights_base_url"] == "swift://host/AUTH_weights/releases/test"


def test_backlog_campaign_can_supersede_legacy_success_without_force(
    tmp_path, monkeypatch,
):
    path = _db_path(tmp_path)
    preprocess = _successful_preprocess(path)
    db.create_stage_run(
        sequence_name="sequence_a", dataset="dataset_a",
        stage=db.RECONSTRUCTION_STAGE, pipeline_version="1.0.0",
        workflow_name="legacy-pass", status="SUCCEEDED",
        preprocess_run_id=preprocess["stage_run_id"], db_path=path,
    )
    campaign = db.create_campaign(
        name="backlog", campaign_type="BACKLOG_REPROCESSING",
        dataset="dataset_a", pipeline_version="1.0.0",
        output_uri="swift://host/AUTH_bucket/root/data_export_2",
        created_by="operator", db_path=path,
    )
    campaign = db.freeze_campaign(
        campaign["id"], inventory_uri="swift://inventory.json",
        inventory_sha256="a" * 64, configuration_uri="swift://config.json",
        configuration_sha256="b" * 64, inventory_sequence_count=1,
        db_path=path,
    )
    request = db.create_stage_request(
        sequence_name="sequence_a", dataset="dataset_a", stage="preprocess",
        pipeline_version="1.0.0", campaign=campaign["id"], cohort="BULK",
        parameters={},
        db_path=path,
    )
    db.upsert_blacklisted_sequence(
        "dataset_a", "sequence_a", reason="legacy timeout box", db_path=path,
    )
    monkeypatch.setattr(submit, "DB_PATH", path)
    monkeypatch.setattr(submit, "_generate_workflow_name", lambda *_: "campaign-preprocess")
    monkeypatch.setattr(
        submit, "get_s3_client", lambda *_args, **_kwargs: (object(), "bucket", "root")
    )
    monkeypatch.setattr(
        submit, "get_hoi_metadata",
        lambda *_args, **_kwargs: {"calib_seq_name": "calib", "object": {"id": "box"}},
    )
    monkeypatch.setattr(submit, "path_exists", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        submit, "resolve_mesh_url",
        lambda *_args, **_kwargs: "swift://host/AUTH_mesh/meshes/box/model",
    )
    monkeypatch.setattr(submit, "osmo_submit", lambda *_args, **_kwargs: "campaign-preprocess-1")

    result = submit.submit_sequence(
        "sequence_a", "dataset_a", _dataset_cfg(), db.PREPROCESS_STAGE,
        pipeline_version="1.0.0", trigger="MIGRATION", request_id=request["id"],
    )

    assert result
    assert db.get_stage_request(request["id"], db_path=path)["status"] == "RUNNING"
    assert db.get_blacklisted_sequence(
        "dataset_a", "sequence_a", db_path=path,
    )["reason"] == "legacy timeout box"


def test_submit_reserves_before_osmo_and_promotes_after_acceptance(tmp_path, monkeypatch):
    path = _db_path(tmp_path)
    observed = {}
    monkeypatch.setattr(submit, "DB_PATH", path)
    monkeypatch.setattr(submit, "_generate_workflow_name", lambda *_: "reserved-run")
    monkeypatch.setattr(
        submit, "get_s3_client", lambda *_args, **_kwargs: (object(), "bucket", "root")
    )

    def accept(*_args, **_kwargs):
        row = db.get_workflow("reserved-run", db_path=path)
        observed["status_during_submit"] = row["run_status"]
        observed["is_current_during_submit"] = row["is_current"]
        return "reserved-run-1"

    monkeypatch.setattr(submit, "osmo_submit", accept)
    submit.submit_sequence(
        "calibration", "dataset_a", _dataset_cfg(), db.CALIBRATION_STAGE,
        pipeline_version="1.0.0",
    )
    assert observed == {"status_during_submit": "SUBMITTING", "is_current_during_submit": 0}
    row = db.get_workflow("reserved-run", db_path=path)
    assert row["run_status"] == "RUNNING"
    assert row["is_current"] == 1
    assert row["osmo_workflow_id"] == "reserved-run-1"


def test_ambiguous_submit_remains_current_unknown_and_is_not_relaunched(
    tmp_path, monkeypatch
):
    path = _db_path(tmp_path)
    error = subprocess.CalledProcessError(
        10, "osmo", output="Read timed out", stderr=""
    )
    monkeypatch.setattr(submit, "DB_PATH", path)
    monkeypatch.setattr(submit, "_generate_workflow_name", lambda *_: "ambiguous-run")
    monkeypatch.setattr(
        submit, "get_s3_client", lambda *_args, **_kwargs: (object(), "bucket", "root")
    )
    monkeypatch.setattr(
        submit,
        "osmo_submit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(submit.AmbiguousSubmitError(error)),
    )
    result = submit.submit_sequence(
        "calibration", "dataset_a", _dataset_cfg(), db.CALIBRATION_STAGE,
        pipeline_version="1.0.0",
    )
    assert result.ambiguous
    row = db.get_workflow("ambiguous-run", db_path=path)
    assert row["status"] == "UNKNOWN"
    assert row["is_current"] == 1

    monkeypatch.setattr(
        submit, "osmo_submit", lambda *_args, **_kwargs: pytest.fail("must not relaunch")
    )
    assert submit.submit_sequence(
        "calibration", "dataset_a", _dataset_cfg(), db.CALIBRATION_STAGE,
        pipeline_version="1.0.0",
    ) is None


def test_clear_submit_failure_is_historical_and_keeps_previous_owner(
    tmp_path, monkeypatch
):
    path = _db_path(tmp_path)
    previous = _successful_calibration(path, "calibration")
    error = subprocess.CalledProcessError(2, "osmo", output="", stderr="invalid spec")
    monkeypatch.setattr(submit, "DB_PATH", path)
    monkeypatch.setattr(submit, "_generate_workflow_name", lambda *_: "failed-resubmit")
    monkeypatch.setattr(
        submit, "get_s3_client", lambda *_args, **_kwargs: (object(), "bucket", "root")
    )
    monkeypatch.setattr(
        submit, "osmo_submit", lambda *_args, **_kwargs: (_ for _ in ()).throw(error)
    )
    assert submit.submit_sequence(
        "calibration",
        "dataset_a",
        _dataset_cfg(),
        db.CALIBRATION_STAGE,
        force=True,
        pipeline_version="1.0.0",
    ) is None
    assert db.get_current_stage_run(
        "calibration", "dataset_a", db.CALIBRATION_STAGE, db_path=path
    )["stage_run_id"] == previous["stage_run_id"]
    failed = db.get_workflow("failed-resubmit", db_path=path)
    assert failed["run_status"] == "FAILED"
    assert failed["is_current"] == 0


class _Body:
    def __init__(self, payload: bytes):
        self.payload = payload

    def read(self):
        return self.payload


class _BBoxClient:
    def __init__(self):
        self.payloads = {
            "data_output/sequence_a/mv_preprocess/labeled_bboxes/a.json": b'{"a":1}',
            "data_output/sequence_a/mv_preprocess/labeled_bboxes/b.json": b'{"b":2}',
        }

    def get_paginator(self, _name):
        client = self

        class Paginator:
            def paginate(self, **_kwargs):
                yield {
                    "Contents": [
                        {"Key": key, "ETag": f'"etag-{index}"'}
                        for index, key in enumerate(client.payloads)
                    ]
                }

        return Paginator()

    def get_object(self, *, Bucket, Key):
        del Bucket
        return {"Body": _Body(self.payloads[Key])}


def test_reconstruction_records_and_passes_exact_bbox_snapshot(tmp_path, monkeypatch):
    path = _db_path(tmp_path)
    preprocess = _successful_preprocess(path)
    client = _BBoxClient()
    captured = {}
    monkeypatch.setattr(submit, "DB_PATH", path)
    monkeypatch.setattr(submit, "_generate_workflow_name", lambda *_: "reconstruction-run")
    monkeypatch.setattr(
        submit, "get_s3_client", lambda *_args, **_kwargs: (client, "root", "root")
    )
    monkeypatch.setattr(submit, "path_exists", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(submit, "object_exists", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(submit, "prefix_has_json_files", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        submit,
        "get_s3_text",
        lambda _client, _bucket, key: (
            json.dumps([{"cameras": [{"transform": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]}]}])
            if key.endswith("/edex")
            else "manual_labeled_bboxes"
            if key.endswith("object_bbox_source.txt")
            else None
        ),
    )

    def accept(_yaml, _pool, set_vars, **_kwargs):
        captured.update(set_vars)
        return "reconstruction-run-1"

    monkeypatch.setattr(submit, "osmo_submit", accept)
    result = submit.submit_sequence(
        "sequence_a", "dataset_a", _dataset_cfg(), db.RECONSTRUCTION_STAGE,
        pipeline_version="1.0.0",
    )
    assert result
    row = db.get_workflow("reconstruction-run", db_path=path)
    assert row["preprocess_run_id"] == preprocess["stage_run_id"]
    assert captured["swift_output_base"].endswith(
        f"/data_output/sequence_a/reconstruction_{row['request_id']}"
    )
    assert captured["mv_preprocess_url"] == (
        "swift://host/AUTH_bucket/root/data_output/sequence_a/mv_preprocess/"
    )
    assert captured["face_detector_url"] == (
        "swift://host/AUTH_bucket/root/data_output/sequence_a/face_detector/"
    )
    assert row["labeled_bboxes_sha256"] == captured["expected_labeled_bboxes_sha256"]
    assert captured["weights_base_url"] == "swift://host/AUTH_weights/releases/test"
    manifest = json.loads(row["labeled_bboxes_manifest_json"])
    assert json.loads(base64.b64decode(
        captured["expected_labeled_bboxes_manifest_b64"]
    )) == manifest
    assert [entry["name"] for entry in manifest] == ["a.json", "b.json"]
    assert all({"name", "size", "etag", "sha256"} == set(entry) for entry in manifest)


def test_refresh_completed_reconstruction_waits_for_qc(tmp_path, monkeypatch):
    path = _db_path(tmp_path)
    preprocess = _successful_preprocess(path)
    run = db.create_stage_run(
        sequence_name="sequence_a",
        dataset="dataset_a",
        stage=db.RECONSTRUCTION_STAGE,
        pipeline_version="1.0.0",
        workflow_name="reconstruction-running",
        osmo_workflow_id="reconstruction-running-1",
        status="RUNNING",
        preprocess_run_id=preprocess["stage_run_id"],
        db_path=path,
    )
    monkeypatch.setattr(
        query, "osmo_query", lambda _id: {"status": "COMPLETED", "tasks": {}}
    )
    query.refresh_waiting(
        "dataset_a", pipeline_type=db.RECONSTRUCTION_STAGE, db_path=path
    )
    refreshed = db.get_stage_run(run["stage_run_id"], db_path=path)
    assert refreshed["run_status"] == "SUCCEEDED"
    assert refreshed["status"] == "WAITING_QC"


def test_reconstruction_accuracy_failure_ingests_only_stable_categories(
    tmp_path, monkeypatch,
):
    path = _db_path(tmp_path)
    preprocess = _successful_preprocess(path)
    request = db.create_stage_request(
        sequence_name="sequence_a",
        dataset="dataset_a",
        stage="reconstruction",
        pipeline_version="1.0.0",
        db_path=path,
    )
    reservation = db.reserve_request_with_execution(
        request["id"],
        reserved_by="pytest",
        workflow_name="reconstruction-running",
        pipeline_version="1.0.0",
        workflow_spec_path="workflow.yaml",
        pool="pool",
        db_path=path,
    )
    run = db.create_stage_run(
        sequence_name="sequence_a",
        dataset="dataset_a",
        stage=db.RECONSTRUCTION_STAGE,
        pipeline_version="1.0.0",
        status="WAITING_WF",
        execution_id=reservation["execution_id"],
        preprocess_run_id=preprocess["stage_run_id"],
        request_id=request["id"],
        output_uri="swift://host/account/bucket/data_output/sequence_a/reconstruction_1",
        db_path=path,
    )
    db.apply_execution_observation(
        reservation["execution_id"],
        execution_status="RUNNING",
        osmo_workflow_id="reconstruction-running-1",
        request_outcomes=[{"request_id": request["id"], "status": "RUNNING"}],
        db_path=path,
    )
    monkeypatch.setattr(
        query,
        "osmo_query",
        lambda _id: {
            "status": "FAILED",
            "tasks": {
                "check_accuracy": "FAILED",
                "upload": "FAILED_UPSTREAM",
            },
        },
    )
    monkeypatch.setattr(
        query,
        "_read_accuracy_report",
        lambda _uri: {
            "status": "FAIL",
            "reason": (
                "object silhouette bbox containment failed: camera "
                "bad frame fraction 0.45 > 0.05"
            ),
            "checks": {
                "chamfer_object": "PASS",
                "chamfer_human": "PASS",
                "object_mask_containment": "PASS",
                "object_silhouette_alignment": "FAIL",
            },
            "object_silhouette_alignment": {
                "per_camera": {"camera": {"bad_frame_fraction": 0.45}},
            },
        },
    )

    query.refresh_waiting(
        "dataset_a", pipeline_type=db.RECONSTRUCTION_STAGE, db_path=path,
    )

    refreshed_run = db.get_stage_run(
        run["stage_run_id"], stage=db.RECONSTRUCTION_STAGE, db_path=path,
    )
    refreshed_request = db.get_stage_request(request["id"], db_path=path)
    expected_summary = {
        "failure_category": "accuracy_check_failed",
        "failed_accuracy_checks": ["object_silhouette_alignment"],
    }
    assert refreshed_run["details"] == (
        "task_failed: check_accuracy; "
        "accuracy_checks_failed: object_silhouette_alignment"
    )
    assert json.loads(refreshed_request["result_summary_json"]) == expected_summary
    assert "0.45" not in refreshed_request["details"]
    assert "camera" not in refreshed_request["details"]
    assert "reason" not in json.loads(refreshed_request["result_summary_json"])


def test_accuracy_failure_category_falls_back_to_normalized_reason() -> None:
    assert query.categorize_accuracy_failure({
        "status": "FAIL",
        "reason": "object chamfer 41.4 > threshold 40.0",
    }) == {
        "failure_category": "accuracy_check_failed",
        "failed_accuracy_checks": ["chamfer_object"],
    }
    assert query.categorize_accuracy_failure({
        "status": "FAIL",
        "reason": "some sequence-specific failure",
    }) is None


def test_backfill_enriches_existing_generic_accuracy_failure(tmp_path, monkeypatch):
    path = _db_path(tmp_path)
    preprocess = _successful_preprocess(path)
    request = db.create_stage_request(
        sequence_name="sequence_a", dataset="dataset_a",
        stage="reconstruction", pipeline_version="1.0.0", db_path=path,
    )
    reservation = db.reserve_request_with_execution(
        request["id"], reserved_by="pytest", workflow_name="reconstruction-failed",
        pipeline_version="1.0.0", workflow_spec_path="workflow.yaml",
        pool="pool", db_path=path,
    )
    run = db.create_stage_run(
        sequence_name="sequence_a", dataset="dataset_a",
        stage=db.RECONSTRUCTION_STAGE, pipeline_version="1.0.0",
        status="WAITING_WF", execution_id=reservation["execution_id"],
        preprocess_run_id=preprocess["stage_run_id"], request_id=request["id"],
        output_uri="swift://host/account/bucket/data_output/sequence_a/reconstruction_1",
        db_path=path,
    )
    db.apply_execution_observation(
        reservation["execution_id"], execution_status="FAILED",
        details="task_failed: check_accuracy",
        run_outcomes=[{
            "run_id": run["stage_run_id"],
            "stage": db.RECONSTRUCTION_STAGE,
            "status": "FAILED",
        }],
        db_path=path,
    )
    monkeypatch.setattr(
        query,
        "_read_accuracy_report",
        lambda _uri: {
            "checks": {
                "chamfer_object": "FAIL",
                "chamfer_human": "PASS",
                "object_mask_containment": "PASS",
                "object_silhouette_alignment": "PASS",
            },
            "object_chamfer_median": 41.438,
        },
    )

    assert query.backfill_accuracy_failure_summaries(
        "dataset_a", db_path=path,
    ) == 1
    refreshed = db.get_stage_request(request["id"], db_path=path)
    assert refreshed["details"].endswith("accuracy_checks_failed: chamfer_object")
    assert json.loads(refreshed["result_summary_json"]) == {
        "failure_category": "accuracy_check_failed",
        "failed_accuracy_checks": ["chamfer_object"],
    }
    assert query.backfill_accuracy_failure_summaries(
        "dataset_a", db_path=path,
    ) == 0


def test_ambiguous_not_found_stays_active_for_later_recovery(tmp_path, monkeypatch):
    path = _db_path(tmp_path)
    run = db.create_stage_run(
        sequence_name="calibration",
        dataset="dataset_a",
        stage=db.CALIBRATION_STAGE,
        pipeline_version="1.0.0",
        workflow_name="ambiguous",
        osmo_workflow_id="ambiguous-1",
        status="UNKNOWN",
        details="submit_ambiguous: timeout",
        db_path=path,
    )
    monkeypatch.setattr(
        query,
        "osmo_query",
        lambda _id: {"status": "UNKNOWN", "tasks": {}, "not_found": True},
    )
    monkeypatch.setattr(
        query, "_ambiguous_submit_not_found_grace_expired", lambda _wf: False,
    )
    submitted_at = db.get_workflow_execution(
        run["execution_id"], db_path=path,
    )["submitted_at"]
    query.refresh_waiting(
        "dataset_a", pipeline_type=db.CALIBRATION_STAGE, db_path=path
    )
    refreshed = db.get_stage_run(run["stage_run_id"], db_path=path)
    assert refreshed["run_status"] == "UNKNOWN"
    assert refreshed["is_current"] == 1
    assert db.get_workflow_execution(
        run["execution_id"], db_path=path,
    )["submitted_at"] == submitted_at


def test_auto_discovery_upserts_sequences_before_scheduling(tmp_path, monkeypatch):
    path = _db_path(tmp_path)
    monkeypatch.setattr(submit, "DB_PATH", path)
    monkeypatch.setattr(
        submit, "get_s3_client", lambda *_args, **_kwargs: (object(), "bucket", "root")
    )
    monkeypatch.setattr(submit, "list_sequences", lambda *_args, **_kwargs: ["calib-a"])
    monkeypatch.setattr(
        submit,
        "submit_sequence",
        lambda *_args, **_kwargs: submit.SubmitResult("submitted"),
    )
    submit.auto_submit(
        "dataset_a", _dataset_cfg(), db.CALIBRATION_STAGE, pipeline_version="1.0.0"
    )
    status = db.get_sequence_status("dataset_a", "calib-a", db_path=path)[0]
    assert status["sequence_kind"] == "calibration"
    assert status["calibration_status"] == "NOT_STARTED"
