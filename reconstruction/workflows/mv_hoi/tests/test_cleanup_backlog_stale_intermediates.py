from io import BytesIO
from pathlib import Path
import sys

import pytest


WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

from migration import cleanup_backlog_stale_intermediates as recovery


class Client:
    def __init__(self, objects):
        self.objects = objects

    def get_paginator(self, _name):
        objects = self.objects

        class Paginator:
            def paginate(self, *, Bucket, Prefix):
                del Bucket
                yield {"Contents": [
                    {"Key": key, "Size": len(value), "ETag": '"etag"'}
                    for key, value in objects.items() if key.startswith(Prefix)
                ]}

        return Paginator()

    def get_object(self, *, Bucket, Key):
        del Bucket
        return {"Body": BytesIO(self.objects[Key]), "ETag": '"etag"'}


def _config():
    return {"datasets": {"dataset": {
        "swift_base": "swift://host/account/bucket/root",
        "pipelines": {"mv_preprocess": {"output_path": "data_output"}},
    }}}


def _install(monkeypatch, *, active=False, reset_preprocess_succeeded=False):
    campaign = {
        "id": 7, "name": "backlog", "dataset": "dataset",
        "campaign_type": "BACKLOG_REPROCESSING", "pipeline_version": "1.2.3",
    }
    requests = [
        {"id": 10, "sequence_name": "done", "stage": "preprocess", "status": "SUCCEEDED"},
        {"id": 20, "sequence_name": "reset", "stage": "preprocess", "status": "FAILED",
         "parameters_json": "{}", "source_manifest_json": "{}"},
        {"id": 21, "sequence_name": "reset", "stage": "reconstruction",
         "status": "RUNNING" if active else "FAILED"},
    ]
    objects = {
        "root/data_output/reset/mv_preprocess/labeled_bboxes/camera.json": b"label",
        "root/data_output/reset/mv_preprocess/images/camera.h5": b"rgb",
        "root/data_output/reset/face_detector/images/camera.h5": b"face",
        "root/data_output/reset/reconstruction_21/check_accuracy/check_accuracy.json": b"metric",
        "root/data_output/reset/reconstruction_999/orphan.bin": b"orphan",
    }
    client = Client(objects)
    monkeypatch.setattr(recovery.db, "get_campaign", lambda *_a, **_k: campaign)
    monkeypatch.setattr(recovery.db, "list_stage_requests", lambda **_k: requests)

    def run_for_request(request_id, *, stage, db_path):
        del stage, db_path
        if request_id == 10:
            return {"stage_run_id": 100, "run_status": "SUCCEEDED"}
        if request_id == 20:
            return {
                "stage_run_id": 200,
                "run_status": (
                    "SUCCEEDED" if reset_preprocess_succeeded else "FAILED"
                ),
            }
        if request_id == 21:
            return {
                "stage_run_id": 210, "run_status": "FAILED",
                "output_uri": "swift://host/account/bucket/root/data_output/reset/reconstruction_21",
            }
        return None

    monkeypatch.setattr(recovery.db, "get_stage_run_by_request", run_for_request)
    monkeypatch.setattr(
        recovery.cleanup, "_client",
        lambda url: (client, "bucket", url.split("/bucket/", 1)[1].rstrip("/")),
    )
    return objects


def test_audit_deletes_only_resolved_stale_keys(monkeypatch):
    _install(monkeypatch)
    report = recovery.audit(
        "backlog", expected_campaign_id=7, db_path="db", config=_config(),
    )
    assert report["reset_count"] == 1
    reset = next(item for item in report["records"] if item["sequence"] == "reset")
    keys = {item["key"] for item in reset["deletions"]}
    assert "root/data_output/reset/mv_preprocess/images/camera.h5" in keys
    assert "root/data_output/reset/face_detector/images/camera.h5" in keys
    assert any("reconstruction_21/" in key for key in keys)
    assert not any("labeled_bboxes" in key for key in keys)
    assert not any("reconstruction_999" in key for key in keys)


def test_audit_refuses_unreconciled_campaign_work(monkeypatch):
    _install(monkeypatch, active=True)
    with pytest.raises(RuntimeError, match="cancel and reconcile"):
        recovery.audit(
            "backlog", expected_campaign_id=7, db_path="db", config=_config(),
        )


def test_canceled_reconstruction_retries_without_deleting_preprocess(monkeypatch):
    _install(monkeypatch, reset_preprocess_succeeded=True)
    report = recovery.audit(
        "backlog", expected_campaign_id=7, db_path="db", config=_config(),
    )
    reset = next(item for item in report["records"] if item["sequence"] == "reset")
    assert reset["recovery_stage"] == "reconstruction"
    keys = {item["key"] for item in reset["deletions"]}
    assert any("reconstruction_21/" in key for key in keys)
    assert not any("mv_preprocess/images" in key for key in keys)
    assert not any("face_detector" in key for key in keys)
