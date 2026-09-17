from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys

import pytest
from botocore.exceptions import ClientError


WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

from migration import cleanup_backlog_pending_intermediates as pending


class Client:
    def __init__(self, objects):
        self.objects = dict(objects)

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        client = self

        class Paginator:
            def paginate(self, *, Bucket, Prefix):
                assert Bucket == "bucket"
                yield {"Contents": [
                    {"Key": key, "Size": len(value), "ETag": f'"etag-{len(value)}"'}
                    for key, value in sorted(client.objects.items())
                    if key.startswith(Prefix)
                ]}

        return Paginator()

    def get_object(self, *, Bucket, Key):
        assert Bucket == "bucket"
        if Key not in self.objects:
            raise KeyError(Key)
        value = self.objects[Key]
        return {"Body": BytesIO(value), "ETag": f'"etag-{len(value)}"'}

    def put_object(self, *, Bucket, Key, Body, **_kwargs):
        assert Bucket == "bucket"
        self.objects[Key] = Body

    def delete_objects(self, *, Bucket, Delete):
        assert Bucket == "bucket"
        for item in Delete["Objects"]:
            self.objects.pop(item["Key"], None)
        return {}

    def delete_object(self, *, Bucket, Key):
        assert Bucket == "bucket"
        self.objects.pop(Key, None)


class StaleDeleteListingClient(Client):
    """Return one stale pre-delete listing before exposing the deletion."""

    def __init__(self, objects):
        super().__init__(objects)
        self.stale_listing = None
        self.delete_calls = 0

    def get_paginator(self, name):
        if self.stale_listing is None:
            return super().get_paginator(name)
        contents = self.stale_listing
        self.stale_listing = None

        class Paginator:
            def paginate(self, *, Bucket, Prefix):
                assert Bucket == "bucket"
                yield {"Contents": [item for item in contents if item["Key"].startswith(Prefix)]}

        return Paginator()

    def delete_objects(self, *, Bucket, Delete):
        if self.delete_calls == 0:
            self.stale_listing = [
                {"Key": key, "Size": len(value), "ETag": f'"etag-{len(value)}"'}
                for key, value in sorted(self.objects.items())
            ]
        self.delete_calls += 1
        return super().delete_objects(Bucket=Bucket, Delete=Delete)


def _request(
    request_id, sequence, status, created_at, *, run_status=None, reason=None,
):
    return {
        "id": request_id,
        "sequence_id": request_id,
        "sequence_name": sequence,
        "dataset": "dataset",
        "campaign_id": 491,
        "stage": "preprocess",
        "status": status,
        "pipeline_version": "1.6.29",
        "created_at": created_at,
        "reason": reason,
        "preprocess_run_status": run_status,
    }


def test_candidate_selection_includes_exact_quota_retry_and_excludes_unsafe_work():
    preprocess = [
        _request(1, "ordinary", "PENDING", "2026-01-01"),
        _request(10, "quota", "SUCCEEDED", "2026-01-01", run_status="SUCCEEDED"),
        _request(
            11, "quota", "PENDING", "2026-01-02",
            reason="missing_face_videos_retry_of_request_10",
        ),
        _request(20, "legit", "SUCCEEDED", "2026-01-01", run_status="SUCCEEDED"),
        _request(21, "legit", "PENDING", "2026-01-02", reason="unrelated"),
        _request(30, "active", "PENDING", "2026-01-01"),
        _request(40, "newer", "PENDING", "2026-01-01"),
    ]
    all_requests = [
        {key: row[key] for key in (
            "id", "sequence_id", "sequence_name", "campaign_id", "stage",
            "status", "created_at",
        )}
        for row in preprocess
    ] + [
        {
            "id": 31, "sequence_id": 30, "sequence_name": "active",
            "campaign_id": 999, "stage": "revalidation", "status": "RUNNING",
            "created_at": "2026-01-02",
        },
        {
            "id": 41, "sequence_id": 40, "sequence_name": "newer",
            "campaign_id": 999, "stage": "revalidation", "status": "CANCELED",
            "created_at": "2026-01-02",
        },
    ]
    quota = {
        "preprocess_retries": [{
            "sequence": "quota", "source_request_id": 10,
            "retry_request_id": 11, "status": "PENDING",
        }],
    }

    selected = pending._select_candidates(
        {"id": 491}, quota,
        {"preprocess": preprocess, "reconstruction": [], "all_requests": all_requests},
    )

    assert [(item["sequence"], item["candidate_reason"]) for item in selected] == [
        ("ordinary", "NO_SUCCESSFUL_CAMPAIGN_PREPROCESS"),
        ("quota", "QUOTA_RETRY_MISSING_FACE_VIDEOS"),
    ]


def test_candidate_selection_rejects_quota_manifest_lineage_mismatch():
    rows = [
        _request(10, "quota", "SUCCEEDED", "2026-01-01", run_status="SUCCEEDED"),
        _request(11, "quota", "PENDING", "2026-01-02", reason="wrong"),
    ]
    with pytest.raises(ValueError, match="does not match database lineage"):
        pending._select_candidates(
            {"id": 491},
            {"preprocess_retries": [{
                "sequence": "quota", "source_request_id": 10,
                "retry_request_id": 11,
            }]},
            {"preprocess": rows, "reconstruction": [], "all_requests": rows},
        )


def _inspected(client, monkeypatch):
    root = "root/data_output/sequence"
    candidate = {
        "sequence": "sequence",
        "candidate_reason": "QUOTA_RETRY_MISSING_FACE_VIDEOS",
        "request_id": 11,
        "request_status": "PENDING",
        "request_pipeline_version": "1.6.29",
        "request_created_at": "2026-01-02",
        "quota_retry": {"source_request_id": 10, "retry_request_id": 11},
        "quota_source_request_status": "SUCCEEDED",
        "successful_preprocess_request_ids": [10],
        "reconstruction_runs": [{
            "request_id": 50,
            "reconstruction_run_id": 60,
            "reconstruction_run_status": "FAILED",
            "output_uri": "swift://host/account/bucket/root/data_output/sequence/reconstruction_50",
            "pipeline_version": "1.6.28",
            "details": "task_failed: upload_hitl",
            "workflow_name": "reconstruction",
        }],
    }
    monkeypatch.setattr(
        pending.cleanup, "_client",
        lambda url: (client, "bucket", url.split("/bucket/", 1)[1].rstrip("/")),
    )
    return pending._inspect_candidate(
        candidate, client=client, bucket="bucket", output_root="root/data_output",
    ), root


def test_storage_inventory_and_apply_preserve_assets_metrics_and_unknowns(monkeypatch):
    root = "root/data_output/sequence"
    metric = b'{"status":"FAIL"}'
    protected = {
        f"{root}/mv_preprocess/edex": b"edex",
        f"{root}/mv_preprocess/videos/front.mp4": b"video",
        f"{root}/mv_preprocess/labeled_bboxes/front.json": b"{}",
        f"{root}/mv_preprocess/object_bbox_source.txt": b"manual_labeled_bboxes",
        f"{root}/mv_preprocess/object_mesh/output_aligned.glb": b"mesh",
    }
    heavy = {
        f"{root}/rosbag_to_edex/raw.bin": b"raw" * 10,
        f"{root}/mv_preprocess/images/front.h5": b"rgb" * 20,
        f"{root}/face_detector/images/front.h5": b"face" * 30,
        f"{root}/reconstruction_50/check_accuracy/check_accuracy.json": metric,
        f"{root}/reconstruction_50/foundation_pose/poses.npy": b"pose" * 15,
    }
    unknown = {f"{root}/reconstruction_orphan/unknown.bin": b"keep"}
    client = Client({**protected, **heavy, **unknown})
    record, root = _inspected(client, monkeypatch)
    report = {
        "schema": pending.SCHEMA,
        "generated_at": "now",
        "campaign_id": 491,
        "campaign_name": "campaign",
        "storage": pending._storage_summary([record]),
        "records": [record],
    }
    report["audit_sha256"] = pending._payload_sha256(report)

    assert record["reclaimable_bytes"] == sum(map(len, heavy.values()))
    assert record["retained_bytes"] == sum(map(len, {**protected, **unknown}.values()))
    assert report["storage"]["by_category"]["face_detector"]["bytes"] == len(
        heavy[f"{root}/face_detector/images/front.h5"]
    )

    result = pending._apply_record(report, record, client=client, bucket="bucket")

    assert result["status"] == "COMPLETE"
    assert result["actual_reclaimed_logical_bytes_this_run"] == record["reclaimable_bytes"]
    assert all(client.objects[key] == value for key, value in protected.items())
    assert client.objects[f"{root}/reconstruction_orphan/unknown.bin"] == b"keep"
    assert not set(heavy) & set(client.objects)
    copied = f"{root}/metrics/reconstruction/50/accuracy/check_accuracy.json"
    assert client.objects[copied] == metric
    assert f"{root}/{pending.COMPLETE_NAME}" in client.objects
    assert f"{root}/{pending.PENDING_NAME}" not in client.objects

    repeated = pending._apply_record(report, record, client=client, bucket="bucket")
    assert repeated["status"] == "ALREADY_COMPLETE"
    assert repeated["idempotent_noop"] is True


def test_apply_refuses_retained_storage_drift(monkeypatch):
    root = "root/data_output/sequence"
    client = Client({
        f"{root}/mv_preprocess/videos/front.mp4": b"video",
        f"{root}/mv_preprocess/images/front.h5": b"rgb",
    })
    record, _root = _inspected(client, monkeypatch)
    report = {
        "schema": pending.SCHEMA,
        "campaign_id": 491,
        "campaign_name": "campaign",
        "storage": pending._storage_summary([record]),
        "records": [record],
    }
    report["audit_sha256"] = pending._payload_sha256(report)
    client.objects[f"{root}/mv_preprocess/videos/front.mp4"] = b"changed"

    with pytest.raises(RuntimeError, match="protected storage changed"):
        pending._apply_record(report, record, client=client, bucket="bucket")


def test_apply_preserves_new_unrecognized_retained_object(monkeypatch):
    root = "root/data_output/sequence"
    client = Client({
        f"{root}/mv_preprocess/videos/front.mp4": b"video",
        f"{root}/mv_preprocess/images/front.h5": b"rgb",
    })
    record, _root = _inspected(client, monkeypatch)
    report = {
        "schema": pending.SCHEMA,
        "campaign_id": 491,
        "campaign_name": "campaign",
        "storage": pending._storage_summary([record]),
        "records": [record],
    }
    report["audit_sha256"] = pending._payload_sha256(report)
    unknown = f"{root}/legacy_unknown/result.bin"
    client.objects[unknown] = b"preserve me"

    result = pending._apply_record(report, record, client=client, bucket="bucket")

    assert result["status"] == "COMPLETE"
    assert client.objects[unknown] == b"preserve me"


def test_apply_refuses_new_deletable_object(monkeypatch):
    root = "root/data_output/sequence"
    client = Client({f"{root}/mv_preprocess/images/front.h5": b"rgb"})
    record, _root = _inspected(client, monkeypatch)
    report = {
        "schema": pending.SCHEMA,
        "campaign_id": 491,
        "campaign_name": "campaign",
        "storage": pending._storage_summary([record]),
        "records": [record],
    }
    report["audit_sha256"] = pending._payload_sha256(report)
    client.objects[f"{root}/face_detector/images/front.h5"] = b"new"

    with pytest.raises(RuntimeError, match="new deletable object"):
        pending._apply_record(report, record, client=client, bucket="bucket")


def test_apply_refuses_changed_audited_deletion(monkeypatch):
    root = "root/data_output/sequence"
    key = f"{root}/mv_preprocess/images/front.h5"
    client = Client({key: b"rgb"})
    record, _root = _inspected(client, monkeypatch)
    report = {
        "schema": pending.SCHEMA,
        "campaign_id": 491,
        "campaign_name": "campaign",
        "storage": pending._storage_summary([record]),
        "records": [record],
    }
    report["audit_sha256"] = pending._payload_sha256(report)
    client.objects[key] = b"changed"

    with pytest.raises(RuntimeError, match="audited deletion key changed"):
        pending._apply_record(report, record, client=client, bucket="bucket")


def test_audit_hash_detects_tampering():
    report = {"schema": pending.SCHEMA, "records": []}
    report["audit_sha256"] = pending._payload_sha256(report)
    pending._validate_audit(report)
    report["records"].append({"sequence": "tampered"})
    with pytest.raises(ValueError, match="SHA-256"):
        pending._validate_audit(report)


def test_list_retries_css_throttling(monkeypatch):
    attempts = []

    def throttled(*_args):
        attempts.append(1)
        if len(attempts) < 3:
            raise ClientError(
                {
                    "Error": {"Code": "429", "Message": "Too Many Requests"},
                    "ResponseMetadata": {"HTTPStatusCode": 429},
                },
                "ListObjectsV2",
            )
        return [{"key": "root/item", "size": 1, "etag": "etag"}]

    monkeypatch.setattr(pending.cleanup, "_list", throttled)
    monkeypatch.setattr(pending.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(pending.random, "random", lambda: 0)

    assert pending._list_with_backoff(object(), "bucket", "root") == [
        {"key": "root/item", "size": 1, "etag": "etag"}
    ]
    assert len(attempts) == 3


def test_apply_retries_stale_post_delete_listing(monkeypatch):
    root = "root/data_output/sequence"
    heavy_key = f"{root}/mv_preprocess/images/front.h5"
    client = StaleDeleteListingClient({
        f"{root}/mv_preprocess/videos/front.mp4": b"video",
        heavy_key: b"rgb",
    })
    record, _root = _inspected(client, monkeypatch)
    report = {
        "schema": pending.SCHEMA,
        "campaign_id": 491,
        "campaign_name": "campaign",
        "storage": pending._storage_summary([record]),
        "records": [record],
    }
    report["audit_sha256"] = pending._payload_sha256(report)
    monkeypatch.setattr(pending.time, "sleep", lambda _seconds: None)

    result = pending._apply_record(report, record, client=client, bucket="bucket")

    assert result["status"] == "COMPLETE"
    assert heavy_key not in client.objects
    assert client.delete_calls >= 2


def test_apply_recovers_delayed_complete_marker_visibility(monkeypatch):
    root = "root/data_output/sequence"
    heavy_key = f"{root}/mv_preprocess/images/front.h5"
    client = Client({
        f"{root}/mv_preprocess/videos/front.mp4": b"video",
        heavy_key: b"rgb",
    })
    record, _root = _inspected(client, monkeypatch)
    report = {
        "schema": pending.SCHEMA,
        "campaign_id": 491,
        "campaign_name": "campaign",
        "storage": pending._storage_summary([record]),
        "records": [record],
    }
    report["audit_sha256"] = pending._payload_sha256(report)
    client.objects.pop(heavy_key)
    visible_complete = {
        "schema": pending.COMPLETE_SCHEMA,
        "status": "COMPLETE",
        "audit_sha256": report["audit_sha256"],
        "sequence": "sequence",
    }
    reads = iter((None, None, visible_complete, None))
    monkeypatch.setattr(pending, "_optional_json", lambda *_a, **_k: next(reads))
    monkeypatch.setattr(pending.time, "sleep", lambda _seconds: None)

    result = pending._apply_record(report, record, client=client, bucket="bucket")

    assert result["status"] == "ALREADY_COMPLETE"
    assert result["actual_reclaimed_logical_bytes_total"] == record["reclaimable_bytes"]


def test_apply_wrapper_revalidates_and_dispatches_exact_audit(monkeypatch, tmp_path):
    record = {
        "sequence": "sequence", "request_id": 11, "request_status": "PENDING",
        "candidate_reason": "NO_SUCCESSFUL_CAMPAIGN_PREPROCESS",
        "reclaimable_bytes": 10,
    }
    report = {
        "schema": pending.SCHEMA,
        "campaign_id": 491,
        "campaign_name": "campaign",
        "quota_retry_plan_sha256": "quota-sha",
        "records": [record],
    }
    report["audit_sha256"] = pending._payload_sha256(report)
    campaign = {"id": 491, "name": "campaign", "dataset": "dataset"}
    monkeypatch.setattr(pending.db, "get_campaign", lambda *_a, **_k: campaign)
    monkeypatch.setattr(
        pending, "_load_quota_plan", lambda *_a, **_k: ({}, "quota-sha"),
    )
    monkeypatch.setattr(pending, "_database_snapshot", lambda *_a, **_k: {})
    monkeypatch.setattr(
        pending, "_select_candidates", lambda *_a, **_k: [record],
    )
    monkeypatch.setattr(
        pending, "_output_root",
        lambda *_a, **_k: (object(), "bucket", "root", "swift://root"),
    )
    monkeypatch.setattr(
        pending, "_apply_record",
        lambda *_a, **_k: {
            "status": "COMPLETE", "sequence": "sequence",
            "actual_reclaimed_logical_bytes_total": 10,
            "actual_reclaimed_logical_bytes_this_run": 10,
        },
    )

    result = pending.apply_audit(
        report, quota_retry_plan=tmp_path / "quota.json", db_path="db", config={},
    )

    assert result["completed_sequence_count"] == 1
    assert result["verified_reclaimed_logical_bytes"] == 10
    assert result["variance_bytes"] == 0
    assert result["actual_reclaimed_logical_bytes_this_run"] == 10
    assert result["variance_bytes"] == 0
