from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import sys

import pytest
from botocore.exceptions import ClientError


WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

from migration import cleanup_backlog_legacy_flat_intermediates as legacy


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

    def head_object(self, *, Bucket, Key):
        assert Bucket == "bucket"
        if Key not in self.objects:
            raise ClientError(
                {
                    "Error": {"Code": "NoSuchKey", "Message": "missing"},
                    "ResponseMetadata": {"HTTPStatusCode": 404},
                },
                "HeadObject",
            )
        value = self.objects[Key]
        return {"ContentLength": len(value), "ETag": f'"etag-{len(value)}"'}

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


def _item(key: str, value: bytes) -> dict:
    return {"key": key, "size": len(value), "etag": f"etag-{len(value)}"}


def _fixture(tmp_path):
    root = "root/data_output/sequence"
    objects = {
        f"{root}/foundation_stereo/front/depth.h5": b"depth",
        f"{root}/grounding_dino/front.json": b'{"boxes": []}',
        f"{root}/check_accuracy/check_accuracy.json": b'{"status": "FAIL"}',
        f"{root}/mv_preprocess/videos/front.mp4": b"video",
        f"{root}/mv_preprocess/labeled_bboxes/front.json": b"{}",
        f"{root}/mv_preprocess/object_bbox_source.txt": b"manual_labeled_bboxes",
    }
    client = Client(objects)
    legacy_objects = [
        _item(key, value) for key, value in objects.items()
        if key.split(root + "/", 1)[1].split("/", 1)[0] in legacy.LEGACY_PREFIXES
    ]
    protected = [
        _item(key, value) for key, value in objects.items()
        if key not in {item["key"] for item in legacy_objects}
    ]
    run = {
        "sequence_name": "sequence", "reconstruction_run_id": 50,
        "run_status": "FAILED", "pipeline_version": "1.5.1",
        "legacy_source_table": "pipelines", "legacy_source_id": 10,
        "request_id": None, "created_at": "2026-01-01",
        "workflow_name": "legacy-workflow",
    }
    shard = {
        "schema": legacy.SHARD_SCHEMA,
        "sequence": "sequence",
        "sequence_root": root,
        "legacy_runs": [run],
        "objects": legacy_objects,
        "protected_objects": protected,
        "protected_identity_sha256": legacy.cleanup._identity(protected),
        "evidence_keys": [f"{root}/check_accuracy/check_accuracy.json"],
    }
    shard_path = tmp_path / "sequence.json.gz"
    shard_sha = legacy._write_shard(shard_path, shard)
    record = {
        "sequence": "sequence", "legacy_run_count": 1,
        "legacy_run_ids": [50], "latest_legacy_run_id": 50,
        "object_count": len(legacy_objects),
        "logical_bytes": sum(item["size"] for item in legacy_objects),
        "evidence_object_count": 1, "by_prefix": {},
        "shard": str(shard_path), "shard_sha256": shard_sha,
    }
    report = {
        "schema": legacy.SCHEMA, "campaign_id": 491,
        "campaign_name": "campaign", "output_root_uri": "swift://root",
        "records": [record],
    }
    report["audit_sha256"] = legacy._sha256(report)
    return client, objects, shard_path, record, report


def test_legacy_prefixes_include_historical_grounding_output():
    assert "grounding_dino" in legacy.LEGACY_PREFIXES


def test_legacy_lineage_is_limited_to_source_campaign_members():
    source = {"records": [{"sequence": "selected"}]}
    rows = [
        {"sequence_name": "selected", "run_status": "FAILED", "id": 1},
        {"sequence_name": "other", "run_status": "SUCCEEDED", "id": 2},
    ]
    assert list(legacy._legacy_by_sequence(source, rows)) == ["selected"]


def test_apply_deletes_only_flat_tasks_and_preserves_evidence_and_labels(tmp_path):
    client, original, _shard, record, report = _fixture(tmp_path)

    result = legacy._apply_record(report, record, client=client, bucket="bucket")

    assert result["status"] == "COMPLETE"
    assert result["actual_reclaimed_logical_bytes_total"] == record["logical_bytes"]
    assert not any(
        key.split("root/data_output/sequence/", 1)[1].split("/", 1)[0]
        in legacy.LEGACY_PREFIXES
        for key in client.objects
        if key.startswith("root/data_output/sequence/")
    )
    for suffix in (
        "mv_preprocess/videos/front.mp4",
        "mv_preprocess/labeled_bboxes/front.json",
        "mv_preprocess/object_bbox_source.txt",
    ):
        key = f"root/data_output/sequence/{suffix}"
        assert client.objects[key] == original[key]
    copied = (
        "root/data_output/sequence/metrics/reconstruction/50/legacy_flat/"
        "check_accuracy/check_accuracy.json"
    )
    assert client.objects[copied] == b'{"status": "FAIL"}'
    assert "root/data_output/sequence/backlog_legacy_flat_cleanup.pending.json" not in (
        client.objects
    )

    repeated = legacy._apply_record(report, record, client=client, bucket="bucket")
    assert repeated["status"] == "ALREADY_COMPLETE"
    assert repeated["idempotent_noop"] is True


def test_apply_refuses_changed_legacy_object(tmp_path):
    client, _original, _shard, record, report = _fixture(tmp_path)
    key = "root/data_output/sequence/foundation_stereo/front/depth.h5"
    client.objects[key] = b"changed"

    with pytest.raises(RuntimeError, match="audited legacy key changed"):
        legacy._apply_record(report, record, client=client, bucket="bucket")


def test_apply_resumes_after_evidence_copy_and_source_deletion(tmp_path):
    client, _original, shard_path, record, report = _fixture(tmp_path)
    shard = legacy._read_shard(shard_path, record["shard_sha256"])
    source = "root/data_output/sequence/check_accuracy/check_accuracy.json"
    destination = (
        "root/data_output/sequence/metrics/reconstruction/50/legacy_flat/"
        "check_accuracy/check_accuracy.json"
    )
    client.objects[destination] = client.objects[source]
    for item in shard["objects"]:
        client.objects.pop(item["key"], None)
    pending_key = "root/data_output/sequence/backlog_legacy_flat_cleanup.pending.json"
    client.objects[pending_key] = json.dumps({
        "schema": legacy.PENDING_SCHEMA,
        "status": "PENDING",
        "audit_sha256": report["audit_sha256"],
    }).encode()

    result = legacy._apply_record(report, record, client=client, bucket="bucket")

    assert result["status"] == "COMPLETE"
    assert client.objects[destination] == b'{"status": "FAIL"}'
    assert pending_key not in client.objects


def test_delete_verification_retries_stale_listing(monkeypatch):
    item = {"key": "root/foundation_stereo/depth.h5", "size": 5, "etag": "etag"}
    shard = {
        "sequence": "sequence", "sequence_root": "root", "objects": [item],
    }
    listings = iter(([item], [], []))
    deleted = []
    monkeypatch.setattr(legacy, "_list_legacy", lambda *_a, **_k: next(listings))
    monkeypatch.setattr(
        legacy.pending, "_delete_with_backoff",
        lambda _client, _bucket, keys: deleted.append(list(keys)),
    )
    monkeypatch.setattr(legacy.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(legacy.random, "random", lambda: 0)

    assert legacy._delete_and_verify(object(), "bucket", shard, [item]) == []
    assert deleted == [[item["key"]], [item["key"]]]


def test_apply_heads_an_object_omitted_by_initial_listing(tmp_path, monkeypatch):
    client, _original, shard_path, record, report = _fixture(tmp_path)
    shard = legacy._read_shard(shard_path, record["shard_sha256"])
    omitted = shard["objects"][0]
    real_list = legacy._list_legacy
    calls = 0

    def list_with_initial_omission(client_arg, bucket, sequence_root):
        nonlocal calls
        calls += 1
        listed = real_list(client_arg, bucket, sequence_root)
        if calls == 1:
            return [item for item in listed if item["key"] != omitted["key"]]
        return listed

    monkeypatch.setattr(legacy, "_list_legacy", list_with_initial_omission)
    monkeypatch.setattr(legacy.time, "sleep", lambda _seconds: None)

    result = legacy._apply_record(report, record, client=client, bucket="bucket")

    assert result["status"] == "COMPLETE"
    assert result["actual_deleted_object_count_this_run"] == record["object_count"]
    assert omitted["key"] not in client.objects


def test_apply_rechecks_transient_unexpected_listing(tmp_path, monkeypatch):
    client, _original, _shard_path, record, report = _fixture(tmp_path)
    real_list = legacy._list_legacy
    unexpected = {
        "key": "root/data_output/sequence/foundation_stereo/transient.tmp",
        "size": 1,
        "etag": "transient",
    }
    calls = 0

    def list_with_transient_extra(client_arg, bucket, sequence_root):
        nonlocal calls
        calls += 1
        listed = real_list(client_arg, bucket, sequence_root)
        if calls == 1:
            return listed + [unexpected]
        return listed

    monkeypatch.setattr(legacy, "_list_legacy", list_with_transient_extra)
    monkeypatch.setattr(legacy.time, "sleep", lambda _seconds: None)

    result = legacy._apply_record(report, record, client=client, bucket="bucket")

    assert result["status"] == "COMPLETE"
    assert calls >= 4  # initial, drift recheck, and two empty confirmations


def test_shard_hash_detects_tampering(tmp_path):
    _client, _original, shard, record, _report = _fixture(tmp_path)
    with shard.open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises((ValueError, OSError, EOFError)):
        legacy._read_shard(shard, record["shard_sha256"])
