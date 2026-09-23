import io
import json
from pathlib import Path
import sys
import types

import pytest


WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

from orchestration import export_commit


def test_supported_schemas_include_v3_and_preserve_v1_v2_compatibility():
    assert export_commit.SUPPORTED_SCHEMAS == {
        "v2d.mv_hoi.export_commit.v1",
        "v2d.mv_hoi.export_commit.v2",
        "v2d.mv_hoi.export_commit.v3",
        "v2d.mv_hoi.revalidation_export_commit.v1",
        "v2d.mv_hoi.revalidation_export_commit.v2",
        "v2d.mv_hoi.revalidation_export_commit.v3",
    }


class _Paginator:
    def paginate(self, **kwargs):
        assert kwargs["Bucket"] == "container"
        return [{"Contents": [
            {"Key": "root/sequence/payload.bin", "Size": 3},
            {"Key": "root/sequence/commit.json", "Size": 100},
        ]}]


class _Client:
    def __init__(self, commit):
        self.commit = commit

    def get_object(self, **kwargs):
        assert kwargs == {"Bucket": "container", "Key": "root/sequence/commit.json"}
        return {"Body": io.BytesIO(json.dumps(self.commit).encode())}

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return _Paginator()


def test_publish_requires_distinct_candidate_and_destination():
    with pytest.raises(ValueError, match="must differ"):
        export_commit.publish_remote_export_commit(
            "swift://storage.example.com/AUTH_example/container/root/sequence",
            "swift://storage.example.com/AUTH_example/container/root/sequence/",
        )


def test_remote_commit_uses_swift_container_and_validates_exact_payload(monkeypatch):
    commit = {
        "schema": "v2d.mv_hoi.export_commit.v1",
        "complete": True,
        "file_count": 1,
        "total_bytes": 3,
        "files": [{"path": "payload.bin", "size": 3, "sha256": "unused"}],
    }
    monkeypatch.setattr(
        export_commit.boto3, "client", lambda *args, **kwargs: _Client(commit),
    )

    assert export_commit.verify_remote_export_commit(
        "swift://storage.example.com/AUTH_example/container/root/sequence/"
    ) == commit


def test_publish_replaces_destination_and_copies_commit_last(monkeypatch):
    commit = {
        "schema": "v2d.mv_hoi.export_commit.v1",
        "complete": True,
        "file_count": 1,
        "total_bytes": 3,
        "files": [{"path": "payload.bin", "size": 3, "sha256": "unused"}],
    }
    storage = {
        "request_1/payload.bin": b"new",
        "request_1/commit.json": json.dumps(commit).encode(),
        "destination/stale.bin": b"stale",
        "destination/commit.json": b"old",
    }
    operations = []

    class Client:
        meta = types.SimpleNamespace(endpoint_url="https://storage.example.com")

        def get_object(self, *, Bucket, Key):
            assert Bucket == "container"
            return {"Body": io.BytesIO(storage[Key])}

        def get_paginator(self, name):
            assert name == "list_objects_v2"

            class Paginator:
                def paginate(self, *, Bucket, Prefix):
                    assert Bucket == "container"
                    yield {"Contents": [
                        {"Key": key, "Size": len(value)}
                        for key, value in sorted(storage.items())
                        if key.startswith(Prefix)
                    ]}

            return Paginator()

        def delete_objects(self, *, Bucket, Delete):
            assert Bucket == "container"
            for item in Delete["Objects"]:
                operations.append(("delete", item["Key"]))
                storage.pop(item["Key"], None)

        def delete_object(self, *, Bucket, Key):
            assert Bucket == "container"
            operations.append(("delete", Key))
            storage.pop(Key, None)

        def list_objects_v2(self, *, Bucket, Prefix, MaxKeys):
            assert Bucket == "container"
            assert MaxKeys == 1
            contents = [
                {"Key": key, "Size": len(value)}
                for key, value in sorted(storage.items())
                if key.startswith(Prefix)
            ][:1]
            return {"Contents": contents} if contents else {}

        def copy_object(self, *, Bucket, Key, CopySource):
            assert Bucket == "container"
            operations.append(("copy", Key))
            storage[Key] = storage[CopySource["Key"]]

    monkeypatch.setattr(
        export_commit.boto3, "client", lambda *args, **kwargs: Client(),
    )

    assert export_commit.publish_remote_export_commit(
        "swift://storage.example.com/AUTH_example/container/request_1",
        "swift://storage.example.com/AUTH_example/container/destination",
    ) == commit
    assert "destination/stale.bin" not in storage
    assert storage["destination/payload.bin"] == b"new"
    assert any(key.startswith("request_1/") for key in storage)
    assert export_commit.cleanup_promoted_export_candidate(
        "swift://storage.example.com/AUTH_example/container/request_1",
        "swift://storage.example.com/AUTH_example/container/destination",
        expected_commit=commit,
    )["deleted_object_count"] == 2
    assert not any(key.startswith("request_1/") for key in storage)
    assert operations[-1] == ("delete", "request_1/commit.json")
