import hashlib
import io
from pathlib import Path
import sys
import tarfile

import pytest


WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

from migration import migrate_weights_to_css as migration


class MissingObject(Exception):
    response = {"Error": {"Code": "NoSuchKey"}}


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.events = []

    def get_object(self, *, Bucket, Key):
        del Bucket
        if Key not in self.objects:
            raise MissingObject(Key)
        body = self.objects[Key]
        return {"Body": io.BytesIO(body), "ContentLength": len(body)}

    def upload_file(self, filename, bucket, key, ExtraArgs=None):
        del bucket, ExtraArgs
        self.events.append(("upload", key))
        self.objects[key] = Path(filename).read_bytes()

    def put_object(self, *, Bucket, Key, Body, **kwargs):
        del Bucket, kwargs
        self.events.append(("put", Key))
        self.objects[Key] = Body


def test_release_uses_tao_foundationpose_assets():
    foundation_pose_paths = [
        spec.path for spec in migration.PAYLOAD_SPECS
        if spec.path.startswith("foundation_pose/")
    ]
    assert migration.DEFAULT_RELEASE == "20260722"
    assert foundation_pose_paths == [
        "foundation_pose/nvidia_tensorrt/deployable_v1.0/refiner_net.onnx",
        "foundation_pose/nvidia_tensorrt/deployable_v1.0/score_net.onnx",
        "foundation_pose/nvidia_tensorrt/deployable_v1.0/manifest.json",
    ]


def _prepared(tmp_path: Path, name: str = "model.bin", data: bytes = b"weights"):
    path = tmp_path / name
    path.write_bytes(data)
    return migration.PreparedPayload(
        path=name,
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        source_path=name,
        role="test",
        local_path=path,
    )


def test_dinov3_archive_is_deterministic_and_excludes_generated_files(tmp_path):
    source = tmp_path / "dinov3"
    (source / "models").mkdir(parents=True)
    (source / "models" / "backbone.py").write_text("MODEL = True\n")
    (source / ".git").mkdir()
    (source / ".git" / "index").write_bytes(b"do not publish")
    (source / "__pycache__").mkdir()
    (source / "__pycache__" / "x.pyc").write_bytes(b"cache")
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"

    migration.create_deterministic_dinov3_archive(source, first)
    migration.create_deterministic_dinov3_archive(source, second)

    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first, "r:gz") as archive:
        names = archive.getnames()
    assert "facebookresearch_dinov3_main/models/backbone.py" in names
    assert not any(".git" in name or "__pycache__" in name for name in names)


def test_prepare_release_selects_only_explicit_payloads_and_manifest_fields(tmp_path):
    root = tmp_path / "weights"
    root.mkdir()
    (root / "keep.bin").write_bytes(b"keep")
    (root / "engine.engine").write_bytes(b"exclude")
    dino = root / migration.DINO_SOURCE
    dino.mkdir(parents=True)
    (dino / "hubconf.py").write_text("def model(): pass\n")
    specs = (migration.PayloadSpec("keep.bin", "module/keep.bin", "runtime"),)

    payloads, manifest, manifest_bytes = migration.prepare_release(
        root, tmp_path / "work", release="test", payload_specs=specs
    )

    assert [payload.path for payload in payloads] == [
        "module/keep.bin",
        "sam3d_body/dinov3_repo.tar.gz",
    ]
    assert manifest["schema"] == migration.MANIFEST_SCHEMA
    assert manifest["release"] == "test"
    assert manifest["complete"] is True
    assert manifest["file_count"] == 2
    assert b"engine.engine" not in manifest_bytes


def test_publish_is_dry_run_by_default_and_commits_manifest_last(tmp_path):
    s3 = FakeS3()
    payload = _prepared(tmp_path)
    manifest = b'{"complete":true}\n'

    result = migration.publish_release(
        s3, "recordings", "release", [payload], manifest, apply=False
    )
    assert result["dry_run"] is True
    assert s3.objects == {}

    result = migration.publish_release(
        s3, "recordings", "release", [payload], manifest, apply=True
    )
    assert result["verified"] == ["model.bin"]
    assert s3.events[-1] == ("put", "release/manifest.json")
    assert s3.objects["release/model.bin"] == b"weights"


def test_publish_resumes_matching_object_and_refuses_conflicts(tmp_path):
    payload = _prepared(tmp_path)
    manifest = b'{"complete":true}\n'
    s3 = FakeS3()
    s3.objects["release/model.bin"] = b"weights"

    result = migration.publish_release(
        s3, "recordings", "release", [payload], manifest, apply=True
    )
    assert result["resumed"] == ["model.bin"]
    assert not any(event[0] == "upload" for event in s3.events)

    conflicting = FakeS3()
    conflicting.objects["release/model.bin"] = b"other"
    with pytest.raises(RuntimeError, match="Conflicting immutable object"):
        migration.publish_release(
            conflicting, "recordings", "release", [payload], manifest, apply=True
        )


def test_publish_refuses_conflicting_committed_manifest(tmp_path):
    s3 = FakeS3()
    s3.objects["release/manifest.json"] = b"old manifest"
    with pytest.raises(RuntimeError, match="conflicting"):
        migration.publish_release(
            s3,
            "recordings",
            "release",
            [_prepared(tmp_path)],
            b"new manifest",
            apply=True,
        )


def test_parse_swift_url():
    assert migration.parse_swift_url(
        "swift://pdx.s8k.io/AUTH_team-isaac/recordings/path/to/release"
    ) == ("https://pdx.s8k.io", "recordings", "path/to/release")
