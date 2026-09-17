import json
import hashlib

import pytest

from v2d.mv.postprocess.lib.verify_bbox_manifest import (
    logical_bbox_manifest,
    verify_bbox_manifest,
)


def test_bbox_manifest_round_trip_and_change_detection(tmp_path):
    (tmp_path / "b.json").write_text(json.dumps({"bbox": [1, 2, 3, 4]}))
    (tmp_path / "a.json").write_text(json.dumps({"bbox": [5, 6, 7, 8]}))

    entries, digest = logical_bbox_manifest(tmp_path)
    assert [entry["name"] for entry in entries] == ["a.json", "b.json"]
    assert verify_bbox_manifest(tmp_path, digest) == entries

    (tmp_path / "a.json").write_text(json.dumps({"bbox": [0, 0, 1, 1]}))
    with pytest.raises(ValueError, match="changed after reconstruction submission"):
        verify_bbox_manifest(tmp_path, digest)


def test_bbox_manifest_rejects_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="No labeled bbox JSON"):
        logical_bbox_manifest(tmp_path)


def test_transport_manifest_identity_and_local_content_are_both_verified(tmp_path):
    (tmp_path / "a.json").write_text('{"bbox":[1,2,3,4]}')
    logical, _ = logical_bbox_manifest(tmp_path)
    manifest = [{**logical[0], "etag": "css-object-etag"}]
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()

    assert verify_bbox_manifest(tmp_path, digest, manifest) == logical
    with pytest.raises(ValueError, match="manifest does not match"):
        verify_bbox_manifest(tmp_path, "0" * 64, manifest)

    manifest[0]["size"] += 1
    changed_canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    changed_digest = hashlib.sha256(changed_canonical.encode()).hexdigest()
    with pytest.raises(ValueError, match="content changed"):
        verify_bbox_manifest(tmp_path, changed_digest, manifest)
