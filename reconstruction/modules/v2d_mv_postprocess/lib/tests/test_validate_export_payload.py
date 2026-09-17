import sys
from datetime import datetime
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest


LIB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIB_DIR))

import validate_export_payload as validation
import interaction_trim


def test_video_frame_count_uses_declared_count_and_decodes_one_probe(
    tmp_path, monkeypatch,
):
    path = tmp_path / "video.mp4"
    path.write_bytes(b"container")
    decoded = 0

    class Stream:
        frames = 123

    class Container:
        streams = type("Streams", (), {"video": [Stream()]})()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def decode(self, _stream):
            nonlocal decoded
            decoded += 1
            yield object()
            raise AssertionError("validator decoded more than one probe frame")

    monkeypatch.setattr(validation.av, "open", lambda _path: Container())

    assert validation._video_frame_count(path) == 123
    assert decoded == 1


def _touch(path: Path, payload: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _trim_manifest(frame_count: int = 2, start: int = 0) -> dict:
    manifest = {
        "schema": interaction_trim.TRIM_SCHEMA,
        "source_frame_count": frame_count + start,
        "export_source_start_frame": start,
        "export_source_end_frame": frame_count + start,
        "export_frame_count": frame_count,
        "contact_frame_source": start,
        "first_contact_frame_source": start,
        "last_contact_frame_source": frame_count + start - 1,
        "trimmed_prefix_frames": start,
        "trimmed_suffix_frames": 0,
        "trimmed_suffix_seconds": 0.0,
    }
    manifest["decision_sha256"] = interaction_trim._canonical_sha256(manifest)
    return manifest


def test_exact_camera_files_requires_every_camera_and_rejects_extras(tmp_path):
    for camera in validation.RGB_CAMERAS:
        _touch(tmp_path / f"{camera}.mp4")
    files = validation._exact_camera_files(tmp_path, ".mp4", validation.RGB_CAMERAS)
    assert set(files) == set(validation.RGB_CAMERAS)

    (tmp_path / "front_stereo_camera_right.mp4").unlink()
    with pytest.raises(ValueError, match="missing"):
        validation._exact_camera_files(tmp_path, ".mp4", validation.RGB_CAMERAS)

    _touch(tmp_path / "front_stereo_camera_right.mp4")
    _touch(tmp_path / "unexpected_camera.mp4")
    with pytest.raises(ValueError, match="unexpected"):
        validation._exact_camera_files(tmp_path, ".mp4", validation.RGB_CAMERAS)


def test_same_frame_identity_rejects_unbalanced_stems():
    records = {
        "a": {"n_frames": 2, "stems": ["0", "1"]},
        "b": {"n_frames": 2, "stems": ["0", "2"]},
    }
    with pytest.raises(ValueError, match="not balanced"):
        validation._same_frame_identity("images", records)


def test_ffv1_group_reuses_known_payload_hash(tmp_path, monkeypatch):
    directory = tmp_path / "images"
    metadata_path = _touch(directory / "camera.h5")
    sidecar_path = _touch(directory / "camera.ffv1.digest.mkv")
    digest = "a" * 64
    monkeypatch.setattr(
        validation,
        "read_ffv1_metadata",
        lambda _path: {
            "kind": "rgb",
            "sidecar_path": sidecar_path,
            "sidecar_basename": sidecar_path.name,
            "sidecar_sha256": digest,
        },
    )
    monkeypatch.setattr(
        validation,
        "verify_ffv1_sidecar",
        lambda *_args, **_kwargs: pytest.fail("sidecar was redundantly rehashed"),
    )

    records = validation._ffv1_group(
        tmp_path,
        directory,
        ("camera",),
        "rgb",
        {"images/camera.ffv1.digest.mkv": digest},
    )

    assert records["camera"]["sidecar_sha256"] == digest
    assert metadata_path.is_file()
    with pytest.raises(ValueError, match="does not match metadata"):
        validation._ffv1_group(
            tmp_path,
            directory,
            ("camera",),
            "rgb",
            {"images/camera.ffv1.digest.mkv": "b" * 64},
        )


def test_mask_group_accepts_balanced_legacy_png_directories(tmp_path):
    for camera in validation.LEFT_CAMERAS:
        for index in range(2):
            path = tmp_path / camera / f"{index:06d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.zeros((8, 9), dtype=np.uint8)).save(path)

    records = validation._mask_group(tmp_path, validation.LEFT_CAMERAS)

    assert set(records) == set(validation.LEFT_CAMERAS)
    assert {record["n_frames"] for record in records.values()} == {2}
    assert {tuple(record["stems"]) for record in records.values()} == {
        ("000000", "000001")
    }


def test_mask_group_rejects_missing_or_mixed_png_directories(tmp_path):
    for camera in validation.LEFT_CAMERAS[:-1]:
        _touch(tmp_path / camera / "000000.png")
    with pytest.raises(ValueError, match="missing"):
        validation._mask_group(tmp_path, validation.LEFT_CAMERAS)

    _touch(tmp_path / f"{validation.LEFT_CAMERAS[-1]}.h5")
    with pytest.raises(ValueError, match="Mixed H5 and PNG-directory"):
        validation._mask_group(tmp_path, validation.LEFT_CAMERAS)


def test_load_metadata_accepts_yaml_timestamps(tmp_path):
    metadata_path = _touch(
        tmp_path / "hoi_metadata.yaml",
        b"recorded_at: 2026-07-21T23:01:42Z\nobject:\n  id: luggage\n",
    )

    metadata = validation._load_metadata(metadata_path)

    assert metadata["recorded_at"] == datetime.fromisoformat("2026-07-21T23:01:42+00:00")
    assert metadata["object"] == {"id": "luggage"}


def test_soma_validation_accepts_legacy_payload_without_optional_bone_lengths():
    frame_count = 6
    legacy = {
        "poses": np.zeros((frame_count, 77, 3)),
        "transl": np.zeros((frame_count, 3)),
        "identity_coeffs": np.zeros((frame_count, 45)),
        "scale_params": np.zeros((frame_count, 68)),
    }

    validation._validate_soma_temporal_arrays(legacy, frame_count)

    extended = dict(legacy)
    extended["bone_length_flexibles"] = np.zeros((frame_count, 6))
    validation._validate_soma_temporal_arrays(extended, frame_count)

    extended["bone_length_flexibles"] = np.zeros((frame_count - 1, 6))
    with pytest.raises(ValueError, match="bone_length_flexibles differs from RGB"):
        validation._validate_soma_temporal_arrays(extended, frame_count)


def test_non_camera_assets_require_symmetry_metadata(tmp_path):
    trim = _trim_manifest(frame_count=1)
    _touch(
        tmp_path / "hoi_metadata.yaml",
        (
            "frame_count: 1\n"
            "interaction_trim:\n"
            f"  decision_sha256: {trim['decision_sha256']}\n"
        ).encode(),
    )
    _touch(
        tmp_path / "edex",
        json.dumps({
            "frame_start": 0,
            "frame_end": 1,
            "cameras": [{} for _ in validation.RGB_CAMERAS],
        }).encode(),
    )
    import trimesh
    mesh_path = tmp_path / "object_mesh" / "output_aligned.glb"
    mesh_path.parent.mkdir(parents=True)
    trimesh.creation.box().export(mesh_path)

    with pytest.raises(ValueError, match="output_symmetry.json"):
        validation._validate_non_camera_assets(tmp_path, 1, trim)


def test_validate_payload_checks_all_groups_before_final_assets(tmp_path, monkeypatch):
    calls = []

    def fake_ffv1(_root, path, cameras, kind, _known_payload_sha256=None):
        calls.append((path.name, len(cameras), kind))
        return {
            camera: {"n_frames": 2, "stems": ["000000", "000001"]}
            for camera in cameras
        }

    def fake_mask(path, cameras):
        calls.append((path.name, len(cameras), "mask"))
        return {
            camera: {"n_frames": 2, "stems": ["000000", "000001"]}
            for camera in cameras
        }

    monkeypatch.setattr(validation, "_ffv1_group", fake_ffv1)
    monkeypatch.setattr(validation, "_mask_group", fake_mask)
    monkeypatch.setattr(
        validation, "_exact_camera_files",
        lambda path, _suffix, cameras: {camera: path / f"{camera}.mp4" for camera in cameras},
    )
    monkeypatch.setattr(validation, "_video_frame_count", lambda _path: 2)
    monkeypatch.setattr(
        validation,
        "_validate_non_camera_assets",
        lambda _root, count, _trim, _known_payload_sha256=None: {
            "frames": count
        },
    )
    _touch(
        tmp_path / "interaction_trim.json",
        json.dumps(_trim_manifest()).encode(),
    )

    result = validation.validate_export_payload(tmp_path)

    assert calls == [
        ("images", 8, "rgb"),
        ("images_anonymized", 8, "rgb"),
        ("depth", 4, "depth"),
        ("object_masks", 4, "mask"),
        ("human_masks", 4, "mask"),
    ]
    assert result["camera_counts"] == {
        "images": 8, "videos": 8,
        "images_anonymized": 8, "videos_anonymized": 8,
        "depth": 4, "object_masks": 4, "human_masks": 4,
    }
