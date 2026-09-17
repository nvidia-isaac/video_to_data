import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


LIB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIB_DIR))

import finalize_revalidation_export as finalize
import interaction_trim


MESH_BYTES = b"aligned-mesh"
SYMMETRY_BYTES = b'{"alignment":{}}'


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_object_pose_pair(
    output: Path, *, poses: np.ndarray | None = None,
) -> None:
    (output / "object_mesh").mkdir(parents=True, exist_ok=True)
    poses = (
        np.arange(16, dtype=np.float32).reshape(1, 4, 4)
        if poses is None else poses
    )
    with (output / "poses.npy").open("wb") as stream:
        np.save(stream, poses, allow_pickle=False)
    (output / "object_mesh" / "output_aligned.glb").write_bytes(MESH_BYTES)
    (output / "object_mesh" / "output_symmetry.json").write_bytes(SYMMETRY_BYTES)


def _json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def _args(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("foundation_pose", "face_detector", "render_hoi_overlay"):
        (source / name).mkdir()
    commercial = tmp_path / "commercial"
    face = tmp_path / "face"
    overlay = tmp_path / "overlay"
    for path in (commercial, face, overlay):
        path.mkdir()
    commercial_poses = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    with (commercial / "poses.npy").open("wb") as stream:
        np.save(stream, commercial_poses, allow_pickle=False)
    comparison = _json(tmp_path / "comparison.json", {
        "status": "PASS",
        "comparison_pose_frame": "aligned",
        "commercial_pose_sha256": finalize._sha256(commercial / "poses.npy"),
        "mesh_sha256": _digest(MESH_BYTES),
        "symmetry_sha256": _digest(SYMMETRY_BYTES),
    })
    accuracy = _json(tmp_path / "accuracy.json", {"status": "PASS"})
    metric_failure_segments = tmp_path / "metric_failure_segments.json"
    metric_failure_segments.write_text("[]\n")
    human_silhouette = _json(tmp_path / "human_silhouette.json", {"frame_metrics": {}})
    human_chamfer = _json(tmp_path / "human_chamfer.json", {"per_camera": {}})
    silhouette = _json(
        tmp_path / "silhouette.json", {"quality_gate": {"status": "PASS"}},
    )
    failure_segments = tmp_path / "failure_segments.json"
    failure_segments.write_text("[]\n")
    return {
        "source_dir": source,
        "commercial_foundation_pose_dir": commercial,
        "anonymized_rgb_dir": face,
        "overlay_dir": overlay,
        "pose_comparison_path": comparison,
        "check_accuracy_path": accuracy,
        "metric_failure_segments_path": metric_failure_segments,
        "object_silhouette_path": silhouette,
        "human_silhouette_path": human_silhouette,
        "human_chamfer_path": human_chamfer,
        "failure_segments_path": failure_segments,
        "output_dir": tmp_path / "output",
        "request_id": 7,
        "campaign_name": "campaign",
        "sequence_name": "sequence_01",
        "pipeline_version": "1.0.0",
        "source_manifest_sha256": "a" * 64,
        "configuration_sha256": "b" * 64,
    }


def _trim_manifest() -> dict:
    manifest = {
        "schema": interaction_trim.TRIM_SCHEMA,
        "source_frame_count": 1,
        "export_source_start_frame": 0,
        "export_source_end_frame": 1,
        "export_frame_count": 1,
        "contact_frame_source": 0,
        "first_contact_frame_source": 0,
        "last_contact_frame_source": 0,
        "trimmed_prefix_frames": 0,
        "trimmed_suffix_frames": 0,
        "trimmed_suffix_seconds": 0.0,
    }
    manifest["decision_sha256"] = interaction_trim._canonical_sha256(manifest)
    return manifest


def _stub_trim(monkeypatch, observed_prepare: dict | None = None) -> None:
    manifest = _trim_manifest()
    monkeypatch.setattr(
        finalize, "detect_interaction_trim", lambda *_args, **_kwargs: manifest,
    )
    def fake_prepare(source, _output, _manifest, **kwargs):
        if observed_prepare is not None:
            observed_prepare.update(kwargs)
        return Path(source)

    monkeypatch.setattr(finalize, "prepare_trimmed_source", fake_prepare)


def test_commit_covers_all_payloads_and_detects_mutation(tmp_path, monkeypatch):
    args = _args(tmp_path)
    sha256_calls = []
    real_sha256 = finalize._sha256
    observed_export = {}

    def fake_export_sequence(**kwargs):
        observed_export.update(kwargs)
        output = Path(kwargs["output_dir"])
        (output / "images").mkdir(parents=True)
        (output / "images" / "camera.h5").write_bytes(b"metadata")
        _write_object_pose_pair(output)

    def counting_sha256(path):
        sha256_calls.append(path)
        return real_sha256(path)

    monkeypatch.setattr(finalize, "export_sequence", fake_export_sequence)
    observed_prepare = {}
    _stub_trim(monkeypatch, observed_prepare)
    monkeypatch.setattr(finalize, "_sha256", counting_sha256)
    monkeypatch.setattr(
        finalize, "validate_export_payload",
        lambda _root, **_kwargs: {
            "schema": "v2d.mv_hoi.export_validation.v2", "frame_count": 1,
        },
    )
    commit = finalize.finalize_revalidation_export(**args)
    assert commit["complete"] is True
    assert commit["authorization_type"] == "REVALIDATION"
    assert commit["sequence_name"] == "sequence_01"
    assert commit["object_pose_artifacts"]["pose_frame"] == "aligned"
    assert {item["path"] for item in commit["files"]} >= {
        "images/camera.h5", "poses.npy",
        "failure_segments.json",
        "revalidation/foundation_pose_comparison.json",
        "metrics/revalidation/7/foundation_pose/foundation_pose_comparison.json",
        "metrics/revalidation/7/failure_segments/source_failure_segments.json",
        "metrics/revalidation/7/failure_segments/export_failure_segments.json",
        "metrics/revalidation/7/manifest.json",
    }
    assert json.loads((args["output_dir"] / "failure_segments.json").read_text()) == []
    assert {
        key: commit["human_qc"][key] for key in (
            "source_index_space", "export_index_space",
            "source_failure_segment_count", "export_failure_segment_count",
        )
    } == {
        "source_index_space": "source_frames",
        "export_index_space": "export_frames",
        "source_failure_segment_count": 0,
        "export_failure_segment_count": 0,
    }
    assert len(sha256_calls) >= commit["file_count"]
    assert observed_prepare["defer_frame_archives"] is True
    assert observed_export["source_start_frame"] == 0
    assert observed_export["source_end_frame"] == 1
    assert observed_export["max_camera_workers"] >= 1
    assert finalize.verify_export_commit(args["output_dir"])["request_id"] == 7
    (args["output_dir"] / "poses.npy").write_bytes(b"p0se")
    with pytest.raises(ValueError, match="(size|SHA-256) mismatch"):
        finalize.verify_export_commit(args["output_dir"])


def test_preexisting_mounted_output_root_is_preserved(tmp_path, monkeypatch):
    args = _args(tmp_path)
    output = args["output_dir"]
    output.mkdir()
    original_rmdir = Path.rmdir

    def reject_output_rmdir(path):
        if path == output:
            raise OSError(16, "Device or resource busy", str(path))
        return original_rmdir(path)

    def fake_export_sequence(**kwargs):
        _write_object_pose_pair(Path(kwargs["output_dir"]))

    monkeypatch.setattr(Path, "rmdir", reject_output_rmdir)
    monkeypatch.setattr(finalize, "export_sequence", fake_export_sequence)
    _stub_trim(monkeypatch)
    monkeypatch.setattr(
        finalize, "validate_export_payload",
        lambda _root, **_kwargs: {"schema": "v2d.mv_hoi.export_validation.v2"},
    )

    commit = finalize.finalize_revalidation_export(**args)
    assert commit["complete"] is True
    assert output.is_dir()


def test_legacy_silhouette_gate_is_diagnostic_only(tmp_path, monkeypatch):
    args = _args(tmp_path)
    _json(args["object_silhouette_path"], {"quality_gate": {"status": "INCONCLUSIVE"}})
    monkeypatch.setattr(
        finalize,
        "export_sequence",
        lambda **kwargs: _write_object_pose_pair(Path(kwargs["output_dir"])),
    )
    _stub_trim(monkeypatch)
    monkeypatch.setattr(
        finalize,
        "validate_export_payload",
        lambda _root, **_kwargs: {"schema": "v2d.mv_hoi.export_validation.v2"},
    )

    commit = finalize.finalize_revalidation_export(**args)

    assert commit["silhouette_status"] == "INCONCLUSIVE"
    assert commit["complete"] is True


def test_pose_mesh_hash_mismatch_blocks_commit(tmp_path, monkeypatch):
    args = _args(tmp_path)

    def fake_export_sequence(**kwargs):
        _write_object_pose_pair(
            Path(kwargs["output_dir"]),
            poses=np.full((1, 4, 4), 99, dtype=np.float32),
        )

    monkeypatch.setattr(finalize, "export_sequence", fake_export_sequence)
    _stub_trim(monkeypatch)
    with pytest.raises(ValueError, match="exact retained source slice"):
        finalize.finalize_revalidation_export(**args)
    assert not (args["output_dir"] / "commit.json").exists()


def test_human_failure_segments_are_preserved_and_shifted_for_trim(
    tmp_path, monkeypatch,
):
    args = _args(tmp_path)
    source_segments = [
        {
            "id": "prefix",
            "start_frame": 0,
            "end_frame": 10,
            "failure_category": "Large jolts",
        },
        {
            "id": "crossing",
            "start_frame": 15,
            "end_frame": 30,
            "failure_category": "Mesh Penetration",
            "reason": "human annotation",
        },
        {
            "id": "retained",
            "start_frame": 60,
            "end_frame": 80,
            "failure_category": "Large misalignment/drift",
        },
    ]
    source_bytes = json.dumps(source_segments, separators=(",", ":")).encode()
    args["failure_segments_path"].write_bytes(source_bytes)

    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 100, axis=0)
    with (args["commercial_foundation_pose_dir"] / "poses.npy").open("wb") as stream:
        np.save(stream, poses, allow_pickle=False)
    comparison = json.loads(args["pose_comparison_path"].read_text())
    comparison["commercial_pose_sha256"] = finalize._sha256(
        args["commercial_foundation_pose_dir"] / "poses.npy"
    )
    args["pose_comparison_path"].write_text(json.dumps(comparison))

    trim_manifest = {
        "schema": interaction_trim.TRIM_SCHEMA,
        "source_frame_count": 100,
        "export_source_start_frame": 20,
        "export_source_end_frame": 100,
        "export_frame_count": 80,
        "contact_frame_source": 20,
        "first_contact_frame_source": 20,
        "last_contact_frame_source": 99,
        "trimmed_prefix_frames": 20,
        "trimmed_suffix_frames": 0,
        "trimmed_suffix_seconds": 0.0,
    }
    trim_manifest["decision_sha256"] = interaction_trim._canonical_sha256(
        trim_manifest
    )
    monkeypatch.setattr(
        finalize, "detect_interaction_trim", lambda *_args, **_kwargs: trim_manifest,
    )
    monkeypatch.setattr(
        finalize, "prepare_trimmed_source",
        lambda source, *_args, **_kwargs: Path(source),
    )

    def fake_export_sequence(**kwargs):
        _write_object_pose_pair(
            Path(kwargs["output_dir"]), poses=poses[20:100],
        )

    monkeypatch.setattr(finalize, "export_sequence", fake_export_sequence)
    monkeypatch.setattr(
        finalize,
        "validate_export_payload",
        lambda _root, **_kwargs: {
            "schema": "v2d.mv_hoi.export_validation.v2",
            "frame_count": 80,
        },
    )

    commit = finalize.finalize_revalidation_export(**args)

    assert json.loads((args["output_dir"] / "failure_segments.json").read_text()) == [
        {
            "id": "crossing",
            "start_frame": 0,
            "end_frame": 10,
            "source_start_frame": 15,
            "source_end_frame": 30,
            "failure_category": "Mesh Penetration",
            "reason": "human annotation",
        },
        {
            "id": "retained",
            "start_frame": 40,
            "end_frame": 60,
            "source_start_frame": 60,
            "source_end_frame": 80,
            "failure_category": "Large misalignment/drift",
        },
    ]
    metrics_root = args["output_dir"] / "metrics/revalidation/7/failure_segments"
    assert (
        metrics_root / "human_source_failure_segments.json"
    ).read_bytes() == source_bytes
    assert json.loads(
        (metrics_root / "source_failure_segments.json").read_text()
    ) == source_segments
    assert json.loads(
        (metrics_root / "export_failure_segments.json").read_text()
    ) == json.loads((args["output_dir"] / "failure_segments.json").read_text())
    assert commit["human_qc"]["source_failure_segment_count"] == 3
    assert commit["human_qc"]["export_failure_segment_count"] == 2


def test_existing_candidate_without_human_qc_is_rebuilt(tmp_path, monkeypatch):
    args = _args(tmp_path)
    args["rebuild_incomplete"] = True
    output = args["output_dir"]
    output.mkdir()
    (output / "old.txt").write_text("old candidate")
    old_payload = finalize._files(output)
    (output / "commit.json").write_text(json.dumps({
        "schema": finalize.COMMIT_SCHEMA,
        "complete": True,
        "request_id": 7,
        "files": old_payload,
    }))

    def fake_export_sequence(**kwargs):
        _write_object_pose_pair(Path(kwargs["output_dir"]))

    monkeypatch.setattr(finalize, "export_sequence", fake_export_sequence)
    _stub_trim(monkeypatch)
    monkeypatch.setattr(
        finalize,
        "validate_export_payload",
        lambda _root, **_kwargs: {"schema": "v2d.mv_hoi.export_validation.v2"},
    )

    commit = finalize.finalize_revalidation_export(**args)

    assert not (output / "old.txt").exists()
    assert (output / "failure_segments.json").is_file()
    assert commit["complete"] is True
