import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import run_mesh_to_usd_batch as batch


def _write_alignment(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "alignment": {
                    "centroid": [0.0, 0.0, 0.0],
                    "rotation": np.eye(4).reshape(-1).tolist(),
                }
            }
        ),
        encoding="utf-8",
    )


def _write_sequence(root: Path, name: str, object_id: str, *, moving=False) -> Path:
    sequence = root / name
    mesh_dir = sequence / "object_mesh"
    mesh_dir.mkdir(parents=True)
    (sequence / "hoi_metadata.yaml").write_text(
        f"object:\n  id: {object_id}\n",
        encoding="utf-8",
    )
    poses = np.repeat(np.eye(4)[None, :, :], 60, axis=0)
    if moving:
        poses[:, 0, 3] = np.arange(len(poses)) * 0.01
    np.save(sequence / "poses.npy", poses)
    (sequence / "ground_plane.json").write_text(
        json.dumps({"plane": [0.0, 0.0, 1.0, 0.0]}),
        encoding="utf-8",
    )
    (mesh_dir / "output_aligned.glb").write_bytes(b"recording mesh")
    _write_alignment(mesh_dir / "output_symmetry.json")
    for relative in ("edex", "images", "depth", "object_masks"):
        (sequence / relative).mkdir()
    return sequence


def _write_catalog_mesh(
    root: Path,
    object_id: str,
    method: str,
    data: bytes,
    *,
    filename: str = batch.DEFAULT_MESH_FILENAME,
) -> Path:
    method_dir = root / object_id / method
    method_dir.mkdir(parents=True, exist_ok=True)
    mesh = method_dir / filename
    mesh.write_bytes(data)
    _write_alignment(method_dir / "output_symmetry.json")
    return mesh


def test_discover_meshes_accepts_explicit_output_glb(tmp_path):
    mesh_root = tmp_path / "meshes"
    aligned = _write_catalog_mesh(
        mesh_root,
        "vase",
        "sam3d",
        b"stale aligned mesh",
    )
    raw = _write_catalog_mesh(
        mesh_root,
        "vase",
        "sam3d",
        b"canonical output mesh",
        filename="output.glb",
    )

    meshes, rejected = batch.discover_meshes(
        mesh_root,
        ("sam3d",),
        "output.glb",
    )

    assert aligned != raw
    assert meshes == {"vase": {"sam3d": raw}}
    assert rejected == []


@pytest.mark.parametrize("filename", ("../output.glb", "output.obj", ""))
def test_discover_meshes_rejects_invalid_mesh_filename(tmp_path, filename):
    mesh_root = tmp_path / "meshes"
    mesh_root.mkdir()

    with pytest.raises(ValueError, match="GLB basename"):
        batch.discover_meshes(mesh_root, mesh_filename=filename)


def _batch_options(weights: Path, **overrides) -> batch.BatchOptions:
    options = batch.BatchOptions(
        stable_window_frames=20,
        initial_search_frames=40,
        max_up_deviation_degrees=2.0,
        max_rotation_deviation_degrees=3.0,
        max_translation_deviation_m=0.02,
        foundation_pose_weights_dir=str(weights),
        foundation_pose_config_path=None,
        foundation_pose_debug=0,
        validate=True,
        run_drop_tests=True,
        video=True,
        fail_if_not_standing=True,
        image="generator:test",
        validator_image="validator:test",
        accept_eula=True,
        cache_dir=None,
        dev=False,
        gpu_device="0",
    )
    return replace(options, **overrides)


def test_batch_dry_run_selects_support_per_target_mesh(tmp_path):
    mesh_root = tmp_path / "meshes"
    sequence_root = tmp_path / "sequences"
    _write_catalog_mesh(mesh_root, "vase", "einstar", b"recording mesh")
    _write_catalog_mesh(mesh_root, "vase", "sam3d", b"sam3d mesh")
    _write_catalog_mesh(mesh_root, "unmatched", "einstar", b"other")
    _write_sequence(sequence_root, "2026-01-02_vase_moving", "vase", moving=True)
    stable = _write_sequence(sequence_root, "2026-01-01_vase_stable", "vase")
    weights = tmp_path / "weights"
    weights.mkdir()

    result = batch.run_batch(
        str(mesh_root),
        str(sequence_root),
        str(tmp_path / "output"),
        stable_window_frames=20,
        initial_search_frames=40,
        foundation_pose_weights_dir=str(weights),
        dry_run=True,
    )

    assert result["status"] == "planned"
    assert result["matched_object_count"] == 1
    assert result["planned_job_count"] == 2
    assert result["status_counts"] == {"planned": 2}
    exact_selection = result["support_selection"]["vase"]["einstar"]
    target_selection = result["support_selection"]["vase"]["sam3d"]
    assert exact_selection["selected_sequence"] == str(stable)
    assert exact_selection["support_pose_calibration"] == (
        "exact-mesh-recorded-poses"
    )
    assert len(exact_selection["rejected_candidates"]) == 1
    assert target_selection["selected_sequence"].endswith(
        "2026-01-02_vase_moving"
    )
    assert target_selection["support_pose_calibration"] == (
        "foundation-pose-target-tracking"
    )
    assert target_selection["target_frame_selection_deferred"] is True
    assert target_selection["rejected_candidates"] == []


def test_batch_generates_video_and_reports_each_method(tmp_path, monkeypatch):
    mesh_root = tmp_path / "meshes"
    sequence_root = tmp_path / "sequences"
    _write_catalog_mesh(mesh_root, "vase", "einstar", b"recording mesh")
    _write_catalog_mesh(mesh_root, "vase", "bundlesdf", b"bundlesdf mesh")
    _write_sequence(sequence_root, "2026-01-01_vase", "vase")
    weights = tmp_path / "weights"
    weights.mkdir()

    def fake_prepare(output_dir, **kwargs):
        assert kwargs["image"] == "generator:test"
        assert kwargs["validator_image"] == "validator:test"
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        usd = output / "rigid_object.usd"
        visual = output / "visual_asset.usd"
        support = output / "recorded_support_pose.json"
        generation = output / "generation_report.json"
        usd.write_text("#usda 1.0\n", encoding="utf-8")
        visual.write_text("#usda 1.0\n", encoding="utf-8")
        support.write_text("{}\n", encoding="utf-8")
        generation.write_text("{}\n", encoding="utf-8")
        calibration = (
            "exact-mesh-recorded-poses"
            if Path(kwargs["asset_path"]).read_bytes() == b"recording mesh"
            else "foundation-pose-target-tracking"
        )
        sequence = Path(kwargs["support_sequence_dir"])
        camera_provenance = (
            {
                "mode": "not-applicable-recorded-poses",
                "reason": "support pose read directly from sequence poses.npy",
            }
            if calibration == "exact-mesh-recorded-poses"
            else {
                "mode": "multi-view-foundation-pose",
                "camera_names": ["front", "back"],
                "registration_frame": 0,
                "registration_camera_names": ["front", "back"],
                "highest_visibility_registration_camera": "front",
                "registration_visible_ratios": {
                    "front": 0.8,
                    "back": 0.6,
                },
                "tracking_camera_frame_counts": {
                    "front": 20,
                    "back": 20,
                },
            }
        )
        return {
            "output_usd": str(usd),
            "visual_asset": str(visual),
            "recorded_support_pose": str(support),
            "generation_report": str(generation),
            "support_pose_calibration": calibration,
            "support_provenance": {
                "sequence": str(sequence),
                "sequence_id": sequence.name,
                "pose_calibration": calibration,
                "selected_frame_start": 0,
                "selected_frame_end_exclusive": 20,
                "camera_provenance": camera_provenance,
            },
            "foundation_pose_support_report": None,
            "foundation_pose_support_poses": None,
        }

    def fake_drop(_asset, output_dir, **kwargs):
        assert kwargs["image"] == "generator:test"
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        report = output / "drop_test_result.json"
        video = output / "drop_test.mp4"
        report.write_text("{}\n", encoding="utf-8")
        video.write_bytes(b"video")
        return {
            "report_file": str(report),
            "video_files": [str(video)],
            "standing": True,
        }

    monkeypatch.setattr(batch, "run_mesh_to_usd_workflow", fake_prepare)
    monkeypatch.setattr(batch, "run_drop_test", fake_drop)

    result = batch.run_batch(
        str(mesh_root),
        str(sequence_root),
        str(tmp_path / "output"),
        stable_window_frames=20,
        initial_search_frames=40,
        foundation_pose_weights_dir=str(weights),
        image="generator:test",
        validator_image="validator:test",
        accept_eula=True,
    )

    assert result["status"] == "passed"
    assert result["status_counts"] == {"passed": 2}
    assert all(job["video_files"] for job in result["jobs"])
    assert all(Path(job["video_files"][0]).is_file() for job in result["jobs"])
    assert {
        job["support_pose_calibration"] for job in result["jobs"]
    } == {
        "exact-mesh-recorded-poses",
        "foundation-pose-target-tracking",
    }
    assert all(job["support_provenance"] for job in result["jobs"])
    assert {
        job["support_provenance"]["sequence_id"] for job in result["jobs"]
    } == {"2026-01-01_vase"}


def test_batch_still_generates_usd_when_no_stable_support_exists(
    tmp_path,
    monkeypatch,
):
    mesh_root = tmp_path / "meshes"
    sequence_root = tmp_path / "sequences"
    _write_catalog_mesh(mesh_root, "ball", "einstar", b"recording mesh")
    _write_sequence(sequence_root, "2026-01-01_ball", "ball", moving=True)
    weights = tmp_path / "weights"
    weights.mkdir()
    calls = {"prepare": 0, "drop": 0}

    def fake_prepare(output_dir, **kwargs):
        calls["prepare"] += 1
        assert kwargs["support_sequence_dir"] is None
        assert kwargs["foundation_pose_weights_dir"] is None
        assert kwargs["gpu_device"] == "0"
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        usd = output / "rigid_object.usd"
        visual = output / "visual_asset.usd"
        generation = output / "generation_report.json"
        usd.write_text("#usda 1.0\n", encoding="utf-8")
        visual.write_text("#usda 1.0\n", encoding="utf-8")
        generation.write_text("{}\n", encoding="utf-8")
        return {
            "output_usd": str(usd),
            "visual_asset": str(visual),
            "recorded_support_pose": None,
            "generation_report": str(generation),
        }

    def fake_drop(*_args, **_kwargs):
        calls["drop"] += 1
        raise AssertionError("drop test must not run without recorded support")

    monkeypatch.setattr(batch, "run_mesh_to_usd_workflow", fake_prepare)
    monkeypatch.setattr(batch, "run_drop_test", fake_drop)

    result = batch.run_batch(
        str(mesh_root),
        str(sequence_root),
        str(tmp_path / "output"),
        stable_window_frames=20,
        initial_search_frames=40,
        foundation_pose_weights_dir=str(weights),
        accept_eula=True,
    )

    assert result["status"] == "completed-with-incomplete-drop-tests"
    assert result["status_counts"] == {"generated-without-drop-test": 1}
    assert result["jobs"][0]["support_error"] == (
        "no usable support sequence for target mesh"
    )
    assert Path(result["jobs"][0]["output_usd"]).is_file()
    assert calls == {"prepare": 1, "drop": 0}


def test_batch_cross_mesh_selection_does_not_require_recorded_poses(tmp_path):
    mesh_root = tmp_path / "meshes"
    sequence_root = tmp_path / "sequences"
    _write_catalog_mesh(mesh_root, "vase", "sam3d", b"sam3d mesh")
    sequence = _write_sequence(
        sequence_root,
        "2026-01-01_vase",
        "vase",
    )
    (sequence / "poses.npy").unlink()
    weights = tmp_path / "weights"
    weights.mkdir()

    result = batch.run_batch(
        str(mesh_root),
        str(sequence_root),
        str(tmp_path / "output"),
        stable_window_frames=20,
        initial_search_frames=40,
        foundation_pose_weights_dir=str(weights),
        dry_run=True,
    )

    selection = result["support_selection"]["vase"]["sam3d"]
    assert result["status_counts"] == {"planned": 1}
    assert selection["selected_sequence"] == str(sequence)
    assert selection["support_pose_calibration"] == (
        "foundation-pose-target-tracking"
    )
    assert selection["target_frame_selection_deferred"] is True


def test_batch_does_not_hide_target_tracking_failure_with_older_sequence(
    tmp_path,
    monkeypatch,
):
    mesh_root = tmp_path / "meshes"
    sequence_root = tmp_path / "sequences"
    _write_catalog_mesh(mesh_root, "vase", "sam3d", b"sam3d mesh")
    newest = _write_sequence(
        sequence_root,
        "2026-01-02_vase",
        "vase",
    )
    _write_sequence(sequence_root, "2026-01-01_vase", "vase")
    weights = tmp_path / "weights"
    weights.mkdir()
    attempted_sequences = []

    def failed_target_tracking(_output_dir, **kwargs):
        attempted_sequences.append(kwargs["support_sequence_dir"])
        raise ValueError(
            "no stable segment of still frames was found; "
            "No fallback was attempted"
        )

    monkeypatch.setattr(
        batch,
        "run_mesh_to_usd_workflow",
        failed_target_tracking,
    )

    result = batch.run_batch(
        str(mesh_root),
        str(sequence_root),
        str(tmp_path / "output"),
        stable_window_frames=20,
        initial_search_frames=40,
        foundation_pose_weights_dir=str(weights),
        accept_eula=True,
    )

    assert result["status"] == "completed-with-failures"
    assert result["status_counts"] == {"failed": 1}
    assert attempted_sequences == [str(newest)]
    assert result["jobs"][0]["error"].endswith(
        "No fallback was attempted"
    )


def test_batch_resume_rejects_result_from_different_support_sequence(tmp_path):
    mesh = tmp_path / "mesh.glb"
    mesh.write_bytes(b"mesh")
    first_sequence = tmp_path / "sequence-1"
    second_sequence = tmp_path / "sequence-2"
    first_sequence.mkdir()
    second_sequence.mkdir()
    package = tmp_path / "mesh_to_usd"
    package.mkdir()
    usd = package / "rigid_object.usd"
    usd.write_text("#usda 1.0\n", encoding="utf-8")
    visual = package / "visual_asset.usd"
    visual.write_text("#usda 1.0\n", encoding="utf-8")
    generation = package / "generation_report.json"
    generation.write_text("{}\n", encoding="utf-8")
    support = package / "recorded_support_pose.json"
    support.write_text("{}\n", encoding="utf-8")
    drop_report = tmp_path / "drop_test_result.json"
    drop_report.write_text("{}\n", encoding="utf-8")
    video = tmp_path / "drop_test.mp4"
    video.write_bytes(b"video")
    report_path = tmp_path / "batch_job_report.json"
    signature = {
        "schema_version": batch.BATCH_RESUME_SCHEMA_VERSION,
        "sha256": "a" * 64,
    }
    report = {
        "status": "passed",
        "mesh": str(mesh),
        "sequence": str(first_sequence),
        "resume_signature": signature,
        "support_pose_calibration": (
            "foundation-pose-target-tracking"
        ),
        "support_provenance": {
            "sequence": str(first_sequence),
            "sequence_id": first_sequence.name,
            "pose_calibration": "foundation-pose-target-tracking",
            "selected_frame_start": 0,
            "selected_frame_end_exclusive": 30,
            "camera_provenance": {
                "mode": "multi-view-foundation-pose",
            },
        },
        "output_usd": str(usd),
        "visual_asset": str(visual),
        "mesh_to_usd_dir": str(package),
        "generation_report": str(generation),
        "recorded_support_pose": str(support),
        "drop_test_report": str(drop_report),
        "video_files": [str(video)],
    }
    report["artifact_fingerprints"] = batch._capture_artifact_fingerprints(
        report
    )
    report_path.write_text(
        json.dumps(report),
        encoding="utf-8",
    )

    assert batch._resume_result(
        report_path,
        expected_mesh=mesh,
        expected_sequence=first_sequence,
        expected_resume_signature=signature,
        require_validation=False,
        require_drop_test=True,
        require_video=True,
        expected_support_pose_calibration=(
            "foundation-pose-target-tracking"
        ),
    ) is not None
    assert batch._resume_result(
        report_path,
        expected_mesh=mesh,
        expected_sequence=second_sequence,
        expected_resume_signature=signature,
        require_validation=False,
        require_drop_test=True,
        require_video=True,
        expected_support_pose_calibration=(
            "foundation-pose-target-tracking"
        ),
    ) is None


def test_batch_resume_rejects_replaced_mesh_and_changed_options(tmp_path):
    mesh_root = tmp_path / "meshes"
    mesh = _write_catalog_mesh(mesh_root, "vase", "sam3d", b"old mesh")
    weights = tmp_path / "weights"
    weights.mkdir()
    options = _batch_options(weights, run_drop_tests=False, video=False)
    first = batch._resume_signature(
        mesh=mesh,
        sequence=None,
        support_pose_calibration=None,
        options=options,
        digest_cache={},
    )

    mesh.write_bytes(b"replacement mesh at the same path")
    replaced = batch._resume_signature(
        mesh=mesh,
        sequence=None,
        support_pose_calibration=None,
        options=options,
        digest_cache={},
    )
    changed_options = batch._resume_signature(
        mesh=mesh,
        sequence=None,
        support_pose_calibration=None,
        options=replace(options, validate=False),
        digest_cache={},
    )
    changed_generator = batch._resume_signature(
        mesh=mesh,
        sequence=None,
        support_pose_calibration=None,
        options=replace(options, image="generator:replacement"),
        digest_cache={},
    )
    changed_validator = batch._resume_signature(
        mesh=mesh,
        sequence=None,
        support_pose_calibration=None,
        options=replace(options, validator_image="validator:replacement"),
        digest_cache={},
    )

    assert first["sha256"] != replaced["sha256"]
    assert replaced["sha256"] != changed_options["sha256"]
    assert replaced["sha256"] != changed_generator["sha256"]
    assert replaced["sha256"] != changed_validator["sha256"]


def test_batch_honors_accept_eula_environment(tmp_path, monkeypatch):
    mesh_root = tmp_path / "meshes"
    sequence_root = tmp_path / "sequences"
    mesh_root.mkdir()
    sequence_root.mkdir()
    monkeypatch.setenv("ACCEPT_EULA", "Y")

    with pytest.raises(RuntimeError, match="no matched"):
        batch.run_batch(
            str(mesh_root),
            str(sequence_root),
            str(tmp_path / "output"),
        )


def test_batch_resume_rejects_modified_output_artifact(tmp_path):
    mesh = tmp_path / "mesh.glb"
    mesh.write_bytes(b"mesh")
    package = tmp_path / "mesh_to_usd"
    package.mkdir()
    usd = package / "rigid_object.usd"
    usd.write_text("#usda 1.0\n", encoding="utf-8")
    visual = package / "visual_asset.usd"
    visual.write_text("#usda 1.0\n", encoding="utf-8")
    generation = package / "generation_report.json"
    generation.write_text("{}\n", encoding="utf-8")
    signature = {
        "schema_version": batch.BATCH_RESUME_SCHEMA_VERSION,
        "sha256": "b" * 64,
    }
    report = {
        "status": "passed",
        "mesh": str(mesh),
        "sequence": None,
        "resume_signature": signature,
        "support_pose_calibration": None,
        "output_usd": str(usd),
        "visual_asset": str(visual),
        "mesh_to_usd_dir": str(package),
        "generation_report": str(generation),
        "video_files": [],
    }
    report["artifact_fingerprints"] = batch._capture_artifact_fingerprints(
        report
    )
    report_path = tmp_path / "batch_job_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    usd.write_text("#usda 1.0\n# modified\n", encoding="utf-8")

    assert batch._resume_result(
        report_path,
        expected_mesh=mesh,
        expected_sequence=None,
        expected_resume_signature=signature,
        require_validation=False,
        require_drop_test=False,
        require_video=False,
        expected_support_pose_calibration=None,
    ) is None


def test_batch_resume_rejects_missing_visual_asset_dependency(tmp_path):
    mesh = tmp_path / "mesh.glb"
    mesh.write_bytes(b"mesh")
    package = tmp_path / "mesh_to_usd"
    package.mkdir()
    usd = package / "rigid_object.usd"
    usd.write_text(
        '#usda 1.0\ndef Xform "Asset" (references = @./visual_asset.usd@) {}\n',
        encoding="utf-8",
    )
    visual = package / "visual_asset.usd"
    visual.write_text("#usda 1.0\n", encoding="utf-8")
    generation = package / "generation_report.json"
    generation.write_text("{}\n", encoding="utf-8")
    signature = {
        "schema_version": batch.BATCH_RESUME_SCHEMA_VERSION,
        "sha256": "c" * 64,
    }
    report = {
        "status": "passed",
        "mesh": str(mesh),
        "sequence": None,
        "resume_signature": signature,
        "support_pose_calibration": None,
        "output_usd": str(usd),
        "visual_asset": str(visual),
        "mesh_to_usd_dir": str(package),
        "generation_report": str(generation),
        "video_files": [],
    }
    report["artifact_fingerprints"] = batch._capture_artifact_fingerprints(
        report
    )
    report_path = tmp_path / "batch_job_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    visual.unlink()

    assert batch._resume_result(
        report_path,
        expected_mesh=mesh,
        expected_sequence=None,
        expected_resume_signature=signature,
        require_validation=False,
        require_drop_test=False,
        require_video=False,
        expected_support_pose_calibration=None,
    ) is None


def test_batch_artifact_check_covers_unreported_package_dependencies(tmp_path):
    package = tmp_path / "mesh_to_usd"
    textures = package / "textures"
    textures.mkdir(parents=True)
    usd = package / "rigid_object.usd"
    visual = package / "visual_asset.usd"
    generation = package / "generation_report.json"
    texture = textures / "albedo.png"
    usd.write_text("#usda 1.0\n", encoding="utf-8")
    visual.write_text("#usda 1.0\n", encoding="utf-8")
    generation.write_text("{}\n", encoding="utf-8")
    texture.write_bytes(b"original texture")
    job = {
        "output_usd": str(usd),
        "visual_asset": str(visual),
        "generation_report": str(generation),
        "mesh_to_usd_dir": str(package),
        "video_files": [],
    }
    job["artifact_fingerprints"] = batch._capture_artifact_fingerprints(job)

    texture.write_bytes(b"modified texture")

    assert not batch._artifacts_match(job)


def test_batch_rejects_zero_matched_jobs(tmp_path):
    mesh_root = tmp_path / "meshes"
    sequence_root = tmp_path / "sequences"
    mesh_root.mkdir()
    sequence_root.mkdir()
    output = tmp_path / "output"

    with pytest.raises(RuntimeError, match="no matched"):
        batch.run_batch(
            str(mesh_root),
            str(sequence_root),
            str(output),
            accept_eula=True,
        )

    report = json.loads(
        (output / batch.BATCH_REPORT_NAME).read_text(encoding="utf-8")
    )
    assert report["status"] == "no-matched-jobs"
    assert report["planned_job_count"] == 0
