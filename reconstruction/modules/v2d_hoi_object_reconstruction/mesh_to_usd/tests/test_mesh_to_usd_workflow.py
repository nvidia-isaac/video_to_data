import json
from pathlib import Path

import numpy as np
import pytest

import run_mesh_to_usd_workflow as workflow
from recorded_support import file_sha256, recorded_support_pose_load


def _camera_provenance(registration_frame=0):
    return {
        "mode": "multi-view-foundation-pose",
        "camera_names": [
            "front_stereo_camera_left",
            "back_stereo_camera_left",
        ],
        "registration_frame": registration_frame,
        "registration_camera_names": [
            "front_stereo_camera_left",
            "back_stereo_camera_left",
        ],
        "highest_visibility_registration_camera": (
            "front_stereo_camera_left"
        ),
        "registration_visible_ratios": {
            "front_stereo_camera_left": 0.8,
            "back_stereo_camera_left": 0.6,
        },
        "tracking_camera_frame_counts": {
            "front_stereo_camera_left": 40,
            "back_stereo_camera_left": 35,
        },
    }


def _sequence(tmp_path, *, moving=False):
    sequence = tmp_path / "sequence"
    (sequence / "object_mesh").mkdir(parents=True)
    mesh = sequence / "object_mesh" / "output_aligned.glb"
    mesh.write_bytes(b"mesh")
    poses = np.repeat(np.eye(4)[None, :, :], 60, axis=0)
    if moving:
        poses[:, 0, 3] = np.arange(len(poses)) * 0.01
    np.save(sequence / "poses.npy", poses)
    (sequence / "ground_plane.json").write_text(
        json.dumps({"plane": [0.0, 0.0, 1.0, 0.0]}),
        encoding="utf-8",
    )
    return sequence


def _write_alignment(path):
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


def _fake_generation(asset_path, output_dir, **_kwargs):
    output = Path(output_dir)
    rigid_object = output / "rigid_object.usd"
    visual_asset = output / "visual_asset.usd"
    rigid_object.write_text("#usda 1.0\n", encoding="utf-8")
    visual_asset.write_text("#usda 1.0\n", encoding="utf-8")
    (output / "generation_report.json").write_text(
        json.dumps({"status": "generated"}),
        encoding="utf-8",
    )
    return {
        "status": "generated",
        "output_usd": str(rigid_object),
        "visual_asset": str(visual_asset),
        "input_asset_file_sha256": file_sha256(asset_path),
    }


def _fake_foundation_pose_support(
    sequence_dir,
    target_mesh_path,
    weights_dir,
    output_dir,
    *,
    frame_end_exclusive,
    **_kwargs,
):
    assert weights_dir == "test-weights"
    assert _kwargs["allow_shorter_prefix"] is True
    assert _kwargs["minimum_output_frames"] == 20
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    poses_path = output / "poses.npy"
    poses = np.repeat(np.eye(4)[None, :, :], frame_end_exclusive, axis=0)
    np.save(poses_path, poses)
    target_mesh = Path(target_mesh_path)
    target_symmetry = target_mesh.with_name("output_symmetry.json")
    report_path = output / "foundation_pose_support_report.json"
    tracking_metadata = output / "pose_tracking_metadata.json"
    tracking_metadata.write_text(
        json.dumps(
            {
                "status": "completed",
                "source_frame_start": 0,
                "source_frame_end_exclusive": frame_end_exclusive,
                "pose_count": frame_end_exclusive,
            }
        ),
        encoding="utf-8",
    )
    report = {
        "status": "completed",
        "sequence_id": Path(sequence_dir).name,
        "target_mesh_file_sha256": file_sha256(target_mesh),
        "target_symmetry_file_sha256": file_sha256(target_symmetry),
        "poses_file_sha256": file_sha256(poses_path),
        "frame_start": 0,
        "frame_end_exclusive": frame_end_exclusive,
        "camera_provenance": _camera_provenance(),
        "output_poses": str(poses_path),
        "tracking_metadata": str(tracking_metadata),
        "report_file": str(report_path),
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report


def test_asset_workflow_generates_usd_without_recorded_support(tmp_path, monkeypatch):
    asset = tmp_path / "object.glb"
    asset.write_bytes(b"mesh")
    output = tmp_path / "output"
    output.mkdir()
    stale_support = output / workflow.RECORDED_SUPPORT_NAME
    stale_support.write_text("stale", encoding="utf-8")
    stale_foundation_pose = output / "foundation_pose_support"
    stale_foundation_pose.mkdir()
    stale_poses = stale_foundation_pose / "poses.npy"
    stale_report = (
        stale_foundation_pose / "foundation_pose_support_report.json"
    )
    stale_poses.write_bytes(b"stale")
    stale_report.write_text("stale", encoding="utf-8")
    monkeypatch.setattr(workflow, "run_mesh_to_usd", _fake_generation)

    result = workflow.run_mesh_to_usd_workflow(
        str(output),
        asset_path=str(asset),
    )

    assert result["input_mode"] == "asset"
    assert result["recorded_support_pose"] is None
    assert not stale_support.exists()
    assert not stale_poses.exists()
    assert not stale_report.exists()
    assert Path(result["output_usd"]).is_file()
    assert Path(result["visual_asset"]).is_file()
    assert Path(result["workflow_report"]).is_file()


def test_sequence_workflow_generates_usd_and_support_json(tmp_path, monkeypatch):
    sequence = _sequence(tmp_path)
    output = tmp_path / "output"
    monkeypatch.setattr(workflow, "run_mesh_to_usd", _fake_generation)

    result = workflow.run_mesh_to_usd_workflow(
        str(output),
        sequence_dir=str(sequence),
        stable_window_frames=20,
        initial_search_frames=40,
    )

    support_path = Path(result["recorded_support_pose"])
    support = json.loads(support_path.read_text(encoding="utf-8"))
    assert result["input_mode"] == "recorded-sequence"
    assert result["source_asset"].endswith("object_mesh/output_aligned.glb")
    assert support["frame_start"] == 0
    assert support["frame_end_exclusive"] == 20
    assert support["frame_selection"]["policy"] == "initial-stable"


def test_asset_workflow_retargets_recorded_support_with_full_provenance(
    tmp_path,
    monkeypatch,
):
    sequence = _sequence(tmp_path)
    _write_alignment(sequence / "object_mesh" / "output_symmetry.json")
    target_dir = tmp_path / "catalog" / "bundlesdf"
    target_dir.mkdir(parents=True)
    target = target_dir / "output_aligned.glb"
    target.write_bytes(b"different mesh")
    _write_alignment(target_dir / "output_symmetry.json")
    # Cross-mesh support must come entirely from the target-mesh
    # FoundationPose run, not from the recording's pre-existing object poses.
    (sequence / "poses.npy").unlink()
    output = tmp_path / "output"
    monkeypatch.setattr(workflow, "run_mesh_to_usd", _fake_generation)
    monkeypatch.setattr(
        workflow,
        "run_foundation_pose_support",
        _fake_foundation_pose_support,
    )

    result = workflow.run_mesh_to_usd_workflow(
        str(output),
        asset_path=str(target),
        support_sequence_dir=str(sequence),
        stable_window_frames=20,
        initial_search_frames=40,
        foundation_pose_weights_dir="test-weights",
    )

    support = json.loads(
        Path(result["recorded_support_pose"]).read_text(encoding="utf-8")
    )
    assert result["input_mode"] == "asset-with-recorded-support"
    assert result["support_sequence"] == str(sequence.resolve())
    assert support["mesh_file_sha256"] == file_sha256(target)
    assert support["recording_mesh_file_sha256"] == file_sha256(target)
    assert result["support_pose_calibration"] == (
        "foundation-pose-target-tracking"
    )
    assert Path(result["foundation_pose_support_report"]).is_file()
    assert Path(result["foundation_pose_support_poses"]).is_file()
    assert support["local_up"] == pytest.approx([0.0, 0.0, 1.0])
    assert support["frame_start"] == 0
    assert support["frame_end_exclusive"] == 20
    assert support["frame_selection"]["search_end_exclusive"] == 40
    assert support["mesh_frame_transfer"]["policy"] == (
        "foundation-pose-target-tracking"
    )
    assert support["mesh_frame_transfer"]["local_up_transform"] == (
        "direct-target-pose-inference"
    )
    assert result["support_provenance"]["sequence_id"] == sequence.name
    assert result["support_provenance"]["camera_provenance"] == (
        _camera_provenance()
    )
    assert support["mesh_frame_transfer"]["camera_provenance"] == (
        _camera_provenance()
    )
    assert "reference_poses_file_sha256" not in support["mesh_frame_transfer"]


def test_asset_workflow_offsets_late_foundation_pose_registration(
    tmp_path,
    monkeypatch,
):
    sequence = _sequence(tmp_path)
    target_dir = tmp_path / "catalog" / "sam3d"
    target_dir.mkdir(parents=True)
    target = target_dir / "output_aligned.glb"
    target.write_bytes(b"different mesh")
    _write_alignment(target_dir / "output_symmetry.json")
    output = tmp_path / "output"

    def late_foundation_pose(
        sequence_dir,
        target_mesh_path,
        _weights_dir,
        output_dir,
        *,
        frame_end_exclusive,
        **_kwargs,
    ):
        assert frame_end_exclusive == 40
        pose_output = Path(output_dir)
        pose_output.mkdir(parents=True, exist_ok=True)
        poses_path = pose_output / "poses.npy"
        poses = np.repeat(np.eye(4)[None, :, :], 30, axis=0)
        np.save(poses_path, poses)
        target_symmetry = Path(target_mesh_path).with_name(
            "output_symmetry.json"
        )
        report_path = (
            pose_output / "foundation_pose_support_report.json"
        )
        tracking_metadata = pose_output / "pose_tracking_metadata.json"
        tracking_metadata.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "source_frame_start": 10,
                    "source_frame_end_exclusive": 40,
                    "pose_count": 30,
                }
            ),
            encoding="utf-8",
        )
        report = {
            "status": "completed",
            "sequence_id": Path(sequence_dir).name,
            "target_mesh_file_sha256": file_sha256(target_mesh_path),
            "target_symmetry_file_sha256": file_sha256(target_symmetry),
            "poses_file_sha256": file_sha256(poses_path),
            "frame_start": 10,
            "frame_end_exclusive": 40,
            "camera_provenance": _camera_provenance(10),
            "output_poses": str(poses_path),
            "tracking_metadata": str(tracking_metadata),
            "report_file": str(report_path),
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    monkeypatch.setattr(workflow, "run_mesh_to_usd", _fake_generation)
    monkeypatch.setattr(
        workflow,
        "run_foundation_pose_support",
        late_foundation_pose,
    )

    result = workflow.run_mesh_to_usd_workflow(
        str(output),
        asset_path=str(target),
        support_sequence_dir=str(sequence),
        stable_window_frames=20,
        initial_search_frames=40,
        foundation_pose_weights_dir="test-weights",
    )

    support = json.loads(
        Path(result["recorded_support_pose"]).read_text(encoding="utf-8")
    )
    assert support["frame_start"] == 10
    assert support["frame_end_exclusive"] == 30
    assert support["frame_selection"]["search_start"] == 10
    assert support["frame_selection"]["search_end_exclusive"] == 40
    loaded = recorded_support_pose_load(result["recorded_support_pose"])
    assert loaded.frame_start == 10


def test_explicit_support_window_sets_tracking_minimum_to_its_span(
    tmp_path,
    monkeypatch,
):
    sequence = _sequence(tmp_path)
    target = tmp_path / "catalog" / "sam3d" / "output_aligned.glb"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"different mesh")

    class ExpectedCall(Exception):
        pass

    def inspect_foundation_pose_call(
        _sequence_dir,
        _target_mesh_path,
        _weights_dir,
        _output_dir,
        *,
        frame_end_exclusive,
        **kwargs,
    ):
        assert frame_end_exclusive == 25
        assert kwargs["allow_shorter_prefix"] is False
        assert kwargs["minimum_output_frames"] == 15
        raise ExpectedCall

    monkeypatch.setattr(
        workflow,
        "run_foundation_pose_support",
        inspect_foundation_pose_call,
    )

    with pytest.raises(ExpectedCall):
        workflow.run_mesh_to_usd_workflow(
            str(tmp_path / "output"),
            asset_path=str(target),
            support_sequence_dir=str(sequence),
            frame_start=10,
            frame_end_exclusive=25,
            stable_window_frames=30,
            foundation_pose_weights_dir="test-weights",
        )


def test_asset_workflow_reuses_exact_recording_mesh_poses(
    tmp_path,
    monkeypatch,
):
    sequence = _sequence(tmp_path)
    target = sequence / "object_mesh" / "output_aligned.glb"
    monkeypatch.setattr(workflow, "run_mesh_to_usd", _fake_generation)

    def unexpected_foundation_pose(*_args, **_kwargs):
        raise AssertionError("exact recording mesh must not rerun FoundationPose")

    monkeypatch.setattr(
        workflow,
        "run_foundation_pose_support",
        unexpected_foundation_pose,
    )

    result = workflow.run_mesh_to_usd_workflow(
        str(tmp_path / "output"),
        asset_path=str(target),
        support_sequence_dir=str(sequence),
        stable_window_frames=20,
        initial_search_frames=40,
    )

    support = json.loads(
        Path(result["recorded_support_pose"]).read_text(encoding="utf-8")
    )
    assert result["support_pose_calibration"] == "exact-mesh-recorded-poses"
    assert result["foundation_pose_support_report"] is None
    assert support["mesh_frame_transfer"]["policy"] == "exact-mesh"
    assert result["support_provenance"]["sequence_id"] == sequence.name
    assert result["support_provenance"]["camera_provenance"]["mode"] == (
        "not-applicable-recorded-poses"
    )


def test_asset_workflow_tracks_exact_mesh_when_recorded_poses_are_missing(
    tmp_path,
    monkeypatch,
):
    sequence = _sequence(tmp_path)
    target = sequence / "object_mesh" / "output_aligned.glb"
    _write_alignment(sequence / "object_mesh" / "output_symmetry.json")
    (sequence / "poses.npy").unlink()
    monkeypatch.setattr(workflow, "run_mesh_to_usd", _fake_generation)
    monkeypatch.setattr(
        workflow,
        "run_foundation_pose_support",
        _fake_foundation_pose_support,
    )

    result = workflow.run_mesh_to_usd_workflow(
        str(tmp_path / "output"),
        asset_path=str(target),
        support_sequence_dir=str(sequence),
        stable_window_frames=20,
        initial_search_frames=40,
        foundation_pose_weights_dir="test-weights",
    )

    support = json.loads(
        Path(result["recorded_support_pose"]).read_text(encoding="utf-8")
    )
    assert result["support_pose_calibration"] == (
        "foundation-pose-target-tracking"
    )
    assert support["recording_mesh_file_sha256"] == file_sha256(target)
    assert support["frame_selection"]["search_end_exclusive"] == 40


def test_asset_recorded_support_rejects_missing_alignment_metadata(
    tmp_path,
    monkeypatch,
):
    sequence = _sequence(tmp_path)
    _write_alignment(sequence / "object_mesh" / "output_symmetry.json")
    target_dir = tmp_path / "catalog" / "sam3d"
    target_dir.mkdir(parents=True)
    target = target_dir / "output_aligned.glb"
    target.write_bytes(b"different mesh")
    weights = tmp_path / "weights"
    weights.mkdir()
    for relative in ("edex", "images", "depth", "object_masks"):
        (sequence / relative).mkdir()

    def unexpected_conversion(*_args, **_kwargs):
        raise AssertionError("mesh conversion must not run without alignment metadata")

    monkeypatch.setattr(workflow, "run_mesh_to_usd", unexpected_conversion)

    with pytest.raises(ValueError, match="requires alignment metadata"):
        workflow.run_mesh_to_usd_workflow(
            str(tmp_path / "output"),
            asset_path=str(target),
            support_sequence_dir=str(sequence),
            stable_window_frames=20,
            initial_search_frames=40,
            foundation_pose_weights_dir=str(weights),
        )


def test_sequence_workflow_quits_before_conversion_without_still_frames(
    tmp_path,
    monkeypatch,
):
    sequence = _sequence(tmp_path, moving=True)
    output = tmp_path / "output"
    output.mkdir()
    stale_support = output / workflow.RECORDED_SUPPORT_NAME
    stale_support.write_text("stale", encoding="utf-8")

    def unexpected_conversion(*_args, **_kwargs):
        raise AssertionError("mesh conversion must not run after inference failure")

    monkeypatch.setattr(workflow, "run_mesh_to_usd", unexpected_conversion)

    with pytest.raises(ValueError, match="no stable segment.*No fallback"):
        workflow.run_mesh_to_usd_workflow(
            str(output),
            sequence_dir=str(sequence),
            stable_window_frames=20,
            initial_search_frames=40,
        )

    assert not stale_support.exists()
    assert not (output / workflow.WORKFLOW_REPORT_NAME).exists()


def test_sequence_workflow_rejects_generated_mesh_hash_mismatch(tmp_path, monkeypatch):
    sequence = _sequence(tmp_path)
    output = tmp_path / "output"

    def wrong_hash(asset_path, output_dir, **kwargs):
        result = _fake_generation(asset_path, output_dir, **kwargs)
        result["input_asset_file_sha256"] = "0" * 64
        return result

    monkeypatch.setattr(workflow, "run_mesh_to_usd", wrong_hash)

    with pytest.raises(RuntimeError, match="source hash does not match"):
        workflow.run_mesh_to_usd_workflow(
            str(output),
            sequence_dir=str(sequence),
            stable_window_frames=20,
            initial_search_frames=40,
        )

    assert not (output / workflow.RECORDED_SUPPORT_NAME).exists()


def test_workflow_requires_exactly_one_input(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        workflow.run_mesh_to_usd_workflow(str(tmp_path / "output"))
    with pytest.raises(ValueError, match="exactly one"):
        workflow.run_mesh_to_usd_workflow(
            str(tmp_path / "output"),
            asset_path="mesh.glb",
            sequence_dir="sequence",
        )
    with pytest.raises(ValueError, match="requires asset_path"):
        workflow.run_mesh_to_usd_workflow(
            str(tmp_path / "output"),
            sequence_dir="sequence",
            support_sequence_dir="support",
        )
