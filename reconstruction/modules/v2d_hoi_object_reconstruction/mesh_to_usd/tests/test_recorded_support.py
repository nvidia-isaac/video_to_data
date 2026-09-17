import json

import numpy as np
import pytest

from recorded_support import (
    file_sha256,
    foundation_pose_calibrated_support_pose,
    poses_to_recorded_support,
    recorded_support_pose_load,
    recorded_support_pose_save,
    recording_to_support_pose,
    retarget_recorded_support_pose,
)


def _poses(count=20):
    poses = np.repeat(np.eye(4)[None, :, :], count, axis=0)
    poses[:, :3, :3] = np.asarray(
        (
            (0.0, 0.0, -1.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
        )
    )
    return poses


def _support(
    poses=None,
    *,
    frame_start=0,
    frame_end_exclusive=None,
    **kwargs,
):
    poses = _poses() if poses is None else poses
    mesh_file_sha256 = kwargs.pop("mesh_file_sha256", "a" * 64)
    frame_end_exclusive = (
        len(poses) if frame_end_exclusive is None else frame_end_exclusive
    )
    return poses_to_recorded_support(
        poses,
        np.asarray((0.0, 0.0, 1.0, 0.0)),
        frame_start=frame_start,
        frame_end_exclusive=frame_end_exclusive,
        sequence_id="sequence",
        mesh_file_sha256=mesh_file_sha256,
        poses_file_sha256="b" * 64,
        ground_plane_file_sha256="d" * 64,
        **kwargs,
    )


def _write_sequence(tmp_path, poses):
    sequence = tmp_path / "sequence"
    (sequence / "object_mesh").mkdir(parents=True)
    (sequence / "object_mesh" / "output_aligned.glb").write_bytes(b"mesh")
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


def test_recorded_support_maps_observed_local_up_to_world_z():
    support = _support()

    assert support.pose_source == "recorded"
    assert support.fallback_used is False
    assert support.local_up == pytest.approx((1.0, 0.0, 0.0))
    w, x, y, z = support.initial_rotation_wxyz
    assert (w, x, y, z) == pytest.approx(
        (np.sqrt(0.5), 0.0, -np.sqrt(0.5), 0.0)
    )
    assert support.max_up_deviation_degrees == pytest.approx(0.0)
    assert support.frame_selection["policy"] == "explicit"


def test_recording_auto_selects_first_stable_window_from_initial_prefix(tmp_path):
    poses = _poses(60)
    poses[:5, 0, 3] = 0.1
    sequence = _write_sequence(tmp_path, poses)

    support = recording_to_support_pose(
        sequence,
        stable_window_frames=20,
        initial_search_frames=40,
    )

    assert support.frame_start == 5
    assert support.frame_end_exclusive == 25
    assert support.frame_selection == {
        "policy": "initial-stable",
        "search_start": 0,
        "search_end_exclusive": 40,
        "window_frames": 20,
    }
    assert support.fallback_used is False


def test_recording_auto_selection_fails_without_initial_still_frames(tmp_path):
    poses = _poses(60)
    poses[:, 0, 3] = np.arange(len(poses)) * 0.01
    sequence = _write_sequence(tmp_path, poses)

    with pytest.raises(ValueError, match="no stable segment.*No fallback"):
        recording_to_support_pose(
            sequence,
            stable_window_frames=20,
            initial_search_frames=40,
        )


def test_recording_auto_selection_does_not_search_later_frames(tmp_path):
    poses = _poses(70)
    poses[:40, 0, 3] = np.arange(40) * 0.01
    poses[40:, 0, 3] = poses[39, 0, 3]
    sequence = _write_sequence(tmp_path, poses)

    with pytest.raises(ValueError, match=r"initial frames \[0, 40\)"):
        recording_to_support_pose(
            sequence,
            stable_window_frames=20,
            initial_search_frames=40,
        )


def test_recording_manual_frame_override_must_be_paired(tmp_path):
    sequence = _write_sequence(tmp_path, _poses())

    with pytest.raises(ValueError, match="must be provided together"):
        recording_to_support_pose(sequence, frame_start=0)


def test_recorded_support_rejects_unstable_translation():
    poses = _poses()
    poses[-1, :3, 3] = (0.0, 0.0, 0.1)

    with pytest.raises(ValueError, match="translation spread"):
        _support(poses, max_translation_deviation_m=0.02)


def test_recorded_support_rejects_malformed_selected_pose():
    poses = _poses()
    poses[5, 0, 0] = np.nan

    with pytest.raises(ValueError, match="selected poses must be finite"):
        _support(poses)


def test_recorded_support_only_validates_selected_poses():
    poses = _poses(21)
    poses[20, 0, 0] = np.nan

    support = _support(
        poses,
        frame_start=0,
        frame_end_exclusive=20,
    )

    assert support.selected_frame_count == 20


def test_recording_input_uses_poses_without_valid_mask(tmp_path):
    sequence = _write_sequence(tmp_path, _poses())

    support = recording_to_support_pose(
        sequence,
        frame_start=0,
        frame_end_exclusive=20,
    )

    data = support.to_dict()
    assert support.selected_frame_count == 20
    assert "pose_valid_mask_file_sha256" not in data
    assert "valid_frame_count" not in data
    assert "valid_frame_fraction" not in data


def test_recorded_support_round_trip_preserves_strict_provenance(tmp_path):
    output = recorded_support_pose_save(_support(), tmp_path / "support.json")
    loaded = recorded_support_pose_load(output)

    assert loaded.mesh_file_sha256 == "a" * 64
    assert loaded.pose_source == "recorded"
    assert loaded.fallback_used is False


def test_recorded_support_rejects_cross_mesh_identity_transfer(tmp_path):
    recording_dir = tmp_path / "einstar"
    target_dir = tmp_path / "sam3d"
    recording_dir.mkdir()
    target_dir.mkdir()
    recording_mesh = recording_dir / "output_aligned.glb"
    target_mesh = target_dir / "output_aligned.glb"
    recording_mesh.write_bytes(b"recording mesh")
    target_mesh.write_bytes(b"target mesh")
    _write_alignment(recording_dir / "output_symmetry.json")
    _write_alignment(target_dir / "output_symmetry.json")
    support = _support(mesh_file_sha256=file_sha256(recording_mesh))

    with pytest.raises(ValueError, match="identity aligned-frame transfer"):
        retarget_recorded_support_pose(
            support,
            recording_mesh_path=recording_mesh,
            target_mesh_path=target_mesh,
        )


def test_recorded_support_accepts_exact_mesh_transfer(tmp_path):
    mesh_dir = tmp_path / "einstar"
    mesh_dir.mkdir()
    mesh = mesh_dir / "output_aligned.glb"
    mesh.write_bytes(b"recording mesh")
    support = _support(mesh_file_sha256=file_sha256(mesh))

    retargeted = retarget_recorded_support_pose(
        support,
        recording_mesh_path=mesh,
        target_mesh_path=mesh,
    )

    assert retargeted.local_up == support.local_up
    assert retargeted.mesh_file_sha256 == file_sha256(mesh)
    assert retargeted.recording_mesh_file_sha256 == file_sha256(mesh)
    assert retargeted.mesh_frame_transfer["policy"] == "exact-mesh"


def test_foundation_pose_calibration_uses_target_mesh_local_up(tmp_path):
    target_mesh = tmp_path / "target.glb"
    target_mesh.write_bytes(b"target mesh")
    target_poses = np.repeat(np.eye(4)[None, :, :], 20, axis=0)
    target = _support(
        target_poses,
        mesh_file_sha256=file_sha256(target_mesh),
    )
    report_path = tmp_path / "foundation_pose_support_report.json"
    report_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "sequence_id": target.sequence_id,
                "target_mesh_file_sha256": target.mesh_file_sha256,
                "target_symmetry_file_sha256": "e" * 64,
                "poses_file_sha256": target.poses_file_sha256,
                "frame_start": target.frame_start,
                "frame_end_exclusive": target.frame_end_exclusive,
                "camera_provenance": {
                    "mode": "multi-view-foundation-pose",
                    "camera_names": ["front", "back"],
                    "registration_frame": target.frame_start,
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
                },
            }
        ),
        encoding="utf-8",
    )

    calibrated = foundation_pose_calibrated_support_pose(
        target,
        calibration_report_path=report_path,
    )
    output = recorded_support_pose_save(calibrated, tmp_path / "support.json")
    loaded = recorded_support_pose_load(output)

    assert loaded.local_up == pytest.approx((0.0, 0.0, 1.0))
    assert loaded.recording_mesh_file_sha256 == target.mesh_file_sha256
    assert loaded.mesh_frame_transfer["policy"] == (
        "foundation-pose-target-tracking"
    )
    assert loaded.mesh_frame_transfer["local_up_transform"] == (
        "direct-target-pose-inference"
    )
    assert loaded.mesh_frame_transfer["camera_provenance"][
        "highest_visibility_registration_camera"
    ] == "front"
    assert "reference_poses_file_sha256" not in loaded.mesh_frame_transfer


def test_recorded_support_loader_rejects_legacy_identity_transfer(tmp_path):
    output = tmp_path / "support.json"
    data = _support().to_dict()
    data["mesh_file_sha256"] = "f" * 64
    data["recording_mesh_file_sha256"] = "a" * 64
    data["mesh_frame_transfer"] = {
        "policy": "shared-output-aligned-frame",
        "local_up_transform": "identity",
        "recording_mesh_file_sha256": "a" * 64,
        "target_mesh_file_sha256": "f" * 64,
        "recording_symmetry_file_sha256": "c" * 64,
        "target_symmetry_file_sha256": "d" * 64,
    }
    output.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported.*transfer"):
        recorded_support_pose_load(output)


def test_recorded_support_loader_rejects_fallback_provenance(tmp_path):
    output = tmp_path / "support.json"
    data = _support().to_dict()
    data["fallback_used"] = True
    output.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="no fallback"):
        recorded_support_pose_load(output)
