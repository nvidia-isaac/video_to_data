from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat

import numpy as np
import pytest

from inpainting.adapters import parallel_jaw_from_tracking as parallel_jaw
from inpainting.contracts import TRACKING_SCHEMA


def _identity(values: np.ndarray) -> np.ndarray:
    return np.asarray(values).copy()


def _tracking_arrays(
    frame_count: int = 4,
    *,
    coordinate_frame: str = "world",
    tracker: str = "phantom",
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(TRACKING_SCHEMA),
        "tracker": np.asarray(tracker),
        "coordinate_frame": np.asarray(coordinate_frame),
        "frame_indices": np.arange(frame_count, dtype=np.int32),
    }
    for side_index, side in enumerate(("left", "right")):
        joints = np.zeros((frame_count, 21, 3), dtype=np.float32)
        side_offset = np.array([0.0, float(side_index), 0.0])
        for frame in range(frame_count):
            width = 0.04 + 0.01 * frame
            joints[frame] += side_offset + [0.1 * frame, 0.0, 0.0]
            joints[frame, 4] += [width / 2.0, 0.0, 0.0]
            joints[frame, 8] += [-width / 2.0, 0.0, 0.0]
            joints[frame, 12] += [0.0, 0.04, 0.0]
            joints[frame, 5] += [0.0, 0.0, 0.04]
        wrist_quaternions = np.zeros((frame_count, 4), dtype=np.float32)
        wrist_quaternions[:, 0] = 1.0
        arrays[f"{side}_valid"] = np.ones(frame_count, dtype=np.bool_)
        arrays[f"{side}_wrist_position"] = joints[:, 0].copy()
        arrays[f"{side}_wrist_wxyz"] = wrist_quaternions
        arrays[f"{side}_joints_3d"] = joints
    return arrays


def _write_tracking(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **arrays)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_raw_geometry_matches_phantom_without_robot_mount_rotation() -> None:
    joints = np.zeros((1, 21, 3), dtype=np.float64)
    joints[:, 4] = [1.0, 0.0, 0.0]
    joints[:, 8] = [-1.0, 0.0, 0.0]
    joints[:, 12] = [0.0, 2.0, 0.0]
    joints[:, 5] = [0.0, 0.0, 1.0]

    positions, rotations, widths, diagnostics = (
        parallel_jaw.derive_parallel_jaw_geometry(joints)
    )

    np.testing.assert_allclose(positions, [[0.5, 1.0, 0.0]])
    np.testing.assert_allclose(widths, [2.0])
    # x remains the thumb-minus-index direction. Phantom's later Rz(90)
    # gripper-mount correction would move this semantic axis.
    np.testing.assert_allclose(rotations[0, :, 0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(rotations[0], np.diag([1.0, -1.0, -1.0]))
    assert np.linalg.det(rotations[0]) == pytest.approx(1.0)
    assert diagnostics["minimum_thumb_index_distance_m"] == pytest.approx(2.0)


def test_camera_joints_use_inverse_T_camera_world_and_world_is_preserved() -> None:
    camera_tracking = _tracking_arrays(2, coordinate_frame="camera")
    world_to_camera = np.repeat(np.eye(4)[None], 2, axis=0)
    world_to_camera[:, :3, 3] = [1.0, 2.0, 3.0]

    converted = parallel_jaw.joints_in_world(
        camera_tracking, world_to_camera=world_to_camera
    )
    np.testing.assert_allclose(
        converted["left"],
        np.asarray(camera_tracking["left_joints_3d"]) - [1.0, 2.0, 3.0],
    )

    world_tracking = _tracking_arrays(2, coordinate_frame="world")
    preserved = parallel_jaw.joints_in_world(world_tracking, world_to_camera=None)
    np.testing.assert_array_equal(preserved["right"], world_tracking["right_joints_3d"])


def test_camera_tracking_never_infers_identity_calibration() -> None:
    tracking = _tracking_arrays(2, coordinate_frame="camera")
    with pytest.raises(
        parallel_jaw.ParallelJawRetargetError,
        match="identity transform is never inferred",
    ):
        parallel_jaw.joints_in_world(tracking, world_to_camera=None)


def test_grasp_span_cap_matches_phantom_sequence_policy() -> None:
    widths = np.array([1.0, 0.1, 0.6, 0.2, 1.0])
    capped, diagnostics = parallel_jaw.apply_grasp_span_width_cap(widths)

    # threshold = 0.1 + 20% * (1.0 - 0.1) = 0.28. The grasp interval spans
    # first-to-last below-threshold observation, so its middle is capped too.
    np.testing.assert_allclose(capped, [1.0, 0.1, 0.28, 0.2, 1.0])
    assert diagnostics["threshold_m"] == pytest.approx(0.28)
    assert diagnostics["first_grasp_frame"] == 1
    assert diagnostics["last_grasp_frame"] == 3
    assert diagnostics["capped_frame_count"] == 1

    constant, constant_diagnostics = parallel_jaw.apply_grasp_span_width_cap(
        np.full(3, 0.05)
    )
    np.testing.assert_allclose(constant, 0.05)
    assert constant_diagnostics["first_grasp_frame"] is None


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda arrays: arrays.pop("left_joints_3d"),
            "requires left_joints_3d",
        ),
        (
            lambda arrays: arrays["right_valid"].__setitem__(1, False),
            "right_valid must cover all",
        ),
        (
            lambda arrays: arrays["left_joints_3d"].__setitem__(
                (0, 8), arrays["left_joints_3d"][0, 4]
            ),
            "coincident thumb/index tips",
        ),
        (
            lambda arrays: arrays["right_joints_3d"].__setitem__(
                (0, 5),
                (arrays["right_joints_3d"][0, 4] + arrays["right_joints_3d"][0, 8])
                / 2.0,
            ),
            "degenerate index-MCP palm axis",
        ),
    ],
)
def test_invalid_or_degenerate_tracking_fails_closed(mutator, message: str) -> None:
    arrays = _tracking_arrays(3)
    mutator(arrays)
    with pytest.raises(parallel_jaw.ParallelJawRetargetError, match=message):
        parallel_jaw.retarget_tracking_arrays(
            arrays,
            position_smoother=_identity,
            width_smoother=_identity,
            orientation_smoother=_identity,
        )


def test_retarget_output_uses_exact_renderer_keys_and_all_valid_frames() -> None:
    arrays = _tracking_arrays(3, tracker="v2d")
    output, diagnostics = parallel_jaw.retarget_tracking_arrays(
        arrays,
        position_smoother=_identity,
        width_smoother=_identity,
        orientation_smoother=_identity,
    )

    assert set(output) == parallel_jaw.PARALLEL_JAW_KEYS
    assert output["schema_version"].item() == ("v2d.inpainting.parallel-jaw-target/v1")
    assert output["tracker"].item() == "v2d"
    assert output["coordinate_frame"].item() == "world"
    assert output["frame_indices"].dtype == np.int32
    for side in ("left", "right"):
        assert output[f"{side}_valid"].dtype == np.bool_
        assert output[f"{side}_valid"].all()
        assert output[f"{side}_position"].shape == (3, 3)
        assert output[f"{side}_position"].dtype == np.float32
        assert output[f"{side}_wxyz"].shape == (3, 4)
        assert output[f"{side}_wxyz"].dtype == np.float32
        assert output[f"{side}_aperture_m"].shape == (3,)
        assert output[f"{side}_aperture_m"].dtype == np.float32
        assert len(diagnostics["sides"][side]["raw_aperture_m"]) == 3
    assert parallel_jaw.validate_parallel_jaw_arrays(output) == 3


def test_execute_atomically_writes_hashed_npz_and_json(tmp_path: Path) -> None:
    tracking = tmp_path / "tracking.npz"
    _write_tracking(tracking, _tracking_arrays(4, tracker="ground_truth"))
    source_hash = _sha256(tracking)

    metadata = parallel_jaw.execute(
        tracking=tracking,
        output_dir=tmp_path / "output",
        position_smoother=_identity,
        width_smoother=_identity,
        orientation_smoother=_identity,
    )

    output = tmp_path / "output" / parallel_jaw.TRAJECTORY_FILENAME
    sidecar = tmp_path / "output" / parallel_jaw.METADATA_FILENAME
    assert output.is_file()
    assert sidecar.is_file()
    assert stat.S_IMODE(output.stat().st_mode) == 0o644
    assert stat.S_IMODE(sidecar.stat().st_mode) == 0o644
    assert not list((tmp_path / "output").glob("*.partial.*"))
    assert parallel_jaw.validate_parallel_jaw_file(output, expected_frames=4) == 4
    parsed = json.loads(sidecar.read_text())
    assert parsed == metadata
    assert parsed["state"] == "complete"
    assert parsed["all_frames_valid"] is True
    assert parsed["source"]["tracking"]["sha256"] == source_hash
    assert parsed["source"]["transform_policy"] == "preserve_world_coordinates"
    assert parsed["algorithm"]["geometry"]["robot_mount_rotation"] is None
    assert parsed["output"]["trajectory"]["sha256"] == _sha256(output)
    assert parsed["output"]["trajectory"]["size_bytes"] == output.stat().st_size


def test_execute_refuses_commit_if_source_changes_mid_run(tmp_path: Path) -> None:
    tracking = tmp_path / "tracking.npz"
    arrays = _tracking_arrays(3)
    _write_tracking(tracking, arrays)
    mutated = False

    def mutate_once(values: np.ndarray) -> np.ndarray:
        nonlocal mutated
        if not mutated:
            mutated = True
            changed = {key: value.copy() for key, value in arrays.items()}
            changed["left_joints_3d"][0, 0, 0] += 1.0
            _write_tracking(tracking, changed)
        return np.asarray(values).copy()

    with pytest.raises(
        parallel_jaw.ParallelJawRetargetError,
        match="changed while retargeting",
    ):
        parallel_jaw.execute(
            tracking=tracking,
            output_dir=tmp_path / "output",
            position_smoother=mutate_once,
            width_smoother=_identity,
            orientation_smoother=_identity,
        )
    assert not (tmp_path / "output" / parallel_jaw.TRAJECTORY_FILENAME).exists()
    assert not (tmp_path / "output" / parallel_jaw.METADATA_FILENAME).exists()
    assert not list((tmp_path / "output").glob("*.partial.*"))


def test_malformed_transform_and_schema_extras_are_rejected() -> None:
    with pytest.raises(parallel_jaw.ParallelJawRetargetError, match="orthonormal"):
        parallel_jaw._validate_transform_batch(
            np.diag([2.0, 1.0, 1.0, 1.0]), frame_count=2
        )

    arrays = _tracking_arrays(2)
    output, _ = parallel_jaw.retarget_tracking_arrays(
        arrays,
        position_smoother=_identity,
        width_smoother=_identity,
        orientation_smoother=_identity,
    )
    output["raw_width"] = np.zeros(2)
    with pytest.raises(parallel_jaw.ParallelJawRetargetError, match="unexpected"):
        parallel_jaw.validate_parallel_jaw_arrays(output)
