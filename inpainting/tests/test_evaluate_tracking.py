from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from inpainting import evaluate_tracking
from inpainting.contracts import ContractError, TRACKING_SCHEMA, VideoGeometry
from inpainting.evaluate_tracking import evaluate_arrays, evaluate_files
from inpainting.taco_camera import TacoCamera


def _tracking(
    *,
    tracker: str,
    coordinate_frame: str,
    joints: np.ndarray,
    valid: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    frame_count = joints.shape[0]
    quaternions = np.zeros((frame_count, 4), dtype=np.float32)
    quaternions[:, 0] = 1.0
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(TRACKING_SCHEMA),
        "tracker": np.asarray(tracker),
        "coordinate_frame": np.asarray(coordinate_frame),
        "frame_indices": np.arange(frame_count, dtype=np.int32),
    }
    for side in ("left", "right"):
        side_joints = joints.copy()
        if side == "right":
            side_joints[..., 1] += 0.2
        side_valid = (
            valid[side].copy()
            if valid is not None
            else np.ones(frame_count, dtype=np.bool_)
        )
        arrays.update(
            {
                f"{side}_valid": side_valid,
                f"{side}_wrist_position": side_joints[:, 0].astype(np.float32),
                f"{side}_wrist_wxyz": quaternions.copy(),
                f"{side}_joints_3d": side_joints.astype(np.float32),
            }
        )
    return arrays


def _fixture() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], TacoCamera]:
    frame_count = 3
    joints_world = np.zeros((frame_count, 21, 3), dtype=np.float64)
    joints_world[..., 2] = 2.0
    joints_world[:, :, 0] = np.arange(frame_count)[:, None] * 0.02
    joints_world[:, :, 1] = np.arange(21)[None] * 0.001
    transforms = np.repeat(np.eye(4)[None], frame_count, axis=0)
    for index, angle in enumerate((0.0, np.pi / 2.0, np.pi)):
        transforms[index, :3, :3] = [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    transforms[:, 0, 3] = [0.0, 0.2, 0.4]
    camera = TacoCamera(
        intrinsic=np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]),
        world_to_camera=transforms,
    )
    ground_truth = _tracking(
        tracker="ground_truth",
        coordinate_frame="world",
        joints=joints_world,
    )
    prediction_world = joints_world.copy()
    prediction_world[..., 0] += 0.1
    prediction_camera = np.einsum(
        "nij,nkj->nki", transforms[:, :3, :3], prediction_world
    )
    prediction_camera += transforms[:, None, :3, 3]
    prediction = _tracking(
        tracker="phantom",
        coordinate_frame="camera",
        joints=prediction_camera,
        valid={
            "left": np.ones(frame_count, dtype=np.bool_),
            "right": np.array([True, False, True], dtype=np.bool_),
        },
    )
    return prediction, ground_truth, camera


def test_metrics_use_coordinate_metadata_calibration_and_valid_masks() -> None:
    prediction, ground_truth, camera = _fixture()
    result = evaluate_arrays(
        prediction=prediction,
        ground_truth=ground_truth,
        camera=camera,
        sequence_id="synthetic",
    )

    left = result["sides"]["left"]
    assert left["metrics"]["wrist_3d_error_m"]["median"] == pytest.approx(0.1)
    assert left["metrics"]["joint_3d_mpjpe"]["per_frame_mpjpe_m"][
        "median"
    ] == pytest.approx(0.1)
    assert left["metrics"]["projected_2d_mpjpe"]["per_frame_mpjpe_px"][
        "median"
    ] == pytest.approx(5.0)
    # The camera translates by 0.2 m per frame, but temporal measurements are
    # correctly converted back to world and retain only the 0.02 m hand step.
    assert left["metrics"]["temporal_joint_step"]["prediction"][
        "per_frame_mean_joint_step_m"
    ]["median"] == pytest.approx(0.02)

    right = result["sides"]["right"]
    assert right["validity"]["prediction"]["valid_count"] == 2
    assert right["validity"]["prediction"]["invalid_gaps"] == [
        {"start_frame": 1, "end_frame": 1, "length": 1}
    ]
    assert right["validity"]["paired"]["valid_count"] == 2
    assert (
        right["metrics"]["temporal_joint_step"]["prediction"][
            "adjacent_frame_pair_count"
        ]
        == 0
    )
    assert result["ground_truth_policy"] == {
        "usage": "evaluation_only",
        "enters_prediction_or_retargeting": False,
    }


def test_joint_metrics_are_explicitly_unavailable_without_common_semantics() -> None:
    prediction, ground_truth, camera = _fixture()
    prediction.pop("left_joints_3d")
    result = evaluate_arrays(
        prediction=prediction,
        ground_truth=ground_truth,
        camera=camera,
        sequence_id="synthetic",
    )
    left = result["sides"]["left"]
    assert left["joint_semantics"]["compatible"] is False
    assert left["metrics"]["joint_3d_mpjpe"]["status"] == "not_computed"
    assert left["metrics"]["projected_2d_mpjpe"]["status"] == "not_computed"
    assert left["metrics"]["wrist_3d_error_m"]["status"] == "computed"


def test_reference_must_be_ground_truth_and_prediction_must_be_learned() -> None:
    prediction, ground_truth, camera = _fixture()
    ground_truth["tracker"] = np.asarray("v2d")
    with pytest.raises(ContractError, match="tracker='ground_truth'"):
        evaluate_arrays(
            prediction=prediction,
            ground_truth=ground_truth,
            camera=camera,
            sequence_id="synthetic",
        )

    prediction["tracker"] = np.asarray("ground_truth")
    ground_truth["tracker"] = np.asarray("ground_truth")
    with pytest.raises(ContractError, match="Prediction tracker"):
        evaluate_arrays(
            prediction=prediction,
            ground_truth=ground_truth,
            camera=camera,
            sequence_id="synthetic",
        )


def test_file_evaluation_is_deterministic_atomic_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prediction, ground_truth, camera = _fixture()
    prediction_path = tmp_path / "prediction.npz"
    ground_truth_path = tmp_path / "ground_truth.npz"
    video_path = tmp_path / "video.mp4"
    intrinsic_path = tmp_path / "intrinsic.txt"
    extrinsic_path = tmp_path / "extrinsic.npy"
    output_path = tmp_path / "evaluation.json"
    np.savez_compressed(prediction_path, **prediction)
    np.savez_compressed(ground_truth_path, **ground_truth)
    video_path.write_bytes(b"fingerprinted test video")
    np.savetxt(intrinsic_path, camera.intrinsic)
    np.save(extrinsic_path, camera.world_to_camera)
    monkeypatch.setattr(
        evaluate_tracking,
        "probe_video",
        lambda _: VideoGeometry(frame_count=3, width=100, height=80, fps=30.0),
    )

    evaluate_files(
        prediction_path=prediction_path,
        ground_truth_path=ground_truth_path,
        video_path=video_path,
        intrinsic_path=intrinsic_path,
        extrinsic_path=extrinsic_path,
        output_path=output_path,
        sequence_id="synthetic",
    )
    first = output_path.read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        evaluate_files(
            prediction_path=prediction_path,
            ground_truth_path=ground_truth_path,
            video_path=video_path,
            intrinsic_path=intrinsic_path,
            extrinsic_path=extrinsic_path,
            output_path=output_path,
            sequence_id="synthetic",
        )
    evaluate_files(
        prediction_path=prediction_path,
        ground_truth_path=ground_truth_path,
        video_path=video_path,
        intrinsic_path=intrinsic_path,
        extrinsic_path=extrinsic_path,
        output_path=output_path,
        sequence_id="synthetic",
        overwrite=True,
    )
    assert output_path.read_bytes() == first
    assert not list(tmp_path.glob("*.partial"))
    parsed = json.loads(first)
    assert parsed["inputs"]["prediction"]["sha256"]
    assert parsed["inputs"]["camera"]["world_to_camera"]["shape"] == [3, 4, 4]
