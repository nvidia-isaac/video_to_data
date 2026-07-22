from __future__ import annotations

import unittest

import numpy as np

from inpainting.contracts import (
    ROBOT_TRAJECTORY_SCHEMA,
    TRACKING_SCHEMA,
    ContractError,
    VideoGeometry,
    validate_depth_array,
    validate_mask_array,
    validate_robot_trajectory_arrays,
    validate_tracking_arrays,
)
from inpainting.adapters.taco_ground_truth import arrays_from_row


def _valid_tracking(frame_count: int = 3) -> dict[str, np.ndarray]:
    data: dict[str, np.ndarray] = {
        "schema_version": np.asarray(TRACKING_SCHEMA),
        "tracker": np.asarray("v2d"),
        "coordinate_frame": np.asarray("camera"),
        "frame_indices": np.arange(frame_count),
    }
    for side in ("left", "right"):
        data[f"{side}_valid"] = np.ones(frame_count, dtype=bool)
        data[f"{side}_wrist_position"] = np.zeros((frame_count, 3), dtype=np.float32)
        quat = np.zeros((frame_count, 4), dtype=np.float32)
        quat[:, 0] = 1.0
        data[f"{side}_wrist_wxyz"] = quat
    return data


def _valid_trajectory(frame_count: int = 4, joints: int = 3) -> dict[str, np.ndarray]:
    data: dict[str, np.ndarray] = {
        "schema_version": np.asarray(ROBOT_TRAJECTORY_SCHEMA),
        "coordinate_frame": np.asarray("world"),
        "robot": np.asarray("dexmate_vega"),
        "gripper": np.asarray("sharpa_wave"),
        "frame_indices": np.arange(frame_count),
    }
    for side in ("left", "right"):
        data[f"{side}_valid"] = np.ones(frame_count, dtype=bool)
        data[f"{side}_wrist_position"] = np.zeros((frame_count, 3), dtype=np.float32)
        quaternion = np.zeros((frame_count, 4), dtype=np.float32)
        quaternion[:, 0] = 1
        data[f"{side}_wrist_wxyz"] = quaternion
        data[f"{side}_finger_joints"] = np.zeros(
            (frame_count, joints), dtype=np.float32
        )
        data[f"{side}_finger_joint_names"] = np.asarray(
            [f"joint_{index}" for index in range(joints)]
        )
    return data


class TrackingContractTest(unittest.TestCase):
    def test_valid_archive(self) -> None:
        self.assertEqual(validate_tracking_arrays(_valid_tracking(), expected_frames=3), 3)

    def test_non_unit_quaternion_rejected(self) -> None:
        data = _valid_tracking()
        data["left_wrist_wxyz"][1] = 0
        with self.assertRaisesRegex(ContractError, "non-unit"):
            validate_tracking_arrays(data)

    def test_non_unit_optional_joint_quaternion_rejected(self) -> None:
        data = _valid_tracking()
        joint_quaternions = np.zeros((3, 21, 4), dtype=np.float32)
        joint_quaternions[..., 0] = 1.0
        joint_quaternions[1, 5] = 0.0
        data["left_joints_wxyz"] = joint_quaternions
        with self.assertRaisesRegex(ContractError, "left_joints_wxyz.*non-unit"):
            validate_tracking_arrays(data)

    def test_non_contiguous_frames_rejected(self) -> None:
        data = _valid_tracking()
        data["frame_indices"][1] = 9
        with self.assertRaisesRegex(ContractError, "contiguous"):
            validate_tracking_arrays(data)

    def test_float_frame_indices_rejected(self) -> None:
        data = _valid_tracking()
        data["frame_indices"] = data["frame_indices"].astype(np.float32)
        with self.assertRaisesRegex(ContractError, "integer dtype"):
            validate_tracking_arrays(data)

    def test_non_boolean_validity_rejected(self) -> None:
        data = _valid_tracking()
        data["left_valid"] = np.ones(3, dtype=np.uint8)
        with self.assertRaisesRegex(ContractError, "boolean dtype"):
            validate_tracking_arrays(data)

    def test_non_finite_valid_tracking_row_rejected(self) -> None:
        data = _valid_tracking()
        data["left_wrist_position"][1, 0] = np.nan
        with self.assertRaisesRegex(ContractError, "non-finite"):
            validate_tracking_arrays(data)

    def test_non_scalar_metadata_rejected(self) -> None:
        data = _valid_tracking()
        data["tracker"] = np.asarray(["v2d"])
        with self.assertRaisesRegex(ContractError, "scalar string"):
            validate_tracking_arrays(data)


class MaskContractTest(unittest.TestCase):
    def test_numeric_mask_is_rejected(self) -> None:
        geometry = VideoGeometry(frame_count=2, width=4, height=3, fps=30)
        with self.assertRaisesRegex(ContractError, "boolean dtype"):
            validate_mask_array(np.ones((2, 3, 4), dtype=np.uint8), geometry)

    def test_boolean_mask_is_preserved(self) -> None:
        geometry = VideoGeometry(frame_count=2, width=4, height=3, fps=30)
        masks = np.ones((2, 3, 4), dtype=np.bool_)
        self.assertIs(validate_mask_array(masks, geometry), masks)

    def test_geometry_mismatch_rejected(self) -> None:
        geometry = VideoGeometry(frame_count=2, width=4, height=3, fps=30)
        with self.assertRaisesRegex(ContractError, "shape"):
            validate_mask_array(np.zeros((2, 4, 3), dtype=bool), geometry)


class DepthContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.geometry = VideoGeometry(frame_count=2, width=4, height=3, fps=30)
        self.mask = np.zeros((2, 3, 4), dtype=np.bool_)
        self.mask[:, 1, 2] = True
        self.depth = np.full((2, 3, 4), np.inf, dtype=np.float32)
        self.depth[self.mask] = 0.75

    def test_valid_metric_depth_is_preserved(self) -> None:
        self.assertIs(
            validate_depth_array(self.depth, self.mask, self.geometry), self.depth
        )

    def test_depth_requires_float32(self) -> None:
        with self.assertRaisesRegex(ContractError, "float32"):
            validate_depth_array(
                self.depth.astype(np.float64), self.mask, self.geometry
            )

    def test_masked_depth_must_be_finite_and_positive(self) -> None:
        self.depth[0, 1, 2] = 0.0
        with self.assertRaisesRegex(ContractError, "non-positive"):
            validate_depth_array(self.depth, self.mask, self.geometry)

    def test_unmasked_depth_must_be_positive_infinity(self) -> None:
        self.depth[0, 0, 0] = 1.0
        with self.assertRaisesRegex(ContractError, "outside"):
            validate_depth_array(self.depth, self.mask, self.geometry)


class RobotTrajectoryContractTest(unittest.TestCase):
    def test_valid_trajectory(self) -> None:
        self.assertEqual(validate_robot_trajectory_arrays(_valid_trajectory()), 4)

    def test_robot_and_gripper_must_be_scalar_strings(self) -> None:
        data = _valid_trajectory()
        data["robot"] = np.asarray(["dexmate_vega"])
        with self.assertRaisesRegex(ContractError, "scalar string"):
            validate_robot_trajectory_arrays(data)

    def test_non_finite_valid_robot_target_rejected(self) -> None:
        data = _valid_trajectory()
        data["right_finger_joints"][2, 1] = np.inf
        with self.assertRaisesRegex(ContractError, "non-finite"):
            validate_robot_trajectory_arrays(data)

    def test_non_finite_invalid_robot_target_allowed(self) -> None:
        data = _valid_trajectory()
        data["right_valid"][2] = False
        data["right_wrist_position"][2] = np.nan
        data["right_wrist_wxyz"][2] = np.nan
        data["right_finger_joints"][2] = np.nan
        self.assertEqual(validate_robot_trajectory_arrays(data), 4)

    def test_joint_names_require_string_dtype(self) -> None:
        data = _valid_trajectory()
        data["left_finger_joint_names"] = np.arange(3)
        with self.assertRaisesRegex(ContractError, "string dtype"):
            validate_robot_trajectory_arrays(data)


class TacoGroundTruthAdapterTest(unittest.TestCase):
    def test_robot_validity_uses_every_robot_target_field(self) -> None:
        frame_count, hand_joints, finger_joints = 4, 21, 3
        mano_positions = np.zeros((frame_count, hand_joints, 3), dtype=np.float32)
        mano_quaternions = np.zeros((frame_count, hand_joints, 4), dtype=np.float32)
        mano_quaternions[..., 0] = 1.0
        robot_positions = np.zeros((frame_count, 3), dtype=np.float32)
        robot_quaternions = np.zeros((frame_count, 4), dtype=np.float32)
        robot_quaternions[:, 0] = 1.0
        fingers = np.zeros((frame_count, finger_joints), dtype=np.float32)
        row = {
            "robot_name": "sharpa_wave",
            "mano_left_joints": mano_positions.copy(),
            "mano_right_joints": mano_positions.copy(),
            "mano_left_joints_wxyz": mano_quaternions.copy(),
            "mano_right_joints_wxyz": mano_quaternions.copy(),
            "robot_left_wrist_position": robot_positions.copy(),
            "robot_right_wrist_position": robot_positions.copy(),
            "robot_left_wrist_wxyz": robot_quaternions.copy(),
            "robot_right_wrist_wxyz": robot_quaternions.copy(),
            "robot_left_finger_joints": fingers.copy(),
            "robot_right_finger_joints": fingers.copy(),
            "left_robot_finger_joint_names": ["a", "b", "c"],
            "right_robot_finger_joint_names": ["a", "b", "c"],
        }
        row["robot_left_wrist_position"][1, 0] = np.nan
        row["robot_left_wrist_wxyz"][2, 1] = np.nan
        row["robot_left_finger_joints"][3, 2] = np.nan

        tracking, trajectory = arrays_from_row(row)

        np.testing.assert_array_equal(tracking["left_valid"], np.ones(4, dtype=bool))
        np.testing.assert_array_equal(
            trajectory["left_valid"], [True, False, False, False]
        )
        np.testing.assert_array_equal(trajectory["right_valid"], np.ones(4, dtype=bool))


if __name__ == "__main__":
    unittest.main()
