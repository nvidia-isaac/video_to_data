from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from inpainting.robot_renderer.backend import (
    _build_flange_targets,
    _metadata_base,
    flange_to_hand_transforms,
    validate_render_visibility,
    world_wrist_rows,
)
from inpainting.robot_renderer.backend import RenderBackendError
from inpainting.robot_renderer.external_ik import FLANGES
from inpainting.robot_renderer.transforms import validate_rigid_transform


class BackendPureTests(unittest.TestCase):
    def test_calibrated_flange_mount_direction(self) -> None:
        flange_to_hand = flange_to_hand_transforms()
        for transform in flange_to_hand.values():
            validate_rigid_transform(transform)
        inputs = SimpleNamespace(
            frame_count=1,
            left_world_wrist=flange_to_hand[FLANGES[0]][None, ...],
            right_world_wrist=flange_to_hand[FLANGES[1]][None, ...],
        )
        targets = _build_flange_targets(inputs, np.eye(4))
        for flange in FLANGES:
            np.testing.assert_allclose(targets[0][flange][0], np.zeros(3), atol=1e-12)
            np.testing.assert_allclose(targets[0][flange][1], (1, 0, 0, 0), atol=1e-8)

    def test_world_wrist_rows_preserve_pose(self) -> None:
        poses = np.tile(np.eye(4), (2, 1, 1))
        poses[:, :3, 3] = ((1, 2, 3), (4, 5, 6))
        rows = world_wrist_rows(poses)
        np.testing.assert_array_equal(rows[:, :3], poses[:, :3, 3])
        np.testing.assert_allclose(rows[:, 3:], ((1, 0, 0, 0), (1, 0, 0, 0)))

    def test_blank_render_cannot_be_committed(self) -> None:
        with self.assertRaisesRegex(RenderBackendError, "blank or nearly blank"):
            validate_render_visibility([0] * 20, width=640, height=480)
        statistics = validate_render_visibility([20] * 2 + [0] * 18, width=640, height=480)
        self.assertEqual(statistics["visible_frame_count"], 2)

    def test_render_metadata_records_selected_kinematics_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trajectory = root / "trajectory.npz"
            intrinsic = root / "intrinsic.txt"
            world_to_camera = root / "world_to_camera.npy"
            for path in (trajectory, intrinsic, world_to_camera):
                path.write_bytes(path.name.encode())
            inputs = SimpleNamespace(
                geometry=SimpleNamespace(
                    as_dict=lambda: {
                        "frame_count": 2,
                        "width": 6,
                        "height": 4,
                        "fps": 30.0,
                    }
                ),
                trajectory_path=trajectory,
                coordinate_frame="world",
                intrinsic_path=intrinsic,
                world_to_camera_path=world_to_camera,
                projection_report=lambda: {},
            )
            assets = SimpleNamespace(as_dict=lambda: {})
            metadata = _metadata_base(
                inputs,
                assets,
                output_dir=root,
                background_rgb=(0, 0, 0),
                max_position_residual_m=0.012,
                max_joint_step_rad=0.35,
            )
            self.assertEqual(
                metadata["kinematics_policy"],
                {
                    "max_position_residual_m": 0.012,
                    "max_joint_step_rad": 0.35,
                },
            )


if __name__ == "__main__":
    unittest.main()
