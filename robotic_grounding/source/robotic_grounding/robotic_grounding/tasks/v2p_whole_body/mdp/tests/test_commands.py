# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for tracking command."""

import unittest
from typing import Any

import torch
import yaml
from robotic_grounding.tests.utils import APP_IS_READY

if APP_IS_READY:
    from isaaclab.envs import ManagerBasedRLEnv
    from robotic_grounding.assets import MOTION_ASSET_DIR, SCENE_CONFIG_DIR
    from robotic_grounding.tasks.v2p_whole_body import G1SonicEEEnvCfg


@unittest.skipIf(not APP_IS_READY, "App is not ready")
class TestTrackingCommand(unittest.TestCase):
    """Test suite for tracking command."""

    env_cfg: Any
    motion_file: str
    env: Any
    tracking_cmd: Any
    expected_root_pos: torch.Tensor
    expected_root_quat: torch.Tensor
    expected_object_pos: torch.Tensor
    expected_object_quat: torch.Tensor

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the test environment once for all tests."""
        cls.env_cfg = G1SonicEEEnvCfg(
            scene_config_path=f"{SCENE_CONFIG_DIR}/apple_pick_optimized.yaml"
        )
        cls.env_cfg.commands.motion.object_pos_offset = [0.0, 0.0, 0.0]
        cls.env_cfg.commands.motion.robot_anchor_pos_offset = [0.0, 0.0, 0.0]
        cls.env_cfg.scene.num_envs = 2
        cls.motion_file = (
            f"{MOTION_ASSET_DIR}/object_pick_and_place_optimized_motion_with_ee.yaml"
        )
        cls.env = ManagerBasedRLEnv(cfg=cls.env_cfg)

        # Get tracking command
        cls.tracking_cmd = cls.env.command_manager.get_term("motion")

        # Load motion data to compare against
        with open(cls.motion_file, "r") as f:
            data = yaml.safe_load(f)
            qpos_data = torch.tensor(data["qpos"]).to(cls.env.device)
            object_pos_w = torch.tensor(data["object_position"]).to(cls.env.device)
            object_quat_w = torch.tensor(data["object_wxyz"]).to(cls.env.device)

        # Extract components
        cls.expected_root_pos = qpos_data[:, :3].float()
        cls.expected_root_quat = qpos_data[:, 3:7].float()
        cls.expected_object_pos = object_pos_w.float()
        cls.expected_object_quat = object_quat_w.float()

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up after all tests."""
        cls.env.close()

    def test_object_pos_w(self) -> None:
        """Test that object positions are correctly computed."""
        expected_shape = (self.env.num_envs, 3)
        self.assertEqual(
            self.tracking_cmd.command_object_pos_w.shape,
            expected_shape,
            f"Object positions should have shape {expected_shape}",
        )

        first_frame_object_pos = self.tracking_cmd.command_object_pos_w
        expected_first_frame_object_pos = (
            self.expected_object_pos[0] + self.env.scene.env_origins
        )
        self.assertTrue(
            torch.allclose(
                first_frame_object_pos, expected_first_frame_object_pos, atol=1e-5
            ),
            f"First frame object positions should match. Max diff: {torch.abs(first_frame_object_pos - expected_first_frame_object_pos).max()}",
        )

    def test_object_quat_w(self) -> None:
        """Test that object quaternions are correctly computed."""
        expected_shape = (self.env.num_envs, 4)
        self.assertEqual(
            self.tracking_cmd.command_object_quat_w.shape,
            expected_shape,
            f"Object quaternions should have shape {expected_shape}",
        )

        first_frame_object_quat = self.tracking_cmd.command_object_quat_w[0]
        expected_first_frame_object_quat = self.expected_object_quat[0]
        self.assertTrue(
            torch.allclose(
                first_frame_object_quat, expected_first_frame_object_quat, atol=1e-5
            ),
            f"First frame object quaternions should match. Max diff: {torch.abs(first_frame_object_quat - expected_first_frame_object_quat).max()}",
        )

    def test_command_anchor_z_multi_future(self) -> None:
        """Test that future z positions are correctly computed."""
        expected_shape = (self.env.num_envs, self.tracking_cmd.cfg.num_future_frames)
        self.assertEqual(
            self.tracking_cmd.command_anchor_z_multi_future.shape,
            expected_shape,
            f"Future z positions should have shape {expected_shape}",
        )

        first_frame_z = self.tracking_cmd.command_anchor_z_multi_future[0, 0]
        expected_first_frame_z = self.expected_root_pos[0, 2]
        self.assertTrue(
            torch.allclose(first_frame_z, expected_first_frame_z, atol=1e-4),
            f"First frame z position should match. Max diff: {torch.abs(first_frame_z - expected_first_frame_z).max()}",
        )

    def test_root_rot_dif_l_multi_future_shape(self) -> None:
        """Test that future root rotation differences are correctly computed."""
        root_rot_dif = self.tracking_cmd.command_anchor_rot_diff_l_multi_future
        expected_shape = (self.env.num_envs, self.tracking_cmd.cfg.num_future_frames, 6)

        self.assertEqual(
            root_rot_dif.shape,
            expected_shape,
            f"Future root rotation differences should have shape {expected_shape}",
        )

    def test_joint_pos_multi_future_shape(self) -> None:
        """Test that future joint positions have correct shape."""
        joint_pos_multi_future = self.tracking_cmd.command_joint_pos_multi_future
        expected_shape = (
            self.env.num_envs,
            self.tracking_cmd.cfg.num_future_frames,
            len(self.env.scene["robot"].data.joint_names),
        )

        self.assertEqual(
            joint_pos_multi_future.shape,
            expected_shape,
            f"Future joint positions should have shape {expected_shape}",
        )

    def test_joint_vel_multi_future_shape(self) -> None:
        """Test that future joint velocities have correct shape."""
        joint_vel_multi_future = self.tracking_cmd.command_joint_vel_multi_future
        expected_shape = (
            self.env.num_envs,
            self.tracking_cmd.cfg.num_future_frames,
            len(self.env.scene["robot"].data.joint_names),
        )

        self.assertEqual(
            joint_vel_multi_future.shape,
            expected_shape,
            f"Future joint velocities should have shape {expected_shape}",
        )

    def test_ee_pos_multi_future_shape(self) -> None:
        """Test that future EE positions have correct shape."""
        ee_pos_multi_future = self.tracking_cmd.command_ee_pos_w_multi_future
        num_ee_links = len(self.tracking_cmd.cfg.ee_link_names)
        expected_shape = (
            self.env.num_envs,
            self.tracking_cmd.cfg.num_future_frames,
            num_ee_links,
            3,
        )

        self.assertEqual(
            ee_pos_multi_future.shape,
            expected_shape,
            f"Future EE positions should have shape {expected_shape}",
        )

    def test_ee_pos_w_shape(self) -> None:
        """Test that current EE positions have correct shape."""
        ee_pos_w = self.tracking_cmd.command_ee_pos_w
        num_ee_links = len(self.tracking_cmd.cfg.ee_link_names)
        expected_shape = (self.env.num_envs, num_ee_links, 3)

        self.assertEqual(
            ee_pos_w.shape,
            expected_shape,
            f"EE positions should have shape {expected_shape}",
        )

    def test_ee_quat_w_shape(self) -> None:
        """Test that current EE quaternions have correct shape."""
        ee_quat_w = self.tracking_cmd.command_ee_quat_w
        num_ee_links = len(self.tracking_cmd.cfg.ee_link_names)
        expected_shape = (self.env.num_envs, num_ee_links, 4)

        self.assertEqual(
            ee_quat_w.shape,
            expected_shape,
            f"EE quaternions should have shape {expected_shape}",
        )

    def test_update_command(self) -> None:
        """Test that update_command correctly increments timestep."""
        self.tracking_cmd.timestep[:] = 0
        initial_timesteps = self.tracking_cmd.timestep.clone()
        self.tracking_cmd._update_command()
        expected_timesteps = initial_timesteps + 1
        self.assertTrue(
            torch.all(self.tracking_cmd.timestep == expected_timesteps),
            f"Timesteps should increment by 1. Expected: {expected_timesteps}, Got: {self.tracking_cmd.timestep}",
        )

    def test_update_command_wraps_at_end(self) -> None:
        """Test that update_command resets timestep when reaching end of trajectory."""
        self.tracking_cmd.timestep[:] = self.tracking_cmd.num_timesteps - 1
        self.tracking_cmd._update_command()
        self.assertTrue(
            torch.all(
                self.tracking_cmd.timestep == self.tracking_cmd.num_timesteps - 1
            ),
            f"Timesteps should stay at end till reset. Got: {self.tracking_cmd.timestep}",
        )

    def test_update_metrics_populates_expected_metric_shapes(self) -> None:
        """Test that update_metrics fills the expected per-env metric tensors."""
        self.tracking_cmd._update_metrics()

        expected_metric_names = [
            "anchor_position_error",
            "anchor_wxyz_error",
            "joint_pos_error",
            "left_hand_wrist_position_error",
            "left_hand_wrist_wxyz_error",
            "left_hand_finger_joints_error",
            "right_hand_wrist_position_error",
            "right_hand_wrist_wxyz_error",
            "right_hand_finger_joints_error",
            "object_body_position_error",
            "object_body_wxyz_error",
            "virtual_object_controller_scale_factor",
        ]

        for metric_name in expected_metric_names:
            self.assertIn(metric_name, self.tracking_cmd.metrics)
            metric = self.tracking_cmd.metrics[metric_name]
            self.assertEqual(
                metric.shape,
                (self.env.num_envs,),
                f"{metric_name} should be shape {(self.env.num_envs,)}",
            )
            self.assertTrue(
                torch.isfinite(metric).all(),
                f"{metric_name} should contain only finite values",
            )

    def test_update_metrics_tracks_per_env_voc_scale(self) -> None:
        """Test that VOC metric mirrors the currently applied per-env scale."""
        expected = torch.tensor([[0.25], [0.75]], device=self.env.device)
        self.tracking_cmd.virtual_object_controller_scale_factor_per_env[:] = expected

        self.tracking_cmd._update_metrics()

        self.assertTrue(
            torch.allclose(
                self.tracking_cmd.metrics["virtual_object_controller_scale_factor"],
                expected.squeeze(-1),
            ),
            "VOC metric should match per-env applied VOC scale",
        )


if __name__ == "__main__":
    unittest.main()
