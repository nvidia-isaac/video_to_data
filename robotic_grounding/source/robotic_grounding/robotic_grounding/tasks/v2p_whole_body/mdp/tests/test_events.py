# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for environment events."""

import unittest
from typing import Any

import h5py
import torch
from robotic_grounding.tests.utils import APP_IS_READY

if APP_IS_READY:
    from isaaclab.envs import ManagerBasedRLEnv
    from robotic_grounding.assets import MOTION_ASSET_DIR, SCENE_CONFIG_DIR
    from robotic_grounding.tasks.v2p_whole_body import G1SonicEnvCfg


@unittest.skipIf(not APP_IS_READY, "App is not ready")
class TestEvents(unittest.TestCase):
    """Test suite for events."""

    env_cfg: Any
    motion_file: str
    env: Any

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the test environment once for all tests."""
        cls.env_cfg = G1SonicEnvCfg(
            scene_config_path=f"{SCENE_CONFIG_DIR}/apple_pick.yaml"
        )
        cls.env_cfg.scene.num_envs = 2
        cls.env_cfg.commands.motion.object_pos_offset = [0.0, 0.0, 0.0]
        cls.env_cfg.commands.motion.robot_anchor_pos_offset = [0.0, 0.0, 0.0]
        cls.env_cfg.events.reset_to_motion_start.params["trajectory_time_index"] = (
            0,
            1,
        )
        cls.motion_file = (
            f"{MOTION_ASSET_DIR}/apple_pick_and_place_retarget_motion_w_body.h5"
        )
        cls.env = ManagerBasedRLEnv(cfg=cls.env_cfg)

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up after all tests."""
        cls.env.close()

    def test_reset_to_trajectory_start(self) -> None:
        """Test that reset_robot_to_trajectory_start correctly positions the robot."""
        # Load the motion data to compare against
        with h5py.File(self.motion_file, "r") as f:
            qpos_data = torch.from_numpy(f["qpos"][()]).to(self.env.device)

        # Extract first frame data
        expected_root_pos = qpos_data[0, :3]  # (3,)
        expected_root_quat = qpos_data[0, 3:7]  # (4,)

        # Reset the environment
        obs_dict, _ = self.env.reset()

        # Get tracking command
        tracking_cmd = self.env.command_manager.get_term("motion")
        robot = self.env.scene["robot"]

        # Tracking timestep should be 0
        self.assertTrue(
            torch.all(tracking_cmd.timestep == 0),
            f"Tracking timestep should be 0, got {tracking_cmd.timestep}",
        )

        # Root position should match first frame (plus env origins)
        for env_id in range(self.env.num_envs):
            expected_pos_w = expected_root_pos + self.env.scene.env_origins[env_id]
            actual_pos_w = robot.data.root_pos_w[env_id]

            pos_diff = torch.abs(expected_pos_w - actual_pos_w)
            self.assertTrue(
                torch.all(pos_diff < 0.01),  # 1cm tolerance
                f"Env {env_id}: Root position mismatch. "
                f"Expected: {expected_pos_w}, Got: {actual_pos_w}, Diff: {pos_diff}",
            )

        # Root orientation should match first frame
        for env_id in range(self.env.num_envs):
            expected_quat = expected_root_quat
            actual_quat = robot.data.root_quat_w[env_id]

            # Check quaternion similarity (q and -q represent same rotation)
            quat_diff = torch.min(
                torch.abs(expected_quat - actual_quat),
                torch.abs(expected_quat + actual_quat),
            )

            self.assertTrue(
                torch.all(quat_diff < 0.01),
                f"Env {env_id}: Root orientation mismatch. "
                f"Expected: {expected_quat}, Got: {actual_quat}, Diff: {quat_diff}",
            )


if __name__ == "__main__":
    # Run tests
    unittest.main()
