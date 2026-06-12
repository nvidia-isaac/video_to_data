# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for environment observations."""

import unittest
from typing import Any

import torch
from robotic_grounding.tests.utils import APP_IS_READY

if APP_IS_READY:
    from isaaclab.envs import ManagerBasedRLEnv
    from robotic_grounding.assets import MOTION_ASSET_DIR, SCENE_CONFIG_DIR
    from robotic_grounding.tasks.v2p_whole_body import G1SonicEEEnvCfg
    from robotic_grounding.tasks.v2p_whole_body.mdp.observations import (
        policy_observations as obs,
    )
    from robotic_grounding.tasks.v2p_whole_body.mdp.observations import (
        sonic_tokenizer_observations as tokenizer_obs,
    )


@unittest.skipIf(not APP_IS_READY, "App is not ready")
class TestObservations(unittest.TestCase):
    """Test suite for observations (including EE observations)."""

    env_cfg: Any
    motion_file: str
    env: Any

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the test environment once for all tests."""
        cls.env_cfg = G1SonicEEEnvCfg(
            scene_config_path=f"{SCENE_CONFIG_DIR}/apple_pick_optimized.yaml"
        )
        cls.env_cfg.scene.num_envs = 2
        cls.motion_file = (
            f"{MOTION_ASSET_DIR}/object_pick_and_place_optimized_motion_with_ee.yaml"
        )
        cls.env = ManagerBasedRLEnv(cfg=cls.env_cfg)

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up after all tests."""
        cls.env.close()

    def test_encoder_mode(self) -> None:
        """Test that the encoder mode is correctly computed."""
        encoder_mode = tokenizer_obs.encoder_mode(self.env, "motion")
        self.assertEqual(
            encoder_mode.shape,
            (self.env.num_envs, 4),
            "Encoder mode should have shape (num_envs, 4)",
        )
        expected = torch.zeros((self.env.num_envs, 4), device=self.env.device)
        self.assertTrue(
            torch.allclose(encoder_mode, expected), "Encoder mode should be all zeros"
        )

    def test_command_joint_pos(self) -> None:
        """Test that the command joint positions are correctly computed."""
        command_joint_pos = tokenizer_obs.command_joint_pos(self.env, "motion")
        num_joints = len(self.env.scene["robot"].data.joint_names)
        expected_shape = (
            self.env.num_envs,
            num_joints
            * self.env.command_manager.get_term("motion").cfg.num_future_frames,
        )
        self.assertEqual(
            command_joint_pos.shape,
            expected_shape,
            f"Command joint positions should have shape {expected_shape}",
        )

    def test_command_joint_vel(self) -> None:
        """Test that the command joint velocities are correctly computed."""
        command_joint_vel = tokenizer_obs.command_joint_vel(self.env, "motion")
        num_joints = len(self.env.scene["robot"].data.joint_names)
        expected_shape = (
            self.env.num_envs,
            num_joints
            * self.env.command_manager.get_term("motion").cfg.num_future_frames,
        )
        self.assertEqual(
            command_joint_vel.shape,
            expected_shape,
            f"Command joint velocities should have shape {expected_shape}",
        )

    def test_motion_anchor_ori_b(self) -> None:
        """Test that the motion anchor orientation differences are correctly computed."""
        motion_anchor_ori_b = tokenizer_obs.motion_anchor_ori_b(self.env, "motion")
        expected_shape = (
            self.env.num_envs,
            6 * self.env.command_manager.get_term("motion").cfg.num_future_frames,
        )
        self.assertEqual(
            motion_anchor_ori_b.shape,
            expected_shape,
            f"Motion anchor orientation differences should have shape {expected_shape}",
        )

    def test_encoder_padding(self) -> None:
        """Test that the encoder padding is correctly computed."""
        encoder_padding = tokenizer_obs.encoder_padding(self.env, 17)
        expected_shape = (self.env.num_envs, 17)
        self.assertEqual(
            encoder_padding.shape,
            expected_shape,
            f"Encoder padding should have shape {expected_shape}",
        )

    def test_motion_anchor_pos_b(self) -> None:
        """Test that the motion anchor positions are correctly computed."""
        motion_anchor_pos_b = obs.motion_anchor_pos_b(self.env, "motion")
        expected_shape = (
            self.env.num_envs,
            3 * self.env.command_manager.get_term("motion").cfg.num_future_frames,
        )
        self.assertEqual(
            motion_anchor_pos_b.shape,
            expected_shape,
            f"Motion anchor positions should have shape {expected_shape}",
        )

    def test_motion_joint_pos_delta(self) -> None:
        """Test that the motion joint position deltas are correctly computed."""
        motion_joint_pos_delta = obs.motion_joint_pos_delta(self.env, "motion")
        expected_shape = (
            self.env.num_envs,
            self.env.command_manager.get_term("motion").cfg.num_future_frames
            * len(self.env.scene["robot"].data.joint_names),
        )
        self.assertEqual(
            motion_joint_pos_delta.shape,
            expected_shape,
            f"Motion joint position deltas should have shape {expected_shape}",
        )

    def test_object_pos_delta(self) -> None:
        """Test that the object position deltas are correctly computed."""
        object_pos_delta = obs.object_pos_delta(self.env, "motion")
        expected_shape = (
            self.env.num_envs,
            3 * self.env.command_manager.get_term("motion").cfg.num_future_frames,
        )
        self.assertEqual(
            object_pos_delta.shape,
            expected_shape,
            f"Object position deltas should have shape {expected_shape}",
        )

    def test_command_trajectory_progress(self) -> None:
        """Test that the command trajectory progress is correctly computed."""
        command_trajectory_progress = obs.command_trajectory_progress(
            self.env, "motion"
        )
        expected_shape = (self.env.num_envs, 1)
        self.assertEqual(
            command_trajectory_progress.shape,
            expected_shape,
            f"Command trajectory progress should have shape {expected_shape}",
        )
        self.assertTrue(
            torch.allclose(
                command_trajectory_progress,
                torch.zeros((self.env.num_envs, 1), device=self.env.device),
            ),
            "Command trajectory progress should be 0",
        )

    def test_motion_ee_pos_delta(self) -> None:
        """Test that motion EE position deltas are correctly computed."""
        motion_ee_pos_delta = obs.motion_ee_pos_delta(self.env, "motion")
        num_future_frames = self.env.command_manager.get_term(
            "motion"
        ).cfg.num_future_frames
        num_ee_links = len(
            self.env.command_manager.get_term("motion").cfg.ee_link_names
        )
        expected_shape = (self.env.num_envs, num_future_frames * num_ee_links * 3)
        self.assertEqual(
            motion_ee_pos_delta.shape,
            expected_shape,
            f"Motion EE position deltas should have shape {expected_shape}",
        )

    def test_motion_ee_quat_delta(self) -> None:
        """Test that motion EE quaternion deltas are correctly computed."""
        motion_ee_quat_delta = obs.motion_ee_quat_delta(self.env, "motion")
        num_future_frames = self.env.command_manager.get_term(
            "motion"
        ).cfg.num_future_frames
        num_ee_links = len(
            self.env.command_manager.get_term("motion").cfg.ee_link_names
        )
        expected_shape = (self.env.num_envs, num_future_frames * num_ee_links * 6)
        self.assertEqual(
            motion_ee_quat_delta.shape,
            expected_shape,
            f"Motion EE quaternion deltas should have shape {expected_shape}",
        )


if __name__ == "__main__":
    unittest.main()
