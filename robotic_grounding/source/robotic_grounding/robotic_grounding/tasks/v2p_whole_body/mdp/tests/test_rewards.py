"""Unit tests for tracking rewards."""

import unittest
from typing import Any

import torch
from robotic_grounding.tests.utils import APP_IS_READY

if APP_IS_READY:
    from isaaclab.envs import ManagerBasedRLEnv
    from robotic_grounding.assets import MOTION_ASSET_DIR, SCENE_CONFIG_DIR
    from robotic_grounding.tasks.v2p_whole_body import G1SonicEEEnvCfg
    from robotic_grounding.tasks.v2p_whole_body.mdp.rewards import tracking_rewards


@unittest.skipIf(not APP_IS_READY, "App is not ready")
class TestTrackingRewards(unittest.TestCase):
    """Test suite for tracking rewards (including EE rewards)."""

    env_cfg: Any
    motion_file: str
    env: Any
    tracking_cmd: Any

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
        cls.tracking_cmd = cls.env.command_manager.get_term("motion")

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up after all tests."""
        cls.env.close()

    def test_motion_global_anchor_position_error_exp_zero_error(self) -> None:
        """Test that anchor position reward is 1 when tracking error is 0."""
        self.tracking_cmd.robot_anchor_pos_w[:] = (
            self.tracking_cmd.command_anchor_pos_w.clone()
        )
        reward = tracking_rewards.motion_global_anchor_position_error_exp(
            self.env, command_name="motion", std=1.0
        )
        self.assertEqual(
            reward.shape,
            (self.env.num_envs,),
            f"Reward should have shape ({self.env.num_envs},)",
        )
        expected = torch.ones(self.env.num_envs, device=self.env.device)
        self.assertTrue(
            torch.allclose(reward, expected, atol=1e-5),
            f"Reward should be 1 when error is 0. Got: {reward}",
        )

    def test_motion_global_anchor_position_error_exp_nonzero_error(self) -> None:
        """Test that anchor position reward is less than 1 when tracking error is nonzero."""
        self.tracking_cmd.robot_anchor_pos_w[:] = (
            self.tracking_cmd.command_anchor_pos_w.clone()
        )
        self.tracking_cmd.robot_anchor_pos_w[:, 0] += 1.0
        reward = tracking_rewards.motion_global_anchor_position_error_exp(
            self.env, command_name="motion", std=1.0
        )
        self.assertTrue(
            torch.all(reward < 1.0),
            f"Reward should be less than 1 when error is nonzero. Got: {reward}",
        )

    def test_motion_global_anchor_orientation_error_exp_zero_error(self) -> None:
        """Test that anchor orientation reward is 1 when tracking error is 0."""
        self.tracking_cmd.robot_anchor_quat_w[:] = (
            self.tracking_cmd.command_anchor_quat_w.clone()
        )
        reward = tracking_rewards.motion_global_anchor_orientation_error_exp(
            self.env, command_name="motion", std=1.0
        )
        self.assertEqual(
            reward.shape,
            (self.env.num_envs,),
            f"Reward should have shape ({self.env.num_envs},)",
        )
        expected = torch.ones(self.env.num_envs, device=self.env.device)
        self.assertTrue(
            torch.allclose(reward, expected, atol=1e-5),
            f"Reward should be 1 when error is 0. Got: {reward}",
        )

    def test_motion_global_anchor_orientation_error_exp_nonzero_error(self) -> None:
        """Test that anchor orientation reward is less than 1 when tracking error is nonzero."""
        self.tracking_cmd.robot_anchor_quat_w[:] = torch.tensor(
            [0.7071, 0.0, 0.0, 0.7071], device=self.env.device
        ).repeat(self.env.num_envs, 1)
        reward = tracking_rewards.motion_global_anchor_orientation_error_exp(
            self.env, command_name="motion", std=1.0
        )
        self.assertTrue(
            torch.all(reward < 1.0),
            f"Reward should be less than 1 when error is nonzero. Got: {reward}",
        )

    def test_motion_object_position_error_exp_zero_error(self) -> None:
        """Test that object position reward is 1 when tracking error is 0."""
        self.env.scene["object"].data.root_pos_w[
            :
        ] = self.tracking_cmd.command_object_pos_w.clone()
        reward = tracking_rewards.motion_object_position_error_exp(
            self.env, command_name="motion", std=1.0
        )
        self.assertEqual(
            reward.shape,
            (self.env.num_envs,),
            f"Reward should have shape ({self.env.num_envs},)",
        )
        expected = torch.ones(self.env.num_envs, device=self.env.device)
        self.assertTrue(
            torch.allclose(reward, expected, atol=1e-5),
            f"Reward should be 1 when error is 0. Got: {reward}",
        )

    def test_motion_object_position_error_exp_nonzero_error(self) -> None:
        """Test that object position reward is less than 1 when tracking error is nonzero."""
        self.env.scene["object"].data.root_pos_w[
            :
        ] = self.tracking_cmd.command_object_pos_w.clone()
        self.env.scene["object"].data.root_pos_w[:, 0] += 1.0
        reward = tracking_rewards.motion_object_position_error_exp(
            self.env, command_name="motion", std=1.0
        )
        self.assertTrue(
            torch.all(reward < 1.0),
            f"Reward should be less than 1 when error is nonzero. Got: {reward}",
        )

    def test_motion_object_orientation_error_exp_zero_error(self) -> None:
        """Test that object orientation reward is 1 when tracking error is 0."""
        self.env.scene["object"].data.root_quat_w[
            :
        ] = self.tracking_cmd.command_object_quat_w.clone()
        reward = tracking_rewards.motion_object_orientation_error_exp(
            self.env, command_name="motion", std=1.0
        )
        self.assertEqual(
            reward.shape,
            (self.env.num_envs,),
            f"Reward should have shape ({self.env.num_envs},)",
        )
        expected = torch.ones(self.env.num_envs, device=self.env.device)
        self.assertTrue(
            torch.allclose(reward, expected, atol=1e-5),
            f"Reward should be 1 when error is 0. Got: {reward}",
        )

    def test_motion_object_orientation_error_exp_nonzero_error(self) -> None:
        """Test that object orientation reward is less than 1 when tracking error is nonzero."""
        self.env.scene["object"].data.root_quat_w[:] = torch.tensor(
            [0.7071, 0.0, 0.0, 0.7071], device=self.env.device
        ).repeat(self.env.num_envs, 1)
        reward = tracking_rewards.motion_object_orientation_error_exp(
            self.env, command_name="motion", std=1.0
        )
        self.assertTrue(
            torch.all(reward < 1.0),
            f"Reward should be less than 1 when error is nonzero. Got: {reward}",
        )

    def test_motion_ee_position_error_exp(self) -> None:
        """Test EE position reward has correct shape and range."""
        reward = tracking_rewards.motion_ee_position_error_exp(
            self.env, command_name="motion", std=1.0
        )
        self.assertEqual(
            reward.shape,
            (self.env.num_envs,),
            f"Reward should have shape ({self.env.num_envs},)",
        )
        self.assertTrue(
            torch.all(reward >= 0.0) and torch.all(reward <= 1.0),
            f"Reward should be in [0, 1]. Got: {reward}",
        )

    def test_motion_ee_orientation_error_exp(self) -> None:
        """Test EE orientation reward has correct shape and range."""
        reward = tracking_rewards.motion_ee_orientation_error_exp(
            self.env, command_name="motion", std=1.0
        )
        self.assertEqual(
            reward.shape,
            (self.env.num_envs,),
            f"Reward should have shape ({self.env.num_envs},)",
        )
        self.assertTrue(
            torch.all(reward >= 0.0) and torch.all(reward <= 1.0),
            f"Reward should be in [0, 1]. Got: {reward}",
        )


if __name__ == "__main__":
    unittest.main()
