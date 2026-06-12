# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for SONIC environment instantiation."""

import unittest
from typing import Any

import torch
from robotic_grounding.tests.utils import APP_IS_READY

if APP_IS_READY:
    from isaaclab.envs import ManagerBasedRLEnv
    from robotic_grounding.assets import SCENE_CONFIG_DIR
    from robotic_grounding.tasks.v2p_whole_body import G1SonicEnvCfg


@unittest.skipIf(not APP_IS_READY, "App is not ready")
class TestSonicG1Env(unittest.TestCase):
    """Test that SONIC G1 environment can be instantiated correctly."""

    env: Any
    cfg: Any

    @classmethod
    def setUpClass(cls) -> None:
        """Create one environment for all tests in this class."""
        cfg = G1SonicEnvCfg(scene_config_path=f"{SCENE_CONFIG_DIR}/apple_pick.yaml")
        cfg.scene.num_envs = 2
        cfg.scene.env_spacing = 2.0
        cls.env = ManagerBasedRLEnv(cfg=cfg)
        cls.cfg = cfg

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up the environment after all tests."""
        cls.env.close()

    def test_config_instantiation(self) -> None:
        """Test that the SONIC configuration can be instantiated."""
        self.assertIsNotNone(self.cfg, "Configuration should be created")
        self.assertIsNotNone(self.cfg.scene, "Scene configuration should exist")
        self.assertIsNotNone(self.cfg.actions, "Actions configuration should exist")
        self.assertIsNotNone(
            self.cfg.observations, "Observations configuration should exist"
        )

    def test_environment_creation(self) -> None:
        """Test that the SONIC environment can be created."""
        self.assertIsNotNone(self.env, "Environment should be created")
        self.assertEqual(self.env.num_envs, 2, "Number of environments should match")

    def test_robot_in_scene(self) -> None:
        """Test that the robot is present in the scene."""
        # Check if robot exists in scene
        robot = self.env.scene["robot"]
        self.assertIsNotNone(robot, "Robot should not be None")

    def test_action_space(self) -> None:
        """Test that the action space is correctly defined."""
        # Check action space
        self.assertIsNotNone(self.env.action_space, "Action space should exist")
        self.assertGreater(
            self.env.action_space.shape[-1], 0, "Action space should have dimensions"
        )

    def test_observation_space(self) -> None:
        """Test that the observation space is correctly defined."""
        # Check observation space
        self.assertIsNotNone(
            self.env.observation_space, "Observation space should exist"
        )

        # Check observation groups
        if hasattr(self.cfg.observations, "sonic_tokenizer"):
            self.assertIn(
                "sonic_tokenizer",
                self.env.observation_space.keys(),
                "Tokenizer observation group should exist",
            )

        if hasattr(self.cfg.observations, "sonic_policy"):
            self.assertIn(
                "sonic_policy",
                self.env.observation_space.keys(),
                "Policy observation group should exist",
            )

    def test_step(self) -> None:
        """Test that the environment can step with actions."""
        # Reset first
        obs, info = self.env.reset()

        # Create random actions
        actions = torch.zeros(
            self.env.num_envs, self.env.action_space.shape[-1], device=self.env.device
        )

        # Step the environment
        obs, rewards, terminated, truncated, info = self.env.step(actions)

        # Check outputs
        self.assertIsNotNone(obs, "Observations should not be None")
        self.assertIsNotNone(rewards, "Rewards should not be None")
        self.assertIsNotNone(terminated, "Terminated should not be None")
        self.assertIsNotNone(truncated, "Truncated should not be None")

        # Check shapes
        self.assertEqual(
            rewards.shape[0], self.env.num_envs, "Rewards should match num_envs"
        )
        self.assertEqual(
            terminated.shape[0], self.env.num_envs, "Terminated should match num_envs"
        )
        self.assertEqual(
            truncated.shape[0], self.env.num_envs, "Truncated should match num_envs"
        )


if __name__ == "__main__":
    # Run tests
    unittest.main()
