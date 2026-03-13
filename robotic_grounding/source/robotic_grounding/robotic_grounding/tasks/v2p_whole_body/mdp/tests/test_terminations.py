"""Unit tests for terminations."""

import unittest
from typing import Any

import torch
from robotic_grounding.tests.utils import APP_IS_READY

if APP_IS_READY:
    from isaaclab.envs import ManagerBasedRLEnv
    from robotic_grounding.assets import MOTION_ASSET_DIR, SCENE_CONFIG_DIR
    from robotic_grounding.tasks.v2p_whole_body import G1SonicEEEnvCfg
    from robotic_grounding.tasks.v2p_whole_body.mdp.terminations import (
        ee_position_error,
        ee_quat_error,
        timestep_termination,
    )


@unittest.skipIf(not APP_IS_READY, "App is not ready")
class TestTerminations(unittest.TestCase):
    """Test suite for terminations (including EE terminations)."""

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

    def test_timestep_termination(self) -> None:
        """Test that the timestep termination is correctly computed."""
        termination = timestep_termination(self.env, "motion")
        self.assertEqual(
            termination.shape,
            (self.env.num_envs,),
            "Termination should have shape (num_envs,)",
        )
        self.assertTrue(torch.all(termination == 0), "Termination should be 0")

        self.tracking_cmd.timestep += self.tracking_cmd.num_timesteps
        termination = timestep_termination(self.env, "motion")
        self.assertEqual(
            termination.shape,
            (self.env.num_envs,),
            "Termination should have shape (num_envs,)",
        )
        self.assertTrue(torch.all(termination == 1), "Termination should be 1")

        # Reset timestep for other tests
        self.tracking_cmd.timestep[:] = 0

    def test_ee_position_error(self) -> None:
        """Test that EE position error termination has correct shape and dtype."""
        termination = ee_position_error(self.env, "motion", threshold=1.0)
        self.assertEqual(
            termination.shape,
            (self.env.num_envs,),
            f"Termination should have shape ({self.env.num_envs},)",
        )
        self.assertEqual(
            termination.dtype,
            torch.bool,
            f"Termination should be boolean. Got: {termination.dtype}",
        )

    def test_ee_quat_error(self) -> None:
        """Test that EE quaternion error termination has correct shape and dtype."""
        termination = ee_quat_error(self.env, "motion", threshold=1.0)
        self.assertEqual(
            termination.shape,
            (self.env.num_envs,),
            f"Termination should have shape ({self.env.num_envs},)",
        )
        self.assertEqual(
            termination.dtype,
            torch.bool,
            f"Termination should be boolean. Got: {termination.dtype}",
        )


if __name__ == "__main__":
    unittest.main()
