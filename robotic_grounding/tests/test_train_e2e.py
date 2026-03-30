#!/usr/bin/env python3

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""End-to-end tests for train script.

These tests verify that training can start and run for a few iterations.

⚠️ These tests REQUIRE a GPU to run.
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch


class TestTrainE2E(unittest.TestCase):
    """End-to-end tests for training tasks."""

    # Class variables set in setUpClass
    project_root: Path
    scripts_dir: Path
    isaaclab_path: str | None
    isaaclab_script: str

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the test environment."""
        cls.project_root = Path(__file__).parent.parent.absolute()
        cls.scripts_dir = cls.project_root / "scripts" / "rsl_rl"

        # Check for Isaac Lab
        cls.isaaclab_path = os.environ.get("ISAACLAB_PATH")
        if not cls.isaaclab_path:
            raise unittest.SkipTest("ISAACLAB_PATH environment variable is not set")

        cls.isaaclab_script = os.path.join(cls.isaaclab_path, "isaaclab.sh")
        if not os.path.exists(cls.isaaclab_script):
            raise unittest.SkipTest(
                f"IsaacLab script not found at {cls.isaaclab_script}"
            )

        # Check for GPU
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available - E2E tests require GPU")

        print(f"GPU available: {torch.cuda.get_device_name(0)}")

    @classmethod
    def get_tasks_to_test(cls) -> list[str]:
        """Get tasks to test for training.

        ⚠️ IMPORTANT: When you add a new task, add it here to ensure it works!
        """
        return [
            # ====================================================================
            # V2P TASKS
            # ====================================================================
            "Sharpa-V2P-v0",
            # ====================================================================
            # 🆕 ADD YOUR NEW TASKS HERE!
            # ====================================================================
        ]

    def _get_env_vars(self) -> dict:
        """Get environment variables for running commands."""
        env = dict(os.environ)
        env["WANDB_MODE"] = "disabled"
        env["OMNI_HEADLESS"] = "1"
        env["DISPLAY"] = ":1"
        return env

    def test_train_tasks(self) -> None:
        """Test that each task can be trained for a few iterations."""
        train_script = self.scripts_dir / "train.py"
        if not train_script.exists():
            self.skipTest(f"Train script not found at {train_script}")

        # Keep iterations small for CI
        num_iterations = 3
        num_envs = 4

        failed_tasks = []

        for task in self.get_tasks_to_test():
            with self.subTest(task=task):
                print(f"\n{'=' * 60}")
                print(f"Testing training: {task}")
                print("=" * 60)

                with tempfile.TemporaryDirectory() as temp_dir:
                    cmd = [
                        self.isaaclab_script,
                        "-p",
                        str(train_script),
                        "--task",
                        task,
                        "--max_iterations",
                        str(num_iterations),
                        "--num_envs",
                        str(num_envs),
                        "--headless",
                        # Hydra override to use temp directory
                        f"hydra.run.dir={temp_dir}",
                        "--motion_file",
                        "arctic_processed/arctic_s01_mixer_use_01/sharpa_wave",
                    ]

                    print(f"Command: {' '.join(cmd)}")

                    try:
                        result = subprocess.run(
                            cmd,
                            check=True,
                            timeout=180,  # 3 minutes timeout
                            capture_output=True,
                            text=True,
                            env=self._get_env_vars(),
                        )
                        print(f"✅ Training {task} passed")

                        if os.environ.get("VERBOSE_E2E_TESTS") == "true":
                            print("STDOUT (last 1000 chars):")
                            print(result.stdout[-1000:])

                    except subprocess.CalledProcessError as e:
                        failed_tasks.append(task)
                        print(f"❌ Training {task} failed (exit code {e.returncode})")
                        print("STDERR (last 2000 chars):")
                        print(e.stderr[-2000:] if e.stderr else "No stderr")

                    except subprocess.TimeoutExpired as e:
                        failed_tasks.append(task)
                        print(f"❌ Training {task} timed out")
                        if e.stdout:
                            print("Partial output:")
                            print(e.stdout[-2000:])

        # Summary
        print(f"\n{'=' * 60}")
        print("Training Test Summary")
        print("=" * 60)
        print(f"Total: {len(self.get_tasks_to_test())}")
        print(f"Passed: {len(self.get_tasks_to_test()) - len(failed_tasks)}")
        print(f"Failed: {len(failed_tasks)}")

        if failed_tasks:
            print("\nFailed tasks:")
            for task in failed_tasks:
                print(f"  - {task}")
            self.fail(f"{len(failed_tasks)} task(s) failed training test")
        else:
            print("\n✅ All training tests passed!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
