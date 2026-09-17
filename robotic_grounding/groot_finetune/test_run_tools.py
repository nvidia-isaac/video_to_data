# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the canonical GR00T run utilities and closed-loop launcher."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from groot_finetune.closed_loop.seeded_gr00t_server import _parse_args
from groot_finetune.contracts import VEGA_SHARPA_JOINT

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_EVAL = REPO_ROOT / "robotic_grounding/groot_finetune/closed_loop/run_eval.sh"
TASK_PROFILE = REPO_ROOT / (
    "robotic_grounding/groot_finetune/task_profiles/tissue_box_lift_hold.json"
)


def _run(*args: str | Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT / "robotic_grounding")
    return subprocess.run(
        tuple(str(value) for value in args),
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=environment,
    )


class CompatibilityTests(unittest.TestCase):
    def test_seeded_server_argument_split(self) -> None:
        seed, remainder = _parse_args(
            ["--seed", "17", "--model-path", "/tmp/model", "--port", "5555"]
        )
        self.assertEqual(seed, 17)
        self.assertEqual(remainder, ["--model-path", "/tmp/model", "--port", "5555"])

    def test_closed_loop_dry_run_is_seeded_and_scene_explicit(self) -> None:
        result = _run(
            "bash",
            RUN_EVAL,
            "--gr00t-dir",
            "/tmp/gr00t",
            "--model",
            "/tmp/checkpoint",
            "--container",
            "isaac",
            "--expected-mount-source",
            REPO_ROOT,
            "--client-workdir",
            "/workspace/video_to_data/robotic_grounding",
            "--task",
            "VegaSharpa-WholeBody-Gr00t-Joint-Inference-v0",
            "--contract",
            "/workspace/e2e/contracts/embodiment.json",
            "--task-profile",
            "/workspace/e2e/contracts/task_profile.json",
            "--motion-file",
            "ego_recon/processed/sequence_id=tissue_box_simple/robot_name=vega_sharpa",
            "--human-motion-data-dir",
            "/workspace/human_motion_data",
            "--expected-sequence-id",
            "tissue_box_simple",
            "--expected-robot-name",
            "vega_sharpa",
            "--output-json",
            "/workspace/eval.json",
            "--episodes",
            "20",
            "--num-envs",
            "4",
            "--episode-horizon",
            "519",
            "--model-seed",
            "17",
            "--dry-run",
        )
        self.assertIn("seeded_gr00t_server.py", result.stdout)
        self.assertIn("--seed 17", result.stdout)
        self.assertIn("HUMAN_MOTION_DATA_DIR", result.stdout)
        self.assertIn("--require_partitioned_motion", result.stdout)
        self.assertIn("--require_support_surface", result.stdout)
        self.assertIn("--checkpoint_manifest_sha256", result.stdout)
        self.assertIn("--eval_episode_horizon\\ 519", result.stdout)

    def test_planner_accepts_collection_only_measurements(self) -> None:
        result = _run(
            sys.executable,
            "-m",
            "groot_finetune.tools.plan_groot_run",
            "--target-successes",
            "10",
            "--measured-success-rate",
            "0.5",
            "--collection-safety-factor",
            "1.125",
        )
        plan = json.loads(result.stdout)
        self.assertEqual(plan["planned_collection_envs"], 23)
        self.assertIsNone(plan["optimizer_steps"])

    def test_planner_reports_joint_training_quantities(self) -> None:
        args = (
            "--target-successes",
            "10",
            "--measured-success-rate",
            "0.5",
            "--episodes",
            "10",
            "--frames-per-episode",
            "20",
            "--action-horizon",
            "4",
            "--epochs",
            "1",
            "--global-batch-size",
            "8",
        )
        result = _run(
            sys.executable,
            "-m",
            "groot_finetune.tools.plan_groot_run",
            *args,
        )
        plan = json.loads(result.stdout)
        self.assertEqual(plan["action_horizon"], 4)
        self.assertEqual(plan["usable_samples_per_episode"], 17)
        self.assertEqual(plan["optimizer_steps"], 22)

    def test_selector_writes_exact_success_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            source = tmp_path / "source"
            source.mkdir()
            (source / "manifest.json").write_text(
                json.dumps(
                    {
                        "format": "joint_rollout",
                        "schema_version": 1,
                        "embodiment_contract": VEGA_SHARPA_JOINT.contract_id,
                        "embodiment_contract_sha256": VEGA_SHARPA_JOINT.sha256,
                        "fps": VEGA_SHARPA_JOINT.fps,
                        "joint_names": list(VEGA_SHARPA_JOINT.joint_names),
                        "object_names": ["object"],
                        "timeout_termination": "timeout",
                    }
                )
            )
            np.savez(
                source / "episode_000000.npz",
                source_success=np.asarray(True),
                joint_pos=np.ones((3, 2), dtype=np.float32),
                action_target=np.ones((3, 2), dtype=np.float32),
                object_pose=np.ones((3, 1, 7), dtype=np.float32),
                termination_reasons=np.asarray(["timeout"]),
            )
            output = tmp_path / "selected"
            common = ("--input", source, "--target", "1", "--expected-frames", "3")
            _run(
                sys.executable,
                "-m",
                "groot_finetune.tools.select_successful_episodes",
                *common,
                "--output",
                output,
            )
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["selected_episode_count"], 1)
            self.assertEqual(manifest["expected_frames"], 3)

            audit = _run(
                sys.executable,
                "-m",
                "groot_finetune.tools.audit_groot_run",
                "--root",
                tmp_path,
                "--episodes",
                "1",
                "--frames",
                "3",
                "--contract",
                "vega_sharpa_joint",
                "--task-profile",
                TASK_PROFILE,
                "--selected-export",
                "selected",
            )
            self.assertTrue(json.loads(audit.stdout)["ok"])

            (output / "stale.txt").write_text("prior selection")
            with self.assertRaises(subprocess.CalledProcessError):
                _run(
                    sys.executable,
                    "-m",
                    "groot_finetune.tools.select_successful_episodes",
                    *common,
                    "--output",
                    output,
                )
            _run(
                sys.executable,
                "-m",
                "groot_finetune.tools.select_successful_episodes",
                *common,
                "--output",
                output,
                "--replace",
            )
            self.assertFalse((output / "stale.txt").exists())

    def test_closed_loop_checkpoint_fingerprint_includes_weight_bytes(self) -> None:
        def fingerprint(model: Path) -> str:
            result = _run(
                "bash",
                RUN_EVAL,
                "--gr00t-dir",
                "/tmp/gr00t",
                "--model",
                model,
                "--container",
                "isaac",
                "--expected-mount-source",
                REPO_ROOT,
                "--client-workdir",
                "/workspace/video_to_data/robotic_grounding",
                "--task",
                "VegaSharpa-WholeBody-Gr00t-Joint-Inference-v0",
                "--contract",
                "/workspace/e2e/contracts/embodiment.json",
                "--task-profile",
                "/workspace/e2e/contracts/task_profile.json",
                "--motion-file",
                "ego_recon/processed/sequence_id=test/robot_name=vega_sharpa",
                "--human-motion-data-dir",
                "/workspace/human_motion_data",
                "--expected-sequence-id",
                "test",
                "--expected-robot-name",
                "vega_sharpa",
                "--output-json",
                "/workspace/eval.json",
                "--episodes",
                "1",
                "--num-envs",
                "1",
                "--episode-horizon",
                "8",
                "--dry-run",
            )
            digests = re.findall(
                r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", result.stdout
            )
            self.assertEqual(len(digests), 1, result.stdout)
            return digests[0]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "checkpoint-a"
            second = root / "checkpoint-b"
            for checkpoint, weights in ((first, b"first"), (second, b"second")):
                checkpoint.mkdir()
                (checkpoint / "config.json").write_text("{}\n")
                (checkpoint / "model-00001-of-00001.safetensors").write_bytes(weights)

            self.assertNotEqual(fingerprint(first), fingerprint(second))

    def test_audit_reports_clean_empty_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            args = (
                "--root",
                tmp_path,
                "--episodes",
                "1",
                "--frames",
                "3",
                "--contract",
                "vega_sharpa_joint",
                "--task-profile",
                TASK_PROFILE,
            )
            result = _run(
                sys.executable,
                "-m",
                "groot_finetune.tools.audit_groot_run",
                *args,
            )
            audit = json.loads(result.stdout)
            self.assertTrue(audit["ok"])
            self.assertEqual(audit["errors"], [])


if __name__ == "__main__":
    unittest.main()
