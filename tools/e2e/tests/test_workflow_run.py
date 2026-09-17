"""Lightweight subprocess tests for the noninteractive container lifecycle."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_RUNNER = REPO_ROOT / "robotic_grounding/workflow/run.sh"
CONTAINER_NAME = "robotic-grounding-latest-gpu0"


class WorkflowRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run_root = self.root / "run"
        self.run_root.mkdir()
        self.fake_bin = self.root / "bin"
        self.fake_bin.mkdir()
        self.log = self.root / "docker.log"
        self.state = self.root / "running"
        self.state.write_text(CONTAINER_NAME)
        docker = self.fake_bin / "docker"
        docker.write_text(
            """#!/usr/bin/env bash
set -eu
printf '%s\n' "$*" >> "$FAKE_DOCKER_LOG"
if [ "$1" = "ps" ]; then
  if [ -f "$FAKE_DOCKER_STATE" ]; then printf '%s\n' 'robotic-grounding-latest-gpu0'; fi
elif [ "$1" = "inspect" ]; then
  printf '%s|%s\n' "$FAKE_REPO_ROOT" /workspace/video_to_data
  if [ "$FAKE_DOCKER_MODE" = "reuse" ]; then
    printf '%s|%s\n' "$FAKE_RUN_ROOT" /workspace/e2e
  fi
elif [ "$1" = "container" ] && [ "$2" = "inspect" ]; then
  test -f "$FAKE_DOCKER_STATE"
elif [ "$1" = "stop" ] || [ "$1" = "rm" ]; then
  rm -f "$FAKE_DOCKER_STATE"
elif [ "$1" = "run" ]; then
  printf '%s\n' 'robotic-grounding-latest-gpu0' > "$FAKE_DOCKER_STATE"
elif [ "$1" = "exec" ]; then
  exit 0
fi
"""
        )
        docker.chmod(0o755)
        xhost = self.fake_bin / "xhost"
        xhost.write_text("#!/usr/bin/env bash\nexit 0\n")
        xhost.chmod(0o755)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_e2e(
        self, mode: str, human_motion_data_dir: str = ""
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.update(
            {
                "PATH": f"{self.fake_bin}:{environment['PATH']}",
                "HOME": str(self.root / "home"),
                "DISPLAY": "",
                "SSH_AUTH_SOCK": "",
                "HUMAN_MOTION_DATA_DIR": human_motion_data_dir,
                "WANDB_API_KEY": "",
                "FAKE_DOCKER_LOG": str(self.log),
                "FAKE_DOCKER_STATE": str(self.state),
                "FAKE_DOCKER_MODE": mode,
                "FAKE_REPO_ROOT": str(REPO_ROOT),
                "FAKE_RUN_ROOT": str(self.run_root),
            }
        )
        return subprocess.run(
            [
                "bash",
                str(WORKFLOW_RUNNER),
                "e2e-run",
                "latest",
                "0",
                "--run-data",
                str(self.run_root),
                "--recreate-on-mount-change",
                "--",
                "true",
            ],
            cwd=REPO_ROOT / "robotic_grounding",
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_matching_mounts_reuse_running_container(self) -> None:
        result = self.run_e2e("reuse")
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.log.read_text()
        self.assertIn("exec --workdir", commands)
        self.assertNotIn("stop ", commands)
        self.assertNotIn("run --rm", commands)

    def test_mount_mismatch_recreates_before_exec(self) -> None:
        result = self.run_e2e("mismatch")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("required mount changed", result.stdout)
        commands = self.log.read_text()
        self.assertIn(f"stop {CONTAINER_NAME}", commands)
        self.assertIn(f"rm {CONTAINER_NAME}", commands)
        self.assertIn("run --rm", commands)
        self.assertIn("exec --workdir", commands)

    def test_external_motion_data_mounts_each_dataset(self) -> None:
        external_data = self.root / "human_motion_data"
        for dataset in ("arctic", "mano"):
            (external_data / dataset).mkdir(parents=True)

        result = self.run_e2e("mismatch", str(external_data))

        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.log.read_text()
        container_data = (
            "/workspace/video_to_data/robotic_grounding/source/robotic_grounding/"
            "robotic_grounding/assets/human_motion_data"
        )
        for dataset in ("arctic", "mano"):
            self.assertIn(
                f"-v {external_data / dataset}:{container_data}/{dataset}", commands
            )
        self.assertNotIn(f"-v {external_data}:{container_data}", commands)


if __name__ == "__main__":
    unittest.main()
