# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


DOCKER_PACKAGE = (
    Path(__file__).resolve().parents[3]
    / "reconstruction"
    / "modules"
    / "v2d_docker"
)
_SPEC = importlib.util.spec_from_file_location(
    "v2d_docker_container", DOCKER_PACKAGE / "container.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
run_in_container = _MODULE.run_in_container


class ContainerSecurityTests(unittest.TestCase):
    def test_offline_run_mounts_input_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "inputs"
            output_dir = root / "outputs"
            input_dir.mkdir()
            source = input_dir / "clip.mp4"
            source.write_bytes(b"video")
            output = output_dir / "result.mp4"

            with mock.patch("subprocess.run") as run:
                run_in_container(
                    image="stage:test",
                    module="example.stage",
                    inputs={"input_video": str(source)},
                    outputs={"output_video": str(output)},
                    network_disabled=True,
                )

            command = run.call_args.args[0]
            self.assertIn("none", command)
            self.assertEqual(command[command.index("--network") + 1], "none")
            mounts = [command[index + 1] for index, value in enumerate(command) if value == "-v"]
            self.assertIn(f"{input_dir}:/data/input_video:ro", mounts)
            self.assertIn(f"{output_dir}:/data/output_video", mounts)

    def test_explicit_gpu_device_does_not_expose_all_gpus(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input" / "clip.mp4"
            source.parent.mkdir()
            source.write_bytes(b"video")
            output = root / "output" / "result.mp4"

            with mock.patch("subprocess.run") as run:
                run_in_container(
                    image="stage:test",
                    module="example.stage",
                    inputs={"input_video": str(source)},
                    outputs={"output_video": str(output)},
                    gpus=True,
                    gpu_device=2,
                    env={"CUDA_VISIBLE_DEVICES": "0"},
                )

            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--gpus") + 1], "device=2")
            self.assertNotIn("all", command)
            self.assertIn("CUDA_VISIBLE_DEVICES=0", command)

    def test_legacy_gpu_flag_remains_backward_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "clip.mp4"
            source.write_bytes(b"video")
            output = root / "output" / "result.mp4"

            with mock.patch("subprocess.run") as run:
                run_in_container(
                    image="legacy:test",
                    module="example.stage",
                    inputs={"input_video": str(source)},
                    outputs={"output_video": str(output)},
                    gpus=True,
                )

            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--gpus") + 1], "all")

    def test_invalid_explicit_gpu_device_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative physical GPU index"):
            run_in_container(
                image="stage:test",
                module="example.stage",
                inputs={},
                outputs={},
                gpu_device=-1,
            )


if __name__ == "__main__":
    unittest.main()
