from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from inpainting.robot_renderer.container_runner import (
    DEFAULT_IMAGE,
    ContainerConfig,
    build_docker_command,
    resolve_local_image_id,
    validate_gpu_selector,
)


IMAGE_ID = "sha256:" + "a" * 64


class ContainerRunnerTests(unittest.TestCase):
    def test_command_is_read_only_and_gpu_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "repo"
            assets = root / "assets"
            scene_utils = root / "scene_utils"
            output = root / "output"
            for directory in (repository, assets, scene_utils, output):
                directory.mkdir()
            trajectory = root / "trajectory.npz"
            intrinsic = root / "intrinsic.txt"
            w2c = root / "w2c.npy"
            for path in (trajectory, intrinsic, w2c):
                path.write_bytes(b"test")
            base = ContainerConfig(
                trajectory=trajectory,
                intrinsics=intrinsic,
                world_to_camera=w2c,
                asset_root=assets,
                scene_utils_root=scene_utils,
                output_dir=output,
                width=640,
                height=480,
                fps=30.0,
                repository_root=repository,
                image_id=IMAGE_ID,
                dry_run=True,
            )
            command = build_docker_command(base)
            self.assertNotIn("--gpus", command)
            self.assertNotIn("--user", command)
            self.assertIn("--dry-run", command)
            self.assertTrue(any("/external_assets,readonly" in value for value in command))
            self.assertTrue(any(value.startswith("V2D_RENDER_HOST_UID=") for value in command))
            self.assertIn(f"V2D_RENDER_CONTAINER_IMAGE={DEFAULT_IMAGE}", command)
            self.assertIn(f"V2D_RENDER_CONTAINER_IMAGE_ID={IMAGE_ID}", command)
            self.assertEqual(command[command.index("--entrypoint") + 2], IMAGE_ID)
            self.assertEqual(base.max_ik_residual_m, 0.01)
            self.assertEqual(base.max_joint_step_rad, 0.4)
            with_gpu = ContainerConfig(**{**base.__dict__, "gpu": "1"})
            gpu_command = build_docker_command(with_gpu)
            self.assertEqual(gpu_command[gpu_command.index("--gpus") + 1], "device=1")

    def test_input_file_mounts_allow_taco_commas_and_spaces(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "repo"
            assets = root / "assets"
            scene_utils = root / "scene_utils"
            output = root / "output"
            taco = root / "(dust, brush, cup)"
            for directory in (repository, assets, scene_utils, output, taco):
                directory.mkdir()
            trajectory = taco / "robot trajectory.npz"
            intrinsic = taco / "egocentric intrinsic.txt"
            w2c = taco / "egocentric frame extrinsic.npy"
            for path in (trajectory, intrinsic, w2c):
                path.write_bytes(b"test")
            config = ContainerConfig(
                trajectory=trajectory,
                intrinsics=intrinsic,
                world_to_camera=w2c,
                asset_root=assets,
                scene_utils_root=scene_utils,
                output_dir=output,
                width=640,
                height=480,
                fps=30.0,
                repository_root=repository,
                image_id=IMAGE_ID,
                dry_run=True,
            )
            command = build_docker_command(config)
            volume_specs = [
                command[index + 1]
                for index, token in enumerate(command)
                if token == "--volume"
            ]
            self.assertTrue(any(str(intrinsic) in value for value in volume_specs))
            self.assertTrue(any(str(w2c) in value for value in volume_specs))

    def test_gpu_selector_accepts_one_device_and_rejects_broad_access(self) -> None:
        gpu_uuid = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        self.assertEqual(validate_gpu_selector("0"), "0")
        self.assertEqual(validate_gpu_selector("17"), "17")
        self.assertEqual(validate_gpu_selector(gpu_uuid), gpu_uuid)
        self.assertIsNone(validate_gpu_selector(None))
        for invalid in ("", "all", "0,1", " 0", "GPU-short", "-1"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    validate_gpu_selector(invalid)

    @mock.patch("inpainting.robot_renderer.container_runner.subprocess.run")
    def test_resolves_and_validates_immutable_local_image_id(self, run: mock.Mock) -> None:
        run.return_value = SimpleNamespace(returncode=0, stdout=IMAGE_ID + "\n", stderr="")
        self.assertEqual(resolve_local_image_id(DEFAULT_IMAGE), IMAGE_ID)
        run.assert_called_once_with(
            ["docker", "image", "inspect", "--format", "{{.Id}}", DEFAULT_IMAGE],
            capture_output=True,
            text=True,
            check=False,
        )
        run.return_value = SimpleNamespace(returncode=0, stdout="not-a-digest\n", stderr="")
        with self.assertRaisesRegex(RuntimeError, "invalid immutable image ID"):
            resolve_local_image_id(DEFAULT_IMAGE)


if __name__ == "__main__":
    unittest.main()
