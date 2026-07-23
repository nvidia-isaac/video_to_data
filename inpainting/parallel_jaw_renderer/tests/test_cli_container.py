from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

import numpy as np

from inpainting.parallel_jaw_renderer.cli import load_world_hub_from_metadata
from inpainting.parallel_jaw_renderer.container_runner import (
    ContainerConfig,
    build_docker_command,
)


class CliAndContainerTests(unittest.TestCase):
    def test_world_hub_metadata_requires_completed_explicit_mount(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "render_metadata.json"
            transform = np.eye(4)
            transform[2, 3] = 1.2
            path.write_text(
                json.dumps(
                    {
                        "state": "complete",
                        "kinematics": {"arm_center_world": transform.tolist()},
                    }
                )
            )
            np.testing.assert_allclose(load_world_hub_from_metadata(path), transform)
            path.write_text(json.dumps({"state": "complete", "kinematics": {}}))
            with self.assertRaisesRegex(ValueError, "arm_center_world"):
                load_world_hub_from_metadata(path)

    def test_container_command_mounts_every_external_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("target.npz", "bundle.json", "K.txt", "w2c.npy", "hub.json"):
                (root / name).write_bytes(name.encode())
            for name in ("assets", "scene_utils", "repository", "output"):
                (root / name).mkdir()
            config = ContainerConfig(
                target=root / "target.npz",
                bundle=root / "bundle.json",
                intrinsics=root / "K.txt",
                world_to_camera=root / "w2c.npy",
                robot_asset_root=root / "assets",
                scene_utils_root=root / "scene_utils",
                output_dir=root / "output",
                width=1920,
                height=1080,
                fps=30.0,
                repository_root=root / "repository",
                image_id="sha256:" + "a" * 64,
                T_world_hub_metadata=root / "hub.json",
                gpu="0",
                preview_frame_index=37,
            )
            command = build_docker_command(config)
            joined = " ".join(command)
            self.assertIn("/robot_assets", joined)
            self.assertIn("/robot_bundle/bundle.json", joined)
            self.assertIn("--T-world-hub-metadata", command)
            self.assertIn("--max-orientation-residual-deg", command)
            self.assertIn("--preview-frame-index", command)
            self.assertIn("device=0", command)


if __name__ == "__main__":
    unittest.main()
