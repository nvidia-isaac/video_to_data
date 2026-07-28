from __future__ import annotations

import tempfile
from pathlib import Path
import sys
import unittest

from inpainting.robot_renderer.external_ik import (
    load_external_ik,
    resolve_external_ik_sources,
)

from inpainting.robot_renderer.tests.helpers import create_fake_scene_utils


class ExternalIKTests(unittest.TestCase):
    def test_isolated_loader_resolves_relative_imports_without_package_init(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = create_fake_scene_utils(Path(temporary) / "scene_utils")
            before = {name for name in sys.modules if name.startswith("robotic_grounding")}
            modules = load_external_ik(root)
            self.assertEqual(modules.arm_ik.ArmIK.__name__, "ArmIK")
            placement = modules.arm_mount_opt.place_hub_from_wrists()
            self.assertEqual(placement[0], (0, 0, 0))
            self.assertNotIn("robotic_grounding", modules.arm_ik.__name__)
            after = {name for name in sys.modules if name.startswith("robotic_grounding")}
            self.assertEqual(after, before)

    def test_source_resolution_requires_both_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "arm_ik.py").write_text("class ArmIK: pass\n")
            with self.assertRaises(FileNotFoundError):
                resolve_external_ik_sources(root)


if __name__ == "__main__":
    unittest.main()
