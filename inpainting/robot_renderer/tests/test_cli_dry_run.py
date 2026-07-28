from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
import tempfile
from pathlib import Path
import unittest

from inpainting.robot_renderer.cli import main

from inpainting.robot_renderer.tests.helpers import (
    create_fake_scene_utils,
    create_synthetic_asset_tree,
    save_inputs,
)


class CliDryRunTests(unittest.TestCase):
    def test_end_to_end_dry_run_needs_no_gpu_or_real_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trajectory, intrinsic, w2c = save_inputs(root)
            assets = create_synthetic_asset_tree(root / "assets")
            scene_utils = create_fake_scene_utils(root / "scene_utils")
            output = root / "not_created"
            captured = io.StringIO()
            with redirect_stdout(captured):
                result = main(
                    [
                        "--trajectory",
                        str(trajectory),
                        "--intrinsics",
                        str(intrinsic),
                        "--world-to-camera",
                        str(w2c),
                        "--width",
                        "640",
                        "--height",
                        "480",
                        "--fps",
                        "30",
                        "--asset-root",
                        str(assets),
                        "--scene-utils-root",
                        str(scene_utils),
                        "--output-dir",
                        str(output),
                        "--dry-run",
                    ]
                )
            self.assertEqual(result, 0)
            report = json.loads(captured.getvalue())
            self.assertEqual(report["state"], "validated")
            self.assertFalse(report["outputs_created"])
            self.assertTrue(report["ready_for_full_render"])
            self.assertFalse(output.exists())
            self.assertEqual(
                report["provenance"]["inputs"]["trajectory"]["sha256"],
                hashlib.sha256(trajectory.read_bytes()).hexdigest(),
            )
            source_paths = [
                record["path"]
                for record in report["provenance"]["renderer_source_files"]
            ]
            self.assertEqual(source_paths, sorted(source_paths))
            self.assertIn("inpainting/robot_renderer/backend.py", source_paths)
            self.assertIn("inpainting/robot_renderer/provenance.py", source_paths)
            self.assertNotIn("inpainting/robot_renderer/enrich_metadata.py", source_paths)
            for part in ("arms", "left_hand", "right_hand"):
                self.assertEqual(len(report["assets"][part]["urdf_file"]["sha256"]), 64)
                self.assertTrue(report["assets"][part]["referenced_asset_files"])


if __name__ == "__main__":
    unittest.main()
