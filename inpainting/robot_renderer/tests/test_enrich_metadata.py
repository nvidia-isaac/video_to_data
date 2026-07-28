from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from inpainting.robot_renderer.assets import resolve_robot_assets
from inpainting.robot_renderer.enrich_metadata import (
    MetadataEnrichmentError,
    enrich_render_metadata,
)
from inpainting.robot_renderer.inputs import load_render_inputs
from inpainting.robot_renderer.provenance import sha256_file
from inpainting.robot_renderer.tests.helpers import (
    create_synthetic_asset_tree,
    save_inputs,
)


IMAGE_ID = "sha256:" + "b" * 64
IMAGE = "robotic-grounding:photo-render-v6"


class EnrichMetadataTests(unittest.TestCase):
    def _bundle(self, root: Path) -> dict[str, Path]:
        trajectory, intrinsic, world_to_camera = save_inputs(root, frame_count=3)
        asset_root = create_synthetic_asset_tree(root / "assets")
        bundle = root / "robot_render"
        bundle.mkdir()
        rgb = bundle / "robot_rgb.mp4"
        mask_path = bundle / "robot_mask.npy"
        depth_path = bundle / "robot_depth.npy"
        metadata_path = bundle / "render_metadata.json"

        writer = cv2.VideoWriter(
            str(rgb), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 480)
        )
        self.assertTrue(writer.isOpened())
        for index in range(3):
            frame = np.full((480, 640, 3), index * 20, dtype=np.uint8)
            writer.write(frame)
        writer.release()

        mask = np.zeros((3, 480, 640), dtype=np.bool_)
        mask[:, 20:40, 30:50] = True
        depth = np.full(mask.shape, np.inf, dtype=np.float32)
        depth[mask] = 1.25
        np.save(mask_path, mask)
        np.save(depth_path, depth)
        counts = np.count_nonzero(mask, axis=(1, 2))

        assets = resolve_robot_assets(asset_root).as_dict()
        for part in ("arms", "left_hand", "right_hand"):
            assets[part].pop("urdf_file")
            assets[part].pop("referenced_asset_files")
        metadata = {
            "schema_version": "v2d.inpainting.robot-render/v1",
            "state": "complete",
            "container_image": IMAGE,
            "host_output_dir": str(bundle.resolve()),
            "geometry": {"frame_count": 3, "width": 640, "height": 480, "fps": 30.0},
            "trajectory": "/inputs/robot_trajectory.npz",
            "intrinsic": "/inputs/intrinsic.txt",
            "world_to_camera": "/inputs/world_to_camera.npy",
            "trajectory_coordinate_frame": "world",
            "projection_validation": {
                "left": {
                    "sample_count": 3,
                    "positive_depth_count": 3,
                    "inside_image_count": 3,
                    "depth_m_range": [2.0, 2.0],
                    "pixel_bounds": {"u": [295.0, 295.0], "v": [240.0, 240.0]},
                },
                "right": {
                    "sample_count": 3,
                    "positive_depth_count": 3,
                    "inside_image_count": 3,
                    "depth_m_range": [2.0, 2.0],
                    "pixel_bounds": {"u": [345.0, 345.0], "v": [240.0, 240.0]},
                },
            },
            "assets": assets,
            "artifacts": {
                "rgb": "/output/robot_rgb.mp4",
                "mask": "/output/robot_mask.npy",
                "depth": "/output/robot_depth.npy",
            },
            "artifact_bytes": {
                "rgb": rgb.stat().st_size,
                "mask": mask_path.stat().st_size,
                "depth": depth_path.stat().st_size,
            },
            "render_statistics": {
                "robot_pixel_count": int(counts.sum()),
                "mean_robot_pixels_per_frame": float(counts.mean()),
                "min_robot_pixels_per_frame": int(counts.min()),
                "max_robot_pixels_per_frame": int(counts.max()),
                "visibility_pixel_threshold": 16,
                "visible_frame_count": 3,
                "required_visible_frame_count": 1,
                "video_verification": {
                    "decoded_frame_count": 3,
                    "width": 640,
                    "height": 480,
                    "fps": 30.0,
                    "verification_backend": "opencv",
                },
            },
        }
        validated_inputs = load_render_inputs(
            trajectory_path=trajectory,
            intrinsic_path=intrinsic,
            world_to_camera_path=world_to_camera,
            width=640,
            height=480,
            fps=30.0,
        )
        metadata["trajectory_coordinate_frame"] = validated_inputs.coordinate_frame
        metadata["projection_validation"] = validated_inputs.projection_report()
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        return {
            "metadata": metadata_path,
            "trajectory": trajectory,
            "intrinsic": intrinsic,
            "world_to_camera": world_to_camera,
            "asset_root": asset_root,
            "rgb": rgb,
            "mask": mask_path,
            "depth": depth_path,
        }

    def _enrich(self, paths: dict[str, Path], *, write: bool = True) -> dict:
        return enrich_render_metadata(
            metadata_path=paths["metadata"],
            trajectory=paths["trajectory"],
            intrinsic=paths["intrinsic"],
            world_to_camera=paths["world_to_camera"],
            asset_root=paths["asset_root"],
            repository_root=Path(__file__).resolve().parents[3],
            image=IMAGE,
            image_id=IMAGE_ID,
            write=write,
        )

    def test_enriches_complete_bundle_atomically_and_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._bundle(Path(temporary))
            artifact_mtimes = {
                key: paths[key].stat().st_mtime_ns for key in ("rgb", "mask", "depth")
            }
            enriched = self._enrich(paths)
            on_disk = json.loads(paths["metadata"].read_text())
            self.assertEqual(enriched, on_disk)
            self.assertEqual(on_disk["container_image_id"], IMAGE_ID)
            self.assertEqual(
                on_disk["provenance"]["capture_mode"], "retrospective_enrichment"
            )
            for key in ("rgb", "mask", "depth"):
                self.assertEqual(on_disk["artifact_sha256"][key], sha256_file(paths[key]))
                self.assertEqual(paths[key].stat().st_mtime_ns, artifact_mtimes[key])
            for part in ("arms", "left_hand", "right_hand"):
                self.assertEqual(len(on_disk["assets"][part]["urdf_file"]["sha256"]), 64)
                self.assertTrue(on_disk["assets"][part]["referenced_asset_files"])
            self.assertFalse(list(paths["metadata"].parent.glob("*.partial")))

            first_bytes = paths["metadata"].read_bytes()
            self._enrich(paths)
            self.assertEqual(paths["metadata"].read_bytes(), first_bytes)

    def test_verify_only_does_not_replace_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._bundle(Path(temporary))
            before = paths["metadata"].read_bytes()
            result = self._enrich(paths, write=False)
            self.assertIn("artifact_sha256", result)
            self.assertEqual(paths["metadata"].read_bytes(), before)

    def test_refuses_non_complete_or_mismatched_bundle_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._bundle(Path(temporary))
            metadata = json.loads(paths["metadata"].read_text())
            metadata["state"] = "committing"
            paths["metadata"].write_text(json.dumps(metadata))
            before = paths["metadata"].read_bytes()
            with self.assertRaisesRegex(MetadataEnrichmentError, "non-complete"):
                self._enrich(paths)
            self.assertEqual(paths["metadata"].read_bytes(), before)

        with tempfile.TemporaryDirectory() as temporary:
            paths = self._bundle(Path(temporary))
            metadata = json.loads(paths["metadata"].read_text())
            metadata["artifact_bytes"]["depth"] += 1
            paths["metadata"].write_text(json.dumps(metadata))
            before = paths["metadata"].read_bytes()
            with self.assertRaisesRegex(MetadataEnrichmentError, "byte count mismatch"):
                self._enrich(paths)
            self.assertEqual(paths["metadata"].read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
