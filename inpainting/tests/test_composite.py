from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from inpainting.composite_robot import (
    composite_robot,
    depth_visible_robot_mask,
    validate_robot_render_bundle,
    validate_taco_object_render_bundle,
)
from inpainting.contracts import ContractError, VideoGeometry
from inpainting.robot_renderer.backend import RENDER_METADATA_SCHEMA
from inpainting.taco_object_depth import OBJECT_RENDER_SCHEMA
from inpainting.video_io import probe_video


def _write_video(path: Path, values: list[int], size: tuple[int, int] = (16, 12)) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, size)
    if not writer.isOpened():
        raise RuntimeError("test video writer failed")
    for value in values:
        writer.write(np.full((size[1], size[0], 3), value, dtype=np.uint8))
    writer.release()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_robot_bundle(root: Path, frame_count: int = 3) -> tuple[Path, Path, Path, Path]:
    video = root / "robot_rgb.mp4"
    mask_path = root / "robot_mask.npy"
    depth_path = root / "robot_depth.npy"
    metadata_path = root / "render_metadata.json"
    _write_video(video, [220] * frame_count)
    mask = np.zeros((frame_count, 12, 16), dtype=np.bool_)
    mask[:, :, :8] = True
    depth = np.full(mask.shape, np.inf, dtype=np.float32)
    depth[mask] = 1.0
    np.save(mask_path, mask)
    np.save(depth_path, depth)
    metadata = {
        "schema_version": RENDER_METADATA_SCHEMA,
        "state": "complete",
        "run_id": "robot-test",
        "host_output_dir": str(root.resolve()),
        "geometry": probe_video(video).as_dict(),
        "artifacts": {
            "rgb": "/output/robot_rgb.mp4",
            "mask": "/output/robot_mask.npy",
            "depth": "/output/robot_depth.npy",
        },
        "artifact_bytes": {
            "rgb": video.stat().st_size,
            "mask": mask_path.stat().st_size,
            "depth": depth_path.stat().st_size,
        },
        "artifact_sha256": {
            "rgb": _sha256(video),
            "mask": _sha256(mask_path),
            "depth": _sha256(depth_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata))
    return video, mask_path, depth_path, metadata_path


def _write_object_bundle(root: Path, frame_count: int = 3) -> tuple[Path, Path, Path]:
    root.mkdir()
    mask_path = root / "object_mask.npy"
    depth_path = root / "object_depth.npy"
    metadata_path = root / "object_render_metadata.json"
    mask = np.zeros((frame_count, 12, 16), dtype=np.bool_)
    mask[:, :, :4] = True
    depth = np.full(mask.shape, np.inf, dtype=np.float32)
    depth[mask] = 0.5
    np.save(mask_path, mask)
    np.save(depth_path, depth)
    metadata = {
        "schema_version": OBJECT_RENDER_SCHEMA,
        "state": "complete",
        "run_id": "object-test",
        "host_output_dir": str(root.resolve()),
        "geometry": {
            "frame_count": frame_count,
            "width": 16,
            "height": 12,
            "fps": 10.0,
        },
        "artifacts": {
            "mask": "/output/object_mask.npy",
            "depth": "/output/object_depth.npy",
        },
        "artifact_bytes": {
            "mask": mask_path.stat().st_size,
            "depth": depth_path.stat().st_size,
        },
        "artifact_sha256": {
            "mask": _sha256(mask_path),
            "depth": _sha256(depth_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata))
    return mask_path, depth_path, metadata_path


class CompositeTest(unittest.TestCase):
    def test_hard_composite_preserves_geometry_and_requires_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.mp4"
            output = root / "out.mp4"
            _write_video(base, [10, 10, 10])
            robot, mask, _, sidecar = _write_robot_bundle(root)
            metadata = composite_robot(
                base,
                robot,
                mask,
                output,
                robot_metadata_path=sidecar,
            )
            self.assertEqual(metadata["frames_written"], 3)
            self.assertEqual(metadata["compositing"], "hard_robot_mask")
            self.assertEqual(probe_video(output).frame_count, 3)
            self.assertTrue(output.with_suffix(".json").is_file())

    def test_depth_aware_composite_removes_closer_object_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.mp4"
            output = root / "out.mp4"
            _write_video(base, [10, 10, 10])
            robot, mask, _, sidecar = _write_robot_bundle(root)
            object_mask, object_depth, object_sidecar = _write_object_bundle(
                root / "objects"
            )
            metadata = composite_robot(
                base,
                robot,
                mask,
                output,
                robot_metadata_path=sidecar,
                object_mask_path=object_mask,
                object_depth_path=object_depth,
                object_metadata_path=object_sidecar,
            )
            self.assertEqual(metadata["compositing"], "taco_object_depth")
            self.assertEqual(metadata["depth_guard_m"], 0.003)
            self.assertEqual(metadata["object_occluded_robot_pixels"], 3 * 12 * 4)

    def test_depth_mask_formula_honors_guard(self) -> None:
        robot_mask = np.array([True, True, True, False])
        object_mask = np.array([False, True, True, True])
        robot_depth = np.array([1.0, 1.0, 1.0, np.inf], dtype=np.float32)
        object_depth = np.array([np.inf, 0.995, 0.5, 0.5], dtype=np.float32)
        visible = depth_visible_robot_mask(
            robot_mask,
            robot_depth,
            object_mask,
            object_depth,
            depth_guard_m=0.01,
        )
        np.testing.assert_array_equal(visible, [True, True, False, False])

    def test_object_inputs_are_all_or_none(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.mp4"
            _write_video(base, [10, 10, 10])
            robot, mask, _, sidecar = _write_robot_bundle(root)
            object_mask = root / "object_mask.npy"
            np.save(object_mask, np.zeros((3, 12, 16), dtype=bool))
            with self.assertRaisesRegex(ContractError, "all-or-none"):
                composite_robot(
                    base,
                    robot,
                    mask,
                    root / "out.mp4",
                    robot_metadata_path=sidecar,
                    object_mask_path=object_mask,
                )

    def test_output_requires_explicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.mp4"
            output = root / "out.mp4"
            _write_video(base, [10, 10, 10])
            robot, mask, _, sidecar = _write_robot_bundle(root)
            kwargs = {"robot_metadata_path": sidecar}
            composite_robot(base, robot, mask, output, **kwargs)
            with self.assertRaises(FileExistsError):
                composite_robot(base, robot, mask, output, **kwargs)
            metadata = composite_robot(
                base, robot, mask, output, overwrite=True, **kwargs
            )
            self.assertTrue(metadata["overwrite"])

    def test_resolved_path_alias_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.mp4"
            _write_video(base, [10, 10, 10])
            robot, mask, _, sidecar = _write_robot_bundle(root)
            with self.assertRaisesRegex(ContractError, "Path alias"):
                composite_robot(
                    base,
                    robot,
                    mask,
                    base,
                    robot_metadata_path=sidecar,
                    overwrite=True,
                )

    def test_bundle_rejects_incomplete_state_and_size_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            robot, _, _, sidecar = _write_robot_bundle(root)
            geometry = probe_video(robot)
            metadata = json.loads(sidecar.read_text())
            metadata["state"] = "committing"
            sidecar.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ContractError, "state"):
                validate_robot_render_bundle(sidecar, geometry)
            metadata["state"] = "complete"
            metadata["artifact_bytes"]["mask"] += 1
            sidecar.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ContractError, "size mismatch"):
                validate_robot_render_bundle(sidecar, geometry)

    def test_object_bundle_rejects_same_size_hash_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask, _, sidecar = _write_object_bundle(root / "objects")
            geometry = VideoGeometry(frame_count=3, width=16, height=12, fps=10.0)
            payload = bytearray(mask.read_bytes())
            payload[-1] ^= 1
            mask.write_bytes(payload)
            with self.assertRaisesRegex(ContractError, "SHA-256 mismatch"):
                validate_taco_object_render_bundle(sidecar, geometry)

    def test_robot_bundle_requires_hashes_and_rejects_same_size_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            robot, mask, _, sidecar = _write_robot_bundle(root)
            geometry = probe_video(robot)
            metadata = json.loads(sidecar.read_text())
            metadata.pop("artifact_sha256")
            sidecar.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ContractError, "artifact_sha256"):
                validate_robot_render_bundle(sidecar, geometry)

            metadata["artifact_sha256"] = {
                "rgb": _sha256(robot),
                "mask": _sha256(mask),
                "depth": _sha256(root / "robot_depth.npy"),
            }
            sidecar.write_text(json.dumps(metadata))
            payload = bytearray(mask.read_bytes())
            payload[-1] ^= 1
            mask.write_bytes(payload)
            with self.assertRaisesRegex(ContractError, "SHA-256 mismatch"):
                validate_robot_render_bundle(sidecar, geometry)

    def test_transient_input_mutation_after_streaming_aborts_before_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.mp4"
            output = root / "out.mp4"
            _write_video(base, [10, 10, 10])
            robot, mask, _, sidecar = _write_robot_bundle(root)
            original_mask = mask.read_bytes()
            original_status = mask.stat()
            real_probe = probe_video
            mutated = False

            def mutate_after_encode(path: str | Path) -> VideoGeometry:
                nonlocal mutated
                geometry = real_probe(path)
                candidate = Path(path)
                if ".partial" in candidate.name and not mutated:
                    payload = bytearray(original_mask)
                    payload[-1] ^= 1
                    mask.write_bytes(payload)
                    mask.write_bytes(original_mask)
                    os.utime(
                        mask,
                        ns=(
                            original_status.st_atime_ns,
                            original_status.st_mtime_ns + 1_000_000_000,
                        ),
                    )
                    mutated = True
                return geometry

            with mock.patch(
                "inpainting.composite_robot.probe_video",
                side_effect=mutate_after_encode,
            ):
                with self.assertRaisesRegex(ContractError, "robot mask changed"):
                    composite_robot(
                        base,
                        robot,
                        mask,
                        output,
                        robot_metadata_path=sidecar,
                    )
            self.assertTrue(mutated)
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".json").exists())


if __name__ == "__main__":
    unittest.main()
