from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import inpainting.run_parallel_jaw_comparison as batch
from inpainting.run_parallel_jaw_comparison import (
    CONDITIONS,
    GRID_SCHEMA,
    PlanOptions,
    build_plan,
)


class ParallelJawBatchFixture:
    sequence_id = "sequence_demo"
    frame_count = 2
    width = 6
    height = 4
    fps = 30.0

    def __init__(self, root: Path):
        self.root = root
        self.run_root = root / "run"
        self.repository_root = root / "repository"
        self.camera_root = root / "camera"
        self.source_root = root / "source"
        self.old_package = root / "robotic_grounding"
        self.old_assets = self.old_package / "assets"
        self.scene_utils = self.old_package / "tasks" / "scene_utils"
        self.robot_asset_root = root / "parallel_robot_assets"
        self.bundle_root = root / "bundle"
        for directory in (
            self.run_root,
            self.repository_root,
            self.camera_root,
            self.source_root,
            self.old_assets,
            self.scene_utils,
            self.robot_asset_root,
            self.bundle_root,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        (self.scene_utils / "arm_ik.py").write_text("class ArmIK: pass\n")

        self.source_video = self.source_root / "color.mp4"
        self.source_video.write_bytes(b"source")
        camera_dir = self.camera_root / self.sequence_id
        camera_dir.mkdir()
        self.intrinsic = camera_dir / "egocentric_intrinsic.txt"
        self.world_to_camera = camera_dir / "egocentric_frame_extrinsic.npy"
        np.savetxt(
            self.intrinsic,
            np.asarray(((3.0, 0.0, 3.0), (0.0, 3.0, 2.0), (0.0, 0.0, 1.0))),
        )
        np.save(
            self.world_to_camera,
            np.tile(np.eye(4, dtype=np.float64), (self.frame_count, 1, 1)),
        )

        self.render_urdf = self.bundle_root / "render.urdf"
        self.ik_urdf = self.bundle_root / "ik.urdf"
        self.render_urdf.write_text("<robot name='render'><link name='root'/></robot>\n")
        self.ik_urdf.write_text("<robot name='ik'><link name='root'/></robot>\n")
        self.bundle = self.bundle_root / "bundle_manifest.json"
        self.bundle.write_text(
            json.dumps(
                {
                    "schema_version": "v2d.inpainting.parallel-jaw-robot-bundle/v1",
                    "robot_id": "fixture_robot",
                    "render_urdf": self.render_urdf.name,
                    "ik_urdf": self.ik_urdf.name,
                }
            )
        )

        sequence_root = self.run_root / self.sequence_id
        (sequence_root / "shared_arm_mask").mkdir(parents=True)
        (sequence_root / "shared_inpaint").mkdir()
        (sequence_root / "ground_truth" / "object_render").mkdir(parents=True)
        gt_robot = sequence_root / "ground_truth" / "robot_render"
        gt_robot.mkdir()
        (sequence_root / "shared_arm_mask" / "arm_mask.npy").write_bytes(b"mask")
        (sequence_root / "shared_inpaint" / "e2fgvi_960.mp4").write_bytes(
            b"inpaint"
        )
        (sequence_root / "shared_inpaint" / "e2fgvi_960.json").write_text("{}")
        (gt_robot / "render_metadata.json").write_text(
            json.dumps(
                {
                    "state": "complete",
                    "kinematics": {"arm_center_world": np.eye(4).tolist()},
                }
            )
        )
        for name in ("object_mask.npy", "object_depth.npy"):
            (sequence_root / "ground_truth" / "object_render" / name).write_bytes(
                name.encode()
            )
        (
            sequence_root
            / "ground_truth"
            / "object_render"
            / "object_render_metadata.json"
        ).write_text("{}")
        for condition in CONDITIONS:
            self._write_target(sequence_root, condition)

        self.manifest = self.run_root / "manifest.resolved.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "schema_version": "v2d.inpainting.resolved-experiment/v1",
                    "experiment_id": "fixture",
                    "roots": {
                        "camera": str(self.camera_root.resolve()),
                        "robot_assets": str(self.old_assets.resolve()),
                    },
                    "sequences": [
                        {
                            "sequence_id": self.sequence_id,
                            "video": {
                                "path": str(self.source_video.resolve()),
                                "frame_count": self.frame_count,
                                "width": self.width,
                                "height": self.height,
                                "fps": self.fps,
                            },
                            "camera": {
                                "intrinsic": str(self.intrinsic.resolve()),
                                "extrinsic": str(self.world_to_camera.resolve()),
                                "available": True,
                            },
                            "conditions": {
                                condition: {
                                    "tracker": condition,
                                    "blockers": [],
                                }
                                for condition in CONDITIONS
                            },
                        }
                    ],
                }
            )
        )

    def _write_target(self, sequence_root: Path, condition: str) -> None:
        output = sequence_root / "parallel_jaw" / "targets" / condition
        output.mkdir(parents=True)
        trajectory = output / "parallel_jaw_trajectory.npz"
        position = np.asarray(((0.0, 0.0, 1.0), (0.01, 0.0, 1.0)))
        quaternion = np.asarray(((1.0, 0.0, 0.0, 0.0),) * self.frame_count)
        np.savez(
            trajectory,
            schema_version=np.asarray("v2d.inpainting.parallel-jaw-target/v1"),
            tracker=np.asarray(condition),
            coordinate_frame=np.asarray("world"),
            frame_indices=np.arange(self.frame_count, dtype=np.int64),
            left_valid=np.ones(self.frame_count, dtype=bool),
            right_valid=np.ones(self.frame_count, dtype=bool),
            left_position=position,
            right_position=position,
            left_wxyz=quaternion,
            right_wxyz=quaternion,
            left_aperture_m=np.full(self.frame_count, 0.04),
            right_aperture_m=np.full(self.frame_count, 0.04),
        )
        metadata = output / "parallel_jaw_trajectory.json"
        metadata.write_text(
            json.dumps(
                {
                    "schema_version": (
                        "v2d.inpainting.parallel-jaw-retarget-run/v1"
                    ),
                    "state": "complete",
                    "tracker": condition,
                    "frame_count": self.frame_count,
                    "output": {
                        "trajectory": {
                            "filename": trajectory.name,
                            "size_bytes": trajectory.stat().st_size,
                            "sha256": hashlib.sha256(
                                trajectory.read_bytes()
                            ).hexdigest(),
                        }
                    },
                }
            )
        )

    def options(self, **overrides: object) -> PlanOptions:
        values: dict[str, object] = {
            "manifest_path": self.manifest,
            "sequence_id": self.sequence_id,
            "bundle": self.bundle,
            "robot_asset_root": self.robot_asset_root,
            "repository_root": self.repository_root,
            "python_executable": Path(sys.executable),
            "gpu": "2",
        }
        values.update(overrides)
        return PlanOptions(**values)


class ParallelJawComparisonBatchTests(unittest.TestCase):
    def test_read_only_plan_has_three_renders_three_composites_and_fixed_grid(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ParallelJawBatchFixture(Path(temporary))
            output_root = (
                fixture.run_root
                / fixture.sequence_id
                / "parallel_jaw"
                / "fixture_robot"
            )
            with (
                patch.object(batch, "_validate_inpaint", return_value=[]),
                patch.object(
                    batch,
                    "_object_depth_complete",
                    return_value=(True, "fixture object bundle"),
                ),
            ):
                plan = build_plan(fixture.options())

            self.assertFalse(output_root.exists())
            self.assertEqual(plan["summary"], {"pending": 7})
            actions = plan["actions"]
            self.assertEqual(
                [(item["condition"], item["stage"]) for item in actions],
                [
                    ("ground_truth", "render"),
                    ("v2d", "render"),
                    ("phantom", "render"),
                    ("ground_truth", "composite"),
                    ("v2d", "composite"),
                    ("phantom", "composite"),
                    (None, "grid"),
                ],
            )
            mount = str(
                fixture.run_root
                / fixture.sequence_id
                / "ground_truth"
                / "robot_render"
                / "render_metadata.json"
            )
            for action in actions[:3]:
                command = action["command"]
                self.assertIn(
                    "inpainting.parallel_jaw_renderer.container_runner", command
                )
                self.assertEqual(
                    command[command.index("--T-world-hub-metadata") + 1], mount
                )
                self.assertEqual(command[command.index("--gpu") + 1], "2")
                self.assertTrue(
                    action["outputs"][0].endswith(
                        f"fixture_robot/{action['condition']}/"
                        "robot_render/robot_rgb.mp4"
                    )
                )
            for action in actions[3:6]:
                command = action["command"]
                self.assertIn("inpainting.composite_robot", command)
                self.assertEqual(
                    command[command.index("--depth-guard-m") + 1], "0.003"
                )
            grid = actions[-1]["command"]
            self.assertIn("inpainting.make_video_grid", grid)
            self.assertEqual(grid.count("--video"), 5)
            self.assertEqual(grid[grid.index("--tile-width") + 1], "640")
            self.assertEqual(grid[grid.index("--columns") + 1], "3")
            self.assertEqual(
                plan["grid_specification"]["labels"][:2],
                ["Source", "E2FGVI (arms masked)"],
            )

    def test_changed_target_hash_blocks_that_render_and_downstream_grid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ParallelJawBatchFixture(Path(temporary))
            target = (
                fixture.run_root
                / fixture.sequence_id
                / "parallel_jaw"
                / "targets"
                / "v2d"
                / "parallel_jaw_trajectory.npz"
            )
            target.write_bytes(target.read_bytes() + b"changed")
            with (
                patch.object(batch, "_validate_inpaint", return_value=[]),
                patch.object(
                    batch,
                    "_object_depth_complete",
                    return_value=(True, "fixture object bundle"),
                ),
            ):
                plan = build_plan(fixture.options())
            by_key = {
                (item["condition"], item["stage"]): item for item in plan["actions"]
            }
            self.assertEqual(by_key[("v2d", "render")]["status"], "blocked")
            self.assertIn(
                "output size differs", by_key[("v2d", "render")]["reason"]
            )
            self.assertEqual(by_key[("v2d", "composite")]["status"], "blocked")
            self.assertEqual(by_key[(None, "grid")]["status"], "blocked")

    def test_grid_only_overwrite_does_not_schedule_render_or_composite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ParallelJawBatchFixture(Path(temporary))
            with (
                patch.object(batch, "_validate_inpaint", return_value=[]),
                patch.object(
                    batch,
                    "_object_depth_complete",
                    return_value=(True, "fixture object bundle"),
                ),
                patch.object(
                    batch,
                    "_parallel_render_complete",
                    return_value=(True, "complete render"),
                ),
                patch.object(
                    batch,
                    "_composite_complete",
                    return_value=(True, "complete composite"),
                ),
            ):
                plan = build_plan(
                    fixture.options(stages=("grid",), overwrite=True, gpu=None)
                )
            self.assertEqual(plan["selected_stages"], ["grid"])
            self.assertEqual(len(plan["actions"]), 1)
            self.assertEqual(plan["actions"][0]["stage"], "grid")
            self.assertEqual(plan["actions"][0]["status"], "pending_overwrite")
            self.assertNotIn("--gpu", plan["actions"][0]["command"])

    def test_grid_sidecar_is_atomic_and_hashes_inputs_lineage_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            videos = []
            for index in range(5):
                video = root / f"input_{index}.mp4"
                video.write_bytes(f"input-{index}".encode())
                videos.append(video)
            output = root / "final_5panel_comparison.mp4"
            output.write_bytes(b"grid-output")
            metadata = root / "final_5panel_comparison.json"
            lineage_file = root / "bundle.json"
            lineage_file.write_bytes(b"lineage")
            plan = {
                "sequence_id": "sequence_demo",
                "robot_id": "fixture_robot",
                "grid_specification": {
                    "videos": [str(path) for path in videos],
                    "labels": ["Source", "E2FGVI", "GT", "V2D", "Phantom"],
                    "tile_width": 640,
                    "columns": 3,
                    "max_frames": None,
                },
                "pipeline_policy": {"depth_guard_m": 0.003},
                "lineage_paths": {"robot_bundle": str(lineage_file)},
            }
            action = {
                "outputs": [str(output), str(metadata)],
            }
            geometry = SimpleNamespace(
                as_dict=lambda: {
                    "frame_count": 2,
                    "width": 1920,
                    "height": 720,
                    "fps": 30.0,
                }
            )
            with patch.object(batch, "probe_video", return_value=geometry):
                batch._write_grid_metadata(action, plan)

            payload = json.loads(metadata.read_text())
            self.assertEqual(payload["schema_version"], GRID_SCHEMA)
            self.assertEqual(payload["state"], "complete")
            self.assertEqual(
                set(payload["input_fingerprints"]),
                {"source", "e2fgvi", "ground_truth", "v2d", "phantom"},
            )
            self.assertEqual(
                payload["output_fingerprint"]["sha256"],
                hashlib.sha256(output.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                payload["lineage_fingerprints"]["robot_bundle"]["sha256"],
                hashlib.sha256(lineage_file.read_bytes()).hexdigest(),
            )
            self.assertEqual(list(root.glob(".*.partial")), [])


if __name__ == "__main__":
    unittest.main()
