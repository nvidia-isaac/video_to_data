from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from inpainting.run_ground_truth_batch import (
    BatchPlanError,
    GRID_SCHEMA,
    PlanOptions,
    build_plan,
    execute_plan,
)
from inpainting.taco_object_depth import (
    OBJECT_RENDER_IMPLEMENTATION_FILES,
    OBJECT_RENDER_PROVENANCE_SCHEMA,
    object_render_source_records,
)


IMAGE_ID = "sha256:" + "a" * 64


class BatchFixture:
    frame_count = 2
    width = 6
    height = 4
    fps = 30.0
    sequence_id = "sequence_demo"

    def __init__(self, root: Path):
        self.root = root
        self.run_root = root / "run"
        self.repository_root = root / "repo"
        self.asset_package = root / "robotic_grounding"
        self.asset_root = self.asset_package / "assets"
        self.scene_utils = self.asset_package / "tasks" / "scene_utils"
        self.object_mesh_root = (
            self.asset_root
            / "human_motion_data"
            / "taco_v2.0"
            / "object_assets"
            / "meshes"
            / "taco"
        )
        self.camera_root = root / "official_camera"
        self.source_root = root / "rgb"
        for directory in (
            self.run_root,
            self.repository_root,
            self.asset_root,
            self.scene_utils,
            self.object_mesh_root,
            self.camera_root,
            self.source_root,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        for relative in OBJECT_RENDER_IMPLEMENTATION_FILES:
            source = self.repository_root / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(f"# fixture source: {relative}\n")
        (self.scene_utils / "arm_ik.py").write_text("class ArmIK: pass\n")
        (self.scene_utils / "arm_mount_opt.py").write_text(
            "def place_hub_from_wrists(*args, **kwargs): return None\n"
        )

        self.source_video = self.source_root / "color.mp4"
        self.source_video.write_bytes(b"source-video")
        self.motion_parquet = self.root / "motion" / "sequence.parquet"
        self.motion_parquet.parent.mkdir()
        self.motion_parquet.write_bytes(b"parquet-placeholder")
        camera_directory = self.camera_root / self.sequence_id
        camera_directory.mkdir()
        self.intrinsic = camera_directory / "egocentric_intrinsic.txt"
        self.world_to_camera = camera_directory / "egocentric_frame_extrinsic.npy"
        np.savetxt(
            self.intrinsic,
            np.array(((3.0, 0.0, 3.0), (0.0, 3.0, 2.0), (0.0, 0.0, 1.0))),
        )
        np.save(
            self.world_to_camera,
            np.tile(np.eye(4, dtype=np.float64), (self.frame_count, 1, 1)),
        )

        sequence_root = self.run_root / self.sequence_id
        tracking = sequence_root / "ground_truth" / "tracking"
        shared_arm_mask = sequence_root / "shared_arm_mask"
        shared_inpaint = sequence_root / "shared_inpaint"
        tracking.mkdir(parents=True)
        shared_arm_mask.mkdir(parents=True)
        shared_inpaint.mkdir(parents=True)
        self.trajectory = tracking / "robot_trajectory.npz"
        self.inpaint_masks = shared_arm_mask / "arm_mask.npy"
        self.inpaint_video = shared_inpaint / "e2fgvi_960.mp4"
        self.inpaint_metadata = shared_inpaint / "e2fgvi_960.json"
        self._write_trajectory()
        np.save(
            self.inpaint_masks,
            np.zeros((self.frame_count, self.height, self.width), dtype=bool),
        )
        self.inpaint_video.write_bytes(b"inpaint-video")
        self.inpaint_metadata.write_text(
            json.dumps(
                {
                    "schema": "v2d.e2fgvi.inpainting.v1",
                    "status": "completed",
                    "implementation": {
                        "container_image": "v2d_e2fgvi",
                        "container_image_id": IMAGE_ID,
                        "container_image_provenance": "recorded_immutable_id",
                    },
                    "inputs": {
                        "video": {
                            "name": self.source_video.name,
                            **self._fingerprint(self.source_video),
                        },
                        "masks": {
                            "name": self.inpaint_masks.name,
                            **self._fingerprint(self.inpaint_masks),
                        },
                        "checkpoint": {
                            "name": "E2FGVI-HQ-CVPR22.pth",
                            "bytes": 164535938,
                            "sha256": "f" * 64,
                        },
                    },
                    "output": {
                        "frame_count": self.frame_count,
                        "width": self.width,
                        "height": self.height,
                        "fps": self.fps,
                        "name": self.inpaint_video.name,
                        **self._fingerprint(self.inpaint_video),
                    },
                }
            )
        )
        # Keep the fixture's causal ordering deterministic instead of relying
        # on the host clock (completed artifacts below use increasing mtimes).
        self._set_mtime(
            (
                self.source_video,
                self.inpaint_masks,
                self.inpaint_video,
                self.inpaint_metadata,
            ),
            1_000_000_000,
        )

        self.manifest = self.run_root / "manifest.resolved.json"
        self.manifest.write_text(json.dumps(self._manifest_payload()))

    @property
    def condition(self) -> Path:
        return self.run_root / self.sequence_id / "ground_truth"

    @property
    def robot_dir(self) -> Path:
        return self.condition / "robot_render"

    def options(self, **overrides) -> PlanOptions:
        values = {
            "manifest_path": self.manifest,
            "repository_root": self.repository_root,
            "python_executable": Path(sys.executable),
        }
        values.update(overrides)
        return PlanOptions(**values)

    def _manifest_payload(self) -> dict:
        return {
            "schema_version": "v2d.inpainting.resolved-experiment/v1",
            "source_schema_version": "v2d.inpainting.experiment/v1",
            "experiment_id": "test_gt_batch",
            "trackers": ["phantom", "v2d", "ground_truth"],
            "inpainting": "e2fgvi_hq",
            "robot": "dexmate_vega",
            "gripper": "sharpa_wave",
            "roots": {
                "rgb": str(self.source_root.resolve()),
                "motion": str((self.root / "motion").resolve()),
                "camera": str(self.camera_root.resolve()),
                "robot_assets": str(self.asset_root.resolve()),
            },
            "robot_assets_available": True,
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
                    "motion": {
                        "path": str(self.motion_parquet.resolve()),
                        "size_bytes": self.motion_parquet.stat().st_size,
                        "frame_lengths": {},
                    },
                    "camera": {
                        "directory": str(self.intrinsic.parent.resolve()),
                        "intrinsic": str(self.intrinsic.resolve()),
                        "extrinsic": str(self.world_to_camera.resolve()),
                        "available": True,
                    },
                    "conditions": {
                        "ground_truth": {
                            "tracker": "ground_truth",
                            "state": "source_inputs_resolved",
                            "blockers": [],
                        }
                    },
                }
            ],
        }

    def _write_trajectory(self) -> None:
        quaternion = np.tile(
            np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
            (self.frame_count, 1),
        )
        position = np.tile(
            np.array((0.0, 0.0, 1.0), dtype=np.float32),
            (self.frame_count, 1),
        )
        np.savez(
            self.trajectory,
            schema_version=np.asarray("v2d.inpainting.robot-trajectory/v1"),
            coordinate_frame=np.asarray("world"),
            robot=np.asarray("dexmate_vega"),
            gripper=np.asarray("sharpa_wave"),
            frame_indices=np.arange(self.frame_count, dtype=np.int32),
            left_valid=np.ones(self.frame_count, dtype=bool),
            right_valid=np.ones(self.frame_count, dtype=bool),
            left_wrist_position=position,
            right_wrist_position=position,
            left_wrist_wxyz=quaternion,
            right_wrist_wxyz=quaternion,
            left_finger_joints=np.zeros((self.frame_count, 1), dtype=np.float32),
            right_finger_joints=np.zeros((self.frame_count, 1), dtype=np.float32),
            left_finger_joint_names=np.asarray(("left_joint",)),
            right_finger_joint_names=np.asarray(("right_joint",)),
        )

    def write_complete_render(self) -> None:
        self.robot_dir.mkdir(parents=True, exist_ok=True)
        rgb = self.robot_dir / "robot_rgb.mp4"
        mask = self.robot_dir / "robot_mask.npy"
        depth = self.robot_dir / "robot_depth.npy"
        metadata = self.robot_dir / "render_metadata.json"
        rgb.write_bytes(b"robot-video")
        np.save(mask, np.zeros((self.frame_count, self.height, self.width), dtype=bool))
        np.save(
            depth,
            np.full(
                (self.frame_count, self.height, self.width), np.inf, dtype=np.float32
            ),
        )
        self._set_mtime((rgb, mask, depth), 2_000_000_000)
        metadata.write_text(
            json.dumps(
                {
                    "schema_version": "v2d.inpainting.robot-render/v1",
                    "state": "complete",
                    "run_id": "test-run",
                    "geometry": self.geometry,
                    "host_output_dir": str(self.robot_dir.resolve()),
                    "artifacts": {
                        "rgb": str(rgb.resolve()),
                        "mask": str(mask.resolve()),
                        "depth": str(depth.resolve()),
                    },
                    "render_statistics": {
                        "video_verification": {
                            "decoded_frame_count": self.frame_count,
                            "width": self.width,
                            "height": self.height,
                            "fps": self.fps,
                        }
                    },
                    "artifact_bytes": {
                        "rgb": rgb.stat().st_size,
                        "mask": mask.stat().st_size,
                        "depth": depth.stat().st_size,
                    },
                    "artifact_sha256": {
                        "rgb": self._sha256(rgb),
                        "mask": self._sha256(mask),
                        "depth": self._sha256(depth),
                    },
                }
            )
        )
        self._set_mtime((metadata,), 3_000_000_000)

    def write_complete_object_depth(self) -> None:
        object_dir = self.condition / "object_render"
        object_dir.mkdir(parents=True, exist_ok=True)
        mask = object_dir / "object_mask.npy"
        depth = object_dir / "object_depth.npy"
        metadata = object_dir / "object_render_metadata.json"
        np.save(mask, np.zeros((self.frame_count, self.height, self.width), dtype=bool))
        np.save(
            depth,
            np.full(
                (self.frame_count, self.height, self.width), np.inf, dtype=np.float32
            ),
        )
        self._set_mtime((mask, depth), 4_000_000_000)
        metadata.write_text(
            json.dumps(
                {
                    "schema_version": "v2d.inpainting.taco-object-render/v1",
                    "state": "complete",
                    "run_id": "test-object-run",
                    "container_image": "robotic-grounding:photo-render-v6",
                    "container_image_id": IMAGE_ID,
                    "geometry": self.geometry,
                    "host_output_dir": str(object_dir.resolve()),
                    "artifacts": {
                        "mask": str(mask.resolve()),
                        "depth": str(depth.resolve()),
                    },
                    "artifact_bytes": {
                        "mask": mask.stat().st_size,
                        "depth": depth.stat().st_size,
                    },
                    "artifact_sha256": {
                        "mask": self._sha256(mask),
                        "depth": self._sha256(depth),
                    },
                    "provenance": {
                        "schema_version": OBJECT_RENDER_PROVENANCE_SCHEMA,
                        "hash_algorithm": "sha256",
                        "inputs": {
                            "source_parquet": {
                                "container_path": "/inputs/motion.parquet",
                                "host_path": str(self.motion_parquet.resolve()),
                                "bytes": self.motion_parquet.stat().st_size,
                                "sha256": self._sha256(self.motion_parquet),
                            },
                            "source_video": {
                                "container_path": "/inputs/source.mp4",
                                "host_path": str(self.source_video.resolve()),
                                "bytes": self.source_video.stat().st_size,
                                "sha256": self._sha256(self.source_video),
                            },
                            "intrinsic": {
                                "container_path": "/inputs/intrinsic.txt",
                                "host_path": str(self.intrinsic.resolve()),
                                "bytes": self.intrinsic.stat().st_size,
                                "sha256": self._sha256(self.intrinsic),
                            },
                            "world_to_camera": {
                                "container_path": "/inputs/world_to_camera.npy",
                                "host_path": str(self.world_to_camera.resolve()),
                                "bytes": self.world_to_camera.stat().st_size,
                                "sha256": self._sha256(self.world_to_camera),
                            },
                        },
                        "implementation_sources": object_render_source_records(
                            self.repository_root
                        ),
                    },
                }
            )
        )
        self._set_mtime((metadata,), 5_000_000_000)

    def write_complete_composite(self) -> None:
        output = self.condition / "final_overlay.mp4"
        metadata = self.condition / "final_overlay.json"
        fingerprint_paths = {
            "base_video": self.inpaint_video,
            "robot_metadata": self.robot_dir / "render_metadata.json",
            "robot_rgb": self.robot_dir / "robot_rgb.mp4",
            "robot_mask": self.robot_dir / "robot_mask.npy",
            "robot_depth": self.robot_dir / "robot_depth.npy",
            "object_mask": self.condition / "object_render" / "object_mask.npy",
            "object_depth": self.condition / "object_render" / "object_depth.npy",
            "object_metadata": (
                self.condition / "object_render" / "object_render_metadata.json"
            ),
        }
        output.write_bytes(b"composite-video")
        self._set_mtime((output,), 6_000_000_000)
        metadata.write_text(
            json.dumps(
                {
                    "schema_version": "v2d.inpainting.composite/v1",
                    "state": "complete",
                    "base_video": str(self.inpaint_video.resolve()),
                    "robot_video": str((self.robot_dir / "robot_rgb.mp4").resolve()),
                    "robot_mask": str((self.robot_dir / "robot_mask.npy").resolve()),
                    "robot_metadata": str(
                        (self.robot_dir / "render_metadata.json").resolve()
                    ),
                    "object_mask": str(
                        (self.condition / "object_render" / "object_mask.npy").resolve()
                    ),
                    "object_depth": str(
                        (
                            self.condition / "object_render" / "object_depth.npy"
                        ).resolve()
                    ),
                    "object_metadata": str(
                        (
                            self.condition
                            / "object_render"
                            / "object_render_metadata.json"
                        ).resolve()
                    ),
                    "output_video": str(output.resolve()),
                    "geometry": self.geometry,
                    "frames_written": self.frame_count,
                    "compositing": "taco_object_depth",
                    "depth_guard_m": 0.003,
                    "input_fingerprints": {
                        name: self._fingerprint(path)
                        for name, path in fingerprint_paths.items()
                    },
                    "output_fingerprint": self._fingerprint(output),
                }
            )
        )
        self._set_mtime((metadata,), 7_000_000_000)

    def write_complete_grid(self, options: PlanOptions) -> None:
        output = self.condition / "final_comparison_grid.mp4"
        metadata = self.condition / "final_comparison_grid.json"
        output.write_bytes(b"grid-video")
        self._set_mtime((output,), 8_000_000_000)
        metadata.write_text(
            json.dumps(
                {
                    "schema_version": GRID_SCHEMA,
                    "state": "complete",
                    "specification": {
                        "videos": [
                            str(self.source_video.resolve()),
                            str(self.inpaint_video.resolve()),
                            str((self.condition / "final_overlay.mp4").resolve()),
                        ],
                        "labels": ["Source", "E2FGVI", "GT Vega + Sharpa"],
                        "tile_width": options.grid_tile_width,
                        "columns": options.grid_columns,
                        "max_frames": options.grid_max_frames,
                    },
                    "geometry": {
                        "frame_count": self.frame_count,
                        "width": self.width * 3,
                        "height": self.height,
                        "fps": self.fps,
                    },
                }
            )
        )
        self._set_mtime((metadata,), 9_000_000_000)

    @property
    def geometry(self) -> dict:
        return {
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }

    @staticmethod
    def _set_mtime(paths: tuple[Path, ...], timestamp_ns: int) -> None:
        for path in paths:
            os.utime(path, ns=(timestamp_ns, timestamp_ns))

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @classmethod
    def _fingerprint(cls, path: Path) -> dict[str, int | str]:
        return {"bytes": path.stat().st_size, "sha256": cls._sha256(path)}


class GroundTruthBatchTests(unittest.TestCase):
    def test_default_is_read_only_plan_and_withholds_full_renderer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = BatchFixture(Path(temporary))
            before = {
                path.relative_to(fixture.root) for path in fixture.root.rglob("*")
            }
            plan = build_plan(fixture.options())
            after = {path.relative_to(fixture.root) for path in fixture.root.rglob("*")}
            self.assertEqual(after, before)
            self.assertEqual(plan["mode"], "plan")
            self.assertEqual(
                plan["selected_stages"],
                ["render", "object_depth", "composite", "grid"],
            )
            actions = {action["stage"]: action for action in plan["actions"]}
            self.assertEqual(actions["render"]["status"], "pending")
            self.assertIsNone(actions["render"]["command"])
            self.assertIn("withheld", actions["render"]["reason"])
            self.assertEqual(actions["object_depth"]["status"], "pending")
            self.assertIsNone(actions["object_depth"]["command"])
            self.assertIn("withheld", actions["object_depth"]["reason"])
            self.assertEqual(actions["composite"]["status"], "pending")
            self.assertEqual(actions["grid"]["status"], "pending")
            for action in plan["actions"]:
                for path in (*action["inputs"], *action["outputs"]):
                    self.assertTrue(Path(path).is_absolute())

    def test_gpu_plan_has_exact_full_renderer_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = BatchFixture(Path(temporary))
            plan = build_plan(
                fixture.options(
                    stages=("render",),
                    sequence_ids=(fixture.sequence_id,),
                    gpu="1",
                )
            )
            action = plan["actions"][0]
            command = action["command"]
            self.assertIn("--execute", command)
            self.assertEqual(command[command.index("--gpu") + 1], "1")
            self.assertNotIn("--overwrite", command)
            self.assertEqual(Path(command[0]), Path(sys.executable).resolve())

    @mock.patch(
        "inpainting.run_ground_truth_batch.resolve_local_image_id",
        return_value=IMAGE_ID,
    )
    def test_composite_consumes_renderer_bundle_and_forwards_overwrite(
        self, _resolve_image: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = BatchFixture(Path(temporary))
            fixture.write_complete_render()
            fixture.write_complete_object_depth()
            plan = build_plan(fixture.options(stages=("composite",)))
            command = plan["actions"][0]["command"]
            self.assertEqual(
                Path(command[command.index("--robot-metadata") + 1]),
                (fixture.robot_dir / "render_metadata.json").resolve(),
            )
            self.assertEqual(
                Path(command[command.index("--object-metadata") + 1]),
                (
                    fixture.condition / "object_render" / "object_render_metadata.json"
                ).resolve(),
            )
            self.assertNotIn("--overwrite", command)

            overwrite = build_plan(
                fixture.options(stages=("composite",), overwrite=True)
            )
            self.assertIn("--overwrite", overwrite["actions"][0]["command"])

    def test_hard_composite_requires_explicit_escape_hatch_and_is_labelled(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = BatchFixture(Path(temporary))
            fixture.write_complete_render()
            blocked = build_plan(fixture.options(stages=("composite",)))
            self.assertEqual(blocked["actions"][0]["status"], "blocked")
            self.assertIn("object-depth render", blocked["actions"][0]["reason"])

            fallback = build_plan(
                fixture.options(
                    stages=("composite", "grid"),
                    allow_hard_composite=True,
                )
            )
            composite, grid = fallback["actions"]
            self.assertEqual(composite["status"], "pending")
            self.assertIn("HARD-MASK FALLBACK", composite["reason"])
            self.assertNotIn("--object-metadata", composite["command"])
            labels = [
                grid["command"][index + 1]
                for index, value in enumerate(grid["command"][:-1])
                if value == "--label"
            ]
            self.assertEqual(labels[-1], "GT Vega + Sharpa (HARD-MASK FALLBACK)")

    def test_execute_refuses_pending_renderer_without_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = BatchFixture(Path(temporary))
            options = fixture.options(stages=("render",))
            plan = build_plan(options)
            with self.assertRaisesRegex(BatchPlanError, "explicit --gpu"):
                execute_plan(plan, options, run_command=lambda *args, **kwargs: None)

    def test_object_depth_plan_has_sequence_id_and_requires_gpu_to_execute(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = BatchFixture(Path(temporary))
            options = fixture.options(stages=("object_depth",), gpu="2")
            plan = build_plan(options)
            command = plan["actions"][0]["command"]
            self.assertEqual(
                command[command.index("--sequence-id") + 1], fixture.sequence_id
            )
            self.assertEqual(command[command.index("--gpu") + 1], "2")
            self.assertEqual(
                Path(command[command.index("--parquet") + 1]),
                fixture.motion_parquet.resolve(),
            )

            no_gpu_options = fixture.options(stages=("object_depth",))
            no_gpu = build_plan(no_gpu_options)
            with self.assertRaisesRegex(BatchPlanError, "explicit --gpu"):
                execute_plan(
                    no_gpu,
                    no_gpu_options,
                    run_command=lambda *args, **kwargs: None,
                )

    def test_incomplete_output_blocks_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = BatchFixture(Path(temporary))
            fixture.robot_dir.mkdir(parents=True)
            (fixture.robot_dir / "robot_rgb.mp4").write_bytes(b"partial")
            plan = build_plan(fixture.options(stages=("render",), gpu="0"))
            self.assertEqual(plan["actions"][0]["status"], "blocked")
            overwrite = build_plan(
                fixture.options(stages=("render",), gpu="0", overwrite=True)
            )
            action = overwrite["actions"][0]
            self.assertEqual(action["status"], "pending_overwrite")
            self.assertIn("--overwrite", action["command"])

    @mock.patch(
        "inpainting.run_ground_truth_batch.resolve_local_image_id",
        return_value=IMAGE_ID,
    )
    def test_complete_artifacts_are_resumed_without_subprocesses(
        self, _resolve_image: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = BatchFixture(Path(temporary))
            options = fixture.options()
            fixture.write_complete_render()
            fixture.write_complete_object_depth()
            fixture.write_complete_composite()
            fixture.write_complete_grid(options)
            plan = build_plan(options)
            self.assertTrue(
                all(
                    action["status"] == "skipped_complete" for action in plan["actions"]
                )
            )

            def forbidden(*args, **kwargs):
                raise AssertionError(
                    "resume-only execution must not launch subprocesses"
                )

            result = execute_plan(plan, options, run_command=forbidden)
            self.assertEqual(result["state"], "complete")
            self.assertEqual(result["executed"], [])
            self.assertEqual(len(result["resumed"]), 4)

    @mock.patch(
        "inpainting.run_ground_truth_batch.resolve_local_image_id",
        return_value=IMAGE_ID,
    )
    def test_inpaint_resume_rejects_same_path_output_mutation(
        self, _resolve_image: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = BatchFixture(Path(temporary))
            fixture.write_complete_render()
            fixture.write_complete_object_depth()
            fixture.inpaint_video.write_bytes(b"X" * len(b"inpaint-video"))
            fixture._set_mtime((fixture.inpaint_video,), 1_000_000_000)

            plan = build_plan(fixture.options(stages=("composite",)))

            action = plan["actions"][0]
            self.assertEqual(action["status"], "blocked")
            self.assertIn("output fingerprint", action["reason"])

    @mock.patch(
        "inpainting.run_ground_truth_batch.resolve_local_image_id",
        return_value=IMAGE_ID,
    )
    def test_inpaint_resume_rejects_changed_shared_arm_mask(
        self, _resolve_image: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = BatchFixture(Path(temporary))
            fixture.write_complete_render()
            fixture.write_complete_object_depth()
            masks = np.load(fixture.inpaint_masks, allow_pickle=False)
            masks[0, 0, 0] = True
            np.save(fixture.inpaint_masks, masks)
            fixture._set_mtime((fixture.inpaint_masks,), 1_000_000_000)

            plan = build_plan(fixture.options(stages=("composite",)))

            action = plan["actions"][0]
            self.assertEqual(action["status"], "blocked")
            self.assertIn("input fingerprint", action["reason"])

    @mock.patch(
        "inpainting.run_ground_truth_batch.resolve_local_image_id",
        return_value=IMAGE_ID,
    )
    def test_inpaint_resume_rejects_changed_source_video(
        self, _resolve_image: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = BatchFixture(Path(temporary))
            fixture.write_complete_render()
            fixture.write_complete_object_depth()
            fixture.source_video.write_bytes(b"changed-source")
            fixture._set_mtime((fixture.source_video,), 1_000_000_000)

            plan = build_plan(fixture.options(stages=("composite",)))

            action = plan["actions"][0]
            self.assertEqual(action["status"], "blocked")
            self.assertIn("shared E2FGVI input fingerprint", action["reason"])

    @mock.patch(
        "inpainting.run_ground_truth_batch.resolve_local_image_id",
        return_value=IMAGE_ID,
    )
    def test_composite_resume_rejects_same_path_output_mutation(
        self, _resolve_image: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = BatchFixture(Path(temporary))
            fixture.write_complete_render()
            fixture.write_complete_object_depth()
            fixture.write_complete_composite()
            output = fixture.condition / "final_overlay.mp4"
            output.write_bytes(b"X" * len(b"composite-video"))
            fixture._set_mtime((output,), 6_000_000_000)

            plan = build_plan(fixture.options(stages=("composite",)))

            action = plan["actions"][0]
            self.assertEqual(action["status"], "blocked")
            self.assertIn("composite fingerprint validation failed", action["reason"])

    @mock.patch(
        "inpainting.run_ground_truth_batch.resolve_local_image_id",
        return_value=IMAGE_ID,
    )
    def test_composite_resume_rejects_new_valid_input_generation(
        self, _resolve_image: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = BatchFixture(Path(temporary))
            fixture.write_complete_render()
            fixture.write_complete_object_depth()
            fixture.write_complete_composite()
            rgb = fixture.robot_dir / "robot_rgb.mp4"
            render_metadata_path = fixture.robot_dir / "render_metadata.json"
            rgb.write_bytes(b"R" * len(b"robot-video"))
            render_metadata = json.loads(render_metadata_path.read_text())
            render_metadata["artifact_sha256"]["rgb"] = fixture._sha256(rgb)
            render_metadata_path.write_text(json.dumps(render_metadata))
            fixture._set_mtime((rgb,), 2_000_000_000)
            fixture._set_mtime((render_metadata_path,), 3_000_000_000)

            plan = build_plan(fixture.options(stages=("composite",)))

            action = plan["actions"][0]
            self.assertEqual(action["status"], "blocked")
            self.assertIn("composite fingerprint validation failed", action["reason"])

    @mock.patch(
        "inpainting.run_ground_truth_batch.resolve_local_image_id",
        return_value=IMAGE_ID,
    )
    def test_object_depth_resume_rejects_changed_source_input(
        self, _resolve_image: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = BatchFixture(Path(temporary))
            fixture.write_complete_object_depth()
            fixture.source_video.write_bytes(b"changed-source-video")
            plan = build_plan(fixture.options(stages=("object_depth",), gpu="0"))
            action = plan["actions"][0]
            self.assertEqual(action["status"], "blocked")
            self.assertIn("provenance", action["reason"])

    @mock.patch(
        "inpainting.run_ground_truth_batch.resolve_local_image_id",
        return_value=IMAGE_ID,
    )
    def test_object_depth_resume_rejects_changed_implementation_source(
        self, _resolve_image: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = BatchFixture(Path(temporary))
            fixture.write_complete_object_depth()
            source = fixture.repository_root / "inpainting/taco_object_depth.py"
            source.write_text(source.read_text() + "# changed\n")
            plan = build_plan(fixture.options(stages=("object_depth",), gpu="0"))
            action = plan["actions"][0]
            self.assertEqual(action["status"], "blocked")
            self.assertIn("implementation source", action["reason"])

    @mock.patch(
        "inpainting.run_ground_truth_batch.resolve_local_image_id",
        return_value="sha256:" + "b" * 64,
    )
    def test_object_depth_resume_rejects_retargeted_image(
        self, _resolve_image: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = BatchFixture(Path(temporary))
            fixture.write_complete_object_depth()
            plan = build_plan(fixture.options(stages=("object_depth",), gpu="0"))
            action = plan["actions"][0]
            self.assertEqual(action["status"], "blocked")
            self.assertIn("immutable container image ID", action["reason"])

    def test_composite_only_requires_an_existing_render(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = BatchFixture(Path(temporary))
            plan = build_plan(fixture.options(stages=("composite",)))
            action = plan["actions"][0]
            self.assertEqual(action["status"], "blocked")
            self.assertIn("neither complete nor scheduled", action["reason"])

    def test_camera_unavailable_is_a_planned_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = BatchFixture(Path(temporary))
            manifest = json.loads(fixture.manifest.read_text())
            manifest["sequences"][0]["camera"]["available"] = False
            fixture.manifest.write_text(json.dumps(manifest))
            plan = build_plan(fixture.options(stages=("render",), gpu="0"))
            action = plan["actions"][0]
            self.assertEqual(action["status"], "blocked")
            self.assertIn("not marked available", action["reason"])

    def test_unknown_sequence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = BatchFixture(Path(temporary))
            with self.assertRaisesRegex(BatchPlanError, "unknown sequence"):
                build_plan(fixture.options(sequence_ids=("not_present",)))

    def test_manifest_sequence_cannot_escape_run_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = BatchFixture(Path(temporary))
            manifest = json.loads(fixture.manifest.read_text())
            manifest["sequences"][0]["sequence_id"] = "../escape"
            fixture.manifest.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(BatchPlanError, "safe path segment"):
                build_plan(fixture.options())

    def test_execute_rechecks_no_overwrite_after_planning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = BatchFixture(Path(temporary))
            options = fixture.options(stages=("render",), gpu="0")
            plan = build_plan(options)
            fixture.robot_dir.mkdir(parents=True)
            (fixture.robot_dir / "robot_rgb.mp4").write_bytes(b"appeared-after-plan")
            with self.assertRaisesRegex(BatchPlanError, "appeared after planning"):
                execute_plan(plan, options, run_command=lambda *args, **kwargs: None)


if __name__ == "__main__":
    unittest.main()
