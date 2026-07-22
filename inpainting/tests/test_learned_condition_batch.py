from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import inpainting.run_learned_condition_batch as learned_batch
from inpainting.run_learned_condition_batch import (
    BatchPlanError,
    GRID_SCHEMA,
    PlanOptions,
    build_plan,
    execute_plan,
)
from inpainting.tests.test_ground_truth_batch import BatchFixture


class LearnedBatchFixture:
    condition_name = "v2d"

    def __init__(self, root: Path, *, with_object_depth: bool = True):
        self.base = BatchFixture(root)
        manifest = json.loads(self.base.manifest.read_text())
        manifest["sequences"][0]["conditions"].update(
            {
                "v2d": {
                    "tracker": "v2d",
                    "state": "source_inputs_resolved",
                    "blockers": [],
                },
                "phantom": {
                    "tracker": "phantom",
                    "state": "source_inputs_resolved",
                    "blockers": [],
                },
            }
        )
        self.base.manifest.write_text(json.dumps(manifest))
        self.tracking.mkdir(parents=True)
        self.trajectory.write_bytes(self.base.trajectory.read_bytes())
        if with_object_depth:
            self.base.write_complete_object_depth()

    @property
    def root(self) -> Path:
        return self.base.root

    @property
    def sequence_id(self) -> str:
        return self.base.sequence_id

    @property
    def condition(self) -> Path:
        return self.base.run_root / self.sequence_id / self.condition_name

    @property
    def tracking(self) -> Path:
        return self.condition / "tracking"

    @property
    def trajectory(self) -> Path:
        return self.tracking / "robot_trajectory.npz"

    @property
    def robot_dir(self) -> Path:
        return self.condition / "robot_render"

    @property
    def object_dir(self) -> Path:
        return self.base.condition / "object_render"

    def options(self, **overrides) -> PlanOptions:
        values = {
            "manifest_path": self.base.manifest,
            "repository_root": self.base.repository_root,
            "python_executable": Path(sys.executable),
            "conditions": (self.condition_name,),
        }
        values.update(overrides)
        return PlanOptions(**values)

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _set_mtime(paths: tuple[Path, ...], timestamp_ns: int) -> None:
        for path in paths:
            os.utime(path, ns=(timestamp_ns, timestamp_ns))

    def write_complete_render(
        self,
        *,
        max_ik_residual_m: float = 0.01,
        max_joint_step_rad: float = 0.4,
    ) -> None:
        self.robot_dir.mkdir(parents=True, exist_ok=True)
        rgb = self.robot_dir / "robot_rgb.mp4"
        mask = self.robot_dir / "robot_mask.npy"
        depth = self.robot_dir / "robot_depth.npy"
        metadata = self.robot_dir / "render_metadata.json"
        rgb.write_bytes(b"learned-robot-video")
        np.save(
            mask,
            np.zeros(
                (self.base.frame_count, self.base.height, self.base.width), dtype=bool
            ),
        )
        np.save(
            depth,
            np.full(
                (self.base.frame_count, self.base.height, self.base.width),
                np.inf,
                dtype=np.float32,
            ),
        )
        self._set_mtime((rgb, mask, depth), 2_000_000_000)
        metadata.write_text(
            json.dumps(
                {
                    "schema_version": "v2d.inpainting.robot-render/v1",
                    "state": "complete",
                    "run_id": "learned-render",
                    "geometry": self.base.geometry,
                    "host_output_dir": str(self.robot_dir.resolve()),
                    "artifacts": {
                        "rgb": str(rgb.resolve()),
                        "mask": str(mask.resolve()),
                        "depth": str(depth.resolve()),
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
                    "kinematics_policy": {
                        "max_position_residual_m": max_ik_residual_m,
                        "max_joint_step_rad": max_joint_step_rad,
                    },
                    "render_statistics": {
                        "video_verification": {
                            "decoded_frame_count": self.base.frame_count,
                            "width": self.base.width,
                            "height": self.base.height,
                            "fps": self.base.fps,
                        }
                    },
                }
            )
        )
        self._set_mtime((metadata,), 3_000_000_000)

    def write_complete_composite(self) -> None:
        output = self.condition / "final_overlay.mp4"
        metadata = self.condition / "final_overlay.json"
        fingerprint_paths = {
            "base_video": self.base.inpaint_video,
            "robot_metadata": self.robot_dir / "render_metadata.json",
            "robot_rgb": self.robot_dir / "robot_rgb.mp4",
            "robot_mask": self.robot_dir / "robot_mask.npy",
            "robot_depth": self.robot_dir / "robot_depth.npy",
            "object_mask": self.object_dir / "object_mask.npy",
            "object_depth": self.object_dir / "object_depth.npy",
            "object_metadata": self.object_dir / "object_render_metadata.json",
        }
        output.write_bytes(b"learned-composite-video")
        self._set_mtime((output,), 6_000_000_000)
        metadata.write_text(
            json.dumps(
                {
                    "schema_version": "v2d.inpainting.composite/v1",
                    "state": "complete",
                    "base_video": str(self.base.inpaint_video.resolve()),
                    "robot_video": str((self.robot_dir / "robot_rgb.mp4").resolve()),
                    "robot_mask": str((self.robot_dir / "robot_mask.npy").resolve()),
                    "robot_metadata": str(
                        (self.robot_dir / "render_metadata.json").resolve()
                    ),
                    "object_mask": str((self.object_dir / "object_mask.npy").resolve()),
                    "object_depth": str(
                        (self.object_dir / "object_depth.npy").resolve()
                    ),
                    "object_metadata": str(
                        (self.object_dir / "object_render_metadata.json").resolve()
                    ),
                    "output_video": str(output.resolve()),
                    "geometry": self.base.geometry,
                    "frames_written": self.base.frame_count,
                    "compositing": "taco_object_depth",
                    "depth_guard_m": 0.003,
                    "input_fingerprints": {
                        name: self.base._fingerprint(path)
                        for name, path in fingerprint_paths.items()
                    },
                    "output_fingerprint": self.base._fingerprint(output),
                }
            )
        )
        self._set_mtime((metadata,), 7_000_000_000)

    def write_complete_grid(self, options: PlanOptions) -> None:
        output = self.condition / "final_comparison_grid.mp4"
        metadata = self.condition / "final_comparison_grid.json"
        output.write_bytes(b"learned-grid-video")
        self._set_mtime((output,), 8_000_000_000)
        metadata.write_text(
            json.dumps(
                {
                    "schema_version": GRID_SCHEMA,
                    "state": "complete",
                    "condition": self.condition_name,
                    "specification": {
                        "videos": [
                            str(self.base.source_video.resolve()),
                            str(self.base.inpaint_video.resolve()),
                            str((self.condition / "final_overlay.mp4").resolve()),
                        ],
                        "labels": [
                            "Source",
                            "E2FGVI",
                            "Video2Data Vega + Sharpa",
                        ],
                        "tile_width": options.grid_tile_width,
                        "columns": options.grid_columns,
                        "max_frames": options.grid_max_frames,
                    },
                    "geometry": {
                        "frame_count": self.base.frame_count,
                        "width": self.base.width * 3,
                        "height": self.base.height,
                        "fps": self.base.fps,
                    },
                }
            )
        )
        self._set_mtime((metadata,), 9_000_000_000)


class LearnedConditionBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        # Full current-generation provenance validation is covered by the
        # renderer enrichment tests and the focused tests below. These tiny
        # orchestration fixtures intentionally contain no real robot assets or
        # local Docker image.
        patcher = patch(
            "inpainting.run_learned_condition_batch._verify_current_render_generation"
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_default_is_read_only_and_withholds_only_renderer_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LearnedBatchFixture(Path(temporary))
            before = {path.relative_to(fixture.root) for path in fixture.root.rglob("*")}
            plan = build_plan(fixture.options())
            after = {path.relative_to(fixture.root) for path in fixture.root.rglob("*")}
            self.assertEqual(after, before)
            self.assertEqual(plan["selected_stages"], ["render", "composite", "grid"])
            render, composite, grid = plan["actions"]
            self.assertEqual(render["status"], "pending")
            self.assertIsNone(render["command"])
            self.assertIn("explicit --gpu", render["reason"])
            self.assertEqual(composite["status"], "pending")
            self.assertIsNotNone(composite["command"])
            self.assertNotIn("--gpu", composite["command"])
            self.assertEqual(grid["status"], "pending")
            self.assertEqual(plan["object_depth_inputs"][fixture.sequence_id]["state"], "complete")

    def test_render_command_uses_exact_gpu_and_learned_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LearnedBatchFixture(Path(temporary))
            gpu_uuid = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
            plan = build_plan(
                fixture.options(stages=("render",), gpu=gpu_uuid)
            )
            command = plan["actions"][0]["command"]
            self.assertEqual(command[command.index("--gpu") + 1], gpu_uuid)
            self.assertEqual(
                command[command.index("--max-ik-residual-m") + 1], "0.01"
            )
            self.assertEqual(
                command[command.index("--max-joint-step-rad") + 1], "0.4"
            )
            self.assertEqual(
                Path(command[command.index("--trajectory") + 1]),
                fixture.trajectory.resolve(),
            )
            self.assertNotIn("all", command)

    def test_changed_kinematics_policy_requires_explicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LearnedBatchFixture(Path(temporary))
            fixture.write_complete_render()
            selected = fixture.options(
                stages=("render",),
                gpu="0",
                max_ik_residual_m=0.012,
                max_joint_step_rad=0.35,
            )
            blocked = build_plan(selected)
            action = blocked["actions"][0]
            self.assertEqual(action["status"], "blocked")
            self.assertIn("kinematics policy differs", action["reason"])

            overwrite = build_plan(
                fixture.options(
                    stages=("render",),
                    gpu="0",
                    max_ik_residual_m=0.012,
                    max_joint_step_rad=0.35,
                    overwrite=True,
                )
            )
            action = overwrite["actions"][0]
            self.assertEqual(action["status"], "pending_overwrite")
            command = action["command"]
            self.assertEqual(
                command[command.index("--max-ik-residual-m") + 1], "0.012"
            )
            self.assertEqual(
                command[command.index("--max-joint-step-rad") + 1], "0.35"
            )

    def test_composite_reuses_validated_ground_truth_object_bundle_without_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LearnedBatchFixture(Path(temporary))
            fixture.write_complete_render()
            plan = build_plan(fixture.options(stages=("composite",)))
            action = plan["actions"][0]
            self.assertEqual(action["status"], "pending")
            command = action["command"]
            self.assertEqual(
                Path(command[command.index("--object-metadata") + 1]),
                (fixture.object_dir / "object_render_metadata.json").resolve(),
            )
            self.assertNotIn("--gpu", command)

            def fail_after_launch(*args, **kwargs):
                return SimpleNamespace(returncode=19)

            with self.assertRaisesRegex(RuntimeError, "failed with exit 19"):
                execute_plan(plan, fixture.options(stages=("composite",)), run_command=fail_after_launch)

    def test_missing_object_depth_blocks_composite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LearnedBatchFixture(Path(temporary), with_object_depth=False)
            fixture.write_complete_render()
            plan = build_plan(fixture.options(stages=("composite",)))
            action = plan["actions"][0]
            self.assertEqual(action["status"], "blocked")
            self.assertIn("ground-truth object-depth", action["reason"])
            self.assertIsNone(action["command"])

    def test_final_and_hidden_partial_outputs_block_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LearnedBatchFixture(Path(temporary))
            fixture.write_complete_render()
            fixture.write_complete_composite()
            hidden_partial = fixture.condition / ".final_overlay.crash.partial.mp4"
            hidden_partial.write_bytes(b"partial")
            blocked = build_plan(fixture.options(stages=("composite",)))
            self.assertEqual(blocked["actions"][0]["status"], "blocked")
            self.assertIn("stale partial", blocked["actions"][0]["reason"])
            overwrite = build_plan(
                fixture.options(stages=("composite",), overwrite=True)
            )
            self.assertEqual(overwrite["actions"][0]["status"], "pending_overwrite")
            self.assertIn("--overwrite", overwrite["actions"][0]["command"])

    def test_pending_render_refuses_execute_without_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LearnedBatchFixture(Path(temporary))
            options = fixture.options(stages=("render",))
            plan = build_plan(options)
            with self.assertRaisesRegex(BatchPlanError, "explicit --gpu"):
                execute_plan(plan, options, run_command=lambda *args, **kwargs: None)

    def test_planning_blocks_a_trajectory_with_source_invalid_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LearnedBatchFixture(Path(temporary))
            with np.load(fixture.trajectory, allow_pickle=False) as archive:
                arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
            arrays["right_valid"][0] = False
            for suffix in ("wrist_position", "wrist_wxyz", "finger_joints"):
                arrays[f"right_{suffix}"][0] = np.nan
            np.savez(fixture.trajectory, **arrays)

            plan = build_plan(
                fixture.options(stages=("render",), gpu="0")
            )
            action = plan["actions"][0]
            self.assertEqual(action["status"], "blocked")
            self.assertIn("right_valid marks 1/2 frames invalid", action["reason"])
            self.assertIsNone(action["command"])

    def test_complete_outputs_resume_without_subprocesses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LearnedBatchFixture(Path(temporary))
            options = fixture.options()
            fixture.write_complete_render()
            fixture.write_complete_composite()
            fixture.write_complete_grid(options)
            plan = build_plan(options)
            self.assertTrue(
                all(action["status"] == "skipped_complete" for action in plan["actions"])
            )

            def forbidden(*args, **kwargs):
                raise AssertionError("resume must not launch a subprocess")

            result = execute_plan(plan, options, run_command=forbidden)
            self.assertEqual(result["executed"], [])
            self.assertEqual(len(result["resumed"]), 3)

    def test_unsupported_condition_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LearnedBatchFixture(Path(temporary))
            with self.assertRaisesRegex(BatchPlanError, "conditions must"):
                build_plan(fixture.options(conditions=("ground_truth",)))

    def test_kinematics_policy_must_be_finite_and_positive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LearnedBatchFixture(Path(temporary))
            with self.assertRaisesRegex(BatchPlanError, "max IK residual"):
                build_plan(fixture.options(max_ik_residual_m=0.0))
            with self.assertRaisesRegex(BatchPlanError, "max joint step"):
                build_plan(fixture.options(max_joint_step_rad=float("nan")))

    def test_gpu_selector_rejects_lists_and_all(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LearnedBatchFixture(Path(temporary))
            for selector in ("all", "0,1", "GPU-partial", " 0"):
                with self.subTest(selector=selector):
                    with self.assertRaisesRegex(BatchPlanError, "exactly one"):
                        build_plan(fixture.options(gpu=selector))


class LearnedRenderGenerationTests(unittest.TestCase):
    image_id = "sha256:" + "a" * 64

    @staticmethod
    def _recorded_metadata(
        *, image_id: str, arm_ik_hash: str, arm_mount_hash: str
    ) -> dict:
        return {
            "container_image_id": image_id,
            "provenance": {
                "schema_version": "v2d.inpainting.robot-render-provenance/v1",
                "inputs": {
                    key: {"path": key, "bytes": 1, "sha256": "b" * 64}
                    for key in ("trajectory", "intrinsic", "world_to_camera")
                },
                "renderer_source_files": [
                    {"path": "renderer.py", "bytes": 1, "sha256": "c" * 64}
                ],
            },
            "artifact_sha256": {
                "rgb": "d" * 64,
                "mask": "e" * 64,
                "depth": "f" * 64,
            },
            "assets": {
                part: {
                    "urdf_file": {"path": f"{part}.urdf", "sha256": "1" * 64},
                    "referenced_asset_files": [
                        {"path": f"{part}.stl", "sha256": "2" * 64}
                    ],
                }
                for part in ("arms", "left_hand", "right_hand")
            },
            "kinematics": {
                "external_sources": {
                    "arm_ik_sha256": arm_ik_hash,
                    "arm_mount_opt_sha256": arm_mount_hash,
                }
            },
        }

    def test_strict_generation_check_is_verify_only_and_binds_external_ik(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene = root / "scene"
            scene.mkdir()
            arm_ik = scene / "arm_ik.py"
            arm_mount = scene / "arm_mount_opt.py"
            arm_ik.write_bytes(b"ik")
            arm_mount.write_bytes(b"mount")
            metadata_path = root / "render_metadata.json"
            metadata = self._recorded_metadata(
                image_id=self.image_id,
                arm_ik_hash=hashlib.sha256(b"ik").hexdigest(),
                arm_mount_hash=hashlib.sha256(b"mount").hexdigest(),
            )
            metadata_path.write_text(json.dumps(metadata))
            paths = SimpleNamespace(
                robot_metadata=metadata_path,
                trajectory=root / "trajectory.npz",
                intrinsic=root / "intrinsic.txt",
                world_to_camera=root / "world_to_camera.npy",
            )
            options = PlanOptions(
                manifest_path=root / "manifest.json", renderer_image="render:test"
            )
            with (
                patch.object(
                    learned_batch,
                    "resolve_local_image_id",
                    return_value=self.image_id,
                ),
                patch.object(
                    learned_batch,
                    "enrich_render_metadata",
                    return_value=metadata,
                ) as enrich,
            ):
                learned_batch._verify_current_render_generation(
                    paths=paths,
                    options=options,
                    asset_root=root / "assets",
                    scene_utils_root=scene,
                    repository_root=root / "repo",
                )
            self.assertFalse(enrich.call_args.kwargs["write"])
            self.assertEqual(enrich.call_args.kwargs["image_id"], self.image_id)

    def test_strict_generation_check_rejects_unbound_legacy_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "render_metadata.json"
            metadata.write_text(json.dumps({"state": "complete"}))
            paths = SimpleNamespace(
                robot_metadata=metadata,
                trajectory=root / "trajectory.npz",
                intrinsic=root / "intrinsic.txt",
                world_to_camera=root / "world_to_camera.npy",
            )
            options = PlanOptions(manifest_path=root / "manifest.json")
            with patch.object(
                learned_batch,
                "resolve_local_image_id",
                return_value=self.image_id,
            ):
                with self.assertRaisesRegex(
                    BatchPlanError, "immutable image ID"
                ):
                    learned_batch._verify_current_render_generation(
                        paths=paths,
                        options=options,
                        asset_root=root / "assets",
                        scene_utils_root=root / "scene",
                        repository_root=root / "repo",
                    )


if __name__ == "__main__":
    unittest.main()
