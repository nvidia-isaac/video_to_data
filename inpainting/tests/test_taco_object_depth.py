from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from inpainting.contracts import ContractError, VideoGeometry
from inpainting.taco_object_depth import (
    TacoObjectInputs,
    TacoObjectRenderError,
    _metadata_base,
    _resolve_meshes,
    _validated_object_arrays,
    render_taco_object_depth,
)
from inpainting.taco_object_depth_container import (
    DEFAULT_IMAGE,
    ObjectDepthContainerConfig,
    build_docker_command,
    main as object_container_main,
)


SEQUENCE_ID = "taco_dust__brush__cup_20231005_253"
IMAGE_ID = "sha256:" + "a" * 64


def _valid_row(frame_count: int = 3) -> dict:
    quaternions = np.zeros((frame_count, 2, 4), dtype=np.float32)
    quaternions[..., 0] = 1.0
    return {
        "fps": 30.0,
        "object_body_names": ["tool", "target"],
        "object_mesh_paths": ["/stale/179_cm.obj", "/stale/107_cm.obj"],
        "object_body_position": np.zeros((frame_count, 2, 3), dtype=np.float32),
        "object_body_wxyz": quaternions,
    }


class TacoObjectInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.geometry = VideoGeometry(frame_count=3, width=16, height=12, fps=30.0)

    def test_valid_arrays_preserve_explicit_sequence(self) -> None:
        sequence_id, names, paths, positions, quaternions = _validated_object_arrays(
            _valid_row(), self.geometry, sequence_id=SEQUENCE_ID
        )
        self.assertEqual(sequence_id, SEQUENCE_ID)
        self.assertEqual(names, ("tool", "target"))
        self.assertEqual(len(paths), 2)
        self.assertEqual(positions.shape, (3, 2, 3))
        self.assertEqual(quaternions.shape, (3, 2, 4))

    def test_body_order_is_strict(self) -> None:
        row = _valid_row()
        row["object_body_names"] = ["target", "tool"]
        with self.assertRaisesRegex(ContractError, "body order"):
            _validated_object_arrays(row, self.geometry, sequence_id=SEQUENCE_ID)

    def test_non_unit_object_quaternion_is_rejected(self) -> None:
        row = _valid_row()
        row["object_body_wxyz"][1, 0] = 0.0
        with self.assertRaisesRegex(ContractError, "not unit"):
            _validated_object_arrays(row, self.geometry, sequence_id=SEQUENCE_ID)

    def test_meshes_reroot_by_basename_and_reject_lfs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tool = root / "179_cm.obj"
            target = root / "107_cm.obj"
            tool.write_text("v 0 0 0\n")
            target.write_text("v 0 0 0\n")
            resolved = _resolve_meshes(
                root, ["/container/stale/179_cm.obj", "/container/stale/107_cm.obj"]
            )
            self.assertEqual(resolved, (tool.resolve(), target.resolve()))
            target.write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
            )
            with self.assertRaisesRegex(ContractError, "LFS"):
                _resolve_meshes(root, [str(tool), str(target)])

    def test_precommit_overwrite_failure_preserves_complete_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "object_render"
            output.mkdir()
            metadata_path = output / "object_render_metadata.json"
            previous = {
                "schema_version": "v2d.inpainting.taco-object-render/v1",
                "state": "complete",
                "run_id": "previous-complete-run",
            }
            metadata_path.write_text(json.dumps(previous) + "\n")
            quaternions = np.zeros((3, 2, 4), dtype=np.float64)
            quaternions[..., 0] = 1.0
            inputs = TacoObjectInputs(
                sequence_id=SEQUENCE_ID,
                source_parquet=root / "motion.parquet",
                source_video=root / "source.mp4",
                intrinsic_path=root / "intrinsic.txt",
                world_to_camera_path=root / "world_to_camera.npy",
                mesh_root=root / "meshes",
                geometry=self.geometry,
                body_names=("tool", "target"),
                stored_mesh_paths=("179_cm.obj", "107_cm.obj"),
                mesh_paths=(root / "179_cm.obj", root / "107_cm.obj"),
                positions_world=np.zeros((3, 2, 3), dtype=np.float64),
                quaternions_wxyz=quaternions,
                intrinsic=np.eye(3, dtype=np.float64),
                world_to_camera=np.repeat(
                    np.eye(4, dtype=np.float64)[None, ...], 3, axis=0
                ),
            )
            with mock.patch.dict(sys.modules, {"pyrender": None}):
                with self.assertRaises(TacoObjectRenderError):
                    render_taco_object_depth(inputs, output, overwrite=True)
            self.assertEqual(json.loads(metadata_path.read_text()), previous)

    def test_metadata_fingerprints_immutable_image_and_host_source_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = {
                "source_parquet": root / "motion.parquet",
                "source_video": root / "source.mp4",
                "intrinsic": root / "intrinsic.txt",
                "world_to_camera": root / "world_to_camera.npy",
            }
            for name, path in sources.items():
                path.write_bytes(name.encode())
            quaternions = np.zeros((3, 2, 4), dtype=np.float64)
            quaternions[..., 0] = 1.0
            inputs = TacoObjectInputs(
                sequence_id=SEQUENCE_ID,
                source_parquet=sources["source_parquet"],
                source_video=sources["source_video"],
                intrinsic_path=sources["intrinsic"],
                world_to_camera_path=sources["world_to_camera"],
                mesh_root=root / "meshes",
                geometry=self.geometry,
                body_names=("tool", "target"),
                stored_mesh_paths=("179_cm.obj", "107_cm.obj"),
                mesh_paths=(root / "179_cm.obj", root / "107_cm.obj"),
                positions_world=np.zeros((3, 2, 3), dtype=np.float64),
                quaternions_wxyz=quaternions,
                intrinsic=np.eye(3, dtype=np.float64),
                world_to_camera=np.repeat(
                    np.eye(4, dtype=np.float64)[None, ...], 3, axis=0
                ),
            )
            host_paths = {
                name: f"/host/canonical/{path.name}" for name, path in sources.items()
            }
            with mock.patch.dict(
                "os.environ",
                {
                    "V2D_RENDER_CONTAINER_IMAGE": "requested:tag",
                    "V2D_RENDER_CONTAINER_IMAGE_ID": IMAGE_ID,
                },
            ):
                metadata = _metadata_base(
                    inputs,
                    root,
                    "run-id",
                    recorded_source_paths=host_paths,
                )
            self.assertEqual(metadata["container_image"], "requested:tag")
            self.assertEqual(metadata["container_image_id"], IMAGE_ID)
            records = metadata["provenance"]["inputs"]
            self.assertEqual(set(records), set(sources))
            for name, source in sources.items():
                self.assertEqual(records[name]["container_path"], str(source.resolve()))
                self.assertEqual(records[name]["host_path"], host_paths[name])
                self.assertEqual(records[name]["bytes"], source.stat().st_size)
                self.assertEqual(
                    records[name]["sha256"],
                    hashlib.sha256(source.read_bytes()).hexdigest(),
                )


class TacoObjectContainerTests(unittest.TestCase):
    def _config(self, root: Path) -> ObjectDepthContainerConfig:
        repository = root / "repo"
        meshes = root / "meshes"
        output = root / "output"
        partition = root / f"sequence_id={SEQUENCE_ID}" / "robot_name=sharpa_wave"
        taco = root / "(dust, brush, cup)"
        for directory in (repository, meshes, output, partition, taco):
            directory.mkdir(parents=True, exist_ok=True)
        parquet = partition / "data.parquet"
        source_video = taco / "color video.mp4"
        intrinsics = taco / "egocentric intrinsic.txt"
        world_to_camera = taco / "egocentric frame extrinsic.npy"
        for path in (parquet, source_video, intrinsics, world_to_camera):
            path.write_bytes(b"test")
        return ObjectDepthContainerConfig(
            sequence_id=SEQUENCE_ID,
            parquet=parquet,
            source_video=source_video,
            intrinsics=intrinsics,
            world_to_camera=world_to_camera,
            mesh_root=meshes,
            output_dir=output,
            repository_root=repository,
            image_id=IMAGE_ID,
            dry_run=True,
        )

    def test_command_is_offline_read_only_and_gpu_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory))
            command = build_docker_command(config)
            self.assertIn("none", command)
            self.assertNotIn("--gpus", command)
            self.assertIn("--sequence-id", command)
            self.assertEqual(command[command.index("--sequence-id") + 1], SEQUENCE_ID)
            self.assertTrue(
                any("/external_taco_meshes,readonly" in part for part in command)
            )
            entrypoint = command.index("/workspace/isaaclab/isaaclab.sh")
            self.assertEqual(command[entrypoint + 1], IMAGE_ID)
            self.assertIn(f"V2D_RENDER_CONTAINER_IMAGE_ID={IMAGE_ID}", command)
            self.assertIn("--source-video-recorded-path", command)
            self.assertEqual(
                command[command.index("--source-video-recorded-path") + 1],
                str(config.source_video.resolve()),
            )
            with_gpu = ObjectDepthContainerConfig(**{**config.__dict__, "gpu": "0"})
            gpu_command = build_docker_command(with_gpu)
            self.assertEqual(gpu_command[gpu_command.index("--gpus") + 1], "device=0")

    def test_explicit_sequence_must_match_host_hive_partition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory))
            mismatch = ObjectDepthContainerConfig(
                **{**config.__dict__, "sequence_id": "taco_wrong"}
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                build_docker_command(mismatch)

    def test_main_resolves_requested_image_once_before_building_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory))
            argv = [
                "--sequence-id",
                config.sequence_id,
                "--parquet",
                str(config.parquet),
                "--source-video",
                str(config.source_video),
                "--intrinsics",
                str(config.intrinsics),
                "--world-to-camera",
                str(config.world_to_camera),
                "--mesh-root",
                str(config.mesh_root),
                "--output-dir",
                str(config.output_dir),
                "--repository-root",
                str(config.repository_root),
            ]
            with (
                mock.patch(
                    "inpainting.taco_object_depth_container.resolve_local_image_id",
                    return_value=IMAGE_ID,
                ) as resolve,
                redirect_stdout(io.StringIO()) as output,
            ):
                return_code = object_container_main(argv)
            self.assertEqual(return_code, 0)
            resolve.assert_called_once_with(DEFAULT_IMAGE)
            self.assertIn(IMAGE_ID, output.getvalue())


if __name__ == "__main__":
    unittest.main()
