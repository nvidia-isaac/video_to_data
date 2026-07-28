# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
E2FGVI_LIB_SRC = REPOSITORY_ROOT / "inpainting" / "e2fgvi" / "lib" / "src"
sys.path.insert(0, str(E2FGVI_LIB_SRC))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# The shared Docker utility's setuptools package mapping does not mirror its
# source-tree directory layout, so register it under its installed import name.
docker_package = types.ModuleType("v2d.docker")
docker_package.__path__ = []
sys.modules.setdefault("v2d.docker", docker_package)
DOCKER_CONTAINER = _load_module(
    "v2d.docker.container",
    REPOSITORY_ROOT / "reconstruction" / "modules" / "v2d_docker" / "container.py",
)


e2fgvi_docker_root = (
    REPOSITORY_ROOT
    / "inpainting"
    / "e2fgvi"
    / "docker"
    / "src"
    / "v2d"
    / "inpainting"
    / "e2fgvi"
    / "docker"
)
e2fgvi_test_package = types.ModuleType("_e2fgvi_docker_test")
e2fgvi_test_package.__path__ = [str(e2fgvi_docker_root)]
sys.modules[e2fgvi_test_package.__name__] = e2fgvi_test_package
E2FGVI_RUNNER = _load_module(
    f"{e2fgvi_test_package.__name__}.run_inpaint",
    e2fgvi_docker_root / "run_inpaint.py",
)


sam2_root = REPOSITORY_ROOT / "reconstruction" / "modules" / "v2d_sam2"
sam2_package = types.ModuleType("v2d.sam2")
sam2_package.__path__ = [str(sam2_root)]
sys.modules.setdefault("v2d.sam2", sam2_package)
sam2_docker_package = types.ModuleType("v2d.sam2.docker")
sam2_docker_package.__path__ = [str(sam2_root / "docker")]
sys.modules.setdefault("v2d.sam2.docker", sam2_docker_package)
sam2_config = types.ModuleType("v2d.sam2.docker._config")
sam2_config.IMAGE_NAME = "v2d_sam2"
sam2_config.MODULES_DIR = str(sam2_root)
sys.modules.setdefault("v2d.sam2.docker._config", sam2_config)

SAM2_VIDEO_RUNNER = _load_module(
    "v2d.sam2.docker.run_video_to_masks",
    sam2_root / "docker" / "run_video_to_masks.py",
)
SAM2_MV_RUNNER = _load_module(
    "v2d.sam2.docker.run_mv_videos_to_masks",
    sam2_root / "docker" / "run_mv_videos_to_masks.py",
)
SAM2_SHELL_RUNNER = _load_module(
    "v2d.sam2.docker.run_shell",
    sam2_root / "docker" / "run_shell.py",
)
SAM2_ANNOTATE_RUNNER = _load_module(
    "v2d.sam2.docker.run_annotate",
    sam2_root / "docker" / "run_annotate.py",
)
SAM2_IMAGE_ID = "sha256:" + "2" * 64


def _complete_sam2_generation(**arguments) -> None:
    staged = Path(arguments["outputs"]["masks_dir"])
    staged.mkdir()
    (staged / SAM2_VIDEO_RUNNER.RUN_GENERATION_FILENAME).write_text("{}")


wilor_root = REPOSITORY_ROOT / "reconstruction" / "modules" / "v2d_wilor"
wilor_package = types.ModuleType("v2d.wilor")
wilor_package.__path__ = [str(wilor_root)]
sys.modules.setdefault("v2d.wilor", wilor_package)
wilor_docker_package = types.ModuleType("v2d.wilor.docker")
wilor_docker_package.__path__ = [str(wilor_root / "docker")]
sys.modules.setdefault("v2d.wilor.docker", wilor_docker_package)
wilor_config = types.ModuleType("v2d.wilor.docker._config")
wilor_config.IMAGE_NAME = "v2d_wilor"
wilor_config.MODULES_DIR = str(wilor_root)
sys.modules.setdefault("v2d.wilor.docker._config", wilor_config)

WILOR_DOWNLOAD_RUNNER = _load_module(
    "v2d.wilor.docker.run_download_weights",
    wilor_root / "docker" / "run_download_weights.py",
)
WILOR_VIDEO_RUNNER = _load_module(
    "v2d.wilor.docker.run_video_to_hands",
    wilor_root / "docker" / "run_video_to_hands.py",
)
WILOR_RENDER_RUNNER = _load_module(
    "v2d.wilor.docker.run_render_hands_video",
    wilor_root / "docker" / "run_render_hands_video.py",
)
WILOR_SHELL_RUNNER = _load_module(
    "v2d.wilor.docker.run_shell",
    wilor_root / "docker" / "run_shell.py",
)


class ExplicitGpuRunnerTests(unittest.TestCase):
    def test_e2fgvi_exposes_only_selected_physical_gpu_as_cuda_zero(self) -> None:
        image_id = "sha256:" + "e" * 64
        with mock.patch.object(
            E2FGVI_RUNNER, "resolve_local_image_id", return_value=image_id
        ) as resolve, mock.patch.object(E2FGVI_RUNNER, "run_in_container") as run:
            E2FGVI_RUNNER.run_inpaint(
                "source.mp4",
                "masks.npy",
                "checkpoint.pth",
                "output.mp4",
                gpu=3,
                device="cuda:0",
            )

        arguments = run.call_args.kwargs
        resolve.assert_called_once_with(E2FGVI_RUNNER.IMAGE_NAME)
        self.assertEqual(arguments["image"], image_id)
        self.assertEqual(arguments["extra_args"]["container_image_id"], image_id)
        self.assertEqual(
            arguments["extra_args"]["container_image"], E2FGVI_RUNNER.IMAGE_NAME
        )
        self.assertNotIn("gpus", arguments)
        self.assertEqual(arguments["gpu_device"], 3)
        self.assertEqual(arguments["env"], {"CUDA_VISIBLE_DEVICES": "0"})
        self.assertTrue(arguments["strict_io_isolation"])

    def test_sam2_video_exposes_only_selected_physical_gpu_as_cuda_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts = root / "prompts.json"
            prompts.write_text(json.dumps({"prompts": []}))
            output = root / "masks"
            with mock.patch.object(
                SAM2_VIDEO_RUNNER,
                "run_in_container",
                side_effect=_complete_sam2_generation,
            ) as run:
                SAM2_VIDEO_RUNNER.run_video_to_masks(
                    "source.mp4",
                    str(prompts),
                    str(output),
                    "weights",
                    gpu=4,
                    image_id=SAM2_IMAGE_ID,
                )
            self.assertTrue(
                (output / SAM2_VIDEO_RUNNER.RUN_GENERATION_FILENAME).is_file()
            )

        arguments = run.call_args.kwargs
        self.assertEqual(arguments["image"], SAM2_IMAGE_ID)
        self.assertEqual(arguments["extra_args"], {"image_id": SAM2_IMAGE_ID})
        self.assertNotIn("gpus", arguments)
        self.assertEqual(arguments["gpu_device"], 4)
        self.assertEqual(arguments["env"], {"CUDA_VISIBLE_DEVICES": "0"})
        self.assertTrue(arguments["network_disabled"])
        self.assertTrue(arguments["strict_io_isolation"])
        self.assertEqual(arguments["input_directories"], {"weights_dir"})
        self.assertEqual(arguments["input_files"], {"video_path"})
        self.assertEqual(arguments["atomic_output_directories"], {"masks_dir"})
        self.assertNotEqual(Path(arguments["outputs"]["masks_dir"]), output)

    def test_sam2_multiview_uses_selected_physical_gpu(self) -> None:
        with mock.patch.object(SAM2_MV_RUNNER, "run_in_container") as run:
            SAM2_MV_RUNNER.run_mv_videos_to_masks(
                "boxes",
                "rgb",
                "output",
                "weights",
                gpu=2,
            )

        arguments = run.call_args.kwargs
        self.assertEqual(arguments["gpu_device"], 2)
        self.assertEqual(arguments["env"]["CUDA_VISIBLE_DEVICES"], "0")
        self.assertTrue(arguments["network_disabled"])
        self.assertTrue(arguments["strict_io_isolation"])
        self.assertEqual(
            arguments["input_directories"],
            {"bbox_dir", "rgb_dir", "weights_dir"},
        )
        self.assertEqual(arguments["output_directories"], {"output_dir"})

    def test_sam2_preserves_image_directory_input_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = root / "frames"
            frames.mkdir()
            prompts = root / "prompts.json"
            prompts.write_text(json.dumps({"prompts": []}))

            with mock.patch.object(
                SAM2_VIDEO_RUNNER,
                "run_in_container",
                side_effect=_complete_sam2_generation,
            ) as run:
                SAM2_VIDEO_RUNNER.run_video_to_masks(
                    str(frames),
                    str(prompts),
                    str(root / "masks"),
                    str(root / "weights"),
                    image_id=SAM2_IMAGE_ID,
                )

        arguments = run.call_args.kwargs
        self.assertEqual(
            arguments["input_directories"], {"video_path", "weights_dir"}
        )
        self.assertEqual(arguments["input_files"], set())

    def test_sam2_stages_prompt_masks_without_arbitrary_host_mounts(self) -> None:
        staged_parent: Path | None = None
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt_mask = root / "sam2_hand_prompt_masks" / "2" / "000007.png"
            prompt_mask.parent.mkdir(parents=True)
            prompt_mask.write_bytes(b"small-prompt-mask")
            prompts = root / "sam2_prompts.json"
            prompts.write_text(
                json.dumps(
                    {
                        "prompts": [
                            {
                                "object_id": 2,
                                "frame_index": 7,
                                "mask_path": str(prompt_mask.relative_to(root)),
                            }
                        ]
                    }
                )
            )

            def inspect_staging(**arguments):
                nonlocal staged_parent
                staged_prompts = Path(arguments["inputs"]["prompts_path"])
                staged_parent = staged_prompts.parent
                staged = json.loads(staged_prompts.read_text())
                container_mask = staged["prompts"][0]["mask_path"]
                self.assertTrue(container_mask.startswith("/data/prompts_path/"))
                relative = container_mask.removeprefix("/data/prompts_path/")
                self.assertEqual(
                    (staged_prompts.parent / relative).read_bytes(),
                    b"small-prompt-mask",
                )
                self.assertNotIn("extra_volumes", arguments)
                self.assertEqual(arguments["input_files"], {"video_path"})
                _complete_sam2_generation(**arguments)

            with mock.patch.object(
                SAM2_VIDEO_RUNNER,
                "run_in_container",
                side_effect=inspect_staging,
            ):
                SAM2_VIDEO_RUNNER.run_video_to_masks(
                    "source.mp4",
                    str(prompts),
                    str(root / "masks"),
                    "weights",
                    image_id=SAM2_IMAGE_ID,
                )

        assert staged_parent is not None
        self.assertFalse(staged_parent.exists())

    def test_sam2_documented_nested_output_layout_keeps_strict_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            run_root = data_root / "outputs" / "clip"
            run_root.mkdir(parents=True)
            prompts = run_root / "sam2_prompts.json"
            prompts.write_text(json.dumps({"prompts": []}))
            source = data_root / "source.mp4"
            source.write_bytes(b"video")
            weights = data_root / "weights" / "sam2"
            weights.mkdir(parents=True)
            masks = run_root / "masks"

            def complete_generation(command, *, check):
                self.assertTrue(check)
                container_output = command[command.index("--masks_dir") + 1]
                output_mount = next(
                    command[index + 1]
                    for index, value in enumerate(command)
                    if value == "-v" and command[index + 1].endswith(":/data/masks_dir")
                )
                host_parent = Path(output_mount.removesuffix(":/data/masks_dir"))
                staged = host_parent / Path(container_output).name
                staged.mkdir()
                (staged / SAM2_VIDEO_RUNNER.RUN_GENERATION_FILENAME).write_text("{}")

            with mock.patch.object(
                DOCKER_CONTAINER.subprocess,
                "run",
                side_effect=complete_generation,
            ) as run:
                SAM2_VIDEO_RUNNER.run_video_to_masks(
                    str(source),
                    str(prompts),
                    str(masks),
                    str(weights),
                    image_id=SAM2_IMAGE_ID,
                )

        command = run.call_args.args[0]
        mounts = [
            command[index + 1]
            for index, value in enumerate(command)
            if value == "-v"
        ]
        self.assertIn(f"{source}:/data/video_path/source.mp4:ro", mounts)
        self.assertIn(f"{weights}:/data/weights_dir:ro", mounts)
        self.assertNotIn(f"{run_root}:/data/masks_dir", mounts)
        self.assertTrue(
            any(
                mount.startswith(str(run_root / ".masks.container."))
                and mount.endswith(":/data/masks_dir")
                for mount in mounts
            )
        )
        self.assertTrue(
            any(
                mount.startswith(tempfile.gettempdir() + "/sam2_prompts_")
                and mount.endswith(":/data/prompts_path:ro")
                for mount in mounts
            )
        )
        self.assertEqual(command[command.index("--network") + 1], "none")
        self.assertEqual(command[command.index("--image_id") + 1], SAM2_IMAGE_ID)

    def test_sam2_crash_cleans_private_stage_without_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts = root / "prompts.json"
            prompts.write_text(json.dumps({"prompts": []}))
            output = root / "masks"

            with (
                mock.patch.object(
                    SAM2_VIDEO_RUNNER,
                    "run_in_container",
                    side_effect=RuntimeError("container failed"),
                ),
                self.assertRaisesRegex(RuntimeError, "container failed"),
            ):
                SAM2_VIDEO_RUNNER.run_video_to_masks(
                    "source.mp4",
                    str(prompts),
                    str(output),
                    "weights",
                    image_id=SAM2_IMAGE_ID,
                )

            self.assertFalse(output.exists())
            self.assertFalse(list(root.glob(".masks.container.*")))

    def test_sam2_refuses_uncommitted_existing_output_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts = root / "prompts.json"
            prompts.write_text(json.dumps({"prompts": []}))
            output = root / "masks"
            partial = output / "1" / "000000.png"
            partial.parent.mkdir(parents=True)
            partial.write_bytes(b"partial")

            with (
                mock.patch.object(SAM2_VIDEO_RUNNER, "run_in_container") as run,
                self.assertRaisesRegex(FileExistsError, "not a committed generation"),
            ):
                SAM2_VIDEO_RUNNER.run_video_to_masks(
                    "source.mp4",
                    str(prompts),
                    str(output),
                    "weights",
                    image_id=SAM2_IMAGE_ID,
                )

            run.assert_not_called()
            self.assertEqual(partial.read_bytes(), b"partial")

    def test_sam2_existing_commit_is_validation_only_and_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts = root / "prompts.json"
            prompts.write_text(json.dumps({"prompts": []}))
            output = root / "masks"
            output.mkdir()
            marker = output / SAM2_VIDEO_RUNNER.RUN_GENERATION_FILENAME
            marker.write_text("{}")

            with mock.patch.object(SAM2_VIDEO_RUNNER, "run_in_container") as run:
                SAM2_VIDEO_RUNNER.run_video_to_masks(
                    "source.mp4",
                    str(prompts),
                    str(output),
                    "weights",
                    image_id=SAM2_IMAGE_ID,
                )

        arguments = run.call_args.kwargs
        self.assertEqual(arguments["outputs"], {})
        self.assertEqual(arguments["inputs"]["masks_dir"], str(output))
        self.assertIn("masks_dir", arguments["input_directories"])
        self.assertEqual(arguments["atomic_output_directories"], set())

    def test_sam2_direct_docker_tools_never_request_all_gpus(self) -> None:
        with mock.patch.object(SAM2_SHELL_RUNNER.subprocess, "run") as run:
            SAM2_SHELL_RUNNER.run_shell(gpu=5)
        shell_command = run.call_args.args[0]
        self.assertEqual(shell_command[shell_command.index("--gpus") + 1], "device=5")
        self.assertNotIn("all", shell_command)
        self.assertIn("CUDA_VISIBLE_DEVICES=0", shell_command)

        with tempfile.TemporaryDirectory() as directory:
            prompts = Path(directory) / "prompts.json"
            with mock.patch.object(SAM2_ANNOTATE_RUNNER.subprocess, "run") as run:
                SAM2_ANNOTATE_RUNNER.run_annotate(
                    "source.mp4",
                    str(prompts),
                    gpu=6,
                )
        annotate_command = run.call_args.args[0]
        self.assertEqual(
            annotate_command[annotate_command.index("--gpus") + 1],
            "device=6",
        )
        self.assertNotIn("all", annotate_command)
        self.assertIn("CUDA_VISIBLE_DEVICES=0", annotate_command)
        annotate_mounts = [
            annotate_command[index + 1]
            for index, value in enumerate(annotate_command)
            if value == "-v"
        ]
        self.assertTrue(
            any(mount.endswith(":/data/video:ro") for mount in annotate_mounts)
        )
        self.assertTrue(
            any(
                mount.endswith(":/data/prompts") and not mount.endswith(":ro")
                for mount in annotate_mounts
            )
        )

    def test_wilor_inference_is_offline_and_uses_only_selected_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run" / "hands"

            def complete_generation(**arguments):
                staged = Path(arguments["outputs"]["output_dir"])
                staged.mkdir()
                (staged / WILOR_VIDEO_RUNNER.RUN_GENERATION_FILENAME).write_text("{}")

            with mock.patch.object(
                WILOR_VIDEO_RUNNER,
                "run_in_container",
                side_effect=complete_generation,
            ) as run:
                WILOR_VIDEO_RUNNER.run_video_to_hands(
                    "source.mp4",
                    str(output),
                    "weights",
                    image_id="sha256:" + "a" * 64,
                    gpu=2,
                )

            self.assertTrue(
                (output / WILOR_VIDEO_RUNNER.RUN_GENERATION_FILENAME).is_file()
            )

        arguments = run.call_args.kwargs
        self.assertNotIn("gpus", arguments)
        self.assertEqual(arguments["gpu_device"], 2)
        self.assertEqual(arguments["env"], {"CUDA_VISIBLE_DEVICES": "0"})
        self.assertTrue(arguments["network_disabled"])
        self.assertTrue(arguments["strict_io_isolation"])
        self.assertEqual(arguments["input_directories"], {"weights_dir"})
        self.assertEqual(arguments["input_files"], {"video_path"})
        self.assertEqual(arguments["atomic_output_directories"], {"output_dir"})
        self.assertEqual(arguments["image"], "sha256:" + "a" * 64)
        self.assertEqual(arguments["extra_args"]["image_id"], "sha256:" + "a" * 64)
        staged_output = Path(arguments["outputs"]["output_dir"])
        self.assertNotEqual(staged_output, output)
        self.assertEqual(staged_output.parent.parent, output.parent)
        self.assertFalse(staged_output.parent.exists())

    def test_wilor_resolves_local_tag_once_when_image_id_is_omitted(self) -> None:
        image_id = "sha256:" + "c" * 64
        completed = types.SimpleNamespace(stdout=image_id + "\n")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "hands"

            def complete_generation(**arguments):
                staged = Path(arguments["outputs"]["output_dir"])
                staged.mkdir()
                (staged / WILOR_VIDEO_RUNNER.RUN_GENERATION_FILENAME).write_text("{}")

            with (
                mock.patch.object(
                    WILOR_VIDEO_RUNNER.subprocess,
                    "run",
                    return_value=completed,
                ) as inspect,
                mock.patch.object(
                    WILOR_VIDEO_RUNNER,
                    "run_in_container",
                    side_effect=complete_generation,
                ) as run,
            ):
                WILOR_VIDEO_RUNNER.run_video_to_hands(
                    "source.mp4",
                    str(output),
                    "weights",
                )

        inspect.assert_called_once_with(
            [
                "docker",
                "image",
                "inspect",
                "v2d_wilor",
                "--format",
                "{{.Id}}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(run.call_args.kwargs["image"], image_id)
        self.assertEqual(run.call_args.kwargs["extra_args"]["image_id"], image_id)

    def test_wilor_existing_generation_is_mounted_read_only_for_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "wilor_raw"
            output.mkdir()
            (output / WILOR_VIDEO_RUNNER.RUN_GENERATION_FILENAME).write_text("{}")

            with mock.patch.object(WILOR_VIDEO_RUNNER, "run_in_container") as run:
                WILOR_VIDEO_RUNNER.run_video_to_hands(
                    "source.mp4",
                    str(output),
                    "weights",
                    image_id="sha256:" + "d" * 64,
                )

        arguments = run.call_args.kwargs
        self.assertEqual(arguments["outputs"], {})
        self.assertEqual(arguments["inputs"]["output_dir"], str(output))
        self.assertEqual(
            arguments["input_directories"], {"weights_dir", "output_dir"}
        )
        self.assertEqual(arguments["atomic_output_directories"], set())

    def test_wilor_incomplete_existing_output_is_refused_without_staging(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "wilor_raw"
            output.mkdir()
            partial_frame = output / "000000.json"
            partial_frame.write_text("[]")

            with mock.patch.object(
                WILOR_VIDEO_RUNNER,
                "run_in_container",
                side_effect=FileExistsError("no run_generation.json"),
            ) as run, self.assertRaisesRegex(
                FileExistsError, "no run_generation.json"
            ):
                WILOR_VIDEO_RUNNER.run_video_to_hands(
                    "source.mp4",
                    str(output),
                    "weights",
                    image_id="sha256:" + "e" * 64,
                )

            arguments = run.call_args.kwargs
            self.assertEqual(arguments["outputs"], {})
            self.assertEqual(arguments["inputs"]["output_dir"], str(output))
            self.assertEqual(partial_frame.read_text(), "[]")
            self.assertFalse(list(output.parent.glob(".wilor_raw.container.*")))

    def test_wilor_documented_nested_output_uses_private_rw_mount(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory) / "data"
            source = data_root / "clip.mp4"
            source.parent.mkdir()
            source.write_bytes(b"video")
            weights = data_root / "weights" / "wilor"
            weights.mkdir(parents=True)
            output = data_root / "outputs" / "clip" / "wilor_raw"

            def complete_generation(command, *, check):
                self.assertTrue(check)
                container_output = command[command.index("--output_dir") + 1]
                output_mount = next(
                    command[index + 1]
                    for index, value in enumerate(command)
                    if value == "-v"
                    and command[index + 1].endswith(":/data/output_dir")
                )
                host_parent = Path(output_mount.removesuffix(":/data/output_dir"))
                staged = host_parent / Path(container_output).name
                staged.mkdir()
                (staged / WILOR_VIDEO_RUNNER.RUN_GENERATION_FILENAME).write_text("{}")

            with mock.patch.object(
                DOCKER_CONTAINER.subprocess,
                "run",
                side_effect=complete_generation,
            ) as run:
                WILOR_VIDEO_RUNNER.run_video_to_hands(
                    str(source),
                    str(output),
                    str(weights),
                    image_id="sha256:" + "f" * 64,
                )

            command = run.call_args.args[0]
            mounts = [
                command[index + 1]
                for index, value in enumerate(command)
                if value == "-v"
            ]
            self.assertIn(
                f"{source}:/data/video_path/{source.name}:ro",
                mounts,
            )
            self.assertNotIn(f"{output.parent}:/data/output_dir", mounts)
            self.assertTrue(
                (output / WILOR_VIDEO_RUNNER.RUN_GENERATION_FILENAME).is_file()
            )

    def test_wilor_download_is_cpu_only_and_render_uses_selected_gpu(self) -> None:
        with mock.patch.object(WILOR_DOWNLOAD_RUNNER, "run_in_container") as run:
            WILOR_DOWNLOAD_RUNNER.run_download_weights("weights")
        self.assertNotIn("gpu_device", run.call_args.kwargs)
        self.assertNotIn("gpus", run.call_args.kwargs)

        with mock.patch.object(WILOR_RENDER_RUNNER, "run_in_container") as run:
            WILOR_RENDER_RUNNER.run_render_hands_video(
                "frames", "hands", "mano", "overlay.mp4", gpu=4
            )
        arguments = run.call_args.kwargs
        self.assertEqual(arguments["gpu_device"], 4)
        self.assertTrue(arguments["network_disabled"])

    def test_wilor_rejects_boolean_or_negative_gpu(self) -> None:
        for gpu in (True, -1):
            with self.subTest(gpu=gpu), self.assertRaises(ValueError):
                WILOR_VIDEO_RUNNER.run_video_to_hands(
                    "source.mp4",
                    "hands",
                    "weights",
                    image_id="sha256:" + "a" * 64,
                    gpu=gpu,
                )

    def test_wilor_preserves_legacy_positional_bboxes_and_dev_slots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "hands"

            def complete_generation(**arguments):
                staged = Path(arguments["outputs"]["output_dir"])
                staged.mkdir()
                (staged / WILOR_VIDEO_RUNNER.RUN_GENERATION_FILENAME).write_text("{}")

            with mock.patch.object(
                WILOR_VIDEO_RUNNER,
                "run_in_container",
                side_effect=complete_generation,
            ) as run:
                WILOR_VIDEO_RUNNER.run_video_to_hands(
                    "source.mp4",
                    str(output),
                    "weights",
                    "bboxes",
                    True,
                    image_id="sha256:" + "b" * 64,
                )

        arguments = run.call_args.kwargs
        self.assertEqual(arguments["inputs"]["bboxes_dir"], "bboxes")
        self.assertTrue(arguments["dev"])
        self.assertEqual(arguments["image"], "sha256:" + "b" * 64)

    def test_wilor_shell_never_requests_all_gpus(self) -> None:
        with mock.patch.object(WILOR_SHELL_RUNNER.subprocess, "run") as run:
            WILOR_SHELL_RUNNER.run_shell(gpu=7)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--gpus") + 1], "device=7")
        self.assertNotIn("all", command)
        self.assertIn("CUDA_VISIBLE_DEVICES=0", command)
        self.assertFalse(any(value.startswith("HOME=") for value in command))


class WilorOutputClassificationTests(unittest.TestCase):
    def test_pre_manifest_reference_output_remains_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "wilor_raw"
            self.assertEqual(
                WILOR_VIDEO_RUNNER.classify_existing_output(str(output), 7),
                WILOR_VIDEO_RUNNER.OUTPUT_STATE_INCOMPLETE,
            )

            output.mkdir()
            (output / "000006.json").write_text("[]")
            self.assertEqual(
                WILOR_VIDEO_RUNNER.classify_existing_output(str(output), 7),
                WILOR_VIDEO_RUNNER.OUTPUT_STATE_INCOMPLETE,
            )

            (output / "000007.json").write_text("[]")
            self.assertEqual(
                WILOR_VIDEO_RUNNER.classify_existing_output(str(output), 7),
                WILOR_VIDEO_RUNNER.OUTPUT_STATE_INCOMPLETE,
            )

    def test_manifest_presence_routes_output_to_strict_container_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "wilor_raw"
            output.mkdir()
            (output / "000007.json").write_text("[]")
            # Classification deliberately does not trust or parse this file;
            # the container-side generation validator owns that decision.
            (output / WILOR_VIDEO_RUNNER.RUN_GENERATION_FILENAME).write_text("broken")
            self.assertEqual(
                WILOR_VIDEO_RUNNER.classify_existing_output(str(output), 7),
                WILOR_VIDEO_RUNNER.OUTPUT_STATE_COMMITTED,
            )


if __name__ == "__main__":
    unittest.main()
