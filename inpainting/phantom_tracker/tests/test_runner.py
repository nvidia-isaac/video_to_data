from __future__ import annotations

import argparse
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from inpainting.phantom_tracker import IMAGE_NAME
from inpainting.phantom_tracker.runner import _mount, acquire, infer


IMAGE_ID = "sha256:" + "a" * 64


class MountTests(unittest.TestCase):
    def test_taco_tuple_path_is_one_read_only_volume_argument(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "(dust, brush, cup)"
            source.mkdir()
            self.assertEqual(
                _mount(source, "/input", True),
                ["-v", f"{source.resolve()}:/input:ro"],
            )


def _infer_args(root: Path, *, dry_run: bool) -> argparse.Namespace:
    video = root / "video.mp4"
    intrinsics = root / "intrinsics.txt"
    models = root / "models"
    mano = root / "mano"
    video.write_bytes(b"video")
    intrinsics.write_text("1 0 0\n0 1 0\n0 0 1\n")
    (models / "hamer/_DATA/hamer_ckpts/checkpoints").mkdir(parents=True)
    (models / "hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt").write_bytes(b"hamer")
    (models / "grounding-dino-base").mkdir(parents=True)
    (models / "grounding-dino-base/config.json").write_text("{}")
    mano.mkdir()
    (mano / "MANO_LEFT.pkl").write_bytes(b"left")
    (mano / "MANO_RIGHT.pkl").write_bytes(b"right")
    return argparse.Namespace(
        video=video,
        intrinsics=intrinsics,
        models_dir=models,
        mano_dir=mano,
        output_dir=root / "new" / "tracking",
        sequence_id="test_sequence",
        gpu=0,
        box_threshold=0.2,
        text_threshold=0.2,
        min_valid_fraction=0.5,
        max_ambiguous_fraction=0.15,
        batch_size=16,
        minimum_box_area_fraction=0.001,
        maximum_box_area_fraction=0.12,
        overwrite=False,
        dry_run=dry_run,
    )


class InferBoundaryTests(unittest.TestCase):
    def test_dry_run_does_not_create_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = _infer_args(Path(directory), dry_run=True)
            with (
                patch(
                    "inpainting.phantom_tracker.runner._inspect_image",
                    return_value={"id": IMAGE_ID, "labels": {}},
                ),
                patch("inpainting.phantom_tracker.runner.subprocess.run") as execute,
                redirect_stdout(io.StringIO()),
            ):
                plan = infer(args)
            self.assertFalse(args.output_dir.exists())
            self.assertEqual(plan["output_dir"], str(args.output_dir.resolve()))
            command = plan["command"]
            self.assertEqual(command[command.index("python") - 1], IMAGE_ID)
            self.assertNotEqual(command[command.index("python") - 1], IMAGE_NAME)
            self.assertEqual(plan["image"], IMAGE_NAME)
            self.assertEqual(plan["container_image_id"], IMAGE_ID)
            execute.assert_not_called()

    def test_hash_guard_runs_before_container_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = _infer_args(Path(directory), dry_run=False)
            events: list[str] = []

            def verify(*_args: object) -> None:
                events.append("verify")

            def execute(*_args: object, **_kwargs: object) -> None:
                events.append("execute")
                raise OSError("stop after boundary assertion")

            with (
                patch(
                    "inpainting.phantom_tracker.runner._inspect_image",
                    return_value={"id": IMAGE_ID, "labels": {}},
                ),
                patch(
                    "inpainting.phantom_tracker.runner.verify_pinned_inference_assets",
                    side_effect=verify,
                ),
                patch(
                    "inpainting.phantom_tracker.runner.subprocess.run",
                    side_effect=execute,
                ),
                self.assertRaisesRegex(OSError, "boundary assertion"),
            ):
                infer(args)
            self.assertEqual(events, ["verify", "execute"])


class AcquisitionImageTests(unittest.TestCase):
    def test_grounding_dino_acquisition_executes_inspected_image_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch(
                    "inpainting.phantom_tracker.runner._inspect_image",
                    return_value={"id": IMAGE_ID, "labels": {}},
                ),
                patch(
                    "inpainting.phantom_tracker.runner.acquire_hamer",
                    return_value={"state": "complete"},
                ),
                patch(
                    "inpainting.phantom_tracker.runner.sha256_tree",
                    return_value=("tree-hash", []),
                ),
                patch("inpainting.phantom_tracker.runner.subprocess.run") as execute,
            ):
                manifest = acquire(root / "downloads", root / "models")
            command = execute.call_args.args[0]
            self.assertEqual(command[command.index("python") - 1], IMAGE_ID)
            self.assertNotEqual(command[command.index("python") - 1], IMAGE_NAME)
            self.assertEqual(manifest["container_image_id"], IMAGE_ID)


if __name__ == "__main__":
    unittest.main()
