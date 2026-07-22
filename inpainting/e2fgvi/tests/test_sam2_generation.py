from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
GENERATION_PATH = (
    REPOSITORY_ROOT
    / "reconstruction"
    / "modules"
    / "v2d_sam2"
    / "lib"
    / "generation.py"
)
SPEC = importlib.util.spec_from_file_location("_sam2_generation_test", GENERATION_PATH)
assert SPEC is not None and SPEC.loader is not None
GENERATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATION)
IMAGE_ID = "sha256:" + "4" * 64


class Sam2GenerationTest(unittest.TestCase):
    def _inputs(self, root: Path) -> tuple[Path, Path, Path, dict]:
        video = root / "source.mp4"
        video.write_bytes(b"video-generation-one")
        prompts = root / "prompts.json"
        prompt_data = {
            "prompts": [
                {
                    "frame_index": 0,
                    "object_id": 1,
                    "points": None,
                    "point_labels": None,
                    "box": {"x0": 1, "y0": 2, "x1": 3, "y1": 4},
                    "mask_path": None,
                }
            ]
        }
        prompts.write_text(json.dumps(prompt_data))
        weights = root / "weights"
        weights.mkdir()
        (weights / "sam2.1_hiera_large.pt").write_bytes(b"checkpoint")
        return video, prompts, weights, prompt_data

    def _static(self, root: Path) -> tuple[dict, Path]:
        video, prompts, weights, prompt_data = self._inputs(root)
        static = GENERATION.build_static_identity(
            str(video),
            str(prompts),
            str(weights),
            prompt_data["prompts"],
            IMAGE_ID,
        )
        return static, prompts

    @staticmethod
    def _write_masks(output: Path, frame_count: int = 2) -> None:
        object_dir = output / "1"
        object_dir.mkdir(parents=True)
        for index in range(frame_count):
            (object_dir / f"{index:06d}.png").write_bytes(f"mask-{index}".encode())

    def test_complete_generation_validates_and_detects_same_path_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            static, _ = self._static(root)
            output = root / "masks"
            self._write_masks(output)

            GENERATION.commit_generation(output, static, [1], 2)
            manifest = GENERATION.validate_generation(output, static)
            self.assertEqual(manifest["state"], "complete")

            (output / "1" / "000001.png").write_bytes(b"changed-mask")
            with self.assertRaisesRegex(RuntimeError, "hashes no longer match"):
                GENERATION.validate_generation(output, static)

            (output / "1" / "000001.png").unlink()
            with self.assertRaisesRegex(RuntimeError, "exactly 2 frames"):
                GENERATION.validate_generation(output, static)

    def test_partial_or_stale_tree_cannot_receive_complete_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            static, _ = self._static(root)
            output = root / "masks"
            self._write_masks(output, frame_count=1)

            with self.assertRaisesRegex(RuntimeError, "exactly 2 frames"):
                GENERATION.commit_generation(output, static, [1], 2)
            self.assertFalse((output / GENERATION.RUN_GENERATION_FILENAME).exists())

    def test_changed_prompt_identity_refuses_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            static, prompts = self._static(root)
            output = root / "masks"
            self._write_masks(output)
            GENERATION.commit_generation(output, static, [1], 2)

            prompts.write_text('{"prompts": []}')
            changed_static = GENERATION.build_static_identity(
                str(root / "source.mp4"),
                str(prompts),
                str(root / "weights"),
                [],
                IMAGE_ID,
            )
            with self.assertRaisesRegex(RuntimeError, "different video/prompt"):
                GENERATION.validate_generation(output, changed_static)


if __name__ == "__main__":
    unittest.main()
