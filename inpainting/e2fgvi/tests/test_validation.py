# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

LIB_SRC = Path(__file__).resolve().parents[1] / "lib" / "src"
sys.path.insert(0, str(LIB_SRC))

from v2d.inpainting.e2fgvi.validation import (  # noqa: E402
    E2FGVI_COMMIT,
    InferenceConfig,
    InputValidationError,
    RunPlan,
    VideoInfo,
    build_metadata,
    canonical_json,
    compute_processing_size,
    enrich_completed_metadata,
    select_reference_indices,
    validate_config,
    validate_mask_array,
    validate_output_paths,
    write_metadata,
)
from v2d.inpainting.e2fgvi.cli import main as cli_main  # noqa: E402
from v2d.inpainting.e2fgvi.inference import _write_video_atomic  # noqa: E402


def _write_test_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        12.5,
        (16, 12),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open test video writer for {path}")
    for value in (0, 80, 160):
        writer.write(np.full((12, 16, 3), value, dtype=np.uint8))
    writer.release()


class ProcessingSizeTests(unittest.TestCase):
    def test_downscale_then_caps_longest_edge(self) -> None:
        self.assertEqual(compute_processing_size(1920, 1080, 2.0, 800), (800, 450))

    def test_disabled_cap_preserves_source_at_scale_one(self) -> None:
        self.assertEqual(compute_processing_size(641, 359, 1.0, 0), (641, 359))

    def test_invalid_scale_is_rejected(self) -> None:
        with self.assertRaisesRegex(InputValidationError, "downscale"):
            compute_processing_size(100, 50, 0.5, 0)


class MaskValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.video = VideoInfo(width=8, height=6, frame_count=3, fps=29.97)

    def test_accepts_exact_boolean_contract(self) -> None:
        masks = np.zeros((3, 6, 8), dtype=bool)
        self.assertEqual(validate_mask_array(masks, self.video), (3, 6, 8))

    def test_rejects_integer_masks(self) -> None:
        masks = np.zeros((3, 6, 8), dtype=np.uint8)
        with self.assertRaisesRegex(InputValidationError, "boolean dtype"):
            validate_mask_array(masks, self.video)

    def test_rejects_frame_count_mismatch(self) -> None:
        masks = np.zeros((2, 6, 8), dtype=bool)
        with self.assertRaisesRegex(InputValidationError, "frame count"):
            validate_mask_array(masks, self.video)

    def test_rejects_resolution_mismatch(self) -> None:
        masks = np.zeros((3, 5, 8), dtype=bool)
        with self.assertRaisesRegex(InputValidationError, "resolution"):
            validate_mask_array(masks, self.video)


class SamplingAndConfigTests(unittest.TestCase):
    def test_reference_limit_prefers_nearest_candidates(self) -> None:
        references = select_reference_indices(
            center=5,
            neighbor_indices=[3, 4, 5, 6, 7],
            frame_count=10,
            ref_stride=2,
            num_ref=2,
        )
        self.assertEqual(references, [2, 8])

    def test_all_reference_mode_is_ordered(self) -> None:
        self.assertEqual(
            select_reference_indices(3, [2, 3, 4], 8, 2, -1),
            [0, 6],
        )

    def test_even_dilation_kernel_is_rejected(self) -> None:
        with self.assertRaisesRegex(InputValidationError, "positive odd"):
            validate_config(InferenceConfig(dilation_kernel=4))

    def test_invalid_device_is_rejected(self) -> None:
        with self.assertRaisesRegex(InputValidationError, "device"):
            validate_config(InferenceConfig(device="mps"))

    def test_output_and_metadata_must_be_distinct(self) -> None:
        with self.assertRaisesRegex(InputValidationError, "metadata_path"):
            validate_output_paths("video.mp4", "masks.npy", "model.pth", "out.mp4", "out.mp4")


class MetadataTests(unittest.TestCase):
    def test_metadata_is_canonical_and_repeatable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.mp4"
            masks = root / "masks.npy"
            checkpoint = root / "model.pth"
            output = root / "result.mp4"
            video.write_bytes(b"video")
            masks.write_bytes(b"masks")
            checkpoint.write_bytes(b"checkpoint")
            plan = RunPlan(
                video=VideoInfo(width=8, height=6, frame_count=3, fps=24.0),
                mask_shape=(3, 6, 8),
                processing_width=8,
                processing_height=6,
                config=InferenceConfig(device="cpu"),
            )
            first = build_metadata(plan, video, masks, checkpoint, output, "validated")
            second = build_metadata(plan, video, masks, checkpoint, output, "validated")
            self.assertEqual(canonical_json(first), canonical_json(second))
            parsed = json.loads(canonical_json(first))
            self.assertEqual(parsed["implementation"]["commit"], E2FGVI_COMMIT)
            self.assertNotIn(str(root), canonical_json(first))

            sidecar = root / "result.mp4.json"
            write_metadata(sidecar, first)
            self.assertEqual(sidecar.read_text(encoding="utf-8"), canonical_json(first))

    def test_completed_metadata_fingerprints_validated_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.avi"
            output = root / "result.avi"
            _write_test_video(video)
            output.write_bytes(video.read_bytes())
            masks = root / "masks.npy"
            np.save(masks, np.zeros((3, 12, 16), dtype=bool))
            checkpoint = root / "model.pth"
            checkpoint.write_bytes(b"checkpoint")
            plan = RunPlan(
                video=VideoInfo(width=16, height=12, frame_count=3, fps=12.5),
                mask_shape=(3, 12, 16),
                processing_width=16,
                processing_height=12,
                config=InferenceConfig(device="cpu", codec="MJPG"),
            )

            report = build_metadata(plan, video, masks, checkpoint, output, "completed")

            self.assertEqual(report["output"]["bytes"], output.stat().st_size)
            self.assertEqual(
                report["output"]["sha256"], hashlib.sha256(output.read_bytes()).hexdigest()
            )

    def test_legacy_enrichment_validates_inputs_and_output_before_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.avi"
            output = root / "result.avi"
            _write_test_video(video)
            output.write_bytes(video.read_bytes())
            masks = root / "masks.npy"
            np.save(masks, np.zeros((3, 12, 16), dtype=bool))
            checkpoint = root / "model.pth"
            checkpoint.write_bytes(b"checkpoint")
            plan = RunPlan(
                video=VideoInfo(width=16, height=12, frame_count=3, fps=12.5),
                mask_shape=(3, 12, 16),
                processing_width=16,
                processing_height=12,
                config=InferenceConfig(device="cpu", codec="MJPG"),
            )
            legacy = build_metadata(plan, video, masks, checkpoint, output, "validated")
            legacy["status"] = "completed"
            sidecar = root / "result.json"
            write_metadata(sidecar, legacy)

            enriched = enrich_completed_metadata(
                sidecar, video, masks, checkpoint, output
            )

            self.assertEqual(enriched["output"]["bytes"], output.stat().st_size)
            self.assertEqual(
                enriched["implementation"]["container_image_provenance"],
                "legacy_unrecorded",
            )
            self.assertEqual(json.loads(sidecar.read_text()), enriched)


class ValidationCliTests(unittest.TestCase):
    def test_validate_only_needs_neither_torch_nor_a_model(self) -> None:
        self.assertNotIn("torch", sys.modules)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.avi"
            _write_test_video(video)
            masks = root / "masks.npy"
            np.save(masks, np.zeros((3, 12, 16), dtype=bool))
            checkpoint = root / "checkpoint.pth"
            checkpoint.write_bytes(b"validation does not deserialize this")
            output = root / "result.mp4"
            sidecar = root / "result.json"

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = cli_main(
                    [
                        "--input-video",
                        str(video),
                        "--masks",
                        str(masks),
                        "--checkpoint",
                        str(checkpoint),
                        "--output-video",
                        str(output),
                        "--metadata-path",
                        str(sidecar),
                        "--validate-only",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertFalse(output.exists())
            self.assertEqual(json.loads(sidecar.read_text())["status"], "validated")
            self.assertNotIn("torch", sys.modules)

    def test_failed_overwrite_leaves_committing_marker_not_old_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.avi"
            output = root / "result.avi"
            sidecar = root / "result.json"
            _write_test_video(video)
            output.write_bytes(video.read_bytes())
            sidecar.write_text(json.dumps({"status": "completed", "old": True}))
            masks = root / "masks.npy"
            np.save(masks, np.zeros((3, 12, 16), dtype=bool))
            checkpoint = root / "model.pth"
            checkpoint.write_bytes(b"checkpoint")

            def fail_after_commit(*args, **kwargs):
                self.assertEqual(json.loads(sidecar.read_text())["status"], "committing")
                raise RuntimeError("simulated inference crash")

            with mock.patch(
                "v2d.inpainting.e2fgvi.inference.run_inference",
                side_effect=fail_after_commit,
            ), self.assertRaisesRegex(RuntimeError, "simulated inference crash"):
                cli_main(
                    [
                        "--input-video",
                        str(video),
                        "--masks",
                        str(masks),
                        "--checkpoint",
                        str(checkpoint),
                        "--output-video",
                        str(output),
                        "--metadata-path",
                        str(sidecar),
                        "--device",
                        "cpu",
                        "--codec",
                        "MJPG",
                        "--overwrite",
                    ]
                )

            marker = json.loads(sidecar.read_text())
            self.assertEqual(marker["status"], "committing")
            self.assertNotIn("old", marker)

    def test_success_commits_fingerprint_only_after_inputs_remain_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.avi"
            output = root / "result.avi"
            sidecar = root / "result.json"
            _write_test_video(video)
            masks = root / "masks.npy"
            np.save(masks, np.zeros((3, 12, 16), dtype=bool))
            checkpoint = root / "model.pth"
            checkpoint.write_bytes(b"checkpoint")

            def copy_output(*args, **kwargs):
                output.write_bytes(video.read_bytes())

            with mock.patch(
                "v2d.inpainting.e2fgvi.inference.run_inference",
                side_effect=copy_output,
            ), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    cli_main(
                        [
                            "--input-video",
                            str(video),
                            "--masks",
                            str(masks),
                            "--checkpoint",
                            str(checkpoint),
                            "--output-video",
                            str(output),
                            "--metadata-path",
                            str(sidecar),
                            "--device",
                            "cpu",
                            "--codec",
                            "MJPG",
                        ]
                    ),
                    0,
                )

            report = json.loads(sidecar.read_text())
            self.assertEqual(report["status"], "completed")
            self.assertEqual(
                report["output"]["sha256"], hashlib.sha256(output.read_bytes()).hexdigest()
            )

    def test_input_change_during_inference_keeps_committing_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.avi"
            output = root / "result.avi"
            sidecar = root / "result.json"
            _write_test_video(video)
            masks = root / "masks.npy"
            np.save(masks, np.zeros((3, 12, 16), dtype=bool))
            checkpoint = root / "model.pth"
            checkpoint.write_bytes(b"checkpoint")

            def mutate_input(*args, **kwargs):
                output.write_bytes(video.read_bytes())
                changed = np.zeros((3, 12, 16), dtype=bool)
                changed[0, 0, 0] = True
                np.save(masks, changed)

            with mock.patch(
                "v2d.inpainting.e2fgvi.inference.run_inference",
                side_effect=mutate_input,
            ), self.assertRaisesRegex(InputValidationError, "changed during inference"):
                cli_main(
                    [
                        "--input-video",
                        str(video),
                        "--masks",
                        str(masks),
                        "--checkpoint",
                        str(checkpoint),
                        "--output-video",
                        str(output),
                        "--metadata-path",
                        str(sidecar),
                        "--device",
                        "cpu",
                        "--codec",
                        "MJPG",
                    ]
                )

            self.assertEqual(json.loads(sidecar.read_text())["status"], "committing")

    def test_video_writer_preserves_geometry_frame_count_and_fractional_fps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.mp4"
            frames = [np.full((12, 16, 3), value, dtype=np.uint8) for value in (0, 60, 120)]
            _write_video_atomic(output, frames, fps=12.5, codec="mp4v")
            capture = cv2.VideoCapture(str(output))
            self.assertTrue(capture.isOpened())
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 16)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 12)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
            self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 12.5, places=3)
            capture.release()


if __name__ == "__main__":
    unittest.main()
