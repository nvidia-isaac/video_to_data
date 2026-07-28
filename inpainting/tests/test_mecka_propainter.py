from __future__ import annotations

import json
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from inpainting.mecka_panda import propainter
from inpainting.mecka_panda.contracts import sha256


def _write_video(
    path: Path,
    *,
    frame_count: int = 3,
    size: tuple[int, int] = (64, 32),
    fps: float = 10.0,
) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        size,
    )
    if not writer.isOpened():
        raise RuntimeError("test video writer failed")
    for frame_index in range(frame_count):
        writer.write(
            np.full(
                (size[1], size[0], 3),
                20 + frame_index,
                dtype=np.uint8,
            )
        )
    writer.release()


def _write_sentinel_video(
    path: Path,
    *,
    values: tuple[int, ...] = (12, 45, 82, 127, 176, 225),
    size: tuple[int, int] = (64, 32),
    fps: float = 10.0,
) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        size,
    )
    if not writer.isOpened():
        raise RuntimeError("test video writer failed")
    for value in values:
        writer.write(
            np.full(
                (size[1], size[0], 3),
                value,
                dtype=np.uint8,
            )
        )
    writer.release()


def _decoded_frame_means(path: Path) -> list[float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode test video {path}")
    means = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        means.append(float(frame.mean()))
    capture.release()
    return means


def _write_fake_propainter(root: Path) -> tuple[Path, Path]:
    propainter_dir = root / "ProPainter"
    weights_dir = propainter_dir / "weights"
    weights_dir.mkdir(parents=True)
    (propainter_dir / "inference_propainter.py").write_text("# fake\n")
    for index, directory_name in enumerate(propainter.PROPAINTER_SOURCE_DIRECTORIES):
        source_dir = propainter_dir / directory_name
        source_dir.mkdir(parents=True)
        (source_dir / f"runtime_{index}.py").write_text(f"SENTINEL = {index}\n")
    for name in propainter.PROPAINTER_WEIGHT_FILENAMES:
        (weights_dir / name).write_bytes(name.encode())
    python_path = root / "python"
    python_path.write_text("#!/bin/sh\n")
    python_path.chmod(python_path.stat().st_mode | stat.S_IXUSR)
    return propainter_dir, python_path


def _fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
    def argument(name: str) -> Path:
        return Path(command[command.index(name) + 1])

    source_video = argument("--video")
    masks_dir = argument("--mask")
    results_dir = argument("--output")
    resize_ratio = float(command[command.index("--resize_ratio") + 1])
    result_root = results_dir / source_video.stem
    frames_dir = result_root / "frames"
    frames_dir.mkdir(parents=True)
    capture = cv2.VideoCapture(str(source_video))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    backend_size = (
        int(resize_ratio * width),
        int(resize_ratio * height),
    )
    writer = cv2.VideoWriter(
        str(result_root / "inpaint_out.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        backend_size,
    )
    for frame_index, _ in enumerate(sorted(masks_dir.glob("*.png"))):
        ok, _ = capture.read()
        if not ok:
            raise RuntimeError("fake backend source ended early")
        backend = np.full(
            (backend_size[1], backend_size[0], 3),
            (220, 60 + frame_index, 100),
            dtype=np.uint8,
        )
        writer.write(backend)
        cv2.imwrite(str(frames_dir / f"{frame_index:06d}.png"), backend)
    writer.release()
    capture.release()
    return subprocess.CompletedProcess(command, 0)


class ProPainterStageTest(unittest.TestCase):
    def test_preflight_identifies_runtime_and_rejects_missing_weight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            propainter_dir, python_path = _write_fake_propainter(root)
            records = propainter.preflight(propainter_dir, python_path)
            self.assertEqual(
                set(records["weights"]),
                set(propainter.PROPAINTER_WEIGHT_FILENAMES),
            )
            self.assertEqual(
                records["inference_script"]["path"],
                str((propainter_dir / "inference_propainter.py").resolve()),
            )
            self.assertEqual(
                records["source_tree"]["root"],
                str(propainter_dir.resolve()),
            )
            self.assertEqual(
                records["source_tree"]["file_count"],
                1 + len(propainter.PROPAINTER_SOURCE_DIRECTORIES),
            )
            missing_weight = (
                propainter_dir / "weights" / propainter.PROPAINTER_WEIGHT_FILENAMES[0]
            )
            missing_weight.unlink()
            with self.assertRaises(FileNotFoundError):
                propainter.preflight(propainter_dir, python_path)

    def test_source_tree_digest_tracks_added_python_but_ignores_pycache(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            propainter_dir, _ = _write_fake_propainter(root)
            initial = propainter.source_tree_identity(propainter_dir)
            cache = propainter_dir / "model" / "__pycache__"
            cache.mkdir()
            (cache / "runtime.cpython-310.pyc").write_bytes(b"generated")
            after_cache = propainter.source_tree_identity(propainter_dir)
            self.assertEqual(
                after_cache["tree_sha256"],
                initial["tree_sha256"],
            )

            added = propainter_dir / "RAFT" / "new_runtime.py"
            added.write_text("VALUE = 1\n")
            after_add = propainter.source_tree_identity(propainter_dir)
            self.assertNotEqual(
                after_add["tree_sha256"],
                initial["tree_sha256"],
            )
            relative_paths = {record["relative_path"] for record in after_add["files"]}
            self.assertIn("RAFT/new_runtime.py", relative_paths)

    def test_execute_publishes_hashed_bundle_and_legacy_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "window.mp4"
            mask_path = root / "arm_mask.npy"
            output = root / "output"
            _write_video(source, frame_count=5)
            mask = np.zeros((3, 32, 64), dtype=np.bool_)
            mask[:, 8:24, 16:48] = True
            np.save(mask_path, mask)
            propainter_dir, python_path = _write_fake_propainter(root)

            with mock.patch.object(
                propainter.subprocess,
                "run",
                side_effect=_fake_run,
            ) as run:
                metadata = propainter.execute(
                    source_video=source,
                    mask=mask_path,
                    output_dir=output,
                    source_start_frame=1,
                    propainter_dir=propainter_dir,
                    propainter_python=python_path,
                )

            command = run.call_args.args[0]
            self.assertIn("--save_frames", command)
            self.assertIn("--fp16", command)
            self.assertEqual(command[command.index("--resize_ratio") + 1], "0.5")
            self.assertEqual(command[command.index("--subvideo_length") + 1], "40")
            self.assertEqual(command[command.index("--neighbor_length") + 1], "6")
            self.assertEqual(command[command.index("--ref_stride") + 1], "10")
            self.assertNotEqual(Path(command[command.index("--video") + 1]), source)
            self.assertEqual(run.call_args.kwargs["cwd"], propainter_dir)
            self.assertNotIn("HTTP_PROXY", run.call_args.kwargs["env"])

            result = output / propainter.OUTPUT_FILENAME
            sidecar = output / propainter.METADATA_FILENAME
            self.assertTrue(result.is_file())
            self.assertTrue(sidecar.is_file())
            self.assertEqual(metadata["schema_version"], propainter.PROPAINTER_SCHEMA)
            self.assertEqual(metadata["statistics"]["masked_pixel_count"], 1536)
            self.assertEqual(
                metadata["source_window"],
                {"start_frame": 1, "stop_frame_exclusive": 4},
            )
            self.assertEqual(metadata["output"]["video"]["sha256"], sha256(result))
            self.assertEqual(
                metadata["hashes"]["output_sha256"],
                metadata["output"]["video"]["sha256"],
            )
            self.assertEqual(
                metadata["backend"]["geometry"]["requested_frame"],
                {"width": 32, "height": 16},
            )
            self.assertEqual(
                metadata["compositing"]["policy"],
                "full_resolution_formal_bool_mask_only",
            )
            self.assertEqual(
                metadata["source"]["implementation"]["source_tree"]["root"],
                str(propainter_dir.resolve()),
            )
            self.assertEqual(
                json.loads(sidecar.read_text(encoding="utf-8")),
                metadata,
            )
            result_capture = cv2.VideoCapture(str(result))
            source_capture = cv2.VideoCapture(str(source))
            source_capture.set(cv2.CAP_PROP_POS_FRAMES, 1)
            source_ok, source_frame = source_capture.read()
            result_ok, result_frame = result_capture.read()
            source_capture.release()
            result_capture.release()
            self.assertTrue(source_ok and result_ok)
            self.assertEqual(result_frame.shape, (32, 64, 3))
            outside = ~mask[0]
            outside_error = np.mean(
                np.abs(
                    result_frame[outside].astype(np.int16)
                    - source_frame[outside].astype(np.int16)
                )
            )
            inside_change = np.mean(
                np.abs(
                    result_frame[mask[0]].astype(np.int16)
                    - source_frame[mask[0]].astype(np.int16)
                )
            )
            self.assertLess(outside_error, 8.0)
            self.assertGreater(inside_change, 30.0)

            with self.assertRaises(FileExistsError):
                propainter.execute(
                    source_video=source,
                    mask=mask_path,
                    output_dir=output,
                    source_start_frame=1,
                    propainter_dir=propainter_dir,
                    propainter_python=python_path,
                )

    def test_nonzero_start_uses_sequential_decode_and_exact_window_source(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sentinels.mp4"
            mask_path = root / "arm_mask.npy"
            output = root / "output"
            _write_sentinel_video(source)
            np.save(mask_path, np.zeros((3, 32, 64), dtype=np.bool_))
            propainter_dir, python_path = _write_fake_propainter(root)

            original_video_capture = cv2.VideoCapture
            source_grabs = 0

            class NoFrameSeekCapture:
                def __init__(self, capture_source: object) -> None:
                    self._capture_source = str(capture_source)
                    self._capture = original_video_capture(capture_source)

                def __getattr__(self, name: str) -> object:
                    return getattr(self._capture, name)

                def grab(self) -> bool:
                    nonlocal source_grabs
                    if self._capture_source == str(source):
                        source_grabs += 1
                    return bool(self._capture.grab())

                def set(self, property_id: int, value: float) -> bool:
                    if property_id == cv2.CAP_PROP_POS_FRAMES:
                        raise AssertionError(
                            f"frame seek is forbidden in this test: {value}"
                        )
                    return bool(self._capture.set(property_id, value))

            with (
                mock.patch.object(
                    propainter.cv2,
                    "VideoCapture",
                    NoFrameSeekCapture,
                ),
                mock.patch.object(
                    propainter.subprocess,
                    "run",
                    side_effect=_fake_run,
                ),
            ):
                propainter.execute(
                    source_video=source,
                    mask=mask_path,
                    output_dir=output,
                    source_start_frame=2,
                    propainter_dir=propainter_dir,
                    propainter_python=python_path,
                )

            self.assertEqual(source_grabs, 2)
            source_means = _decoded_frame_means(source)
            output_means = _decoded_frame_means(output / propainter.OUTPUT_FILENAME)
            self.assertEqual(len(output_means), 3)
            closest_source_indices = [
                int(np.argmin(np.abs(np.asarray(source_means) - output_mean)))
                for output_mean in output_means
            ]
            self.assertEqual(closest_source_indices, [2, 3, 4])

    def test_rejects_non_boolean_or_misaligned_mask_before_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "window.mp4"
            mask_path = root / "arm_mask.npy"
            _write_video(source)
            propainter_dir, python_path = _write_fake_propainter(root)
            np.save(mask_path, np.zeros((3, 32, 64), dtype=np.uint8))
            with (
                mock.patch.object(propainter.subprocess, "run") as run,
                self.assertRaisesRegex(ValueError, "non-empty bool"),
            ):
                propainter.execute(
                    source_video=source,
                    mask=mask_path,
                    output_dir=root / "output",
                    propainter_dir=propainter_dir,
                    propainter_python=python_path,
                )
            run.assert_not_called()

            np.save(mask_path, np.zeros((3, 31, 64), dtype=np.bool_))
            with (
                mock.patch.object(propainter.subprocess, "run") as run,
                self.assertRaisesRegex(ValueError, "spatial geometry"),
            ):
                propainter.execute(
                    source_video=source,
                    mask=mask_path,
                    output_dir=root / "output",
                    propainter_dir=propainter_dir,
                    propainter_python=python_path,
                )
            run.assert_not_called()

    def test_failure_does_not_publish_partial_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "window.mp4"
            mask_path = root / "arm_mask.npy"
            output = root / "output"
            _write_video(source)
            np.save(mask_path, np.zeros((3, 32, 64), dtype=np.bool_))
            propainter_dir, python_path = _write_fake_propainter(root)
            with (
                mock.patch.object(
                    propainter.subprocess,
                    "run",
                    side_effect=subprocess.CalledProcessError(1, ["fake"]),
                ),
                self.assertRaises(subprocess.CalledProcessError),
            ):
                propainter.execute(
                    source_video=source,
                    mask=mask_path,
                    output_dir=output,
                    propainter_dir=propainter_dir,
                    propainter_python=python_path,
                )
            self.assertFalse((output / propainter.OUTPUT_FILENAME).exists())
            self.assertFalse((output / propainter.METADATA_FILENAME).exists())


if __name__ == "__main__":
    unittest.main()
