"""Remove tracked hands from an exact video window with ProPainter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from inpainting.mecka_panda.contracts import artifact, write_json_atomic
from inpainting.mecka_panda.video_io import Mp4Writer, probe_video

PROPAINTER_SCHEMA = "v2d.inpainting.propainter-run/v1"
OUTPUT_FILENAME = "hand_removed.mp4"
METADATA_FILENAME = "hand_removed.json"
PROPAINTER_WEIGHT_FILENAMES = (
    "ProPainter.pth",
    "raft-things.pth",
    "recurrent_flow_completion.pth",
)
PROPAINTER_SOURCE_DIRECTORIES = ("model", "RAFT", "core", "utils")
DEFAULT_PROPAINTER_DIR = (
    Path(__file__).resolve().parents[2] / "debug" / "third_party" / "ProPainter"
)
DEFAULT_PROPAINTER_PYTHON = (
    Path.home() / "miniconda3" / "envs" / "vlmevalkit" / "bin" / "python"
)


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _without_proxy() -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
        "SOCKS_PROXY",
        "SOCKS5_PROXY",
        "socks_proxy",
        "socks5_proxy",
        "GIT_HTTP_PROXY",
        "GIT_HTTPS_PROXY",
    ):
        environment.pop(name, None)
    environment["NO_PROXY"] = "localhost,127.0.0.1"
    environment["no_proxy"] = "localhost,127.0.0.1"
    return environment


def _validate_parameters(
    *,
    resize_ratio: float,
    subvideo_length: int,
    neighbor_length: int,
    ref_stride: int,
) -> None:
    if not np.isfinite(resize_ratio) or resize_ratio <= 0.0:
        raise ValueError("resize_ratio must be positive and finite")
    if subvideo_length <= 0:
        raise ValueError("subvideo_length must be positive")
    if neighbor_length < 2:
        raise ValueError("neighbor_length must be at least 2")
    if ref_stride <= 0:
        raise ValueError("ref_stride must be positive")


def _validate_decoded_video(
    path: Path,
    *,
    frame_count: int,
    width: int,
    height: int,
) -> None:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot decode video {path}")
    try:
        for frame_index in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(
                    f"{path} ended at frame {frame_index}; expected {frame_count}"
                )
            if frame.shape != (height, width, 3):
                raise ValueError(
                    f"{path} frame {frame_index} has shape {frame.shape}; "
                    f"expected {(height, width, 3)}"
                )
        has_extra_frame, _ = capture.read()
        if has_extra_frame:
            raise ValueError(f"{path} contains more than {frame_count} frames")
    finally:
        capture.release()


def _load_mask(path: Path) -> np.ndarray:
    loaded = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(loaded, np.ndarray):
        loaded.close()
        raise TypeError("mask must be a single NPY array, not an NPZ archive")
    if (
        loaded.ndim != 3
        or loaded.shape[0] <= 0
        or loaded.shape[1] <= 0
        or loaded.shape[2] <= 0
        or loaded.dtype != np.bool_
    ):
        raise ValueError("mask must be a non-empty bool array with shape (N,H,W)")
    return loaded


def _write_exact_window(
    source: Path,
    destination: Path,
    *,
    source_start_frame: int,
    frame_count: int,
    width: int,
    height: int,
    fps: float,
) -> None:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Cannot decode video {source}")
    for skipped_frame in range(source_start_frame):
        if not capture.grab():
            capture.release()
            raise RuntimeError(
                f"{source} ended while skipping source frame {skipped_frame}"
            )
    writer = Mp4Writer(destination, fps, (width, height))
    try:
        for frame_index in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(
                    f"{source} ended while extracting window frame {frame_index}"
                )
            if frame.shape != (height, width, 3):
                raise RuntimeError(
                    f"{source} frame has shape {frame.shape}; "
                    f"expected {(height, width, 3)}"
                )
            writer.write(frame)
    finally:
        writer.close()
        capture.release()


def _materialize_masks(mask: np.ndarray, destination: Path) -> int:
    destination.mkdir(parents=True)
    masked_pixels = 0
    for frame_index in range(mask.shape[0]):
        frame_mask = np.asarray(mask[frame_index], dtype=np.uint8)
        masked_pixels += int(frame_mask.sum())
        written = cv2.imwrite(
            str(destination / f"{frame_index:06d}.png"),
            frame_mask * np.uint8(255),
            [cv2.IMWRITE_PNG_COMPRESSION, 1],
        )
        if not written:
            raise RuntimeError(f"Could not write mask PNG for frame {frame_index}")
    return masked_pixels


def _macroblock_dimension(value: int, macroblock_size: int = 16) -> int:
    return ((value + macroblock_size - 1) // macroblock_size) * macroblock_size


def _validate_backend_outputs(
    result_video: Path,
    *,
    frame_count: int,
    width: int,
    height: int,
    fps: float,
    resize_ratio: float,
) -> tuple[list[Path], dict[str, Any]]:
    frames_dir = result_video.parent / "frames"
    frame_paths = sorted(frames_dir.glob("*.png"))
    if len(frame_paths) != frame_count:
        raise RuntimeError(
            "ProPainter did not save the expected number of frames: "
            f"{len(frame_paths)} != {frame_count}"
        )
    requested_width = int(resize_ratio * width)
    requested_height = int(resize_ratio * height)
    if requested_width <= 0 or requested_height <= 0:
        raise ValueError(
            "resize_ratio makes the ProPainter backend geometry empty: "
            f"{requested_width}x{requested_height}"
        )
    encoded_width = _macroblock_dimension(requested_width)
    encoded_height = _macroblock_dimension(requested_height)
    geometry = probe_video(result_video)
    actual = (
        int(geometry["frame_count"]),
        int(geometry["width"]),
        int(geometry["height"]),
    )
    expected = (frame_count, encoded_width, encoded_height)
    if actual != expected:
        raise RuntimeError(
            f"ProPainter encoded result geometry is {actual}; expected {expected}"
        )
    if not np.isclose(float(geometry["fps"]), fps, rtol=0.0, atol=1e-3):
        raise RuntimeError(
            f"ProPainter result FPS is {geometry['fps']}; expected {fps}"
        )
    _validate_decoded_video(
        result_video,
        frame_count=frame_count,
        width=encoded_width,
        height=encoded_height,
    )
    return frame_paths, {
        "frame_count": frame_count,
        "fps": fps,
        "requested_frame": {
            "width": requested_width,
            "height": requested_height,
        },
        "encoded_video": {
            "width": encoded_width,
            "height": encoded_height,
            "macroblock_size": 16,
        },
    }


def _composite_full_resolution(
    *,
    source_video: Path,
    backend_frames: list[Path],
    mask: np.ndarray,
    destination: Path,
    width: int,
    height: int,
    fps: float,
    backend_width: int,
    backend_height: int,
) -> None:
    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise ValueError(f"Cannot decode video {source_video}")
    writer = Mp4Writer(destination, fps, (width, height))
    try:
        for frame_index, backend_path in enumerate(backend_frames):
            source_ok, source_frame = capture.read()
            if not source_ok:
                raise RuntimeError(
                    f"{source_video} ended at window frame {frame_index}"
                )
            backend_frame = cv2.imread(str(backend_path), cv2.IMREAD_COLOR)
            expected_shape = (backend_height, backend_width, 3)
            if backend_frame is None or backend_frame.shape != expected_shape:
                actual_shape = None if backend_frame is None else backend_frame.shape
                raise RuntimeError(
                    f"ProPainter saved frame {backend_path} has shape "
                    f"{actual_shape}; expected {expected_shape}"
                )
            upsampled = cv2.resize(
                backend_frame,
                (width, height),
                interpolation=cv2.INTER_CUBIC,
            )
            result = source_frame.copy()
            formal_mask = np.asarray(mask[frame_index])
            result[formal_mask] = upsampled[formal_mask]
            writer.write(result)
    finally:
        writer.close()
        capture.release()


def _configuration(
    *,
    source_start_frame: int,
    resize_ratio: float,
    subvideo_length: int,
    neighbor_length: int,
    ref_stride: int,
    fp16: bool,
) -> dict[str, Any]:
    return {
        "backend": "propainter",
        "fp16": fp16,
        "neighbor_length": neighbor_length,
        "ref_stride": ref_stride,
        "resize_ratio": resize_ratio,
        "save_frames": True,
        "source_start_frame": source_start_frame,
        "subvideo_length": subvideo_length,
    }


def source_tree_identity(propainter_dir: str | Path) -> dict[str, Any]:
    """Hash the exact Python source closure imported by ProPainter inference."""
    root = Path(propainter_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    inference_script = root / "inference_propainter.py"
    if not inference_script.is_file():
        raise FileNotFoundError(inference_script)
    candidates = [inference_script]
    for directory_name in PROPAINTER_SOURCE_DIRECTORIES:
        directory = root / directory_name
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        candidates.extend(
            candidate for candidate in directory.rglob("*.py") if candidate.is_file()
        )
    candidates = sorted(
        set(candidates),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    digest = hashlib.sha256()
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        relative = candidate.relative_to(root).as_posix()
        record = artifact(candidate)
        record["relative_path"] = relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["size_bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(record["sha256"].encode("ascii"))
        digest.update(b"\n")
        records.append(record)
    return {
        "root": str(root),
        "tree_sha256": digest.hexdigest(),
        "file_count": len(records),
        "files": records,
    }


def _implementation_records(
    propainter_dir: Path,
    propainter_python: Path,
) -> dict[str, Any]:
    script = propainter_dir / "inference_propainter.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    if not propainter_python.is_file():
        raise FileNotFoundError(propainter_python)
    if not os.access(propainter_python, os.X_OK):
        raise PermissionError(
            f"ProPainter Python is not executable: {propainter_python}"
        )
    weights = {
        name: artifact(propainter_dir / "weights" / name)
        for name in PROPAINTER_WEIGHT_FILENAMES
    }
    return {
        "adapter": artifact(__file__),
        "inference_script": artifact(script),
        "source_tree": source_tree_identity(propainter_dir),
        "python": artifact(propainter_python),
        "weights": weights,
    }


def preflight(
    propainter_dir: str | Path = DEFAULT_PROPAINTER_DIR,
    propainter_python: str | Path = DEFAULT_PROPAINTER_PYTHON,
) -> dict[str, Any]:
    """Validate the complete local ProPainter runtime and return its identity."""
    propainter_root = Path(propainter_dir).expanduser().resolve()
    python_path = Path(propainter_python).expanduser().resolve()
    if not propainter_root.is_dir():
        raise FileNotFoundError(propainter_root)
    return _implementation_records(propainter_root, python_path)


def _hash_implementation(records: dict[str, Any]) -> str:
    return _canonical_sha256(
        {
            "adapter": records["adapter"]["sha256"],
            "inference_script": records["inference_script"]["sha256"],
            "source_tree": records["source_tree"]["tree_sha256"],
            "python": records["python"]["sha256"],
            "weights": {
                name: record["sha256"]
                for name, record in sorted(records["weights"].items())
            },
        }
    )


def execute(
    *,
    source_video: str | Path,
    mask: str | Path,
    output_dir: str | Path,
    source_start_frame: int = 0,
    propainter_dir: str | Path = DEFAULT_PROPAINTER_DIR,
    propainter_python: str | Path = DEFAULT_PROPAINTER_PYTHON,
    resize_ratio: float = 0.5,
    subvideo_length: int = 40,
    neighbor_length: int = 6,
    ref_stride: int = 10,
    fp16: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run ProPainter and atomically publish one validated hand-removed video."""
    _validate_parameters(
        resize_ratio=resize_ratio,
        subvideo_length=subvideo_length,
        neighbor_length=neighbor_length,
        ref_stride=ref_stride,
    )
    if source_start_frame < 0:
        raise ValueError("source_start_frame must be non-negative")
    source_path = Path(source_video).expanduser().resolve()
    mask_path = Path(mask).expanduser().resolve()
    if source_path.suffix.lower() != ".mp4":
        raise ValueError("source_video must be an MP4 exact-window clip")
    source_record = artifact(source_path)
    mask_record = artifact(mask_path)
    source_geometry = probe_video(source_path)
    source_frame_count = int(source_geometry["frame_count"])
    width = int(source_geometry["width"])
    height = int(source_geometry["height"])
    fps = float(source_geometry["fps"])
    mask_array = _load_mask(mask_path)
    frame_count, mask_height, mask_width = map(int, mask_array.shape)
    if (mask_width, mask_height) != (width, height):
        raise ValueError(
            "mask spatial geometry must match source video: "
            f"{(mask_width, mask_height)} != {(width, height)}"
        )
    if source_start_frame + frame_count > source_frame_count:
        raise ValueError(
            "source video does not cover the requested mask window: "
            f"{source_start_frame}+{frame_count}>{source_frame_count}"
        )

    propainter_root = Path(propainter_dir).expanduser().resolve()
    python_path = Path(propainter_python).expanduser().resolve()
    implementation = preflight(propainter_root, python_path)
    configuration = _configuration(
        source_start_frame=source_start_frame,
        resize_ratio=resize_ratio,
        subvideo_length=subvideo_length,
        neighbor_length=neighbor_length,
        ref_stride=ref_stride,
        fp16=fp16,
    )
    config_sha256 = _canonical_sha256(configuration)
    input_sha256 = _canonical_sha256(
        {
            "mask": mask_record["sha256"],
            "source_video": source_record["sha256"],
        }
    )
    implementation_sha256 = _hash_implementation(implementation)
    run_sha256 = _canonical_sha256(
        {
            "config_sha256": config_sha256,
            "implementation_sha256": implementation_sha256,
            "input_sha256": input_sha256,
            "schema_version": PROPAINTER_SCHEMA,
        }
    )

    output = Path(output_dir).expanduser().resolve()
    output_video = output / OUTPUT_FILENAME
    output_metadata = output / METADATA_FILENAME
    existing = [path for path in (output_video, output_metadata) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite: {existing}")
    output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        dir=output,
        prefix=".propainter-",
    ) as temporary_name:
        temporary = Path(temporary_name)
        masks_dir = temporary / "masks"
        results_dir = temporary / "results"
        exact_window = temporary / "source_window.mp4"
        _write_exact_window(
            source_path,
            exact_window,
            source_start_frame=source_start_frame,
            frame_count=frame_count,
            width=width,
            height=height,
            fps=fps,
        )
        _validate_decoded_video(
            exact_window,
            frame_count=frame_count,
            width=width,
            height=height,
        )
        masked_pixels = _materialize_masks(mask_array, masks_dir)
        command = [
            str(python_path),
            "inference_propainter.py",
            "--video",
            str(exact_window),
            "--mask",
            str(masks_dir),
            "--output",
            str(results_dir),
            "--save_frames",
        ]
        if fp16:
            command.append("--fp16")
        command.extend(
            [
                "--resize_ratio",
                f"{resize_ratio:.12g}",
                "--subvideo_length",
                str(subvideo_length),
                "--neighbor_length",
                str(neighbor_length),
                "--ref_stride",
                str(ref_stride),
            ]
        )
        subprocess.run(
            command,
            cwd=propainter_root,
            check=True,
            env=_without_proxy(),
        )
        result_videos = sorted(results_dir.rglob("inpaint_out.mp4"))
        if len(result_videos) != 1:
            raise RuntimeError(
                "ProPainter must produce exactly one inpaint_out.mp4; "
                f"found {len(result_videos)} under {results_dir}"
            )
        result_video = result_videos[0]
        backend_frames, backend_geometry = _validate_backend_outputs(
            result_video,
            frame_count=frame_count,
            width=width,
            height=height,
            fps=fps,
            resize_ratio=resize_ratio,
        )
        staged_video = temporary / OUTPUT_FILENAME
        _composite_full_resolution(
            source_video=exact_window,
            backend_frames=backend_frames,
            mask=mask_array,
            destination=staged_video,
            width=width,
            height=height,
            fps=fps,
            backend_width=int(backend_geometry["requested_frame"]["width"]),
            backend_height=int(backend_geometry["requested_frame"]["height"]),
        )
        _validate_decoded_video(
            staged_video,
            frame_count=frame_count,
            width=width,
            height=height,
        )
        output_record = artifact(staged_video)
        output_record["path"] = str(output_video)
        output_sha256 = output_record["sha256"]
        metadata = {
            "schema_version": PROPAINTER_SCHEMA,
            "state": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "cache_key": run_sha256,
            "geometry": {
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "fps": fps,
            },
            "source_window": {
                "start_frame": source_start_frame,
                "stop_frame_exclusive": source_start_frame + frame_count,
            },
            "backend": {
                "geometry": backend_geometry,
                "inpaint_video_sha256": artifact(result_video)["sha256"],
            },
            "compositing": {
                "candidate_interpolation": "opencv.INTER_CUBIC",
                "candidate_source": "propainter_saved_frames",
                "inside_mask": "upsampled_propainter",
                "outside_mask": "source_video_window",
                "policy": "full_resolution_formal_bool_mask_only",
            },
            "configuration": configuration,
            "statistics": {
                "masked_fraction": masked_pixels / float(frame_count * height * width),
                "masked_pixel_count": masked_pixels,
            },
            "hashes": {
                "config_sha256": config_sha256,
                "implementation_sha256": implementation_sha256,
                "input_sha256": input_sha256,
                "output_sha256": output_sha256,
                "run_sha256": run_sha256,
            },
            "source": {
                "mask": mask_record,
                "source_video": source_record,
                "implementation": implementation,
            },
            "output": {"video": output_record},
        }
        staged_metadata = temporary / METADATA_FILENAME
        write_json_atomic(staged_metadata, metadata)
        os.replace(staged_video, output_video)
        os.replace(staged_metadata, output_metadata)
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-video", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-start-frame", type=int, default=0)
    parser.add_argument(
        "--propainter-dir",
        type=Path,
        default=DEFAULT_PROPAINTER_DIR,
    )
    parser.add_argument(
        "--propainter-python",
        type=Path,
        default=DEFAULT_PROPAINTER_PYTHON,
    )
    parser.add_argument("--resize-ratio", type=float, default=0.5)
    parser.add_argument("--subvideo-length", type=int, default=40)
    parser.add_argument("--neighbor-length", type=int, default=6)
    parser.add_argument("--ref-stride", type=int, default=10)
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    metadata = execute(
        source_video=arguments.source_video,
        mask=arguments.mask,
        output_dir=arguments.output_dir,
        source_start_frame=arguments.source_start_frame,
        propainter_dir=arguments.propainter_dir,
        propainter_python=arguments.propainter_python,
        resize_ratio=arguments.resize_ratio,
        subvideo_length=arguments.subvideo_length,
        neighbor_length=arguments.neighbor_length,
        ref_stride=arguments.ref_stride,
        fp16=arguments.fp16,
        overwrite=arguments.overwrite,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
