# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only validation and deterministic run-description helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import cv2
import numpy as np

E2FGVI_REPOSITORY = "https://github.com/MarionLepert/phantom-E2FGVI.git"
E2FGVI_COMMIT = "5b45ffe400288006facb350e00d319bfc6c5cbd3"
METADATA_SCHEMA = "v2d.e2fgvi.inpainting.v1"


class InputValidationError(ValueError):
    """Raised when an inpainting input or option violates the stage contract."""


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    frame_count: int
    fps: float

    def as_metadata(self) -> dict[str, int | float]:
        return {
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
            "fps": round(self.fps, 8),
        }


@dataclass(frozen=True)
class InferenceConfig:
    downscale: float = 1.0
    max_size: int = 0
    dilation_iterations: int = 4
    dilation_kernel: int = 3
    neighbor_stride: int = 5
    ref_stride: int = 20
    num_ref: int = -1
    device: str = "cuda:0"
    codec: str = "mp4v"
    seed: int = 0


@dataclass(frozen=True)
class RunPlan:
    video: VideoInfo
    mask_shape: tuple[int, int, int]
    processing_width: int
    processing_height: int
    config: InferenceConfig

    def as_metadata(self) -> dict[str, Any]:
        return {
            "source_video": self.video.as_metadata(),
            "masks": {
                "dtype": "bool",
                "shape": list(self.mask_shape),
                "true_means": "inpaint",
            },
            "processing_resolution": {
                "width": self.processing_width,
                "height": self.processing_height,
            },
            "parameters": asdict(self.config),
        }


def validate_config(config: InferenceConfig) -> None:
    """Validate all pure inference options."""
    if not math.isfinite(config.downscale) or config.downscale < 1.0:
        raise InputValidationError("downscale must be a finite value >= 1.0")
    if config.max_size < 0:
        raise InputValidationError("max_size must be 0 (disabled) or a positive integer")
    if config.dilation_iterations < 0:
        raise InputValidationError("dilation_iterations must be >= 0")
    if config.dilation_kernel <= 0 or config.dilation_kernel % 2 == 0:
        raise InputValidationError("dilation_kernel must be a positive odd integer")
    if config.neighbor_stride <= 0:
        raise InputValidationError("neighbor_stride must be > 0")
    if config.ref_stride <= 0:
        raise InputValidationError("ref_stride must be > 0")
    if config.num_ref != -1 and config.num_ref <= 0:
        raise InputValidationError("num_ref must be -1 (all) or a positive integer")
    if not re.fullmatch(r"(?:cpu|cuda(?::\d+)?)", config.device):
        raise InputValidationError("device must be 'cpu', 'cuda', or 'cuda:<index>'")
    if len(config.codec) != 4 or any(not 32 <= ord(character) <= 126 for character in config.codec):
        raise InputValidationError("codec must contain four printable ASCII characters")
    if config.seed < 0:
        raise InputValidationError("seed must be >= 0")


def compute_processing_size(
    width: int,
    height: int,
    downscale: float = 1.0,
    max_size: int = 0,
) -> tuple[int, int]:
    """Compute an aspect-preserving processing size.

    ``downscale=2`` halves each dimension. A non-zero ``max_size`` then caps
    the longest edge.
    """
    if width <= 0 or height <= 0:
        raise InputValidationError("source width and height must be positive")
    if not math.isfinite(downscale) or downscale < 1.0:
        raise InputValidationError("downscale must be a finite value >= 1.0")
    if max_size < 0:
        raise InputValidationError("max_size must be >= 0")

    target_width = max(1, int(math.floor(width / downscale + 0.5)))
    target_height = max(1, int(math.floor(height / downscale + 0.5)))
    longest = max(target_width, target_height)
    if max_size and longest > max_size:
        scale = max_size / longest
        target_width = max(1, int(math.floor(target_width * scale + 0.5)))
        target_height = max(1, int(math.floor(target_height * scale + 0.5)))
    return target_width, target_height


def validate_mask_array(masks: np.ndarray, video: VideoInfo) -> tuple[int, int, int]:
    """Validate the strict ``[frames, height, width]`` boolean mask contract."""
    if masks.dtype != np.dtype(np.bool_):
        raise InputValidationError(
            f"masks must have boolean dtype, got {masks.dtype}; do not pass 0/1 integers"
        )
    if masks.ndim != 3:
        raise InputValidationError(
            f"masks must have shape [frames, height, width], got {masks.shape}"
        )
    expected = (video.frame_count, video.height, video.width)
    actual = tuple(int(value) for value in masks.shape)
    if actual[0] != expected[0]:
        raise InputValidationError(
            f"mask frame count {actual[0]} does not match decoded video frame count {expected[0]}"
        )
    if actual[1:] != expected[1:]:
        raise InputValidationError(
            "mask resolution "
            f"{actual[2]}x{actual[1]} does not match video resolution {expected[2]}x{expected[1]}"
        )
    return actual


def probe_video(path: str | os.PathLike[str]) -> VideoInfo:
    """Decode a video on CPU to obtain an exact frame count and stable geometry."""
    video_path = Path(path)
    if not video_path.is_file():
        raise InputValidationError(f"input video does not exist: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise InputValidationError(f"could not open input video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise InputValidationError(f"input video reports invalid FPS: {fps}")

    frame_count = 0
    width = height = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            current_height, current_width = frame.shape[:2]
            if frame_count == 0:
                width, height = current_width, current_height
            elif (current_width, current_height) != (width, height):
                raise InputValidationError(
                    "input video changes resolution at frame "
                    f"{frame_count}: {current_width}x{current_height} vs {width}x{height}"
                )
            frame_count += 1
    finally:
        capture.release()

    if frame_count == 0:
        raise InputValidationError(f"input video contains no decodable frames: {video_path}")
    return VideoInfo(width=width, height=height, frame_count=frame_count, fps=fps)


def validate_run(
    input_video: str | os.PathLike[str],
    masks_path: str | os.PathLike[str],
    checkpoint_path: str | os.PathLike[str],
    config: InferenceConfig,
) -> RunPlan:
    """Validate files and options without importing PyTorch or E2FGVI."""
    validate_config(config)
    mask_file = Path(masks_path)
    checkpoint_file = Path(checkpoint_path)
    if mask_file.suffix.lower() != ".npy" or not mask_file.is_file():
        raise InputValidationError(f"masks must be an existing .npy file: {mask_file}")
    if not checkpoint_file.is_file():
        raise InputValidationError(f"checkpoint does not exist: {checkpoint_file}")
    if checkpoint_file.stat().st_size == 0:
        raise InputValidationError(f"checkpoint is empty: {checkpoint_file}")

    video = probe_video(input_video)
    if video.frame_count < 2:
        raise InputValidationError("E2FGVI requires a video with at least two decoded frames")
    try:
        masks = np.load(mask_file, allow_pickle=False, mmap_mode="r")
    except (OSError, ValueError) as exc:
        raise InputValidationError(f"could not load masks from {mask_file}: {exc}") from exc
    shape = validate_mask_array(masks, video)
    processing_width, processing_height = compute_processing_size(
        video.width,
        video.height,
        config.downscale,
        config.max_size,
    )
    return RunPlan(
        video=video,
        mask_shape=shape,
        processing_width=processing_width,
        processing_height=processing_height,
        config=config,
    )


def validate_output_paths(
    input_video: str | os.PathLike[str],
    masks_path: str | os.PathLike[str],
    checkpoint_path: str | os.PathLike[str],
    output_video: str | os.PathLike[str],
    metadata_path: str | os.PathLike[str],
) -> None:
    """Prevent an output or sidecar from overwriting an input or each other."""
    inputs = {
        Path(input_video).resolve(),
        Path(masks_path).resolve(),
        Path(checkpoint_path).resolve(),
    }
    output = Path(output_video).resolve()
    metadata = Path(metadata_path).resolve()
    if output in inputs:
        raise InputValidationError("output_video must be distinct from every input file")
    if metadata in inputs or metadata == output:
        raise InputValidationError(
            "metadata_path must be distinct from the input files and output_video"
        )


def select_reference_indices(
    center: int,
    neighbor_indices: Sequence[int],
    frame_count: int,
    ref_stride: int,
    num_ref: int,
) -> list[int]:
    """Select deterministic global references, preferring frames near the center."""
    if not 0 <= center < frame_count:
        raise InputValidationError("center must identify a frame in the video")
    if ref_stride <= 0:
        raise InputValidationError("ref_stride must be > 0")
    if num_ref != -1 and num_ref <= 0:
        raise InputValidationError("num_ref must be -1 or positive")
    neighbors = set(neighbor_indices)
    candidates = [index for index in range(0, frame_count, ref_stride) if index not in neighbors]
    if num_ref != -1:
        candidates = sorted(candidates, key=lambda index: (abs(index - center), index))[:num_ref]
        candidates.sort()
    return candidates


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_identity(stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _file_record(path: str | os.PathLike[str]) -> dict[str, str | int]:
    file_path = Path(path)
    before = file_path.stat()
    digest = sha256_file(file_path)
    after = file_path.stat()
    if _stat_identity(before) != _stat_identity(after):
        raise InputValidationError(f"file changed while it was being fingerprinted: {file_path}")
    return {
        "name": file_path.name,
        "bytes": after.st_size,
        "sha256": digest,
    }


def _video_matches_record(video: VideoInfo, record: dict[str, Any]) -> bool:
    try:
        return (
            video.width == int(record["width"])
            and video.height == int(record["height"])
            and video.frame_count == int(record["frame_count"])
            and math.isclose(video.fps, float(record["fps"]), rel_tol=1e-4, abs_tol=1e-3)
        )
    except (KeyError, TypeError, ValueError):
        return False


def _completed_output_record(
    output_video: str | os.PathLike[str], expected: VideoInfo
) -> dict[str, str | int | float]:
    output_path = Path(output_video)
    before = output_path.stat()
    decoded = probe_video(output_path)
    if not _video_matches_record(decoded, expected.as_metadata()):
        raise InputValidationError(
            "completed output geometry/FPS does not match the validated input video: "
            f"got {decoded.width}x{decoded.height}/{decoded.frame_count} at {decoded.fps}, "
            f"expected {expected.width}x{expected.height}/{expected.frame_count} at {expected.fps}"
        )
    record: dict[str, str | int | float] = {
        **expected.as_metadata(),
        **_file_record(output_path),
    }
    after = output_path.stat()
    if _stat_identity(before) != _stat_identity(after):
        raise InputValidationError(
            f"completed output changed while it was being validated: {output_path}"
        )
    return record


def _implementation_record(
    container_image: str | None,
    container_image_id: str | None,
) -> dict[str, str]:
    implementation = {
        "model": "E2FGVI-HQ",
        "repository": E2FGVI_REPOSITORY,
        "commit": E2FGVI_COMMIT,
    }
    if container_image is None and container_image_id is None:
        return implementation
    if not isinstance(container_image, str) or not container_image.strip():
        raise InputValidationError(
            "container_image must be non-empty when container_image_id is recorded"
        )
    if not isinstance(container_image_id, str) or re.fullmatch(
        r"sha256:[0-9a-f]{64}", container_image_id
    ) is None:
        raise InputValidationError(
            "container_image_id must be sha256: followed by 64 lowercase hex digits"
        )
    implementation.update(
        {
            "container_image": container_image,
            "container_image_id": container_image_id,
            "container_image_provenance": "recorded_immutable_id",
        }
    )
    return implementation


def build_metadata(
    plan: RunPlan,
    input_video: str | os.PathLike[str],
    masks_path: str | os.PathLike[str],
    checkpoint_path: str | os.PathLike[str],
    output_video: str | os.PathLike[str],
    status: Literal["validated", "committing", "completed"],
    *,
    container_image: str | None = None,
    container_image_id: str | None = None,
) -> dict[str, Any]:
    """Build sidecar content with no clocks, durations, or machine-specific paths."""
    output: dict[str, Any] = {
        "name": Path(output_video).name,
        **plan.video.as_metadata(),
    }
    if status == "completed":
        output = _completed_output_record(output_video, plan.video)
    return {
        "schema": METADATA_SCHEMA,
        "status": status,
        "implementation": _implementation_record(container_image, container_image_id),
        "inputs": {
            "video": _file_record(input_video),
            "masks": _file_record(masks_path),
            "checkpoint": _file_record(checkpoint_path),
        },
        "output": output,
        "run": plan.as_metadata(),
    }


def enrich_completed_metadata(
    metadata_path: str | os.PathLike[str],
    input_video: str | os.PathLike[str],
    masks_path: str | os.PathLike[str],
    checkpoint_path: str | os.PathLike[str],
    output_video: str | os.PathLike[str],
) -> dict[str, Any]:
    """Safely add an output fingerprint to a legacy completed sidecar.

    This does not pretend to recover an immutable container ID that the
    original run did not record.  Instead, it marks that provenance gap
    explicitly after revalidating all recorded inputs and decoded output
    geometry.
    """

    sidecar = Path(metadata_path)
    try:
        report = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputValidationError(f"could not read legacy metadata {sidecar}: {exc}") from exc
    if not isinstance(report, dict):
        raise InputValidationError(f"legacy metadata must contain a JSON object: {sidecar}")
    if report.get("schema") != METADATA_SCHEMA or report.get("status") != "completed":
        raise InputValidationError(
            "only completed v2d.e2fgvi.inpainting.v1 metadata can be enriched"
        )

    expected_inputs = {
        "video": _file_record(input_video),
        "masks": _file_record(masks_path),
        "checkpoint": _file_record(checkpoint_path),
    }
    if report.get("inputs") != expected_inputs:
        raise InputValidationError(
            "current video, masks, or checkpoint do not match the recorded legacy inputs"
        )

    output = report.get("output")
    if not isinstance(output, dict) or output.get("name") != Path(output_video).name:
        raise InputValidationError("legacy metadata output name does not match output_video")
    decoded = probe_video(output_video)
    if not _video_matches_record(decoded, output):
        raise InputValidationError(
            "legacy output decoded geometry/FPS does not match its metadata"
        )
    output_fingerprint = _file_record(output_video)

    # Recheck the source artifacts after probing/hashing the output so a
    # concurrent input replacement cannot be committed as a valid migration.
    if expected_inputs != {
        "video": _file_record(input_video),
        "masks": _file_record(masks_path),
        "checkpoint": _file_record(checkpoint_path),
    }:
        raise InputValidationError("an E2FGVI input changed while metadata was enriched")

    enriched = dict(report)
    enriched_output = dict(output)
    enriched_output.update(output_fingerprint)
    enriched["output"] = enriched_output
    implementation = report.get("implementation")
    if not isinstance(implementation, dict):
        raise InputValidationError("legacy metadata implementation must be an object")
    enriched_implementation = dict(implementation)
    existing_image_id = enriched_implementation.get("container_image_id")
    if existing_image_id is None:
        enriched_implementation["container_image_provenance"] = "legacy_unrecorded"
    elif not isinstance(existing_image_id, str) or re.fullmatch(
        r"sha256:[0-9a-f]{64}", existing_image_id
    ) is None:
        raise InputValidationError("legacy metadata contains an invalid container image ID")
    enriched["implementation"] = enriched_implementation
    write_metadata(sidecar, enriched)
    return enriched


def canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_metadata(path: str | os.PathLike[str], data: dict[str, Any]) -> None:
    """Atomically write canonical JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(canonical_json(data), encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
