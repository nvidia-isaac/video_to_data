#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run target-mesh FoundationPose tracking for recorded-support inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from recorded_support import FOUNDATION_POSE_CAMERA_MODE, file_sha256
from workflow_io import utc_now, write_json


FOUNDATION_POSE_SUPPORT_REPORT_NAME = "foundation_pose_support_report.json"
FOUNDATION_POSE_SUPPORT_POSES_NAME = "poses.npy"
FOUNDATION_POSE_TRACKING_METADATA_NAME = "pose_tracking_metadata.json"
DEFAULT_REGISTRATION_MAX_ATTEMPTS = 6
DEFAULT_REGISTRATION_ATTEMPT_STRIDE = 5
DEFAULT_FOUNDATION_POSE_WEIGHTS_DIR = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "weights"
    / "foundationpose"
)
DEFAULT_FOUNDATION_POSE_CONFIG = (
    Path(__file__).resolve().with_name("foundation_pose_hoi_h5_paths.yaml")
)
FOUNDATION_POSE_SEQUENCE_REQUIRED_PATHS = (
    "edex",
    "images",
    "depth",
    "object_masks",
    "ground_plane.json",
)


class RetryableFoundationPoseFailure(RuntimeError):
    """A registration or tracking outcome that may improve on a later frame."""


def _registration_failure_metadata(
    metadata_path: Path,
    *,
    source_frame: int,
) -> dict[str, object] | None:
    """Return a matching registration-failure report, if one was produced."""

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema_version") != 1
        or metadata.get("status") != "failed"
        or metadata.get("source_frame_start") != source_frame
        or metadata.get("source_frame_end_exclusive") != source_frame
        or metadata.get("pose_count") != 0
        or not isinstance(metadata.get("error"), str)
        or not metadata["error"]
    ):
        return None
    return metadata


def _short_tracking_metadata(
    metadata_path: Path,
    *,
    source_frame: int,
    requested_frame_end_exclusive: int,
    pose_count: int | None,
) -> dict[str, object] | None:
    """Return a valid completed-prefix report that is shorter than requested."""

    if pose_count is None:
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(metadata, dict):
        return None
    frame_end = metadata.get("source_frame_end_exclusive")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("status") != "completed"
        or metadata.get("source_frame_start") != source_frame
        or type(frame_end) is not int
        or not source_frame < frame_end < requested_frame_end_exclusive
        or metadata.get("pose_count") != pose_count
        or frame_end - source_frame != pose_count
    ):
        return None
    return metadata


def _registration_candidate_frames(
    frame_end_exclusive: int,
    *,
    max_attempts: int,
    stride: int,
    minimum_output_frames: int,
) -> list[int]:
    """Return source frames with enough room for a usable pose window."""

    last_start = frame_end_exclusive - minimum_output_frames
    return [
        source_frame
        for source_frame in range(0, frame_end_exclusive, stride)
        if source_frame <= last_start
    ][:max_attempts]


def resolve_foundation_pose_weights_dir(
    weights_dir: str | Path | None,
) -> Path:
    """Resolve an override or the repository-standard weights directory."""

    return Path(
        weights_dir or DEFAULT_FOUNDATION_POSE_WEIGHTS_DIR
    ).expanduser().resolve()


def foundation_pose_support_inputs(
    sequence_dir: str | Path,
    target_mesh_path: str | Path,
    weights_dir: str | Path | None,
    *,
    config_path: str | Path | None = None,
) -> dict[str, Path]:
    """Validate and resolve every input required for target-mesh tracking."""

    sequence = Path(sequence_dir).expanduser().resolve()
    target_mesh = Path(target_mesh_path).expanduser().resolve()
    weights = resolve_foundation_pose_weights_dir(weights_dir)
    missing = [
        sequence / relative
        for relative in FOUNDATION_POSE_SEQUENCE_REQUIRED_PATHS
        if not (sequence / relative).exists()
    ]
    if missing:
        raise ValueError(
            "FoundationPose support input is incomplete; missing: "
            + ", ".join(str(path) for path in missing)
        )
    if not target_mesh.is_file():
        raise ValueError(f"missing FoundationPose target mesh: {target_mesh}")
    symmetry_path = target_mesh.with_name("output_symmetry.json")
    if not symmetry_path.is_file():
        raise ValueError(
            "FoundationPose target mesh requires alignment metadata: "
            f"{symmetry_path}"
        )
    if not weights.is_dir():
        raise ValueError(
            f"missing FoundationPose weights directory: {weights}\n"
            "Download them first:\n"
            "  python "
            "modules/v2d_foundation_pose/docker/run_download_weights.py "
            f"--output_dir {weights}"
        )
    config = Path(
        config_path or DEFAULT_FOUNDATION_POSE_CONFIG
    ).expanduser().resolve()
    if not config.is_file():
        raise ValueError(f"missing FoundationPose config: {config}")
    return {
        "sequence": sequence,
        "target_mesh": target_mesh,
        "target_symmetry": symmetry_path,
        "weights": weights,
        "config": config,
    }


def _load_pose_array(
    path: Path,
) -> np.ndarray:
    if not path.is_file():
        raise RuntimeError(f"FoundationPose did not create {path}")
    poses = np.load(path, allow_pickle=False)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise RuntimeError(
            f"FoundationPose poses must have shape (N, 4, 4), got {poses.shape}"
        )
    if len(poses) == 0:
        raise RuntimeError("FoundationPose output must contain at least one pose")
    if not np.all(np.isfinite(poses)):
        raise RuntimeError("FoundationPose output contains non-finite poses")
    return poses


def _load_tracking_metadata(
    path: Path,
    *,
    pose_count: int,
    requested_frame_start: int,
    requested_frame_end_exclusive: int,
    allow_shorter_prefix: bool,
) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"FoundationPose did not create {path}")
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"invalid FoundationPose tracking metadata: {path}"
        ) from error
    if metadata.get("schema_version") != 1:
        raise RuntimeError(
            "FoundationPose tracking metadata has an unsupported schema"
        )
    if metadata.get("status") != "completed":
        raise RuntimeError("FoundationPose tracking metadata is not completed")

    frame_start = metadata.get("source_frame_start")
    frame_end = metadata.get("source_frame_end_exclusive")
    if (
        type(frame_start) is not int
        or type(frame_end) is not int
        or not 0 <= frame_start < frame_end
    ):
        raise RuntimeError(
            "FoundationPose tracking metadata contains an invalid source "
            "frame range"
        )
    if frame_end - frame_start != pose_count:
        raise RuntimeError(
            "FoundationPose pose count does not match its source frame range "
            f"(range=[{frame_start}, {frame_end}), poses={pose_count})"
        )
    if frame_start != requested_frame_start:
        raise RuntimeError(
            "FoundationPose output starts at the wrong source frame "
            f"({frame_start} != {requested_frame_start})"
        )
    if metadata.get("pose_count") != pose_count:
        raise RuntimeError(
            "FoundationPose tracking metadata pose_count does not match "
            f"poses.npy ({metadata.get('pose_count')} != {pose_count})"
        )
    valid_end = (
        frame_end <= requested_frame_end_exclusive
        if allow_shorter_prefix
        else frame_end == requested_frame_end_exclusive
    )
    if not valid_end:
        expectation = (
            f"end at or before frame {requested_frame_end_exclusive}"
            if allow_shorter_prefix
            else f"end exactly at frame {requested_frame_end_exclusive}"
        )
        raise RuntimeError(
            f"FoundationPose output must {expectation}; got {frame_end}"
        )
    metadata["camera_provenance"] = _camera_provenance_from_tracking_metadata(
        metadata,
        pose_count=pose_count,
    )
    return metadata


def _camera_provenance_from_tracking_metadata(
    metadata: dict,
    *,
    pose_count: int,
) -> dict[str, object]:
    camera_names = metadata.get("camera_names")
    if (
        not isinstance(camera_names, list)
        or not camera_names
        or any(not isinstance(name, str) or not name for name in camera_names)
        or len(set(camera_names)) != len(camera_names)
    ):
        raise RuntimeError(
            "FoundationPose tracking metadata contains invalid camera_names"
        )

    registration_camera_names = metadata.get("registration_camera_names")
    if (
        not isinstance(registration_camera_names, list)
        or not registration_camera_names
        or any(
            not isinstance(name, str) or name not in camera_names
            for name in registration_camera_names
        )
        or len(set(registration_camera_names))
        != len(registration_camera_names)
    ):
        raise RuntimeError(
            "FoundationPose tracking metadata contains invalid "
            "registration_camera_names"
        )

    highest_visibility_camera = metadata.get(
        "highest_visibility_registration_camera"
    )
    if highest_visibility_camera not in registration_camera_names:
        raise RuntimeError(
            "FoundationPose tracking metadata contains an invalid "
            "highest-visibility registration camera"
        )

    visible_ratios = metadata.get("registration_visible_ratios")
    if (
        not isinstance(visible_ratios, dict)
        or set(visible_ratios) != set(camera_names)
        or any(
            type(value) not in (int, float)
            or not np.isfinite(value)
            or not 0.0 <= float(value) <= 1.0
            for value in visible_ratios.values()
        )
    ):
        raise RuntimeError(
            "FoundationPose tracking metadata contains invalid registration "
            "visible ratios"
        )

    frame_counts = metadata.get("tracking_camera_frame_counts")
    if (
        not isinstance(frame_counts, dict)
        or set(frame_counts) != set(camera_names)
        or any(
            type(value) is not int or not 0 <= value <= pose_count
            for value in frame_counts.values()
        )
        or sum(frame_counts.values()) < pose_count
    ):
        raise RuntimeError(
            "FoundationPose tracking metadata contains invalid tracking "
            "camera frame counts"
        )

    return {
        "mode": FOUNDATION_POSE_CAMERA_MODE,
        "camera_names": camera_names,
        "registration_frame": metadata["source_frame_start"],
        "registration_camera_names": registration_camera_names,
        "highest_visibility_registration_camera": highest_visibility_camera,
        "registration_visible_ratios": {
            name: float(visible_ratios[name]) for name in camera_names
        },
        "tracking_camera_frame_counts": {
            name: frame_counts[name] for name in camera_names
        },
    }


def _default_runner(**kwargs) -> None:
    from v2d.foundation_pose.docker.run_mv_videos_to_poses import (
        run_mv_videos_to_poses,
    )

    run_mv_videos_to_poses(**kwargs)


def run_foundation_pose_support(
    sequence_dir: str,
    target_mesh_path: str,
    weights_dir: str | None,
    output_dir: str,
    *,
    frame_end_exclusive: int,
    allow_shorter_prefix: bool = False,
    config_path: str | None = None,
    debug: int = 0,
    dev: bool = False,
    registration_max_attempts: int = DEFAULT_REGISTRATION_MAX_ATTEMPTS,
    registration_attempt_stride: int = DEFAULT_REGISTRATION_ATTEMPT_STRIDE,
    minimum_output_frames: int = 1,
    runner: Callable[..., None] | None = None,
) -> dict[str, object]:
    """Retry bounded source slices and publish strict pose provenance."""

    if frame_end_exclusive <= 0:
        raise ValueError("FoundationPose frame_end_exclusive must be positive")
    if debug not in {0, 1, 2}:
        raise ValueError("FoundationPose debug must be 0, 1, or 2")
    if registration_max_attempts <= 0:
        raise ValueError("registration_max_attempts must be positive")
    if registration_attempt_stride <= 0:
        raise ValueError("registration_attempt_stride must be positive")
    if minimum_output_frames <= 0:
        raise ValueError("minimum_output_frames must be positive")
    if minimum_output_frames > frame_end_exclusive:
        raise ValueError(
            "minimum_output_frames cannot exceed "
            "frame_end_exclusive"
        )
    registration_candidates = _registration_candidate_frames(
        frame_end_exclusive,
        max_attempts=registration_max_attempts,
        stride=registration_attempt_stride,
        minimum_output_frames=minimum_output_frames,
    )
    inputs = foundation_pose_support_inputs(
        sequence_dir,
        target_mesh_path,
        weights_dir,
        config_path=config_path,
    )
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    poses_path = output / FOUNDATION_POSE_SUPPORT_POSES_NAME
    report_path = output / FOUNDATION_POSE_SUPPORT_REPORT_NAME
    tracking_metadata_path = (
        output / FOUNDATION_POSE_TRACKING_METADATA_NAME
    )
    poses_path.unlink(missing_ok=True)
    report_path.unlink(missing_ok=True)
    tracking_metadata_path.unlink(missing_ok=True)

    report = {
        "schema_version": 1,
        "status": "running",
        "started_at": utc_now(),
        "sequence": str(inputs["sequence"]),
        "sequence_id": inputs["sequence"].name,
        "target_mesh": str(inputs["target_mesh"]),
        "target_mesh_file_sha256": file_sha256(inputs["target_mesh"]),
        "target_symmetry": str(inputs["target_symmetry"]),
        "target_symmetry_file_sha256": file_sha256(
            inputs["target_symmetry"]
        ),
        "weights_dir": str(inputs["weights"]),
        "config_path": str(inputs["config"]),
        "requested_frame_end_exclusive": frame_end_exclusive,
        "frame_start": None,
        "frame_end_exclusive": None,
        "allow_shorter_prefix": allow_shorter_prefix,
        "registration_max_attempts": registration_max_attempts,
        "registration_attempt_stride": registration_attempt_stride,
        "minimum_output_frames": minimum_output_frames,
        "registration_candidates": registration_candidates,
        "registration_attempts": [],
        "output_poses": str(poses_path),
        "tracking_metadata": str(tracking_metadata_path),
        "camera_provenance": None,
        "report_file": str(report_path),
    }
    write_json(report_path, report)

    invoke = runner or _default_runner
    poses = None
    tracking_metadata = None
    last_error = None
    for candidate in registration_candidates:
        poses_path.unlink(missing_ok=True)
        tracking_metadata_path.unlink(missing_ok=True)
        attempt = {
            "source_frame": candidate,
            "status": "running",
            "started_at": utc_now(),
        }
        report["registration_attempts"].append(attempt)
        write_json(report_path, report)
        try:
            invoke(
                camera_params_path=str(inputs["sequence"] / "edex"),
                rgb_dir=str(inputs["sequence"] / "images"),
                depth_dir=str(inputs["sequence"] / "depth"),
                mask_dir=str(inputs["sequence"] / "object_masks"),
                mesh_path=str(inputs["target_mesh"]),
                symmetry_path=str(inputs["target_symmetry"]),
                weights_dir=str(inputs["weights"]),
                output_dir=str(output),
                config_path=str(inputs["config"]),
                frame_start=candidate,
                frame_end_exclusive=frame_end_exclusive,
                clamp_frame_end_exclusive=allow_shorter_prefix,
                debug=debug,
                dev=dev,
            )
            poses = _load_pose_array(poses_path)
            tracking_metadata = _load_tracking_metadata(
                tracking_metadata_path,
                pose_count=len(poses),
                requested_frame_start=candidate,
                requested_frame_end_exclusive=frame_end_exclusive,
                allow_shorter_prefix=allow_shorter_prefix,
            )
            if len(poses) < minimum_output_frames:
                raise RetryableFoundationPoseFailure(
                    "FoundationPose produced too few valid poses for support "
                    f"inference ({len(poses)} < {minimum_output_frames})"
                )
        except Exception as error:
            last_error = error
            failure_metadata = _registration_failure_metadata(
                tracking_metadata_path,
                source_frame=candidate,
            )
            short_tracking_metadata = _short_tracking_metadata(
                tracking_metadata_path,
                source_frame=candidate,
                requested_frame_end_exclusive=frame_end_exclusive,
                pose_count=len(poses) if poses is not None else None,
            )
            retryable = (
                isinstance(error, RetryableFoundationPoseFailure)
                or failure_metadata is not None
                or short_tracking_metadata is not None
            )
            attempt.update(
                {
                    "status": "failed",
                    "completed_at": utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "retryable": retryable,
                }
            )
            if failure_metadata is not None:
                attempt["registration_error"] = failure_metadata["error"]
            write_json(report_path, report)
            poses = None
            tracking_metadata = None
            if retryable:
                continue
            poses_path.unlink(missing_ok=True)
            report["status"] = "failed"
            report["completed_at"] = utc_now()
            report["error_type"] = type(error).__name__
            report["error"] = str(error)
            write_json(report_path, report)
            raise

        attempt.update(
            {
                "status": "completed",
                "completed_at": utc_now(),
                "source_frame_end_exclusive": tracking_metadata[
                    "source_frame_end_exclusive"
                ],
                "pose_count": int(len(poses)),
            }
        )
        write_json(report_path, report)
        break

    if poses is None or tracking_metadata is None:
        poses_path.unlink(missing_ok=True)
        tracking_metadata_path.unlink(missing_ok=True)
        report["status"] = "failed"
        report["completed_at"] = utc_now()
        report["error_type"] = "RuntimeError"
        report["error"] = (
            "FoundationPose registration/tracking failed on all eligible "
            f"source frames: {registration_candidates}"
        )
        write_json(report_path, report)
        raise RuntimeError(report["error"]) from last_error

    report["status"] = "completed"
    report["completed_at"] = utc_now()
    report["pose_count"] = int(len(poses))
    report["frame_start"] = tracking_metadata["source_frame_start"]
    report["frame_end_exclusive"] = tracking_metadata[
        "source_frame_end_exclusive"
    ]
    report["camera_provenance"] = tracking_metadata["camera_provenance"]
    report["poses_file_sha256"] = file_sha256(poses_path)
    report["tracking_metadata_file_sha256"] = file_sha256(
        tracking_metadata_path
    )
    write_json(report_path, report)
    return report
