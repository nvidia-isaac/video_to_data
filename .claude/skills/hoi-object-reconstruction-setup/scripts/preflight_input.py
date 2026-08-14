#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the calibrated-stereo input contract for HOI reconstruction."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXPECTED_SENSORS = {
    "left": "front_stereo_camera_left",
    "right": "front_stereo_camera_right",
}
MAX_REPORTED_ERRORS = 20


def _camera_params(cameras: dict[str, Any], camera_id: Any) -> dict[str, Any] | None:
    value = cameras.get(str(camera_id))
    return value if isinstance(value, dict) else None


def _as_positive_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _inside(root: Path, candidate: Path) -> bool:
    try:
        return os.path.commonpath((str(root), str(candidate))) == str(root)
    except ValueError:
        return False


def validate(mapping_data_dir: Path) -> tuple[list[str], list[str], dict[str, int]]:
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict[str, int] = {}

    def error(message: str) -> None:
        if len(errors) < MAX_REPORTED_ERRORS:
            errors.append(message)

    root = mapping_data_dir.expanduser().resolve()
    if not root.is_dir():
        return [f"mapping_data_dir is not a directory: {root}"], warnings, stats

    metadata_path = root / "frames_meta.json"
    if not metadata_path.is_file():
        return [f"missing required file: {metadata_path}"], warnings, stats

    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"cannot parse {metadata_path}: {exc}"], warnings, stats

    if not isinstance(metadata, dict):
        return ["frames_meta.json must contain a JSON object"], warnings, stats

    required_keys = (
        "keyframes_metadata",
        "camera_params_id_to_camera_params",
        "stereo_pair",
        "initial_pose_type",
    )
    for key in required_keys:
        if key not in metadata:
            error(f"frames_meta.json is missing required key: {key}")

    frames = metadata.get("keyframes_metadata")
    cameras = metadata.get("camera_params_id_to_camera_params")
    stereo_pairs = metadata.get("stereo_pair")
    if not isinstance(frames, list) or not frames:
        error("keyframes_metadata must be a nonempty list")
        frames = []
    if not isinstance(cameras, dict) or not cameras:
        error("camera_params_id_to_camera_params must be a nonempty object")
        cameras = {}
    if not isinstance(stereo_pairs, list) or not stereo_pairs:
        error("stereo_pair must be a nonempty list")
        stereo_pairs = []
    if not metadata.get("initial_pose_type"):
        error("initial_pose_type must be nonempty")

    pair = stereo_pairs[0] if stereo_pairs and isinstance(stereo_pairs[0], dict) else {}
    left_id = pair.get("left_camera_param_id")
    right_id = pair.get("right_camera_param_id")
    if left_id is None or right_id is None:
        error("the first stereo_pair must identify left and right camera IDs")
    if _as_positive_number(pair.get("baseline_meters")) is None:
        error("the first stereo_pair baseline_meters must be positive")

    pair_ids = {"left": left_id, "right": right_id}
    for side, camera_id in pair_ids.items():
        params = _camera_params(cameras, camera_id)
        if params is None:
            error(f"{side} camera ID {camera_id!r} is missing from camera parameters")
            continue

        sensor_metadata = params.get("sensor_meta_data")
        sensor = (
            sensor_metadata.get("sensor_name")
            if isinstance(sensor_metadata, dict)
            else None
        )
        if sensor != EXPECTED_SENSORS[side]:
            error(
                f"{side} camera {camera_id!r} sensor_name is {sensor!r}; "
                f"expected {EXPECTED_SENSORS[side]!r}"
            )

        calibration = params.get("calibration_parameters")
        if not isinstance(calibration, dict):
            error(f"{side} camera {camera_id!r} is missing calibration_parameters")
            continue
        if _as_positive_number(calibration.get("image_width")) is None:
            error(f"{side} camera {camera_id!r} has invalid image_width")
        if _as_positive_number(calibration.get("image_height")) is None:
            error(f"{side} camera {camera_id!r} has invalid image_height")

        projection = calibration.get("projection_matrix")
        if not isinstance(projection, dict):
            error(f"{side} camera {camera_id!r} is missing projection_matrix")
            continue
        if projection.get("row_count") != 3 or projection.get("column_count") != 4:
            error(f"{side} camera {camera_id!r} projection_matrix must be 3 by 4")
        data = projection.get("data")
        if not isinstance(data, list) or len(data) != 12:
            error(f"{side} camera {camera_id!r} projection_matrix.data must have 12 values")
        else:
            try:
                [float(value) for value in data]
            except (TypeError, ValueError):
                error(
                    f"{side} camera {camera_id!r} projection_matrix.data "
                    "must contain numeric values"
                )

    sync_ids: dict[str, Counter[str]] = defaultdict(Counter)
    referenced_images = 0
    non_jpeg_images = 0
    known_camera_ids = {str(key) for key in cameras}
    side_by_camera_id = {
        str(camera_id): side
        for side, camera_id in pair_ids.items()
        if camera_id is not None
    }

    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            error(f"keyframes_metadata[{index}] must be an object")
            continue

        camera_id = str(frame.get("camera_params_id"))
        if camera_id not in known_camera_ids:
            error(
                f"keyframes_metadata[{index}] references unknown camera ID "
                f"{frame.get('camera_params_id')!r}"
            )

        image_name = frame.get("image_name")
        if not isinstance(image_name, str) or not image_name:
            error(f"keyframes_metadata[{index}] has no image_name")
        else:
            relative_path = Path(image_name)
            if relative_path.is_absolute():
                error(f"keyframes_metadata[{index}] image_name must be relative: {image_name}")
            else:
                image_path = (root / relative_path).resolve()
                if not _inside(root, image_path):
                    error(f"keyframes_metadata[{index}] image_name escapes input root: {image_name}")
                elif not image_path.is_file():
                    error(f"referenced image does not exist: {image_name}")
                else:
                    referenced_images += 1
                    if image_path.suffix.lower() not in {".jpg", ".jpeg"}:
                        non_jpeg_images += 1

        timestamp = frame.get("timestamp_microseconds")
        if not isinstance(timestamp, str) or not timestamp.isdecimal():
            error(
                f"keyframes_metadata[{index}] timestamp_microseconds "
                "must be a decimal string"
            )

        side = side_by_camera_id.get(camera_id)
        if side is not None:
            synced_sample_id = frame.get("synced_sample_id")
            if synced_sample_id is None or str(synced_sample_id) == "":
                error(f"keyframes_metadata[{index}] has no synced_sample_id")
            else:
                sync_ids[side][str(synced_sample_id)] += 1

    for side, counts in sync_ids.items():
        duplicates = [sample_id for sample_id, count in counts.items() if count > 1]
        if duplicates:
            error(
                f"{side} camera has duplicate synced_sample_id values; "
                f"first duplicate: {duplicates[0]}"
            )

    paired = set(sync_ids["left"]) & set(sync_ids["right"])
    if not paired:
        error("no synchronized left/right image pairs were found")
    incomplete = (set(sync_ids["left"]) | set(sync_ids["right"])) - paired
    if incomplete:
        warnings.append(
            f"{len(incomplete)} synchronized sample IDs are incomplete and will be dropped"
        )
    if non_jpeg_images:
        warnings.append(
            f"{non_jpeg_images} referenced images are not JPEG; JPEG is the "
            "known-good complete-pipeline format"
        )

    stats["keyframe_records"] = len(frames)
    stats["referenced_images"] = referenced_images
    stats["stereo_pairs"] = len(paired)
    return errors, warnings, stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate an HOI reconstruction mapping-data directory"
    )
    parser.add_argument("mapping_data_dir", type=Path)
    args = parser.parse_args()

    errors, warnings, stats = validate(args.mapping_data_dir)
    for message in warnings:
        print(f"WARNING: {message}", file=sys.stderr)
    for message in errors:
        print(f"ERROR: {message}", file=sys.stderr)

    if errors:
        if len(errors) == MAX_REPORTED_ERRORS:
            print(
                f"ERROR: stopped after {MAX_REPORTED_ERRORS} reported errors",
                file=sys.stderr,
            )
        return 1

    print(
        "input_preflight=pass "
        f"keyframe_records={stats['keyframe_records']} "
        f"referenced_images={stats['referenced_images']} "
        f"stereo_pairs={stats['stereo_pairs']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
