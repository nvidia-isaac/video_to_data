# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Required camera-calibration contract for BundleSDF reconstruction."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_NUMERIC_FIELDS = ("fx", "fy", "cx", "cy", "baseline")


def calibration_candidates(output_path: Path) -> tuple[Path, Path]:
    """Return the legacy-compatible calibration lookup order."""
    return (
        output_path / "calibration.json",
        output_path.parent / "calibration.json",
    )


def _finite_number(calibration: dict[str, Any], field: str) -> float:
    value = calibration.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Camera calibration field {field!r} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Camera calibration field {field!r} must be finite")
    if field in {"fx", "fy", "baseline"} and number <= 0:
        raise ValueError(f"Camera calibration field {field!r} must be positive")
    return number


def _positive_integer(calibration: dict[str, Any], field: str) -> int:
    value = calibration.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Camera calibration field {field!r} must be an integer")
    number = float(value)
    if not math.isfinite(number) or not number.is_integer() or number <= 0:
        raise ValueError(f"Camera calibration field {field!r} must be a positive integer")
    return int(number)


def _first_image_size(images_dir: Path) -> tuple[int, int, Path]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"BundleSDF RGB image directory is missing: {images_dir}")
    images = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"BundleSDF RGB image directory is empty: {images_dir}")
    image = cv2.imread(str(images[0]), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim < 2:
        raise ValueError(f"Cannot read BundleSDF RGB image dimensions: {images[0]}")
    height, width = image.shape[:2]
    return int(width), int(height), images[0]


def load_required_calibration(
    output_path: Path,
    *,
    allow_default_hawk_intrinsics: bool = False,
) -> tuple[Path, dict[str, Any]] | None:
    """Load calibration and verify it describes the actual RGB image size.

    The parent-directory lookup preserves the existing prepared-job layout. The
    embedded Hawk matrix is available only through an explicit compatibility
    opt-in so a missing dataset file cannot silently change reconstruction scale.
    """
    candidates = calibration_candidates(output_path)
    calibration_path = next((path for path in candidates if path.is_file()), None)
    if calibration_path is None:
        if allow_default_hawk_intrinsics:
            actual_width, actual_height, first_image = _first_image_size(
                output_path / "left"
            )
            if (actual_width, actual_height) != (1920, 1200):
                raise ValueError(
                    "Embedded Hawk intrinsics describe 1920x1200 images, but "
                    f"the RGB images are {actual_width}x{actual_height} "
                    f"(checked {first_image})"
                )
            return None
        searched = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            "BundleSDF camera calibration is required; searched "
            f"{searched}. Pass --intrinsics_file or, only for a deliberate "
            "1920x1200 Hawk run, --allow-default-hawk-intrinsics."
        )

    try:
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid camera calibration JSON {calibration_path}: {error}"
        ) from error
    if not isinstance(calibration, dict):
        raise ValueError(f"Camera calibration must be a JSON object: {calibration_path}")

    normalized = dict(calibration)
    for field in _NUMERIC_FIELDS:
        normalized[field] = _finite_number(calibration, field)
    normalized["width"] = _positive_integer(calibration, "width")
    normalized["height"] = _positive_integer(calibration, "height")

    actual_width, actual_height, first_image = _first_image_size(output_path / "left")
    declared_size = (normalized["width"], normalized["height"])
    actual_size = (actual_width, actual_height)
    if declared_size != actual_size:
        raise ValueError(
            f"Camera calibration resolution {declared_size[0]}x{declared_size[1]} "
            f"does not match RGB images {actual_size[0]}x{actual_size[1]} "
            f"(checked {first_image})"
        )
    return calibration_path, normalized


def intrinsic_matrix(calibration: dict[str, Any]) -> list[float]:
    """Convert normalized scalar intrinsics to BundleSDF's flattened matrix."""
    return [
        calibration["fx"],
        0,
        calibration["cx"],
        0,
        calibration["fy"],
        calibration["cy"],
        0,
        0,
        1,
    ]
