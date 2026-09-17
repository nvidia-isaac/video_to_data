# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from v2d_bundlesdf.lib.calibration import (
    intrinsic_matrix,
    load_required_calibration,
)


def _write_image(path, *, width: int = 960, height: int = 600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _calibration(**overrides):
    calibration = {
        "fx": 426.205322265625,
        "fy": 426.205322265625,
        "cx": 473.22314453125,
        "cy": 278.40216064453125,
        "width": 960,
        "height": 600,
        "baseline": 0.1495617627365213,
    }
    calibration.update(overrides)
    return calibration


def test_calibration_is_required_by_default(tmp_path) -> None:
    output = tmp_path / "job" / "stage1_recon"
    _write_image(output / "left" / "000000.jpg")

    with pytest.raises(FileNotFoundError, match="camera calibration is required"):
        load_required_calibration(output)


def test_default_hawk_intrinsics_require_explicit_opt_in(tmp_path) -> None:
    output = tmp_path / "job" / "stage1_recon"
    _write_image(output / "left" / "000000.jpg", width=1920, height=1200)

    assert load_required_calibration(
        output, allow_default_hawk_intrinsics=True
    ) is None


def test_default_hawk_intrinsics_reject_other_image_sizes(tmp_path) -> None:
    output = tmp_path / "job" / "stage1_recon"
    _write_image(output / "left" / "000000.jpg", width=960, height=600)

    with pytest.raises(ValueError, match="1920x1200.*960x600"):
        load_required_calibration(
            output, allow_default_hawk_intrinsics=True
        )


def test_parent_job_calibration_is_loaded_and_normalized(tmp_path) -> None:
    output = tmp_path / "job" / "stage1_recon"
    _write_image(output / "left" / "000000.jpg")
    calibration_path = output.parent / "calibration.json"
    calibration_path.write_text(json.dumps(_calibration()), encoding="utf-8")

    resolved_path, calibration = load_required_calibration(output)

    assert resolved_path == calibration_path
    assert calibration["width"] == 960
    assert calibration["height"] == 600
    assert intrinsic_matrix(calibration) == [
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


def test_calibration_requires_complete_schema(tmp_path) -> None:
    output = tmp_path / "job" / "merged_recon"
    _write_image(output / "left" / "000000.jpg")
    calibration = _calibration()
    calibration.pop("baseline")
    (output.parent / "calibration.json").write_text(
        json.dumps(calibration), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="'baseline' must be numeric"):
        load_required_calibration(output)


def test_calibration_resolution_must_match_rgb_images(tmp_path) -> None:
    output = tmp_path / "job" / "stage1_recon"
    _write_image(output / "left" / "000000.jpg", width=960, height=600)
    (output.parent / "calibration.json").write_text(
        json.dumps(_calibration(width=1920, height=1200)), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="1920x1200.*960x600"):
        load_required_calibration(output)
