# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared setup and Warp primitives for residual policy inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import warp as wp

InputMode = Literal["raw", "normalized"]
NormalizedMapping = Literal["linear", "rational"]
InputMapping = Literal["identity", "linear", "rational"]


@dataclass(frozen=True)
class ResidualMapping:
    """Validated policy-input mapping and per-dimension physical residual bounds."""

    input_mode: InputMode
    normalized_mapping: NormalizedMapping
    input_mapping: InputMapping
    input_mapping_code: int
    input_scale: np.ndarray
    scale: np.ndarray
    clip: np.ndarray
    ema: float


def build_residual_mapping(
    scale,
    clip,
    *,
    ema: float,
    input_mode: InputMode,
    normalized_mapping: NormalizedMapping,
) -> ResidualMapping:
    """Validate and derive one residual mapping without action-specific semantics."""
    if input_mode not in ("raw", "normalized"):
        raise ValueError(f"input_mode must be 'raw' or 'normalized', got {input_mode!r}")
    if normalized_mapping not in ("linear", "rational"):
        raise ValueError(
            f"normalized_mapping must be 'linear' or 'rational', got {normalized_mapping!r}"
        )
    if not np.isfinite(ema) or not 0.0 <= ema <= 1.0:
        raise ValueError(f"ema must be finite and in [0, 1], got {ema}")
    if input_mode == "normalized" and normalized_mapping == "rational" and ema >= 1.0:
        raise ValueError("normalized rational mapping requires ema to be less than 1")

    scale = np.asarray(scale, dtype=np.float32)
    clip = np.asarray(clip, dtype=np.float32)
    if scale.ndim != 1 or scale.shape != clip.shape or not scale.size:
        raise ValueError(f"residual scale and clip must be matching nonempty vectors, got {scale.shape}/{clip.shape}")
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("residual scales must be positive finite float32 values")
    if not np.all(np.isfinite(clip)) or np.any(clip <= 0.0):
        raise ValueError("residual clips must be positive finite float32 values")

    input_scale = np.ones_like(scale)
    if input_mode == "normalized":
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if normalized_mapping == "linear":
                input_scale = clip / scale
            else:
                input_scale = clip / ((1.0 - ema) * scale)
        if not np.all(np.isfinite(input_scale)) or np.any(input_scale <= 0.0):
            raise ValueError("normalized input endpoint bounds must be positive finite float32 values")
    input_mapping = "identity" if input_mode == "raw" else normalized_mapping
    return ResidualMapping(
        input_mode=input_mode,
        normalized_mapping=normalized_mapping,
        input_mapping=input_mapping,
        input_mapping_code={"identity": 0, "linear": 1, "rational": 2}[input_mapping],
        input_scale=np.ascontiguousarray(input_scale),
        scale=np.ascontiguousarray(scale),
        clip=np.ascontiguousarray(clip),
        ema=float(ema),
    )


@wp.func
def map_residual_input(value: float, mapping: int, input_scale: float) -> float:
    """Map one raw or normalized policy scalar into residual units."""
    if mapping == 0:
        return value
    normalized = wp.clamp(value, -1.0, 1.0)
    if mapping == 1:
        return input_scale * normalized
    magnitude = wp.abs(normalized)
    denominator = (1.0 - magnitude) + magnitude / input_scale
    return normalized / denominator


@wp.func
def filter_residual(value: float, previous: float, scale: float, clip: float, ema: float) -> float:
    """Apply physical scale, EMA, and symmetric clipping to one residual scalar."""
    filtered = ema * previous + (1.0 - ema) * value * scale
    return wp.clamp(filtered, -clip, clip)
