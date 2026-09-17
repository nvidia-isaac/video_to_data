# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light helpers for keeping retargeted joints inside model limits."""

from __future__ import annotations

import numpy as np


def clamp_position_limits(
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    margin: float,
) -> np.ndarray:
    """Clamp configurations to a finite margin inside their position limits.

    Unbounded sides remain unbounded. If a finite interval is narrower than twice
    ``margin``, it collapses to its midpoint.
    """
    value_array = np.asarray(values, dtype=np.float64)
    lower_array = np.asarray(lower, dtype=np.float64)
    upper_array = np.asarray(upper, dtype=np.float64)
    if value_array.ndim == 0:
        raise ValueError("values must have at least one dimension")
    expected = (value_array.shape[-1],)
    if lower_array.shape != expected or upper_array.shape != expected:
        raise ValueError(
            f"limits must have shape {expected}, got {lower_array.shape} and "
            f"{upper_array.shape}"
        )
    if not np.all(np.isfinite(value_array)):
        raise ValueError("values must be finite")
    if np.any(np.isnan(lower_array)) or np.any(np.isnan(upper_array)):
        raise ValueError("limits must not contain NaN")
    if np.any(lower_array > upper_array):
        raise ValueError("lower limits must not exceed upper limits")
    if isinstance(margin, (bool, np.bool_)):
        raise ValueError("margin must be finite and nonnegative")
    margin_value = float(margin)
    if not np.isfinite(margin_value) or margin_value < 0.0:
        raise ValueError("margin must be finite and nonnegative")

    lower_interior = lower_array.copy()
    upper_interior = upper_array.copy()
    finite_lower = np.isfinite(lower_interior)
    finite_upper = np.isfinite(upper_interior)
    lower_interior[finite_lower] += margin_value
    upper_interior[finite_upper] -= margin_value

    collapsed = finite_lower & finite_upper & (lower_interior > upper_interior)
    midpoint = lower_array / 2.0 + upper_array / 2.0
    lower_interior[collapsed] = midpoint[collapsed]
    upper_interior[collapsed] = midpoint[collapsed]
    return np.clip(value_array, lower_interior, upper_interior)
