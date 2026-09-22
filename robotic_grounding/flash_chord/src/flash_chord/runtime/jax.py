# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared zero-copy array views between Warp and JAX."""

from __future__ import annotations

import math
from typing import Any

import warp as wp


def to_jax(array: wp.array, shape: tuple[int, ...] | None = None) -> Any:
    """Return a zero-copy JAX view of a contiguous Warp array."""
    if not array.is_contiguous:
        raise ValueError("JAX interop requires a contiguous Warp array")
    view = array
    if shape is not None:
        if math.prod(shape) != array.size:
            raise ValueError(f"requested shape {shape} has {math.prod(shape)} elements; array has {array.size}")
        view = array.reshape(shape)
    return wp.to_jax(view)


def from_jax(array: Any, dtype=None) -> wp.array:
    """Return a zero-copy Warp view of a JAX array."""
    return wp.from_jax(array, dtype=dtype)
