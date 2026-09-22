# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Quaternion helpers — numpy (host) conversions + Warp (device) ops.

Newton/Warp quaternions are ``xyzw``; the retargeted parquet / :class:`Reference` are ``wxyz``. The
numpy functions accept a single quaternion ``(4,)`` or a batched ``(..., 4)`` array; the ``@wp.func``
ops run inside Warp kernels.
"""

from __future__ import annotations

import numpy as np
import warp as wp


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.asarray(q)[..., [1, 2, 3, 0]]


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.asarray(q)[..., [3, 0, 1, 2]]


def wxyz_to_rotvec(q: np.ndarray) -> np.ndarray:
    """wxyz quaternion(s) -> rotation vector (axis * angle), shortest path. ``(..., 4) -> (..., 3)``."""
    q = np.asarray(q, dtype=float)
    sign = np.where(q[..., 0] < 0.0, -1.0, 1.0)  # shortest: ensure w >= 0
    w = q[..., 0] * sign
    xyz = q[..., 1:4] * sign[..., None]
    s = np.linalg.norm(xyz, axis=-1)
    small = s < 1e-8
    scale = np.where(small, 2.0, 2.0 * np.arctan2(s, w) / np.where(small, 1.0, s))
    return xyz * scale[..., None]


def quat_rotate_xyzw(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` by quaternion ``q`` (xyzw, Newton/Warp layout)."""
    q = np.asarray(q_xyzw)
    u, w = q[:3], q[3]
    return np.asarray(v) + 2.0 * np.cross(u, np.cross(u, v) + w * v)


# --- Warp (device) ops, for use inside @wp.kernel ---


@wp.func
def quat_mul_xyzw(a: wp.quat, b: wp.quat) -> wp.quat:
    """Multiply two quaternions (xyzw)."""
    return wp.quat(
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    )


@wp.func
def quat_inv_xyzw(q: wp.quat) -> wp.quat:
    """Invert a unit quaternion (xyzw)."""
    return wp.quat(-q[0], -q[1], -q[2], q[3])


@wp.func
def quat_geodesic_angle(a: wp.quat, b: wp.quat) -> wp.float32:
    """Geodesic angle [rad] between two unit quaternions (xyzw); double-cover aware."""
    d = wp.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])
    return 2.0 * wp.acos(wp.clamp(d, 0.0, 1.0))


@wp.func
def quat_to_rotvec(q_in: wp.quat) -> wp.vec3:
    """Convert a quaternion (xyzw) to a rotation vector (axis * angle), shortest path."""
    q = wp.normalize(q_in)
    if q[3] < 0.0:
        q = wp.quat(-q[0], -q[1], -q[2], -q[3])
    v = wp.vec3(q[0], q[1], q[2])
    s = wp.length(v)
    if s < 1.0e-8:
        return 2.0 * v
    angle = 2.0 * wp.atan2(s, q[3])
    return v * (angle / s)
