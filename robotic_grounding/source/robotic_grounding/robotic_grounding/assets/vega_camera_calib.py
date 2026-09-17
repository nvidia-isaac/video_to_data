# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Calibrated GR00T cameras for the fixed-base Vega with dual Sharpa hands.

Transforms use ROS/OpenCV optical convention (x right, y down, z forward) and can be
passed directly to ``CameraCfg.OffsetCfg(convention="ros")``. The wrist camera
calibration is expressed in the hand mount frame; the reduced URDF merges that fixed
link into ``R/L_arm_l7``, so :func:`wrist_cam_offset_in_l7` composes the URDF mount.

The ego calibration was measured at 30 degrees of head pitch. The close-to-chest lift
uses 50 degrees for deploy and simulation; the calibrated camera residual is propagated
through the URDF head kinematics so both sides retain the same mount contract.
"""

from __future__ import annotations

import math
from typing import cast

CALIB_EGO_CAM_OFFSET_SIM = (
    (-0.087257, 0.010587, 1.439830),
    (0.227843, -0.666526, 0.667340, -0.241845),
)
CALIB_RIGHT_WRIST_CAM_OFFSET = (
    (0.037290, -0.043113, 0.009431),
    (0.526983, 0.199401, 0.140530, 0.814113),
)
CALIB_LEFT_WRIST_CAM_OFFSET = (
    (0.037290, 0.043113, 0.009431),
    (0.814113, 0.140530, 0.199401, 0.526983),
)

_R_L7_T_HAND = ((0.1058, 0.0, -0.03242), (1.57079, 0.0, 1.57079))
_L_L7_T_HAND = ((0.1058, 0.0, -0.03242), (-1.57079, 0.0, -1.57079))

OAK_INTRINSICS = {
    "width": 640,
    "height": 480,
    "fx": None,
    "fy": None,
    "cx": None,
    "cy": None,
    "distortion": None,
    "nominal_hfov_deg": 69.0,
}
CAMERA_RENDER_WIDTH = 320
CAMERA_RENDER_HEIGHT = 240


def pinhole_focal_length_mm(horizontal_aperture_mm: float = 20.955) -> float:
    """Return the pinhole focal length matching the calibrated or nominal OAK FOV."""
    focal_x = OAK_INTRINSICS["fx"]
    if focal_x is not None:
        hfov = 2.0 * math.atan(
            cast(int, OAK_INTRINSICS["width"]) / (2.0 * float(focal_x))
        )
    else:
        hfov = math.radians(cast(float, OAK_INTRINSICS["nominal_hfov_deg"]))
    return horizontal_aperture_mm / (2.0 * math.tan(hfov / 2.0))


def _quat_mul(a: tuple, b: tuple) -> tuple:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _quat_rotate(q: tuple, v: tuple) -> tuple:
    qw, qx, qy, qz = q
    uvx = qy * v[2] - qz * v[1]
    uvy = qz * v[0] - qx * v[2]
    uvz = qx * v[1] - qy * v[0]
    uuvx = qy * uvz - qz * uvy
    uuvy = qz * uvx - qx * uvz
    uuvz = qx * uvy - qy * uvx
    return (
        v[0] + 2.0 * (qw * uvx + uuvx),
        v[1] + 2.0 * (qw * uvy + uuvy),
        v[2] + 2.0 * (qw * uvz + uuvz),
    )


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> tuple:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _compose(parent_t: tuple, child_t: tuple) -> tuple:
    """Compose ``(position, rpy)`` with ``(position, quat_wxyz)``."""
    p_ab, rpy_ab = parent_t
    q_ab = _quat_from_rpy(*rpy_ab)
    p_bc, q_bc = child_t
    pos = tuple(p + d for p, d in zip(p_ab, _quat_rotate(q_ab, p_bc), strict=True))
    return pos, _quat_mul(q_ab, q_bc)


def wrist_cam_offset_in_l7(side: str) -> tuple:
    """Return the calibrated wrist-camera transform in the merged arm-l7 frame."""
    if side == "right":
        return _compose(_R_L7_T_HAND, CALIB_RIGHT_WRIST_CAM_OFFSET)
    if side == "left":
        return _compose(_L_L7_T_HAND, CALIB_LEFT_WRIST_CAM_OFFSET)
    raise ValueError(f"side must be left/right, got {side!r}")


def _quat_conj(q: tuple) -> tuple:
    return q[0], -q[1], -q[2], -q[3]


def _quat_from_axis_angle(axis: tuple, angle: float) -> tuple:
    norm = math.sqrt(sum(value * value for value in axis))
    scale = math.sin(angle / 2.0) / norm
    return (
        math.cos(angle / 2.0),
        axis[0] * scale,
        axis[1] * scale,
        axis[2] * scale,
    )


def _transform_mul(a: tuple, b: tuple) -> tuple:
    pa, qa = a
    pb, qb = b
    pos = tuple(p + d for p, d in zip(pa, _quat_rotate(qa, pb), strict=True))
    return pos, _quat_mul(qa, qb)


def _transform_inv(transform: tuple) -> tuple:
    position, quat = transform
    inverse_quat = _quat_conj(quat)
    inverse_position = tuple(-v for v in _quat_rotate(inverse_quat, position))
    return inverse_position, inverse_quat


_BASE_T_ZED_CHAIN = (
    ((-0.235, 0.0, 0.248), (0.0, 0.0, 0.0), (0, -1, 0), 0.78),
    ((0.396, 0.0, 0.082), (0.0, 0.0, 0.0), (0, 1, 0), 1.57),
    ((-0.40718, 0.0, 0.09764), (0.0, 0.0, 0.0), (0, -1, 0), 0.44),
    ((-0.05908, 0.0, 0.44528), (0.0, 0.0, 0.0), None, None),
    ((-0.0735, -0.0725, 0.014), (0.0, 0.0, 0.0), (0, 1, 0), "HEAD"),
    ((0.0, 0.0725, -0.0035), (0.0, 0.0, 0.0), None, None),
    ((0.0, 0.002, 0.0495), (0.0, 0.0, 0.0), None, None),
    ((0.0365, 0.023, 0.0489), (-1.57079, 0.0, -1.57079), None, None),
)

CALIB_HEAD_PITCH_RAD = math.pi / 6
EGO_HEAD_PITCH_RAD = math.radians(50.0)


def _fk_base_t_zed(head_pitch: float) -> tuple:
    transform = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    for xyz, rpy, axis, value in _BASE_T_ZED_CHAIN:
        transform = _transform_mul(transform, (xyz, _quat_from_rpy(*rpy)))
        if axis is not None:
            if value == "HEAD":
                joint_value = head_pitch
            elif isinstance(value, (float, int)):
                joint_value = float(value)
            else:
                raise ValueError(f"invalid joint value {value!r} for axis {axis}")
            transform = _transform_mul(
                transform,
                ((0.0, 0.0, 0.0), _quat_from_axis_angle(axis, joint_value)),
            )
    return transform


_ZED_T_CAM_RESIDUAL = _transform_mul(
    _transform_inv(_fk_base_t_zed(CALIB_HEAD_PITCH_RAD)),
    CALIB_EGO_CAM_OFFSET_SIM,
)


def ego_cam_offset_in_base(head_pitch: float | None = None) -> tuple:
    """Return the ego-camera transform in the reduced URDF's merged root frame."""
    pitch = EGO_HEAD_PITCH_RAD if head_pitch is None else head_pitch
    return _transform_mul(_fk_base_t_zed(pitch), _ZED_T_CAM_RESIDUAL)
