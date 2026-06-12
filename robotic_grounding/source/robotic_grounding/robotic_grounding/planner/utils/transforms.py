# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coordinate transforms for the G1 whole-body planner.

Two layers coexist here:

- High-level pipeline (`transform_reference` and the `apply_*` helpers) maps a
  V2P retargeted reference into the planner workspace frame: local frame fix,
  heading-toward-object yaw, position alignment with the nominal EE midpoint,
  and an optional workspace offset.
- Low-level rigid-transform primitives (`quat_*`, `transform_primary_*`,
  `transform_contact_*_by_part`) reproduce the same per-frame rigid transform
  for arrays the pipeline doesn't itself touch (hand keypoints, per-body
  contact positions/normals), so downstream consumers see a single coherent
  frame across every field of the output parquet.

Quaternions use the wxyz convention throughout, matching the on-disk motion
schema.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

# -- Sharpa → G1 frame rotation matrices -------------------------------------------
# Sharpa hand_C_MC body frame:
#   Left:  X=right, Y=down,  Z=forward
#   Right: X=left,  Y=up,    Z=forward
# G1 wrist_yaw_link frame (both hands):
#   X=forward, Y=left, Z=up

_R_S2G_LEFT = Rotation.from_matrix(
    [
        [0, 0, 1],  # G1 X (fwd) = Sharpa Z (fwd)
        [-1, 0, 0],  # G1 Y (left) = -Sharpa X (right)
        [0, -1, 0],  # G1 Z (up) = -Sharpa Y (down)
    ]
)

_R_S2G_RIGHT = Rotation.from_matrix(
    [
        [0, 0, 1],  # G1 X (fwd) = Sharpa Z (fwd)
        [1, 0, 0],  # G1 Y (left) = Sharpa X (left)
        [0, 1, 0],  # G1 Z (up) = Sharpa Y (up)
    ]
)

# Data conversion: post-multiply by R_s2g.inv()
_R_LOCAL_FIX_LEFT = _R_S2G_LEFT.inv()
_R_LOCAL_FIX_RIGHT = _R_S2G_RIGHT.inv()


# -- Quaternion conversions ---------------------------------------------------


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Permute a single quaternion or quaternion array from xyzw to wxyz."""
    q = np.asarray(q)
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """Permute a single quaternion or quaternion array from wxyz to xyzw."""
    q = np.asarray(q)
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def fix_quat_wxyz(q_wxyz: np.ndarray, r_fix: Rotation) -> np.ndarray:
    """Post-multiply wxyz quaternion(s) by a rotation.

    Args:
        q_wxyz: (T, 4) or (4,) quaternions in wxyz format.
        r_fix: Rotation to post-multiply.

    Returns:
        Corrected quaternions in wxyz format, same shape as input.
    """
    squeeze = q_wxyz.ndim == 1
    if squeeze:
        q_wxyz = q_wxyz[np.newaxis]

    # wxyz → xyzw for scipy
    q_xyzw = q_wxyz[:, [1, 2, 3, 0]]
    r_src = Rotation.from_quat(q_xyzw)
    r_out = r_src * r_fix
    out_xyzw = r_out.as_quat()
    out_wxyz = out_xyzw[:, [3, 0, 1, 2]]

    return out_wxyz[0] if squeeze else out_wxyz


# -- Low-level rigid-transform primitives -------------------------------------


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Return the conjugate of a wxyz quaternion (negate the vector part)."""
    return np.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], axis=-1)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quaternions, broadcast over batch dims."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` by wxyz quaternion ``q`` (q v q*)."""
    vq = np.concatenate([np.zeros_like(v[..., :1]), v], axis=-1)
    return quat_mul(quat_mul(q, vq), quat_conj(q))[..., 1:]


def broadcast_time_axis(
    arr: np.ndarray, shape_no_last: tuple[int, ...], last: int
) -> np.ndarray:
    """Broadcast a (T, last) array against a leading (T, *inner) target shape."""
    expand = (1,) * (len(shape_no_last) - 1)
    return np.broadcast_to(
        arr.reshape(arr.shape[0], *expand, last), (*shape_no_last, last)
    )


def transform_primary_pos(
    arr: np.ndarray,
    raw_obj_pos: np.ndarray,
    dst_obj_pos: np.ndarray,
    r_rel: np.ndarray,
) -> np.ndarray:
    """Rotate positions about the primary object body's per-frame frame change.

    ``arr`` has shape ``(T, ..., 3)``; the leading T axis is broadcast against
    ``r_rel`` / ``raw_obj_pos`` / ``dst_obj_pos`` (all shape ``(T, 4)`` or
    ``(T, 3)``).
    """
    arr = np.asarray(arr, dtype=np.float32)
    r = broadcast_time_axis(r_rel, arr.shape[:-1], 4)
    raw_p = broadcast_time_axis(raw_obj_pos, arr.shape[:-1], 3)
    dst_p = broadcast_time_axis(dst_obj_pos, arr.shape[:-1], 3)
    return quat_rotate(r, arr - raw_p) + dst_p


def transform_primary_quat(arr: np.ndarray, r_rel: np.ndarray) -> np.ndarray:
    """Rotate quaternions by the primary object body's per-frame frame change."""
    arr = np.asarray(arr, dtype=np.float32)
    r = broadcast_time_axis(r_rel, arr.shape[:-1], 4)
    return quat_mul(r, arr)


def transform_contact_pos_by_part(
    arr: np.ndarray,
    raw_obj_pos_all: np.ndarray,
    dst_obj_pos_all: np.ndarray,
    raw_obj_quat_all: np.ndarray,
    dst_obj_quat_all: np.ndarray,
    part_ids: np.ndarray | None,
) -> np.ndarray:
    """Per-body rigid transform on ``(T, N, 3)`` contact positions.

    Each contact point uses its own body's pose change (rather than the
    primary body's), which matters when the hand contacts a non-primary
    body of a multi-body or articulated object. ``part_ids`` is 1-indexed
    into the object's body list; entries equal to 0 are treated as
    inactive and left untouched.
    """
    out = np.asarray(arr, dtype=np.float32).copy()
    if part_ids is None:
        return out
    num_bodies = raw_obj_pos_all.shape[1]
    xyz = out[..., :3]
    for body_idx in range(num_bodies):
        mask = part_ids == body_idx + 1
        if not np.any(mask):
            continue
        t_idx, contact_idx = np.where(mask)
        r_rel = quat_mul(
            dst_obj_quat_all[t_idx, body_idx],
            quat_conj(raw_obj_quat_all[t_idx, body_idx]),
        )
        xyz[t_idx, contact_idx] = (
            quat_rotate(
                r_rel, xyz[t_idx, contact_idx] - raw_obj_pos_all[t_idx, body_idx]
            )
            + dst_obj_pos_all[t_idx, body_idx]
        )
    out[..., :3] = xyz
    return out


def transform_contact_dir_by_part(
    arr: np.ndarray,
    raw_obj_quat_all: np.ndarray,
    dst_obj_quat_all: np.ndarray,
    part_ids: np.ndarray | None,
) -> np.ndarray:
    """Per-body rotation on ``(T, N, 3)`` contact normals — translation skipped.

    Direction-only counterpart to :func:`transform_contact_pos_by_part`.
    Zero-length normals (contact-inactive frames) are left untouched.
    """
    out = np.asarray(arr, dtype=np.float32).copy()
    if part_ids is None:
        return out
    num_bodies = raw_obj_quat_all.shape[1]
    xyz = out[..., :3]
    for body_idx in range(num_bodies):
        mask = (part_ids == body_idx + 1) & (np.linalg.norm(xyz, axis=-1) > 1e-8)
        if not np.any(mask):
            continue
        t_idx, contact_idx = np.where(mask)
        r_rel = quat_mul(
            dst_obj_quat_all[t_idx, body_idx],
            quat_conj(raw_obj_quat_all[t_idx, body_idx]),
        )
        xyz[t_idx, contact_idx] = quat_rotate(r_rel, xyz[t_idx, contact_idx])
    out[..., :3] = xyz
    return out


# -- High-level transform pipeline --------------------------------------------


def apply_local_frame_fix(
    data: dict,
    robot_type: str = "sharpa",
) -> dict:
    """Apply Sharpa→G1 local frame correction to wrist quaternions.

    Args:
        data: Dict with keys left_quat, right_quat (wxyz), and positions.
        robot_type: "sharpa" or "dex3". Dex3 is already aligned.

    Returns:
        Copy of data with corrected quaternions.
    """
    out = dict(data)
    if robot_type == "sharpa":
        out["left_quat"] = fix_quat_wxyz(data["left_quat"], _R_LOCAL_FIX_LEFT)
        out["right_quat"] = fix_quat_wxyz(data["right_quat"], _R_LOCAL_FIX_RIGHT)
    return out


def compute_heading_toward_object(
    left_pos: np.ndarray,
    right_pos: np.ndarray,
    obj_pos: np.ndarray | None,
    frame_index: int = 0,
) -> float:
    """Compute heading angle from wrist midpoint toward the object.

    Falls back to hand-perpendicular heading if no object data.

    Args:
        left_pos: (T, 3) left wrist positions.
        right_pos: (T, 3) right wrist positions.
        obj_pos: (T, 3) object positions, or None.
        frame_index: Reference frame to read positions from (clipped to range).

    Returns:
        Heading angle in radians.
    """
    idx = int(np.clip(frame_index, 0, len(left_pos) - 1))
    mid = 0.5 * (left_pos[idx] + right_pos[idx])
    if obj_pos is not None and len(obj_pos) > 0:
        obj_idx = int(np.clip(idx, 0, len(obj_pos) - 1))
        fwd = obj_pos[obj_idx, :2] - mid[:2]
        if np.linalg.norm(fwd) > 1e-4:
            return float(np.arctan2(fwd[1], fwd[0]))

    # Fallback: perpendicular to L→R vector
    lr = right_pos[idx] - left_pos[idx]
    return float(np.arctan2(-lr[0], lr[1]))


def apply_yaw_correction(
    data: dict,
    delta_yaw: float,
    midpoint: np.ndarray | None = None,
) -> dict:
    """Rotate all positions and quaternions by delta_yaw around Z axis.

    Args:
        data: Dict with left_pos, left_quat, right_pos, right_quat,
              and optionally object_pos, object_quat.
        delta_yaw: Rotation angle in radians.
        midpoint: Center of rotation in XY. If None, uses frame-0 wrist midpoint.

    Returns:
        Copy of data with rotated values.
    """
    if midpoint is None:
        midpoint = 0.5 * (data["left_pos"][0] + data["right_pos"][0])
    midpoint = np.array(midpoint, dtype=np.float64)

    r_yaw = Rotation.from_euler("z", delta_yaw)
    out = dict(data)

    for pos_key in ("left_pos", "right_pos", "object_pos"):
        if pos_key in out and out[pos_key] is not None:
            p = np.array(out[pos_key], dtype=np.float64)
            p = r_yaw.apply(p - midpoint) + midpoint
            out[pos_key] = p.astype(np.float32)

    for quat_key in ("left_quat", "right_quat", "object_quat"):
        if quat_key in out and out[quat_key] is not None:
            q = np.array(out[quat_key], dtype=np.float64)
            q_xyzw = q[:, [1, 2, 3, 0]]
            r_src = Rotation.from_quat(q_xyzw)
            r_out = r_yaw * r_src
            q_out = r_out.as_quat()[:, [3, 0, 1, 2]]
            out[quat_key] = q_out.astype(np.float32)

    return out


def apply_position_offset(
    left_pos: np.ndarray,
    right_pos: np.ndarray,
    nominal_ee: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shift trajectory so frame-0 wrist midpoint aligns with nominal EE midpoint.

    Args:
        left_pos: (T, 3) left wrist positions.
        right_pos: (T, 3) right wrist positions.
        nominal_ee: Dict with left_pos, right_pos from FK at nominal pose.

    Returns:
        (shifted_left, shifted_right, offset_vector)
    """
    nom_mid = 0.5 * (nominal_ee["left_pos"] + nominal_ee["right_pos"])
    traj_mid = 0.5 * (left_pos[0] + right_pos[0])
    offset = nom_mid - traj_mid
    return left_pos + offset, right_pos + offset, offset


def transform_reference(
    ref_raw: dict,
    nominal_ee: dict,
    workspace_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    robot_type: str = "sharpa",
    delta_yaw_offset: float = 0.0,
    heading_frame: int = 0,
) -> dict:
    """Full transform pipeline: local frame fix → yaw → position align → offset.

    Args:
        ref_raw: Raw reference data from load_v2p_reference().
        nominal_ee: Nominal EE dict from get_nominal_ee().
        workspace_offset: Additional (x, y, z) offset.
        robot_type: "sharpa" or "dex3".
        delta_yaw_offset: Extra yaw rotation in radians added on top of the
            heading-toward-object correction. Used for local search over
            starting heading.
        heading_frame: Reference frame used to compute heading toward object.

    Returns:
        Dict with transformed positions, quaternions, finger joints, and metadata
        (delta_yaw, offset vector).
    """
    ws = np.array(workspace_offset, dtype=np.float32)

    # Step 1: local frame fix
    data = apply_local_frame_fix(ref_raw, robot_type=robot_type)

    # Step 2: yaw correction
    heading_src = compute_heading_toward_object(
        data["left_pos"],
        data["right_pos"],
        data.get("object_pos"),
        frame_index=heading_frame,
    )
    g1_forward = 0.0  # G1 faces +X
    delta_yaw = g1_forward - heading_src + float(delta_yaw_offset)
    data = apply_yaw_correction(data, delta_yaw)

    # Step 3: position offset
    lp, rp, offset = apply_position_offset(
        data["left_pos"],
        data["right_pos"],
        nominal_ee,
    )
    data["left_pos"] = lp + ws
    data["right_pos"] = rp + ws
    if data.get("object_pos") is not None:
        data["object_pos"] = data["object_pos"] + offset + ws

    # Transform all object bodies (multi-body objects)
    if data.get("object_pos_all") is not None:
        obj_all = np.array(data["object_pos_all"], dtype=np.float64)
        r_yaw = Rotation.from_euler("z", delta_yaw)
        midpoint = 0.5 * (ref_raw["left_pos"][0] + ref_raw["right_pos"][0])
        for b in range(obj_all.shape[1]):
            obj_all[:, b] = r_yaw.apply(obj_all[:, b] - midpoint) + midpoint
            obj_all[:, b] += offset + ws
        data["object_pos_all"] = obj_all.astype(np.float32)
    if data.get("object_quat_all") is not None:
        quat_all = np.array(data["object_quat_all"], dtype=np.float64)
        r_yaw = Rotation.from_euler("z", delta_yaw)
        for b in range(quat_all.shape[1]):
            q_xyzw = quat_all[:, b, [1, 2, 3, 0]]
            r_src = Rotation.from_quat(q_xyzw)
            r_out = r_yaw * r_src
            quat_all[:, b] = r_out.as_quat()[:, [3, 0, 1, 2]]
        data["object_quat_all"] = quat_all.astype(np.float32)

    data["delta_yaw"] = delta_yaw
    data["offset"] = offset + ws
    data["heading_frame"] = int(heading_frame)
    return data
