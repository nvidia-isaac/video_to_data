"""Frame alignment transforms for Sharpa/Dex3 → G1 planner pipeline.

Handles local frame correction, yaw alignment, and workspace positioning.
"""

# ruff: noqa: ANN001, ANN201, ANN202, ANN204, D102, D103, D107, D417
# Planner is still in active development and this file is likely to change
# significantly with the new groot planner. Suppress annotation/docstring
# lint for now; real code issues are fixed individually.

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
) -> float:
    """Compute heading angle from wrist midpoint toward the object.

    Falls back to hand-perpendicular heading if no object data.

    Args:
        left_pos: (T, 3) left wrist positions.
        right_pos: (T, 3) right wrist positions.
        obj_pos: (T, 3) object positions, or None.

    Returns:
        Heading angle in radians.
    """
    mid = 0.5 * (left_pos[0] + right_pos[0])
    if obj_pos is not None and len(obj_pos) > 0:
        fwd = obj_pos[0, :2] - mid[:2]
        if np.linalg.norm(fwd) > 1e-4:
            return float(np.arctan2(fwd[1], fwd[0]))

    # Fallback: perpendicular to L→R vector
    lr = right_pos[0] - left_pos[0]
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
) -> dict:
    """Full transform pipeline: local frame fix → yaw → position align → offset.

    Args:
        ref_raw: Raw reference data from load_v2p_reference().
        nominal_ee: Nominal EE dict from get_nominal_ee().
        workspace_offset: Additional (x, y, z) offset.
        robot_type: "sharpa" or "dex3".

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
    )
    g1_forward = 0.0  # G1 faces +X
    delta_yaw = g1_forward - heading_src
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
    return data
