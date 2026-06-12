# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kinematics helpers for the MotionBricks planner adapter.

Pure math + MuJoCo FK; no torch.package or bundle awareness. The agent
(``motionbricks_inference.MotionInferenceAgent``) calls into this module to
build a canonicalization seed and to convert decoded body transforms back
into MuJoCo qpos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# G1 body order expected by the planner. Must match the model's
# training-time body order; do not reorder.
G1_BODY_NAMES_ISAACLAB: list[str] = [
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]

DEFAULT_SEED_XML: Path = (
    Path(__file__).parent.parent / "assets" / "mujoco" / "g1_29dof.xml"
)

# Joint name ordering for qpos[7:] when FK'd against ``DEFAULT_SEED_XML``.
G1_BODY_JOINT_NAMES: list[str] = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# Indices into ``G1_BODY_NAMES_ISAACLAB`` for the lower-body bodies that
# get pinned when fix_lower_body is on (hips, knees, ankles — both sides).
LOWER_BODY_BODY_INDICES_ISAACLAB: tuple[int, ...] = (
    1,  # left_hip_pitch_link
    2,  # right_hip_pitch_link
    4,  # left_hip_roll_link
    5,  # right_hip_roll_link
    7,  # left_hip_yaw_link
    8,  # right_hip_yaw_link
    10,  # left_knee_link
    11,  # right_knee_link
    14,  # left_ankle_pitch_link
    15,  # right_ankle_pitch_link
    18,  # left_ankle_roll_link
    19,  # right_ankle_roll_link
)


# V2P retargeting reports wrist positions at the HAND ROOT body
# (palm_link/hand_C_MC), but the planner is trained on the wrist_yaw_link
# body. These local-frame offsets convert hand-root world positions to
# wrist_yaw_link world positions:
#   wrist_yaw_link_world = hand_root_world + R(hand_root_world_quat) @ offset
# Values are read from the dex3 hand MJCF (palm_link is a fixed-joint child
# of wrist_yaw_link with identity body_quat, so palm_link's world quat
# equals wrist_yaw_link's).
HAND_ROOT_TO_WRIST_OFFSET_LOCAL_LEFT: tuple[float, float, float] = (
    -0.0415,
    -0.003,
    0.0,
)
HAND_ROOT_TO_WRIST_OFFSET_LOCAL_RIGHT: tuple[float, float, float] = (
    -0.0415,
    0.003,
    0.0,
)


def apply_hand_root_to_wrist_offset(
    pos_w: np.ndarray,
    quat_wxyz: np.ndarray,
    offset_local: tuple[float, float, float],
) -> np.ndarray:
    """Shift hand-root world positions to wrist_yaw_link world positions.

    Args:
        pos_w: ``(T, 3)`` hand-root world positions.
        quat_wxyz: ``(T, 4)`` hand-root world quaternions in wxyz order.
        offset_local: ``(3,)`` constant offset in the hand-root local frame.

    Returns:
        ``(T, 3)`` wrist_yaw_link world positions.
    """
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    rot = Rotation.from_quat(quat_xyzw)
    offset_w = rot.apply(np.asarray(offset_local, dtype=np.float64))
    return (pos_w + offset_w).astype(np.float32)


# Nominal upper-body + leg pose for the canonicalization seed. Values are
# joint angles in radians; arms slightly tucked, legs in a soft crouch.
NOMINAL_BODY_JOINTS: dict[str, float] = {
    "left_shoulder_pitch_joint": -0.5,
    "left_shoulder_roll_joint": 0.2,
    "left_elbow_joint": 0.0,
    "right_shoulder_pitch_joint": -0.5,
    "right_shoulder_roll_joint": -0.2,
    "right_elbow_joint": 0.0,
    "left_hip_pitch_joint": -0.1,
    "left_knee_joint": 0.4,
    "left_ankle_pitch_joint": -0.2,
    "right_hip_pitch_joint": -0.1,
    "right_knee_joint": 0.4,
    "right_ankle_pitch_joint": -0.2,
}


# ---------------------------------------------------------------------------
# Seed qpos + MuJoCo FK
# ---------------------------------------------------------------------------


def build_seed_qpos(
    num_frames: int, root_height: float
) -> tuple[np.ndarray, list[str]]:
    """Build a static-pose seed qpos for canonicalization FK.

    Args:
        num_frames: Number of frames in the trajectory.
        root_height: World-frame Z height of the root body.

    Returns:
        ``(qpos, joint_names)`` where ``qpos`` is ``(num_frames, 36)`` with
        layout ``[root_pos(3), root_wxyz(4), body_joints(29)]`` and
        ``joint_names`` matches the qpos[7:] ordering.
    """
    qpos = np.zeros((num_frames, 36), dtype=np.float32)
    qpos[:, 2] = root_height
    qpos[:, 3] = 1.0  # wxyz identity (w=1)
    for jname, val in NOMINAL_BODY_JOINTS.items():
        idx = G1_BODY_JOINT_NAMES.index(jname)
        qpos[:, 7 + idx] = val
    return qpos, list(G1_BODY_JOINT_NAMES)


def qpos_to_body_world(
    qpos: np.ndarray, joint_names: list[str], xml_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Run MuJoCo FK and return G1 body world poses in IsaacLab body order.

    Args:
        qpos: ``(T, model.nq)`` qpos with root pos+wxyz at qpos[:7].
        joint_names: Joint name ordering of qpos[7:].
        xml_path: Path to the MuJoCo XML used for FK.

    Returns:
        ``(body_pos, body_wxyz)`` with shapes ``(T, 30, 3)`` and
        ``(T, 30, 4)`` in IsaacLab body order.
    """
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    if qpos.shape[-1] != model.nq:
        raise ValueError(
            f"seed qpos dim {qpos.shape[-1]} != MuJoCo nq={model.nq}; "
            f"check {xml_path}"
        )

    src_idx = {n: i for i, n in enumerate(joint_names)}
    model_joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(model.njnt)
        if int(model.jnt_qposadr[j]) >= 7
    ]
    qpos_perm = [src_idx[n] for n in model_joint_names]

    body_ids = []
    for name in G1_BODY_NAMES_ISAACLAB:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"MuJoCo XML missing G1 body {name!r} in {xml_path}")
        body_ids.append(bid)

    T = qpos.shape[0]
    body_pos = np.zeros((T, len(body_ids), 3), dtype=np.float32)
    body_wxyz = np.zeros((T, len(body_ids), 4), dtype=np.float32)
    for t in range(T):
        data.qpos[:7] = qpos[t, :7]
        data.qpos[7:] = qpos[t, 7:][qpos_perm]
        mujoco.mj_forward(model, data)
        for j, bid in enumerate(body_ids):
            body_pos[t, j] = data.xpos[bid]
            xyzw = Rotation.from_matrix(data.xmat[bid].reshape(3, 3)).as_quat()
            body_wxyz[t, j] = xyzw[[3, 0, 1, 2]]

    return body_pos, body_wxyz


# ---------------------------------------------------------------------------
# Qpos reconstruction
# ---------------------------------------------------------------------------
# The exporter ships kinematic chain tensors (parents, dof_axis, local
# rotations) as plain data so this module can convert decoded body
# transforms back to MuJoCo qpos without pulling open3d / hydra / lxml into
# v2d's runtime.


def _rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """Convert rot6d (first two columns of R) to a rotation matrix.

    Args:
        rot6d: ``[..., 6]`` packed as the first two columns of a rotation
            matrix flattened to 6 channels.

    Returns:
        ``[..., 3, 3]`` rotation matrix recovered via Gram-Schmidt + cross.
    """
    cols = rot6d.reshape(*rot6d.shape[:-1], 3, 2)
    c0 = cols[..., :, 0]
    c0 = c0 / c0.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    c1 = cols[..., :, 1]
    c1 = c1 - (c0 * c1).sum(dim=-1, keepdim=True) * c0
    c1 = c1 / c1.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    c2 = torch.cross(c0, c1, dim=-1)
    return torch.stack([c0, c1, c2], dim=-1)


def _global_to_local_rotations(
    global_rots: torch.Tensor, parents: torch.Tensor
) -> torch.Tensor:
    """Convert global body rotations to per-joint local rotations.

    Args:
        global_rots: ``[..., J, 3, 3]`` global rotation matrices.
        parents: ``[J]`` parent body index per joint, with ``-1`` marking
            roots (which pass through unchanged).

    Returns:
        ``[..., J, 3, 3]`` local rotations such that
        ``parent_rot @ local == global_rot``.
    """
    p = parents[: global_rots.shape[-3]].clone()
    root_mask = p == -1
    p[root_mask] = 0  # placeholder; overwritten below
    parent_rot = global_rots[..., p.long(), :, :]
    local = torch.matmul(parent_rot.transpose(-1, -2), global_rots)
    if root_mask.any():
        local[..., root_mask, :, :] = global_rots[..., root_mask, :, :]
    return local


def _rotation_matrices_to_dof(
    rot_mats: torch.Tensor, dof_axis: torch.Tensor
) -> torch.Tensor:
    """Project per-joint rotation matrices onto their actuated axis.

    Args:
        rot_mats: ``[..., N, 3, 3]`` per-joint rotations in the joint's
            local frame.
        dof_axis: ``[N, 3]`` one-hot-style axis selector per joint.

    Returns:
        ``[..., N]`` scalar joint angles around the selected axis.
    """
    R = rot_mats
    x_angle = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    y_angle = torch.atan2(R[..., 0, 2], R[..., 0, 0])
    z_angle = torch.atan2(R[..., 1, 0], R[..., 1, 1])
    xyz = torch.stack([x_angle, y_angle, z_angle], dim=-1)
    axis = dof_axis.to(R.device, R.dtype)
    for _ in range(xyz.dim() - 2):
        axis = axis.unsqueeze(0)
    axis = axis.expand(*xyz.shape[:-1], 3)
    return (xyz * axis).sum(dim=-1)


def _hamming_smooth(signal: np.ndarray, half_width: int) -> np.ndarray:
    """Edge-padded Hamming smoothing along the first axis."""
    if half_width <= 0 or signal.shape[0] <= 2:
        return signal.copy()
    signal = np.asarray(signal, dtype=np.float32)
    pad_left = np.repeat(signal[:1], half_width, axis=0)
    pad_right = np.repeat(signal[-1:], half_width, axis=0)
    padded = np.concatenate([pad_left, signal, pad_right], axis=0)
    kernel = np.hamming(2 * half_width + 1).astype(np.float32)
    kernel /= kernel.sum()
    flat = padded.reshape(padded.shape[0], -1)
    out = np.empty((signal.shape[0], flat.shape[1]), dtype=np.float32)
    for i in range(signal.shape[0]):
        out[i] = (flat[i : i + kernel.shape[0]] * kernel[:, None]).sum(axis=0)
    return out.reshape(signal.shape)


def _smooth_quat_wxyz(quat: np.ndarray, half_width: int) -> np.ndarray:
    """Hamming smoothing for a wxyz quaternion sequence with sign continuity."""
    if half_width <= 0 or quat.shape[0] <= 2:
        return quat.copy()
    aligned = np.asarray(quat, dtype=np.float32).copy()
    for i in range(1, aligned.shape[0]):
        if float(np.dot(aligned[i - 1], aligned[i])) < 0.0:
            aligned[i] *= -1.0
    smoothed = _hamming_smooth(aligned, half_width)
    smoothed /= np.clip(np.linalg.norm(smoothed, axis=-1, keepdims=True), 1e-8, None)
    return smoothed.astype(np.float32)


def _nlerp_quat_wxyz(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Sign-corrected normalized linear quaternion interpolation."""
    q0 = np.asarray(q0, dtype=np.float32)
    q1 = np.asarray(q1, dtype=np.float32).copy()
    flip = np.sum(q0 * q1, axis=-1, keepdims=True) < 0.0
    q1 = np.where(flip, -q1, q1)
    out = q0 * (1.0 - alpha) + q1 * alpha
    out /= np.clip(np.linalg.norm(out, axis=-1, keepdims=True), 1e-8, None)
    return out.astype(np.float32)


def _smooth_g1_qpos_wxyz(
    qpos: np.ndarray,
    *,
    pos_width: int = 3,
    quat_width: int = 3,
    joint_width: int = 2,
) -> np.ndarray:
    """Smooth a G1 qpos trajectory with wxyz root quaternion."""
    out = np.asarray(qpos, dtype=np.float32).copy()
    out[:, :3] = _hamming_smooth(out[:, :3], pos_width)
    out[:, 3:7] = _smooth_quat_wxyz(out[:, 3:7], quat_width)
    out[:, 7:] = _hamming_smooth(out[:, 7:], joint_width)
    return out


def chunk_boundary_centers(chunk_infos: list[dict[str, Any]] | None) -> list[int]:
    """Frame indices at the centre of each chunk-overlap region."""
    if not chunk_infos:
        return []
    centers: list[int] = []
    for prev, cur in zip(chunk_infos[:-1], chunk_infos[1:], strict=False):
        overlap_start = int(cur["start_frame"])
        overlap_end = min(int(prev["end_frame"]), int(cur["end_frame"]))
        if overlap_end > overlap_start:
            centers.append(overlap_start + (overlap_end - overlap_start) // 2)
        else:
            centers.append(overlap_start)
    return centers


def smooth_qpos_global(qpos: np.ndarray, *, nfpt: int) -> np.ndarray:
    """Mild global Hamming smoothing for the entire qpos trajectory.

    Equivalent to the legacy planner's first ``smooth_qpos(...)`` pass
    before targeted boundary smoothing. Half-widths are conservative so
    natural motion is preserved.
    """
    return _smooth_g1_qpos_wxyz(
        qpos, pos_width=2, quat_width=2, joint_width=max(1, nfpt // 2)
    )


def smooth_qpos_at_boundaries(
    qpos: np.ndarray,
    boundary_centers: list[int],
    *,
    nfpt: int,
) -> np.ndarray:
    """Apply Hamming smoothing locally around each chunk boundary.

    A Gaussian mask centred on each overlap midpoint blends smoothed
    Hamming output with the raw qpos. Quaternions use sign-corrected NLerp;
    positions and joints use linear blend. Half-widths match the legacy
    planner's ``boundary_radius = 3 * nfpt`` for joints.
    """
    if not boundary_centers:
        return qpos.copy()
    raw = np.asarray(qpos, dtype=np.float32)
    boundary_radius = max(3 * nfpt, 6)
    smooth = _smooth_g1_qpos_wxyz(
        raw,
        pos_width=max(boundary_radius // 2, 2),
        quat_width=max(boundary_radius // 2, 2),
        joint_width=boundary_radius,
    )
    T = raw.shape[0]
    frames = np.arange(T, dtype=np.float32)
    mask = np.zeros((T, 1), dtype=np.float32)
    sigma = max(float(nfpt) * 1.5, 1.0)
    for center in boundary_centers:
        gaussian = np.exp(-0.5 * ((frames - float(center)) / sigma) ** 2)
        mask = np.maximum(mask, gaussian.reshape(T, 1).astype(np.float32))
    out = raw.copy()
    out[:, :3] = raw[:, :3] * (1.0 - mask) + smooth[:, :3] * mask
    out[:, 7:] = raw[:, 7:] * (1.0 - mask) + smooth[:, 7:] * mask
    out[:, 3:7] = _nlerp_quat_wxyz(raw[:, 3:7], smooth[:, 3:7], mask)
    return out.astype(np.float32)


def features_to_qpos(
    pred_joints: np.ndarray,
    pred_root_xy: np.ndarray,
    kin: dict[str, Any],
    body_reorder: list[int],
) -> np.ndarray:
    """Reconstruct G1 MuJoCo qpos from decoded body transforms + root XY.

    Args:
        pred_joints: ``(F, num_bodies * 9)`` packed body transforms in
            IsaacLab body order — per body: ``[pos(3), rot6d(6)]``.
        pred_root_xy: ``(F, 2)`` predicted root XY in the same heading
            frame.
        kin: Kinematics dict with ``num_bodies``, ``num_dof``,
            ``dof_axis``, ``parents``, ``local_rotation_mat``.
        body_reorder: 30-int IsaacLab → MuJoCo body index permutation.

    Returns:
        ``(F, 7 + num_dof)`` MuJoCo qpos with root quaternion in wxyz
        order.
    """
    F = min(pred_joints.shape[0], pred_root_xy.shape[0])
    num_bodies = kin["num_bodies"]
    pj = torch.from_numpy(np.asarray(pred_joints[:F], dtype=np.float32))
    pj = pj.reshape(F, num_bodies, 9)

    reorder = body_reorder[:num_bodies]
    pj = pj[:, reorder]

    body_pos = pj[..., :3].clone()
    body_rot = _rot6d_to_matrix(pj[..., 3:9])

    rxy = torch.from_numpy(np.asarray(pred_root_xy[:F], dtype=np.float32))
    body_pos[..., 0] += rxy[:, None, 0]
    body_pos[..., 1] += rxy[:, None, 1]

    parents = kin["parents"].to(body_rot.dtype)
    local_rots = _global_to_local_rotations(body_rot, parents)

    root_mat_np = local_rots[..., 0, :, :].cpu().numpy()
    root_quat_xyzw = Rotation.from_matrix(root_mat_np.reshape(-1, 3, 3)).as_quat()
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]].reshape(F, 4)

    local_rot_mat = kin["local_rotation_mat"].to(local_rots.device, local_rots.dtype)
    joint_rot_mat = torch.matmul(
        local_rot_mat[:, 1:num_bodies].transpose(-1, -2),
        local_rots[..., 1:num_bodies, :, :],
    )
    dof_angles = _rotation_matrices_to_dof(joint_rot_mat, kin["dof_axis"])
    dof_angles = dof_angles.squeeze(0).cpu().numpy()

    qpos = np.zeros((F, 7 + kin["num_dof"]), dtype=np.float32)
    qpos[:, :3] = body_pos[:, 0].cpu().numpy()
    qpos[:, 3:7] = root_quat_wxyz
    qpos[:, 7:] = dof_angles
    return qpos
