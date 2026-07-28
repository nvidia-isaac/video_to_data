# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""G1 kinematics: seed-pose forward kinematics and transforms -> qpos.

Provides the static-pose seed used to establish the model's coordinate
convention (via MuJoCo forward kinematics), the hand-root -> wrist offset for EE
inputs, and the inverse step that turns decoded global body transforms back into
MuJoCo qpos ``[pos(3), wxyz_quat(4), joints(29)]``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from robotic_grounding.planner.motionbricks.rotations import (
    _rot6d_first_two_cols_to_matrix,
)

DEFAULT_SEED_XML: Path = (
    Path(__file__).parent.parent / "assets" / "mujoco" / "g1_29dof.xml"
)

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


def apply_hand_root_to_wrist_offset(
    pos_w: np.ndarray, quat_wxyz: np.ndarray, offset_local: tuple[float, float, float]
) -> np.ndarray:
    """Shift hand-root world positions to wrist_yaw_link world positions."""
    quat_xyzw = np.asarray(quat_wxyz)[:, [1, 2, 3, 0]]
    rot = Rotation.from_quat(quat_xyzw)
    offset_w = rot.apply(np.asarray(offset_local, dtype=np.float64))
    return (np.asarray(pos_w) + offset_w).astype(np.float32)


def build_seed_qpos(
    num_frames: int, root_height: float
) -> tuple[np.ndarray, list[str]]:
    """Build a static nominal-pose seed qpos for canonicalization forward kinematics."""
    qpos = np.zeros((num_frames, 36), dtype=np.float32)
    qpos[:, 2] = root_height
    qpos[:, 3] = 1.0  # wxyz identity (w=1)
    for jname, val in NOMINAL_BODY_JOINTS.items():
        idx = G1_BODY_JOINT_NAMES.index(jname)
        qpos[:, 7 + idx] = val
    return qpos, list(G1_BODY_JOINT_NAMES)


def qpos_to_body_world(
    qpos: np.ndarray, joint_names: list[str], xml_path: str | Path
) -> tuple[np.ndarray, np.ndarray]:
    """Run MuJoCo forward kinematics -> G1 body world poses in IsaacLab body order."""
    import mujoco  # noqa: PLC0415 — mujoco is heavy and only needed for the seed FK.

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    if qpos.shape[-1] != model.nq:
        raise ValueError(f"seed qpos dim {qpos.shape[-1]} != MuJoCo nq={model.nq}")
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
            raise ValueError(f"MuJoCo XML missing G1 body {name!r}")
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


def _global_to_local_rotations(
    global_rots: np.ndarray, parents: np.ndarray
) -> np.ndarray:
    """Global body rotations -> per-joint local rotations (parent^T @ global)."""
    p = np.asarray(parents).reshape(-1)[: global_rots.shape[-3]].copy()
    root_mask = p == -1
    p[root_mask] = 0
    parent_rot = global_rots[..., p, :, :]
    local = np.matmul(np.swapaxes(parent_rot, -1, -2), global_rots)
    if root_mask.any():
        local[..., root_mask, :, :] = global_rots[..., root_mask, :, :]
    return local


def _rotation_matrices_to_dof(rot_mats: np.ndarray, dof_axis: np.ndarray) -> np.ndarray:
    """Project per-joint rotation matrices onto their actuated axis."""
    R = rot_mats
    x_angle = np.arctan2(R[..., 2, 1], R[..., 2, 2])
    y_angle = np.arctan2(R[..., 0, 2], R[..., 0, 0])
    z_angle = np.arctan2(R[..., 1, 0], R[..., 1, 1])
    xyz = np.stack([x_angle, y_angle, z_angle], axis=-1)
    axis = np.asarray(dof_axis, dtype=xyz.dtype)
    for _ in range(xyz.ndim - 2):
        axis = axis[None]
    axis = np.broadcast_to(axis, xyz.shape)
    return (xyz * axis).sum(axis=-1)


def features_to_qpos_np(
    pred_joints: np.ndarray,
    pred_root_xy: np.ndarray,
    kin: dict[str, Any],
    body_reorder: np.ndarray,
) -> np.ndarray:
    """Reconstruct G1 MuJoCo qpos (wxyz root) from decoded body transforms + root XY."""
    F = min(pred_joints.shape[0], pred_root_xy.shape[0])
    num_bodies = int(kin["num_bodies"])
    num_dof = int(kin["num_dof"])
    pj = np.asarray(pred_joints[:F], dtype=np.float64).reshape(F, num_bodies, 9)
    reorder = np.asarray(body_reorder)[:num_bodies]
    pj = pj[:, reorder]

    body_pos = pj[..., :3].copy()
    body_rot = _rot6d_first_two_cols_to_matrix(pj[..., 3:9])  # (F, B, 3, 3)

    rxy = np.asarray(pred_root_xy[:F], dtype=np.float64)
    body_pos[..., 0] += rxy[:, None, 0]
    body_pos[..., 1] += rxy[:, None, 1]

    parents = np.asarray(kin["parents"])
    local_rots = _global_to_local_rotations(body_rot, parents)  # (F, B, 3, 3)

    root_mat = local_rots[..., 0, :, :]  # (F, 3, 3)
    root_quat_xyzw = Rotation.from_matrix(root_mat.reshape(-1, 3, 3)).as_quat()
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]].reshape(F, 4)

    local_rot_mat = np.asarray(
        kin["local_rotation_mat"], dtype=np.float64
    )  # (1, 33, 3, 3)
    lrm = local_rot_mat[:, 1:num_bodies]  # (1, B-1, 3, 3)
    joint_rot_mat = np.matmul(
        np.swapaxes(lrm, -1, -2), local_rots[..., 1:num_bodies, :, :]
    )  # (F, B-1, 3, 3)
    dof_angles = _rotation_matrices_to_dof(joint_rot_mat, np.asarray(kin["dof_axis"]))

    qpos = np.zeros((F, 7 + num_dof), dtype=np.float32)
    qpos[:, :3] = body_pos[:, 0]
    qpos[:, 3:7] = root_quat_wxyz
    qpos[:, 7:] = dof_angles
    return qpos
