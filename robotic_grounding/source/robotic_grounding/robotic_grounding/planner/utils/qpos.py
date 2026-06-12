# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MuJoCo-side qpos assembly + scoring for the G1 whole-body planner.

Owns the constants and helpers that turn the model's chunked-AR output
(`planned_qpos`) plus per-side V2P finger data into a single
`(T, model.nq)` qpos trajectory: body-joint name → qpos-index mapping,
finger-joint resolution, root-component pinning, and an FK-based wrist
tracking error used to score heading-offset candidates.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from robotic_grounding.planner.utils.transforms import xyzw_to_wxyz

# -- Constants ---------------------------------------------------------------

ROOT_HEIGHT = 0.793
ROOT_FIX_COMPONENTS = ("x", "y", "z", "roll", "pitch", "yaw")

LEG_OVERRIDES = {
    "left_hip_pitch_joint": -0.1,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.4,
    "left_ankle_pitch_joint": -0.2,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.1,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.4,
    "right_ankle_pitch_joint": -0.2,
    "right_ankle_roll_joint": 0.0,
}

LEFT_WRIST = "left_wrist_yaw_link"
RIGHT_WRIST = "right_wrist_yaw_link"

G1_BODY_JOINT_NAMES = [
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


# -- Mappings ----------------------------------------------------------------


def build_body_joint_mapping(model: mujoco.MjModel) -> dict[int, int]:
    """Map 29 body DOFs to combined model qpos indices."""
    mapping = {}
    for dof_idx, jname in enumerate(G1_BODY_JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            mapping[dof_idx] = int(model.jnt_qposadr[jid])
    return mapping


def build_finger_mapping(model: mujoco.MjModel, joint_names: list[str]) -> list[int]:
    """Map finger joint names to model qpos indices."""
    result = []
    for jname in joint_names:
        try:
            result.append(int(model.jnt_qposadr[model.joint(jname).id]))
        except Exception:
            result.append(-1)
    return result


# -- Wrist tracking error ----------------------------------------------------


def wrist_ee_error_from_qpos(
    full_qpos: np.ndarray,
    ref_data: dict,
    model: mujoco.MjModel,
) -> float:
    """Mean L2 distance from FK'd wrist_yaw_link bodies to V2P wrist targets.

    Used to score heading-offset candidates during the local search.
    """
    data = mujoco.MjData(model)
    li = model.body(LEFT_WRIST).id
    ri = model.body(RIGHT_WRIST).id
    target_l = np.asarray(ref_data["left_pos"], dtype=np.float64)
    target_r = np.asarray(ref_data["right_pos"], dtype=np.float64)
    T = min(full_qpos.shape[0], target_l.shape[0], target_r.shape[0])
    err_l = np.empty(T, dtype=np.float64)
    err_r = np.empty(T, dtype=np.float64)
    for t in range(T):
        data.qpos[: full_qpos.shape[1]] = full_qpos[t]
        mujoco.mj_forward(model, data)
        err_l[t] = np.linalg.norm(data.xpos[li] - target_l[t])
        err_r[t] = np.linalg.norm(data.xpos[ri] - target_r[t])
    return float(((err_l + err_r) / 2.0).mean())


# -- Root pinning ------------------------------------------------------------


def root_fix_component_set(
    components: tuple[str, ...] | list[str] = (),
    *,
    fix_root_pos: bool = False,
    fix_root_rot: bool = False,
    fix_root_z: bool = False,
    fix_root_rp: bool = False,
) -> set[str]:
    """Merge the generalized root component list with legacy root flags."""
    result = set(components or ())
    invalid = result.difference(ROOT_FIX_COMPONENTS)
    if invalid:
        raise ValueError(f"Unknown root fix component(s): {sorted(invalid)}")
    if fix_root_pos:
        result.update(("x", "y", "z"))
    if fix_root_z:
        result.add("z")
    if fix_root_rot:
        result.update(("roll", "pitch", "yaw"))
    if fix_root_rp:
        result.update(("roll", "pitch"))
    return result


def root_wxyz_with_fixed_components(
    q_xyzw: np.ndarray, fixed_components: set[str]
) -> np.ndarray:
    """Apply roll/pitch/yaw root clamps while preserving free components."""
    if not fixed_components.intersection(("roll", "pitch", "yaw")):
        return xyzw_to_wxyz(q_xyzw)

    euler_xyz = Rotation.from_quat(q_xyzw).as_euler("xyz", degrees=False)
    if "roll" in fixed_components:
        euler_xyz[0] = 0.0
    if "pitch" in fixed_components:
        euler_xyz[1] = 0.0
    if "yaw" in fixed_components:
        euler_xyz[2] = 0.0
    q_fixed_xyzw = Rotation.from_euler("xyz", euler_xyz).as_quat()
    return xyzw_to_wxyz(q_fixed_xyzw).astype(np.float32)


# -- Full qpos assembly ------------------------------------------------------


def build_full_qpos(
    planned_qpos: np.ndarray,
    ref_data: dict[str, Any],
    model: mujoco.MjModel,
    T_save: int,
    fix_lower_body: bool = False,
    fix_root_pos: bool = False,
    fix_root_rot: bool = False,
    fix_root_z: bool = False,
    fix_root_rp: bool = False,
    fix_root_components: tuple[str, ...] | list[str] = (),
) -> tuple[np.ndarray, dict[int, int], list[int], list[int]]:
    """Combine planned body + reference fingers + (optionally fixed) parts."""
    nq = model.nq
    full_qpos = np.zeros((T_save, nq), dtype=np.float32)
    body_map = build_body_joint_mapping(model)
    l_finger_map = build_finger_mapping(model, ref_data.get("left_joint_names", []))
    r_finger_map = build_finger_mapping(model, ref_data.get("right_joint_names", []))
    fixed_root = root_fix_component_set(
        fix_root_components,
        fix_root_pos=fix_root_pos,
        fix_root_rot=fix_root_rot,
        fix_root_z=fix_root_z,
        fix_root_rp=fix_root_rp,
    )

    for t in range(T_save):
        t_plan = min(t, planned_qpos.shape[0] - 1)
        full_qpos[t, :3] = planned_qpos[t_plan, :3]
        if "x" in fixed_root:
            full_qpos[t, 0] = 0.0
        if "y" in fixed_root:
            full_qpos[t, 1] = 0.0
        if "z" in fixed_root:
            full_qpos[t, 2] = ROOT_HEIGHT
        full_qpos[t, 3:7] = root_wxyz_with_fixed_components(
            planned_qpos[t_plan, 3:7], fixed_root
        )
        for dof_idx, qi in body_map.items():
            full_qpos[t, qi] = planned_qpos[t_plan, 7 + dof_idx]
        if fix_lower_body:
            for jname, val in LEG_OVERRIDES.items():
                try:
                    full_qpos[t, int(model.jnt_qposadr[model.joint(jname).id])] = val
                except Exception:
                    pass
        f_idx = min(t, ref_data["left_finger_joints"].shape[0] - 1)
        for j, qi in enumerate(l_finger_map):
            if qi >= 0 and j < ref_data["left_finger_joints"].shape[1]:
                full_qpos[t, qi] = ref_data["left_finger_joints"][f_idx, j]
        for j, qi in enumerate(r_finger_map):
            if qi >= 0 and j < ref_data["right_finger_joints"].shape[1]:
                full_qpos[t, qi] = ref_data["right_finger_joints"][f_idx, j]

    return full_qpos, body_map, l_finger_map, r_finger_map
