# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MuJoCo FK and coordinate conversion for the planner.

Handles:
- Parsing G1 skeleton XML for joint axes and hierarchy
- MuJoCo z-up ↔ motion-model y-up coordinate transforms
- Forward kinematics for converting model output to qpos
- Inverse: converting qpos to model features
"""

# ruff: noqa: ANN001, ANN201, ANN202, ANN204, D102, D103, D107, D417
# Planner is still in active development and this file is likely to change
# significantly with the new groot planner. Suppress annotation/docstring
# lint for now; real code issues are fixed individually.

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import torch
from scipy.spatial.transform import Rotation

# Coordinate transform: MuJoCo z-up (+X forward) ↔ motion model y-up (+Z forward)
# MuJoCo (x,y,z) → model (z,x,y)
_MUJOCO_TO_MODEL = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float64)
_MODEL_TO_MUJOCO = _MUJOCO_TO_MODEL.T

# As scipy Rotations
R_MUJOCO_TO_MODEL = Rotation.from_matrix(_MUJOCO_TO_MODEL)
R_MODEL_TO_MUJOCO = Rotation.from_matrix(_MODEL_TO_MUJOCO)

# Combined transform: MuJoCo z-up → model y-up
# R_zup_to_yup = Rotation.from_euler("x", -90, degrees=True)
# R_x_to_z = Rotation.from_euler("z", -90, degrees=True)
# MUJOCO_TO_MODEL_ROT = R_zup_to_yup * R_x_to_z
R_ZUP_TO_YUP = Rotation.from_euler("x", -np.pi / 2)
R_X_TO_Z_FORWARD = Rotation.from_euler("z", -np.pi / 2)
MUJOCO_TO_MODEL_ROT = R_ZUP_TO_YUP * R_X_TO_Z_FORWARD


def parse_skeleton_xml(xml_path: str) -> dict:
    """Parse a G1 MuJoCo XML to extract joint axes, names, and parent hierarchy.

    Args:
        xml_path: Path to MuJoCo XML file.

    Returns:
        Dict with keys:
        - joint_names: list of joint name strings
        - joint_axes: dict mapping joint_name → axis string ("x", "y", or "z")
        - joint_axis_vectors: dict mapping joint_name → (3,) unit vector
        - body_names: list of body name strings
        - parent_map: dict mapping body_name → parent body_name
        - joint_to_body: dict mapping joint_name → body_name
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    joint_names = []
    joint_axes = {}
    joint_axis_vectors = {}
    body_names = []
    parent_map = {}
    joint_to_body = {}

    def _traverse(elem, parent_body=None):
        if elem.tag == "body":
            bname = elem.get("name", "")
            body_names.append(bname)
            if parent_body is not None:
                parent_map[bname] = parent_body
            parent_body = bname

        if elem.tag == "joint":
            jname = elem.get("name", "")
            axis_str = elem.get("axis", "0 0 1")
            axis_vec = np.array([float(x) for x in axis_str.split()], dtype=np.float64)
            axis_vec /= np.linalg.norm(axis_vec) + 1e-12

            joint_names.append(jname)
            joint_axis_vectors[jname] = axis_vec
            joint_to_body[jname] = parent_body

            # Map axis vector to named axis in model space
            # After coordinate transform, MuJoCo [1,0,0] → model "z",
            # [0,1,0] → "x", [0,0,1] → "y"
            abs_axis = np.abs(axis_vec)
            max_idx = int(np.argmax(abs_axis))
            axis_label = {0: "z", 1: "x", 2: "y"}[max_idx]
            joint_axes[jname] = axis_label

        for child in elem:
            _traverse(child, parent_body)

    worldbody = root.find("worldbody")
    if worldbody is not None:
        _traverse(worldbody)

    return {
        "joint_names": joint_names,
        "joint_axes": joint_axes,
        "joint_axis_vectors": joint_axis_vectors,
        "body_names": body_names,
        "parent_map": parent_map,
        "joint_to_body": joint_to_body,
    }


def joint_angle_to_rotation_matrix(
    angle: float | np.ndarray,
    axis: str,
) -> np.ndarray:
    """Create a rotation matrix from a joint angle and named axis.

    Args:
        angle: Scalar or (T,) array of joint angles in radians.
        axis: One of "x", "y", "z".

    Returns:
        (3, 3) or (T, 3, 3) rotation matrix.
    """
    return Rotation.from_euler(axis, angle).as_matrix()


def transform_root_to_model(
    root_pos_mj: np.ndarray,
    root_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform root pose from MuJoCo z-up to model y-up frame.

    Args:
        root_pos_mj: (T, 3) root position in MuJoCo frame.
        root_wxyz: (T, 4) root quaternion in wxyz format, MuJoCo frame.

    Returns:
        (root_pos_model, root_quat_wxyz_model) both (T, 3) and (T, 4).
    """
    # Position: apply coordinate transform
    root_pos_model = MUJOCO_TO_MODEL_ROT.apply(root_pos_mj)

    # Rotation: R_model = R_transform * R_mujoco * R_transform^-1
    root_xyzw = root_wxyz[:, [1, 2, 3, 0]]
    r_mj = Rotation.from_quat(root_xyzw)
    r_model = MUJOCO_TO_MODEL_ROT * r_mj * MUJOCO_TO_MODEL_ROT.inv()
    out_xyzw = r_model.as_quat()
    root_quat_model = out_xyzw[:, [3, 0, 1, 2]]  # back to wxyz

    return root_pos_model.astype(np.float32), root_quat_model.astype(np.float32)


def transform_root_to_mujoco(
    root_pos_model: np.ndarray | torch.Tensor,
    root_quat_model: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform root pose from model y-up to MuJoCo z-up frame.

    Inverse of transform_root_to_model.

    Args:
        root_pos_model: (T, 3) or (B, T, 3) root position in model frame.
        root_quat_model: (T, 4) or (B, T, 4) root quaternion wxyz in model frame.

    Returns:
        (root_pos_mj, root_quat_wxyz_mj)
    """
    if isinstance(root_pos_model, torch.Tensor):
        root_pos_model = root_pos_model.cpu().numpy()
    if isinstance(root_quat_model, torch.Tensor):
        root_quat_model = root_quat_model.cpu().numpy()

    orig_shape = root_pos_model.shape
    root_pos_model = root_pos_model.reshape(-1, 3)
    root_quat_model = root_quat_model.reshape(-1, 4)

    inv_rot = MUJOCO_TO_MODEL_ROT.inv()
    root_pos_mj = inv_rot.apply(root_pos_model)

    q_xyzw = root_quat_model[:, [1, 2, 3, 0]]
    r_model = Rotation.from_quat(q_xyzw)
    r_mj = inv_rot * r_model * MUJOCO_TO_MODEL_ROT
    out_xyzw = r_mj.as_quat()
    root_quat_mj = out_xyzw[:, [3, 0, 1, 2]]

    pos_shape = orig_shape[:-1] + (3,)
    quat_shape = orig_shape[:-1] + (4,)
    return root_pos_mj.reshape(pos_shape).astype(np.float32), root_quat_mj.reshape(
        quat_shape
    ).astype(np.float32)


def extract_joint_dof_from_rotation(rot_matrix: np.ndarray, axis: str) -> np.ndarray:
    """Extract a single-axis joint angle from a rotation matrix.

    Args:
        rot_matrix: (..., 3, 3) rotation matrix.
        axis: "x", "y", or "z".

    Returns:
        (...,) joint angle in radians.
    """
    if axis == "x":
        return np.arctan2(rot_matrix[..., 2, 1], rot_matrix[..., 2, 2])
    elif axis == "y":
        return np.arctan2(
            -rot_matrix[..., 2, 0],
            np.sqrt(rot_matrix[..., 2, 1] ** 2 + rot_matrix[..., 2, 2] ** 2),
        )
    elif axis == "z":
        return np.arctan2(rot_matrix[..., 1, 0], rot_matrix[..., 0, 0])
    else:
        raise ValueError(f"Unknown axis: {axis}")
