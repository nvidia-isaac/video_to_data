# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data adapters: MuJoCo arrays → motion model features.

Ported 1:1 from the original data_adapters module. The only external
dependency is the motion representation objects (global_motion_rep,
local_motion_rep, motion_rep) which are loaded from the bundled
torch.package at runtime.
"""

# ruff: noqa: ANN001, ANN201, ANN202, ANN204, D102, D103, D107, D417
# Planner is still in active development and this file is likely to change
# significantly with the new groot planner. Suppress annotation/docstring
# lint for now; real code issues are fixed individually.

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import einops
import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


def parquet_to_mfm(root_pos_mj, root_wxyz, joint_positions, skeleton, xml_path):
    """Convert MuJoCo-space robot data to local joint rotations + root translation.

    Coordinate transform: MuJoCo (z-up, x-forward) -> model (y-up, z-forward)
    via R_x(-90) * R_z(-90). Joint axes parsed from MuJoCo XML.

    Args:
        root_pos_mj: [T, 3] root position in MuJoCo coords.
        root_wxyz: [T, 4] root quaternion [w, x, y, z].
        joint_positions: [T, 29] joint DOFs (radians).
        skeleton: Skeleton object with bone_order_names.
        xml_path: Path to g1_29dof.xml.

    Returns:
        root_trans: [T, 3] root translation in model coords.
        joint_rots: [T, J, 3, 3] local joint rotation matrices.
    """
    T_frames = root_pos_mj.shape[0]
    num_bones = len(skeleton.bone_order_names)

    R_zup_to_yup = Rotation.from_euler("x", -90, degrees=True)
    x_to_y_forward = Rotation.from_euler("z", -90, degrees=True)
    mujoco_to_mfm = R_zup_to_yup * x_to_y_forward

    root_trans = mujoco_to_mfm.apply(root_pos_mj)
    root_rot = Rotation.from_quat(root_wxyz[:, [1, 2, 3, 0]])
    root_rot_mfm = mujoco_to_mfm * root_rot * mujoco_to_mfm.inv()

    joint_rots = np.tile(np.eye(3), (T_frames, num_bones, 1, 1))
    joint_rots[:, 0] = root_rot_mfm.as_matrix()

    tree = ET.parse(xml_path)
    root_el = tree.getroot()
    for dof_idx, joint_el in enumerate(
        root_el.find("worldbody").findall(".//joint")[1:]
    ):
        joint_name = joint_el.get("name").replace("_joint", "_skel")
        assert joint_name in skeleton.bone_order_names
        bone_idx = skeleton.bone_order_names.index(joint_name)

        axis_vals = [float(x) for x in joint_el.get("axis").split(" ")]
        axis_in_mfm = ["z", "x", "y"][np.argmax(axis_vals)]
        joint_rot = Rotation.from_euler(axis_in_mfm, joint_positions[:, dof_idx])
        joint_rots[:, bone_idx] = joint_rot.as_matrix()

    return root_trans, joint_rots


_SKEL_ASSETS = Path(__file__).parent / "assets" / "skeleton"


def _batch_rigid_transform(local_rots, neutral_joints, joint_parents, root_idx=0):
    """Forward kinematics: local rotations → global rotations.

    Args:
        local_rots: (B, J, 3, 3)
        neutral_joints: (J, 3) rest-pose joint positions
        joint_parents: (J,) parent indices (-1 for root)

    Returns:
        global_rots: (B, J, 3, 3)
    """
    B, J = local_rots.shape[:2]
    global_rots = torch.zeros_like(local_rots)
    parents = (
        joint_parents if isinstance(joint_parents, list) else joint_parents.tolist()
    )
    for j in range(J):
        p = int(parents[j])
        if p < 0 or j == root_idx:
            global_rots[:, j] = local_rots[:, j]
        else:
            global_rots[:, j] = global_rots[:, p] @ local_rots[:, j]
    return global_rots


def change_t_pose_local_mats(
    local_mats, skeleton, t_pose_from="capture", t_pose_to="standard"
):
    """Convert local rotation matrices between T-pose conventions.

    Uses pre-exported offset matrices from planner/assets/skeleton/.
    No runtime file I/O beyond loading the .npy offsets.

    Args:
        local_mats: (T, J, 3, 3) or (B, T, J, 3, 3) local rotation matrices.
        skeleton: Skeleton object with neutral_joints, joint_parents, nbjoints, root_idx.
        t_pose_from: Source T-pose convention.
        t_pose_to: Target T-pose convention.

    Returns:
        Converted local rotation matrices, same shape as input.
    """
    if t_pose_from == t_pose_to:
        return local_mats

    orig_shape = local_mats.shape
    device = local_mats.device
    dtype = local_mats.dtype
    J = skeleton.nbjoints
    local_mats = local_mats.reshape(-1, J, 3, 3)
    B = local_mats.shape[0]

    # Load pre-exported global offset matrices
    def _load_offset(pose_name):
        if pose_name == "capture":
            return torch.eye(3).repeat(J, 1, 1)
        path = _SKEL_ASSETS / f"t_pose_offset_{pose_name}.npy"
        return torch.from_numpy(np.load(path))

    offset_from = _load_offset(t_pose_from).to(device=device, dtype=dtype)
    offset_to = _load_offset(t_pose_to).to(device=device, dtype=dtype)

    # FK: local → global
    neutral = skeleton.neutral_joints
    batched_neutral = einops.repeat(neutral, "j k -> b j k", b=B).to(
        dtype=dtype, device=device
    )
    global_mats = _batch_rigid_transform(
        local_mats, batched_neutral, skeleton.joint_parents, skeleton.root_idx
    )

    # Apply T-pose conversion
    new_global = torch.einsum(
        "... N m n, N n o, N p o -> ... N m p",
        global_mats,
        offset_from,
        offset_to,
    )

    # Global → local
    new_local = torch.zeros_like(new_global)
    parents = skeleton.joint_parents
    if isinstance(parents, torch.Tensor):
        parents = parents.tolist()
    for j in range(J):
        p = int(parents[j])
        if p < 0 or j == skeleton.root_idx:
            new_local[:, j] = new_global[:, j]
        else:
            new_local[:, j] = new_global[:, p].transpose(-2, -1) @ new_global[:, j]

    return new_local.reshape(orig_shape)


def arrays_to_mfm_features(
    root_pos,
    root_wxyz,
    dof_29,
    skeleton,
    global_motion_rep,
    local_motion_rep,
    motion_rep,
    device,
    xml_path,
    change_t_pose_fn=None,  # change_t_pose_fn kept for compat but unused
):
    """Convert raw numpy arrays to global and local motion features.

    Pipeline:
      1. parquet_to_mfm(): MuJoCo coords -> local joint rotations
      2. change_t_pose_local_mats(): capture T-pose -> standard T-pose
      3. global_motion_rep(): forward pass to get normalized global features
      4. global_to_local(): convert to local representation
      5. Unnormalize both for use as inference inputs

    Args:
        root_pos: [T, 3] root position in MuJoCo coords.
        root_wxyz: [T, 4] root quaternion [w, x, y, z].
        dof_29: [T, 29] joint DOFs (radians).
        skeleton: Skeleton object (from bundle).
        global_motion_rep: Global motion rep module (from bundle).
        local_motion_rep: Local motion rep module (from bundle).
        motion_rep: Dual motion rep (from bundle, has dual_rep).
        device: Torch device string.
        xml_path: Path to g1_29dof.xml.
        change_t_pose_fn: T-pose correction function (from bundle).

    Returns:
        (global_motions, local_motions): both [1, T, D] torch tensors, unnormalized.
    """
    T_total = root_pos.shape[0]

    # Step 1: coordinate conversion + joint axis mapping
    root_trans, local_joint_rots = parquet_to_mfm(
        root_pos, root_wxyz, dof_29, skeleton, xml_path=xml_path
    )

    root_trans_t = torch.from_numpy(root_trans).float().to(device)
    local_rots_t = torch.from_numpy(local_joint_rots).float().to(device)

    # Step 2: T-pose correction (capture -> standard)
    local_rots_std = change_t_pose_local_mats(
        local_rots_t, skeleton, t_pose_from="capture", t_pose_to="standard"
    )

    # Step 3-4: forward pass through motion representation
    input_dict = {
        "local_joint_rots": local_rots_std[None],
        "translation": root_trans_t[None],
    }
    global_motions_norm = global_motion_rep(
        input_dict, to_normalize=True, lengths=torch.tensor([T_total], device=device)
    )

    # Step 5: unnormalize for inference
    global_motions = global_motion_rep.unnormalize(global_motions_norm)
    local_motions = local_motion_rep.unnormalize(
        motion_rep.dual_rep.global_to_local(
            global_motions_norm,
            is_normalized=True,
            to_normalize=True,
            lengths=torch.tensor([T_total], device=device),
        )
    )

    return global_motions, local_motions


def resample_qpos(qpos, src_fps, dst_fps):
    """Resample a [T, D] qpos trajectory. Linear for pos/joints, Slerp for quaternions.

    Layout: [pos(3), wxyz_quat(4), joints(N)]
    """
    T_src = qpos.shape[0]
    duration = (T_src - 1) / src_fps
    T_dst = int(round(duration * dst_fps)) + 1

    t_src = np.linspace(0, duration, T_src)
    t_dst = np.linspace(0, duration, T_dst)

    pos_joints = np.concatenate([qpos[:, :3], qpos[:, 7:]], axis=1)
    lerp = interp1d(t_src, pos_joints, axis=0, kind="linear")
    pos_joints_dst = lerp(t_dst)

    quat_wxyz = qpos[:, 3:7]
    rots = Rotation.from_quat(quat_wxyz[:, [1, 2, 3, 0]])
    slerp = Slerp(t_src, rots)
    quat_dst_xyzw = slerp(t_dst).as_quat()
    quat_dst_wxyz = quat_dst_xyzw[:, [3, 0, 1, 2]]

    out = np.zeros((T_dst, qpos.shape[1]))
    out[:, :3] = pos_joints_dst[:, :3]
    out[:, 3:7] = quat_dst_wxyz
    out[:, 7:] = pos_joints_dst[:, 3:]
    return out


def build_gt_qpos(root_pos, root_wxyz, dof_29):
    """Build qpos array. Converts root quaternion from wxyz to xyzw."""
    T = root_pos.shape[0]
    gt_qpos = np.zeros((T, 36))
    gt_qpos[:, :3] = root_pos
    gt_qpos[:, 3:7] = root_wxyz[:, [1, 2, 3, 0]]
    gt_qpos[:, 7:] = dof_29
    return gt_qpos


def extract_feature_from_bones_rep(
    x,
    motion_rep,
    feature,
    fetch_feat_idx=False,
    joint_groups=None,
    get_ee_pose_indices_fn=None,
):
    """Extract named features from motion representation tensors by index slicing.

    Args:
        x: [B, T, D] motion features tensor (or None if fetch_feat_idx=True).
        motion_rep: Motion rep with .indices dict and .root_mode attribute.
        feature: Feature name string.
        fetch_feat_idx: If True, return indices instead of sliced tensor.
        joint_groups: Optional joint group filter for EE features.
        get_ee_pose_indices_fn: Function for EE index lookup (from bundle).

    Returns:
        Sliced tensor [B, T, D_feat] or index array if fetch_feat_idx.
    """
    if feature == "root":
        idx = motion_rep.indices["root"]
        if fetch_feat_idx:
            return idx
        return x if x.shape[-1] == len(idx) else x[:, :, idx]

    elif feature == "root_without_hip_height":
        if motion_rep.root_mode == "global":
            idx = np.concatenate(
                [
                    motion_rep.indices["global_root_pos_2d"],
                    motion_rep.indices["global_root_heading"],
                ]
            )
        else:
            idx = np.concatenate(
                [
                    motion_rep.indices["local_root_rot_vel"],
                    motion_rep.indices["local_root_vel"],
                ]
            )
        if fetch_feat_idx:
            return idx
        return x if x.shape[-1] == len(idx) else x[:, :, idx]

    elif feature == "joint_positions_and_rotations_and_hip_height":
        if motion_rep.root_mode == "global":
            idx = np.concatenate(
                [
                    np.array(
                        [
                            i
                            for i in motion_rep.indices["global_root_pos"]
                            if i not in motion_rep.indices["global_root_pos_2d"]
                        ]
                    ),
                    motion_rep.indices["ric_data"],
                    motion_rep.indices["global_rot_data"],
                ]
            )
        else:
            idx = np.concatenate(
                [
                    motion_rep.indices["global_root_y"],
                    motion_rep.indices["ric_data"],
                    motion_rep.indices["global_rot_data"],
                ]
            )
        if fetch_feat_idx:
            return idx
        return x if x.shape[-1] == len(idx) else x[:, :, idx]

    elif feature in ("ee_pose", "end_effector_positions_and_rotations"):
        assert (
            get_ee_pose_indices_fn is not None
        ), "get_ee_pose_indices_fn required for ee_pose feature"
        _, _, feat_idx = get_ee_pose_indices_fn(motion_rep, joint_groups=joint_groups)
        if fetch_feat_idx:
            return feat_idx
        return x if x.shape[-1] == len(feat_idx) else x[:, :, feat_idx]

    elif feature == "pose":
        if fetch_feat_idx:
            return motion_rep.indices["all"]
        assert x.shape[-1] == len(motion_rep.indices["all"])
        return x

    else:
        raise NotImplementedError(f"Unknown feature: {feature}")


def interpolate_robot_motion_data(motion_data, target_fps):
    """Interpolate motion data fields to target FPS.

    Handles linear (positions, joints), Slerp (quaternions), and
    contact-aware interpolation (zero between non-contact frames).
    """
    n_frames = len(motion_data.robot_right_wrist_position)
    src_times = np.arange(n_frames) / motion_data.fps
    tgt_times = np.linspace(0, src_times[-1], int(src_times[-1] * target_fps))

    def _linear(data):
        return interp1d(src_times, np.asarray(data), kind="linear", axis=0)(
            tgt_times
        ).tolist()

    def _slerp(quat_data):
        quats = np.asarray(quat_data)
        return (
            Slerp(src_times, Rotation.from_quat(quats, scalar_first=True))(tgt_times)
            .as_quat(scalar_first=True)
            .tolist()
        )

    def _slerp_batch(quat_data):
        arr = np.asarray(quat_data)
        results = [_slerp(arr[:, i, :]) for i in range(arr.shape[1])]
        return np.array(results).transpose(1, 0, 2).tolist()

    def _frames(frame_data):
        arr = np.asarray(frame_data)
        pos = np.array(_linear(arr[:, :, :3]))
        rot = np.array(_slerp_batch(arr[:, :, 3:]))
        return np.concatenate([pos, rot], axis=2).tolist()

    def _contact_linear(data, part_ids):
        if not data:
            return []
        H, N = len(data), len(data[0])
        arr = np.concatenate(
            [np.asarray(data), np.asarray(part_ids).reshape(H, N, 1)], axis=-1
        )
        interp_result = interp1d(src_times, arr, kind="linear", axis=0)(tgt_times)
        nonzero_src = np.abs(arr) > 1e-8
        idx_lo = np.clip(
            np.searchsorted(src_times, tgt_times, side="right") - 1,
            0,
            len(src_times) - 1,
        )
        idx_hi = np.minimum(idx_lo + 1, len(src_times) - 1)
        mask = nonzero_src[idx_lo] & nonzero_src[idx_hi]
        return np.where(mask, interp_result, 0.0).tolist()

    motion_data.object_articulation = _linear(motion_data.object_articulation)
    motion_data.robot_right_finger_joints = _linear(
        motion_data.robot_right_finger_joints
    )
    motion_data.robot_left_finger_joints = _linear(motion_data.robot_left_finger_joints)
    motion_data.robot_right_wrist_position = _linear(
        motion_data.robot_right_wrist_position
    )
    motion_data.robot_left_wrist_position = _linear(
        motion_data.robot_left_wrist_position
    )
    motion_data.robot_right_wrist_wxyz = _slerp(motion_data.robot_right_wrist_wxyz)
    motion_data.robot_left_wrist_wxyz = _slerp(motion_data.robot_left_wrist_wxyz)
    motion_data.object_body_position = _linear(motion_data.object_body_position)
    motion_data.object_body_wxyz = _slerp_batch(motion_data.object_body_wxyz)
    motion_data.robot_right_frames = _frames(motion_data.robot_right_frames)
    motion_data.robot_left_frames = _frames(motion_data.robot_left_frames)

    for side in ("right", "left"):
        part_ids = getattr(motion_data, f"mano_{side}_object_contact_part_ids", [])
        for field in (
            "link_contact_positions",
            "object_contact_positions",
            "object_contact_normals",
        ):
            attr = f"mano_{side}_{field}"
            val = getattr(motion_data, attr, [])
            if val:
                setattr(motion_data, attr, _contact_linear(val, part_ids))

    return motion_data
