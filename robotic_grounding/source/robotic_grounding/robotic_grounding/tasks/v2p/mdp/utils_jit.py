# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JIT utility functions for the V2P environment."""

from typing import Tuple

import torch
from isaaclab.utils.math import quat_inv


@torch.jit.script
def refresh_jit(
    right_forces_w: torch.Tensor,
    left_forces_w: torch.Tensor,
    retargeted_left_contact_wrench_supports: torch.Tensor,
    retargeted_right_contact_wrench_supports: torch.Tensor,
    timestep_counter: torch.Tensor,
    right_contact_wrench_supports: torch.Tensor,
    left_contact_wrench_supports: torch.Tensor,
    object_position_e: torch.Tensor,
    object_orientation_e: torch.Tensor,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Pure-tensor derivations for DualHandsObjectTrackingCommand.refresh_tensors.

    Fuses the force-reduction, in-contact, reference-indexing, 1e-3 wrench-mask
    and pose-squeeze computations into a single scripted graph. The wrench support
    body loop stays in Python (``_compute_contact_wrench_supports``) because it
    calls external utility functions and iterates over object bodies.

    Returns (in order):
        0  right_force_sq_per_link                    (N, num_right_links)
        1  left_force_sq_per_link                     (N, num_left_links)
        2  right_link_in_contact                      (N, num_right_links) bool
        3  left_link_in_contact                       (N, num_left_links)  bool
        4  right_in_contact                           (N,) bool
        5  left_in_contact                            (N,) bool
        6  in_contact                                 (N,) bool
        7  ref_L                                      (N, num_bodies, B)
        8  ref_R                                      (N, num_bodies, B)
        9  mask_L                                     (N, num_bodies, B) bool
       10  mask_R                                     (N, num_bodies, B) bool
       11  ref_active_per_cell                        (N, num_bodies, B) bool
       12  ref_active_per_body                        (N, num_bodies)   bool
       13  ref_active_global                          (N,)              bool
       14  right_wrench_cmd_active                    (N, num_bodies, B) bool
       15  left_wrench_cmd_active                     (N, num_bodies, B) bool
       16  right_wrench_cur_active                    (N, num_bodies, B) bool
       17  left_wrench_cur_active                     (N, num_bodies, B) bool
       18  right_wrench_cmd_active_per_body           (N, num_bodies)   bool
       19  left_wrench_cmd_active_per_body            (N, num_bodies)   bool
       20  right_wrench_cur_active_per_body           (N, num_bodies)   bool
       21  left_wrench_cur_active_per_body            (N, num_bodies)   bool
       22  object_position_e_sq                       (N, 3)
       23  object_wxyz_e_sq                           (N, 4)
    """
    eps = 1e-6
    in_contact_force_threshold = 1e-3
    support_thr = 1e-3

    # Per-link squared-force reduction (contact_force_reward / contact_force_range_reward).
    right_force_sq_per_link = right_forces_w.square().sum(dim=-1).mean(dim=1).sum(dim=1)
    left_force_sq_per_link = left_forces_w.square().sum(dim=-1).mean(dim=1).sum(dim=1)
    right_link_in_contact = right_force_sq_per_link > in_contact_force_threshold
    left_link_in_contact = left_force_sq_per_link > in_contact_force_threshold

    # Per-env in-contact (contact_wrench_reward / contact_wrench_continuous_reward).
    # Use torch.linalg.vector_norm explicitly — Tensor.norm(dim=...) trips the
    # TorchScript overload resolver on the default `p` argument.
    right_force_mag = torch.linalg.vector_norm(
        torch.linalg.vector_norm(right_forces_w, dim=1), dim=-1
    )
    left_force_mag = torch.linalg.vector_norm(
        torch.linalg.vector_norm(left_forces_w, dim=1), dim=-1
    )
    right_in_contact = (
        (right_force_mag > in_contact_force_threshold).any(dim=-1).any(dim=-1)
    )
    left_in_contact = (
        (left_force_mag > in_contact_force_threshold).any(dim=-1).any(dim=-1)
    )
    in_contact = right_in_contact | left_in_contact

    # Reference wrench indexing and active masks at eps=1e-6 ("any non-zero").
    ref_L = retargeted_left_contact_wrench_supports[timestep_counter]
    ref_R = retargeted_right_contact_wrench_supports[timestep_counter]
    mask_L = ref_L > eps
    mask_R = ref_R > eps
    ref_active_per_cell = mask_L | mask_R
    ref_active_per_body = ref_active_per_cell.any(dim=-1)
    ref_active_global = ref_active_per_body.any(dim=-1)

    # 1e-3 "meaningful support" masks (contact_wrench_support_reward,
    # unintended_contact_penalty, missed_contact_penalty).
    right_wrench_cmd_active = ref_R > support_thr
    left_wrench_cmd_active = ref_L > support_thr
    right_wrench_cur_active = right_contact_wrench_supports > support_thr
    left_wrench_cur_active = left_contact_wrench_supports > support_thr
    right_wrench_cmd_active_per_body = right_wrench_cmd_active.any(dim=-1)
    left_wrench_cmd_active_per_body = left_wrench_cmd_active.any(dim=-1)
    right_wrench_cur_active_per_body = right_wrench_cur_active.any(dim=-1)
    left_wrench_cur_active_per_body = left_wrench_cur_active.any(dim=-1)

    # Object pose body-dim squeeze.
    object_position_e_sq = object_position_e.squeeze(1)
    object_wxyz_e_sq = object_orientation_e.squeeze(1)

    return (
        right_force_sq_per_link,
        left_force_sq_per_link,
        right_link_in_contact,
        left_link_in_contact,
        right_in_contact,
        left_in_contact,
        in_contact,
        ref_L,
        ref_R,
        mask_L,
        mask_R,
        ref_active_per_cell,
        ref_active_per_body,
        ref_active_global,
        right_wrench_cmd_active,
        left_wrench_cmd_active,
        right_wrench_cur_active,
        left_wrench_cur_active,
        right_wrench_cmd_active_per_body,
        left_wrench_cmd_active_per_body,
        right_wrench_cur_active_per_body,
        left_wrench_cur_active_per_body,
        object_position_e_sq,
        object_wxyz_e_sq,
    )


@torch.jit.script
def _quat_rotate_broadcast(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply quaternion rotation to vectors, supporting broadcasting.

    Unlike quat_apply which reshapes to (-1, N) and requires matching shapes,
    this uses element-wise ops that broadcast naturally.

    Args:
        q: Quaternion (wxyz). Shape is (..., 4), broadcastable against *v*.
        v: Vectors. Shape is (..., 3).

    Returns:
        Rotated vectors. Shape is broadcast(q[..., :3], v).
    """
    xyz = q[..., 1:]
    t = torch.linalg.cross(xyz, v) * 2
    return v + q[..., 0:1] * t + torch.linalg.cross(xyz, t)


@torch.jit.script
def wrench_preprocess_jit(
    contact_positions_w: torch.Tensor,
    contact_forces_first_hist_w: torch.Tensor,
    object_com_position_w: torch.Tensor,
    object_com_orientation_w: torch.Tensor,
    num_envs: int,
    num_bodies: int,
    num_robot_contacts: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse the per-refresh wrench-support preprocessing into one scripted graph.

    Inlines the `subtract_frame_transforms` logic so we don't pay the non-scripted
    Python wrapper. Returns contact positions and contact normals, both expressed in
    the object COM frame and masked by the active-contact flag.

    Args:
        contact_positions_w: (N, bodies, num_robot_contacts, 3) world-frame positions.
        contact_forces_first_hist_w: (N, bodies, num_robot_contacts, 3) world-frame forces,
            already sliced to the latest history entry.
        object_com_position_w: (N, bodies, 1, 3) unexpanded COM position.
        object_com_orientation_w: (N, bodies, 1, 4) unexpanded COM quat (wxyz).
        num_envs: Number of environments; used to reshape the per-cell active mask.
        num_bodies: Number of object bodies; used to reshape the active mask.
        num_robot_contacts: Number of per-body robot contact points; used to
            reshape the active mask.

    Returns:
        contact_positions_com: (N, bodies, num_robot_contacts, 3)
        contact_normals_com:   (N, bodies, num_robot_contacts, 3)
    """
    active_contact = (
        torch.linalg.vector_norm(contact_positions_w, dim=-1) > 1e-3
    )  # (N, bodies, num_robot_contacts)
    active_mask = active_contact.view(num_envs, num_bodies, num_robot_contacts, 1).to(
        contact_positions_w.dtype
    )

    # quat_inv on (N, bodies, 1, 4) — num_robot_contacts times cheaper
    # than the old (N, bodies, num_robot_contacts, 4) path.
    q10 = quat_inv(object_com_orientation_w)

    # Subtraction broadcasts (N,B,C,3) - (N,B,1,3). The rotation helper
    # broadcasts q10 (N,B,1,4) against the (N,B,C,3) result via cross/mul.
    contact_positions_com = _quat_rotate_broadcast(
        q10, contact_positions_w - object_com_position_w
    )
    contact_positions_com = contact_positions_com * active_mask

    # Normals = unit-normalized force direction (contact sensors report normal forces only).
    normals_w = contact_forces_first_hist_w / torch.linalg.vector_norm(
        contact_forces_first_hist_w, dim=-1, keepdim=True
    ).clamp(min=1e-5)
    contact_normals_com = _quat_rotate_broadcast(q10, normals_w)
    contact_normals_com = contact_normals_com * active_mask

    return contact_positions_com, contact_normals_com


@torch.jit.script
def friction_cone_edges_jit(
    normals: torch.Tensor,
    cos_t: torch.Tensor,
    sin_t: torch.Tensor,
    friction_coefficients: float,
    eps: float,
) -> torch.Tensor:
    """Polyhedral friction-cone rays with the contact normal appended.

    Mirrors `v2p.mdp.utils.compute_friction_cone_edges` but is JIT-compiled and
    uses `torch.linalg.vector_norm` (Tensor.norm's default-p overload trips JIT).

    Args:
        normals: (batch_size, num_contacts, 3) contact normals.
        cos_t: (1, K, 1) cosines of the friction-cone edge phase angles.
        sin_t: (1, K, 1) sines of the friction-cone edge phase angles.
        friction_coefficients: Scalar friction coefficient.
        eps: Scalar epsilon for safe divisions / sign handling.

    Returns:
        (batch_size, num_contacts, K+1, 3)
    """
    batch_size = normals.shape[0]
    num_contacts = normals.shape[1]

    # Frisvad 2012 tangent basis, inlined (compute_tangent_basis).
    normals_flat = normals.reshape(-1, 3)
    nx = normals_flat[:, 0]
    ny = normals_flat[:, 1]
    nz = normals_flat[:, 2]
    sign = torch.where(
        nz >= 0,
        torch.ones_like(nz),
        -torch.ones_like(nz),
    )
    den = sign + nz
    den = torch.where(den.abs() < eps, sign * eps, den)
    a = -1.0 / den
    b = nx * ny * a
    t1 = torch.stack(
        (1.0 + sign * nx * nx * a, sign * b, -sign * nx),
        dim=-1,
    )
    t2 = torch.stack(
        (b, sign + ny * ny * a, -ny),
        dim=-1,
    )
    t1 = t1 / torch.linalg.vector_norm(t1, dim=-1, keepdim=True).clamp(min=eps)
    t2 = t2 / torch.linalg.vector_norm(t2, dim=-1, keepdim=True).clamp(min=eps)

    n_exp = normals_flat.unsqueeze(1)  # (B*C, 1, 3)
    t1_exp = t1.unsqueeze(1)
    t2_exp = t2.unsqueeze(1)

    edges = n_exp + friction_coefficients * (cos_t * t1_exp + sin_t * t2_exp)
    edges = edges / torch.linalg.vector_norm(edges, dim=-1, keepdim=True).clamp(min=eps)
    edges = torch.cat([edges, n_exp], dim=1)  # append_normal=True

    num_edges = cos_t.shape[1] + 1
    return edges.view(batch_size, num_contacts, num_edges, 3)


@torch.jit.script
def wrench_support_one_body_jit(
    contact_points: torch.Tensor,
    contact_normals: torch.Tensor,
    cos_t: torch.Tensor,
    sin_t: torch.Tensor,
    basis: torch.Tensor,
    rc: float,
    friction_coefficients: float,
) -> torch.Tensor:
    """Compute the wrench-space support for one object body.

    Fuses `compute_wrench_space` + `compute_wrench_space_support_function` for a
    single body.

    Args:
        contact_points:  (N, num_contacts, 3) in object COM frame.
        contact_normals: (N, num_contacts, 3) in object COM frame.
        cos_t:           (1, K, 1) cosines of the friction-cone edge phase angles.
        sin_t:           (1, K, 1) sines of the friction-cone edge phase angles.
        basis:           (num_basis, 6) sampled wrench-space basis directions.
        rc:              body radius scale for torque.
        friction_coefficients: friction coefficient scalar.

    Returns:
        support: (N, num_basis) non-negative support function.
    """
    eps = 1e-6
    batch_size = contact_points.shape[0]

    # Re-normalize and active mask for the per-body subset.
    normals_norm = torch.linalg.vector_norm(contact_normals, dim=-1, keepdim=True)
    normals = contact_normals / normals_norm.clamp(min=eps)
    contact_is_active = torch.linalg.vector_norm(normals, dim=-1) > 1e-3

    forces = friction_cone_edges_jit(
        normals, cos_t, sin_t, friction_coefficients, eps
    )  # (N, num_contacts, K+1, 3)

    torques = torch.cross(contact_points.unsqueeze(2).expand_as(forces), forces, dim=-1)

    wrench_space = torch.cat(
        (forces, torques / rc), dim=-1
    )  # (N, num_contacts, K+1, 6)
    wrench_space = wrench_space * contact_is_active.view(batch_size, -1, 1, 1).to(
        wrench_space.dtype
    )
    wrench_space = wrench_space.view(batch_size, -1, 6).transpose(1, 2).contiguous()

    # Support function: max over wrench-space contributions along each basis direction.
    support = torch.matmul(basis.unsqueeze(0), wrench_space).amax(dim=-1)
    return torch.clamp(support, min=0.0)


@torch.jit.script
def resample_compute_tensors_jit(
    tc: torch.Tensor,
    env_origins_sel: torch.Tensor,
    retargeted_object_body_position: torch.Tensor,
    retargeted_object_body_wxyz: torch.Tensor,
    retargeted_right_wrist_position: torch.Tensor,
    retargeted_right_wrist_wxyz: torch.Tensor,
    retargeted_left_wrist_position: torch.Tensor,
    retargeted_left_wrist_wxyz: torch.Tensor,
    retargeted_right_finger_joints: torch.Tensor,
    retargeted_left_finger_joints: torch.Tensor,
    right_soft_joint_pos_limits_sel: torch.Tensor,
    left_soft_joint_pos_limits_sel: torch.Tensor,
    reset_finger_openness: float,
    n: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Fused pure-tensor derivations for DualHandsObjectTrackingCommand._resample_command.

    Takes the timestep indices ``tc`` (already gathered from ``timestep_counter[env_ids]``),
    the selected ``env_origins`` offsets, the retargeted motion buffers, and the per-env
    soft joint limit slices. Returns every tensor the Python wrapper needs to scatter into
    ``self.*`` buffers or pass to ``write_*_to_sim``.

    Returns (in order):
        0  object_pose                  (n, num_bodies, 7) world-frame
        1  object_velocity              (n, num_bodies, 6) zeros
        2  right_hand_wrist_position_e  (n, 3) env-frame (for reset_*_wrist scatter)
        3  right_hand_wrist_wxyz        (n, 4)
        4  left_hand_wrist_position_e   (n, 3)
        5  left_hand_wrist_wxyz         (n, 4)
        6  right_hand_wrist_pose        (n, 7) world-frame
        7  left_hand_wrist_pose         (n, 7)
        8  wrist_zero_velocity          (n, 6) shared by both wrists
        9  right_hand_finger_joint_pos  (n, num_right_finger_joints)
       10  left_hand_finger_joint_pos   (n, num_left_finger_joints)
       11  finger_zero_velocity         (n, num_right_finger_joints) shared across hands
    """
    # ── Object pose ──────────────────────────────────────────────────────────
    object_position_e = retargeted_object_body_position[tc]
    object_wxyz = retargeted_object_body_wxyz[tc]
    object_pose = torch.cat(
        [object_position_e + env_origins_sel.unsqueeze(1), object_wxyz], dim=-1
    )
    object_velocity = torch.zeros(
        object_pose.size(0),
        object_pose.size(1),
        6,
        device=object_pose.device,
        dtype=object_pose.dtype,
    )

    # ── Wrist poses (both env and world frames) ──────────────────────────────
    right_hand_wrist_position_e = retargeted_right_wrist_position[tc]
    right_hand_wrist_wxyz = retargeted_right_wrist_wxyz[tc]
    left_hand_wrist_position_e = retargeted_left_wrist_position[tc]
    left_hand_wrist_wxyz = retargeted_left_wrist_wxyz[tc]

    right_hand_wrist_pose = torch.cat(
        [right_hand_wrist_position_e + env_origins_sel, right_hand_wrist_wxyz],
        dim=-1,
    )
    left_hand_wrist_pose = torch.cat(
        [left_hand_wrist_position_e + env_origins_sel, left_hand_wrist_wxyz],
        dim=-1,
    )
    wrist_zero_velocity = torch.zeros(
        n,
        6,
        device=right_hand_wrist_pose.device,
        dtype=right_hand_wrist_pose.dtype,
    )

    # ── Finger joint targets (scaled then clamped to soft limits) ────────────
    finger_factor = torch.rand(n, 1, device=tc.device) * reset_finger_openness
    right_hand_finger_joint_pos = finger_factor * retargeted_right_finger_joints[tc]
    left_hand_finger_joint_pos = finger_factor * retargeted_left_finger_joints[tc]
    right_hand_finger_joint_pos.clamp_(
        right_soft_joint_pos_limits_sel[..., 0],
        right_soft_joint_pos_limits_sel[..., 1],
    )
    left_hand_finger_joint_pos.clamp_(
        left_soft_joint_pos_limits_sel[..., 0],
        left_soft_joint_pos_limits_sel[..., 1],
    )
    finger_zero_velocity = torch.zeros_like(right_hand_finger_joint_pos)

    return (
        object_pose,
        object_velocity,
        right_hand_wrist_position_e,
        right_hand_wrist_wxyz,
        left_hand_wrist_position_e,
        left_hand_wrist_wxyz,
        right_hand_wrist_pose,
        left_hand_wrist_pose,
        wrist_zero_velocity,
        right_hand_finger_joint_pos,
        left_hand_finger_joint_pos,
        finger_zero_velocity,
    )


@torch.jit.script
def contact_wrench_support_reward_jit(
    right_cmd_active: torch.Tensor,
    right_cur_active: torch.Tensor,
    left_cmd_active: torch.Tensor,
    left_cur_active: torch.Tensor,
    right_cmd_active_per_body: torch.Tensor,
    left_cmd_active_per_body: torch.Tensor,
    right_cmd_supports: torch.Tensor,
    right_cur_supports: torch.Tensor,
    left_cmd_supports: torch.Tensor,
    left_cur_supports: torch.Tensor,
    tolerance: float,
    var: float,
) -> torch.Tensor:
    """Pure-tensor body of contact_wrench_support_reward.

    Stacks right+left hands along a new leading axis so each downstream op
    (clamp / square / exp / sum / division) runs once on a doubled tensor
    instead of twice on separate tensors. Cuts kernel-launch overhead roughly
    in half for the hot reward path.

    Inputs are all already cached by ``DualHandsObjectTrackingCommand.refresh_tensors``;
    the wrapper just hands them to this scripted graph.
    """
    # Fuse right+left into one leading-axis batch.
    #   per-cell bool masks:          (N, B, K) → (2, N, B, K)
    #   per-body bool masks:          (N, B)    → (2, N, B)
    #   wrench-support scalars:       (N, B, K) → (2, N, B, K)
    cmd_active = torch.stack((right_cmd_active, left_cmd_active), dim=0)
    cur_active = torch.stack((right_cur_active, left_cur_active), dim=0)
    cmd_active_per_body = torch.stack(
        (right_cmd_active_per_body, left_cmd_active_per_body), dim=0
    )
    cmd_supports = torch.stack((right_cmd_supports, left_cmd_supports), dim=0)
    cur_supports = torch.stack((right_cur_supports, left_cur_supports), dim=0)

    # Counts (float-cast so downstream divisions stay in float).
    cmd_num = cmd_active.sum(dim=-1).float().clamp(min=1e-6)  # (2, N, B)
    cmd_num_body = cmd_active_per_body.sum(dim=-1).float().clamp(min=1e-6)  # (2, N)
    num_active_hands = (cmd_num_body > 1e-3).float().sum(dim=0).clamp(min=1e-6)  # (N,)

    # Current supports must be within (1 ± tolerance) of the commanded supports.
    # Both directional excess terms get squared and summed into a per-cell loss.
    better = ((1.0 - tolerance) * cmd_supports - cur_supports).clamp(min=0.0)
    too_large = (cur_supports - (1.0 + tolerance) * cmd_supports).clamp(min=0.0)
    loss = better.square() + too_large.square()

    # Per-body inclusion reward: only count cells where the command and the agent
    # both have meaningful support, then average over the commanded basis count.
    reward = ((cmd_active & cur_active).float() * torch.exp(-loss / var)).sum(
        dim=-1
    ) / cmd_num  # (2, N, B)
    per_hand_reward = reward.sum(dim=-1) / cmd_num_body  # (2, N)
    return per_hand_reward.sum(dim=0) / num_active_hands  # (N,)


@torch.jit.script
def unintended_contact_penalty_jit(
    right_cmd_active_per_body: torch.Tensor,
    right_cur_active_per_body: torch.Tensor,
    left_cmd_active_per_body: torch.Tensor,
    left_cur_active_per_body: torch.Tensor,
    right_cur_supports: torch.Tensor,
    left_cur_supports: torch.Tensor,
    num_bodies: int,
) -> torch.Tensor:
    """Pure-tensor body of unintended_contact_penalty."""
    right_cmd_num = right_cmd_active_per_body.sum(dim=-1).float()
    left_cmd_num = left_cmd_active_per_body.sum(dim=-1).float()

    right_unintended = torch.logical_and(
        ~right_cmd_active_per_body, right_cur_active_per_body
    )
    left_unintended = torch.logical_and(
        ~left_cmd_active_per_body, left_cur_active_per_body
    )

    # Continuous penalty: when the command says "no contact" on a body, score
    # the squared-then-mean-over-basis support magnitude there.
    right_unintended_support = (~right_cmd_active_per_body).float() * (
        right_cur_supports.clamp(min=0.0).square().mean(dim=-1)
    )
    right_unintended_support = right_unintended_support.sum(dim=-1) / (
        float(num_bodies) - right_cmd_num
    ).clamp(min=1e-3)

    left_unintended_support = (~left_cmd_active_per_body).float() * (
        left_cur_supports.clamp(min=0.0).square().mean(dim=-1)
    )
    left_unintended_support = left_unintended_support.sum(dim=-1) / (
        float(num_bodies) - left_cmd_num
    ).clamp(min=1e-3)

    return (
        right_unintended.float().mean(dim=-1)
        + right_unintended_support
        + left_unintended.float().mean(dim=-1)
        + left_unintended_support
    )


@torch.jit.script
def hand_keypoints_tracking_jit(
    left_wrist_cmd: torch.Tensor,
    right_wrist_cmd: torch.Tensor,
    left_fingertip_cmd: torch.Tensor,
    right_fingertip_cmd: torch.Tensor,
    left_wrist_cur: torch.Tensor,
    right_wrist_cur: torch.Tensor,
    left_fingertip_cur: torch.Tensor,
    right_fingertip_cur: torch.Tensor,
    var: float,
    threshold: float,
) -> torch.Tensor:
    """Pure-tensor body of hand_keypoints_tracking_exp.

    Fuses per-hand keypoint assembly (wrist + fingertips) + L2 error + the
    ``exp(-(err - threshold).clamp(min=0)/var)`` decay + the per-keypoint mean
    into a single scripted graph. The two hands return the symmetric mean.

    Args:
        left_wrist_cmd:       (N, 3) commanded left wrist position in env frame.
        right_wrist_cmd:      (N, 3) commanded right wrist position in env frame.
        left_fingertip_cmd:   (N, F_left, 3) commanded left fingertip positions.
        right_fingertip_cmd:  (N, F_right, 3) commanded right fingertip positions.
        left_wrist_cur:       (N, 3) current left wrist position.
        right_wrist_cur:      (N, 3) current right wrist position.
        left_fingertip_cur:   (N, F_left, 3) current left fingertip positions.
        right_fingertip_cur:  (N, F_right, 3) current right fingertip positions.
        var:                  Exp decay scale.
        threshold:            Saturation threshold; errors below it are clamped to zero.

    Returns:
        Reward tensor of shape ``(N,)``.
    """
    # Build the per-hand keypoint stacks inline (wrist as the first keypoint).
    left_cmd = torch.cat([left_wrist_cmd.unsqueeze(1), left_fingertip_cmd], dim=1)
    right_cmd = torch.cat([right_wrist_cmd.unsqueeze(1), right_fingertip_cmd], dim=1)
    left_cur = torch.cat([left_wrist_cur.unsqueeze(1), left_fingertip_cur], dim=1)
    right_cur = torch.cat([right_wrist_cur.unsqueeze(1), right_fingertip_cur], dim=1)

    # Per-keypoint squared L2 error.
    left_error = (left_cmd - left_cur).square().sum(dim=-1)
    right_error = (right_cmd - right_cur).square().sum(dim=-1)

    # Saturated exponential decay, averaged across keypoints per hand.
    left_reward = torch.exp(-(left_error - threshold).clamp(min=0.0) / var).mean(dim=-1)
    right_reward = torch.exp(-(right_error - threshold).clamp(min=0.0) / var).mean(
        dim=-1
    )

    return (left_reward + right_reward) / 2.0


@torch.jit.script
def missed_contact_penalty_jit(
    right_cmd_active: torch.Tensor,
    right_cur_active: torch.Tensor,
    left_cmd_active: torch.Tensor,
    left_cur_active: torch.Tensor,
    right_cmd_active_per_body: torch.Tensor,
    left_cmd_active_per_body: torch.Tensor,
) -> torch.Tensor:
    """Pure-tensor body of missed_contact_penalty.

    For each hand, counts the fraction of commanded basis directions per body
    that the agent failed to cover, averages over bodies the command actually
    requests, then averages over hands that have any commanded contact at all.
    """
    # Right hand.
    right_missed = right_cmd_active & ~right_cur_active
    right_n_expected = right_cmd_active.sum(dim=-1).float()
    right_n_missed = right_missed.sum(dim=-1).float()
    right_missing_frac = right_n_missed / right_n_expected.clamp(min=1e-6)
    right_body_has_contact = right_n_expected > 0
    right_n_active_bodies = right_body_has_contact.sum(dim=-1).float().clamp(min=1e-6)
    right_penalty = (right_missing_frac * right_body_has_contact.float()).sum(
        dim=-1
    ) / right_n_active_bodies

    # Left hand.
    left_missed = left_cmd_active & ~left_cur_active
    left_n_expected = left_cmd_active.sum(dim=-1).float()
    left_n_missed = left_missed.sum(dim=-1).float()
    left_missing_frac = left_n_missed / left_n_expected.clamp(min=1e-6)
    left_body_has_contact = left_n_expected > 0
    left_n_active_bodies = left_body_has_contact.sum(dim=-1).float().clamp(min=1e-6)
    left_penalty = (left_missing_frac * left_body_has_contact.float()).sum(
        dim=-1
    ) / left_n_active_bodies

    # Hand-active normalization (count hands that have any commanded contact).
    right_hand_active = right_cmd_active_per_body.any(dim=-1)
    left_hand_active = left_cmd_active_per_body.any(dim=-1)
    num_active_hands = (right_hand_active.float() + left_hand_active.float()).clamp(
        min=1e-6
    )

    return (right_penalty + left_penalty) / num_active_hands
