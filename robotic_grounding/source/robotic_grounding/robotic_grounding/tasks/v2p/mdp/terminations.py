# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def hand_to_object_away_from_trajectory(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
) -> torch.Tensor:
    """Terminate when hands deviate too far from the commanded trajectory.

    Compares per-hand wrist-to-object distances against the commanded
    wrist-to-object distances and terminates if any hand exceeds
    `threshold` times its commanded distance.

    Args:
        env: The environment instance.
        command_name: The name of the command term.
        threshold: Ratio threshold for termination.

    Returns:
        Tensor of shape (num_envs,) indicating whether to terminate.
    """
    command = env.command_manager.get_term(command_name)

    right_hand_wrist_object_position_difference_command = torch.norm(
        command.right_hand_wrist_pose_command_e[:, :3]
        - command.object_body_position_command_e,
        dim=-1,
    )
    left_hand_wrist_object_position_difference_command = torch.norm(
        command.left_hand_wrist_pose_command_e[:, :3]
        - command.object_body_position_command_e,
        dim=-1,
    )

    right_hand_wrist_object_position_difference = torch.norm(
        command.right_robot.data.body_link_pos_w[:, command.right_wrist_body_id]
        - command.object_position_w,
        dim=-1,
    ).squeeze()
    left_hand_wrist_object_position_difference = torch.norm(
        command.left_robot.data.body_link_pos_w[:, command.left_wrist_body_id]
        - command.object_position_w,
        dim=-1,
    ).squeeze()

    right_hand_difference_ratio = (
        right_hand_wrist_object_position_difference
        / right_hand_wrist_object_position_difference_command
    )
    left_hand_difference_ratio = (
        left_hand_wrist_object_position_difference
        / left_hand_wrist_object_position_difference_command
    )

    return torch.logical_and(
        right_hand_difference_ratio > threshold,
        left_hand_difference_ratio > threshold,
    )


def hand_wrist_away_from_trajectory(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
) -> torch.Tensor:
    """Terminate when the hands are away from the trajectory."""
    command = env.command_manager.get_term(command_name)
    right_hand_position_difference = torch.norm(
        command.right_hand_wrist_pose_command_e[:, :3]
        - command.right_hand_wrist_position_e,
        dim=-1,
    )
    left_hand_position_difference = torch.norm(
        command.left_hand_wrist_pose_command_e[:, :3]
        - command.left_hand_wrist_position_e,
        dim=-1,
    )
    return torch.logical_or(
        right_hand_position_difference > threshold,
        left_hand_position_difference > threshold,
    )


def object_away_from_trajectory_z(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
) -> torch.Tensor:
    """Terminate when the object is away from the trajectory.

    Args:
        env: The environment instance.
        command_name: The name of the command.
        threshold: The threshold for the termination.

    Returns:
        Tensor of shape (num_envs,) indicating whether to terminate.
    """
    command = env.command_manager.get_term(command_name)
    object_position_z_difference = torch.abs(
        command.object_body_position_command_e[..., 2]
        - command.object_position_e[..., 2].squeeze()
    )
    return object_position_z_difference > threshold


def object_away_from_trajectory(
    env: ManagerBasedRLEnv,
    command_name: str,
    position_threshold: float,
    orientation_threshold: float,
    grace_steps: int = 0,
    debug_attr: str | None = None,
) -> torch.Tensor:
    """Terminate when the object is away from the trajectory.

    ``grace_steps`` and ``debug_attr`` are optional compatibility parameters for
    launch snapshots or eval helpers that need a startup grace window or want to
    inspect the per-env object errors without changing the termination result.
    """
    command = env.command_manager.get_term(command_name)
    object_position_difference = torch.norm(
        command.object_body_position_command_e - command.object_position_e,
        dim=-1,
    )
    object_orientation_difference = math_utils.quat_error_magnitude(
        command.object_orientation_e,
        command.object_body_wxyz_command_e,
    )
    over_position = object_position_difference > position_threshold
    over_orientation = object_orientation_difference > orientation_threshold
    done = torch.logical_or(over_position.any(dim=-1), over_orientation.any(dim=-1))
    if grace_steps > 0:
        done = torch.logical_and(done, command.timestep_counter >= grace_steps)
    if debug_attr is not None:
        try:
            setattr(
                command,
                debug_attr,
                {
                    "position_error_m": object_position_difference.detach().clone(),
                    "orientation_error_rad": object_orientation_difference.detach().clone(),
                    "over_position": over_position.detach().clone(),
                    "over_orientation": over_orientation.detach().clone(),
                    "done": done.detach().clone(),
                    "timestep_counter": command.timestep_counter.detach().clone(),
                    "position_threshold": float(position_threshold),
                    "orientation_threshold": float(orientation_threshold),
                },
            )
        except RuntimeError:
            pass
    return done


def timestep_timeout(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Terminate when the command is completed."""
    command = env.command_manager.get_term(command_name)
    return command.timestep_counter >= command.retargeted_horizon - 1


def hand_object_contact_violation(
    env: ManagerBasedRLEnv,
    command_name: str = "dual_hands_object_tracking_command",
    grasp_distance_threshold: float = 0.02,
    contact_force_threshold_n: float = 0.5,
    grace_frames: int = 5,
) -> torch.Tensor:
    """Terminate if MoCap says firm grasp but sim contact force is below threshold.

    Paper Section 3.3: when the reference fingertip-to-object distance falls
    below ``grasp_distance_threshold`` (``xi_c``), the simulated contact force
    on that hand must exceed ``contact_force_threshold_n``. If not, increment
    a per-env violation counter; trigger termination once the counter reaches
    ``grace_frames`` consecutive frames.

    Notes / caveats:
      * Distance is fingertip-to-nearest-object-body-COM in **command** frame
        (i.e. uses the demo's target object pose). This is a per-fingertip
        proxy for the paper's per-MANO-joint signed distance.
      * Force gate is per-hand (env-level), not per-fingertip — the contact-
        sensor plumbing only exposes per-link force magnitudes after a
        reduction, so we use the union "any link on this hand has force >
        threshold" as the contact signal. The MoCap demand still discriminates
        per fingertip via the distance threshold; only the force read-out is
        coarsened.
      * Streak counter is stored on the env (``env._hocv_violation_count``),
        sized to ``num_envs`` and reset at episode boundaries via the
        ``episode_length_buf == 0`` check (lazy reset — no event hook needed).

    Args:
        env: The environment instance.
        command_name: Name of the dual-hand-object tracking command term.
        grasp_distance_threshold: Distance below which MoCap is interpreted as
            firm grasp (m). Default 0.02.
        contact_force_threshold_n: Per-hand contact force threshold (N).
            Default 0.5.
        grace_frames: Consecutive violation frames before triggering
            termination. Default 5.

    Returns:
        Bool tensor of shape ``(num_envs,)`` — True where the episode should
        terminate.
    """
    cmd = env.command_manager.get_term(command_name)

    right_tips_e = cmd.right_hand_fingertip_position_e  # (N, F_r, 3)
    left_tips_e = cmd.left_hand_fingertip_position_e  # (N, F_l, 3)
    obj_pos_cmd_e = cmd.object_body_position_command_e  # (N, num_bodies, 3)

    # Per-fingertip distance to the nearest commanded object body position.
    right_pair_dist = torch.norm(
        right_tips_e.unsqueeze(2) - obj_pos_cmd_e.unsqueeze(1), dim=-1
    )  # (N, F_r, num_bodies)
    right_tip_min_dist = right_pair_dist.min(dim=-1).values  # (N, F_r)
    left_pair_dist = torch.norm(
        left_tips_e.unsqueeze(2) - obj_pos_cmd_e.unsqueeze(1), dim=-1
    )  # (N, F_l, num_bodies)
    left_tip_min_dist = left_pair_dist.min(dim=-1).values  # (N, F_l)

    # Per-hand: does the MoCap demand contact on at least one fingertip?
    right_mocap_demands = (right_tip_min_dist < grasp_distance_threshold).any(dim=-1)
    left_mocap_demands = (left_tip_min_dist < grasp_distance_threshold).any(dim=-1)

    # Per-hand sim contact force: sum the per-link force magnitude norms over
    # the contact-sensor's history mean, then check if any link exceeds
    # ``contact_force_threshold_n``. This bypasses the hard-coded 1e-3 N
    # threshold inside ``refresh_jit``'s ``right_link_in_contact``.
    right_forces_w = cmd.right_hand_object_contact_forces_w  # (N, H, B, L, 3)
    left_forces_w = cmd.left_hand_object_contact_forces_w
    # Reduce over (history, bodies) -> (N, L) per-link force magnitude.
    right_link_force_n = (
        torch.linalg.vector_norm(right_forces_w, dim=-1).mean(dim=1).sum(dim=1)
    )  # (N, L)
    left_link_force_n = (
        torch.linalg.vector_norm(left_forces_w, dim=-1).mean(dim=1).sum(dim=1)
    )  # (N, L)
    right_sim_has_contact = (right_link_force_n > contact_force_threshold_n).any(
        dim=-1
    )  # (N,)
    left_sim_has_contact = (left_link_force_n > contact_force_threshold_n).any(
        dim=-1
    )  # (N,)

    # Per-hand violation: MoCap demands grasp AND sim shows no force above
    # threshold. Combine across hands with OR — either hand failing counts.
    right_violation = right_mocap_demands & (~right_sim_has_contact)
    left_violation = left_mocap_demands & (~left_sim_has_contact)
    step_violation = right_violation | left_violation  # (N,)

    # Per-env consecutive-violation streak counter. Lazy init on env.
    if (
        not hasattr(env, "_hocv_violation_count")
        or env._hocv_violation_count.shape[0] != env.num_envs
    ):
        env._hocv_violation_count = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )

    # Reset streaks at episode boundaries (just-started episodes have
    # episode_length_buf == 0). This matches the lazy-reset pattern used by
    # contact_wrench_cumulative_reward in rewards.py.
    just_reset = env.episode_length_buf == 0
    env._hocv_violation_count = torch.where(
        just_reset,
        torch.zeros_like(env._hocv_violation_count),
        env._hocv_violation_count,
    )

    # Increment on violation, zero on success.
    env._hocv_violation_count = torch.where(
        step_violation,
        env._hocv_violation_count + 1,
        torch.zeros_like(env._hocv_violation_count),
    )

    return env._hocv_violation_count >= int(grace_frames)


# Per-finger base Cartesian error thresholds (METERS) — match ManipTrans
# dexhandimitator.py:1145-1156. The effective threshold scales over training
# as ``base / 0.7 * scale(t)`` where ``scale(t)`` decays from 1.3 -> 0.7 via
# an exponential schedule (see ``_scale_factor`` below).
_FINGER_BASE_THRESHOLDS_M: dict[str, float] = {
    "thumb": 0.040,
    "index": 0.045,
    "middle": 0.050,
    "ring": 0.060,
    "pinky": 0.060,
    "level_1": 0.070,  # _PP bodies (proximal phalanx)
    "level_2": 0.080,  # _MP bodies (middle phalanx)
}


def _scale_factor(
    current_step: int, tighten_factor: float, tighten_steps: int
) -> float:
    """Exp-decay scale factor matching ManipTrans's "exp_decay" tighten method.

    ``scale(t) = (2*e)^(-t / tighten_steps) * (1 - tighten_factor) + tighten_factor``

    At t=0: scale = 1 - tighten_factor + tighten_factor = 1.0; multiplied by the
    1/0.7 prefactor in the threshold formula this gives a 1.43x base (loose).
    For the canonical ``tighten_factor=0.7`` the schedule is:
      * t=0           -> scale = 1.3  -> threshold ~ base / 0.7 * 1.3 = base * 1.857
      * t=tighten_steps -> scale ~= 0.81 -> threshold ~ base * 1.16
      * t -> inf       -> scale -> 0.7 -> threshold -> base
    """
    if tighten_steps <= 0:
        return tighten_factor
    decay = (2.0 * math.e) ** (-float(current_step) / float(tighten_steps))
    return decay * (1.0 - tighten_factor) + tighten_factor


def _disabled_finger_trajectory_termination(
    env: ManagerBasedRLEnv,
    command_name: str = "dual_hands_object_tracking_command",
    grace_frames: int = 20,
    tighten_factor: float = 0.7,
    tighten_steps: int = 128000,
) -> torch.Tensor:
    """ManipTrans imitator-style per-finger early termination.

    Terminates the episode as soon as any of the following per-finger mean
    Cartesian errors (robot body position vs. retargeted ref frame, both in
    env frame) exceeds its annealed threshold AND the episode is past the
    initial ``grace_frames``-frame grace period:

      * thumb_tip   : 0.040 m base
      * index_tip   : 0.045 m base
      * middle_tip  : 0.050 m base
      * ring_tip    : 0.060 m base
      * pinky_tip   : 0.060 m base
      * level_1 (PP): 0.070 m base  (mean over all _PP bodies, both hands)
      * level_2 (MP): 0.080 m base  (mean over all _MP bodies)

    Effective threshold = ``base / 0.7 * scale(t)`` where ``scale(t)`` follows
    the exp-decay schedule in :func:`_scale_factor` driven by
    ``env.common_step_counter``. The first ``grace_frames`` simulation steps
    of every episode are exempt to let the residual policy correct any
    initial-pose error before being penalised.

    All seven sub-tests are evaluated per-hand independently (i.e. the thumb
    tip on the right hand is independent of the thumb tip on the left hand),
    and the final per-env termination is the OR over BOTH hands x all 7
    finger groups.

    NOTE: the finger-to-body classification is precomputed in
    :meth:`DualHandsObjectTrackingCommand._init_hand_data` as
    ``paper_{side}_{finger}_tip_idx`` /
    ``paper_{side}_level1_idxs`` / ``paper_{side}_level2_idxs``. Those tensors
    index INTO the already-filtered intersection of retargeted frames x robot
    bodies (the same intersection used by ``hand_skeleton_tracking_exp`` and
    ``paper_Ej_cm``), NOT into the raw articulation body list.

    Args:
        env: The environment instance.
        command_name: Name of the dual-hand-object tracking command term.
        grace_frames: Episode steps before checks become active. Default 20.
            To DISABLE the term, set this very high (e.g. 100_000) — see note
            on ``tighten_factor`` for why that's the right disable knob.
        tighten_factor: Asymptotic scale (paper default 0.7). NOTE: setting
            this very large does NOT disable the term at
            ``common_step_counter=0`` — the scale formula evaluates to 1.0
            there independent of ``tighten_factor``. The asymptote IS huge
            (≈ 57m thumb threshold at tighten_factor=999), but only after
            ``common_step_counter >> tighten_steps``. To truly disable from
            t=0, raise ``grace_frames`` instead.
        tighten_steps: Decay timescale in env steps (paper default 128000).

    Returns:
        Bool tensor of shape ``(num_envs,)``.
    """
    cmd = env.command_manager.get_term(command_name)

    scale = _scale_factor(
        int(env.common_step_counter), float(tighten_factor), int(tighten_steps)
    )

    env_origins_b = env.scene.env_origins.unsqueeze(1)  # (N, 1, 3)
    terminate = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    for side in ("right", "left"):
        robot = getattr(cmd, f"{side}_robot")
        robot_body_ids = getattr(cmd, f"paper_{side}_robot_body_ids")
        ref_frame_idxs = getattr(cmd, f"paper_{side}_ref_frame_indices")

        # Robot body positions in env frame (filtered to intersection).
        robot_e = (
            robot.data.body_link_pos_w[:, robot_body_ids] - env_origins_b
        )  # (N, F, 3)
        ref_e = getattr(cmd, f"retargeted_{side}_hand_frames")[cmd.timestep_counter][
            :, ref_frame_idxs, :3
        ]  # (N, F, 3)

        err = torch.norm(robot_e - ref_e, dim=-1)  # (N, F)

        # Tip checks (one body each).
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            tip_idxs = getattr(cmd, f"paper_{side}_{finger}_tip_idx")
            if tip_idxs.numel() == 0:
                continue
            base = _FINGER_BASE_THRESHOLDS_M[finger]
            thresh = base / 0.7 * scale
            tip_err = err[:, tip_idxs].mean(dim=-1)  # (N,)
            terminate = terminate | (tip_err > thresh)

        # Level-1 (PP) and level-2 (MP) checks (mean over multiple bodies).
        level1_idxs = getattr(cmd, f"paper_{side}_level1_idxs")
        if level1_idxs.numel() > 0:
            thresh1 = _FINGER_BASE_THRESHOLDS_M["level_1"] / 0.7 * scale
            l1_err = err[:, level1_idxs].mean(dim=-1)
            terminate = terminate | (l1_err > thresh1)
        level2_idxs = getattr(cmd, f"paper_{side}_level2_idxs")
        if level2_idxs.numel() > 0:
            thresh2 = _FINGER_BASE_THRESHOLDS_M["level_2"] / 0.7 * scale
            l2_err = err[:, level2_idxs].mean(dim=-1)
            terminate = terminate | (l2_err > thresh2)

    # Enforce grace period: never terminate before ``grace_frames`` steps.
    in_grace = env.episode_length_buf < int(grace_frames)
    terminate = terminate & ~in_grace
    return terminate


def joint_velocity_blow_up(
    env: ManagerBasedRLEnv,
    command_name: str = "dual_hands_object_tracking_command",
    threshold_rad_s: float = 100.0,
) -> torch.Tensor:
    """Terminate when any robot joint velocity exceeds ``threshold_rad_s``.

    Mirrors ManipTrans's ``error_buf`` safety termination
    (``dexhandmanip_bih.py``): any robot joint that exceeds 100 rad/s is a
    physics-instability signal — the policy has lost control and the rollout
    should end. Used only by the paper-faithful ``--maniptrans_eval`` CLI
    path to match the paper's eval protocol. Not registered by default.

    Args:
        env: The RL environment.
        command_name: Name of the dual-hand-object tracking command term.
        threshold_rad_s: Per-joint velocity magnitude over which the episode
            terminates. Default 100 rad/s matches ManipTrans paper §B.
    """
    cmd = env.command_manager.get_term(command_name)
    terminate = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for side in ("right", "left"):
        try:
            robot = getattr(cmd, f"{side}_robot")
        except AttributeError:
            continue
        # ``data.joint_vel`` shape: (num_envs, num_joints) in rad/s.
        joint_vel_abs = robot.data.joint_vel.abs()
        terminate = terminate | (joint_vel_abs.max(dim=-1).values > threshold_rad_s)
    return terminate
