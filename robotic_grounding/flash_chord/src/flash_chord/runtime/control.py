# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wrist tracking control for floating-hand embodiments.

Prismatic + finger joints are position-controlled via ``control.joint_target_pos``; the ball wrist
joint is effort-controlled by :func:`control_wrist_orientation` (proportional torque toward a target
rotation vector; velocity damping is implicit via ``mujoco:dof_passive_damping``).
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from flash_chord.embodiments.base import EmbodimentLayout
from flash_chord.utils.quat import quat_inv_xyzw, quat_mul_xyzw, quat_to_rotvec


@wp.kernel
def control_wrist_orientation(
    joint_q: wp.array[wp.float32],
    targets: wp.array[wp.float32],
    num_joint_q: int,
    num_joint_dof: int,
    num_wrists: int,
    wrist_qcoord_off: wp.array[wp.int32],  # [num_wrists] ball quat coord offset / world
    wrist_dof_off: wp.array[wp.int32],  # [num_wrists] ball dof offset / world
    wrist_kp: wp.array[wp.float32],
    wrist_tau_max: wp.array[wp.float32],
    joint_f: wp.array[wp.float32],
) -> None:
    """Proportional effort toward each wrist's target rotvec; launched over world_count * num_wrists."""
    tid = wp.tid()
    world_id = tid // num_wrists
    h = tid % num_wrists
    qb = world_id * num_joint_q + wrist_qcoord_off[h]
    db = world_id * num_joint_dof + wrist_dof_off[h]

    cur = wp.normalize(wp.quat(joint_q[qb + 0], joint_q[qb + 1], joint_q[qb + 2], joint_q[qb + 3]))
    rotvec = wp.vec3(targets[db + 0], targets[db + 1], targets[db + 2])
    angle = wp.length(rotvec)
    if angle > 1.0e-6:
        tgt = wp.quat_from_axis_angle(rotvec / angle, angle)
    else:
        tgt = wp.quat_identity()
    if (tgt[0] * cur[0] + tgt[1] * cur[1] + tgt[2] * cur[2] + tgt[3] * cur[3]) < 0.0:
        tgt = wp.quat(-tgt[0], -tgt[1], -tgt[2], -tgt[3])

    # child-frame error (cur^-1 * tgt): the ball joint_f is expressed in the child frame
    rot_err = quat_to_rotvec(quat_mul_xyzw(quat_inv_xyzw(cur), tgt))
    kp = wrist_kp[h]
    tau_max = wrist_tau_max[h]
    joint_f[db + 0] = wp.clamp(kp * rot_err[0], -tau_max, tau_max)
    joint_f[db + 1] = wp.clamp(kp * rot_err[1], -tau_max, tau_max)
    joint_f[db + 2] = wp.clamp(kp * rot_err[2], -tau_max, tau_max)


@dataclass
class WristController:
    """Per-wrist ball-effort controller arrays, built from the embodiment layout."""

    num_wrists: int
    qcoord_off: wp.array  # int32 [num_wrists] ball quat coord / world
    dof_off: wp.array  # int32 [num_wrists] ball rotvec dof start / world
    kp: wp.array  # float32 [num_wrists]
    tau_max: wp.array  # float32 [num_wrists]
    num_joint_q: int  # per world
    num_joint_dof: int  # per world
    world_count: int

    @classmethod
    def from_layout(cls, layout: EmbodimentLayout, model, world_count: int, kp: float = 25.0, tau_max: float = 45.0, device=None):
        qcoord = [h.wrist_orient_q_id for h in layout.hands]
        dof = [h.wrist_orient_dof_ids[0] for h in layout.hands]
        n = len(layout.hands)
        return cls(
            num_wrists=n,
            qcoord_off=wp.array(qcoord, dtype=wp.int32, device=device),
            dof_off=wp.array(dof, dtype=wp.int32, device=device),
            kp=wp.array([kp] * n, dtype=wp.float32, device=device),
            tau_max=wp.array([tau_max] * n, dtype=wp.float32, device=device),
            num_joint_q=model.joint_coord_count // world_count,
            num_joint_dof=model.joint_dof_count // world_count,
            world_count=world_count,
        )

    def write_position_targets(self, control, targets: wp.array) -> None:
        """Route ``targets`` into the solver's position-target buffer (prismatic + finger; the ball
        rotvec slot is ignored by the solver since the ball is effort-controlled). Call once per
        control frame. ``targets`` is the per-world DOF target vector (world_count * num_joint_dof)."""
        wp.copy(control.joint_target_pos, targets)

    def apply_wrist_effort(self, state, control, targets: wp.array) -> None:
        """Launch the ball-joint effort PD toward each wrist's target rotvec (read from ``targets``
        at the ball-dof slots). Call each substep — it reads the evolving ``state.joint_q``."""
        wp.launch(
            control_wrist_orientation,
            dim=self.world_count * self.num_wrists,
            inputs=[
                state.joint_q, targets, self.num_joint_q, self.num_joint_dof, self.num_wrists,
                self.qcoord_off, self.dof_off, self.kp, self.tau_max,
            ],
            outputs=[control.joint_f],
        )
