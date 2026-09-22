# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Virtual Object Controller (VOC) targets for object roots and articulations.

Each object root receives a PD wrench plus gravity compensation through ``state.body_f``. Each declared
articulation receives its reference position through Newton's implicit target drive. The drive's gain and bias
are scaled separately by :class:`~flash_chord.runtime.joint_drive.ScaledJointTargetDrive`, so zero scale leaves
the complete object passive. Targets and root control are written every physics substep after ``clear_forces``
and before ``solver.step``.

By default, root gains are *acceleration*-level (1/s², 1/s): the wrench is the desired acceleration
scaled by the body's mass / inertia. ``mass_normalized_gains=False`` instead treats the configured
gains as direct force/torque gains. The latter mode exists for exact parity with upstream controllers
whose gains were tuned in N/m, N·s/m, N·m/rad, and N·m·s/rad.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import warp as wp

from flash_chord.utils.quat import quat_to_rotvec


@dataclass
class VOCParams:
    k_lin: float = 400.0  # linear stiffness [1/s²] (ω_lin = sqrt(k_lin) = 20 rad/s)
    d_lin: float = 40.0  # linear damping [1/s] (critical at ω_lin)
    k_ang: float = 400.0  # angular stiffness [1/s²]
    d_ang: float = 40.0  # angular damping [1/s]
    max_force: float = 200.0  # N
    max_torque: float = 50.0  # N·m
    gravity: float = 9.81  # magnitude; gravity-comp force adds +m*gravity along +z
    mass_normalized_gains: bool = True

    def __post_init__(self) -> None:
        values = {
            "k_lin": self.k_lin,
            "d_lin": self.d_lin,
            "k_ang": self.k_ang,
            "d_ang": self.d_ang,
            "max_force": self.max_force,
            "max_torque": self.max_torque,
            "gravity": self.gravity,
        }
        invalid = {name: value for name, value in values.items() if not math.isfinite(value) or value < 0.0}
        if invalid:
            raise ValueError(f"VOC parameters must be finite and non-negative, got {invalid}")
        if not isinstance(self.mass_normalized_gains, bool):
            raise TypeError(f"mass_normalized_gains must be bool, got {self.mass_normalized_gains!r}")


@wp.kernel
def _object_control(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_mass: wp.array[wp.float32],
    body_com: wp.array[wp.vec3],
    body_inertia: wp.array[wp.mat33],
    object_root_body_ids: wp.array[wp.int32],
    object_body_offsets: wp.array[wp.int32],
    object_body_ids: wp.array[wp.int32],
    target_pos: wp.array[wp.vec3],
    target_quat: wp.array[wp.quat],  # xyzw
    articulation_dof_ids: wp.array[wp.int32],
    articulation_target_pos: wp.array[wp.float32],
    k_lin: float,
    d_lin: float,
    k_ang: float,
    d_ang: float,
    max_force: float,
    max_torque: float,
    scale: wp.array[wp.float32],
    scale_count: int,
    world_count: int,
    object_count: int,
    articulation_count: int,
    articulations_per_world: int,
    bodies_per_world: int,
    num_joint_dof: int,
    gravity: float,
    mass_normalized_gains: bool,
    body_f: wp.array[wp.spatial_vector],
    joint_target_pos: wp.array[wp.float32],
    joint_target_vel: wp.array[wp.float32],
) -> None:
    i = wp.tid()
    if i < object_count:
        objects_per_world = object_count // world_count
        world = i // objects_per_world
        scale_index = 0
        if scale_count > 1:
            scale_index = world
        scale_value = scale[scale_index]
        object_id = i - world * objects_per_world
        bid = object_root_body_ids[i]
        xf = body_q[bid]
        pos = wp.transform_get_translation(xf)
        quat = wp.transform_get_rotation(xf)
        vel = body_qd[bid]
        lin_vel_com = wp.spatial_top(vel)  # body_qd linear component is measured at the COM
        ang_vel = wp.spatial_bottom(vel)
        com_offset_w = wp.quat_rotate(quat, body_com[bid])
        lin_vel = lin_vel_com - wp.cross(ang_vel, com_offset_w)
        # Warp requires an explicit dynamic float before mutation in this runtime-sized loop.
        mass = float(0.0)  # noqa: UP018
        for body_index in range(object_body_offsets[object_id], object_body_offsets[object_id + 1]):
            object_bid = world * bodies_per_world + object_body_ids[body_index]
            mass += body_mass[object_bid]
        inertia = body_inertia[bid]
        inertia_eff = (inertia[0, 0] + inertia[1, 1] + inertia[2, 2]) / 3.0

        force = k_lin * (target_pos[i] - pos) - d_lin * lin_vel
        q_err = wp.mul(target_quat[i], wp.quat_inverse(quat))
        torque = k_ang * quat_to_rotvec(q_err) - d_ang * ang_vel
        if mass_normalized_gains:
            force = mass * force
            torque = inertia_eff * torque
        force = force + wp.vec3(0.0, 0.0, gravity * mass)

        force = wp.vec3(
            wp.clamp(scale_value * force[0], -max_force, max_force),
            wp.clamp(scale_value * force[1], -max_force, max_force),
            wp.clamp(scale_value * force[2], -max_force, max_force),
        )
        torque = wp.vec3(
            wp.clamp(scale_value * torque[0], -max_torque, max_torque),
            wp.clamp(scale_value * torque[1], -max_torque, max_torque),
            wp.clamp(scale_value * torque[2], -max_torque, max_torque),
        )
        body_f[bid] = wp.spatial_vector(force, torque)

    if i < articulation_count:
        world = i // articulations_per_world
        articulation = i - world * articulations_per_world
        dof_id = world * num_joint_dof + articulation_dof_ids[articulation]
        joint_target_pos[dof_id] = articulation_target_pos[i]
        joint_target_vel[dof_id] = 0.0


def apply_object_control(model, state, control, command, scale, params=None):
    """Apply root wrench and articulation targets from persistent buffers in one graph-safe launch."""
    if scale.shape[0] not in (1, command.world_count):
        raise ValueError(f"VOC scale count must be one or the world count {command.world_count}, got {scale.shape[0]}")
    object_count = command.voc_body_ids_w.shape[0]
    expected_object_count = command.world_count * command.layout.num_objects
    if object_count != expected_object_count:
        raise ValueError(f"VOC has {object_count} root bodies; expected {expected_object_count}")
    articulation_count = command.articulation_target_pos.shape[0]
    expected_articulation_count = command.world_count * command.layout.num_articulations
    if articulation_count != expected_articulation_count:
        raise ValueError(f"VOC has {articulation_count} articulation targets; expected {expected_articulation_count}")
    p = params or VOCParams()
    wp.launch(
        _object_control,
        dim=max(object_count, articulation_count),
        inputs=[
            state.body_q,
            state.body_qd,
            model.body_mass,
            model.body_com,
            model.body_inertia,
            command.voc_body_ids_w,
            command.voc_object_body_offsets,
            command.voc_object_body_ids,
            command.voc_target_pos_w,
            command.voc_target_quat_w,
            command.articulation_dof_ids,
            command.articulation_target_pos,
            p.k_lin,
            p.d_lin,
            p.k_ang,
            p.d_ang,
            p.max_force,
            p.max_torque,
            scale,
            scale.shape[0],
            command.world_count,
            object_count,
            articulation_count,
            command.layout.num_articulations,
            command.bodies_per_world,
            command.num_joint_dof,
            p.gravity,
            p.mass_normalized_gains,
        ],
        outputs=[state.body_f, control.joint_target_pos, control.joint_target_vel],
    )
