# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph-safe per-world scaling for Newton's implicit scalar-joint drives.

Newton represents ``POSITION_VELOCITY`` targets as one MuJoCo position actuator and one velocity
actuator per DOF. Scaling both actuators' gain and bias parameters removes both their force and their
implicit velocity derivative. A zero scale therefore leaves the selected joint physically passive,
while a unit scale preserves Newton's native implicit integration.

Newton does not currently expose a public graph-safe gain synchronization API. This module contains
the only dependency on its MuJoCo actuator mapping and validates that layout at setup time so an
upstream representation change fails loudly instead of silently controlling the wrong actuator.
"""

from __future__ import annotations

from dataclasses import dataclass

from mujoco_warp._src.types import vec10f
import numpy as np
import warp as wp

_JOINT_TARGET_SOURCE = 0


@wp.kernel
def _enforce_joint_velocity_limits(
    joint_qd: wp.array(dtype=wp.float32),
    joint_dof_ids: wp.array(dtype=wp.int32),
    velocity_limit: wp.array(dtype=wp.float32),
    dofs_per_world: int,
) -> None:
    world, joint = wp.tid()
    dof_id = world * dofs_per_world + joint_dof_ids[joint]
    velocity = joint_qd[dof_id]
    limit = velocity_limit[joint]
    if wp.abs(velocity) > limit:
        joint_qd[dof_id] = wp.clamp(velocity, -limit, limit)


@dataclass
class JointActuatorLimits:
    """Enforce authored scalar-joint velocity limits."""

    world_count: int
    joint_count: int
    dofs_per_world: int
    joint_dof_ids: wp.array
    velocity_limit: wp.array

    @classmethod
    def build(
        cls,
        model,
        q_ids: tuple[int, ...],
        dof_ids: tuple[int, ...],
        *,
        world_count: int,
        device=None,
    ) -> "JointActuatorLimits":
        if len(q_ids) != len(dof_ids) or not q_ids:
            raise ValueError("actuator limits require matching nonempty q and DOF selections")
        dofs_per_world = model.joint_dof_count // world_count
        velocity = np.asarray(model.joint_velocity_limit.numpy(), dtype=np.float32).reshape(
            world_count, dofs_per_world
        )[:, dof_ids]
        if not np.array_equal(velocity, np.broadcast_to(velocity[0], velocity.shape)):
            raise ValueError("actuator limits require identical joint velocity limits across worlds")
        if not np.all(np.isfinite(velocity)) or np.any(velocity <= 0.0):
            raise ValueError("selected joint velocity limits must be finite and positive")

        resolved_device = device or model.device
        i32 = lambda values: wp.array(values, dtype=wp.int32, device=resolved_device)  # noqa: E731
        f32 = lambda values: wp.array(values, dtype=wp.float32, device=resolved_device)  # noqa: E731
        return cls(
            world_count=world_count,
            joint_count=len(q_ids),
            dofs_per_world=dofs_per_world,
            joint_dof_ids=i32(dof_ids),
            velocity_limit=f32(velocity[0]),
        )

    def enforce_velocity(self, state) -> None:
        wp.launch(
            _enforce_joint_velocity_limits,
            dim=(self.world_count, self.joint_count),
            inputs=[
                state.joint_qd,
                self.joint_dof_ids,
                self.velocity_limit,
                self.dofs_per_world,
            ],
        )


@dataclass(frozen=True)
class JointPositionIntegralConfig:
    """Slow outer position integral expressed as a velocity target."""

    gain_s_inv2: float = 0.0
    max_effort_fraction: float = 0.15

    def __post_init__(self) -> None:
        if not np.isfinite(self.gain_s_inv2) or self.gain_s_inv2 < 0.0:
            raise ValueError(f"position-integral gain must be finite and non-negative, got {self.gain_s_inv2}")
        if not np.isfinite(self.max_effort_fraction) or not 0.0 < self.max_effort_fraction <= 1.0:
            raise ValueError(
                f"position-integral effort fraction must be finite and in (0, 1], got {self.max_effort_fraction}"
            )


@wp.kernel
def _apply_joint_position_integral(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    joint_target_pos: wp.array(dtype=wp.float32),
    joint_q_ids: wp.array(dtype=wp.int32),
    joint_dof_ids: wp.array(dtype=wp.int32),
    kp: wp.array(dtype=wp.float32),
    kd: wp.array(dtype=wp.float32),
    effort_limit: wp.array(dtype=wp.float32),
    integral_velocity: wp.array(dtype=wp.float32),
    coords_per_world: int,
    dofs_per_world: int,
    dt: float,
    gain_s_inv2: float,
    max_effort_fraction: float,
    joint_target_vel: wp.array(dtype=wp.float32),
) -> None:
    world, joint = wp.tid()
    state_id = world * joint_dof_ids.shape[0] + joint
    q_id = world * coords_per_world + joint_q_ids[joint]
    dof_id = world * dofs_per_world + joint_dof_ids[joint]
    error = joint_target_pos[dof_id] - joint_q[q_id]

    nominal_effort = wp.clamp(
        kp[joint] * error - kd[joint] * joint_qd[dof_id],
        -effort_limit[joint],
        effort_limit[joint],
    )
    velocity = integral_velocity[state_id] + dt * gain_s_inv2 * error
    integral_effort = wp.clamp(
        kd[joint] * velocity,
        -max_effort_fraction * effort_limit[joint],
        max_effort_fraction * effort_limit[joint],
    )
    integral_effort = wp.clamp(
        integral_effort,
        -effort_limit[joint] - nominal_effort,
        effort_limit[joint] - nominal_effort,
    )
    integral_velocity[state_id] = integral_effort / kd[joint]
    joint_target_vel[dof_id] = integral_velocity[state_id]


@wp.kernel
def _reset_joint_position_integral(
    reset_mask: wp.array(dtype=wp.int32),
    joint_count: int,
    integral_velocity: wp.array(dtype=wp.float32),
) -> None:
    world, joint = wp.tid()
    if reset_mask[world] != 0:
        integral_velocity[world * joint_count + joint] = 0.0


@dataclass
class JointPositionIntegral:
    """Graph-safe outer position integral for selected implicit joint drives."""

    world_count: int
    joint_count: int
    coords_per_world: int
    dofs_per_world: int
    joint_q_ids: wp.array
    joint_dof_ids: wp.array
    kp: wp.array
    kd: wp.array
    effort_limit: wp.array
    integral_velocity: wp.array
    dt: float
    gain_s_inv2: float
    max_effort_fraction: float

    @classmethod
    def build(
        cls,
        model,
        q_ids: tuple[int, ...],
        dof_ids: tuple[int, ...],
        *,
        world_count: int,
        dt: float,
        config: JointPositionIntegralConfig,
        device=None,
    ) -> "JointPositionIntegral":
        if len(q_ids) != len(dof_ids) or not q_ids:
            raise ValueError("position integral requires matching nonempty q and DOF selections")
        coords_per_world = model.joint_coord_count // world_count
        dofs_per_world = model.joint_dof_count // world_count
        kp = np.asarray(model.joint_target_ke.numpy(), dtype=np.float32).reshape(world_count, dofs_per_world)
        kd = np.asarray(model.joint_target_kd.numpy(), dtype=np.float32).reshape(world_count, dofs_per_world)
        effort = np.asarray(model.joint_effort_limit.numpy(), dtype=np.float32).reshape(world_count, dofs_per_world)
        selected_kp = kp[:, dof_ids]
        selected_kd = kd[:, dof_ids]
        selected_effort = effort[:, dof_ids]
        if not np.all(selected_kp == selected_kp[0]) or not np.all(selected_kd == selected_kd[0]):
            raise ValueError("position integral requires identical selected feedback gains across worlds")
        if not np.all(selected_effort == selected_effort[0]):
            raise ValueError("position integral requires identical selected effort limits across worlds")
        if np.any(selected_kd <= 0.0) or np.any(selected_effort <= 0.0):
            raise ValueError("position integral requires positive damping gains and effort limits")
        resolved_device = device or model.device
        joint_count = len(dof_ids)
        return cls(
            world_count=world_count,
            joint_count=joint_count,
            coords_per_world=coords_per_world,
            dofs_per_world=dofs_per_world,
            joint_q_ids=wp.array(q_ids, dtype=wp.int32, device=resolved_device),
            joint_dof_ids=wp.array(dof_ids, dtype=wp.int32, device=resolved_device),
            kp=wp.array(selected_kp[0], dtype=wp.float32, device=resolved_device),
            kd=wp.array(selected_kd[0], dtype=wp.float32, device=resolved_device),
            effort_limit=wp.array(selected_effort[0], dtype=wp.float32, device=resolved_device),
            integral_velocity=wp.zeros(world_count * joint_count, dtype=wp.float32, device=resolved_device),
            dt=float(dt),
            gain_s_inv2=config.gain_s_inv2,
            max_effort_fraction=config.max_effort_fraction,
        )

    def apply(self, state, control) -> None:
        wp.launch(
            _apply_joint_position_integral,
            dim=(self.world_count, self.joint_count),
            inputs=[
                state.joint_q,
                state.joint_qd,
                control.joint_target_pos,
                self.joint_q_ids,
                self.joint_dof_ids,
                self.kp,
                self.kd,
                self.effort_limit,
                self.integral_velocity,
                self.coords_per_world,
                self.dofs_per_world,
                self.dt,
                self.gain_s_inv2,
                self.max_effort_fraction,
            ],
            outputs=[control.joint_target_vel],
        )

    def reset(self, reset_mask: wp.array) -> None:
        wp.launch(
            _reset_joint_position_integral,
            dim=(self.world_count, self.joint_count),
            inputs=[reset_mask, self.joint_count],
            outputs=[self.integral_velocity],
        )


@wp.kernel
def _scale_joint_target_drive(
    scale: wp.array[wp.float32],
    position_actuator_ids: wp.array[wp.int32],
    velocity_actuator_ids: wp.array[wp.int32],
    position_gain: wp.array2d[vec10f],
    position_bias: wp.array2d[vec10f],
    velocity_gain: wp.array2d[vec10f],
    velocity_bias: wp.array2d[vec10f],
    actuator_gain: wp.array2d[vec10f],
    actuator_bias: wp.array2d[vec10f],
) -> None:
    world, joint = wp.tid()
    scale_index = 0
    if scale.shape[0] > 1:
        scale_index = world
    value = scale[scale_index]
    position_actuator = position_actuator_ids[joint]
    velocity_actuator = velocity_actuator_ids[joint]
    actuator_gain[world, position_actuator] = value * position_gain[world, joint]
    actuator_bias[world, position_actuator] = value * position_bias[world, joint]
    actuator_gain[world, velocity_actuator] = value * velocity_gain[world, joint]
    actuator_bias[world, velocity_actuator] = value * velocity_bias[world, joint]


def _single_actuator_id(source: np.ndarray, mapping: np.ndarray, encoded_dof: int, label: str) -> int:
    matches = np.flatnonzero((source == _JOINT_TARGET_SOURCE) & (mapping == encoded_dof))
    if matches.size != 1:
        raise ValueError(
            f"selected {label} target {encoded_dof} must map to exactly one MuJoCo actuator, got {matches.tolist()}"
        )
    return int(matches[0])


def _validate_affine_layout(
    gain: np.ndarray,
    bias: np.ndarray,
    *,
    bias_index: int,
    label: str,
) -> None:
    if not np.all(np.isfinite(gain)) or not np.all(np.isfinite(bias)):
        raise ValueError(f"{label} actuator parameters must be finite")
    expected_gain = np.zeros_like(gain)
    expected_gain[..., 0] = gain[..., 0]
    expected_bias = np.zeros_like(bias)
    expected_bias[..., bias_index] = -gain[..., 0]
    if not np.array_equal(gain, expected_gain) or not np.array_equal(bias, expected_bias):
        raise ValueError(
            f"{label} actuator does not use Newton's expected affine target-drive layout; "
            "the MuJoCo adapter must be updated before this solver version is supported"
        )


@dataclass
class ScaledJointTargetDrive:
    """Scale selected implicit position-velocity drives by one global or per-world value."""

    world_count: int
    joint_count: int
    position_actuator_ids: wp.array
    velocity_actuator_ids: wp.array
    position_gain: wp.array
    position_bias: wp.array
    velocity_gain: wp.array
    velocity_bias: wp.array
    actuator_gain: wp.array
    actuator_bias: wp.array

    @classmethod
    def build(
        cls,
        solver,
        dof_ids: tuple[int, ...],
        *,
        world_count: int,
        device=None,
    ) -> "ScaledJointTargetDrive":
        """Resolve selected local DOFs to Newton's paired MuJoCo target actuators once."""
        dof_ids = tuple(int(dof_id) for dof_id in dof_ids)
        if not dof_ids or len(set(dof_ids)) != len(dof_ids) or min(dof_ids) < 0:
            raise ValueError(f"scaled target drives require unique nonnegative DOF IDs, got {dof_ids}")
        if world_count <= 0:
            raise ValueError(f"world_count must be positive, got {world_count}")

        source_array = getattr(solver, "mjc_actuator_ctrl_source", None)
        mapping_array = getattr(solver, "mjc_actuator_to_newton_idx", None)
        mjw_model = getattr(solver, "mjw_model", None)
        if source_array is None or mapping_array is None or mjw_model is None:
            raise TypeError("scaled target drives require Newton SolverMuJoCo actuator mappings")
        source = np.asarray(source_array.numpy(), dtype=np.int32)
        mapping = np.asarray(mapping_array.numpy(), dtype=np.int32)
        if source.ndim != 1 or mapping.shape != source.shape:
            raise ValueError(f"invalid MuJoCo actuator mapping shapes: source={source.shape}, mapping={mapping.shape}")

        position_ids = tuple(_single_actuator_id(source, mapping, dof_id, "position") for dof_id in dof_ids)
        velocity_ids = tuple(_single_actuator_id(source, mapping, -(dof_id + 2), "velocity") for dof_id in dof_ids)
        if set(position_ids).intersection(velocity_ids):
            raise ValueError("position and velocity target actuators must be distinct")

        actuator_gain = mjw_model.actuator_gainprm
        actuator_bias = mjw_model.actuator_biasprm
        if actuator_gain.dtype != vec10f or actuator_bias.dtype != vec10f:
            raise TypeError(
                f"MuJoCo actuator parameters must use vec10f, got "
                f"gain={actuator_gain.dtype}, bias={actuator_bias.dtype}"
            )
        expected_shape = (world_count, source.shape[0])
        if actuator_gain.shape != expected_shape or actuator_bias.shape != expected_shape:
            raise ValueError(
                f"MuJoCo actuator parameters must have shape {expected_shape}, got "
                f"gain={actuator_gain.shape}, bias={actuator_bias.shape}"
            )
        gain = np.asarray(actuator_gain.numpy(), dtype=np.float32)
        bias = np.asarray(actuator_bias.numpy(), dtype=np.float32)
        position_gain = np.ascontiguousarray(gain[:, position_ids, :])
        position_bias = np.ascontiguousarray(bias[:, position_ids, :])
        velocity_gain = np.ascontiguousarray(gain[:, velocity_ids, :])
        velocity_bias = np.ascontiguousarray(bias[:, velocity_ids, :])
        _validate_affine_layout(position_gain, position_bias, bias_index=1, label="position")
        _validate_affine_layout(velocity_gain, velocity_bias, bias_index=2, label="velocity")

        resolved_device = device or actuator_gain.device
        return cls(
            world_count=world_count,
            joint_count=len(dof_ids),
            position_actuator_ids=wp.array(position_ids, dtype=wp.int32, device=resolved_device),
            velocity_actuator_ids=wp.array(velocity_ids, dtype=wp.int32, device=resolved_device),
            position_gain=wp.array(position_gain, dtype=vec10f, device=resolved_device),
            position_bias=wp.array(position_bias, dtype=vec10f, device=resolved_device),
            velocity_gain=wp.array(velocity_gain, dtype=vec10f, device=resolved_device),
            velocity_bias=wp.array(velocity_bias, dtype=vec10f, device=resolved_device),
            actuator_gain=actuator_gain,
            actuator_bias=actuator_bias,
        )

    def apply(self, scale: wp.array) -> None:
        """Apply a persistent global ``[1]`` or per-world ``[world_count]`` scale."""
        if scale.shape[0] not in (1, self.world_count):
            raise ValueError(
                f"joint target scale count must be one or world count {self.world_count}, got {scale.shape[0]}"
            )
        wp.launch(
            _scale_joint_target_drive,
            dim=(self.world_count, self.joint_count),
            inputs=[
                scale,
                self.position_actuator_ids,
                self.velocity_actuator_ids,
                self.position_gain,
                self.position_bias,
                self.velocity_gain,
                self.velocity_bias,
            ],
            outputs=[self.actuator_gain, self.actuator_bias],
        )
