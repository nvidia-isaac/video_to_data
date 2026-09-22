# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device-native policy observations for reference-tracking environments."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import warp as wp

from flash_chord.embodiments.base import ScalarJointSelection
from flash_chord.embodiments.binding import DeviceRobotReference
from flash_chord.embodiments.frames import DeviceBodyFrameMap
from flash_chord.runtime.actions import PolicyAction
from flash_chord.runtime.command import CommandBuffers
from flash_chord.runtime.contact import ContactTracker
from flash_chord.scene.builder import Scene
from flash_chord.utils.quat import quat_inv_xyzw, quat_mul_xyzw

OBSERVATION_TERM_NAMES = (
    "wrist_position",
    "wrist_orientation",
    "wrist_velocity",
    "finger_joint_position",
    "finger_joint_velocity",
    "object_position",
    "object_orientation",
    "object_body_velocity",
    "command_wrist",
    "command_finger",
    "command_object_position",
    "command_object_orientation",
    "raw_action",
    "processed_action",
    "contact_position",
    "contact_direction",
    "arm_joint_position",
    "arm_joint_velocity",
    "arm_joint_reference_delta",
)

(
    _WRIST_POSITION,
    _WRIST_ORIENTATION,
    _WRIST_VELOCITY,
    _FINGER_JOINT_POSITION,
    _FINGER_JOINT_VELOCITY,
    _OBJECT_POSITION,
    _OBJECT_ORIENTATION,
    _OBJECT_BODY_VELOCITY,
    _COMMAND_WRIST,
    _COMMAND_FINGER,
    _COMMAND_OBJECT_POSITION,
    _COMMAND_OBJECT_ORIENTATION,
    _RAW_ACTION,
    _PROCESSED_ACTION,
    _CONTACT_POSITION,
    _CONTACT_DIRECTION,
    _ARM_JOINT_POSITION,
    _ARM_JOINT_VELOCITY,
    _ARM_JOINT_REFERENCE_DELTA,
) = range(len(OBSERVATION_TERM_NAMES))


@dataclass(frozen=True)
class ObservationTermConfig:
    """Inclusion and fixed scaling for one logical observation term."""

    enabled: bool = True
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.scale):
            raise ValueError(f"observation term scale must be finite, got {self.scale}")


@runtime_checkable
class Observation(Protocol):
    """Device observation strategy consumed by an RL or sampling environment."""

    observation_dim: int
    observation: wp.array

    def compute(self, state) -> wp.array:
        """Update and return the persistent observation buffer."""
        ...


@runtime_checkable
class ObservationInto(Protocol):
    """Optional capability for writing an observation into a caller-owned buffer."""

    def compute_into(self, state, output: wp.array) -> wp.array:
        """Update and return ``output`` without modifying the primary observation buffer."""
        ...


@runtime_checkable
class ObservationSpec(Protocol):
    """Setup-time observation builder selected by configuration."""

    def build(
        self,
        scene: Scene,
        action: PolicyAction,
        command: CommandBuffers,
        contact: ContactTracker | None,
        timestep: wp.array,
        episode_step: wp.array,
        reference_joint_q_offset: wp.array,
        device=None,
    ) -> Observation:
        """Bind runtime dependencies and return an observation strategy."""
        ...


@runtime_checkable
class ObservationDiagnostics(Protocol):
    """Optional named flattened ranges for observation health metrics."""

    block_names: tuple[str, ...]
    block_ranges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class PolicyObservationConfig:
    """Ordering and representation choices for the policy observation."""

    sides: tuple[str, ...] = ("right", "left")
    contact_sides: tuple[str, ...] = ("left", "right")
    make_wrist_quat_unique: bool = True
    wrist_position: ObservationTermConfig = field(default_factory=ObservationTermConfig)
    wrist_orientation: ObservationTermConfig = field(default_factory=ObservationTermConfig)
    wrist_velocity: ObservationTermConfig = field(default_factory=ObservationTermConfig)
    arm_joint_position: ObservationTermConfig = field(default_factory=lambda: ObservationTermConfig(enabled=False))
    arm_joint_velocity: ObservationTermConfig = field(default_factory=lambda: ObservationTermConfig(enabled=False))
    arm_joint_reference_delta: ObservationTermConfig = field(
        default_factory=lambda: ObservationTermConfig(enabled=False)
    )
    finger_joint_position: ObservationTermConfig = field(default_factory=ObservationTermConfig)
    finger_joint_velocity: ObservationTermConfig = field(default_factory=ObservationTermConfig)
    object_position: ObservationTermConfig = field(default_factory=ObservationTermConfig)
    object_orientation: ObservationTermConfig = field(default_factory=ObservationTermConfig)
    command_wrist: ObservationTermConfig = field(default_factory=ObservationTermConfig)
    command_finger: ObservationTermConfig = field(default_factory=ObservationTermConfig)
    command_object_position: ObservationTermConfig = field(default_factory=ObservationTermConfig)
    command_object_orientation: ObservationTermConfig = field(default_factory=ObservationTermConfig)
    raw_action: ObservationTermConfig = field(default_factory=ObservationTermConfig)
    processed_action: ObservationTermConfig = field(default_factory=ObservationTermConfig)
    contact_position: ObservationTermConfig = field(default_factory=ObservationTermConfig)
    contact_direction: ObservationTermConfig = field(default_factory=ObservationTermConfig)
    object_body_velocity: ObservationTermConfig = field(default_factory=lambda: ObservationTermConfig(enabled=False))

    def __post_init__(self) -> None:
        object.__setattr__(self, "sides", tuple(self.sides))
        object.__setattr__(self, "contact_sides", tuple(self.contact_sides))
        if len(set(self.sides)) != len(self.sides):
            raise ValueError(f"policy sides must be unique, got {self.sides}")
        if len(set(self.contact_sides)) != len(self.contact_sides):
            raise ValueError(f"contact sides must be unique, got {self.contact_sides}")
        if not set(self.contact_sides).issubset(self.sides):
            raise ValueError(f"contact sides {self.contact_sides} must be a subset of policy sides {self.sides}")

    @property
    def terms(self) -> tuple[ObservationTermConfig, ...]:
        """Logical term configurations in the stable policy-schema order."""
        return tuple(getattr(self, name) for name in OBSERVATION_TERM_NAMES)

    def build(
        self,
        scene: Scene,
        action: PolicyAction,
        command: CommandBuffers,
        contact: ContactTracker | None,
        timestep: wp.array,
        episode_step: wp.array,
        reference_joint_q_offset: wp.array,
        device=None,
    ) -> Observation:
        """Build the fused policy observation."""
        return PolicyObservation.build(
            scene,
            action,
            command,
            contact,
            timestep,
            episode_step,
            config=self,
            device=device,
        )


@dataclass(frozen=True)
class PolicyObservationLayout:
    """Exact flattened policy layout and source-side dimensions."""

    sides: tuple[str, ...]
    contact_sides: tuple[str, ...]
    num_hands: int
    arm_joints: ScalarJointSelection
    finger_joints: ScalarJointSelection
    num_objects: int
    action_dim: int
    processed_dim: int
    wrist_pos_start: int
    wrist_quat_start: int
    wrist_velocity_start: int
    arm_position_start: int
    arm_velocity_start: int
    arm_reference_delta_start: int
    finger_pos_start: int
    finger_velocity_start: int
    object_pos_start: int
    object_quat_start: int
    object_velocity_start: int
    command_wrist_start: int
    command_finger_start: int
    command_object_pos_start: int
    command_object_quat_start: int
    raw_action_start: int
    processed_action_start: int
    contact_position_starts: tuple[int, ...]
    contact_direction_starts: tuple[int, ...]
    contact_slot_counts: tuple[int, ...]
    block_names: tuple[str, ...]
    block_ranges: tuple[tuple[int, int], ...]
    observation_dim: int

    @property
    def total_arms(self) -> int:
        return len(self.arm_joints.q_ids)

    @property
    def total_fingers(self) -> int:
        return len(self.finger_joints.q_ids)

    @classmethod
    def build(
        cls,
        scene: Scene,
        action: PolicyAction,
        command: CommandBuffers,
        contact: ContactTracker | None,
        config: PolicyObservationConfig,
    ) -> "PolicyObservationLayout":
        if set(config.sides) != set(scene.layout.sides) or len(config.sides) != len(scene.layout.sides):
            raise ValueError(f"policy sides {config.sides} must match embodiment sides {scene.layout.sides}")
        if action.sides != config.sides:
            raise ValueError(f"action sides {action.sides} must match observation sides {config.sides}")

        arm_joints = scene.layout.select_scalar_joints(config.sides, ("arm",), require_names=True)
        finger_joints = scene.layout.select_scalar_joints(config.sides, ("finger",))
        total_arms = len(arm_joints.q_ids)
        total_fingers = len(finger_joints.q_ids)
        arm_enabled = (
            config.arm_joint_position.enabled
            or config.arm_joint_velocity.enabled
            or config.arm_joint_reference_delta.enabled
        )
        if arm_enabled and total_arms == 0:
            raise ValueError("arm joint observations require an embodiment with articulated arm joints")
        num_objects = command.layout.num_bodies
        contact_enabled = config.contact_position.enabled or config.contact_direction.enabled
        if contact is None:
            if config.contact_sides and contact_enabled:
                raise ValueError("contact observations require a ContactTracker or contact_sides=()")
        else:
            if set(config.contact_sides) - set(contact.layout.sides):
                raise ValueError(
                    f"contact sides {config.contact_sides} are not available from tracker sides {contact.layout.sides}"
                )
            if contact.layout.num_objects != num_objects:
                raise ValueError(f"contact tracker has {contact.layout.num_objects} objects; command has {num_objects}")

        cursor = 0
        blocks: list[tuple[str, int, int]] = []

        def add_block(term: ObservationTermConfig, name: str, width: int) -> int:
            nonlocal cursor
            if not term.enabled:
                return -1
            start = cursor
            cursor += width
            blocks.append((name, start, cursor))
            return start

        wrist_pos_start = add_block(config.wrist_position, "wrist_position_e_m", 3 * len(config.sides))
        wrist_quat_start = add_block(config.wrist_orientation, "wrist_orientation_e_wxyz", 4 * len(config.sides))
        wrist_velocity_start = add_block(
            config.wrist_velocity,
            "wrist_velocity_b_mps_radps",
            6 * len(config.sides),
        )
        arm_order = ",".join(arm_joints.names)
        arm_position_start = add_block(
            config.arm_joint_position,
            f"arm_joint_position_rad[{arm_order}]",
            total_arms,
        )
        arm_velocity_start = add_block(
            config.arm_joint_velocity,
            f"arm_joint_velocity_radps[{arm_order}]",
            total_arms,
        )
        arm_reference_delta_start = add_block(
            config.arm_joint_reference_delta,
            f"arm_joint_reference_delta_rad[{arm_order}]",
            total_arms,
        )
        finger_pos_start = add_block(
            config.finger_joint_position,
            "finger_joint_position_normalized",
            total_fingers,
        )
        finger_velocity_start = add_block(
            config.finger_joint_velocity,
            "finger_joint_velocity_radps",
            total_fingers,
        )
        object_pos_start = add_block(config.object_position, "object_position_e_m", 3 * num_objects)
        object_quat_start = add_block(config.object_orientation, "object_orientation_e_wxyz", 4 * num_objects)
        object_velocity_start = add_block(
            config.object_body_velocity,
            "object_body_velocity_w_mps_radps",
            6 * num_objects,
        )
        command_wrist_start = add_block(
            config.command_wrist,
            "command_wrist_relative_pose",
            7 * len(config.sides),
        )
        command_finger_start = add_block(
            config.command_finger,
            "command_finger_delta_rad",
            total_fingers,
        )
        command_object_pos_start = add_block(
            config.command_object_position,
            "command_object_relative_position_m",
            3 * num_objects,
        )
        command_object_quat_start = add_block(
            config.command_object_orientation,
            "command_object_relative_orientation_wxyz",
            4 * num_objects,
        )
        raw_action_start = add_block(config.raw_action, "action_raw", action.action_dim)
        processed_action_start = add_block(config.processed_action, "action_processed", action.processed_dim)

        contact_position_starts: list[int] = []
        contact_direction_starts: list[int] = []
        contact_slot_counts: list[int] = []
        if contact is not None and contact_enabled:
            for side in config.contact_sides:
                hand = contact.layout.sides.index(side)
                slot_count = contact.layout.num_objects * contact.layout.link_counts[hand]
                contact_position_starts.append(
                    add_block(config.contact_position, f"contact_{side}_position_wrist_m", 3 * slot_count)
                )
                contact_direction_starts.append(
                    add_block(
                        config.contact_direction,
                        f"contact_{side}_force_direction_wrist_unit",
                        3 * slot_count,
                    )
                )
                contact_slot_counts.append(slot_count)

        return cls(
            sides=config.sides,
            contact_sides=config.contact_sides,
            num_hands=len(config.sides),
            arm_joints=arm_joints,
            finger_joints=finger_joints,
            num_objects=num_objects,
            action_dim=action.action_dim,
            processed_dim=action.processed_dim,
            wrist_pos_start=wrist_pos_start,
            wrist_quat_start=wrist_quat_start,
            wrist_velocity_start=wrist_velocity_start,
            arm_position_start=arm_position_start,
            arm_velocity_start=arm_velocity_start,
            arm_reference_delta_start=arm_reference_delta_start,
            finger_pos_start=finger_pos_start,
            finger_velocity_start=finger_velocity_start,
            object_pos_start=object_pos_start,
            object_quat_start=object_quat_start,
            object_velocity_start=object_velocity_start,
            command_wrist_start=command_wrist_start,
            command_finger_start=command_finger_start,
            command_object_pos_start=command_object_pos_start,
            command_object_quat_start=command_object_quat_start,
            raw_action_start=raw_action_start,
            processed_action_start=processed_action_start,
            contact_position_starts=tuple(contact_position_starts),
            contact_direction_starts=tuple(contact_direction_starts),
            contact_slot_counts=tuple(contact_slot_counts),
            block_names=tuple(name for name, _, _ in blocks),
            block_ranges=tuple((start, end) for _, start, end in blocks),
            observation_dim=cursor,
        )


@wp.func
def unique_quat(q: wp.quat) -> wp.quat:
    """Return the quaternion hemisphere with a non-negative scalar component."""
    if q[3] < 0.0:
        return wp.quat(-q[0], -q[1], -q[2], -q[3])
    return q


@wp.func
def write_wxyz(output: wp.array(dtype=wp.float32), start: int, q: wp.quat, scale: float) -> None:
    """Write one Warp xyzw quaternion in policy wxyz order."""
    output[start] = scale * q[3]
    output[start + 1] = scale * q[0]
    output[start + 2] = scale * q[1]
    output[start + 3] = scale * q[2]


@wp.kernel
def compute_policy_observation(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    timestep: wp.array(dtype=wp.int32),
    episode_step: wp.array(dtype=wp.int32),
    joint_limit_lower: wp.array(dtype=wp.float32),
    joint_limit_upper: wp.array(dtype=wp.float32),
    reference_num_frames: int,
    bodies_per_world: int,
    num_joint_q: int,
    num_joint_dof: int,
    num_hands: int,
    total_arms: int,
    total_fingers: int,
    num_objects: int,
    action_dim: int,
    processed_dim: int,
    observation_dim: int,
    wrist_frame_body_ids: wp.array(dtype=wp.int32),
    wrist_body_to_frame_pos: wp.array(dtype=wp.vec3),
    wrist_body_to_frame_quat: wp.array(dtype=wp.quat),
    arm_q_ids: wp.array(dtype=wp.int32),
    arm_dof_ids: wp.array(dtype=wp.int32),
    finger_q_ids: wp.array(dtype=wp.int32),
    finger_dof_ids: wp.array(dtype=wp.int32),
    object_body_ids: wp.array(dtype=wp.int32),
    reference_wrist_pos_w: wp.array(dtype=wp.vec3),
    reference_wrist_quat_w: wp.array(dtype=wp.quat),
    reference_arm_joint_pos: wp.array(dtype=wp.float32),
    reference_finger_joint_pos: wp.array(dtype=wp.float32),
    reference_object_pos_w: wp.array(dtype=wp.vec3),
    reference_object_quat_w: wp.array(dtype=wp.quat),
    raw_action: wp.array(dtype=wp.float32),
    processed_action: wp.array(dtype=wp.float32),
    term_scales: wp.array(dtype=wp.float32),
    contact_pos_b: wp.array(dtype=wp.vec3),
    contact_direction_b: wp.array(dtype=wp.vec3),
    contact_side_count: int,
    contact_slots_per_world: int,
    contact_source_starts: wp.array(dtype=wp.int32),
    contact_slot_counts: wp.array(dtype=wp.int32),
    contact_position_starts: wp.array(dtype=wp.int32),
    contact_direction_starts: wp.array(dtype=wp.int32),
    make_wrist_quat_unique: bool,
    wrist_pos_start: int,
    wrist_quat_start: int,
    wrist_velocity_start: int,
    arm_position_start: int,
    arm_velocity_start: int,
    arm_reference_delta_start: int,
    finger_pos_start: int,
    finger_velocity_start: int,
    object_pos_start: int,
    object_quat_start: int,
    object_velocity_start: int,
    command_wrist_start: int,
    command_finger_start: int,
    command_object_pos_start: int,
    command_object_quat_start: int,
    raw_action_start: int,
    processed_action_start: int,
    observation: wp.array(dtype=wp.float32),
) -> None:
    """Assemble one world's flattened policy observation."""
    world = wp.tid()
    output = world * observation_dim
    body_base = world * bodies_per_world
    q_base = world * num_joint_q
    dof_base = world * num_joint_dof
    frame = timestep[world]
    if frame < 0:
        frame = 0
    if frame >= reference_num_frames:
        frame = reference_num_frames - 1

    wrist_position_scale = term_scales[_WRIST_POSITION]
    wrist_orientation_scale = term_scales[_WRIST_ORIENTATION]
    wrist_velocity_scale = term_scales[_WRIST_VELOCITY]
    arm_position_scale = term_scales[_ARM_JOINT_POSITION]
    arm_velocity_scale = term_scales[_ARM_JOINT_VELOCITY]
    arm_reference_delta_scale = term_scales[_ARM_JOINT_REFERENCE_DELTA]
    finger_position_scale = term_scales[_FINGER_JOINT_POSITION]
    finger_velocity_scale = term_scales[_FINGER_JOINT_VELOCITY]
    object_position_scale = term_scales[_OBJECT_POSITION]
    object_orientation_scale = term_scales[_OBJECT_ORIENTATION]
    object_body_velocity_scale = term_scales[_OBJECT_BODY_VELOCITY]
    command_wrist_scale = term_scales[_COMMAND_WRIST]
    command_finger_scale = term_scales[_COMMAND_FINGER]
    command_object_position_scale = term_scales[_COMMAND_OBJECT_POSITION]
    command_object_orientation_scale = term_scales[_COMMAND_OBJECT_ORIENTATION]
    raw_action_scale = term_scales[_RAW_ACTION]
    processed_action_scale = term_scales[_PROCESSED_ACTION]
    contact_position_scale = term_scales[_CONTACT_POSITION]
    contact_direction_scale = term_scales[_CONTACT_DIRECTION]

    for hand in range(num_hands):
        wrist_body_xf = body_q[body_base + wrist_frame_body_ids[hand]]
        wrist_body_pos_w = wp.transform_get_translation(wrist_body_xf)
        wrist_body_quat_w = wp.transform_get_rotation(wrist_body_xf)
        wrist_pos_w = wp.transform_point(wrist_body_xf, wrist_body_to_frame_pos[hand])
        wrist_quat_w = wp.normalize(quat_mul_xyzw(wrist_body_quat_w, wrist_body_to_frame_quat[hand]))
        wrist_xf = wp.transform(wrist_pos_w, wrist_quat_w)
        wrist_quat_observation = wrist_quat_w
        if make_wrist_quat_unique:
            wrist_quat_observation = unique_quat(wrist_quat_w)

        if wrist_pos_start >= 0:
            position_output = output + wrist_pos_start + hand * 3
            for axis in range(3):
                observation[position_output + axis] = wrist_position_scale * wrist_pos_w[axis]
        if wrist_quat_start >= 0:
            write_wxyz(
                observation,
                output + wrist_quat_start + hand * 4,
                wrist_quat_observation,
                wrist_orientation_scale,
            )

        wrist_body_velocity_w = body_qd[body_base + wrist_frame_body_ids[hand]]
        angular_velocity_w = wp.spatial_bottom(wrist_body_velocity_w)
        frame_offset_w = wrist_pos_w - wrist_body_pos_w
        linear_velocity_w = wp.spatial_top(wrist_body_velocity_w) + wp.cross(angular_velocity_w, frame_offset_w)
        wrist_inv = wp.transform_inverse(wrist_xf)
        linear_velocity_b = wp.transform_vector(wrist_inv, linear_velocity_w)
        angular_velocity_b = wp.transform_vector(wrist_inv, angular_velocity_w)
        if wrist_velocity_start >= 0:
            velocity_output = output + wrist_velocity_start + hand * 6
            for axis in range(3):
                observation[velocity_output + axis] = wrist_velocity_scale * linear_velocity_b[axis]
                observation[velocity_output + 3 + axis] = wrist_velocity_scale * angular_velocity_b[axis]

        reference_hand = frame * num_hands + hand
        reference_quat = reference_wrist_quat_w[reference_hand]
        if make_wrist_quat_unique:
            reference_quat = unique_quat(reference_quat)
        relative_pos = wp.transform_point(wrist_inv, reference_wrist_pos_w[reference_hand])
        relative_quat = quat_mul_xyzw(quat_inv_xyzw(wrist_quat_observation), reference_quat)
        if command_wrist_start >= 0:
            command_output = output + command_wrist_start + hand * 7
            for axis in range(3):
                observation[command_output + axis] = command_wrist_scale * relative_pos[axis]
            write_wxyz(observation, command_output + 3, relative_quat, command_wrist_scale)

    for arm in range(total_arms):
        position = joint_q[q_base + arm_q_ids[arm]]
        if arm_position_start >= 0:
            observation[output + arm_position_start + arm] = arm_position_scale * position
        if arm_velocity_start >= 0:
            observation[output + arm_velocity_start + arm] = arm_velocity_scale * joint_qd[dof_base + arm_dof_ids[arm]]
        if arm_reference_delta_start >= 0:
            reference_position = reference_arm_joint_pos[frame * total_arms + arm]
            observation[output + arm_reference_delta_start + arm] = arm_reference_delta_scale * (
                reference_position - position
            )

    for finger in range(total_fingers):
        q_id = finger_q_ids[finger]
        dof_id = finger_dof_ids[finger]
        position = joint_q[q_base + q_id]
        if finger_pos_start >= 0:
            lower = joint_limit_lower[dof_base + dof_id]
            upper = joint_limit_upper[dof_base + dof_id]
            normalized = float(0.0)
            if upper - lower > 1.0e-8:
                normalized = 2.0 * (position - lower) / (upper - lower) - 1.0
            observation[output + finger_pos_start + finger] = finger_position_scale * normalized
        if finger_velocity_start >= 0:
            observation[output + finger_velocity_start + finger] = finger_velocity_scale * joint_qd[dof_base + dof_id]
        if command_finger_start >= 0:
            observation[output + command_finger_start + finger] = command_finger_scale * (
                reference_finger_joint_pos[frame * total_fingers + finger] - position
            )

    for object_id in range(num_objects):
        object_xf = body_q[body_base + object_body_ids[object_id]]
        object_pos_w = wp.transform_get_translation(object_xf)
        object_quat_w = wp.transform_get_rotation(object_xf)
        if object_pos_start >= 0:
            position_output = output + object_pos_start + object_id * 3
            for axis in range(3):
                observation[position_output + axis] = object_position_scale * object_pos_w[axis]
        if object_quat_start >= 0:
            write_wxyz(
                observation,
                output + object_quat_start + object_id * 4,
                object_quat_w,
                object_orientation_scale,
            )
        if object_velocity_start >= 0:
            object_velocity_w = body_qd[body_base + object_body_ids[object_id]]
            linear_velocity_w = wp.spatial_top(object_velocity_w)
            angular_velocity_w = wp.spatial_bottom(object_velocity_w)
            velocity_output = output + object_velocity_start + object_id * 6
            for axis in range(3):
                observation[velocity_output + axis] = object_body_velocity_scale * linear_velocity_w[axis]
                observation[velocity_output + 3 + axis] = object_body_velocity_scale * angular_velocity_w[axis]

        object_inv = wp.transform_inverse(object_xf)
        reference_object = frame * num_objects + object_id
        relative_pos = wp.transform_point(object_inv, reference_object_pos_w[reference_object])
        relative_quat = quat_mul_xyzw(
            quat_inv_xyzw(object_quat_w),
            unique_quat(reference_object_quat_w[reference_object]),
        )
        if command_object_pos_start >= 0:
            command_position_output = output + command_object_pos_start + object_id * 3
            for axis in range(3):
                observation[command_position_output + axis] = command_object_position_scale * relative_pos[axis]
        if command_object_quat_start >= 0:
            write_wxyz(
                observation,
                output + command_object_quat_start + object_id * 4,
                relative_quat,
                command_object_orientation_scale,
            )

    if raw_action_start >= 0:
        for action_id in range(action_dim):
            observation[output + raw_action_start + action_id] = (
                raw_action_scale * raw_action[world * action_dim + action_id]
            )
    if processed_action_start >= 0:
        for action_id in range(processed_dim):
            observation[output + processed_action_start + action_id] = (
                processed_action_scale * processed_action[world * processed_dim + action_id]
            )

    for side in range(contact_side_count):
        source = world * contact_slots_per_world + contact_source_starts[side]
        slot_count = contact_slot_counts[side]
        position_output = contact_position_starts[side]
        direction_output = contact_direction_starts[side]
        for slot in range(slot_count):
            contact_position = contact_pos_b[source + slot]
            contact_direction = contact_direction_b[source + slot]
            for axis in range(3):
                position_value = contact_position_scale * contact_position[axis]
                direction_value = contact_direction_scale * contact_direction[axis]
                if episode_step[world] == 0:
                    position_value = 0.0
                    direction_value = 0.0
                if position_output >= 0:
                    observation[output + position_output + slot * 3 + axis] = position_value
                if direction_output >= 0:
                    observation[output + direction_output + slot * 3 + axis] = direction_value


@dataclass
class PolicyObservation:
    """Policy observation assembled by one graph-capturable Warp launch."""

    layout: PolicyObservationLayout
    config: PolicyObservationConfig
    world_count: int
    bodies_per_world: int
    num_joint_q: int
    num_joint_dof: int
    model: object
    timestep: wp.array
    episode_step: wp.array
    action: PolicyAction
    reference: DeviceRobotReference
    command: CommandBuffers
    contact: ContactTracker | None
    contact_slots_per_world: int
    wrist_frames: DeviceBodyFrameMap
    arm_q_ids: wp.array
    arm_dof_ids: wp.array
    finger_q_ids: wp.array
    finger_dof_ids: wp.array
    object_body_ids: wp.array
    contact_source_starts: wp.array
    contact_slot_counts: wp.array
    contact_position_starts: wp.array
    contact_direction_starts: wp.array
    term_scales: wp.array
    empty_contact_vec3: wp.array
    observation: wp.array

    @classmethod
    def build(
        cls,
        scene: Scene,
        action: PolicyAction,
        command: CommandBuffers,
        contact: ContactTracker | None,
        timestep: wp.array,
        episode_step: wp.array,
        config: PolicyObservationConfig | None = None,
        device=None,
    ) -> "PolicyObservation":
        """Precompute source mappings and allocate one persistent output buffer."""
        config = config or PolicyObservationConfig()
        if not isinstance(action, PolicyAction):
            raise TypeError("policy observation requires an action implementing PolicyAction")
        if command.world_count != scene.world_count:
            raise ValueError("command and scene world counts must match")
        if command.reference.num_frames != scene.robot_reference.num_frames:
            raise ValueError(
                f"command has {command.reference.num_frames} reference frames; "
                f"robot binding has {scene.robot_reference.num_frames}"
            )
        layout = PolicyObservationLayout.build(scene, action, command, contact, config)
        reference = DeviceRobotReference.build(
            scene.robot_reference,
            scene.layout,
            config.sides,
            device=device,
        )
        world_count = scene.world_count
        bodies_per_world = scene.model.body_count // world_count
        num_joint_q = scene.model.joint_coord_count // world_count
        num_joint_dof = scene.model.joint_dof_count // world_count

        wrist_frames = [scene.layout.hand(side).palm_frame for side in config.sides]

        contact_source_starts: list[int] = []
        if contact is not None:
            for side in config.contact_sides:
                hand = contact.layout.sides.index(side)
                contact_source_starts.append(contact.layout.hand_slot_starts[hand])
        contact_slots_per_world = contact.layout.slots_per_world if contact is not None else 0

        i32 = lambda values: wp.array(values, dtype=wp.int32, device=device)  # noqa: E731
        return cls(
            layout=layout,
            config=config,
            world_count=world_count,
            bodies_per_world=bodies_per_world,
            num_joint_q=num_joint_q,
            num_joint_dof=num_joint_dof,
            model=scene.model,
            timestep=timestep,
            episode_step=episode_step,
            action=action,
            reference=reference,
            command=command,
            contact=contact,
            contact_slots_per_world=contact_slots_per_world,
            wrist_frames=DeviceBodyFrameMap.build(wrist_frames, device=device),
            arm_q_ids=i32(layout.arm_joints.q_ids),
            arm_dof_ids=i32(layout.arm_joints.dof_ids),
            finger_q_ids=i32(layout.finger_joints.q_ids),
            finger_dof_ids=i32(layout.finger_joints.dof_ids),
            object_body_ids=command.body_ids,
            contact_source_starts=i32(contact_source_starts),
            contact_slot_counts=i32(layout.contact_slot_counts),
            contact_position_starts=i32(layout.contact_position_starts),
            contact_direction_starts=i32(layout.contact_direction_starts),
            term_scales=wp.array([term.scale for term in config.terms], dtype=wp.float32, device=device),
            empty_contact_vec3=wp.empty(0, dtype=wp.vec3, device=device),
            observation=wp.zeros(world_count * layout.observation_dim, dtype=wp.float32, device=device),
        )

    @property
    def observation_dim(self) -> int:
        return self.layout.observation_dim

    @property
    def block_names(self) -> tuple[str, ...]:
        return self.layout.block_names

    @property
    def block_ranges(self) -> tuple[tuple[int, int], ...]:
        return self.layout.block_ranges

    def compute(self, state) -> wp.array:
        """Assemble current state, command, action, and live-contact values."""
        return self.compute_into(state, self.observation)

    def compute_into(self, state, output: wp.array) -> wp.array:
        """Assemble an observation directly into a compatible caller-owned buffer."""
        expected = (self.world_count * self.observation_dim,)
        if output.shape != expected:
            raise ValueError(f"observation output has shape {output.shape}; expected {expected}")
        if output.dtype != wp.float32:
            raise TypeError(f"observation output must be float32, got {output.dtype}")
        if output.device != self.observation.device:
            raise ValueError(
                f"observation output is on {output.device}; expected primary device {self.observation.device}"
            )
        layout = self.layout
        contact_pos_b = self.empty_contact_vec3
        contact_direction_b = self.empty_contact_vec3
        if self.contact is not None:
            contact_pos_b = self.contact.contact_pos_b
            contact_direction_b = self.contact.contact_force_direction_b
        wp.launch(
            compute_policy_observation,
            dim=self.world_count,
            inputs=[
                state.body_q,
                state.body_qd,
                state.joint_q,
                state.joint_qd,
                self.timestep,
                self.episode_step,
                self.model.joint_limit_lower,
                self.model.joint_limit_upper,
                self.reference.num_frames,
                self.bodies_per_world,
                self.num_joint_q,
                self.num_joint_dof,
                layout.num_hands,
                layout.total_arms,
                layout.total_fingers,
                layout.num_objects,
                layout.action_dim,
                layout.processed_dim,
                layout.observation_dim,
                self.wrist_frames.body_ids,
                self.wrist_frames.body_to_frame_pos,
                self.wrist_frames.body_to_frame_quat,
                self.arm_q_ids,
                self.arm_dof_ids,
                self.finger_q_ids,
                self.finger_dof_ids,
                self.object_body_ids,
                self.reference.wrist_pos_w,
                self.reference.wrist_quat_w,
                self.reference.arm_joint_pos,
                self.reference.finger_joint_pos,
                self.command.reference.body_pos_w,
                self.command.reference.body_quat_w,
                self.action.raw_action,
                self.action.processed_target,
                self.term_scales,
                contact_pos_b,
                contact_direction_b,
                len(layout.contact_sides),
                self.contact_slots_per_world,
                self.contact_source_starts,
                self.contact_slot_counts,
                self.contact_position_starts,
                self.contact_direction_starts,
                self.config.make_wrist_quat_unique,
                layout.wrist_pos_start,
                layout.wrist_quat_start,
                layout.wrist_velocity_start,
                layout.arm_position_start,
                layout.arm_velocity_start,
                layout.arm_reference_delta_start,
                layout.finger_pos_start,
                layout.finger_velocity_start,
                layout.object_pos_start,
                layout.object_quat_start,
                layout.object_velocity_start,
                layout.command_wrist_start,
                layout.command_finger_start,
                layout.command_object_pos_start,
                layout.command_object_quat_start,
                layout.raw_action_start,
                layout.processed_action_start,
            ],
            outputs=[output],
        )
        return output
