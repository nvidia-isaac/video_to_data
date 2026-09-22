# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused 486-float ReconBody actor observation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from flash_chord.embodiments.binding import DeviceRobotReference
from flash_chord.embodiments.frames import DeviceBodyFrameMap
from flash_chord.runtime.actions import PolicyAction
from flash_chord.runtime.actions.sonic import RECON_BODY_ACTION_HISTORY_LENGTH, SonicJointResidualAction
from flash_chord.runtime.command import CommandBuffers
from flash_chord.runtime.contact import ContactTracker
from flash_chord.scene.builder import Scene
from flash_chord.utils.quat import quat_inv_xyzw, quat_mul_xyzw

RECON_BODY_OBSERVATION_DIM = 486
RECON_BODY_FUTURE_OFFSETS = (0, 5, 10)

RECON_BODY_OBSERVATION_BLOCKS = (
    ("wrist_position_pelvis_m[right,left]", 6),
    ("wrist_orientation_pelvis_6d[right,left]", 12),
    ("wrist_linear_velocity_pelvis_mps[right,left]", 6),
    ("object_position_pelvis_m", 3),
    ("object_orientation_pelvis_6d", 6),
    ("joint_position_relative_default_rad[43]", 43),
    ("joint_velocity_radps[43]", 43),
    ("future_root_position_w_m[offsets=0,5,10]", 9),
    ("future_root_orientation_delta_6d[offsets=0,5,10]", 18),
    ("future_joint_position_delta_rad[offsets=0,5,10]", 129),
    ("future_palm_position_delta_w_m[offsets=0,5,10;right,left]", 18),
    ("future_palm_orientation_delta_6d[offsets=0,5,10;right,left]", 36),
    ("live_hand_pose_in_object[position,6d;left,right]", 18),
    ("target_object_pose_in_live_object[position,6d]", 9),
    ("normalized_reference_phase", 1),
    ("processed_target_history[oldest,newest;3x43]", 129),
)


@wp.func
def _rotation_6d_component(q: wp.quat, component: int) -> float:
    """First two rotation-matrix columns in row-major flatten order."""
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    value = float(0.0)  # noqa: UP018 - Warp requires an explicitly typed mutable local.
    if component == 0:
        value = 1.0 - 2.0 * (y * y + z * z)
    elif component == 1:
        value = 2.0 * (x * y - z * w)
    elif component == 2:
        value = 2.0 * (x * y + z * w)
    elif component == 3:
        value = 1.0 - 2.0 * (x * x + z * z)
    elif component == 4:
        value = 2.0 * (x * z - y * w)
    else:
        value = 2.0 * (y * z + x * w)
    return value


@wp.func
def _frame_transform(
    body_q: wp.array(dtype=wp.transform),
    body_index: int,
    local_position: wp.array(dtype=wp.vec3),
    local_orientation: wp.array(dtype=wp.quat),
    frame_index: int,
) -> wp.transform:
    body_xf = body_q[body_index]
    position = wp.transform_point(body_xf, local_position[frame_index])
    orientation = wp.normalize(quat_mul_xyzw(wp.transform_get_rotation(body_xf), local_orientation[frame_index]))
    return wp.transform(position, orientation)


@wp.kernel
def compute_recon_body_observation(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    timestep: wp.array(dtype=wp.int32),
    reference_joint_q_offset: wp.array(dtype=wp.float32),
    reference_joint_q: wp.array(dtype=wp.float32, ndim=2),
    reference_wrist_pos_w: wp.array(dtype=wp.vec3),
    reference_wrist_quat_w: wp.array(dtype=wp.quat),
    reference_object_pos_w: wp.array(dtype=wp.vec3),
    reference_object_quat_w: wp.array(dtype=wp.quat),
    actor_action_history: wp.array(dtype=wp.float32),
    wrist_body_ids: wp.array(dtype=wp.int32),
    wrist_local_pos: wp.array(dtype=wp.vec3),
    wrist_local_quat: wp.array(dtype=wp.quat),
    scalar_q_ids: wp.array(dtype=wp.int32),
    scalar_dof_ids: wp.array(dtype=wp.int32),
    default_joint_position: wp.array(dtype=wp.float32),
    object_body_id: int,
    pelvis_body_id: int,
    num_frames: int,
    bodies_per_world: int,
    q_per_world: int,
    dof_per_world: int,
    num_joints: int,
    observation: wp.array(dtype=wp.float32),
) -> None:
    world = wp.tid()
    output = world * RECON_BODY_OBSERVATION_DIM
    q_base = world * q_per_world
    dof_base = world * dof_per_world
    body_base = world * bodies_per_world
    frame = wp.clamp(timestep[world], 0, num_frames - 1)

    pelvis_xf = body_q[body_base + pelvis_body_id]
    pelvis_inverse = wp.transform_inverse(pelvis_xf)
    pelvis_quat_inverse = quat_inv_xyzw(wp.transform_get_rotation(pelvis_xf))
    wrist_xf_0 = _frame_transform(
        body_q,
        body_base + wrist_body_ids[0],
        wrist_local_pos,
        wrist_local_quat,
        0,
    )
    wrist_xf_1 = _frame_transform(
        body_q,
        body_base + wrist_body_ids[1],
        wrist_local_pos,
        wrist_local_quat,
        1,
    )

    cursor = 0
    for hand in range(2):
        wrist_xf = wrist_xf_0
        if hand == 1:
            wrist_xf = wrist_xf_1
        position_pelvis = wp.transform_point(pelvis_inverse, wp.transform_get_translation(wrist_xf))
        for axis in range(3):
            observation[output + cursor + hand * 3 + axis] = position_pelvis[axis]
    cursor += 6

    for hand in range(2):
        wrist_xf = wrist_xf_0
        if hand == 1:
            wrist_xf = wrist_xf_1
        relative = wp.normalize(quat_mul_xyzw(pelvis_quat_inverse, wp.transform_get_rotation(wrist_xf)))
        for component in range(6):
            observation[output + cursor + hand * 6 + component] = _rotation_6d_component(relative, component)
    cursor += 12

    for hand in range(2):
        body_id = wrist_body_ids[hand]
        velocity_w = wp.spatial_top(body_qd[body_base + body_id])
        velocity_pelvis = wp.transform_vector(pelvis_inverse, velocity_w)
        for axis in range(3):
            observation[output + cursor + hand * 3 + axis] = velocity_pelvis[axis]
    cursor += 6

    object_xf = body_q[body_base + object_body_id]
    object_inverse = wp.transform_inverse(object_xf)
    object_pelvis = wp.transform_multiply(pelvis_inverse, object_xf)
    object_position = wp.transform_get_translation(object_pelvis)
    object_orientation = wp.normalize(wp.transform_get_rotation(object_pelvis))
    for axis in range(3):
        observation[output + cursor + axis] = object_position[axis]
    cursor += 3
    for component in range(6):
        observation[output + cursor + component] = _rotation_6d_component(object_orientation, component)
    cursor += 6

    for joint in range(num_joints):
        q_id = scalar_q_ids[joint]
        observation[output + cursor + joint] = joint_q[q_base + q_id] - default_joint_position[joint]
    cursor += num_joints
    for joint in range(num_joints):
        observation[output + cursor + joint] = joint_qd[dof_base + scalar_dof_ids[joint]]
    cursor += num_joints

    current_reference_quat = wp.normalize(
        wp.quat(
            reference_joint_q[frame, 3],
            reference_joint_q[frame, 4],
            reference_joint_q[frame, 5],
            reference_joint_q[frame, 6],
        )
    )
    for future_index in range(3):
        future_frame = wp.min(frame + future_index * 5, num_frames - 1)
        for axis in range(3):
            observation[output + cursor + future_index * 3 + axis] = reference_joint_q[future_frame, axis]
    cursor += 9
    for future_index in range(3):
        future_frame = wp.min(frame + future_index * 5, num_frames - 1)
        future_quat = wp.normalize(
            wp.quat(
                reference_joint_q[future_frame, 3],
                reference_joint_q[future_frame, 4],
                reference_joint_q[future_frame, 5],
                reference_joint_q[future_frame, 6],
            )
        )
        delta = wp.normalize(quat_mul_xyzw(future_quat, quat_inv_xyzw(current_reference_quat)))
        for component in range(6):
            observation[output + cursor + future_index * 6 + component] = _rotation_6d_component(delta, component)
    cursor += 18

    for future_index in range(3):
        future_frame = wp.min(frame + future_index * 5, num_frames - 1)
        for joint in range(num_joints):
            q_id = scalar_q_ids[joint]
            target = reference_joint_q[future_frame, q_id] + reference_joint_q_offset[q_base + q_id]
            observation[output + cursor + future_index * num_joints + joint] = target - joint_q[q_base + q_id]
    cursor += 3 * num_joints

    for future_index in range(3):
        future_frame = wp.min(frame + future_index * 5, num_frames - 1)
        for hand in range(2):
            wrist_xf = wrist_xf_0
            if hand == 1:
                wrist_xf = wrist_xf_1
            live_position = wp.transform_get_translation(wrist_xf)
            target_position = reference_wrist_pos_w[future_frame * 2 + hand]
            delta_position = target_position - live_position
            for axis in range(3):
                observation[output + cursor + (future_index * 2 + hand) * 3 + axis] = delta_position[axis]
    cursor += 18
    for future_index in range(3):
        future_frame = wp.min(frame + future_index * 5, num_frames - 1)
        for hand in range(2):
            wrist_xf = wrist_xf_0
            if hand == 1:
                wrist_xf = wrist_xf_1
            live_quat = wp.normalize(wp.transform_get_rotation(wrist_xf))
            target_quat = reference_wrist_quat_w[future_frame * 2 + hand]
            delta = wp.normalize(quat_mul_xyzw(quat_inv_xyzw(live_quat), target_quat))
            block = (future_index * 2 + hand) * 6
            for component in range(6):
                observation[output + cursor + block + component] = _rotation_6d_component(delta, component)
    cursor += 36

    # This block follows the source's left/right order, the reverse of the wrist blocks above.
    for hand_lr in range(2):
        wrist_xf = wrist_xf_1
        if hand_lr == 1:
            wrist_xf = wrist_xf_0
        hand_object = wp.transform_multiply(object_inverse, wrist_xf)
        position = wp.transform_get_translation(hand_object)
        orientation = wp.normalize(wp.transform_get_rotation(hand_object))
        block = hand_lr * 9
        for axis in range(3):
            observation[output + cursor + block + axis] = position[axis]
        for component in range(6):
            observation[output + cursor + block + 3 + component] = _rotation_6d_component(orientation, component)
    cursor += 18

    target_object_xf = wp.transform(
        reference_object_pos_w[frame],
        reference_object_quat_w[frame],
    )
    target_object_live = wp.transform_multiply(object_inverse, target_object_xf)
    target_position = wp.transform_get_translation(target_object_live)
    target_orientation = wp.normalize(wp.transform_get_rotation(target_object_live))
    for axis in range(3):
        observation[output + cursor + axis] = target_position[axis]
    for component in range(6):
        observation[output + cursor + 3 + component] = _rotation_6d_component(target_orientation, component)
    cursor += 9

    phase = float(0.0)  # noqa: UP018 - Warp requires an explicitly typed mutable local.
    if num_frames > 1:
        phase = float(frame) / float(num_frames - 1)
    observation[output + cursor] = phase
    cursor += 1

    history_base = world * RECON_BODY_ACTION_HISTORY_LENGTH * num_joints
    for index in range(RECON_BODY_ACTION_HISTORY_LENGTH * num_joints):
        observation[output + cursor + index] = actor_action_history[history_base + index]


@dataclass(frozen=True)
class ReconBodyObservationConfig:
    """Source ReconBody schema with a true three-control-step target history."""

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
    ) -> ReconBodyObservation:
        del contact, episode_step
        return ReconBodyObservation.build(
            scene,
            action,
            command,
            timestep,
            reference_joint_q_offset,
            device=device,
        )


@dataclass
class ReconBodyObservation:
    """Persistent ReconBody observation assembled by one Warp launch."""

    world_count: int
    bodies_per_world: int
    q_per_world: int
    dof_per_world: int
    num_joints: int
    pelvis_body_id: int
    object_body_id: int
    timestep: wp.array
    reference_joint_q_offset: wp.array
    reference: DeviceRobotReference
    object_reference_pos_w: wp.array
    object_reference_quat_w: wp.array
    action: SonicJointResidualAction
    wrist_frames: DeviceBodyFrameMap
    scalar_q_ids: wp.array
    scalar_dof_ids: wp.array
    default_joint_position: wp.array
    observation: wp.array

    @classmethod
    def build(
        cls,
        scene: Scene,
        action: PolicyAction,
        command: CommandBuffers,
        timestep: wp.array,
        reference_joint_q_offset: wp.array,
        device=None,
    ) -> ReconBodyObservation:
        if not isinstance(action, SonicJointResidualAction):
            raise TypeError("ReconBody observation requires SonicJointResidualAction")
        if command.layout.num_bodies != 1:
            raise ValueError("ReconBody observation currently requires exactly one rigid object body")
        scalar = scene.layout.scalar_joints
        if scalar is None or len(scalar.names) != 43:
            raise ValueError("ReconBody observation requires the exact 43-joint G1+Dex3 scalar layout")
        pelvis = tuple(frame for frame in scene.layout.semantic_frames if frame.name == "pelvis")
        if len(pelvis) != 1:
            raise ValueError("ReconBody observation requires one pelvis semantic frame")
        sides = ("right", "left")
        reference = DeviceRobotReference.build(scene.robot_reference, scene.layout, sides=sides, device=device)
        wrist_frames = DeviceBodyFrameMap.build(
            [scene.layout.hand(side).palm_frame for side in sides],
            device=device,
        )
        world_count = scene.world_count
        q_per_world = scene.model.joint_coord_count // world_count
        dof_per_world = scene.model.joint_dof_count // world_count
        defaults = np.asarray(scene.model.joint_q.numpy(), dtype=np.float32)[:q_per_world]
        object_id = command.layout.body_ids[0]
        return cls(
            world_count=world_count,
            bodies_per_world=scene.model.body_count // world_count,
            q_per_world=q_per_world,
            dof_per_world=dof_per_world,
            num_joints=len(scalar.names),
            pelvis_body_id=pelvis[0].body_id,
            object_body_id=object_id,
            timestep=timestep,
            reference_joint_q_offset=reference_joint_q_offset,
            reference=reference,
            object_reference_pos_w=command.reference.body_pos_w,
            object_reference_quat_w=command.reference.body_quat_w,
            action=action,
            wrist_frames=wrist_frames,
            scalar_q_ids=wp.array(scalar.q_ids, dtype=wp.int32, device=device),
            scalar_dof_ids=wp.array(scalar.dof_ids, dtype=wp.int32, device=device),
            default_joint_position=wp.array(defaults[list(scalar.q_ids)], dtype=wp.float32, device=device),
            observation=wp.zeros(world_count * RECON_BODY_OBSERVATION_DIM, dtype=wp.float32, device=device),
        )

    @property
    def observation_dim(self) -> int:
        return RECON_BODY_OBSERVATION_DIM

    @property
    def block_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in RECON_BODY_OBSERVATION_BLOCKS)

    @property
    def block_ranges(self) -> tuple[tuple[int, int], ...]:
        ranges = []
        cursor = 0
        for _, width in RECON_BODY_OBSERVATION_BLOCKS:
            ranges.append((cursor, cursor + width))
            cursor += width
        return tuple(ranges)

    def compute(self, state) -> wp.array:
        return self.compute_into(state, self.observation)

    def compute_into(self, state, output: wp.array) -> wp.array:
        expected = (self.world_count * self.observation_dim,)
        if output.shape != expected or output.dtype != wp.float32 or output.device != self.observation.device:
            raise ValueError(f"ReconBody observation output must be float32 {expected} on {self.observation.device}")
        wp.launch(
            compute_recon_body_observation,
            dim=self.world_count,
            inputs=[
                state.body_q,
                state.body_qd,
                state.joint_q,
                state.joint_qd,
                self.timestep,
                self.reference_joint_q_offset,
                self.reference.joint_q,
                self.reference.wrist_pos_w,
                self.reference.wrist_quat_w,
                self.object_reference_pos_w,
                self.object_reference_quat_w,
                self.action.actor_action_history,
                self.wrist_frames.body_ids,
                self.wrist_frames.body_to_frame_pos,
                self.wrist_frames.body_to_frame_quat,
                self.scalar_q_ids,
                self.scalar_dof_ids,
                self.default_joint_position,
                self.object_body_id,
                self.pelvis_body_id,
                self.reference.num_frames,
                self.bodies_per_world,
                self.q_per_world,
                self.dof_per_world,
                self.num_joints,
            ],
            outputs=[output],
        )
        return output
