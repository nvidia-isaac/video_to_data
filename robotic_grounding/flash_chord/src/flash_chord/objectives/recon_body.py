# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BodyRecon tracking objective and binary-label-gated live force closure."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import warp as wp

from flash_chord.data.reference import Reference
from flash_chord.embodiments.base import EmbodimentLayout
from flash_chord.embodiments.binding import DeviceRobotReference, RobotReferenceBinding
from flash_chord.embodiments.frames import DeviceBodyFrameMap
from flash_chord.objectives.wrench import batched_wrench_support, friction_cone_angles, sample_wrench_basis
from flash_chord.runtime.actions import Action, RegularizedAction
from flash_chord.runtime.actions.sonic import SonicJointResidualAction
from flash_chord.runtime.command import CommandBuffers
from flash_chord.runtime.contact import ContactTracker
from flash_chord.utils.quat import quat_geodesic_angle, quat_mul_xyzw

RECON_BODY_OBJECTIVE_TERM_NAMES = (
    "termination",
    "action_rate_l2",
    "action_l2",
    "soft_joint_limit",
    "pelvis_position",
    "pelvis_orientation",
    "body_joint_position",
    "object_position",
    "progress",
    "palm_position",
    "palm_orientation",
    "force_closure",
)
_RECON_BODY_OBJECTIVE_TERM_COUNT = wp.constant(12)


@dataclass(frozen=True)
class ReconBodyObjectiveConfig:
    """Exact BodyRecon weights, shaping scales, and wrench approximation."""

    termination_weight: float = -300.0
    action_rate_l2_weight: float = -1.0e-4
    action_l2_weight: float = -1.0e-6
    soft_joint_limit_weight: float = -1.0e-3
    pelvis_position_weight: float = 1.0
    pelvis_orientation_weight: float = 1.0
    body_joint_position_weight: float = 5.0
    object_position_weight: float = 1.0
    progress_weight: float = 1.0
    palm_position_weight: float = 1.0
    palm_orientation_weight: float = 1.0
    force_closure_weight: float = 5.0
    pelvis_position_std: float = 0.3
    pelvis_orientation_std: float = 0.4
    body_joint_position_std: float = 1.0
    object_position_std: float = 0.2
    palm_position_std: float = 0.2
    palm_orientation_std: float = 0.4
    soft_joint_limit_factor: float = 0.9
    force_closure_min_support: float = 0.01
    num_wrench_basis: int = 512
    num_friction_cone_edges: int = 8
    friction_coefficient: float = 0.1
    wrench_basis_seed: int = 0
    sides: tuple[str, ...] = ("left", "right")

    def __post_init__(self) -> None:
        values = self.weights + (
            self.pelvis_position_std,
            self.pelvis_orientation_std,
            self.body_joint_position_std,
            self.object_position_std,
            self.palm_position_std,
            self.palm_orientation_std,
            self.force_closure_min_support,
            self.friction_coefficient,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("ReconBody objective parameters must be finite")
        if any(
            value <= 0.0
            for value in (
                self.pelvis_position_std,
                self.pelvis_orientation_std,
                self.body_joint_position_std,
                self.object_position_std,
                self.palm_position_std,
                self.palm_orientation_std,
            )
        ):
            raise ValueError("ReconBody shaping standard deviations must be positive")
        if not 0.0 < self.soft_joint_limit_factor <= 1.0:
            raise ValueError("ReconBody soft joint limit factor must be in (0, 1]")
        if self.force_closure_min_support < 0.0 or self.friction_coefficient < 0.0:
            raise ValueError("ReconBody wrench thresholds must be non-negative")
        if self.num_wrench_basis <= 0 or self.num_friction_cone_edges <= 0:
            raise ValueError("ReconBody wrench discretization sizes must be positive")
        object.__setattr__(self, "sides", tuple(self.sides))

    @property
    def weights(self) -> tuple[float, ...]:
        return (
            self.termination_weight,
            self.action_rate_l2_weight,
            self.action_l2_weight,
            self.soft_joint_limit_weight,
            self.pelvis_position_weight,
            self.pelvis_orientation_weight,
            self.body_joint_position_weight,
            self.object_position_weight,
            self.progress_weight,
            self.palm_position_weight,
            self.palm_orientation_weight,
            self.force_closure_weight,
        )

    def build(
        self,
        model,
        embodiment: EmbodimentLayout,
        robot_reference: RobotReferenceBinding,
        reference: Reference,
        command: CommandBuffers,
        contact: ContactTracker | None,
        action: Action,
        world_count: int,
        frame_dt: float,
        device=None,
        reference_joint_q_offset: wp.array | None = None,
        episode_start_frame: wp.array | None = None,
    ) -> ReconBodyObjective:
        if reference_joint_q_offset is None or episode_start_frame is None:
            raise ValueError("ReconBody objective requires reset offset and episode-start buffers")
        return ReconBodyObjective.build(
            model,
            embodiment,
            robot_reference,
            reference,
            command,
            contact,
            action,
            reference_joint_q_offset,
            episode_start_frame,
            world_count,
            frame_dt,
            config=self,
            device=device,
        )


@wp.kernel
def reduce_recon_body_force_closure(
    timestep: wp.array(dtype=wp.int32),
    contact_active: wp.array(dtype=wp.float32),
    current_support: wp.array(dtype=wp.float32),
    num_frames: int,
    num_hands: int,
    num_bodies: int,
    num_basis: int,
    min_support: float,
    force_closure: wp.array(dtype=wp.float32),
) -> None:
    world = wp.tid()
    frame = wp.clamp(timestep[world], 0, num_frames - 1)
    reward = float(0.0)  # noqa: UP018 - Warp requires explicitly typed mutable locals.
    active_hands = float(0.0)  # noqa: UP018
    for hand in range(num_hands):
        if contact_active[frame * num_hands + hand] > 0.5:
            active_hands += 1.0
            covered = float(0.0)  # noqa: UP018
            for basis in range(num_basis):
                best = float(0.0)  # noqa: UP018
                for body in range(num_bodies):
                    index = ((world * num_hands + hand) * num_bodies + body) * num_basis + basis
                    best = wp.max(best, current_support[index])
                if best > min_support:
                    covered += 1.0
            reward += covered / float(num_basis)
    force_closure[world] = reward / wp.max(active_hands, 1.0)


@wp.kernel
def compose_recon_body_objective(
    body_q: wp.array(dtype=wp.transform),
    joint_q: wp.array(dtype=wp.float32),
    timestep: wp.array(dtype=wp.int32),
    episode_start_frame: wp.array(dtype=wp.int32),
    terminated: wp.array(dtype=wp.int32),
    action_rate_l2: wp.array(dtype=wp.float32),
    action_l2: wp.array(dtype=wp.float32),
    force_closure: wp.array(dtype=wp.float32),
    reference_joint_q_offset: wp.array(dtype=wp.float32),
    reference_joint_q: wp.array(dtype=wp.float32, ndim=2),
    reference_wrist_pos_w: wp.array(dtype=wp.vec3),
    reference_wrist_quat_w: wp.array(dtype=wp.quat),
    wrist_body_ids: wp.array(dtype=wp.int32),
    wrist_local_pos: wp.array(dtype=wp.vec3),
    wrist_local_quat: wp.array(dtype=wp.quat),
    scalar_q_ids: wp.array(dtype=wp.int32),
    sonic_q_ids: wp.array(dtype=wp.int32),
    soft_lower: wp.array(dtype=wp.float32),
    soft_upper: wp.array(dtype=wp.float32),
    object_body_id: int,
    object_target_pos_w: wp.array(dtype=wp.vec3),
    pelvis_body_id: int,
    num_frames: int,
    bodies_per_world: int,
    q_per_world: int,
    num_joints: int,
    num_sonic_joints: int,
    pelvis_position_std: float,
    pelvis_orientation_std: float,
    body_joint_position_std: float,
    object_position_std: float,
    palm_position_std: float,
    palm_orientation_std: float,
    weights: wp.array(dtype=wp.float32),
    frame_dt: float,
    terms: wp.array(dtype=wp.float32),
    score: wp.array(dtype=wp.float32),
) -> None:
    world = wp.tid()
    frame = wp.clamp(timestep[world], 0, num_frames - 1)
    q_base = world * q_per_world
    body_base = world * bodies_per_world
    term_base = world * _RECON_BODY_OBJECTIVE_TERM_COUNT

    terms[term_base] = float(terminated[world])
    terms[term_base + 1] = action_rate_l2[world]
    terms[term_base + 2] = action_l2[world]

    limit_error = float(0.0)  # noqa: UP018 - Warp requires explicitly typed mutable locals.
    for joint in range(num_joints):
        value = joint_q[q_base + scalar_q_ids[joint]]
        limit_error += wp.max(soft_lower[joint] - value, 0.0)
        limit_error += wp.max(value - soft_upper[joint], 0.0)
    terms[term_base + 3] = limit_error

    pelvis_xf = body_q[body_base + pelvis_body_id]
    pelvis_position = wp.transform_get_translation(pelvis_xf)
    pelvis_orientation = wp.normalize(wp.transform_get_rotation(pelvis_xf))
    reference_pelvis_position = wp.vec3(
        reference_joint_q[frame, 0],
        reference_joint_q[frame, 1],
        reference_joint_q[frame, 2],
    )
    reference_pelvis_orientation = wp.normalize(
        wp.quat(
            reference_joint_q[frame, 3],
            reference_joint_q[frame, 4],
            reference_joint_q[frame, 5],
            reference_joint_q[frame, 6],
        )
    )
    pelvis_position_error = pelvis_position - reference_pelvis_position
    pelvis_position_sse = wp.dot(pelvis_position_error, pelvis_position_error)
    pelvis_orientation_error = quat_geodesic_angle(pelvis_orientation, reference_pelvis_orientation)
    terms[term_base + 4] = wp.exp(-pelvis_position_sse / (pelvis_position_std * pelvis_position_std))
    terms[term_base + 5] = wp.exp(
        -(pelvis_orientation_error * pelvis_orientation_error) / (pelvis_orientation_std * pelvis_orientation_std)
    )

    joint_sse = float(0.0)  # noqa: UP018
    for joint in range(num_sonic_joints):
        q_id = sonic_q_ids[joint]
        error = joint_q[q_base + q_id] - (reference_joint_q[frame, q_id] + reference_joint_q_offset[q_base + q_id])
        joint_sse += error * error
    terms[term_base + 6] = wp.exp(-joint_sse / (body_joint_position_std * body_joint_position_std))

    object_position = wp.transform_get_translation(body_q[body_base + object_body_id])
    object_error = object_position - object_target_pos_w[world]
    terms[term_base + 7] = wp.exp(-wp.dot(object_error, object_error) / (object_position_std * object_position_std))

    denominator = num_frames - 1 - episode_start_frame[world]
    progress = float(1.0)  # noqa: UP018
    if denominator > 0:
        progress = wp.clamp(
            float(frame - episode_start_frame[world]) / float(denominator),
            0.0,
            1.0,
        )
    terms[term_base + 8] = progress

    palm_position_sse = float(0.0)  # noqa: UP018
    palm_orientation_sse = float(0.0)  # noqa: UP018
    for hand in range(2):
        body_xf = body_q[body_base + wrist_body_ids[hand]]
        palm_position = wp.transform_point(body_xf, wrist_local_pos[hand])
        palm_orientation = wp.normalize(quat_mul_xyzw(wp.transform_get_rotation(body_xf), wrist_local_quat[hand]))
        position_error = palm_position - reference_wrist_pos_w[frame * 2 + hand]
        palm_position_sse += wp.dot(position_error, position_error)
        orientation_error = quat_geodesic_angle(
            palm_orientation,
            reference_wrist_quat_w[frame * 2 + hand],
        )
        palm_orientation_sse += orientation_error * orientation_error
    terms[term_base + 9] = wp.exp(-palm_position_sse / (palm_position_std * palm_position_std))
    terms[term_base + 10] = wp.exp(-palm_orientation_sse / (palm_orientation_std * palm_orientation_std))
    terms[term_base + 11] = force_closure[world]

    total = float(0.0)  # noqa: UP018
    for term in range(_RECON_BODY_OBJECTIVE_TERM_COUNT):
        total += weights[term] * terms[term_base + term]
    score[world] = frame_dt * total


@dataclass
class ReconBodyObjective:
    """Device BodyRecon objective with named diagnostics and mutable curriculum weights."""

    config: ReconBodyObjectiveConfig
    world_count: int
    frame_dt: float
    num_frames: int
    num_hands: int
    num_bodies: int
    num_basis: int
    num_edges: int
    num_joints: int
    bodies_per_world: int
    q_per_world: int
    pelvis_body_id: int
    object_body_id: int
    episode_start_frame: wp.array
    reference_joint_q_offset: wp.array
    reference: DeviceRobotReference
    command: CommandBuffers
    contact: ContactTracker
    action: SonicJointResidualAction
    wrist_frames: DeviceBodyFrameMap
    scalar_q_ids: wp.array
    soft_lower: wp.array
    soft_upper: wp.array
    contact_active: wp.array
    wrench_basis: wp.array
    cone_cos: wp.array
    cone_sin: wp.array
    wrench_contact_enabled: wp.array
    current_wrench_support: wp.array
    force_closure: wp.array
    weights: wp.array
    terms: wp.array
    score: wp.array

    @classmethod
    def build(
        cls,
        model,
        embodiment: EmbodimentLayout,
        robot_reference: RobotReferenceBinding,
        reference: Reference,
        command: CommandBuffers,
        contact: ContactTracker | None,
        action: Action,
        reference_joint_q_offset: wp.array,
        episode_start_frame: wp.array,
        world_count: int,
        frame_dt: float,
        config: ReconBodyObjectiveConfig | None = None,
        device=None,
    ) -> ReconBodyObjective:
        config = config or ReconBodyObjectiveConfig()
        if not isinstance(action, SonicJointResidualAction) or not isinstance(action, RegularizedAction):
            raise TypeError("ReconBody objective requires a regularized SonicJointResidualAction")
        if contact is None:
            raise ValueError("ReconBody force closure requires a ContactTracker")
        if contact.layout.sides != config.sides:
            raise ValueError(
                f"ReconBody objective sides {config.sides} must match contact sides {contact.layout.sides}"
            )
        if command.layout.num_objects != 1 or command.layout.num_bodies != 1:
            raise ValueError("ReconBody objective currently requires one rigid object")
        scalar = embodiment.scalar_joints
        if scalar is None or len(scalar.names) != 43:
            raise ValueError("ReconBody objective requires the exact 43-joint G1+Dex3 layout")
        pelvis = tuple(frame for frame in embodiment.semantic_frames if frame.name == "pelvis")
        if len(pelvis) != 1:
            raise ValueError("ReconBody objective requires one pelvis semantic frame")
        device_reference = DeviceRobotReference.build(
            robot_reference,
            embodiment,
            sides=config.sides,
            device=device,
        )
        wrist_frames = DeviceBodyFrameMap.build(
            [embodiment.hand(side).palm_frame for side in config.sides],
            device=device,
        )
        limits_lower = np.asarray(model.joint_limit_lower.numpy(), dtype=np.float32)
        limits_upper = np.asarray(model.joint_limit_upper.numpy(), dtype=np.float32)
        dof_ids = np.asarray(scalar.dof_ids, dtype=np.int64)
        lower = limits_lower[dof_ids]
        upper = limits_upper[dof_ids]
        center = 0.5 * (lower + upper)
        half = 0.5 * config.soft_joint_limit_factor * (upper - lower)
        if not np.all(np.isfinite(center)) or not np.all(np.isfinite(half)):
            raise ValueError("ReconBody scalar joint limits must be finite")

        labels = np.stack(
            [np.asarray(reference.contact_active(side), dtype=np.float32) for side in config.sides],
            axis=1,
        )
        if labels.shape != (reference.num_frames, len(config.sides)):
            raise ValueError(
                f"ReconBody contact labels have shape {labels.shape}; "
                f"expected {(reference.num_frames, len(config.sides))}"
            )
        basis = np.concatenate(
            [
                sample_wrench_basis(config.num_wrench_basis, 1.0, config.wrench_basis_seed + body)
                for body in range(command.layout.num_bodies)
            ],
            axis=0,
        )
        cone_cos, cone_sin = friction_cone_angles(config.num_friction_cone_edges)
        support_size = world_count * len(config.sides) * command.layout.num_bodies * config.num_wrench_basis
        return cls(
            config=config,
            world_count=world_count,
            frame_dt=frame_dt,
            num_frames=reference.num_frames,
            num_hands=len(config.sides),
            num_bodies=command.layout.num_bodies,
            num_basis=config.num_wrench_basis,
            num_edges=config.num_friction_cone_edges,
            num_joints=len(scalar.names),
            bodies_per_world=model.body_count // world_count,
            q_per_world=model.joint_coord_count // world_count,
            pelvis_body_id=pelvis[0].body_id,
            object_body_id=command.layout.body_ids[0],
            episode_start_frame=episode_start_frame,
            reference_joint_q_offset=reference_joint_q_offset,
            reference=device_reference,
            command=command,
            contact=contact,
            action=action,
            wrist_frames=wrist_frames,
            scalar_q_ids=wp.array(scalar.q_ids, dtype=wp.int32, device=device),
            soft_lower=wp.array(center - half, dtype=wp.float32, device=device),
            soft_upper=wp.array(center + half, dtype=wp.float32, device=device),
            contact_active=wp.array(labels.reshape(-1), dtype=wp.float32, device=device),
            wrench_basis=wp.array(basis, dtype=wp.spatial_vector, device=device),
            cone_cos=wp.array(cone_cos, dtype=wp.float32, device=device),
            cone_sin=wp.array(cone_sin, dtype=wp.float32, device=device),
            wrench_contact_enabled=wp.ones(contact.layout.slots_per_world, dtype=wp.int32, device=device),
            current_wrench_support=wp.zeros(support_size, dtype=wp.float32, device=device),
            force_closure=wp.zeros(world_count, dtype=wp.float32, device=device),
            weights=wp.array(config.weights, dtype=wp.float32, device=device),
            terms=wp.zeros(
                world_count * len(RECON_BODY_OBJECTIVE_TERM_NAMES),
                dtype=wp.float32,
                device=device,
            ),
            score=wp.zeros(world_count, dtype=wp.float32, device=device),
        )

    @property
    def term_names(self) -> tuple[str, ...]:
        return RECON_BODY_OBJECTIVE_TERM_NAMES

    def set_weights(self, weights: Mapping[str, float]) -> None:
        if set(weights) != set(self.term_names):
            raise ValueError(
                f"ReconBody objective weights must contain exactly {self.term_names}, got {tuple(weights)}"
            )
        values = tuple(float(weights[name]) for name in self.term_names)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("ReconBody objective weights must be finite")
        self.weights.assign(values)

    def reset(self, reset_mask: wp.array) -> None:
        del reset_mask

    def evaluate(self, state, timestep: wp.array, terminated: wp.array) -> None:
        wp.launch(
            batched_wrench_support,
            dim=self.world_count * self.num_hands * self.num_bodies * self.num_basis,
            inputs=[
                self.contact.contact_pos_o,
                self.contact.contact_force_direction_o,
                self.wrench_contact_enabled,
                self.contact.hand_slot_starts,
                self.contact.link_counts,
                self.contact.layout.slots_per_world,
                self.num_hands,
                self.num_bodies,
                self.wrench_basis,
                self.cone_cos,
                self.cone_sin,
                self.config.friction_coefficient,
                self.command.object_radius,
                self.num_edges,
                self.num_basis,
            ],
            outputs=[self.current_wrench_support],
        )
        wp.launch(
            reduce_recon_body_force_closure,
            dim=self.world_count,
            inputs=[
                timestep,
                self.contact_active,
                self.current_wrench_support,
                self.num_frames,
                self.num_hands,
                self.num_bodies,
                self.num_basis,
                self.config.force_closure_min_support,
            ],
            outputs=[self.force_closure],
        )
        wp.launch(
            compose_recon_body_objective,
            dim=self.world_count,
            inputs=[
                state.body_q,
                state.joint_q,
                timestep,
                self.episode_start_frame,
                terminated,
                self.action.action_rate_l2,
                self.action.action_l2,
                self.force_closure,
                self.reference_joint_q_offset,
                self.reference.joint_q,
                self.reference.wrist_pos_w,
                self.reference.wrist_quat_w,
                self.wrist_frames.body_ids,
                self.wrist_frames.body_to_frame_pos,
                self.wrist_frames.body_to_frame_quat,
                self.scalar_q_ids,
                self.action.sonic_q_ids,
                self.soft_lower,
                self.soft_upper,
                self.object_body_id,
                self.command.body_target_pos_w,
                self.pelvis_body_id,
                self.num_frames,
                self.bodies_per_world,
                self.q_per_world,
                self.num_joints,
                self.action.sonic_q_ids.shape[0],
                self.config.pelvis_position_std,
                self.config.pelvis_orientation_std,
                self.config.body_joint_position_std,
                self.config.object_position_std,
                self.config.palm_position_std,
                self.config.palm_orientation_std,
                self.weights,
                self.frame_dt,
            ],
            outputs=[self.terms, self.score],
        )
