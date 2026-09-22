# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference-tracking objective strategy built from the objective kernels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np
import warp as wp

from flash_chord.data.reference import Reference
from flash_chord.embodiments.base import EmbodimentLayout
from flash_chord.embodiments.binding import HandKeypointFrame, RobotReferenceBinding
from flash_chord.objectives.composition import ObjectiveBuffers
from flash_chord.objectives.config import OBJECTIVE_TERM_NAMES, ObjectiveConfig
from flash_chord.objectives.contact import (
    advance_contact_force_history,
    contact_force_l2,
    contact_force_per_hand_log,
    contact_wrench_support_objective,
    force_closure_objective,
    missed_contact_penalty,
    reset_contact_force_history,
    unintended_contact_penalty,
)
from flash_chord.objectives.joints import hand_joint_objective
from flash_chord.objectives.keypoints import (
    KEYPOINT_VECS_NP,
    hand_keypoints_objective,
    object_keypoints_objective,
)
from flash_chord.objectives.wrench import (
    batched_wrench_support,
    compute_reference_supports,
    friction_cone_angles,
    merge_wrench_support,
    sample_wrench_basis,
)
from flash_chord.runtime.command import CommandBuffers
from flash_chord.runtime.contact import ContactTracker

_OBJECT_KEYPOINTS = OBJECTIVE_TERM_NAMES.index("object_keypoints")
_HAND_KEYPOINTS = OBJECTIVE_TERM_NAMES.index("hand_keypoints")
_HAND_JOINT_POS = OBJECTIVE_TERM_NAMES.index("hand_joint_pos")
_CONTACT_WRENCH_SUPPORT = OBJECTIVE_TERM_NAMES.index("contact_wrench_support")
_MISSED_CONTACT = OBJECTIVE_TERM_NAMES.index("missed_contact")
_UNINTENDED_CONTACT = OBJECTIVE_TERM_NAMES.index("unintended_contact")
_CONTACT_FORCE_L2 = OBJECTIVE_TERM_NAMES.index("contact_force_l2")
_FORCE_CLOSURE = OBJECTIVE_TERM_NAMES.index("force_closure")


def _hand_reference(
    embodiment: EmbodimentLayout,
    reference: RobotReferenceBinding,
    sides: tuple[str, ...],
    keypoint_frame: HandKeypointFrame,
    include_keypoints: bool,
    include_fingers: bool,
) -> tuple[np.ndarray, tuple[int, ...], np.ndarray, np.ndarray, tuple[int, ...]]:
    """Build wrist/fingertip and finger-joint reference trajectories in simulation order."""
    hands = [embodiment.hand(side) for side in sides]
    keypoint_frames = [
        (
            hand.palm_frame,
            *(hand.dp_frames if keypoint_frame == "dp" else hand.fingertip_frames),
        )
        for hand in hands
    ]
    keypoint_counts = {len(frames) for frames in keypoint_frames}
    finger_counts = {len(hand.finger_q_ids) for hand in hands}
    if len(keypoint_counts) != 1:
        raise ValueError(f"hand keypoint counts must match, got {sorted(keypoint_counts)}")
    if len(finger_counts) != 1:
        raise ValueError(f"finger joint counts must match, got {sorted(finger_counts)}")
    num_keypoints = keypoint_counts.pop()
    num_fingers = finger_counts.pop()

    keypoint_body_ids: list[int] = []
    keypoint_local_pos: list[tuple[float, float, float]] = []
    keypoint_targets: list[np.ndarray] = []
    finger_q_ids: list[int] = []
    finger_targets: list[np.ndarray] = []
    for hand, frames in zip(hands, keypoint_frames, strict=True):
        bound_hand = reference.hand(hand.side)
        keypoint_body_ids.extend(frame.body_id for frame in frames)
        keypoint_local_pos.extend(frame.body_to_frame_pos for frame in frames)
        if include_keypoints:
            keypoint_targets.append(bound_hand.keypoint_pos_w(keypoint_frame))
        else:
            keypoint_targets.append(np.zeros((reference.num_frames, num_keypoints, 3), dtype=np.float32))
        finger_q_ids.extend(hand.finger_q_ids)
        if bound_hand.finger_joint_pos.shape[1] != num_fingers:
            raise ValueError(
                f"{hand.side} binding has {bound_hand.finger_joint_pos.shape[1]} finger joints; "
                f"layout has {num_fingers}"
            )
        if include_fingers:
            finger_targets.append(bound_hand.finger_joint_pos)
        else:
            finger_targets.append(np.zeros((reference.num_frames, num_fingers), dtype=np.float32))

    return (
        np.stack(keypoint_targets, axis=1).reshape(reference.num_frames, len(hands) * num_keypoints, 3),
        tuple(keypoint_body_ids),
        np.asarray(keypoint_local_pos, dtype=np.float32),
        np.stack(finger_targets, axis=1).reshape(reference.num_frames, len(hands) * num_fingers),
        tuple(finger_q_ids),
    )


@wp.kernel
def clear_tracking_objectives(
    num_hands: int,
    object_keypoints: wp.array(dtype=wp.float32),
    hand_keypoints: wp.array(dtype=wp.float32),
    hand_joint_pos: wp.array(dtype=wp.float32),
    contact_wrench_support: wp.array(dtype=wp.float32),
    missed_contact: wp.array(dtype=wp.float32),
    unintended_contact: wp.array(dtype=wp.float32),
    force_closure: wp.array(dtype=wp.float32),
    hand_keypoint_sum: wp.array(dtype=wp.float32),
    hand_joint_value: wp.array(dtype=wp.float32),
) -> None:
    """Clear per-world scalar and per-hand intermediate objective buffers."""
    world = wp.tid()
    object_keypoints[world] = 0.0
    hand_keypoints[world] = 0.0
    hand_joint_pos[world] = 0.0
    contact_wrench_support[world] = 0.0
    missed_contact[world] = 0.0
    unintended_contact[world] = 0.0
    force_closure[world] = 0.0
    for hand in range(num_hands):
        hand_keypoint_sum[world * num_hands + hand] = 0.0
        hand_joint_value[world * num_hands + hand] = 0.0


@wp.kernel
def gather_objective_reference(
    timestep: wp.array(dtype=wp.int32),
    num_frames: int,
    num_hands: int,
    num_keypoints: int,
    num_fingers: int,
    num_bodies: int,
    num_basis: int,
    gather_keypoints: int,
    gather_fingers: int,
    gather_support: int,
    ref_hand_keypoints_w: wp.array(dtype=wp.vec3),
    ref_finger_joint_pos: wp.array(dtype=wp.float32),
    ref_wrench_support: wp.array(dtype=wp.float32),
    hand_keypoint_target_w: wp.array(dtype=wp.vec3),
    finger_joint_target: wp.array(dtype=wp.float32),
    wrench_support_target: wp.array(dtype=wp.float32),
) -> None:
    """Gather each world's current objective reference row."""
    world = wp.tid()
    frame = timestep[world]
    if frame < 0:
        frame = 0
    if frame >= num_frames:
        frame = num_frames - 1

    if gather_keypoints != 0:
        count = num_hands * num_keypoints
        for index in range(count):
            hand_keypoint_target_w[world * count + index] = ref_hand_keypoints_w[frame * count + index]
    if gather_fingers != 0:
        count = num_hands * num_fingers
        for index in range(count):
            finger_joint_target[world * count + index] = ref_finger_joint_pos[frame * count + index]
    if gather_support != 0:
        count = num_hands * num_bodies * num_basis
        for index in range(count):
            wrench_support_target[world * count + index] = ref_wrench_support[frame * count + index]


@wp.kernel
def normalize_tracking_objectives(
    num_hands: int,
    num_keypoints: int,
    num_bodies: int,
    normalize_object: int,
    normalize_hand_keypoints: int,
    normalize_hand_joints: int,
    hand_keypoint_sum: wp.array(dtype=wp.float32),
    hand_joint_value: wp.array(dtype=wp.float32),
    object_keypoints: wp.array(dtype=wp.float32),
    hand_keypoints: wp.array(dtype=wp.float32),
    hand_joint_pos: wp.array(dtype=wp.float32),
) -> None:
    """Reduce atomic/object and per-hand values to one scalar per world."""
    world = wp.tid()
    if normalize_object != 0:
        object_keypoints[world] /= float(num_bodies * 6)
    if normalize_hand_keypoints != 0:
        total = float(0.0)
        for hand in range(num_hands):
            total += hand_keypoint_sum[world * num_hands + hand] / float(num_keypoints)
        hand_keypoints[world] = total / float(num_hands)
    if normalize_hand_joints != 0:
        total = float(0.0)
        for hand in range(num_hands):
            total += hand_joint_value[world * num_hands + hand]
        hand_joint_pos[world] = total / float(num_hands)


@dataclass
class TrackingObjective:
    """Configurable reference tracking, contact support, and regularization objective."""

    config: ObjectiveConfig
    buffers: ObjectiveBuffers
    enabled: tuple[bool, ...]
    world_count: int
    num_frames: int
    num_hands: int
    num_keypoints: int
    num_fingers: int
    num_bodies: int
    num_basis: int
    num_edges: int
    num_joint_q: int
    bodies_per_world: int
    command: CommandBuffers
    contact: ContactTracker | None
    action_rate_l2: wp.array
    action_l2: wp.array
    keypoint_body_ids: wp.array
    keypoint_local_pos: wp.array
    finger_q_ids: wp.array
    ref_hand_keypoints_w: wp.array
    ref_finger_joint_pos: wp.array
    ref_wrench_support: wp.array
    hand_keypoint_target_w: wp.array
    finger_joint_target: wp.array
    wrench_support_target: wp.array
    hand_keypoint_sum: wp.array
    hand_joint_value: wp.array
    object_keypoint_vecs: wp.array
    wrench_basis: wp.array
    cone_cos: wp.array
    cone_sin: wp.array
    object_radius: wp.array
    wrench_contact_enabled: wp.array
    excluded_wrench_contact_enabled: wp.array
    current_wrench_support: wp.array
    unintended_wrench_support: wp.array
    contact_force_history_cursor: wp.array
    contact_force_squared_history: wp.array

    @classmethod
    def build(
        cls,
        model,
        embodiment: EmbodimentLayout,
        robot_reference: RobotReferenceBinding,
        reference: Reference,
        command: CommandBuffers,
        contact: ContactTracker | None,
        action_rate_l2: wp.array,
        action_l2: wp.array,
        world_count: int,
        frame_dt: float,
        config: ObjectiveConfig | None = None,
        device=None,
        reference_joint_q_offset: wp.array | None = None,
        episode_start_frame: wp.array | None = None,
    ) -> "TrackingObjective":
        """Build fixed reference, mapping, term, and support buffers."""
        config = config or ObjectiveConfig()
        enabled = config.enabled
        sides = config.sides
        if set(sides) != set(embodiment.sides) or len(sides) != len(embodiment.sides):
            raise ValueError(f"objective sides {sides} must match embodiment sides {embodiment.sides}")
        if contact is not None and contact.layout.sides != sides:
            raise ValueError(f"objective sides {sides} do not match contact sides {contact.layout.sides}")
        if robot_reference.num_frames != reference.num_frames:
            raise ValueError(
                f"robot binding has {robot_reference.num_frames} frames but task reference has {reference.num_frames}"
            )

        hand_keypoints, keypoint_body_ids, keypoint_local_pos, finger_targets, finger_q_ids = _hand_reference(
            embodiment,
            robot_reference,
            sides,
            keypoint_frame=config.hand_keypoint_frame,
            include_keypoints=enabled[_HAND_KEYPOINTS],
            include_fingers=enabled[_HAND_JOINT_POS],
        )
        num_hands = len(sides)
        num_keypoints = hand_keypoints.shape[1] // num_hands
        num_fingers = finger_targets.shape[1] // num_hands
        num_bodies = command.layout.num_bodies
        num_basis = config.num_wrench_basis
        num_edges = config.num_friction_cone_edges
        wrench_terms_enabled = any(
            enabled[index] for index in (_CONTACT_WRENCH_SUPPORT, _MISSED_CONTACT, _UNINTENDED_CONTACT, _FORCE_CLOSURE)
        )
        contact_enabled = wrench_terms_enabled or enabled[_CONTACT_FORCE_L2]
        if contact_enabled and contact is None:
            raise ValueError("contact objective terms require a ContactTracker")

        reference_support = np.zeros(
            (reference.num_frames, num_hands, num_bodies, num_basis),
            dtype=np.float32,
        )
        if wrench_terms_enabled:
            reference_support = compute_reference_supports(
                reference,
                num_basis=num_basis,
                num_edges=num_edges,
                mu=config.friction_coefficient,
                seed=config.wrench_basis_seed,
                sides=sides,
                device=device,
            )
        basis = np.concatenate(
            [sample_wrench_basis(num_basis, 1.0, config.wrench_basis_seed + body) for body in range(num_bodies)],
            axis=0,
        )
        cone_cos, cone_sin = friction_cone_angles(num_edges)
        i32 = lambda values: wp.array(values, dtype=wp.int32, device=device)  # noqa: E731
        wrench_contact_enabled: tuple[int, ...] = ()
        excluded_wrench_contact_enabled: tuple[int, ...] = ()
        if contact is not None and wrench_terms_enabled:
            excluded = set(config.contact_wrench_support.excluded_sim_link_names)
            mask: list[int] = []
            for side in sides:
                link_names = tuple(link.link_name for link in embodiment.hand(side).contact_links)
                missing = excluded - set(link_names)
                if missing:
                    raise ValueError(f"{side} hand does not have excluded sim contact links {tuple(sorted(missing))}")
                hand_mask = tuple(int(name not in excluded) for name in link_names)
                if not any(hand_mask):
                    raise ValueError(f"excluded sim contact links remove every {side} hand contact link")
                for _ in range(num_bodies):
                    mask.extend(hand_mask)
            wrench_contact_enabled = tuple(mask)
            if len(wrench_contact_enabled) != contact.layout.slots_per_world:
                raise ValueError(
                    f"wrench contact mask has {len(wrench_contact_enabled)} slots; "
                    f"tracker has {contact.layout.slots_per_world}"
                )
            excluded_wrench_contact_enabled = tuple(1 - enabled for enabled in wrench_contact_enabled)
        support_size = world_count * num_hands * num_bodies * num_basis
        current_wrench_support = wp.zeros(support_size, dtype=wp.float32, device=device)
        unintended_wrench_support = (
            wp.zeros(support_size, dtype=wp.float32, device=device)
            if any(excluded_wrench_contact_enabled) and enabled[_UNINTENDED_CONTACT]
            else current_wrench_support
        )
        contact_slots = contact.layout.slots_per_world if contact is not None else 1
        force_history_size = (
            world_count * contact_slots * config.contact_force_l2.history_length
            if enabled[_CONTACT_FORCE_L2] and config.contact_force_l2.mode == "per_hand_log"
            else 0
        )
        return cls(
            config=config,
            buffers=ObjectiveBuffers.build(world_count, frame_dt, config, device=device),
            enabled=enabled,
            world_count=world_count,
            num_frames=reference.num_frames,
            num_hands=num_hands,
            num_keypoints=num_keypoints,
            num_fingers=num_fingers,
            num_bodies=num_bodies,
            num_basis=num_basis,
            num_edges=num_edges,
            num_joint_q=model.joint_coord_count // world_count,
            bodies_per_world=model.body_count // world_count,
            command=command,
            contact=contact,
            action_rate_l2=action_rate_l2,
            action_l2=action_l2,
            keypoint_body_ids=i32(keypoint_body_ids),
            keypoint_local_pos=wp.array(keypoint_local_pos, dtype=wp.vec3, device=device),
            finger_q_ids=i32(finger_q_ids),
            ref_hand_keypoints_w=wp.array(hand_keypoints.reshape(-1, 3), dtype=wp.vec3, device=device),
            ref_finger_joint_pos=wp.array(finger_targets.reshape(-1), dtype=wp.float32, device=device),
            ref_wrench_support=wp.array(reference_support.reshape(-1), dtype=wp.float32, device=device),
            hand_keypoint_target_w=wp.zeros(
                world_count * num_hands * num_keypoints,
                dtype=wp.vec3,
                device=device,
            ),
            finger_joint_target=wp.zeros(
                world_count * num_hands * num_fingers,
                dtype=wp.float32,
                device=device,
            ),
            wrench_support_target=wp.zeros(
                world_count * num_hands * num_bodies * num_basis,
                dtype=wp.float32,
                device=device,
            ),
            hand_keypoint_sum=wp.zeros(world_count * num_hands, dtype=wp.float32, device=device),
            hand_joint_value=wp.zeros(world_count * num_hands, dtype=wp.float32, device=device),
            object_keypoint_vecs=wp.array(KEYPOINT_VECS_NP, dtype=wp.vec3, device=device),
            wrench_basis=wp.array(basis, dtype=wp.spatial_vector, device=device),
            cone_cos=wp.array(cone_cos, dtype=wp.float32, device=device),
            cone_sin=wp.array(cone_sin, dtype=wp.float32, device=device),
            object_radius=command.object_radius,
            wrench_contact_enabled=i32(wrench_contact_enabled),
            excluded_wrench_contact_enabled=i32(excluded_wrench_contact_enabled),
            current_wrench_support=current_wrench_support,
            unintended_wrench_support=unintended_wrench_support,
            contact_force_history_cursor=wp.zeros(1, dtype=wp.int32, device=device),
            contact_force_squared_history=wp.zeros(
                force_history_size,
                dtype=wp.float32,
                device=device,
            ),
        )

    @property
    def score(self) -> wp.array:
        return self.buffers.score

    @property
    def terms(self) -> wp.array:
        return self.buffers.terms

    @property
    def term_names(self) -> tuple[str, ...]:
        return OBJECTIVE_TERM_NAMES

    @property
    def weights(self) -> wp.array:
        return self.buffers.weights

    def set_weights(self, weights: Mapping[str, float]) -> None:
        """Validate and upload one complete set of named objective weights."""
        expected = set(self.term_names)
        actual = set(weights)
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise ValueError(f"objective weights mismatch: missing={missing}, unexpected={unexpected}")
        values = tuple(float(weights[name]) for name in self.term_names)
        invalid = {name: value for name, value in zip(self.term_names, values, strict=True) if not math.isfinite(value)}
        if invalid:
            raise ValueError(f"objective weights must be finite, got {invalid}")
        disabled_nonzero = [
            name
            for name, value, enabled in zip(self.term_names, values, self.enabled, strict=True)
            if not enabled and value != 0.0
        ]
        if disabled_nonzero:
            raise ValueError(f"disabled objective terms must keep zero weight, got {disabled_nonzero}")
        self.buffers.weights.assign(values)

    def reset(self, reset_mask: wp.array) -> None:
        """Clear stateful force-reward history for selected worlds."""
        if (
            not self.enabled[_CONTACT_FORCE_L2]
            or self.config.contact_force_l2.mode != "per_hand_log"
            or self.contact is None
        ):
            return
        wp.launch(
            reset_contact_force_history,
            dim=self.world_count,
            inputs=[
                reset_mask,
                self.contact.layout.slots_per_world,
                self.world_count,
                self.config.contact_force_l2.history_length,
            ],
            outputs=[self.contact_force_squared_history],
        )

    def evaluate(self, state, timestep: wp.array, terminated: wp.array) -> None:
        """Evaluate enabled terms and compose the per-world score."""
        enabled = self.enabled
        buffers = self.buffers
        wrench_contact_enabled = any(
            enabled[index] for index in (_CONTACT_WRENCH_SUPPORT, _MISSED_CONTACT, _UNINTENDED_CONTACT, _FORCE_CLOSURE)
        )
        tracking_enabled = any(
            enabled[index]
            for index in (
                _OBJECT_KEYPOINTS,
                _HAND_KEYPOINTS,
                _HAND_JOINT_POS,
                _CONTACT_WRENCH_SUPPORT,
                _MISSED_CONTACT,
                _UNINTENDED_CONTACT,
                _FORCE_CLOSURE,
            )
        )
        if tracking_enabled:
            wp.launch(
                clear_tracking_objectives,
                dim=self.world_count,
                inputs=[self.num_hands],
                outputs=[
                    buffers.object_keypoints,
                    buffers.hand_keypoints,
                    buffers.hand_joint_pos,
                    buffers.contact_wrench_support,
                    buffers.missed_contact,
                    buffers.unintended_contact,
                    buffers.force_closure,
                    self.hand_keypoint_sum,
                    self.hand_joint_value,
                ],
            )
        reference_gather_enabled = enabled[_HAND_KEYPOINTS] or enabled[_HAND_JOINT_POS] or wrench_contact_enabled
        if reference_gather_enabled:
            wp.launch(
                gather_objective_reference,
                dim=self.world_count,
                inputs=[
                    timestep,
                    self.num_frames,
                    self.num_hands,
                    self.num_keypoints,
                    self.num_fingers,
                    self.num_bodies,
                    self.num_basis,
                    int(enabled[_HAND_KEYPOINTS]),
                    int(enabled[_HAND_JOINT_POS]),
                    int(wrench_contact_enabled),
                    self.ref_hand_keypoints_w,
                    self.ref_finger_joint_pos,
                    self.ref_wrench_support,
                ],
                outputs=[
                    self.hand_keypoint_target_w,
                    self.finger_joint_target,
                    self.wrench_support_target,
                ],
            )
        if enabled[_OBJECT_KEYPOINTS]:
            wp.launch(
                object_keypoints_objective,
                dim=self.world_count * self.num_bodies * 6,
                inputs=[
                    state.body_q,
                    self.command.body_ids,
                    self.command.body_target_pos_w,
                    self.command.body_target_quat_w,
                    self.object_keypoint_vecs,
                    self.num_bodies,
                    self.bodies_per_world,
                    self.config.object_keypoints.var,
                    int(self.config.object_keypoints.shape),
                ],
                outputs=[buffers.object_keypoints],
            )
        if enabled[_HAND_KEYPOINTS]:
            wp.launch(
                hand_keypoints_objective,
                dim=self.world_count * self.num_hands * self.num_keypoints,
                inputs=[
                    state.body_q,
                    self.keypoint_body_ids,
                    self.keypoint_local_pos,
                    self.hand_keypoint_target_w,
                    self.num_keypoints,
                    self.num_hands,
                    self.bodies_per_world,
                    self.config.hand_keypoints.var,
                    int(self.config.hand_keypoints.shape),
                    0.0,
                ],
                outputs=[self.hand_keypoint_sum],
            )
        if enabled[_HAND_JOINT_POS]:
            wp.launch(
                hand_joint_objective,
                dim=self.world_count * self.num_hands,
                inputs=[
                    state.joint_q,
                    self.finger_q_ids,
                    self.finger_joint_target,
                    self.num_fingers,
                    self.num_hands,
                    self.num_joint_q,
                    self.config.hand_joint_pos.var,
                    int(self.config.hand_joint_pos.shape),
                    0.0,
                ],
                outputs=[self.hand_joint_value],
            )
        if wrench_contact_enabled:
            contact = self.contact
            assert contact is not None
            wp.launch(
                batched_wrench_support,
                dim=self.world_count * self.num_hands * self.num_bodies * self.num_basis,
                inputs=[
                    contact.contact_pos_o,
                    contact.contact_force_direction_o,
                    self.wrench_contact_enabled,
                    contact.hand_slot_starts,
                    contact.link_counts,
                    contact.layout.slots_per_world,
                    self.num_hands,
                    self.num_bodies,
                    self.wrench_basis,
                    self.cone_cos,
                    self.cone_sin,
                    self.config.friction_coefficient,
                    self.object_radius,
                    self.num_edges,
                    self.num_basis,
                ],
                outputs=[self.current_wrench_support],
            )
            if self.unintended_wrench_support is not self.current_wrench_support:
                wp.launch(
                    batched_wrench_support,
                    dim=self.world_count * self.num_hands * self.num_bodies * self.num_basis,
                    inputs=[
                        contact.contact_pos_o,
                        contact.contact_force_direction_o,
                        self.excluded_wrench_contact_enabled,
                        contact.hand_slot_starts,
                        contact.link_counts,
                        contact.layout.slots_per_world,
                        self.num_hands,
                        self.num_bodies,
                        self.wrench_basis,
                        self.cone_cos,
                        self.cone_sin,
                        self.config.friction_coefficient,
                        self.object_radius,
                        self.num_edges,
                        self.num_basis,
                    ],
                    outputs=[self.unintended_wrench_support],
                )
                wp.launch(
                    merge_wrench_support,
                    dim=self.world_count * self.num_hands * self.num_bodies * self.num_basis,
                    inputs=[self.current_wrench_support],
                    outputs=[self.unintended_wrench_support],
                )
        if enabled[_CONTACT_FORCE_L2]:
            contact = self.contact
            assert contact is not None
            if self.config.contact_force_l2.mode == "per_hand_log":
                wp.launch(
                    contact_force_per_hand_log,
                    dim=self.world_count,
                    inputs=[
                        contact.contact_force_w,
                        contact.hand_slot_starts,
                        contact.link_counts,
                        contact.layout.num_hands,
                        contact.layout.num_objects,
                        contact.layout.slots_per_world,
                        self.world_count,
                        self.config.contact_force_l2.history_length,
                        self.config.contact_force_l2.log_force_floor**2,
                        math.log(self.config.contact_force_l2.log_force_reference),
                        1.0
                        / (
                            math.log(self.config.contact_force_l2.log_force_reference)
                            - math.log(self.config.contact_force_l2.log_force_floor)
                        ),
                        self.contact_force_history_cursor,
                    ],
                    outputs=[
                        self.contact_force_squared_history,
                        buffers.contact_force_l2,
                    ],
                )
                wp.launch(
                    advance_contact_force_history,
                    dim=1,
                    inputs=[
                        self.contact_force_history_cursor,
                        self.config.contact_force_l2.history_length,
                    ],
                )
            else:
                wp.launch(
                    contact_force_l2,
                    dim=self.world_count,
                    inputs=[
                        contact.contact_force_w,
                        contact.layout.slots_per_world,
                        self.config.contact_force_l2.threshold,
                    ],
                    outputs=[buffers.contact_force_l2],
                )
        if enabled[_CONTACT_WRENCH_SUPPORT]:
            wp.launch(
                contact_wrench_support_objective,
                dim=self.world_count,
                inputs=[
                    self.wrench_support_target,
                    self.current_wrench_support,
                    self.num_hands,
                    self.num_bodies,
                    self.num_basis,
                    self.config.contact_wrench_support.tolerance,
                    self.config.contact_wrench_support.var,
                ],
                outputs=[buffers.contact_wrench_support],
            )
        if enabled[_MISSED_CONTACT]:
            wp.launch(
                missed_contact_penalty,
                dim=self.world_count,
                inputs=[
                    self.wrench_support_target,
                    self.current_wrench_support,
                    self.num_hands,
                    self.num_bodies,
                    self.num_basis,
                ],
                outputs=[buffers.missed_contact],
            )
        if enabled[_UNINTENDED_CONTACT]:
            wp.launch(
                unintended_contact_penalty,
                dim=self.world_count,
                inputs=[
                    self.wrench_support_target,
                    self.unintended_wrench_support,
                    self.num_hands,
                    self.num_bodies,
                    self.num_basis,
                ],
                outputs=[buffers.unintended_contact],
            )
        if enabled[_FORCE_CLOSURE]:
            wp.launch(
                force_closure_objective,
                dim=self.world_count,
                inputs=[
                    self.wrench_support_target,
                    self.current_wrench_support,
                    self.num_hands,
                    self.num_bodies,
                    self.num_basis,
                    self.config.force_closure.min_support,
                ],
                outputs=[buffers.force_closure],
            )
        if enabled[_OBJECT_KEYPOINTS] or enabled[_HAND_KEYPOINTS] or enabled[_HAND_JOINT_POS]:
            wp.launch(
                normalize_tracking_objectives,
                dim=self.world_count,
                inputs=[
                    self.num_hands,
                    self.num_keypoints,
                    self.num_bodies,
                    int(enabled[_OBJECT_KEYPOINTS]),
                    int(enabled[_HAND_KEYPOINTS]),
                    int(enabled[_HAND_JOINT_POS]),
                    self.hand_keypoint_sum,
                    self.hand_joint_value,
                ],
                outputs=[
                    buffers.object_keypoints,
                    buffers.hand_keypoints,
                    buffers.hand_joint_pos,
                ],
            )
        buffers.compose(terminated, self.action_rate_l2, self.action_l2)
