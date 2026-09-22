# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device objective-term storage and weighted composition."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import warp as wp

from flash_chord.objectives.config import OBJECTIVE_TERM_NAMES, ObjectiveConfig

if TYPE_CHECKING:
    from flash_chord.data.reference import Reference
    from flash_chord.embodiments.base import EmbodimentLayout
    from flash_chord.embodiments.binding import RobotReferenceBinding
    from flash_chord.runtime.actions import Action
    from flash_chord.runtime.command import CommandBuffers
    from flash_chord.runtime.contact import ContactTracker

_OBJECTIVE_TERM_COUNT = wp.constant(len(OBJECTIVE_TERM_NAMES))


@runtime_checkable
class Objective(Protocol):
    """Device objective strategy consumed by :class:`BaseEnv`."""

    score: wp.array
    terms: wp.array
    weights: wp.array

    def evaluate(self, state, timestep: wp.array, terminated: wp.array) -> None:
        """Write per-world terms and their weighted score."""
        ...

    def reset(self, reset_mask: wp.array) -> None:
        """Clear objective state for selected worlds."""
        ...


@runtime_checkable
class ObjectiveSpec(Protocol):
    """Setup-time objective builder selected by configuration."""

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
    ) -> Objective:
        """Bind runtime dependencies and return an objective strategy."""
        ...


@runtime_checkable
class ObjectiveDiagnostics(Protocol):
    """Optional named term buffer consumed by training diagnostics."""

    term_names: tuple[str, ...]
    terms: wp.array


@runtime_checkable
class CurriculumObjective(Protocol):
    """Optional objective capability for applying named curriculum weights."""

    term_names: tuple[str, ...]

    def set_weights(self, weights: Mapping[str, float]) -> None:
        """Validate and upload one complete named weight mapping."""
        ...


@wp.kernel
def compose_objective(
    object_keypoints: wp.array(dtype=wp.float32),
    hand_keypoints: wp.array(dtype=wp.float32),
    hand_joint_pos: wp.array(dtype=wp.float32),
    contact_wrench_support: wp.array(dtype=wp.float32),
    missed_contact: wp.array(dtype=wp.float32),
    unintended_contact: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.int32),
    action_rate_l2: wp.array(dtype=wp.float32),
    action_l2: wp.array(dtype=wp.float32),
    contact_force_l2: wp.array(dtype=wp.float32),
    force_closure: wp.array(dtype=wp.float32),
    weights: wp.array(dtype=wp.float32),
    frame_dt: float,
    terms: wp.array(dtype=wp.float32),
    score: wp.array(dtype=wp.float32),
) -> None:
    """Publish named terms and their frame-time-scaled weighted sum."""
    world = wp.tid()
    base = world * _OBJECTIVE_TERM_COUNT
    terms[base] = object_keypoints[world]
    terms[base + 1] = hand_keypoints[world]
    terms[base + 2] = hand_joint_pos[world]
    terms[base + 3] = contact_wrench_support[world]
    terms[base + 4] = missed_contact[world]
    terms[base + 5] = unintended_contact[world]
    terms[base + 6] = float(terminated[world])
    terms[base + 7] = action_rate_l2[world]
    terms[base + 8] = action_l2[world]
    terms[base + 9] = contact_force_l2[world]
    terms[base + 10] = force_closure[world]

    total = float(0.0)
    for term in range(_OBJECTIVE_TERM_COUNT):
        total += weights[term] * terms[base + term]
    score[world] = frame_dt * total


@dataclass
class ObjectiveBuffers:
    """Persistent per-world objective inputs, terms, weights, and score."""

    world_count: int
    frame_dt: float
    object_keypoints: wp.array
    hand_keypoints: wp.array
    hand_joint_pos: wp.array
    contact_wrench_support: wp.array
    missed_contact: wp.array
    unintended_contact: wp.array
    contact_force_l2: wp.array
    force_closure: wp.array
    weights: wp.array
    terms: wp.array
    score: wp.array

    @classmethod
    def build(
        cls,
        world_count: int,
        frame_dt: float,
        config: ObjectiveConfig,
        device=None,
    ) -> "ObjectiveBuffers":
        """Allocate fixed term storage and upload initial weights."""
        scalar = lambda: wp.zeros(world_count, dtype=wp.float32, device=device)  # noqa: E731
        return cls(
            world_count=world_count,
            frame_dt=frame_dt,
            object_keypoints=scalar(),
            hand_keypoints=scalar(),
            hand_joint_pos=scalar(),
            contact_wrench_support=scalar(),
            missed_contact=scalar(),
            unintended_contact=scalar(),
            contact_force_l2=scalar(),
            force_closure=scalar(),
            weights=wp.array(config.weights, dtype=wp.float32, device=device),
            terms=wp.zeros(
                world_count * len(OBJECTIVE_TERM_NAMES),
                dtype=wp.float32,
                device=device,
            ),
            score=scalar(),
        )

    def compose(
        self,
        terminated: wp.array,
        action_rate_l2: wp.array,
        action_l2: wp.array,
    ) -> None:
        """Compose current term buffers into ``terms`` and ``score``."""
        wp.launch(
            compose_objective,
            dim=self.world_count,
            inputs=[
                self.object_keypoints,
                self.hand_keypoints,
                self.hand_joint_pos,
                self.contact_wrench_support,
                self.missed_contact,
                self.unintended_contact,
                terminated,
                action_rate_l2,
                action_l2,
                self.contact_force_l2,
                self.force_closure,
                self.weights,
                self.frame_dt,
            ],
            outputs=[self.terms, self.score],
        )
