# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named objective-term configuration and shared wrench-support parameters."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import warp as wp

from flash_chord.embodiments.binding import HandKeypointFrame
from flash_chord.objectives.shaping import ObjectiveShape

if TYPE_CHECKING:
    from flash_chord.data.reference import Reference
    from flash_chord.embodiments.base import EmbodimentLayout
    from flash_chord.embodiments.binding import RobotReferenceBinding
    from flash_chord.objectives.composition import Objective
    from flash_chord.runtime.actions import Action
    from flash_chord.runtime.command import CommandBuffers
    from flash_chord.runtime.contact import ContactTracker

OBJECTIVE_TERM_NAMES = (
    "object_keypoints",
    "hand_keypoints",
    "hand_joint_pos",
    "contact_wrench_support",
    "missed_contact",
    "unintended_contact",
    "termination",
    "action_rate_l2",
    "action_l2",
    "contact_force_l2",
    "force_closure",
)


@dataclass(frozen=True)
class ObjectiveTermConfig:
    """Inclusion and scalar weight for one objective term."""

    weight: float
    enabled: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.weight):
            raise ValueError(f"objective term weight must be finite, got {self.weight}")


@dataclass(frozen=True)
class ContactForceObjectiveTermConfig(ObjectiveTermConfig):
    """MuJoCo-Warp contact-force objective."""

    threshold: float = 0.0
    mode: str = "l2"
    history_length: int = 3
    log_force_floor: float = 1.0
    log_force_reference: float = 1000.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not math.isfinite(self.threshold) or self.threshold < 0.0:
            raise ValueError(f"contact-force threshold must be finite and non-negative, got {self.threshold}")
        if self.mode not in ("l2", "per_hand_log"):
            raise ValueError(f"contact-force mode must be 'l2' or 'per_hand_log', got {self.mode!r}")
        if self.history_length <= 0:
            raise ValueError(f"contact-force history length must be positive, got {self.history_length}")
        if not math.isfinite(self.log_force_floor) or self.log_force_floor <= 0.0:
            raise ValueError("contact-force log floor must be finite and positive, got " f"{self.log_force_floor}")
        if (
            not math.isfinite(self.log_force_reference)
            or self.log_force_reference <= self.log_force_floor
            or self.log_force_reference >= 1.0e15
        ):
            raise ValueError(
                "contact-force log reference must be finite and greater than "
                "the floor and below 1e15 N, got "
                f"floor={self.log_force_floor}, "
                f"reference={self.log_force_reference}"
            )
        log_span = math.log(self.log_force_reference) - math.log(self.log_force_floor)
        if log_span <= 1.0e-6:
            raise ValueError(
                "contact-force log reference/floor ratio is too close to one "
                f"for stable normalization, got log span {log_span}"
            )


@dataclass(frozen=True)
class ShapedObjectiveTermConfig(ObjectiveTermConfig):
    """Weighted tracking term with an exponential shaping variance."""

    var: float = 0.1
    shape: ObjectiveShape = ObjectiveShape.LAPLACIAN

    def __post_init__(self) -> None:
        super().__post_init__()
        shape = ObjectiveShape[self.shape.upper()] if isinstance(self.shape, str) else ObjectiveShape(self.shape)
        object.__setattr__(self, "shape", shape)
        if self.var <= 0.0:
            raise ValueError(f"objective term var must be positive, got {self.var}")


@dataclass(frozen=True)
class ContactSupportObjectiveTermConfig(ObjectiveTermConfig):
    """Weighted contact-wrench support term with target tolerance and variance."""

    tolerance: float = 0.1
    var: float = 0.1
    excluded_sim_link_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "excluded_sim_link_names", tuple(self.excluded_sim_link_names))
        if self.tolerance < 0.0:
            raise ValueError(f"support tolerance must be non-negative, got {self.tolerance}")
        if self.var <= 0.0:
            raise ValueError(f"support var must be positive, got {self.var}")
        if len(set(self.excluded_sim_link_names)) != len(self.excluded_sim_link_names) or any(
            not isinstance(name, str) or not name for name in self.excluded_sim_link_names
        ):
            raise ValueError(
                "excluded sim contact links must be unique nonempty names, got " f"{self.excluded_sim_link_names}"
            )


@dataclass(frozen=True)
class ForceClosureObjectiveTermConfig(ObjectiveTermConfig):
    """Weighted force-closure term with a minimum per-direction support."""

    min_support: float = 0.01

    def __post_init__(self) -> None:
        super().__post_init__()
        if not math.isfinite(self.min_support) or self.min_support < 0.0:
            raise ValueError(
                "force-closure minimum support must be finite and non-negative, " f"got {self.min_support}"
            )


@dataclass(frozen=True)
class ObjectiveConfig:
    # Weight zero keeps a term captured for curriculum changes; enabled false removes its work.
    object_keypoints: ShapedObjectiveTermConfig = field(
        default_factory=lambda: ShapedObjectiveTermConfig(weight=0.0, var=0.1)
    )
    hand_keypoints: ShapedObjectiveTermConfig = field(
        default_factory=lambda: ShapedObjectiveTermConfig(weight=0.0, var=0.1)
    )
    hand_joint_pos: ShapedObjectiveTermConfig = field(
        default_factory=lambda: ShapedObjectiveTermConfig(weight=0.0, var=1.0)
    )
    contact_wrench_support: ContactSupportObjectiveTermConfig = field(
        default_factory=lambda: ContactSupportObjectiveTermConfig(weight=10.0)
    )
    missed_contact: ObjectiveTermConfig = field(default_factory=lambda: ObjectiveTermConfig(weight=-1.0))
    unintended_contact: ObjectiveTermConfig = field(default_factory=lambda: ObjectiveTermConfig(weight=-10.0))
    termination: ObjectiveTermConfig = field(default_factory=lambda: ObjectiveTermConfig(weight=-100.0))
    action_rate_l2: ObjectiveTermConfig = field(default_factory=lambda: ObjectiveTermConfig(weight=-5.0e-3))
    action_l2: ObjectiveTermConfig = field(default_factory=lambda: ObjectiveTermConfig(weight=-2.0e-3))
    contact_force_l2: ContactForceObjectiveTermConfig = field(
        default_factory=lambda: ContactForceObjectiveTermConfig(weight=0.0, enabled=False)
    )
    force_closure: ForceClosureObjectiveTermConfig = field(
        default_factory=lambda: ForceClosureObjectiveTermConfig(weight=0.0, enabled=False)
    )
    hand_keypoint_frame: HandKeypointFrame = "dp"

    # wrench-space support function
    num_wrench_basis: int = 512
    num_friction_cone_edges: int = 8
    friction_coefficient: float = 0.1
    wrench_basis_seed: int = 0
    sides: tuple[str, ...] = ("right", "left")

    def __post_init__(self) -> None:
        if self.num_wrench_basis <= 0:
            raise ValueError(f"num_wrench_basis must be positive, got {self.num_wrench_basis}")
        if self.num_friction_cone_edges <= 0:
            raise ValueError(f"num_friction_cone_edges must be positive, got {self.num_friction_cone_edges}")
        if self.friction_coefficient < 0.0:
            raise ValueError(f"friction_coefficient must be non-negative, got {self.friction_coefficient}")
        if self.hand_keypoint_frame not in ("dp", "fingertip"):
            raise ValueError(f"hand_keypoint_frame must be 'dp' or 'fingertip', got {self.hand_keypoint_frame!r}")

    @property
    def weights(self) -> tuple[float, ...]:
        """Term weights in :data:`OBJECTIVE_TERM_NAMES` order; disabled terms map to zero."""
        return tuple(term.weight if term.enabled else 0.0 for term in self.terms)

    @property
    def enabled(self) -> tuple[bool, ...]:
        """Whether each term is included in the captured evaluator."""
        return tuple(term.enabled for term in self.terms)

    @property
    def terms(self) -> tuple[ObjectiveTermConfig, ...]:
        """Term configurations in the stable diagnostic and device-buffer order."""
        return tuple(getattr(self, name) for name in OBJECTIVE_TERM_NAMES)

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
        """Build the tracking objective with explicit action capabilities."""
        from flash_chord.objectives.tracking import TrackingObjective
        from flash_chord.runtime.actions import RegularizedAction

        if isinstance(action, RegularizedAction):
            action_rate_l2 = action.action_rate_l2
            action_l2 = action.action_l2
        elif not self.action_rate_l2.enabled and not self.action_l2.enabled:
            action_rate_l2 = wp.zeros(world_count, dtype=wp.float32, device=device)
            action_l2 = wp.zeros(world_count, dtype=wp.float32, device=device)
        else:
            raise TypeError(
                "the tracking objective requires a RegularizedAction when action terms are enabled; "
                "disable those terms or select a compatible objective"
            )
        return TrackingObjective.build(
            model,
            embodiment,
            robot_reference,
            reference,
            command,
            contact,
            action_rate_l2=action_rate_l2,
            action_l2=action_l2,
            world_count=world_count,
            frame_dt=frame_dt,
            config=self,
            device=device,
            reference_joint_q_offset=reference_joint_q_offset,
            episode_start_frame=episode_start_frame,
        )
