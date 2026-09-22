# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Embodiment interface + per-world layout (implemented by e.g. ``sharpa_hands``)."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from flash_chord.embodiments.binding import RobotReferenceBinding


@dataclass
class JointControlSpec:
    """Per-joint control parameters, resolved per joint by an embodiment config's regex overrides.

    Field meaning depends on ``mode``:

    - ``"position_velocity"`` : ``kp`` -> ``joint_target_ke``, ``kd`` -> ``joint_target_kd``,
      ``effort_limit`` -> ``joint_effort_limit``.
    - ``"effort"`` (the ball wrist-orientation joint): ``kp`` -> orientation-controller spring gain,
      ``kd`` -> ``mujoco:dof_passive_damping`` (implicit, stable), ``effort_limit`` -> torque clamp.

    ``friction=None`` leaves the value set at construction / imported from the URDF untouched.
    ``velocity_limit=None`` likewise preserves the imported velocity limit.
    """

    mode: str = "position_velocity"  # "position_velocity" | "effort"
    kp: float = 1.7
    kd: float = 0.1
    default_pos: float = 0.0
    armature: float = 0.01
    effort_limit: float = 10.0
    velocity_limit: float | None = None
    friction: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("position_velocity", "effort"):
            raise ValueError(f"unsupported joint control mode {self.mode!r}")
        for name in ("kp", "kd", "default_pos", "armature", "effort_limit"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"joint control {name} must be finite, got {value}")
        if self.kp < 0.0 or self.kd < 0.0 or self.armature < 0.0:
            raise ValueError("joint control gains and armature must be nonnegative")
        if self.effort_limit <= 0.0:
            raise ValueError(f"joint effort limit must be positive, got {self.effort_limit}")
        if self.velocity_limit is not None and (not math.isfinite(self.velocity_limit) or self.velocity_limit <= 0.0):
            raise ValueError(f"joint velocity limit must be positive and finite, got {self.velocity_limit}")
        if self.friction is not None and (not math.isfinite(self.friction) or self.friction < 0.0):
            raise ValueError(f"joint friction must be nonnegative and finite, got {self.friction}")


def resolve_joint_controls(
    labels: tuple[str, ...] | list[str],
    default: JointControlSpec,
    overrides: Mapping[str, Mapping[str, Any]],
) -> tuple[JointControlSpec, ...]:
    """Resolve one ordered override table after validating every configured pattern."""
    labels = tuple(labels)
    if (
        not labels
        or len(set(labels)) != len(labels)
        or any(not isinstance(label, str) or not label for label in labels)
    ):
        raise ValueError(f"controlled joint labels must be unique and nonempty, got {labels}")

    compiled: list[tuple[str, re.Pattern, dict[str, Any]]] = []
    for pattern, raw_values in overrides.items():
        if not isinstance(pattern, str) or not pattern:
            raise ValueError(f"joint override regex must be a nonempty string, got {pattern!r}")
        if not isinstance(raw_values, Mapping):
            raise TypeError(f"joint override {pattern!r} values must be a mapping")
        try:
            expression = re.compile(pattern)
        except re.error as error:
            raise ValueError(f"invalid joint override regex {pattern!r}: {error}") from error
        values = dict(raw_values)
        try:
            replace(default, **values)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid joint override {pattern!r}: {error}") from error
        if not any(expression.search(label) for label in labels):
            raise ValueError(f"joint override {pattern!r} does not match any controlled joint")
        compiled.append((pattern, expression, values))

    resolved: list[JointControlSpec] = []
    for label in labels:
        spec = replace(default)
        for _, expression, values in compiled:
            if expression.search(label):
                spec = replace(spec, **values)
        resolved.append(spec)
    return tuple(resolved)


@dataclass(frozen=True)
class BodyFrame:
    """A named semantic frame rigidly attached to one retained simulation body."""

    name: str
    body_id: int
    body_to_frame_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    body_to_frame_quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"semantic frame name must be a nonempty string, got {self.name!r}")
        if self.body_id < 0:
            raise ValueError(f"semantic frame body ID must be nonnegative, got {self.body_id}")
        position = np.asarray(self.body_to_frame_pos, dtype=np.float64)
        quaternion = np.asarray(self.body_to_frame_quat_xyzw, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError(f"semantic frame position must be a finite vec3, got {self.body_to_frame_pos}")
        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
            raise ValueError(
                f"semantic frame quaternion must be a finite xyzw value, got {self.body_to_frame_quat_xyzw}"
            )
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1.0e-8:
            raise ValueError("semantic frame quaternion must have nonzero norm")
        object.__setattr__(self, "body_id", int(self.body_id))
        object.__setattr__(self, "body_to_frame_pos", tuple(float(value) for value in position))
        object.__setattr__(
            self,
            "body_to_frame_quat_xyzw",
            tuple(float(value) for value in quaternion / norm),
        )


@dataclass(frozen=True)
class ScalarJointLayout:
    """Authoritative simulation IDs and names for one robot's scalar joints."""

    q_ids: tuple[int, ...]
    dof_ids: tuple[int, ...]
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "q_ids", tuple(self.q_ids))
        object.__setattr__(self, "dof_ids", tuple(self.dof_ids))
        object.__setattr__(self, "names", tuple(self.names))
        if not (len(self.q_ids) == len(self.dof_ids) == len(self.names)):
            raise ValueError("scalar joint q IDs, DOF IDs, and names must have equal lengths")
        if len(set(self.q_ids)) != len(self.q_ids) or any(index < 0 for index in self.q_ids):
            raise ValueError("scalar joint q IDs must be unique and nonnegative")
        if len(set(self.dof_ids)) != len(self.dof_ids) or any(index < 0 for index in self.dof_ids):
            raise ValueError("scalar joint DOF IDs must be unique and nonnegative")
        if len(set(self.names)) != len(self.names) or any(not isinstance(name, str) or not name for name in self.names):
            raise ValueError("scalar joint names must be unique and nonempty")


@dataclass(frozen=True)
class ScalarJointBlock:
    """One contiguous side/group block in a scalar-joint selection."""

    side: str
    group: Literal["arm", "finger"]
    start: int
    count: int

    def __post_init__(self) -> None:
        if not isinstance(self.side, str) or not self.side:
            raise ValueError(f"scalar joint block side must be a nonempty string, got {self.side!r}")
        if self.group not in ("arm", "finger"):
            raise ValueError(f"scalar joint block group must be arm or finger, got {self.group!r}")
        if self.start < 0 or self.count < 0:
            raise ValueError(f"scalar joint block start/count must be nonnegative, got {self.start}/{self.count}")

    @property
    def stop(self) -> int:
        return self.start + self.count


@dataclass(frozen=True)
class ScalarJointSelection:
    """One ordered scalar-joint view derived from an embodiment layout."""

    sides: tuple[str, ...]
    groups: tuple[Literal["arm", "finger"], ...]
    q_ids: tuple[int, ...]
    dof_ids: tuple[int, ...]
    names: tuple[str, ...]
    blocks: tuple[ScalarJointBlock, ...]

    def __post_init__(self) -> None:
        for name in ("sides", "groups", "q_ids", "dof_ids", "names", "blocks"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.sides or len(set(self.sides)) != len(self.sides):
            raise ValueError(f"selected scalar joint sides must be unique and nonempty, got {self.sides}")
        if not self.groups or len(set(self.groups)) != len(self.groups):
            raise ValueError(f"selected scalar joint groups must be unique and nonempty, got {self.groups}")
        if set(self.groups) - {"arm", "finger"}:
            raise ValueError(f"selected scalar joint groups must be arm/finger values, got {self.groups}")
        if len(self.q_ids) != len(self.dof_ids):
            raise ValueError("selected scalar joint q IDs and DOF IDs must have equal lengths")
        if self.names and len(self.names) != len(self.q_ids):
            raise ValueError("selected scalar joint names must be empty or match the selected joint count")
        _validate_unique_nonnegative(self.q_ids, "selected scalar joint q IDs")
        _validate_unique_nonnegative(self.dof_ids, "selected scalar joint DOF IDs")
        if self.names and (
            len(set(self.names)) != len(self.names) or any(not isinstance(name, str) or not name for name in self.names)
        ):
            raise ValueError("selected scalar joint names must be unique and nonempty")

        expected_pairs = tuple((side, group) for side in self.sides for group in self.groups)
        actual_pairs = tuple((block.side, block.group) for block in self.blocks)
        if actual_pairs != expected_pairs:
            raise ValueError("selected scalar joint blocks must cover every side/group pair")
        cursor = 0
        for block in self.blocks:
            if block.start != cursor:
                raise ValueError("selected scalar joint blocks must be contiguous and ordered")
            cursor = block.stop
        if cursor != len(self.q_ids):
            raise ValueError("selected scalar joint blocks must cover the selected joint count")

    @property
    def side_counts(self) -> tuple[int, ...]:
        return tuple(sum(block.count for block in self.blocks if block.side == side) for side in self.sides)

    @property
    def block_starts(self) -> tuple[int, ...]:
        return tuple(block.start for block in self.blocks)

    @property
    def block_counts(self) -> tuple[int, ...]:
        return tuple(block.count for block in self.blocks)


@dataclass(frozen=True)
class HandLinkGeometry:
    """Stable semantic ownership of one hand link's collision shapes."""

    link_name: str
    shape_ids: tuple[int, ...]
    object_contact: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape_ids", tuple(self.shape_ids))
        if not isinstance(self.link_name, str) or not self.link_name:
            raise ValueError(f"hand collision link name must be nonempty, got {self.link_name!r}")
        if not isinstance(self.object_contact, bool):
            raise TypeError(f"hand collision link object_contact must be bool, got {self.object_contact!r}")
        if not self.shape_ids:
            raise ValueError(f"hand collision link {self.link_name!r} must own at least one shape")
        _validate_unique_nonnegative(self.shape_ids, f"hand collision link {self.link_name!r} shape IDs")


@dataclass(frozen=True)
class HandLayout:
    """Where one hand's actuated DOFs and bodies live in the per-world vectors.

    Indices are per-world (the env offsets them by ``world * num_joint_dof`` etc.).
    DOF indices index ``joint_qd`` / target vectors; ``wrist_orient_qcoord`` is the
    start of the ball joint's 4-component quaternion in ``joint_q`` (coord layout).
    """

    side: str  # "left" | "right"
    palm_frame: BodyFrame
    dp_frames: tuple[BodyFrame, ...]
    fingertip_frames: tuple[BodyFrame, ...]
    # the wrist is positioned either by floating wrist joints (prismatic + ball) OR by an
    # arm chain; `wrist_actuation_dof_ids` returns whichever drives it.
    wrist_pos_dof_ids: tuple[int, ...] = ()  # floating wrist: prismatic x/y/z DOFs
    wrist_orient_dof_ids: tuple[int, ...] = ()  # floating wrist: ball-joint rotvec DOFs
    wrist_orient_q_id: int = -1  # floating wrist: ball quaternion (xyzw) start in joint_q
    arm_dof_ids: tuple[int, ...] = ()  # articulated arm: joints that position the wrist
    arm_q_ids: tuple[int, ...] = ()  # articulated arm scalar coordinates
    arm_joint_names: tuple[str, ...] = ()
    finger_dof_ids: tuple[int, ...] = ()  # finger joint DOFs
    link_geometry: tuple[HandLinkGeometry, ...] = ()  # stable semantic shape groups
    # joint_q COORD indices (for setting state; differ from DOF indices when a ball joint is present)
    wrist_pos_q_ids: tuple[int, ...] = ()  # floating wrist: prismatic coords
    finger_q_ids: tuple[int, ...] = ()  # finger coords
    finger_joint_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        tuple_fields = (
            "wrist_pos_dof_ids",
            "wrist_orient_dof_ids",
            "arm_dof_ids",
            "arm_q_ids",
            "arm_joint_names",
            "finger_dof_ids",
            "link_geometry",
            "wrist_pos_q_ids",
            "finger_q_ids",
            "finger_joint_names",
            "dp_frames",
            "fingertip_frames",
        )
        for name in tuple_fields:
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not isinstance(self.side, str) or not self.side:
            raise ValueError(f"hand side must be a nonempty string, got {self.side!r}")
        if not isinstance(self.palm_frame, BodyFrame):
            raise TypeError(f"{self.side} palm frame must be a BodyFrame")

        if not (len(self.arm_q_ids) == len(self.arm_dof_ids) == len(self.arm_joint_names)):
            raise ValueError(f"{self.side} arm q IDs, DOF IDs, and names must have equal lengths")
        if len(self.finger_q_ids) != len(self.finger_dof_ids):
            raise ValueError(f"{self.side} finger q IDs and DOF IDs must have equal lengths")
        if len(self.finger_joint_names) != len(self.finger_q_ids):
            raise ValueError(f"{self.side} finger names must match the finger joint count")

        has_arm = bool(self.arm_q_ids)
        has_floating_wrist = bool(
            self.wrist_pos_dof_ids or self.wrist_orient_dof_ids or self.wrist_pos_q_ids or self.wrist_orient_q_id >= 0
        )
        if has_arm and has_floating_wrist:
            raise ValueError(f"{self.side} hand cannot have both arm-driven and floating-wrist actuation")
        if has_floating_wrist and not (
            len(self.wrist_pos_dof_ids) == 3
            and len(self.wrist_orient_dof_ids) == 3
            and len(self.wrist_pos_q_ids) == 3
            and self.wrist_orient_q_id >= 0
        ):
            raise ValueError(
                f"{self.side} floating wrist requires three position DOFs/q IDs, "
                "three orientation DOFs, and one quaternion q start"
            )

        q_ids = self.wrist_pos_q_ids + self.arm_q_ids + self.finger_q_ids
        if self.wrist_orient_q_id >= 0:
            q_ids += tuple(range(self.wrist_orient_q_id, self.wrist_orient_q_id + 4))
        dof_ids = self.wrist_actuation_dof_ids + self.finger_dof_ids
        _validate_unique_nonnegative(q_ids, f"{self.side} hand q IDs")
        _validate_unique_nonnegative(dof_ids, f"{self.side} hand DOF IDs")
        link_names = tuple(link.link_name for link in self.link_geometry)
        if len(set(link_names)) != len(link_names):
            raise ValueError(f"{self.side} collision link names must be unique, got {link_names}")
        collision_shape_ids = tuple(shape for link in self.link_geometry for shape in link.shape_ids)
        _validate_unique_nonnegative(collision_shape_ids, f"{self.side} collision shape IDs")
        if self.link_geometry and not self.contact_links:
            raise ValueError(f"{self.side} hand geometry must include at least one object-contact link")
        joint_names = self.arm_joint_names + self.finger_joint_names
        if len(set(joint_names)) != len(joint_names) or any(not name for name in joint_names):
            raise ValueError(f"{self.side} scalar joint names must be unique and nonempty")
        semantic_frames = (self.palm_frame,) + self.dp_frames + self.fingertip_frames
        semantic_names = tuple(frame.name for frame in semantic_frames)
        if len(set(semantic_names)) != len(semantic_names):
            raise ValueError(f"{self.side} semantic frame names must be unique")
        if not self.dp_frames or len(self.dp_frames) != len(self.fingertip_frames):
            raise ValueError(f"{self.side} DP and fingertip semantic frames must be nonempty with equal lengths")

    @property
    def wrist_body_id(self) -> int:
        """Compatibility body ID derived from the authoritative semantic palm frame."""
        return self.palm_frame.body_id

    @property
    def fingertip_body_ids(self) -> tuple[int, ...]:
        """Compatibility DP-link body IDs derived from the authoritative DP frames."""
        return tuple(frame.body_id for frame in self.dp_frames)

    @property
    def collision_shape_ids(self) -> tuple[int, ...]:
        """Exact hand-owned shapes in stable semantic link order."""
        return tuple(shape for link in self.link_geometry for shape in link.shape_ids)

    @property
    def contact_links(self) -> tuple[HandLinkGeometry, ...]:
        """Semantic link groups whose shapes may produce manipulation contacts."""
        return tuple(link for link in self.link_geometry if link.object_contact)

    @property
    def contact_shape_ids(self) -> tuple[int, ...]:
        """Exact object-contact shapes in stable semantic link order."""
        return tuple(shape for link in self.contact_links for shape in link.shape_ids)

    @property
    def contact_link_count(self) -> int:
        return len(self.contact_links)

    @property
    def wrist_actuation_dof_ids(self) -> tuple[int, ...]:
        """DOFs that move the wrist: the arm chain if present, else the floating wrist joints."""
        return self.arm_dof_ids or (self.wrist_pos_dof_ids + self.wrist_orient_dof_ids)

    @property
    def joint_q_ids(self) -> tuple[int, ...]:
        """Scalar articulated-arm and finger coordinates, excluding floating wrist coordinates."""
        return self.arm_q_ids + self.finger_q_ids

    @property
    def joint_dof_ids(self) -> tuple[int, ...]:
        """Scalar articulated-arm and finger DOFs, excluding floating wrist DOFs."""
        return self.arm_dof_ids + self.finger_dof_ids

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self.arm_joint_names + self.finger_joint_names


@dataclass(frozen=True)
class EmbodimentLayout:
    """Full per-world DOF/coord/body layout produced by :meth:`Embodiment.build`."""

    num_joint_q: int  # position coords per world (ball joints store a quat)
    num_joint_dof: int  # actuated DOFs per world
    hands: tuple[HandLayout, ...]
    scalar_joints: ScalarJointLayout | None = None
    semantic_frames: tuple[BodyFrame, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "hands", tuple(self.hands))
        if self.num_joint_q < 0 or self.num_joint_dof < 0:
            raise ValueError("embodiment joint dimensions must be nonnegative")
        if not self.hands:
            raise ValueError("embodiment layout requires at least one hand")
        sides = tuple(hand.side for hand in self.hands)
        if len(set(sides)) != len(sides):
            raise ValueError(f"embodiment hand sides must be unique, got {sides}")

        hand_frames = tuple(
            frame for hand in self.hands for frame in (hand.palm_frame, *hand.dp_frames, *hand.fingertip_frames)
        )
        semantic_frames = tuple(self.semantic_frames) or hand_frames
        if any(not isinstance(frame, BodyFrame) for frame in semantic_frames):
            raise TypeError("embodiment semantic frames must contain only BodyFrame values")
        frame_names = tuple(frame.name for frame in semantic_frames)
        if len(set(frame_names)) != len(frame_names):
            raise ValueError(f"embodiment semantic frame names must be unique, got {frame_names}")
        frame_by_name = {frame.name: frame for frame in semantic_frames}
        missing_or_changed = tuple(frame.name for frame in hand_frames if frame_by_name.get(frame.name) != frame)
        if missing_or_changed:
            raise ValueError(f"embodiment semantic frames must include the exact hand frames: {missing_or_changed}")
        object.__setattr__(self, "semantic_frames", semantic_frames)

        q_ids: list[int] = []
        dof_ids: list[int] = []
        collision_shape_ids: list[int] = []
        hand_scalar_triples: list[tuple[int, int, str]] = []
        for hand in self.hands:
            hand_q_ids = list(hand.wrist_pos_q_ids + hand.arm_q_ids + hand.finger_q_ids)
            if hand.wrist_orient_q_id >= 0:
                hand_q_ids.extend(range(hand.wrist_orient_q_id, hand.wrist_orient_q_id + 4))
            q_ids.extend(hand_q_ids)
            dof_ids.extend(hand.wrist_actuation_dof_ids + hand.finger_dof_ids)
            collision_shape_ids.extend(hand.collision_shape_ids)
            hand_scalar_triples.extend(zip(hand.joint_q_ids, hand.joint_dof_ids, hand.joint_names, strict=True))

        _validate_unique_bounded(q_ids, self.num_joint_q, "embodiment q IDs")
        _validate_unique_bounded(dof_ids, self.num_joint_dof, "embodiment DOF IDs")
        _validate_unique_nonnegative(collision_shape_ids, "embodiment collision shape IDs")
        if self.scalar_joints is not None:
            _validate_unique_bounded(self.scalar_joints.q_ids, self.num_joint_q, "scalar joint q IDs")
            _validate_unique_bounded(self.scalar_joints.dof_ids, self.num_joint_dof, "scalar joint DOF IDs")
            scalar_triples = set(
                zip(
                    self.scalar_joints.q_ids,
                    self.scalar_joints.dof_ids,
                    self.scalar_joints.names,
                    strict=True,
                )
            )
            missing = sorted(set(hand_scalar_triples) - scalar_triples)
            if missing:
                raise ValueError(f"hand scalar joints are missing from the embodiment scalar layout: {missing}")

    def hand(self, side: str) -> HandLayout:
        for h in self.hands:
            if h.side == side:
                return h
        raise KeyError(f"no hand for side {side!r}; have {[h.side for h in self.hands]}")

    def frame(self, name: str) -> BodyFrame:
        """Return one exact named semantic frame in the per-world body layout."""
        for frame in self.semantic_frames:
            if frame.name == name:
                return frame
        raise KeyError(f"no semantic frame {name!r}; have {[frame.name for frame in self.semantic_frames]}")

    @property
    def sides(self) -> tuple[str, ...]:
        return tuple(h.side for h in self.hands)

    def select_scalar_joints(
        self,
        sides: tuple[str, ...] | None = None,
        groups: tuple[Literal["arm", "finger"], ...] = ("arm", "finger"),
        *,
        require_names: bool = False,
    ) -> ScalarJointSelection:
        """Select ordered joint identity once for setup-time runtime consumers."""
        sides = self.sides if sides is None else tuple(sides)
        groups = tuple(groups)
        if len(set(sides)) != len(sides) or set(sides) - set(self.sides):
            raise ValueError(f"selected sides {sides} must be unique members of embodiment sides {self.sides}")
        if not groups or len(set(groups)) != len(groups) or set(groups) - {"arm", "finger"}:
            raise ValueError(f"selected scalar joint groups must be unique arm/finger values, got {groups}")

        q_ids: list[int] = []
        dof_ids: list[int] = []
        names: list[str] = []
        blocks: list[ScalarJointBlock] = []
        for side in sides:
            hand = self.hand(side)
            for group in groups:
                block_start = len(q_ids)
                group_q_ids = getattr(hand, f"{group}_q_ids")
                group_dof_ids = getattr(hand, f"{group}_dof_ids")
                group_names = getattr(hand, f"{group}_joint_names")
                q_ids.extend(group_q_ids)
                dof_ids.extend(group_dof_ids)
                names.extend(group_names)
                blocks.append(
                    ScalarJointBlock(
                        side=side,
                        group=group,
                        start=block_start,
                        count=len(group_q_ids),
                    )
                )
        if require_names and len(names) != len(q_ids):
            raise ValueError(f"selected {groups} joints require complete names for sides {sides}")
        return ScalarJointSelection(
            sides=sides,
            groups=groups,
            q_ids=tuple(q_ids),
            dof_ids=tuple(dof_ids),
            names=tuple(names),
            blocks=tuple(blocks),
        )


def _validate_unique_nonnegative(indices: tuple[int, ...] | list[int], label: str) -> None:
    if len(set(indices)) != len(indices) or any(index < 0 for index in indices):
        raise ValueError(f"{label} must be unique and nonnegative, got {tuple(indices)}")


def _validate_unique_bounded(indices: tuple[int, ...] | list[int], size: int, label: str) -> None:
    _validate_unique_nonnegative(indices, label)
    invalid = tuple(index for index in indices if index >= size)
    if invalid:
        raise ValueError(f"{label} must be below {size}, got out-of-range IDs {invalid}")


@runtime_checkable
class Embodiment(Protocol):
    """Robot construction, static layout, and setup-time reference adaptation."""

    name: str

    def build(self, builder: Any) -> EmbodimentLayout:
        """Add the robot (one copy) to ``builder`` and return its layout."""
        ...

    def bind_reference(
        self,
        builder: Any,
        layout: EmbodimentLayout,
        reference: Any,
    ) -> RobotReferenceBinding:
        """Bind one raw reference to the exact built robot topology at setup."""
        ...


EMBODIMENT_REGISTRY: dict[str, type] = {}


def register_embodiment(name: str):
    """Class decorator: register an :class:`Embodiment` implementation under ``name``."""

    def _decorator(cls: type) -> type:
        EMBODIMENT_REGISTRY[name] = cls
        return cls

    return _decorator


def get_embodiment(name: str) -> type:
    """Return the registered embodiment class for ``name`` (raises if unknown)."""
    if name not in EMBODIMENT_REGISTRY:
        raise KeyError(f"unknown embodiment {name!r}; registered: {sorted(EMBODIMENT_REGISTRY)}")
    return EMBODIMENT_REGISTRY[name]


def finger_joint_order(hand: HandLayout, reference: Any) -> tuple[int, ...]:
    """Reference-column indices that place finger joints in ``hand``'s simulation order."""
    joint_pos = np.asarray(reference.finger_joint_pos(hand.side))
    reference_names = reference.finger_joint_names(hand.side)
    if len(reference_names) != joint_pos.shape[-1]:
        raise ValueError(
            f"{hand.side} reference has {joint_pos.shape[-1]} finger columns but {len(reference_names)} names"
        )
    index = {name: i for i, name in enumerate(reference_names)}
    missing = [name for name in hand.finger_joint_names if name not in index]
    if missing:
        raise ValueError(f"{hand.side} reference is missing finger joints: {missing}")
    return tuple(index[name] for name in hand.finger_joint_names)
