# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Setup-time collision ownership and filtering for assembled scenes."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import newton

from flash_chord.embodiments.base import EmbodimentLayout

RobotShapeScope = Literal["all", "contact_only", "none"]


@dataclass(frozen=True)
class ShapeSpan:
    """Half-open shape-ID span owned by one appended scene component."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop < self.start:
            raise ValueError(f"shape span must satisfy 0 <= start <= stop, got ({self.start}, {self.stop})")

    def __len__(self) -> int:
        return self.stop - self.start

    def ids(self) -> range:
        return range(self.start, self.stop)


@dataclass(frozen=True)
class HandCollisionShapes:
    """All robot-owned shapes and the contact-rich subset for one hand."""

    side: str
    shape_ids: tuple[int, ...]
    contact_shape_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.side:
            raise ValueError("hand collision side must be non-empty")
        if not self.shape_ids:
            raise ValueError(f"{self.side} hand must own at least one collision shape")
        if len(set(self.shape_ids)) != len(self.shape_ids) or any(shape < 0 for shape in self.shape_ids):
            raise ValueError(f"{self.side} hand shape IDs must be unique and nonnegative, got {self.shape_ids}")
        if not self.contact_shape_ids:
            raise ValueError(f"{self.side} hand must own at least one contact shape")
        if len(set(self.contact_shape_ids)) != len(self.contact_shape_ids):
            raise ValueError(f"{self.side} contact shape IDs must be unique")
        invalid = set(self.contact_shape_ids) - set(self.shape_ids)
        if invalid:
            raise ValueError(f"{self.side} contact shapes are outside its hand shapes: {sorted(invalid)}")


@dataclass(frozen=True)
class SceneCollisionLayout:
    """Shape ownership for one unreplicated scene copy."""

    robot: ShapeSpan
    objects: ShapeSpan
    support: ShapeSpan
    hands: tuple[HandCollisionShapes, ...]

    def __post_init__(self) -> None:
        if self.robot.stop != self.objects.start or self.objects.stop != self.support.start:
            raise ValueError(
                "scene collision component spans must be contiguous and ordered robot -> objects -> support"
            )
        if len({hand.side for hand in self.hands}) != len(self.hands):
            raise ValueError("hand collision sides must be unique")
        robot_ids = set(self.robot.ids())
        assigned: set[int] = set()
        for hand in self.hands:
            invalid = set(hand.shape_ids) - robot_ids
            overlap = assigned.intersection(hand.shape_ids)
            if invalid:
                raise ValueError(f"{hand.side} hand shapes are outside the robot span: {sorted(invalid)}")
            if overlap:
                raise ValueError(f"hand collision shape ownership overlaps at {sorted(overlap)}")
            assigned.update(hand.shape_ids)

    @property
    def shape_count(self) -> int:
        return self.support.stop

    @property
    def contact_shape_ids(self) -> tuple[int, ...]:
        return tuple(shape for hand in self.hands for shape in hand.contact_shape_ids)


@dataclass(frozen=True)
class CollisionPolicy:
    """Independent scene collision choices applied before world replication."""

    robot_scope: RobotShapeScope = "all"
    hand_self_collision: bool = True
    robot_object_collision: bool = True
    robot_support_collision: bool = True
    ground: bool = True

    def __post_init__(self) -> None:
        if self.robot_scope not in ("all", "contact_only", "none"):
            raise ValueError(
                "robot_scope must be 'all', 'contact_only', or 'none', "
                f"got {self.robot_scope!r}"
            )


@dataclass(frozen=True)
class CollisionPolicyResult:
    disabled_robot_shapes: int
    added_filter_pairs: int


def capture_collision_layout(
    builder: newton.ModelBuilder,
    embodiment: EmbodimentLayout,
    robot: ShapeSpan,
    objects: ShapeSpan,
    support: ShapeSpan,
) -> SceneCollisionLayout:
    """Bind append-time component spans to exact embodiment-owned hand shapes."""
    shape_count = len(builder.shape_body)
    if support.stop != shape_count:
        raise ValueError(
            f"collision ownership covers {support.stop} shapes but the scene builder has {shape_count}"
        )
    hands = tuple(
        HandCollisionShapes(hand.side, hand.collision_shape_ids, hand.contact_shape_ids)
        for hand in embodiment.hands
    )
    return SceneCollisionLayout(robot, objects, support, hands)


def apply_collision_policy(
    builder: newton.ModelBuilder,
    layout: SceneCollisionLayout,
    policy: CollisionPolicy,
) -> CollisionPolicyResult:
    """Apply one subtractive collision policy to an unreplicated scene copy."""
    if len(builder.shape_body) != layout.shape_count:
        raise ValueError(
            f"collision layout covers {layout.shape_count} shapes but the builder has {len(builder.shape_body)}"
        )
    collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
    contact_shapes = set(layout.contact_shape_ids)
    disabled = 0
    for shape in layout.robot.ids():
        keep = policy.robot_scope == "all" or (
            policy.robot_scope == "contact_only" and shape in contact_shapes
        )
        if not keep and builder.shape_flags[shape] & collide:
            builder.shape_flags[shape] &= ~collide
            disabled += 1

    active_robot = _active_shapes(builder, layout.robot)
    active_objects = _active_shapes(builder, layout.objects)
    active_support = _active_shapes(builder, layout.support)
    pairs: list[tuple[int, int]] = []
    if not policy.hand_self_collision:
        active_robot_set = set(active_robot)
        for hand in layout.hands:
            pairs.extend(combinations((shape for shape in hand.shape_ids if shape in active_robot_set), 2))
    if not policy.robot_object_collision:
        pairs.extend((robot_shape, object_shape) for robot_shape in active_robot for object_shape in active_objects)
    if not policy.robot_support_collision:
        pairs.extend((robot_shape, support_shape) for robot_shape in active_robot for support_shape in active_support)
    added = _add_filter_pairs(builder, pairs)
    return CollisionPolicyResult(disabled_robot_shapes=disabled, added_filter_pairs=added)


def _active_shapes(builder: newton.ModelBuilder, span: ShapeSpan) -> tuple[int, ...]:
    collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
    return tuple(shape for shape in span.ids() if builder.shape_flags[shape] & collide)


def _add_filter_pairs(builder: newton.ModelBuilder, pairs) -> int:
    existing = {tuple(sorted(pair)) for pair in builder.shape_collision_filter_pairs}
    added = 0
    for shape_a, shape_b in pairs:
        if shape_a == shape_b:
            continue
        pair = (min(shape_a, shape_b), max(shape_a, shape_b))
        if pair in existing:
            continue
        builder.add_shape_collision_filter_pair(*pair)
        existing.add(pair)
        added += 1
    return added
