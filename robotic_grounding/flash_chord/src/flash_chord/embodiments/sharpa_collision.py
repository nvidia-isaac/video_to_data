# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mechanical collision exclusions shared by Sharpa hand mountings."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping

import newton

from flash_chord.embodiments.base import HandLinkGeometry
from flash_chord.utils.labels import clean_label

_EXCLUDED_CONTACTS = (
    ("hand_C_MC", "thumb_MC"),
    ("thumb_MC", "thumb_PP"),
    ("hand_C_MC", "index_PP"),
    ("hand_C_MC", "middle_PP"),
    ("hand_C_MC", "ring_PP"),
    ("hand_C_MC", "pinky_MC"),
    ("pinky_MC", "pinky_PP"),
    ("hand_C_MC", "thumb_PP"),
    ("hand_C_MC", "pinky_PP"),
    ("index_DP", "index_MCP_VL"),
    ("index_DP", "index_PP"),
    ("index_DP", "middle_MCP_VL"),
    ("index_DP", "pinky_MC"),
    ("index_DP", "pinky_MCP_VL"),
    ("index_DP", "ring_MCP_VL"),
    ("index_MCP_VL", "index_MP"),
    ("index_MCP_VL", "middle_DP"),
    ("index_MCP_VL", "middle_MP"),
    ("index_MCP_VL", "pinky_DP"),
    ("index_MCP_VL", "pinky_MC"),
    ("index_MCP_VL", "pinky_MCP_VL"),
    ("index_MCP_VL", "pinky_MP"),
    ("index_MCP_VL", "pinky_PP"),
    ("index_MCP_VL", "ring_DP"),
    ("index_MCP_VL", "ring_MCP_VL"),
    ("index_MCP_VL", "ring_MP"),
    ("index_MCP_VL", "ring_PP"),
    ("index_MCP_VL", "thumb_MC"),
    ("index_MP", "middle_MCP_VL"),
    ("index_MP", "pinky_MC"),
    ("index_MP", "pinky_MCP_VL"),
    ("index_MP", "ring_MCP_VL"),
    ("index_PP", "pinky_MC"),
    ("index_PP", "pinky_MCP_VL"),
    ("index_PP", "ring_MCP_VL"),
    ("middle_DP", "middle_MCP_VL"),
    ("middle_DP", "middle_PP"),
    ("middle_DP", "pinky_MC"),
    ("middle_DP", "pinky_MCP_VL"),
    ("middle_DP", "ring_MCP_VL"),
    ("middle_MCP_VL", "middle_MP"),
    ("middle_MCP_VL", "pinky_DP"),
    ("middle_MCP_VL", "pinky_MC"),
    ("middle_MCP_VL", "pinky_MCP_VL"),
    ("middle_MCP_VL", "pinky_MP"),
    ("middle_MCP_VL", "pinky_PP"),
    ("middle_MCP_VL", "ring_DP"),
    ("middle_MCP_VL", "ring_MP"),
    ("middle_MCP_VL", "thumb_MC"),
    ("middle_MP", "pinky_MC"),
    ("middle_MP", "pinky_MCP_VL"),
    ("middle_MP", "ring_MCP_VL"),
    ("middle_PP", "pinky_MC"),
    ("middle_PP", "pinky_MCP_VL"),
    ("pinky_DP", "pinky_MCP_VL"),
    ("pinky_DP", "pinky_PP"),
    ("pinky_DP", "ring_MCP_VL"),
    ("pinky_MC", "pinky_MP"),
    ("pinky_MC", "ring_MP"),
    ("pinky_MC", "thumb_MC"),
    ("pinky_MCP_VL", "pinky_MP"),
    ("pinky_MCP_VL", "ring_DP"),
    ("pinky_MCP_VL", "ring_MP"),
    ("pinky_MCP_VL", "thumb_MC"),
    ("pinky_MP", "ring_MCP_VL"),
    ("ring_DP", "ring_MCP_VL"),
    ("ring_DP", "ring_PP"),
    ("ring_MCP_VL", "ring_MP"),
    ("ring_MCP_VL", "thumb_MC"),
    ("thumb_DP", "thumb_MC"),
)
_OBJECT_CONTACT_LINKS = (
    "hand_C_MC",
    "thumb_MC",
    "thumb_PP",
    "thumb_DP",
    "index_PP",
    "index_MP",
    "index_DP",
    "middle_PP",
    "middle_MP",
    "middle_DP",
    "ring_PP",
    "ring_MP",
    "ring_DP",
    "pinky_MC",
    "pinky_PP",
    "pinky_MP",
    "pinky_DP",
)


def sharpa_excluded_contacts() -> list[tuple[str, str]]:
    """Return an independent mutable copy for typed embodiment configuration."""
    return list(_EXCLUDED_CONTACTS)


def sharpa_object_contact_links() -> list[str]:
    """Return an independent explicit object-contact role declaration for Sharpa links."""
    return list(_OBJECT_CONTACT_LINKS)


def capture_sharpa_link_geometry(
    builder: newton.ModelBuilder,
    *,
    palm_body_ids: Mapping[str, int],
    sides: tuple[str, ...],
    object_contact_links: Collection[str],
) -> dict[str, tuple[HandLinkGeometry, ...]]:
    """Group pre-collapse Sharpa shapes by stable logical hand link."""
    if not sides or len(set(sides)) != len(sides):
        raise ValueError(f"Sharpa geometry sides must be unique and nonempty, got {sides}")
    if set(palm_body_ids) != set(sides):
        raise ValueError(
            f"Sharpa palm bodies must cover sides {sides} exactly, got {tuple(palm_body_ids)}"
        )
    body_count = len(builder.body_label)
    invalid_palms = {
        side: body_id
        for side, body_id in palm_body_ids.items()
        if body_id < 0 or body_id >= body_count
    }
    if invalid_palms:
        raise ValueError(f"Sharpa palm body IDs are outside the builder: {invalid_palms}")
    contact_names = tuple(object_contact_links)
    if (
        not contact_names
        or len(set(contact_names)) != len(contact_names)
        or any(not isinstance(name, str) or not name for name in contact_names)
    ):
        raise ValueError(f"Sharpa object-contact links must be unique nonempty names, got {contact_names}")
    contact_name_set = frozenset(contact_names)

    incoming = _incoming_joints(builder)
    protected = frozenset(palm_body_ids.values())
    children: dict[int, list[int]] = defaultdict(list)
    for parent, child in zip(builder.joint_parent, builder.joint_child, strict=True):
        if child >= 0:
            children[int(parent)].append(int(child))

    owned_bodies: set[int] = set()
    links_by_side: dict[str, tuple[HandLinkGeometry, ...]] = {}
    for side in sides:
        descendants = _descendants(int(palm_body_ids[side]), children)
        overlap = owned_bodies.intersection(descendants)
        if overlap:
            raise ValueError(f"Sharpa hand body trees overlap at {tuple(sorted(overlap))}")
        owned_bodies.update(descendants)

        grouped_shapes: dict[int, list[int]] = {}
        for shape_id, body_id in enumerate(builder.shape_body):
            if body_id not in descendants:
                continue
            owner = _fold_fixed_body(int(body_id), incoming, protected)
            if owner < 0 or owner not in descendants:
                raise ValueError(
                    f"{side} hand shape {shape_id} folded outside its palm subtree to body {owner}"
                )
            grouped_shapes.setdefault(owner, []).append(shape_id)
        if not grouped_shapes:
            raise ValueError(f"{side} Sharpa palm subtree owns no collision shapes")

        grouped_links = tuple(
            (clean_label(builder.body_label[owner]), tuple(shape_ids))
            for owner, shape_ids in grouped_shapes.items()
        )
        missing = contact_name_set - {name for name, _ in grouped_links}
        if missing:
            raise ValueError(f"Sharpa object-contact links are missing from {side} geometry: {tuple(sorted(missing))}")
        links_by_side[side] = tuple(
            HandLinkGeometry(
                link_name=name,
                shape_ids=shape_ids,
                object_contact=name in contact_name_set,
            )
            for name, shape_ids in grouped_links
        )
    return links_by_side


def filter_sharpa_collisions(
    builder: newton.ModelBuilder,
    links_by_side: Mapping[str, tuple[HandLinkGeometry, ...]],
    *,
    protected_fixed_bodies: Collection[int],
    excluded_contacts: Iterable[tuple[str, str]],
) -> int:
    """Apply exact logical adjacency and curated exclusions before fixed-joint collapse."""
    if not links_by_side:
        raise ValueError("Sharpa collision filtering requires at least one hand geometry")
    incoming = _incoming_joints(builder)
    protected = frozenset(int(body_id) for body_id in protected_fixed_bodies)
    if any(body_id < 0 or body_id >= len(builder.body_label) for body_id in protected):
        raise ValueError(f"protected fixed bodies are outside the builder: {tuple(sorted(protected))}")
    pairs = list(_logical_adjacent_shape_pairs(builder, incoming, protected))
    pairs.extend(_curated_shape_pairs(links_by_side, excluded_contacts))
    return _add_filter_pairs(builder, pairs)


def _logical_adjacent_shape_pairs(builder, incoming, protected):
    collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
    body_shapes: dict[int, list[int]] = defaultdict(list)
    for shape, body in enumerate(builder.shape_body):
        if body < 0 or not (builder.shape_flags[shape] & collide):
            continue
        owner = _fold_fixed_body(int(body), incoming, protected)
        if owner >= 0:
            body_shapes[owner].append(shape)
    for parent, child in zip(builder.joint_parent, builder.joint_child, strict=True):
        logical_parent = _fold_fixed_body(int(parent), incoming, protected)
        logical_child = _fold_fixed_body(int(child), incoming, protected)
        if logical_parent < 0 or logical_parent == logical_child:
            continue
        yield from (
            (shape_a, shape_b)
            for shape_a in body_shapes.get(logical_parent, ())
            for shape_b in body_shapes.get(logical_child, ())
        )


def _curated_shape_pairs(
    links_by_side: Mapping[str, tuple[HandLinkGeometry, ...]],
    excluded_contacts: Iterable[tuple[str, str]],
):
    configured: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw_pair in excluded_contacts:
        pair = tuple(raw_pair)
        if len(pair) != 2 or any(not isinstance(name, str) or not name for name in pair):
            raise ValueError(f"Sharpa excluded contact must contain two nonempty link names, got {raw_pair!r}")
        name_a, name_b = pair
        if name_a == name_b:
            raise ValueError(f"Sharpa excluded contact cannot pair a link with itself: {pair}")
        normalized_pair = tuple(sorted(pair))
        if normalized_pair in seen:
            raise ValueError(f"duplicate Sharpa excluded contact pair: {pair}")
        seen.add(normalized_pair)
        configured.append(pair)

    for side, links in links_by_side.items():
        by_name = {link.link_name: link for link in links}
        if len(by_name) != len(links):
            raise ValueError(f"{side} Sharpa collision link names must be unique")
        for name_a, name_b in configured:
            missing = tuple(name for name in (name_a, name_b) if name not in by_name)
            if missing:
                raise ValueError(
                    f"{side} Sharpa excluded contact {(name_a, name_b)} has missing links {missing}"
                )
            yield from (
                (shape_a, shape_b)
                for shape_a in by_name[name_a].shape_ids
                for shape_b in by_name[name_b].shape_ids
            )


def _incoming_joints(builder) -> dict[int, tuple[int, int]]:
    incoming: dict[int, tuple[int, int]] = {}
    for parent, child, joint_type in zip(
        builder.joint_parent,
        builder.joint_child,
        builder.joint_type,
        strict=True,
    ):
        child = int(child)
        if child < 0:
            continue
        if child in incoming:
            raise ValueError(f"body {child} has multiple incoming joints")
        incoming[child] = (int(parent), int(joint_type))
    return incoming


def _fold_fixed_body(
    body_id: int,
    incoming: Mapping[int, tuple[int, int]],
    protected: Collection[int],
) -> int:
    while body_id >= 0 and body_id not in protected:
        parent, joint_type = incoming.get(body_id, (-1, -1))
        if joint_type != int(newton.JointType.FIXED):
            break
        body_id = parent
    return body_id


def _descendants(root: int, children: Mapping[int, list[int]]) -> set[int]:
    descendants: set[int] = set()
    stack = [root]
    while stack:
        body_id = stack.pop()
        if body_id in descendants:
            raise ValueError(f"body tree contains a cycle at {body_id}")
        descendants.add(body_id)
        stack.extend(children.get(body_id, ()))
    return descendants


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
