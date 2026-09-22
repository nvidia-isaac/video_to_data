# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for component-owned scene collision filtering."""

from itertools import combinations
from types import SimpleNamespace

import newton
import pytest

from flash_chord.embodiments.base import BodyFrame, EmbodimentLayout, HandLayout, HandLinkGeometry
from flash_chord.scene.collision import (
    CollisionPolicy,
    HandCollisionShapes,
    SceneCollisionLayout,
    ShapeSpan,
    apply_collision_policy,
    capture_collision_layout,
)


class _Builder:
    def __init__(self) -> None:
        collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
        self.shape_body = [-1, 0, 0, 1, 1, 2, 3, -1, 4, 5, 6]
        self.shape_flags = [collide] * len(self.shape_body)
        self.shape_collision_filter_pairs: list[tuple[int, int]] = []

    def add_shape_collision_filter_pair(self, shape_a: int, shape_b: int) -> None:
        self.shape_collision_filter_pairs.append((shape_a, shape_b))


def _spans():
    return ShapeSpan(0, 8), ShapeSpan(8, 10), ShapeSpan(10, 11)


def _frames(side: str, body_id: int) -> dict:
    return {
        "palm_frame": BodyFrame(f"{side}_palm", body_id),
        "dp_frames": (BodyFrame(f"{side}_index_DP", body_id),),
        "fingertip_frames": (BodyFrame(f"{side}_index_fingertip", body_id),),
    }


def _embodiment():
    return EmbodimentLayout(
        num_joint_q=0,
        num_joint_dof=0,
        hands=(
            HandLayout(
                side="left",
                **_frames("left", 0),
                link_geometry=(
                    HandLinkGeometry("palm", (1, 2), True),
                    HandLinkGeometry("virtual", (5,), False),
                ),
            ),
            HandLayout(
                side="right",
                **_frames("right", 1),
                link_geometry=(
                    HandLinkGeometry("palm", (3, 4), True),
                    HandLinkGeometry("virtual", (6,), False),
                ),
            ),
        ),
    )


def _layout(builder: _Builder) -> SceneCollisionLayout:
    return capture_collision_layout(builder, _embodiment(), *_spans())


def _active(builder: _Builder) -> set[int]:
    collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
    return {shape for shape, flags in enumerate(builder.shape_flags) if flags & collide}


def _pairs(builder: _Builder) -> set[tuple[int, int]]:
    return {tuple(sorted(pair)) for pair in builder.shape_collision_filter_pairs}


def test_capture_preserves_exact_component_and_hand_shape_ownership():
    builder = _Builder()

    layout = _layout(builder)

    assert layout.robot == ShapeSpan(0, 8)
    assert layout.objects == ShapeSpan(8, 10)
    assert layout.support == ShapeSpan(10, 11)
    assert layout.hands == (
        HandCollisionShapes("left", (1, 2, 5), (1, 2)),
        HandCollisionShapes("right", (3, 4, 6), (3, 4)),
    )
    assert layout.contact_shape_ids == (1, 2, 3, 4)


def test_capture_rejects_missing_exact_hand_shape_ownership():
    builder = _Builder()
    embodiment = EmbodimentLayout(
        num_joint_q=0,
        num_joint_dof=0,
        hands=(
            HandLayout(
                side="left",
                **_frames("left", 0),
            ),
        ),
    )

    with pytest.raises(ValueError, match="left hand must own at least one collision shape"):
        capture_collision_layout(builder, embodiment, *_spans())


def test_contact_only_and_none_disable_static_robot_shapes_without_touching_objects_or_support():
    contact_builder = _Builder()
    contact_result = apply_collision_policy(
        contact_builder,
        _layout(contact_builder),
        CollisionPolicy(robot_scope="contact_only"),
    )

    assert contact_result.disabled_robot_shapes == 4
    assert _active(contact_builder) == {1, 2, 3, 4, 8, 9, 10}
    assert 0 not in _active(contact_builder) and contact_builder.shape_body[0] == -1
    assert 7 not in _active(contact_builder) and contact_builder.shape_body[7] == -1

    none_builder = _Builder()
    none_result = apply_collision_policy(
        none_builder,
        _layout(none_builder),
        CollisionPolicy(robot_scope="none"),
    )
    assert none_result.disabled_robot_shapes == 8
    assert _active(none_builder) == {8, 9, 10}


def test_pair_switches_are_independent_and_preserve_cross_hand_and_object_support_contacts():
    builder = _Builder()
    builder.shape_collision_filter_pairs.append((1, 2))
    result = apply_collision_policy(
        builder,
        _layout(builder),
        CollisionPolicy(
            hand_self_collision=False,
            robot_object_collision=False,
            robot_support_collision=False,
        ),
    )
    pairs = _pairs(builder)

    assert (1, 2) in pairs and (3, 4) in pairs
    assert (1, 3) not in pairs
    assert all((robot, object_shape) in pairs for robot in range(8) for object_shape in range(8, 10))
    assert all((robot, 10) in pairs for robot in range(8))
    assert (8, 10) not in pairs and (9, 10) not in pairs
    assert result.added_filter_pairs == len(pairs) - 1


@pytest.mark.parametrize("scope", ["contact", "hands", ""])
def test_policy_rejects_unknown_robot_scope(scope):
    with pytest.raises(ValueError, match="robot_scope"):
        CollisionPolicy(robot_scope=scope)


def test_collision_layout_rejects_gaps_overlap_and_stale_builder_shape_count():
    with pytest.raises(ValueError, match="contiguous"):
        SceneCollisionLayout(
            ShapeSpan(0, 2),
            ShapeSpan(3, 4),
            ShapeSpan(4, 5),
            (HandCollisionShapes("left", (0,), (0,)),),
        )
    with pytest.raises(ValueError, match="overlaps"):
        SceneCollisionLayout(
            ShapeSpan(0, 2),
            ShapeSpan(2, 2),
            ShapeSpan(2, 2),
            (
                HandCollisionShapes("left", (0,), (0,)),
                HandCollisionShapes("right", (0,), (0,)),
            ),
        )
    builder = _Builder()
    layout = _layout(builder)
    stale = SimpleNamespace(
        shape_body=builder.shape_body[:-1],
        shape_flags=builder.shape_flags[:-1],
        shape_collision_filter_pairs=[],
        add_shape_collision_filter_pair=builder.add_shape_collision_filter_pair,
    )
    with pytest.raises(ValueError, match="builder has"):
        apply_collision_policy(stale, layout, CollisionPolicy())


@pytest.mark.gpu
@pytest.mark.sequence_data
def test_real_vega_manipulation_policy_reaches_exact_backend_contact_roles():
    import warp as wp

    from flash_chord.assets import ASSETS_DIR
    from flash_chord.assets.registry import support_usda_for_reference
    from flash_chord.data import load_reference
    from flash_chord.embodiments.vega_sharpa import VegaSharpa
    from flash_chord.runtime.sim import SimConfig, build_solver
    from flash_chord.scene.builder import build_scene

    parquet = (
        ASSETS_DIR
        / "human_motion_data"
        / "arctic"
        / "arctic_processed"
        / "sequence_id=dataset_s01_mixer_use_01"
        / "robot_name=vega_sharpa"
        / "data.parquet"
    )
    reference = load_reference(parquet, control_fps=20.0, motion_speed=0.5)
    policy = CollisionPolicy(
        robot_scope="contact_only",
        robot_support_collision=False,
        ground=False,
    )
    with wp.ScopedDevice("cuda:0"):
        scene = build_scene(
            VegaSharpa(),
            reference,
            support_usda=support_usda_for_reference(reference),
            collision=policy,
        )
        solver = build_solver(
            scene.model,
            SimConfig(fps=20.0, substeps=5, njmax=512, nconmax=256, cone="pyramidal"),
        )

    model = scene.model
    layout = scene.collision_layout
    collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
    flags = model.shape_flags.numpy()
    bodies = model.shape_body.numpy()
    active = {shape for shape, value in enumerate(flags) if value & collide}
    active_robot = active.intersection(layout.robot.ids())
    active_objects = active.intersection(layout.objects.ids())
    active_support = active.intersection(layout.support.ids())
    contact_shapes = set(layout.contact_shape_ids)

    assert scene.collision_policy == policy
    assert active_robot <= contact_shapes
    assert len(active_robot) == 44
    assert active_objects
    assert active_objects <= set(layout.objects.ids())
    assert active_support == set(layout.support.ids())
    assert all(shape not in active for shape in layout.robot.ids() if bodies[shape] == -1)
    assert model.shape_count == layout.shape_count

    filters = {tuple(sorted(pair)) for pair in model.shape_collision_filter_pairs}
    assert all(tuple(sorted((hand, support))) in filters for hand in active_robot for support in active_support)
    assert all(tuple(sorted((hand, obj))) not in filters for hand in active_robot for obj in active_objects)
    assert all(tuple(sorted((obj, support))) not in filters for obj in active_objects for support in active_support)

    hand_active = {hand.side: active.intersection(hand.contact_shape_ids) for hand in layout.hands}
    assert all(
        tuple(sorted((left, right))) not in filters for left in hand_active["left"] for right in hand_active["right"]
    )
    assert all(
        any(tuple(sorted(pair)) in filters for pair in combinations(shapes, 2)) for shapes in hand_active.values()
    )

    geom_to_shape = solver.mjc_geom_to_newton_shape.numpy()[0]
    backend_pairs = {
        tuple(sorted((int(geom_to_shape[geom_a]), int(geom_to_shape[geom_b]))))
        for geom_a, geom_b in solver.mjw_model.nxn_geom_pair_filtered.numpy()
    }
    hand_hand = {pair for pair in backend_pairs if pair[0] in active_robot and pair[1] in active_robot}
    hand_object = {
        pair
        for pair in backend_pairs
        if (pair[0] in active_robot and pair[1] in active_objects)
        or (pair[1] in active_robot and pair[0] in active_objects)
    }
    object_support = {
        pair
        for pair in backend_pairs
        if (pair[0] in active_objects and pair[1] in active_support)
        or (pair[1] in active_objects and pair[0] in active_support)
    }
    expected_hand_hand = {
        tuple(sorted(pair)) for pair in combinations(active_robot, 2) if tuple(sorted(pair)) not in filters
    }
    expected_hand_object = {tuple(sorted((hand, obj))) for hand in active_robot for obj in active_objects}
    expected_object_support = {tuple(sorted((obj, support))) for obj in active_objects for support in active_support}
    assert hand_hand == expected_hand_hand
    assert hand_object == expected_hand_object
    assert object_support == expected_object_support
    assert backend_pairs == hand_hand | hand_object | object_support
