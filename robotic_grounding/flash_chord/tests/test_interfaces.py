# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the embodiment and reference interfaces."""

from dataclasses import replace

import pytest

from flash_chord.data.reference import ObjectJointPhysicsSpec, Reference
from flash_chord.embodiments.base import (
    EMBODIMENT_REGISTRY,
    BodyFrame,
    Embodiment,
    EmbodimentLayout,
    HandLayout,
    HandLinkGeometry,
    JointControlSpec,
    ScalarJointBlock,
    ScalarJointLayout,
    finger_joint_order,
    get_embodiment,
    register_embodiment,
    resolve_joint_controls,
)


def _semantic_frames(side: str, body_id: int = 0) -> dict:
    return {
        "palm_frame": BodyFrame(f"{side}_palm", body_id),
        "dp_frames": (BodyFrame(f"{side}_index_DP", body_id),),
        "fingertip_frames": (BodyFrame(f"{side}_index_fingertip", body_id),),
    }


def _layout() -> EmbodimentLayout:
    left = HandLayout(
        side="left",
        **_semantic_frames("left"),
        wrist_pos_dof_ids=(0, 1, 2),
        wrist_orient_dof_ids=(3, 4, 5),
        wrist_orient_q_id=3,
        finger_dof_ids=tuple(range(6, 28)),
        wrist_pos_q_ids=(0, 1, 2),
        finger_q_ids=tuple(range(7, 29)),
        finger_joint_names=tuple(f"left_finger_{index}" for index in range(22)),
    )
    return EmbodimentLayout(num_joint_q=58, num_joint_dof=56, hands=(left,))


def test_protocols_are_runtime_checkable_protocols():
    assert getattr(Embodiment, "_is_protocol", False)
    assert getattr(Reference, "_is_protocol", False)


def test_layout_lookup_and_sides():
    layout = _layout()
    assert layout.sides == ("left",)
    assert layout.hand("left").wrist_orient_q_id == 3
    assert len(layout.hand("left").finger_dof_ids) == 22


def test_layout_semantic_frames_default_to_exact_hand_frames_and_support_lookup():
    layout = _layout()
    hand = layout.hand("left")
    expected = (hand.palm_frame, *hand.dp_frames, *hand.fingertip_frames)

    assert layout.semantic_frames == expected
    assert tuple(layout.frame(frame.name) for frame in expected) == expected
    with pytest.raises(KeyError, match="no semantic frame 'missing'"):
        layout.frame("missing")


def test_layout_semantic_frames_preserve_auxiliary_frames_and_reject_ambiguous_sets():
    layout = _layout()
    camera = BodyFrame("camera", 1, body_to_frame_pos=(0.1, 0.0, 0.0))
    extended = replace(layout, semantic_frames=[*layout.semantic_frames, camera])
    assert isinstance(extended.semantic_frames, tuple)
    assert extended.semantic_frames == (*layout.semantic_frames, camera)
    assert extended.frame("camera") == camera

    duplicate = (*layout.semantic_frames, BodyFrame(layout.semantic_frames[0].name, 1))
    with pytest.raises(ValueError, match="semantic frame names must be unique"):
        replace(layout, semantic_frames=duplicate)
    with pytest.raises(ValueError, match="must include the exact hand frames"):
        replace(layout, semantic_frames=layout.semantic_frames[:-1])
    changed_palm = BodyFrame(layout.semantic_frames[0].name, 1)
    with pytest.raises(ValueError, match="must include the exact hand frames"):
        replace(layout, semantic_frames=(changed_palm, *layout.semantic_frames[1:]))
    with pytest.raises(TypeError, match="only BodyFrame"):
        replace(layout, semantic_frames=(*layout.semantic_frames, "camera"))


def test_articulated_hand_layout_exposes_named_scalar_joint_and_frame_groups():
    palm = BodyFrame(name="left_hand_C_MC", body_id=6)
    dp = BodyFrame(name="left_index_DP", body_id=9)
    tip = BodyFrame(
        name="left_index_fingertip",
        body_id=9,
        body_to_frame_pos=(0.026, 0.0, 0.0),
    )
    hand = HandLayout(
        side="left",
        arm_dof_ids=(0, 1),
        arm_q_ids=(0, 1),
        arm_joint_names=("arm_1", "arm_2"),
        finger_dof_ids=(2, 3),
        finger_q_ids=(2, 3),
        finger_joint_names=("finger_1", "finger_2"),
        link_geometry=(
            HandLinkGeometry("virtual", (10,), False),
            HandLinkGeometry("finger", (11,), True),
        ),
        palm_frame=palm,
        dp_frames=(dp,),
        fingertip_frames=(tip,),
    )
    assert hand.joint_dof_ids == (0, 1, 2, 3)
    assert hand.joint_q_ids == (0, 1, 2, 3)
    assert hand.joint_names == ("arm_1", "arm_2", "finger_1", "finger_2")
    assert hand.palm_frame.body_to_frame_quat_xyzw == (0.0, 0.0, 0.0, 1.0)
    assert hand.fingertip_frames[0].body_to_frame_pos == (0.026, 0.0, 0.0)
    assert hand.contact_shape_ids == (11,)
    scalar_joints = ScalarJointLayout(
        q_ids=hand.joint_q_ids,
        dof_ids=hand.joint_dof_ids,
        names=hand.joint_names,
    )
    layout = EmbodimentLayout(4, 4, (hand,), scalar_joints=scalar_joints)
    assert layout.scalar_joints.names == ("arm_1", "arm_2", "finger_1", "finger_2")
    selection = layout.select_scalar_joints(("left",), ("finger", "arm"), require_names=True)
    assert selection.q_ids == (2, 3, 0, 1)
    assert selection.dof_ids == (2, 3, 0, 1)
    assert selection.names == ("finger_1", "finger_2", "arm_1", "arm_2")
    assert selection.side_counts == (4,)
    assert selection.blocks == (
        ScalarJointBlock(side="left", group="finger", start=0, count=2),
        ScalarJointBlock(side="left", group="arm", start=2, count=2),
    )
    assert selection.block_starts == (0, 2)
    assert selection.block_counts == (2, 2)


def test_scalar_joint_layout_rejects_ambiguous_topology():
    with pytest.raises(ValueError, match="equal lengths"):
        ScalarJointLayout(q_ids=(0,), dof_ids=(0, 1), names=("joint",))
    with pytest.raises(ValueError, match="q IDs must be unique"):
        ScalarJointLayout(q_ids=(0, 0), dof_ids=(0, 1), names=("joint_a", "joint_b"))
    with pytest.raises(ValueError, match="names must be unique"):
        ScalarJointLayout(q_ids=(0, 1), dof_ids=(0, 1), names=("joint", "joint"))


def test_embodiment_layout_rejects_ambiguous_or_mixed_topology():
    with pytest.raises(ValueError, match="both arm-driven and floating-wrist"):
        HandLayout(
            side="left",
            **_semantic_frames("left"),
            wrist_pos_dof_ids=(0, 1, 2),
            wrist_orient_dof_ids=(3, 4, 5),
            wrist_pos_q_ids=(0, 1, 2),
            wrist_orient_q_id=3,
            arm_q_ids=(7,),
            arm_dof_ids=(6,),
            arm_joint_names=("arm",),
        )
    with pytest.raises(ValueError, match="collision shape IDs must be unique"):
        HandLayout(
            side="left",
            link_geometry=(
                HandLinkGeometry("virtual", (1,), False),
                HandLinkGeometry("finger", (1,), True),
            ),
            **_semantic_frames("left"),
        )
    with pytest.raises(ValueError, match="include at least one object-contact link"):
        HandLayout(
            side="left",
            link_geometry=(HandLinkGeometry("virtual", (1,), False),),
            **_semantic_frames("left"),
        )
    with pytest.raises(ValueError, match="finger names must match"):
        HandLayout(
            side="left",
            finger_q_ids=(0,),
            finger_dof_ids=(0,),
            **_semantic_frames("left"),
        )

    hand = HandLayout(
        side="left",
        **_semantic_frames("left"),
        arm_q_ids=(0,),
        arm_dof_ids=(0,),
        arm_joint_names=("arm",),
    )
    with pytest.raises(ValueError, match="sides must be unique"):
        EmbodimentLayout(2, 2, (hand, hand))
    with pytest.raises(ValueError, match="out-of-range"):
        EmbodimentLayout(0, 1, (hand,))


def test_bulk_joint_control_resolution_validates_every_pattern_and_preserves_order():
    resolved = resolve_joint_controls(
        ("robot/L_arm_j1", "robot/R_arm_j1"),
        JointControlSpec(kp=1.0, kd=2.0),
        {
            r"_arm_j": {"kp": 10.0, "velocity_limit": 2.4},
            r"L_arm_j1$": {"kd": 20.0, "effort_limit": 100.0},
        },
    )

    assert tuple(spec.kp for spec in resolved) == (10.0, 10.0)
    assert tuple(spec.kd for spec in resolved) == (20.0, 2.0)
    assert tuple(spec.velocity_limit for spec in resolved) == (2.4, 2.4)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({r"missing$": {"kp": 1.0}}, "does not match"),
        ({r"[": {"kp": 1.0}}, "invalid joint override regex"),
        ({r"_arm": {"unknown": 1.0}}, "invalid joint override"),
        ({r"_arm": {"effort_limit": 0.0}}, "invalid joint override"),
    ],
)
def test_bulk_joint_control_resolution_rejects_inert_or_invalid_configuration(overrides, message):
    with pytest.raises(ValueError, match=message):
        resolve_joint_controls(("robot/L_arm_j1",), JointControlSpec(), overrides)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"mode": "unknown"}, "unsupported"),
        ({"kp": -1.0}, "nonnegative"),
        ({"effort_limit": 0.0}, "must be positive"),
        ({"velocity_limit": float("inf")}, "positive and finite"),
        ({"friction": -1.0}, "nonnegative and finite"),
    ],
)
def test_joint_control_spec_rejects_invalid_physical_parameters(values, message):
    with pytest.raises(ValueError, match=message):
        JointControlSpec(**values)


def test_object_joint_physics_spec_is_passive_and_finite():
    assert ObjectJointPhysicsSpec(armature=0.01, friction=0.1) == ObjectJointPhysicsSpec(
        armature=0.01,
        friction=0.1,
    )
    for values in (
        {"armature": -1.0, "friction": 0.1},
        {"armature": 0.01, "friction": -1.0},
        {"armature": float("inf"), "friction": 0.1},
        {"armature": 0.01, "friction": float("nan")},
    ):
        with pytest.raises(ValueError, match="finite|nonnegative"):
            ObjectJointPhysicsSpec(**values)


def test_layout_missing_side_raises():
    layout = _layout()
    try:
        layout.hand("right")
    except KeyError:
        return
    raise AssertionError("expected KeyError for missing side")


def test_embodiment_registry_roundtrip():
    @register_embodiment("_test_dummy")
    class _Dummy:
        name = "_test_dummy"

    try:
        assert get_embodiment("_test_dummy") is _Dummy
    finally:
        EMBODIMENT_REGISTRY.pop("_test_dummy", None)


def test_get_unknown_embodiment_raises():
    try:
        get_embodiment("_does_not_exist")
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown embodiment")


def test_finger_joint_order_matches_names():
    class _Reference:
        def finger_joint_pos(self, side):
            return [[1.0, 2.0]]

        def finger_joint_names(self, side):
            return ["left_b", "left_a"]

    hand = HandLayout(
        side="left",
        **_semantic_frames("left"),
        finger_dof_ids=(0, 1),
        finger_q_ids=(0, 1),
        finger_joint_names=("left_a", "left_b"),
    )
    assert finger_joint_order(hand, _Reference()) == (1, 0)
