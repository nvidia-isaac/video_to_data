# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Vega + SharpaWave (dexmate_sharpa) embodiment build + layout."""

import pytest

pytestmark = pytest.mark.gpu

_FINGER_SUFFIXES = (
    "thumb_CMC_FE",
    "thumb_CMC_AA",
    "thumb_MCP_FE",
    "thumb_MCP_AA",
    "thumb_IP",
    "index_MCP_FE",
    "index_MCP_AA",
    "index_PIP",
    "index_DIP",
    "middle_MCP_FE",
    "middle_MCP_AA",
    "middle_PIP",
    "middle_DIP",
    "ring_MCP_FE",
    "ring_MCP_AA",
    "ring_PIP",
    "ring_DIP",
    "pinky_CMC",
    "pinky_MCP_FE",
    "pinky_MCP_AA",
    "pinky_PIP",
    "pinky_DIP",
)
_DIGITS = ("thumb", "index", "middle", "ring", "pinky")
_OBJECT_CONTACT_LINKS = {
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
}


def _build(config=None):
    import newton
    import warp as wp

    from flash_chord.embodiments.vega_sharpa import VegaSharpa

    with wp.ScopedDevice("cuda:0"):
        builder = newton.ModelBuilder()
        layout = VegaSharpa(config).build(builder)
    return builder, layout


def _movable_joint_by_name(builder):
    import newton

    return {
        label.rsplit("/", 1)[-1]: joint
        for joint, label in enumerate(builder.joint_label)
        if builder.joint_type[joint] != newton.JointType.FIXED
    }


def _finger_physical_parameters(name):
    if "_CMC_" in name:
        return 0.0032, 3.3, 0.132
    if name.endswith("_pinky_CMC"):
        return 0.00012, 0.5285, 0.012
    if "_MCP_" in name:
        return 0.00265, 1.864, 0.07456
    if name.endswith(("_IP", "_PIP")):
        return 0.0006, 0.638, 0.01276
    if name.endswith("_DIP"):
        return 0.00042, 0.18937, 0.00378738
    raise AssertionError(f"unclassified finger joint {name!r}")


def test_vega_layout_matches_ground_truth():
    import newton

    builder, layout = _build()
    assert layout.num_joint_q == 58  # fixed base, all revolute
    assert layout.num_joint_dof == 58
    assert layout.sides == ("left", "right")
    assert layout.scalar_joints is not None
    assert layout.scalar_joints.q_ids == tuple(range(58))
    assert layout.scalar_joints.dof_ids == tuple(range(58))
    expected_joint_names = tuple(
        label.rsplit("/", 1)[-1]
        for _, label in sorted(
            (
                (builder.joint_q_start[joint], label)
                for joint, label in enumerate(builder.joint_label)
                if builder.joint_type[joint] != newton.JointType.FIXED
            )
        )
    )
    assert layout.scalar_joints.names == expected_joint_names

    left = layout.hand("left")
    assert left.arm_dof_ids == tuple(range(7))  # 7-DOF arm positions the wrist
    assert left.arm_q_ids == tuple(range(7))
    assert left.arm_joint_names == tuple(f"L_arm_j{i}" for i in range(1, 8))
    assert left.finger_dof_ids == tuple(range(7, 29))  # 22 finger DOFs
    assert left.finger_q_ids == tuple(range(7, 29))
    assert left.finger_joint_names == tuple(f"left_{suffix}" for suffix in _FINGER_SUFFIXES)
    assert left.wrist_actuation_dof_ids == left.arm_dof_ids  # arm drives the wrist (no floating joints)
    assert len(left.link_geometry) == 23
    assert left.contact_link_count == 17
    assert len(left.collision_shape_ids) == 54
    assert len(left.contact_shape_ids) == 44
    assert set(left.contact_shape_ids) < set(left.collision_shape_ids)

    right = layout.hand("right")
    assert right.arm_dof_ids == tuple(range(29, 36))
    assert right.arm_q_ids == tuple(range(29, 36))
    assert right.arm_joint_names == tuple(f"R_arm_j{i}" for i in range(1, 8))
    assert right.finger_dof_ids == tuple(range(36, 58))
    assert right.finger_q_ids == tuple(range(36, 58))
    assert right.finger_joint_names == tuple(f"right_{suffix}" for suffix in _FINGER_SUFFIXES)
    assert len(right.link_geometry) == 23
    assert right.contact_link_count == 17
    assert len(right.collision_shape_ids) == 54
    assert len(right.contact_shape_ids) == 44
    assert set(right.contact_shape_ids) < set(right.collision_shape_ids)

    fixed = {
        label.rsplit("/", 1)[-1]
        for label, joint_type in zip(builder.joint_label, builder.joint_type, strict=True)
        if joint_type == newton.JointType.FIXED
    }
    assert fixed == {"head_j3", "L_hand_mount", "R_hand_mount"}


def test_palm_dp_and_true_fingertip_frames_are_explicit():
    builder, layout = _build()
    for side in ("left", "right"):
        hand = layout.hand(side)
        assert hand.palm_frame.name == f"{side}_hand_C_MC"
        assert builder.body_label[hand.palm_frame.body_id].endswith(f"/{side}_hand_C_MC")
        assert hand.palm_frame.body_to_frame_pos == (0.0, 0.0, 0.0)
        assert hand.wrist_body_id == hand.palm_frame.body_id
        contact_names = {link.link_name for link in hand.contact_links}
        assert contact_names == _OBJECT_CONTACT_LINKS

        assert tuple(frame.name for frame in hand.dp_frames) == tuple(f"{side}_{digit}_DP" for digit in _DIGITS)
        assert tuple(frame.name for frame in hand.fingertip_frames) == tuple(
            f"{side}_{digit}_fingertip" for digit in _DIGITS
        )
        assert hand.fingertip_body_ids == tuple(frame.body_id for frame in hand.dp_frames)
        for dp, tip in zip(hand.dp_frames, hand.fingertip_frames, strict=True):
            assert builder.body_label[dp.body_id].endswith(f"/{dp.name}")
            assert dp.body_to_frame_pos == (0.0, 0.0, 0.0)
            assert tip.body_id == dp.body_id
            assert sum(value * value for value in tip.body_to_frame_pos) > 0.0004


def test_palm_collapse_preserves_logical_geometry_and_exact_collision_filters():
    from flash_chord.embodiments.vega_sharpa import VegaSharpaConfig

    retained_builder, retained = _build(VegaSharpaConfig(preserve_palm_bodies=True))
    collapsed_builder, collapsed = _build(VegaSharpaConfig(preserve_palm_bodies=False))
    retained_pairs = {tuple(sorted(pair)) for pair in retained_builder.shape_collision_filter_pairs}
    collapsed_pairs = {tuple(sorted(pair)) for pair in collapsed_builder.shape_collision_filter_pairs}

    assert len(retained_pairs) == 12_819
    assert collapsed_pairs == retained_pairs
    for side in retained.sides:
        assert collapsed.hand(side).link_geometry == retained.hand(side).link_geometry


def test_joint_control_parameters_match_dexmate_and_sharpa_reference_values():
    import newton

    builder, _ = _build()
    joints = _movable_joint_by_name(builder)
    arm = (
        (48.68459466964289, 41.15447085114586, 0.55100, 100.0),
        (27.43237011630257, 23.159532675972144, 0.55100, 100.0),
        (29.62593291473207, 24.986205930444992, 0.19072, 80.0),
        (23.134317278891867, 19.532232049454098, 0.19072, 80.0),
        (2.5463846202007985, 2.148072759799482, 0.07232, 25.0),
        (4.227251607475805, 3.5765041461700604, 0.07232, 25.0),
        (3.4862647411242134, 2.9439891209118327, 0.07232, 25.0),
    )

    visited = set()
    for side in ("L", "R"):
        for index, (kp, kd, armature, effort) in enumerate(arm, start=1):
            name = f"{side}_arm_j{index}"
            joint = joints[name]
            dof = builder.joint_qd_start[joint]
            visited.add(name)
            assert builder.joint_target_ke[dof] == pytest.approx(kp)
            assert builder.joint_target_kd[dof] == pytest.approx(kd)
            assert builder.joint_armature[dof] == pytest.approx(armature)
            assert builder.joint_effort_limit[dof] == pytest.approx(effort)
            assert builder.joint_velocity_limit[dof] == pytest.approx(2.4)
            assert builder.joint_target_mode[dof] == int(newton.JointTargetMode.POSITION_VELOCITY)

    for name, joint in joints.items():
        if name in visited:
            continue
        armature, effort, friction = _finger_physical_parameters(name)
        dof = builder.joint_qd_start[joint]
        visited.add(name)
        assert builder.joint_target_ke[dof] == pytest.approx(1.74533)
        assert builder.joint_target_kd[dof] == pytest.approx(0.01745)
        assert builder.joint_armature[dof] == pytest.approx(armature)
        assert builder.joint_effort_limit[dof] == pytest.approx(effort)
        assert builder.joint_velocity_limit[dof] == pytest.approx(11.62)
        assert builder.joint_friction[dof] == pytest.approx(friction)
        assert builder.joint_target_mode[dof] == int(newton.JointTargetMode.POSITION_VELOCITY)

    assert visited == set(joints)


def test_joint_control_overrides_are_applied_in_order():
    from flash_chord.embodiments.vega_sharpa import VegaSharpaConfig

    config = VegaSharpaConfig()
    config.joint_overrides = dict(config.joint_overrides)
    config.joint_overrides[r"L_arm_j1$"] = {"kp": 123.0}
    builder, _ = _build(config)
    joints = _movable_joint_by_name(builder)

    left_dof = builder.joint_qd_start[joints["L_arm_j1"]]
    right_dof = builder.joint_qd_start[joints["R_arm_j1"]]
    assert builder.joint_target_ke[left_dof] == pytest.approx(123.0)
    assert builder.joint_target_ke[right_dof] == pytest.approx(48.68459466964289)
    assert builder.joint_target_kd[left_dof] == pytest.approx(41.15447085114586)
    assert builder.joint_armature[left_dof] == pytest.approx(0.55100)
