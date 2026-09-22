# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build, control-parameter, and semantic-layout tests for G1+Dex3."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def g1():
    import newton

    from flash_chord.embodiments.g1_dex3 import G1Dex3

    builder = newton.ModelBuilder()
    layout = G1Dex3().build(builder)
    return builder, layout


def _movable_joints(builder):
    import newton

    return {
        label.rsplit("/", 1)[-1]: joint
        for joint, (label, joint_type) in enumerate(zip(builder.joint_label, builder.joint_type, strict=True))
        if joint_type not in (newton.JointType.FIXED, newton.JointType.FREE)
    }


def test_g1_layout_matches_floating_model_12_dex3_contract(g1):
    import newton

    from flash_chord.embodiments.g1_dex3 import G1_BODY_JOINT_NAMES, G1_DEX3_JOINT_NAMES

    builder, layout = g1
    assert (layout.num_joint_q, layout.num_joint_dof) == (50, 49)
    assert (len(builder.joint_label), len(builder.body_label), len(builder.shape_body)) == (54, 54, 94)
    assert layout.sides == ("left", "right")
    assert layout.scalar_joints.q_ids == tuple(range(7, 50))
    assert layout.scalar_joints.dof_ids == tuple(range(6, 49))
    assert layout.scalar_joints.names == (
        G1_BODY_JOINT_NAMES[:22] + G1_DEX3_JOINT_NAMES[:7] + G1_BODY_JOINT_NAMES[22:] + G1_DEX3_JOINT_NAMES[7:]
    )

    free_joints = [joint for joint, joint_type in enumerate(builder.joint_type) if joint_type == newton.JointType.FREE]
    assert free_joints == [0]
    assert builder.joint_dof_dim[0] == (3, 3)
    assert tuple(builder.joint_q[:7]) == pytest.approx((0.0, 0.0, 0.76, 0.0, 0.0, 0.0, 1.0))


def test_g1_hydra_config_instantiates_pinned_asset():
    from hydra.utils import instantiate

    from flash_chord.configuration import compose_config
    from flash_chord.embodiments.g1_dex3 import G1Dex3

    config = compose_config("train", ["embodiment=g1_dex3"])
    robot = instantiate(config.embodiment)

    assert isinstance(robot, G1Dex3)
    assert robot.config.asset_id == "g1_dex3_upstream_main"
    assert robot.config.asset_sha256 == "8c7f768dc8da6c969d8a4b4efb3dcbe3d9d2b05fca5104df12205a69879d19e9"
    assert robot.config.replace_cylinders_with_capsules is True


def test_g1_preserves_upstream_fixed_link_topology(g1):
    import newton

    builder, _ = g1
    body_names = {label.rsplit("/", 1)[-1] for label in builder.body_label}

    assert {"LL_FOOT", "LR_FOOT", "imu_in_pelvis", "imu_in_torso"} <= body_names
    assert sum(joint_type == newton.JointType.FIXED for joint_type in builder.joint_type) == 10


def test_g1_hands_expose_retained_palms_distal_links_and_contact_geometry(g1):
    builder, layout = g1
    for side in layout.sides:
        hand = layout.hand(side)
        assert hand.arm_joint_names == tuple(
            f"{side}_{suffix}_joint"
            for suffix in (
                "shoulder_pitch",
                "shoulder_roll",
                "shoulder_yaw",
                "elbow",
                "wrist_roll",
                "wrist_pitch",
                "wrist_yaw",
            )
        )
        assert hand.finger_joint_names == tuple(
            f"{side}_hand_{digit}_{index}_joint"
            for digit, count in (("thumb", 3), ("middle", 2), ("index", 2))
            for index in range(count)
        )
        assert hand.palm_frame.name == f"{side}_hand_palm_link"
        assert builder.body_label[hand.palm_frame.body_id].endswith(f"/{side}_hand_palm_link")
        assert hand.palm_frame.body_to_frame_pos == (0.0, 0.0, 0.0)
        assert tuple(frame.name for frame in hand.dp_frames) == (
            f"{side}_hand_thumb_2_link",
            f"{side}_hand_middle_1_link",
            f"{side}_hand_index_1_link",
        )
        assert tuple(frame.name for frame in hand.fingertip_frames) == (
            f"{side}_thumb_fingertip",
            f"{side}_middle_fingertip",
            f"{side}_index_fingertip",
        )
        assert len(hand.link_geometry) == 8
        assert len(hand.collision_shape_ids) == 16
        assert hand.contact_shape_ids == hand.collision_shape_ids


def test_g1_cylinder_colliders_are_converted_to_dimensionally_identical_capsules(g1):
    import newton

    from flash_chord.embodiments.g1_dex3 import G1Dex3

    builder, _ = g1
    source_cylinders = ET.parse(G1Dex3().config.urdf_path).getroot().iter("cylinder")
    expected = sorted(
        (float(cylinder.attrib["radius"]), 0.5 * float(cylinder.attrib["length"])) for cylinder in source_cylinders
    )
    actual = sorted(
        (float(builder.shape_scale[shape][0]), float(builder.shape_scale[shape][1]))
        for shape, shape_type in enumerate(builder.shape_type)
        if int(shape_type) == int(newton.GeoType.CAPSULE)
    )

    assert expected
    assert not any(int(shape_type) == int(newton.GeoType.CYLINDER) for shape_type in builder.shape_type)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-8)


def _expected_control(name: str) -> tuple[float, float, float, float, float]:
    frequency = 10.0 * 2.0 * 3.1415926535
    damping_ratio = 2.0
    if "_hand_" in name:
        return 2.0, 0.2, 0.00149, 0.76, 23.0
    if name.endswith(("hip_pitch_joint", "hip_roll_joint", "knee_joint")):
        armature, effort, velocity = 0.025101925, 139.0, 20.0
    elif name.endswith("hip_yaw_joint") or name == "waist_yaw_joint":
        armature, effort, velocity = 0.010177520, 88.0, 32.0
    elif name.endswith(("ankle_pitch_joint", "ankle_roll_joint")) or name in (
        "waist_roll_joint",
        "waist_pitch_joint",
    ):
        armature, effort, velocity = 2.0 * 0.003609725, 50.0, 37.0
    elif name.endswith(("wrist_pitch_joint", "wrist_yaw_joint")):
        armature, effort, velocity = 0.00425, 5.0, 22.0
    else:
        armature, effort, velocity = 0.003609725, 25.0, 37.0
    return (
        armature * frequency**2,
        2.0 * damping_ratio * armature * frequency,
        armature,
        effort,
        velocity,
    )


def test_g1_control_parameters_and_default_pose_match_upstream(g1):
    import newton

    builder, layout = g1
    joints = _movable_joints(builder)
    defaults = {
        "hip_pitch_joint": -0.312,
        "knee_joint": 0.669,
        "ankle_pitch_joint": -0.363,
        "elbow_joint": 0.6,
        "left_shoulder_roll_joint": 0.2,
        "left_shoulder_pitch_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
        "right_shoulder_pitch_joint": 0.2,
    }

    assert set(joints) == set(layout.scalar_joints.names)
    for name, joint in joints.items():
        q_id = builder.joint_q_start[joint]
        dof_id = builder.joint_qd_start[joint]
        kp, kd, armature, effort, velocity = _expected_control(name)
        assert builder.joint_target_ke[dof_id] == pytest.approx(kp)
        assert builder.joint_target_kd[dof_id] == pytest.approx(kd)
        assert builder.joint_armature[dof_id] == pytest.approx(armature)
        assert builder.joint_effort_limit[dof_id] == pytest.approx(effort)
        assert builder.joint_velocity_limit[dof_id] == pytest.approx(velocity)
        assert builder.joint_target_mode[dof_id] == int(newton.JointTargetMode.POSITION)

        expected_default = next(
            (value for suffix, value in defaults.items() if name == suffix or name.endswith(f"_{suffix}")),
            0.0,
        )
        assert builder.joint_q[q_id] == pytest.approx(expected_default)
        assert builder.joint_target_pos[dof_id] == pytest.approx(expected_default)
