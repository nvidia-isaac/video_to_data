# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Newton integration tests for the upstream-derived Vega Sharpa v2 asset."""

from __future__ import annotations

import numpy as np
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
_EXPECTED_JOINTS = tuple(
    name
    for arm_side, hand_side in (("L", "left"), ("R", "right"))
    for name in (
        *(f"{arm_side}_arm_j{index}" for index in range(1, 8)),
        *(f"{hand_side}_{suffix}" for suffix in _FINGER_SUFFIXES),
    )
)
_MIXER_REFERENCE = (
    "src/flash_chord/assets/human_motion_data/arctic/arctic_processed/"
    "sequence_id=dataset_s01_mixer_use_01/robot_name=vega_sharpa/data.parquet"
)


@pytest.fixture(scope="module")
def v2_build():
    import newton

    from flash_chord.configuration import compose_config, instantiate_typed
    from flash_chord.embodiments.base import Embodiment

    config = compose_config(
        "train",
        [
            "embodiment=dexmate_sharpa",
            "action=residual_joint_position",
            "observation=articulated_arm",
            "collision=manipulation",
            "task.parquet=/tmp/reference.parquet",
        ],
    )
    robot = instantiate_typed(config.embodiment, Embodiment)
    builder = newton.ModelBuilder()
    layout = robot.build(builder)
    return robot, builder, layout


def test_v2_hydra_asset_identity_and_exact_runtime_layout(v2_build):
    import hashlib

    import newton

    robot, builder, layout = v2_build
    assert robot.config.asset_id == "vega_sharpa_v2"
    assert hashlib.sha256(robot.config.urdf_path.read_bytes()).hexdigest() == robot.config.asset_sha256
    assert layout.num_joint_q == layout.num_joint_dof == 58
    assert layout.scalar_joints.q_ids == tuple(range(58))
    assert layout.scalar_joints.dof_ids == tuple(range(58))
    assert layout.scalar_joints.names == _EXPECTED_JOINTS

    fixed = tuple(
        label.rsplit("/", 1)[-1]
        for label, joint_type in zip(builder.joint_label, builder.joint_type, strict=True)
        if joint_type == newton.JointType.FIXED
    )
    assert fixed == ("head_j3", "L_hand_mount", "R_hand_mount")


def test_v2_rejects_urdf_that_does_not_match_configured_identity(v2_build):
    from dataclasses import replace

    import newton

    from flash_chord.embodiments.vega_sharpa import VegaSharpa

    robot, _, _ = v2_build
    invalid = VegaSharpa(replace(robot.config, asset_sha256="0" * 64))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        invalid.build(newton.ModelBuilder())


def test_v2_applies_calibrated_arm_controls_and_gravity_compensation(v2_build):
    import newton

    _, builder, _ = v2_build
    joints = {
        label.rsplit("/", 1)[-1]: joint
        for joint, label in enumerate(builder.joint_label)
        if builder.joint_type[joint] != newton.JointType.FIXED
    }
    expected = (
        (48.68459466964289, 41.15447085114586, 0.551, 100.0),
        (27.43237011630257, 23.159532675972144, 0.551, 100.0),
        (29.62593291473207, 24.986205930444992, 0.19072, 80.0),
        (23.134317278891867, 19.532232049454098, 0.19072, 80.0),
        (2.5463846202007985, 2.148072759799482, 0.07232, 25.0),
        (4.227251607475805, 3.5765041461700604, 0.07232, 25.0),
        (3.4862647411242134, 2.9439891209118327, 0.07232, 25.0),
    )
    for side in ("L", "R"):
        for index, (kp, kd, armature, effort) in enumerate(expected, start=1):
            dof = builder.joint_qd_start[joints[f"{side}_arm_j{index}"]]
            assert builder.joint_target_ke[dof] == pytest.approx(kp)
            assert builder.joint_target_kd[dof] == pytest.approx(kd)
            assert builder.joint_armature[dof] == pytest.approx(armature)
            assert builder.joint_effort_limit[dof] == pytest.approx(effort)
            assert builder.joint_velocity_limit[dof] == pytest.approx(2.4)
            assert builder.joint_target_mode[dof] == int(newton.JointTargetMode.POSITION_VELOCITY)

    assert all(builder.custom_attributes["mujoco:jnt_actgravcomp"].values.values())
    assert all(value == 1.0 for value in builder.custom_attributes["mujoco:gravcomp"].values.values())


def test_v2_camera_frames_survive_fixed_collapse(v2_build):
    _, builder, layout = v2_build
    expected_position = {
        "zed_depth_frame": (0.025, 0.023, 0.0489),
        "zed_left_camera": (0.0365, 0.023, 0.0489),
        "zed_right_camera": (0.0365, -0.027, 0.0489),
    }
    frames = tuple(layout.frame(name) for name in expected_position)
    assert len({frame.body_id for frame in frames}) == 1
    assert builder.body_label[frames[0].body_id].endswith("/head_l3")
    for frame in frames:
        np.testing.assert_allclose(frame.body_to_frame_pos, expected_position[frame.name], atol=1.0e-8, rtol=0.0)
        assert np.linalg.norm(frame.body_to_frame_quat_xyzw) == pytest.approx(1.0)


def test_v2_retains_hand_collision_ownership(v2_build):
    _, builder, layout = v2_build
    assert len(builder.shape_body) == 179
    for side in layout.sides:
        hand = layout.hand(side)
        assert len(hand.link_geometry) == 23
        assert hand.contact_link_count == 17
        assert len(hand.collision_shape_ids) == 54
        assert len(hand.contact_shape_ids) == 44
        assert set(hand.contact_shape_ids) < set(hand.collision_shape_ids)


@pytest.mark.sequence_data
def test_v2_binds_the_mixer_reference_with_named_frame_validation(v2_build):
    from flash_chord.data import load_reference

    robot, builder, layout = v2_build
    reference = load_reference(_MIXER_REFERENCE, control_fps=20.0, motion_speed=0.5)

    assert robot.config.validate_named_frames is True
    binding = robot.bind_reference(builder, layout, reference)

    assert binding.joint_q.shape == (reference.num_frames, 58)
    assert binding.joint_target.shape == (reference.num_frames, 58)
    assert binding.hand("left").fingertip_pos_w.shape == (reference.num_frames, 5, 3)
    assert binding.hand("right").fingertip_pos_w.shape == (reference.num_frames, 5, 3)
