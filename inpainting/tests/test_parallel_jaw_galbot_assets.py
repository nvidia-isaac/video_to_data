from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np

from inpainting.parallel_jaw_galbot_assets import (
    GRIPPER_MAX_OPENING,
    GRIPPER_MIN_OPENING,
    _freeze_joint,
    _origin_transform,
    _rpy_matrix,
    _strip_non_arm_visuals,
)


def _joint(
    *,
    joint_type: str = "revolute",
    xyz: str = "0.1 -0.2 0.3",
    rpy: str = "0.2 -0.3 0.4",
    axis: str = "0 0 1",
) -> ET.Element:
    return ET.fromstring(
        f"""
        <joint name="sample" type="{joint_type}">
          <parent link="parent"/>
          <child link="child"/>
          <origin xyz="{xyz}" rpy="{rpy}"/>
          <axis xyz="{axis}"/>
          <limit lower="-3" upper="3" effort="1" velocity="1"/>
          <mimic joint="driver" multiplier="-1"/>
        </joint>
        """
    )


def test_freeze_revolute_bakes_joint_motion_and_removes_actuation() -> None:
    joint = _joint()
    before = _origin_transform(joint)
    angle = 0.73
    expected = before.copy()
    expected[:3, :3] = before[:3, :3] @ _rpy_matrix(np.array((0.0, 0.0, angle)))

    _freeze_joint(joint, angle)

    assert joint.attrib["type"] == "fixed"
    assert joint.find("axis") is None
    assert joint.find("limit") is None
    assert joint.find("mimic") is None
    np.testing.assert_allclose(_origin_transform(joint), expected, atol=2e-8, rtol=0.0)


def test_freeze_prismatic_bakes_normalized_axis_translation() -> None:
    joint = _joint(joint_type="prismatic", axis="2 0 0")
    before = _origin_transform(joint)
    expected = before.copy()
    expected[:3, 3] += before[:3, :3] @ np.array((0.125, 0.0, 0.0))

    _freeze_joint(joint, 0.125)

    np.testing.assert_allclose(_origin_transform(joint), expected, atol=2e-8, rtol=0.0)


def test_robolab_galbot_opening_range_is_physical() -> None:
    assert GRIPPER_MIN_OPENING == 0.0
    assert math.isclose(GRIPPER_MAX_OPENING, 0.12490876627340242, abs_tol=1e-12)


def test_arms_only_visual_scope_keeps_arm_and_gripper_shells() -> None:
    robot = ET.fromstring(
        """
        <robot name="scope">
          <link name="torso"><visual/><visual/></link>
          <link name="left_arm_link1"><visual/></link>
          <link name="right_gripper_base_link"><visual/></link>
          <link name="left_wrist_camera_link"><visual/></link>
        </robot>
        """
    )

    assert _strip_non_arm_visuals(robot) == ("torso",)
    assert robot.find("./link[@name='torso']/visual") is None
    assert robot.find("./link[@name='left_arm_link1']/visual") is not None
    assert robot.find("./link[@name='right_gripper_base_link']/visual") is not None
    assert robot.find("./link[@name='left_wrist_camera_link']/visual") is not None
