# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field

from isaaclab.assets.articulation import ArticulationCfg

from robotic_grounding.assets.g1 import G1_CYLINDER_MODEL_12_HANDS_DEX_DELAYED_CFG
from robotic_grounding.assets.sharpa_wave import (
    FINGER_JOINTS,
    FINGERTIP_BODY_NAME,
    HAND_CONTACT_BODIES,
    LEFT_SHARPA_WAVE_CFG,
    LEFT_SHARPA_WAVE_PRIMITIVE_CFG,
    RIGHT_SHARPA_WAVE_CFG,
    RIGHT_SHARPA_WAVE_PRIMITIVE_CFG,
    WRIST_BODY_NAME,
    WRIST_JOINTS,
)
from robotic_grounding.assets.vega_sharpa import (
    ARM_JOINT_NAMES as VEGA_ARM_JOINT_NAMES,
)
from robotic_grounding.assets.vega_sharpa import (
    EEF_BODY_NAMES as VEGA_EEF_BODY_NAMES,
)
from robotic_grounding.assets.vega_sharpa import (
    HAND_CONTACT_BODIES as VEGA_HAND_CONTACT_BODIES,
)
from robotic_grounding.assets.vega_sharpa import (
    HAND_CONTACT_BODIES_BY_SIDE as VEGA_HAND_CONTACT_BODIES_BY_SIDE,
)
from robotic_grounding.assets.vega_sharpa import (
    VEGA_SHARPA_SYSID_CFG,
)


@dataclass
class RobotSpec:
    """Everything needed to place a robot into a scene and wire up commands.

    Supports two layouts:
      - Dual floating hands: set left_cfg + right_cfg
      - Single robot (humanoid, mobile manip): set robot_cfg
    """

    # Single robot articulation (whole-body, mobile manip, etc.)
    robot_cfg: ArticulationCfg | None = None

    # Dual floating hands
    left_cfg: ArticulationCfg | None = None
    right_cfg: ArticulationCfg | None = None

    # Dual floating hands with primitive URDFs
    left_primitive_cfg: ArticulationCfg | None = None
    right_primitive_cfg: ArticulationCfg | None = None

    # Hand joint/body names for command wiring
    wrist_joint_names: list[str] = field(default_factory=list)
    finger_joint_names: list[str] = field(default_factory=list)
    wrist_body_name: str = ""
    fingertip_body_name: str = ""
    hand_contact_bodies: list[str] = field(default_factory=list)

    # Per-side contact bodies, when `.*` -> side substitution cannot produce them.
    # Callers build a side's contact-sensor filter paths by replacing `.*` in
    # `hand_contact_bodies` with "left"/"right". That assumes one token for the whole
    # hand, which Vega breaks: its arm links are `L_arm_l7`/`R_arm_l7` while its finger
    # links are `left_thumb_MC`/`right_thumb_MC`. The substituted `left_arm_l7` matches
    # no prim, and PhysX only logs it, so the palm silently stops sensing contact.
    # Populate this to name each side's bodies outright; it wins over the substitution.
    hand_contact_bodies_by_side: dict[str, list[str]] = field(default_factory=dict)

    # Arm joint/body names for whole-body command wiring
    arm_joint_names: list[str] = field(default_factory=list)
    eef_body_names: list[str] = field(default_factory=list)

    @property
    def is_dual_hand(self) -> bool:
        """Whether the robot has separate left and right hand configs."""
        return self.left_cfg is not None and self.right_cfg is not None

    def contact_bodies_for_side(self, side: str) -> list[str]:
        """Concrete contact-sensor body names for ``side`` ("left" / "right")."""
        explicit = self.hand_contact_bodies_by_side.get(side)
        if explicit:
            return list(explicit)
        return [b.replace(".*", side) for b in self.hand_contact_bodies]


ROBOT_REGISTRY: dict[str, RobotSpec] = {
    "sharpa_wave": RobotSpec(
        left_cfg=LEFT_SHARPA_WAVE_CFG,
        right_cfg=RIGHT_SHARPA_WAVE_CFG,
        left_primitive_cfg=LEFT_SHARPA_WAVE_PRIMITIVE_CFG,
        right_primitive_cfg=RIGHT_SHARPA_WAVE_PRIMITIVE_CFG,
        wrist_joint_names=WRIST_JOINTS,
        finger_joint_names=FINGER_JOINTS,
        wrist_body_name=WRIST_BODY_NAME,
        fingertip_body_name=FINGERTIP_BODY_NAME,
        hand_contact_bodies=HAND_CONTACT_BODIES,
    ),
    # Whole-body retarget (soma_to_g1.py) uses main_with_hand.urdf — keep asset in sync.
    "g1": RobotSpec(
        robot_cfg=G1_CYLINDER_MODEL_12_HANDS_DEX_DELAYED_CFG,
        wrist_joint_names=[],
        finger_joint_names=[],
        wrist_body_name="",
        fingertip_body_name="",
        hand_contact_bodies=[],
    ),
    # Vega mobile-manipulator base + Sharpa Wave hands (single whole-body articulation).
    # Named `vega_sharpa` to keep it distinct from `sharpa_wave`, the floating-hand-only
    # embodiment that the same source clips are also retargeted to.
    "vega_sharpa": RobotSpec(
        robot_cfg=VEGA_SHARPA_SYSID_CFG,
        arm_joint_names=VEGA_ARM_JOINT_NAMES,
        finger_joint_names=FINGER_JOINTS,
        eef_body_names=VEGA_EEF_BODY_NAMES,
        fingertip_body_name=FINGERTIP_BODY_NAME,
        hand_contact_bodies=VEGA_HAND_CONTACT_BODIES,
        hand_contact_bodies_by_side=VEGA_HAND_CONTACT_BODIES_BY_SIDE,
    ),
}


def get_robot_spec(name: str) -> RobotSpec | None:
    """Return the robot spec for the given name, or None if not found."""
    return ROBOT_REGISTRY.get(name)
