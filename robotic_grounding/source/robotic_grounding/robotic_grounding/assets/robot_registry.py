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

    @property
    def is_dual_hand(self) -> bool:
        """Whether the robot has separate left and right hand configs."""
        return self.left_cfg is not None and self.right_cfg is not None


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
}


def get_robot_spec(name: str) -> RobotSpec | None:
    """Return the robot spec for the given name, or None if not found."""
    return ROBOT_REGISTRY.get(name)
