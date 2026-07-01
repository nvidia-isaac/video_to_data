"""Config for the SimToolReal hammer task with the VEGA + left-Sharpa robot (robot swap only).

Identical task to `HammerEnvCfg` (same claw_hammer tool, prismatic nail/screw, nail-in goals,
reward, obs/action layout) -- the ONLY change is the robot: the IIWA14 arm is replaced by the Vega
humanoid's LEFT 7-DOF arm + LEFT Sharpa hand. The base env reads the robot's joint/palm/fingertip
names from cfg, so this subclass just swaps `robot_cfg` + points those cfg fields at the Vega layout
(see ..vega_sharpa_robot). The original hammer task is untouched.

The IIWA pretrained policy does NOT transfer to a different arm, so `pretrained_compat` is OFF here
(native wxyz convention, USD-derived joint limits) -- this is a train-from-scratch task.
"""

from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from ..hammer.hammer_env_cfg import HammerEnvCfg
from ..vega_sharpa_robot import (
    VEGA_BASE_POS,
    VEGA_FINGERTIP_BODIES,
    VEGA_INIT_JOINT_POS,
    VEGA_JOINT_NAMES,
    VEGA_PALM_BODY,
    VEGA_PALM_OFFSET,
    make_vega_robot_cfg,
    move_scene,
)


@configclass
class VegaHammerEnvCfg(HammerEnvCfg):
    # swap the robot: Vega + dual Sharpa (LEFT arm + LEFT hand drive the task)
    robot_cfg: ArticulationCfg = make_vega_robot_cfg()
    # point the base env's robot identifiers at the Vega left arm + left Sharpa hand
    joint_names: list = VEGA_JOINT_NAMES
    palm_body: str = VEGA_PALM_BODY
    fingertip_bodies: list = VEGA_FINGERTIP_BODIES
    palm_offset: tuple = VEGA_PALM_OFFSET

    def __post_init__(self):
        # run the full hammer-task config (success/goals/cameras/table_dist/genuine-strike guards...)
        super().__post_init__()
        # --- robot swap fix-ups ---
        # different arm -> the IIWA pretrained checkpoint + its compat obs/limits don't apply.
        self.pretrained_compat = False
        # the parent (HammerEnvCfg) __post_init__ wrote the IIWA "startArmHigher" pose onto our
        # robot_cfg (iiwa14_joint_2/4 keys that don't exist on Vega). Restore the clean Vega init
        # (left arm hover over the workspace + right arm tucked down by the side).
        self.robot_cfg.init_state.pos = VEGA_BASE_POS
        self.robot_cfg.init_state.joint_pos = dict(VEGA_INIT_JOINT_POS)
        # --- table + objects (total move from the original): +0.30 up (z), -0.50 in y (front; robot
        #     front is -y). The arm init poses (VEGA_RIGHT_ARM/LEFT_ARM) keep the right hand at the
        #     original hammer hand->table pose (pos+rot) and the left mirrored; re-solved for this table
        #     via scripts/ik_both_arms.py. Recompute those if you move the table again. ---
        move_scene(self, dx=0.0, dy=-0.50, dz=0.30)
