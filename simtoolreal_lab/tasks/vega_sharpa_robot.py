"""Shared Vega + dual-Sharpa robot definition for the SimToolReal robot-SWAP tasks.

The Vega humanoid (fixed base, dual 7-DOF arms + dual 22-DOF Sharpa hands) drops into the SAME
SimToolReal obs/action/reward/goal pipeline as the IIWA14 + left-Sharpa robot, using the LEFT arm
(`L_arm_j1..7`) + the LEFT Sharpa hand. Because the base env (`simtoolreal_env.py`) now reads its
robot identifiers from cfg (cfg.joint_names / palm_body / fingertip_bodies / palm_offset; defaulting
to the original IIWA constants), the Vega tasks just point those at this layout -- no env changes.

The left Sharpa hand is the SAME morphology as the original; only the joint names differ (the Vega
URDF drops the `left_{1..5}_` numeric prefix). So the hand actuator gains are the original Sharpa
gains, re-keyed to the un-prefixed names (1:1, same canonical order).

The right arm + right hand + torso are NOT controlled by the task -- the env's reset writes every
joint to its default and `set_joint_position_target(default)`, and `_apply_action` only re-targets
the canonical 29 (left arm + left hand). A holding actuator on the rest keeps them at the default
pose (robot gravity is disabled, as for the IIWA).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from .simtoolreal.robot_gains import (
    HAND_ARMATURE, HAND_DAMPING, HAND_EFFORT, HAND_FRICTION, HAND_JOINTS, HAND_STIFFNESS,
)

VEGA_ROBOT_USD = "/home/cning/simtoolreal_isaaclab/assets/usd/vega_sharpa/robot.usd"

# --- canonical 29 joints (arm 0:7, hand 7:29) in the SimToolReal order ----------------------------
# Arm = Vega LEFT arm. Hand = the left Sharpa hand, SAME order as JOINT_NAMES_ISAACGYM's hand block
# but with the `left_{1..5}_` prefix dropped (the only naming difference vs the IIWA Sharpa).
VEGA_ARM_JOINTS = [f"L_arm_j{i}" for i in range(1, 8)]
VEGA_HAND_JOINTS = [
    "left_thumb_CMC_FE", "left_thumb_CMC_AA", "left_thumb_MCP_FE", "left_thumb_MCP_AA", "left_thumb_IP",
    "left_index_MCP_FE", "left_index_MCP_AA", "left_index_PIP", "left_index_DIP",
    "left_middle_MCP_FE", "left_middle_MCP_AA", "left_middle_PIP", "left_middle_DIP",
    "left_ring_MCP_FE", "left_ring_MCP_AA", "left_ring_PIP", "left_ring_DIP",
    "left_pinky_CMC", "left_pinky_MCP_FE", "left_pinky_MCP_AA", "left_pinky_PIP", "left_pinky_DIP",
]
VEGA_JOINT_NAMES = VEGA_ARM_JOINTS + VEGA_HAND_JOINTS  # 29

# palm/wrist body (where L_hand_mount attaches the hand -- analogous to iiwa14_link_7) + fingertips
VEGA_PALM_BODY = "L_arm_l7"
VEGA_FINGERTIP_BODIES = ["left_index_DP", "left_middle_DP", "left_ring_DP", "left_thumb_DP", "left_pinky_DP"]

# fixed-base root. x=0 centers the robot's body/face on the table center (table sits at world x=0;
# the shoulder-midpoint + wheeled base both sit at root x). The robot's FRONT is -y (its base/face
# point -y; +y is the back) -> right side is +x. y keeps it by the table; z on the floor.
VEGA_BASE_POS = (0.0, -0.531, 0.0)

# --- arm init poses (solved 6-DOF with scripts/ik_both_arms.py): the RIGHT hand is placed at the
#     ORIGINAL hammer hand->table POSE -- BOTH the position offset AND the orientation (measured in sim;
#     matched to <1 mm / <0.1 deg). "Hand pose" is hand-intrinsic (fingertip-centroid + a frame from the
#     fingertips), so it's robust to the wrist/mount differences vs the IIWA. The LEFT arm is the MIRROR
#     of the right across the robot's center plane (x=0). Targets are relative to the CURRENT (moved)
#     table; if the table is moved again, recompute via scripts/ik_both_arms.py. ---
# RIGHT hand: at the kept hand->table position (spread 0.1 m to the robot's right so the hands don't
# collide), oriented PALM-DOWN (palm normal -> -z; wrist yaw free). LEFT arm = EXACT joint mirror of
# the right: q_L = SIGN * q_R with SIGN=[-1,-1,-1,+1,-1,-1,-1] (from the URDF L/R arm axis analysis;
# only j4 keeps sign) -> a true left-right mirror pose. Solved via scripts/ik_palmdown_mirror.py.
VEGA_RIGHT_ARM = {
    "R_arm_j1": -1.4659, "R_arm_j2": -0.9913, "R_arm_j3": -0.8333, "R_arm_j4": -2.7730,
    "R_arm_j5": -1.6266, "R_arm_j6": -1.0706, "R_arm_j7": 0.6680,
}
VEGA_LEFT_ARM = {    # = [-1,-1,-1,+1,-1,-1,-1] * VEGA_RIGHT_ARM (exact joint mirror)
    "L_arm_j1": 1.4659, "L_arm_j2": 0.9913, "L_arm_j3": 0.8333, "L_arm_j4": -2.7730,
    "L_arm_j5": 1.6266, "L_arm_j6": 1.0706, "L_arm_j7": -0.6680,
}
# full init pose written to the robot; the hands keep their USD default (0 = open).
VEGA_INIT_JOINT_POS = {**VEGA_LEFT_ARM, **VEGA_RIGHT_ARM}
# palm-center reference in the L_arm_l7 frame: ~0.15 m out along the wrist toward the palm (the hand
# extends along L_arm_l7's +x; cf. the IIWA's 0.16 m along its tool axis). Only a (consistent) obs
# reference origin for fingertip/keypoint-rel-palm, so the exact value is not critical.
VEGA_PALM_OFFSET = (0.15, 0.0, -0.012)

# --- arm position-control gains (Vega arm). STIFFENED ~3x (stiffness)/~2.5x (damping)/~2x (effort): during
# ACTIVE motion this tracks the retarget IK command ~3x better (palm EE err median 6.9mm vs 20.6mm for the
# original soft gains, same seeds). MUST pair with DLS lambda_val>=0.05 -- stiff gains + aggressive IK
# (lambda 0.01-0.02) OSCILLATE. (The residual ~32mm object->goal gap is grasp/embodiment, not arm tracking.)
_VEGA_ARM_STIFF = [900.0, 900.0, 600.0, 600.0, 240.0, 240.0, 180.0]
_VEGA_ARM_DAMP = [75.0, 75.0, 50.0, 50.0, 22.0, 22.0, 18.0]
_VEGA_ARM_EFFORT = [300.0, 300.0, 160.0, 160.0, 60.0, 60.0, 60.0]
VEGA_ARM_STIFFNESS = dict(zip(VEGA_ARM_JOINTS, _VEGA_ARM_STIFF))
VEGA_ARM_DAMPING = dict(zip(VEGA_ARM_JOINTS, _VEGA_ARM_DAMP))
VEGA_ARM_EFFORT = dict(zip(VEGA_ARM_JOINTS, _VEGA_ARM_EFFORT))

# --- LEFT hand gains = the original Sharpa gains re-keyed to the un-prefixed Vega names (1:1, same
#     order as HAND_JOINTS), so the hand behaves exactly as on the IIWA. ----------------------------
def _rekey(d):
    return dict(zip(VEGA_HAND_JOINTS, [d[j] for j in HAND_JOINTS]))


VEGA_HAND_STIFFNESS = _rekey(HAND_STIFFNESS)
VEGA_HAND_DAMPING = _rekey(HAND_DAMPING)
VEGA_HAND_ARMATURE = _rekey(HAND_ARMATURE)
VEGA_HAND_FRICTION = _rekey(HAND_FRICTION)
VEGA_HAND_EFFORT = _rekey(HAND_EFFORT)


# --- RIGHT-hand canonical (for the pretrained-policy retarget: the policy drives the right hand) -----
VEGA_RIGHT_ARM_JOINTS = [f"R_arm_j{i}" for i in range(1, 8)]
VEGA_RIGHT_HAND_JOINTS = [n.replace("left_", "right_", 1) for n in VEGA_HAND_JOINTS]
VEGA_RIGHT_JOINT_NAMES = VEGA_RIGHT_ARM_JOINTS + VEGA_RIGHT_HAND_JOINTS  # 29
VEGA_RIGHT_PALM_BODY = "R_arm_l7"
VEGA_RIGHT_FINGERTIP_BODIES = [b.replace("left_", "right_", 1) for b in VEGA_FINGERTIP_BODIES]
_VEGA_RIGHT_ARM_STIFFNESS = dict(zip(VEGA_RIGHT_ARM_JOINTS, _VEGA_ARM_STIFF))
_VEGA_RIGHT_ARM_DAMPING = dict(zip(VEGA_RIGHT_ARM_JOINTS, _VEGA_ARM_DAMP))
_VEGA_RIGHT_ARM_EFFORT = dict(zip(VEGA_RIGHT_ARM_JOINTS, _VEGA_ARM_EFFORT))
_VEGA_RIGHT_HAND_STIFFNESS = dict(zip(VEGA_RIGHT_HAND_JOINTS, [HAND_STIFFNESS[j] for j in HAND_JOINTS]))
_VEGA_RIGHT_HAND_DAMPING = dict(zip(VEGA_RIGHT_HAND_JOINTS, [HAND_DAMPING[j] for j in HAND_JOINTS]))
_VEGA_RIGHT_HAND_ARMATURE = dict(zip(VEGA_RIGHT_HAND_JOINTS, [HAND_ARMATURE[j] for j in HAND_JOINTS]))
_VEGA_RIGHT_HAND_FRICTION = dict(zip(VEGA_RIGHT_HAND_JOINTS, [HAND_FRICTION[j] for j in HAND_JOINTS]))
_VEGA_RIGHT_HAND_EFFORT = dict(zip(VEGA_RIGHT_HAND_JOINTS, [HAND_EFFORT[j] for j in HAND_JOINTS]))


def make_vega_robot_cfg_right() -> ArticulationCfg:
    """Like make_vega_robot_cfg but the RIGHT arm+hand are the CONTROLLED group (the retarget drives
    them via the mirrored pretrained policy) and the LEFT arm+hand are HELD/parked."""
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=VEGA_ROBOT_USD, activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, retain_accelerations=False),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=VEGA_BASE_POS, joint_pos=dict({**VEGA_LEFT_ARM, **VEGA_RIGHT_ARM}),
        ),
        actuators={
            "right_arm": ImplicitActuatorCfg(joint_names_expr=["R_arm_j[1-7]"],
                stiffness=_VEGA_RIGHT_ARM_STIFFNESS, damping=_VEGA_RIGHT_ARM_DAMPING, effort_limit=_VEGA_RIGHT_ARM_EFFORT),
            "right_hand": ImplicitActuatorCfg(joint_names_expr=["right_.*"],
                stiffness=_VEGA_RIGHT_HAND_STIFFNESS, damping=_VEGA_RIGHT_HAND_DAMPING,
                effort_limit=_VEGA_RIGHT_HAND_EFFORT, armature=_VEGA_RIGHT_HAND_ARMATURE, friction=_VEGA_RIGHT_HAND_FRICTION),
            "held": ImplicitActuatorCfg(joint_names_expr=["L_arm_j[1-7]", "left_.*"], stiffness=200.0, damping=20.0),
        },
    )


def make_vega_robot_cfg_bimanual() -> ArticulationCfg:
    """BOTH arms + BOTH hands CONTROLLED with proper gains (nothing parked). For the bimanual retarget:
    the RIGHT arm+hand run the mirrored hammer policy, the LEFT arm+hand run the (un-mirrored) policy to
    hold the thread_tester. Same init pose as the right-only cfg."""
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=VEGA_ROBOT_USD, activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, retain_accelerations=False),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=VEGA_BASE_POS, joint_pos=dict({**VEGA_LEFT_ARM, **VEGA_RIGHT_ARM}),
        ),
        actuators={
            "right_arm": ImplicitActuatorCfg(joint_names_expr=["R_arm_j[1-7]"],
                stiffness=_VEGA_RIGHT_ARM_STIFFNESS, damping=_VEGA_RIGHT_ARM_DAMPING, effort_limit=_VEGA_RIGHT_ARM_EFFORT),
            "right_hand": ImplicitActuatorCfg(joint_names_expr=["right_.*"],
                stiffness=_VEGA_RIGHT_HAND_STIFFNESS, damping=_VEGA_RIGHT_HAND_DAMPING,
                effort_limit=_VEGA_RIGHT_HAND_EFFORT, armature=_VEGA_RIGHT_HAND_ARMATURE, friction=_VEGA_RIGHT_HAND_FRICTION),
            "left_arm": ImplicitActuatorCfg(joint_names_expr=["L_arm_j[1-7]"],
                stiffness=VEGA_ARM_STIFFNESS, damping=VEGA_ARM_DAMPING, effort_limit=VEGA_ARM_EFFORT),
            "left_hand": ImplicitActuatorCfg(joint_names_expr=["left_.*"],
                stiffness=VEGA_HAND_STIFFNESS, damping=VEGA_HAND_DAMPING,
                effort_limit=VEGA_HAND_EFFORT, armature=VEGA_HAND_ARMATURE, friction=VEGA_HAND_FRICTION),
        },
    )


def move_scene(cfg, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0):
    """Translate the work-table + manipulated objects together by a WORLD delta (dx, dy, dz) m.

    The layout randomizer places objects from `cfg.layout_*_range` (xy) and the object INIT-Z (it
    keeps each object's init z), so: shift the table fully, shift the placement ranges in xy, and
    raise the object init-z. The objects' init-xy and the rotation pivot are left alone (the group's
    shape is defined relative to the pivot, so shifting the ranges alone translates it). Call AFTER
    super().__post_init__() so the task-specific table/object/layout values exist."""
    tx, ty, tz = cfg.table_cfg.init_state.pos
    cfg.table_cfg.init_state.pos = (tx + dx, ty + dy, tz + dz)
    for name in ("object_cfg", "screw_cfg", "thread_test_cfg", "screw_asm_cfg"):
        rc = getattr(cfg, name, None)
        if rc is not None:
            ox, oy, oz = rc.init_state.pos
            rc.init_state.pos = (ox, oy, oz + dz)   # z follows; layout sets xy from the ranges below
    cfg.layout_threadtest_center_x_range = (cfg.layout_threadtest_center_x_range[0] + dx, cfg.layout_threadtest_center_x_range[1] + dx)
    cfg.layout_threadtest_center_y_range = (cfg.layout_threadtest_center_y_range[0] + dy, cfg.layout_threadtest_center_y_range[1] + dy)
    cfg.layout_screwdriver_x_range = (cfg.layout_screwdriver_x_range[0] + dx, cfg.layout_screwdriver_x_range[1] + dx)
    cfg.layout_screwdriver_y_range = (cfg.layout_screwdriver_y_range[0] + dy, cfg.layout_screwdriver_y_range[1] + dy)


def make_vega_robot_cfg() -> ArticulationCfg:
    """Vega + dual-Sharpa ArticulationCfg for the SimToolReal pipeline (fixed base baked in the USD).

    Actuator groups: LEFT arm (control) + LEFT hand (control, exact Sharpa gains) drive the task;
    the RIGHT arm/hand + any remaining body joints are HELD at their default pose so they stay put
    (gravity is disabled, matching the IIWA robot_cfg)."""
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=VEGA_ROBOT_USD,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, retain_accelerations=False,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8, solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=VEGA_BASE_POS,
            joint_pos=dict(VEGA_INIT_JOINT_POS),  # left arm hover + right arm tucked; rest keep USD default (0)
        ),
        actuators={
            "left_arm": ImplicitActuatorCfg(
                joint_names_expr=["L_arm_j[1-7]"],
                stiffness=VEGA_ARM_STIFFNESS, damping=VEGA_ARM_DAMPING, effort_limit=VEGA_ARM_EFFORT,
            ),
            "left_hand": ImplicitActuatorCfg(
                joint_names_expr=["left_.*"],
                stiffness=VEGA_HAND_STIFFNESS, damping=VEGA_HAND_DAMPING,
                effort_limit=VEGA_HAND_EFFORT, armature=VEGA_HAND_ARMATURE, friction=VEGA_HAND_FRICTION,
            ),
            # HOLD the uncontrolled side at its default pose (right arm + right hand). Stiff enough to
            # stay put; the env never re-targets these (only the canonical left 29).
            "held": ImplicitActuatorCfg(
                joint_names_expr=["R_arm_j[1-7]", "right_.*"],
                stiffness=200.0, damping=20.0,
            ),
        },
    )
