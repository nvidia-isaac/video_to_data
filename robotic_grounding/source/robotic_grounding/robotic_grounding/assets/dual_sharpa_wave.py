# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robotic_grounding.assets import ASSET_DIR

WRIST_ARMATURE = 0.001
WRIST_FRICTION = 0.001
WRIST_EFFORT_LIMIT = 80.0
WRIST_VELOCITY_LIMIT = 16.0

WRIST_SLIDE_STIFFNESS = 60.0
WRIST_SLIDE_DAMPING = 3.0

WRIST_REVOLUTE_STIFFNESS = 60.0
WRIST_REVOLUTE_DAMPING = 3.0

FINGER_STIFFNESS = 1.74533
FINGER_DAMPING = 0.01745
FINGER_VELOCITY_LIMIT = 11.62

MCP_ARMATURE = 0.00265
MCP_FRICTION = 0.07456
MCP_EFFORT_LIMIT = 1.864

PIP_ARMATURE = 0.0006
PIP_FRICTION = 0.01276
PIP_EFFORT_LIMIT = 0.638

DIP_ARMATURE = 0.00042
DIP_FRICTION = 0.00378738
DIP_EFFORT_LIMIT = 0.18937

CMC_ARMATURE = 0.0032
CMC_FRICTION = 0.132
CMC_EFFORT_LIMIT = 3.3

PCMC_ARMATURE = 0.00012
PCMC_FRICTION = 0.012
PCMC_EFFORT_LIMIT = 0.5285

DUAL_SHARPA_WAVE_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=True,
        asset_path=f"{ASSET_DIR}/urdfs/dual_sharpa_wave.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            enable_gyroscopic_forces=False,
            linear_damping=0.0,
            angular_damping=0.01,
            max_linear_velocity=1000.0,
            max_angular_velocity=64 / math.pi * 180.0,
            max_depenetration_velocity=1000.0,
            max_contact_impulse=1e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0, damping=0
            )
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "right_wrist_y": 0.2,
            "left_wrist_y": -0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "wrists": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_wrist_x",
                ".*_wrist_y",
                ".*_wrist_z",
                ".*_wrist_roll",
                ".*_wrist_pitch",
                ".*_wrist_yaw",
            ],
            effort_limit_sim={
                ".*_wrist_x": WRIST_EFFORT_LIMIT,
                ".*_wrist_y": WRIST_EFFORT_LIMIT,
                ".*_wrist_z": WRIST_EFFORT_LIMIT,
                ".*_wrist_roll": WRIST_EFFORT_LIMIT,
                ".*_wrist_pitch": WRIST_EFFORT_LIMIT,
                ".*_wrist_yaw": WRIST_EFFORT_LIMIT,
            },
            velocity_limit_sim={
                ".*_wrist_x": WRIST_VELOCITY_LIMIT,
                ".*_wrist_y": WRIST_VELOCITY_LIMIT,
                ".*_wrist_z": WRIST_VELOCITY_LIMIT,
                ".*_wrist_roll": WRIST_VELOCITY_LIMIT,
                ".*_wrist_pitch": WRIST_VELOCITY_LIMIT,
                ".*_wrist_yaw": WRIST_VELOCITY_LIMIT,
            },
            stiffness={
                ".*_wrist_x": WRIST_SLIDE_STIFFNESS,
                ".*_wrist_y": WRIST_SLIDE_STIFFNESS,
                ".*_wrist_z": WRIST_SLIDE_STIFFNESS,
                ".*_wrist_roll": WRIST_REVOLUTE_STIFFNESS,
                ".*_wrist_pitch": WRIST_REVOLUTE_STIFFNESS,
                ".*_wrist_yaw": WRIST_REVOLUTE_STIFFNESS,
            },
            damping={
                ".*_wrist_x": WRIST_SLIDE_DAMPING,
                ".*_wrist_y": WRIST_SLIDE_DAMPING,
                ".*_wrist_z": WRIST_SLIDE_DAMPING,
                ".*_wrist_roll": WRIST_REVOLUTE_DAMPING,
                ".*_wrist_pitch": WRIST_REVOLUTE_DAMPING,
                ".*_wrist_yaw": WRIST_REVOLUTE_DAMPING,
            },
            armature={
                ".*_wrist_x": WRIST_ARMATURE,
                ".*_wrist_y": WRIST_ARMATURE,
                ".*_wrist_z": WRIST_ARMATURE,
                ".*_wrist_roll": WRIST_ARMATURE,
                ".*_wrist_pitch": WRIST_ARMATURE,
                ".*_wrist_yaw": WRIST_ARMATURE,
            },
            friction={
                ".*_wrist_x": WRIST_FRICTION,
                ".*_wrist_y": WRIST_FRICTION,
                ".*_wrist_z": WRIST_FRICTION,
                ".*_wrist_roll": WRIST_FRICTION,
                ".*_wrist_pitch": WRIST_FRICTION,
                ".*_wrist_yaw": WRIST_FRICTION,
            },
        ),
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_CMC_.*",  # thumb CMC
                ".*_pinky_CMC",  # pinky CMC
                ".*_MCP_.*",
                ".*_IP",
                ".*_PIP",
                ".*_DIP",
            ],
            effort_limit_sim={
                ".*_CMC_.*": CMC_EFFORT_LIMIT,
                ".*_pinky_CMC": PCMC_EFFORT_LIMIT,
                ".*_MCP_.*": MCP_EFFORT_LIMIT,
                ".*_IP": PIP_EFFORT_LIMIT,
                ".*_PIP": PIP_EFFORT_LIMIT,
                ".*_DIP": DIP_EFFORT_LIMIT,
            },
            velocity_limit_sim={
                ".*_CMC_.*": FINGER_VELOCITY_LIMIT,
                ".*_pinky_CMC": FINGER_VELOCITY_LIMIT,
                ".*_MCP_.*": FINGER_VELOCITY_LIMIT,
                ".*_IP": FINGER_VELOCITY_LIMIT,
                ".*_PIP": FINGER_VELOCITY_LIMIT,
                ".*_DIP": FINGER_VELOCITY_LIMIT,
            },
            stiffness={
                ".*_CMC_.*": FINGER_STIFFNESS,
                ".*_pinky_CMC": FINGER_STIFFNESS,
                ".*_MCP_.*": FINGER_STIFFNESS,
                ".*_IP": FINGER_STIFFNESS,
                ".*_PIP": FINGER_STIFFNESS,
                ".*_DIP": FINGER_STIFFNESS,
            },
            damping={
                ".*_CMC_.*": FINGER_DAMPING,
                ".*_pinky_CMC": FINGER_DAMPING,
                ".*_MCP_.*": FINGER_DAMPING,
                ".*_IP": FINGER_DAMPING,
                ".*_PIP": FINGER_DAMPING,
                ".*_DIP": FINGER_DAMPING,
            },
            armature={
                ".*_CMC_.*": CMC_ARMATURE,
                ".*_pinky_CMC": PCMC_ARMATURE,
                ".*_MCP_.*": MCP_ARMATURE,
                ".*_IP": PIP_ARMATURE,
                ".*_PIP": PIP_ARMATURE,
                ".*_DIP": DIP_ARMATURE,
            },
            friction={
                ".*_CMC_.*": CMC_FRICTION,
                ".*_pinky_CMC": PCMC_FRICTION,
                ".*_MCP_.*": MCP_FRICTION,
                ".*_IP": PIP_FRICTION,
                ".*_PIP": PIP_FRICTION,
                ".*_DIP": DIP_FRICTION,
            },
        ),
    },
)

DUAL_SHARPA_WAVE_ACTION_SCALE = {}
for a in DUAL_SHARPA_WAVE_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            DUAL_SHARPA_WAVE_ACTION_SCALE[n] = 0.25 * e[n] / s[n]

ISAAC_JOINT_ORDER = [
    "left_wrist_x",
    "right_wrist_x",
    "left_wrist_y",
    "right_wrist_y",
    "left_wrist_z",
    "right_wrist_z",
    "left_wrist_roll",
    "right_wrist_roll",
    "left_wrist_pitch",
    "right_wrist_pitch",
    "left_wrist_yaw",
    "right_wrist_yaw",
    "left_index_MCP_FE",
    "left_middle_MCP_FE",
    "left_pinky_CMC",
    "left_ring_MCP_FE",
    "left_thumb_CMC_FE",
    "right_index_MCP_FE",
    "right_middle_MCP_FE",
    "right_pinky_CMC",
    "right_ring_MCP_FE",
    "right_thumb_CMC_FE",
    "left_index_MCP_AA",
    "left_middle_MCP_AA",
    "left_pinky_MCP_FE",
    "left_ring_MCP_AA",
    "left_thumb_CMC_AA",
    "right_index_MCP_AA",
    "right_middle_MCP_AA",
    "right_pinky_MCP_FE",
    "right_ring_MCP_AA",
    "right_thumb_CMC_AA",
    "left_index_PIP",
    "left_middle_PIP",
    "left_pinky_MCP_AA",
    "left_ring_PIP",
    "left_thumb_MCP_FE",
    "right_index_PIP",
    "right_middle_PIP",
    "right_pinky_MCP_AA",
    "right_ring_PIP",
    "right_thumb_MCP_FE",
    "left_index_DIP",
    "left_middle_DIP",
    "left_pinky_PIP",
    "left_ring_DIP",
    "left_thumb_MCP_AA",
    "right_index_DIP",
    "right_middle_DIP",
    "right_pinky_PIP",
    "right_ring_DIP",
    "right_thumb_MCP_AA",
    "left_pinky_DIP",
    "left_thumb_IP",
    "right_pinky_DIP",
    "right_thumb_IP",
]

MUJOCO_JOINT_ORDER = [
    ".*_wrist_x",
    ".*_wrist_y",
    ".*_wrist_z",
    ".*_wrist_roll",
    ".*_wrist_pitch",
    ".*_wrist_yaw",
    ".*_thumb_CMC_FE",
    ".*_thumb_CMC_AA",
    ".*_thumb_MCP_FE",
    ".*_thumb_MCP_AA",
    ".*_thumb_IP",
    ".*_index_MCP_FE",
    ".*_index_MCP_AA",
    ".*_index_PIP",
    ".*_index_DIP",
    ".*_middle_MCP_FE",
    ".*_middle_MCP_AA",
    ".*_middle_PIP",
    ".*_middle_DIP",
    ".*_ring_MCP_FE",
    ".*_ring_MCP_AA",
    ".*_ring_PIP",
    ".*_ring_DIP",
    ".*_pinky_CMC",
    ".*_pinky_MCP_FE",
    ".*_pinky_MCP_AA",
    ".*_pinky_PIP",
    ".*_pinky_DIP",
]

MUJOCO_RIGHT_JOINT_ORDER = [n.replace(".*_", "right_") for n in MUJOCO_JOINT_ORDER]
MUJOCO_LEFT_JOINT_ORDER = [n.replace(".*_", "left_") for n in MUJOCO_JOINT_ORDER]

# Fingertip contact body metadata
# These are the rigid body names where contact sensors should be attached
# Note: URDF importer merges elastomer links into parent distal phalange (_DP) bodies
FINGERTIP_CONTACT_BODIES = [
    "left_thumb_DP",
    "left_index_DP",
    "left_middle_DP",
    "left_ring_DP",
    "left_pinky_DP",
    "right_thumb_DP",
    "right_index_DP",
    "right_middle_DP",
    "right_ring_DP",
    "right_pinky_DP",
]
