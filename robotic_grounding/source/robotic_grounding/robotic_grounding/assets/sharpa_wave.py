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

use_primitive_urdfs = False

#################################################
# Control Parameters
#################################################

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

sharpa_wave_urdf_dir = f"{ASSET_DIR}/urdfs/sharpawave"
dual_sharpa_wave_urdf_path = f"{sharpa_wave_urdf_dir}/dual_sharpa_wave.urdf"
right_sharpa_wave_urdf_path = f"{sharpa_wave_urdf_dir}/right_sharpa_wave.urdf"
left_sharpa_wave_urdf_path = f"{sharpa_wave_urdf_dir}/left_sharpa_wave.urdf"
right_sharpa_wave_primitive_urdf_path = (
    f"{sharpa_wave_urdf_dir}/right_sharpa_wave_primitive.urdf"
)
left_sharpa_wave_primitive_urdf_path = (
    f"{sharpa_wave_urdf_dir}/left_sharpa_wave_primitive.urdf"
)

rigid_props = sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=False,
    retain_accelerations=False,
    enable_gyroscopic_forces=False,
    linear_damping=0.01,
    angular_damping=0.01,
    max_linear_velocity=1000.0,
    max_angular_velocity=64 / math.pi * 180.0,
    max_depenetration_velocity=1.0,
    max_contact_impulse=1e3,
)

articulation_props = sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=True,
    solver_position_iteration_count=8,
    solver_velocity_iteration_count=0,
    sleep_threshold=0.005,
    stabilization_threshold=0.0005,
)

none_joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
    gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
)

finger_actuators = ImplicitActuatorCfg(
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
)

#################################################
# Dual Sharpa Wave with shared base link
#################################################

DUAL_SHARPA_WAVE_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=True,
        asset_path=dual_sharpa_wave_urdf_path,
        activate_contact_sensors=True,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
        joint_drive=none_joint_drive,
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
        "fingers": finger_actuators,
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

#################################################
# Single Sharpa Wave
#################################################

RIGHT_SHARPA_WAVE_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        asset_path=(
            right_sharpa_wave_primitive_urdf_path
            if use_primitive_urdfs
            else right_sharpa_wave_urdf_path
        ),
        activate_contact_sensors=True,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
        joint_drive=none_joint_drive,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "fingers": finger_actuators,
    },
)

LEFT_SHARPA_WAVE_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        asset_path=(
            left_sharpa_wave_primitive_urdf_path
            if use_primitive_urdfs
            else left_sharpa_wave_urdf_path
        ),
        activate_contact_sensors=True,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
        joint_drive=none_joint_drive,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "fingers": finger_actuators,
    },
)

#################################################
# Parameters
#################################################

WRIST_JOINTS = [
    ".*_wrist_x",
    ".*_wrist_y",
    ".*_wrist_z",
    ".*_wrist_roll",
    ".*_wrist_pitch",
    ".*_wrist_yaw",
]

FINGER_JOINTS = [
    ".*_index_MCP_FE",
    ".*_middle_MCP_FE",
    ".*_pinky_CMC",
    ".*_ring_MCP_FE",
    ".*_thumb_CMC_FE",
    ".*_index_MCP_AA",
    ".*_middle_MCP_AA",
    ".*_pinky_MCP_FE",
    ".*_ring_MCP_AA",
    ".*_thumb_CMC_AA",
    ".*_index_PIP",
    ".*_middle_PIP",
    ".*_pinky_MCP_AA",
    ".*_ring_PIP",
    ".*_thumb_MCP_FE",
    ".*_index_DIP",
    ".*_middle_DIP",
    ".*_pinky_PIP",
    ".*_ring_DIP",
    ".*_thumb_MCP_AA",
    ".*_pinky_DIP",
    ".*_thumb_IP",
]

WRIST_BODY_NAME = ".*_hand_C_MC"
FINGERTIP_BODY_NAME = ".*_DP"

# All links in the hand with collision geometry (palm, phalanges).
# Excludes *_fingertip: no collision in URDF, so not created as rigid bodies.
# Excludes *_elastomer: URDF importer merges them into parent _DP bodies.
HAND_CONTACT_BODIES = [
    # Palm
    ".*_hand_C_MC",  # "C_MC" stands for "Central Metacarpal"; palm/base link
    # Thumb
    ".*_thumb_PP",
    ".*_thumb_DP",
    # Index
    ".*_index_PP",
    ".*_index_MP",
    ".*_index_DP",
    # Middle
    ".*_middle_PP",
    ".*_middle_MP",
    ".*_middle_DP",
    # Ring
    ".*_ring_PP",
    ".*_ring_MP",
    ".*_ring_DP",
    # Pinky
    ".*_pinky_MC",
    ".*_pinky_PP",
    ".*_pinky_MP",
    ".*_pinky_DP",
]
