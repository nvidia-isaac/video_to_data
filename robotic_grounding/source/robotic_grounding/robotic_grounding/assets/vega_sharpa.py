# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vega Sharpa configuration."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robotic_grounding.assets import ASSET_DIR

# Body joint parameters from URDF
WHEEL_STEER_EFFORT_LIMIT = 6.0
WHEEL_STEER_VELOCITY_LIMIT = 3.0
WHEEL_DRIVE_EFFORT_LIMIT = 16.0
WHEEL_DRIVE_VELOCITY_LIMIT = 12.0
TORSO_J1_EFFORT_LIMIT = 700.0
TORSO_J2_J3_EFFORT_LIMIT = 380.0
TORSO_VELOCITY_LIMIT = 0.9
HEAD_J1_J3_EFFORT_LIMIT = 6.0
HEAD_J2_EFFORT_LIMIT = 2.5
HEAD_VELOCITY_LIMIT = 3.2
ARM_SHOULDER_EFFORT_LIMIT = 150.0
ARM_SHOULDER_VELOCITY_LIMIT = 2.4
ARM_ELBOW_EFFORT_LIMIT = 80.0
ARM_ELBOW_VELOCITY_LIMIT = 2.7
ARM_WRIST_EFFORT_LIMIT = 25.0
ARM_WRIST_VELOCITY_LIMIT = 2.7

# Body actuator dynamics
DEFAULT_ARMATURE = 0.001  # generic estimate, update with sysid
NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

DEFAULT_BODY_STIFFNESS = DEFAULT_ARMATURE * NATURAL_FREQ**2
DEFAULT_BODY_DAMPING = 2.0 * DAMPING_RATIO * DEFAULT_ARMATURE * NATURAL_FREQ

# Wheel damping: set very high based on isaacsim examples
WHEEL_STEER_STIFFNESS = 1e6
WHEEL_STEER_DAMPING = 1e6
WHEEL_DRIVE_DAMPING = 1e6

# Sharpa hand parameters (from dual_sharpa_wave.py)
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

VEGA_SHARPA_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        asset_path=f"{ASSET_DIR}/urdfs/vega_sharpa/vega_sharpa.urdf",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=32,
            stabilization_threshold=0.00005,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0, damping=0
            )
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.056),
        rot=(1.0, 0.0, 0.0, 0.0),
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "wheel_steer": ImplicitActuatorCfg(
            joint_names_expr=["R_wheel_j1", "L_wheel_j1"],
            effort_limit_sim=WHEEL_STEER_EFFORT_LIMIT,
            velocity_limit_sim=WHEEL_STEER_VELOCITY_LIMIT,
            stiffness=WHEEL_STEER_STIFFNESS,
            damping=WHEEL_STEER_DAMPING,
        ),
        "wheel_drive": ImplicitActuatorCfg(
            joint_names_expr=["R_wheel_j2", "L_wheel_j2"],
            effort_limit_sim=WHEEL_DRIVE_EFFORT_LIMIT,
            velocity_limit_sim=WHEEL_DRIVE_VELOCITY_LIMIT,
            stiffness=0.0,  # velocity control
            damping=WHEEL_DRIVE_DAMPING,
        ),
        "torso": ImplicitActuatorCfg(
            joint_names_expr=["torso_j1", "torso_j2", "torso_j3"],
            effort_limit_sim={
                "torso_j1": TORSO_J1_EFFORT_LIMIT,
                "torso_j2": TORSO_J2_J3_EFFORT_LIMIT,
                "torso_j3": TORSO_J2_J3_EFFORT_LIMIT,
            },
            velocity_limit_sim=TORSO_VELOCITY_LIMIT,
            stiffness=DEFAULT_BODY_STIFFNESS,
            damping=DEFAULT_BODY_DAMPING,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_j1", "head_j2", "head_j3"],
            effort_limit_sim={
                "head_j1": HEAD_J1_J3_EFFORT_LIMIT,
                "head_j2": HEAD_J2_EFFORT_LIMIT,
                "head_j3": HEAD_J1_J3_EFFORT_LIMIT,
            },
            velocity_limit_sim=HEAD_VELOCITY_LIMIT,
            stiffness=DEFAULT_BODY_STIFFNESS,
            damping=DEFAULT_BODY_DAMPING,
        ),
        "arm_shoulder": ImplicitActuatorCfg(
            joint_names_expr=[".*_arm_j1", ".*_arm_j2"],
            effort_limit_sim=ARM_SHOULDER_EFFORT_LIMIT,
            velocity_limit_sim=ARM_SHOULDER_VELOCITY_LIMIT,
            stiffness=DEFAULT_BODY_STIFFNESS,
            damping=DEFAULT_BODY_DAMPING,
        ),
        "arm_elbow": ImplicitActuatorCfg(
            joint_names_expr=[".*_arm_j3", ".*_arm_j4"],
            effort_limit_sim=ARM_ELBOW_EFFORT_LIMIT,
            velocity_limit_sim=ARM_ELBOW_VELOCITY_LIMIT,
            stiffness=DEFAULT_BODY_STIFFNESS,
            damping=DEFAULT_BODY_DAMPING,
        ),
        "arm_wrist": ImplicitActuatorCfg(
            joint_names_expr=[".*_arm_j5", ".*_arm_j6", ".*_arm_j7"],
            effort_limit_sim=ARM_WRIST_EFFORT_LIMIT,
            velocity_limit_sim=ARM_WRIST_VELOCITY_LIMIT,
            stiffness=DEFAULT_BODY_STIFFNESS,
            damping=DEFAULT_BODY_DAMPING,
        ),
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_CMC_.*",
                ".*_pinky_CMC",
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
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Vega Sharpa mobile manipulator robot."""


VEGA_SHARPA_PLANAR_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=True,
        asset_path=f"{ASSET_DIR}/urdfs/vega_sharpa/vega_sharpa_planar.urdf",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=32,
            stabilization_threshold=0.00005,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0, damping=0
            )
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.056),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Omnidirectional base: velocity-controlled forward, lateral, + yaw
        "base_forward": ImplicitActuatorCfg(
            joint_names_expr=["virtual_x"],
            effort_limit_sim=1000.0,
            velocity_limit_sim=2.0,
            stiffness=0.0,
            damping=1e7,
        ),
        "base_lateral": ImplicitActuatorCfg(
            joint_names_expr=["virtual_y"],
            effort_limit_sim=1000.0,
            velocity_limit_sim=2.0,
            stiffness=0.0,
            damping=1e7,
        ),
        "base_yaw": ImplicitActuatorCfg(
            joint_names_expr=["virtual_yaw"],
            effort_limit_sim=1000.0,
            velocity_limit_sim=3.14,
            stiffness=0.0,
            damping=1e7,
        ),
        "torso": ImplicitActuatorCfg(
            joint_names_expr=["torso_j1", "torso_j2", "torso_j3"],
            effort_limit_sim={
                "torso_j1": TORSO_J1_EFFORT_LIMIT,
                "torso_j2": TORSO_J2_J3_EFFORT_LIMIT,
                "torso_j3": TORSO_J2_J3_EFFORT_LIMIT,
            },
            velocity_limit_sim=TORSO_VELOCITY_LIMIT,
            stiffness=DEFAULT_BODY_STIFFNESS,
            damping=DEFAULT_BODY_DAMPING,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_j1", "head_j2", "head_j3"],
            effort_limit_sim={
                "head_j1": HEAD_J1_J3_EFFORT_LIMIT,
                "head_j2": HEAD_J2_EFFORT_LIMIT,
                "head_j3": HEAD_J1_J3_EFFORT_LIMIT,
            },
            velocity_limit_sim=HEAD_VELOCITY_LIMIT,
            stiffness=DEFAULT_BODY_STIFFNESS,
            damping=DEFAULT_BODY_DAMPING,
        ),
        "arm_shoulder": ImplicitActuatorCfg(
            joint_names_expr=[".*_arm_j1", ".*_arm_j2"],
            effort_limit_sim=ARM_SHOULDER_EFFORT_LIMIT,
            velocity_limit_sim=ARM_SHOULDER_VELOCITY_LIMIT,
            stiffness=DEFAULT_BODY_STIFFNESS,
            damping=DEFAULT_BODY_DAMPING,
        ),
        "arm_elbow": ImplicitActuatorCfg(
            joint_names_expr=[".*_arm_j3", ".*_arm_j4"],
            effort_limit_sim=ARM_ELBOW_EFFORT_LIMIT,
            velocity_limit_sim=ARM_ELBOW_VELOCITY_LIMIT,
            stiffness=DEFAULT_BODY_STIFFNESS,
            damping=DEFAULT_BODY_DAMPING,
        ),
        "arm_wrist": ImplicitActuatorCfg(
            joint_names_expr=[".*_arm_j5", ".*_arm_j6", ".*_arm_j7"],
            effort_limit_sim=ARM_WRIST_EFFORT_LIMIT,
            velocity_limit_sim=ARM_WRIST_VELOCITY_LIMIT,
            stiffness=DEFAULT_BODY_STIFFNESS,
            damping=DEFAULT_BODY_DAMPING,
        ),
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_CMC_.*",
                ".*_pinky_CMC",
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
    soft_joint_pos_limit_factor=1.0,
)
