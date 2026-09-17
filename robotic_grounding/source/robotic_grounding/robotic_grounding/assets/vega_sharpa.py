# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vega base with Sharpa Wave hands.

``VEGA_SHARPA_CFG`` is the baseline articulation with the original arm
actuator defaults. ``VEGA_SHARPA_SYSID_CFG`` derives from that baseline and
replaces only the arm actuator with the sim-to-real system-identified model.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robotic_grounding.assets import ASSET_DIR
from robotic_grounding.assets.sharpa_wave import finger_actuators

VEGA_SHARPA_URDF_PATH = f"{ASSET_DIR}/urdfs/vega_sharpa/vega_sharpa_reduced.urdf"

ARM_JOINT_NAMES = [
    ".*_arm_j1",
    ".*_arm_j2",
    ".*_arm_j3",
    ".*_arm_j4",
    ".*_arm_j5",
    ".*_arm_j6",
    ".*_arm_j7",
]

EEF_BODY_NAMES = [
    ".*_arm_l7",
]

HAND_CONTACT_BODIES = [
    # Palm
    ".*_arm_l7",
    # Thumb
    ".*_thumb_MC",
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

# Vega mixes side conventions: the arm links are `L_`/`R_` prefixed while the finger
# links are `left_`/`right_`. Substituting one side token into the `.*` patterns above
# therefore yields `left_arm_l7`, which matches no body -- and PhysX only logs the empty
# contact filter, so the palm silently stops sensing contact. Name both sides outright.
_SIDE_ARM_PREFIX = {"left": "L", "right": "R"}
HAND_CONTACT_BODIES_BY_SIDE = {
    side: [
        (
            f"{_SIDE_ARM_PREFIX[side]}_arm_l7"
            if body == ".*_arm_l7"
            else body.replace(".*", side)
        )
        for body in HAND_CONTACT_BODIES
    ]
    for side in ("left", "right")
}

_ARM_EFFORT_LIMIT = {
    ".*_arm_j1": 100.0,
    ".*_arm_j2": 100.0,
    ".*_arm_j3": 80.0,
    ".*_arm_j4": 80.0,
    ".*_arm_j5": 25.0,
    ".*_arm_j6": 25.0,
    ".*_arm_j7": 25.0,
}

_ARM_VELOCITY_LIMIT = {
    ".*_arm_j1": 2.4,
    ".*_arm_j2": 2.4,
    ".*_arm_j3": 2.4,
    ".*_arm_j4": 2.4,
    ".*_arm_j5": 2.4,
    ".*_arm_j6": 2.4,
    ".*_arm_j7": 2.4,
}

_DEFAULT_ARM_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=ARM_JOINT_NAMES,
    effort_limit_sim=_ARM_EFFORT_LIMIT,
    velocity_limit_sim=_ARM_VELOCITY_LIMIT,
    stiffness={
        ".*_arm_j1": 1000.0,
        ".*_arm_j2": 1000.0,
        ".*_arm_j3": 500.0,
        ".*_arm_j4": 500.0,
        ".*_arm_j5": 500.0,
        ".*_arm_j6": 500.0,
        ".*_arm_j7": 500.0,
    },
    damping={
        ".*_arm_j1": 50.0,
        ".*_arm_j2": 50.0,
        ".*_arm_j3": 25.0,
        ".*_arm_j4": 25.0,
        ".*_arm_j5": 25.0,
        ".*_arm_j6": 25.0,
        ".*_arm_j7": 25.0,
    },
    armature={joint_name: 0.001 for joint_name in ARM_JOINT_NAMES},
    friction={joint_name: 0.01 for joint_name in ARM_JOINT_NAMES},
)

_RIGID_PROPS = sim_utils.RigidBodyPropertiesCfg(
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

_ARTICULATION_PROPS = sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=True,
    solver_position_iteration_count=8,
    solver_velocity_iteration_count=0,
    sleep_threshold=0.005,
    stabilization_threshold=0.0005,
)

_NONE_JOINT_DRIVE = sim_utils.UrdfConverterCfg.JointDriveCfg(
    gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
        stiffness=0,
        damping=0,
    ),
)

VEGA_SHARPA_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=True,
        asset_path=VEGA_SHARPA_URDF_PATH,
        activate_contact_sensors=True,
        rigid_props=_RIGID_PROPS,
        articulation_props=_ARTICULATION_PROPS,
        joint_drive=_NONE_JOINT_DRIVE,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "arm": _DEFAULT_ARM_ACTUATOR_CFG,
        "fingers": finger_actuators,
    },
)
"""Baseline Vega Sharpa articulation with the original default arm gains."""

_SYSID_ARM_ACTUATOR_CFG = DelayedPDActuatorCfg(
    joint_names_expr=ARM_JOINT_NAMES,
    # The measured command latency is approximately 40 ms, or four physics
    # steps in the 100 Hz system-identification setup.
    min_delay=4,
    max_delay=4,
    effort_limit=_ARM_EFFORT_LIMIT,
    effort_limit_sim=_ARM_EFFORT_LIMIT,
    velocity_limit_sim=_ARM_VELOCITY_LIMIT,
    # Identified on the left arm and applied to both arms, which use the same
    # motors. Joint 7 reuses the joint-6 fit.
    stiffness={
        ".*_arm_j1": 56.578,
        ".*_arm_j2": 24.332,
        ".*_arm_j3": 28.290,
        ".*_arm_j4": 26.164,
        ".*_arm_j5": 1.482,
        ".*_arm_j6": 4.233,
        ".*_arm_j7": 4.233,
    },
    damping={
        ".*_arm_j1": 47.150,
        ".*_arm_j2": 20.378,
        ".*_arm_j3": 23.488,
        ".*_arm_j4": 21.740,
        ".*_arm_j5": 1.252,
        ".*_arm_j6": 3.541,
        ".*_arm_j7": 3.541,
    },
    # Motor reflected inertia = rotor inertia * gear ratio**2.
    armature={
        ".*_arm_j1": 0.5510,
        ".*_arm_j2": 0.5510,
        ".*_arm_j3": 0.19072,
        ".*_arm_j4": 0.19072,
        ".*_arm_j5": 0.07232,
        ".*_arm_j6": 0.07232,
        ".*_arm_j7": 0.07232,
    },
)

VEGA_SHARPA_SYSID_CFG = VEGA_SHARPA_CFG.replace(
    actuators={
        **VEGA_SHARPA_CFG.actuators,
        "arm": _SYSID_ARM_ACTUATOR_CFG,
    },
)
"""Vega Sharpa articulation with the system-identified delayed arm actuator."""


# Explicit finger joint order, right hand first. Envs that consume a retargeted reference
# must pass this to the tracking command's ``finger_joint_names``: the reference columns are
# laid out in this order, and resolving the joints by regex instead would silently reorder
# them to the articulation's own ordering, scrambling every finger target.
# Validated against the reference export manifest at replay time.
_RIGHT_FINGER_ORDER = [
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
]
_LEFT_FINGER_ORDER = [n.replace("right_", "left_", 1) for n in _RIGHT_FINGER_ORDER]

VEGA_FINGER_JOINT_ORDER = _RIGHT_FINGER_ORDER + _LEFT_FINGER_ORDER
"""44 finger joints, right side first."""

# Explicit arm and cross-system GR00T orders. Regex expressions are appropriate for
# actuator selection but not for data contracts: they resolve in articulation order,
# which is not guaranteed to match the policy/export order.
VEGA_ARM_JOINT_ORDER = [f"R_arm_j{i}" for i in range(1, 8)] + [
    f"L_arm_j{i}" for i in range(1, 8)
]
"""14 arm joints, right side first."""

VEGA_GROOT_JOINT_ORDER = (
    VEGA_ARM_JOINT_ORDER[:7]
    + VEGA_FINGER_JOINT_ORDER[:22]
    + VEGA_ARM_JOINT_ORDER[7:]
    + VEGA_FINGER_JOINT_ORDER[22:]
)
"""Canonical GR00T action order: right arm/fingers, then left arm/fingers."""

assert len(VEGA_ARM_JOINT_ORDER) == 14
assert len(VEGA_FINGER_JOINT_ORDER) == 44
assert len(VEGA_GROOT_JOINT_ORDER) == 58
assert len(set(VEGA_GROOT_JOINT_ORDER)) == len(VEGA_GROOT_JOINT_ORDER)


# Arm default pose = the RL-training standing pose (palm-down IK); LEFT is the joint mirror
# of RIGHT (from the real-robot arm config).
#
# Load-bearing beyond the initial spawn: envs that reset the robot onto a reference
# trajectory overwrite the joint state every reset, so this is never what the policy starts
# from -- but `init_state.joint_pos` also sets the articulation's `default_joint_pos`, and
# the `joint_pos_rel` observation is `joint_pos - default_joint_pos`. Applying it re-origins
# every joint observation, by up to ~2.8 rad on some joints. VEGA_SHARPA_CFG leaves all
# joints at 0.0, so the URDF does not supply this.
VEGA_RIGHT_ARM_DEFAULT_POSE = {
    "R_arm_j1": -1.4659,
    "R_arm_j2": -0.9913,
    "R_arm_j3": -0.8333,
    "R_arm_j4": -2.7730,
    "R_arm_j5": -1.6266,
    "R_arm_j6": -1.0706,
    "R_arm_j7": 0.6680,
}
VEGA_LEFT_ARM_DEFAULT_POSE = {
    "L_arm_j1": 1.4659,
    "L_arm_j2": 0.9913,
    "L_arm_j3": 0.8333,
    "L_arm_j4": -2.7730,
    "L_arm_j5": 1.6266,
    "L_arm_j6": 1.0706,
    "L_arm_j7": -0.6680,
}

VEGA_ARM_DEFAULT_POSE = {**VEGA_RIGHT_ARM_DEFAULT_POSE, **VEGA_LEFT_ARM_DEFAULT_POSE}
"""Both arms' standing pose, for ``init_state.joint_pos``."""


# Official Sharpa sysid (position-mode) finger calibration, from the real-robot arm config.
# Per group -> (stiffness, damping, armature, friction, velocity_limit, effort_limit).
#
# Distinct from the finger actuators VEGA_SHARPA_CFG uses: those come from
# `sharpa_wave.finger_actuators`, which is UNIFORM across every group (stiffness 1.74533,
# damping 0.01745, no per-group armature or friction). The sysid stiffnesses below span
# 0.90-13.2 -- an order of magnitude of variation the uniform hand cannot express -- which
# is what lets a PhysX grasp hold. Field set matches `sharpa_wave.finger_actuators`
# (effort_limit_sim / velocity_limit_sim) for this IsaacLab version.
_SYSID_FINGER_GROUPS = {
    ".*_CMC_FE": (6.95456, 0.240986, 0.0032, 0.132, 11.8408, 3.3),
    ".*_CMC_AA": (13.20095, 0.451640, 0.0032, 0.132, 11.8408, 3.3),
    ".*_pinky_CMC": (1.38026, 0.039248, 0.00012, 0.013, 35.0742, 0.5285),
    ".*_MCP_FE": (4.76002, 0.183003, 0.00265, 0.104, 16.0769, 1.864),
    ".*_MCP_AA": (6.62167, 0.207984, 0.00265, 0.104, 16.0769, 1.864),
    ".*_IP": (0.90757, 0.039992, 0.00061, 0.02476, 11.6183, 0.638),  # thumb IP = PIP
    ".*_PIP": (0.90757, 0.039992, 0.00061, 0.02476, 11.6183, 0.638),
    ".*_DIP": (0.90413, 0.031513, 0.00042, 0.000418, 14.6659, 0.18937),
}

VEGA_SYSID_FINGER_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=list(_SYSID_FINGER_GROUPS),
    stiffness={k: v[0] for k, v in _SYSID_FINGER_GROUPS.items()},
    damping={k: v[1] for k, v in _SYSID_FINGER_GROUPS.items()},
    armature={k: v[2] for k, v in _SYSID_FINGER_GROUPS.items()},
    friction={k: v[3] for k, v in _SYSID_FINGER_GROUPS.items()},
    velocity_limit_sim={k: v[4] for k, v in _SYSID_FINGER_GROUPS.items()},
    effort_limit_sim={k: v[5] for k, v in _SYSID_FINGER_GROUPS.items()},
)
"""Both hands' 44 finger joints on the official Sharpa sysid position-mode calibration."""
