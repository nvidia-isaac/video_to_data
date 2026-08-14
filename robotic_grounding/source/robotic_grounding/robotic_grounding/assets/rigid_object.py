# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

from robotic_grounding.assets import ASSET_DIR

object_name = "arctic/box_rigid"
object_no_collision_name = "arctic/box_no_collision"

RIGID_OBJECT_CFG = RigidObjectCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        asset_path=f"{ASSET_DIR}/urdfs/{object_name}.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            enable_gyroscopic_forces=False,
            linear_damping=0.01,
            angular_damping=0.01,
            max_linear_velocity=1000.0,
            max_angular_velocity=64 / math.pi * 180.0,
            max_depenetration_velocity=1.0,
            max_contact_impulse=1e3,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0, damping=0.0
            )
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.0,
            rest_offset=0.0,
        ),
        collider_type="convex_decomposition",
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=[1, 0, 0, 0],
    ),
)

RIGID_OBJECT_NO_COLLISION_CFG = RigidObjectCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        asset_path=f"{ASSET_DIR}/urdfs/{object_no_collision_name}.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            enable_gyroscopic_forces=False,
            linear_damping=0.01,
            angular_damping=0.01,
            max_linear_velocity=1000.0,
            max_angular_velocity=64 / math.pi * 180.0,
            max_depenetration_velocity=1.0,
            max_contact_impulse=1e3,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0, damping=0.0
            )
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=[1, 0, 0, 0],
    ),
)
