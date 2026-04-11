# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Apply a SceneConfig to an IsaacLab environment configuration.

Public API:
    apply_scene_objects  — spawn object + fixed objects (viewer and training)
    apply_scene_robot    — place robot hands from registry
    apply_scene_commands — configure dual-hand tracking command
    apply_scene_contact_sensors — set up per-side contact sensors
    apply_scene_config   — all of the above in one call (training entry point)
"""

from __future__ import annotations

import os
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from pxr import Usd, UsdGeom

from robotic_grounding.assets.articulated_object import ARTICULATED_OBJECT_CFG
from robotic_grounding.assets.rigid_object import RIGID_OBJECT_CFG
from robotic_grounding.assets.robot_registry import get_robot_spec
from robotic_grounding.tasks.scene_utils.scene_config import (
    ArticulatedObjectConfig,
    ObjectConfig,
    SceneConfig,
)
from robotic_grounding.tasks.v2p.mdp.actions import (
    VirtualArticulatedObjectControlCfg,
    VirtualRigidObjectControlCfg,
)

###################################################
# Parameters
###################################################

virtual_object_control_linear_stiffness = 50.0
virtual_object_control_linear_damping = 10.0
virtual_object_control_angular_stiffness = 10.0
virtual_object_control_angular_damping = 0.1
virtual_object_control_max_force = 60.0
virtual_object_control_max_torque = 60.0


def _spawn_articulated(
    obj: ArticulatedObjectConfig,
    prim_path: str,
) -> ArticulationCfg:
    """Build an ArticulationCfg for an articulated object."""
    obj_pos = tuple(float(p) for p in obj.init_pos) if obj.init_pos else (0.0, 0.0, 0.0)
    obj_rot = (
        tuple(float(r) for r in obj.init_rot) if obj.init_rot else (1.0, 0.0, 0.0, 0.0)
    )
    return ARTICULATED_OBJECT_CFG.replace(
        prim_path=prim_path,
        spawn=ARTICULATED_OBJECT_CFG.spawn.replace(
            asset_path=obj.urdf_path,
            fix_base=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=obj_pos,
            rot=obj_rot,
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "joint": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim={".*": 75.0},
                velocity_limit_sim={".*": 15.0},
                stiffness={".*": 0.0},
                damping={".*": 0.0},
                armature={".*": 0.01},
                friction={".*": 0.01},
            ),
        },
    )


def _spawn_rigid(
    obj: ObjectConfig,
    prim_path: str,
) -> RigidObjectCfg:
    """Build a RigidObjectCfg for a rigid object."""
    obj_pos = tuple(float(p) for p in obj.init_pos) if obj.init_pos else (0.0, 0.0, 0.0)
    obj_rot = (
        tuple(float(r) for r in obj.init_rot) if obj.init_rot else (1.0, 0.0, 0.0, 0.0)
    )

    if obj.usd_path.endswith(".urdf"):
        return RIGID_OBJECT_CFG.replace(
            prim_path=prim_path,
            spawn=RIGID_OBJECT_CFG.spawn.replace(asset_path=obj.usd_path),
            init_state=RigidObjectCfg.InitialStateCfg(pos=obj_pos, rot=obj_rot),
        )

    obj_scale = tuple(float(s) for s in obj.scale)
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=obj.usd_path,
            scale=obj_scale,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                linear_damping=0.01,
                angular_damping=0.01,
                max_depenetration_velocity=1.0,
                max_contact_impulse=1e3,
                disable_gravity=False,
            ),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=obj_pos, rot=obj_rot),
    )


def apply_scene_objects(env_cfg: Any, scene_config: SceneConfig) -> None:
    """Spawn all scene objects and fixed objects into env_cfg.scene."""
    for obj in scene_config.scene_objects:
        attr_name = obj.name
        prim_path = f"{{ENV_REGEX_NS}}/{attr_name}"

        if isinstance(obj, ArticulatedObjectConfig):
            cfg = _spawn_articulated(obj, prim_path)
        else:
            cfg = _spawn_rigid(obj, prim_path)

        setattr(env_cfg.scene, attr_name, cfg)

    # Fixed objects (support surfaces, etc.)
    for fixed_obj in scene_config.fixed_objects:
        if fixed_obj.init_pos is None or fixed_obj.init_rot is None:
            raise ValueError(
                f"fixed_object {fixed_obj.name} must have init_pos and init_rot"
            )
        stage = Usd.Stage.Open(fixed_obj.usd_path)
        for idx, prim in enumerate(stage.Traverse()):
            if not prim.IsA(UsdGeom.Cylinder):
                continue
            cyl = UsdGeom.Cylinder(prim)
            radius = cyl.GetRadiusAttr().Get()
            height = cyl.GetHeightAttr().Get()
            xf = UsdGeom.Xformable(prim)
            ops = xf.GetOrderedXformOps()
            translate = ops[0].Get() if ops else (0.0, 0.0, 0.0)

            fixed_cfg = AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{fixed_obj.name}_{idx}",
                spawn=sim_utils.CylinderCfg(
                    radius=radius,
                    height=height,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=True
                    ),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=1.0
                    ),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.14, 0.14, 0.14), metallic=0.7
                    ),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=translate,
                    rot=[1.0, 0.0, 0.0, 0.0],
                ),
            )
            setattr(env_cfg.scene, f"{fixed_obj.name}_{idx}", fixed_cfg)


def apply_scene_virtual_object_controls(
    env_cfg: Any, scene_config: SceneConfig
) -> None:
    """Spawn virtual object controls into env_cfg.scene."""
    for obj in scene_config.scene_objects:
        object_name = obj.name
        if isinstance(obj, ArticulatedObjectConfig):
            voc_cfg = VirtualArticulatedObjectControlCfg(
                asset_name=object_name,
                root_body_name=obj.body_names[0],
                tracking_controller_linear_stiffness=virtual_object_control_linear_stiffness,
                tracking_controller_linear_damping=virtual_object_control_linear_damping,  # critical damping: 2 * sqrt(kp * m)
                tracking_controller_angular_stiffness=virtual_object_control_angular_stiffness,
                tracking_controller_angular_damping=virtual_object_control_angular_damping,  # critical damping: 2 * sqrt(kp * I)
                max_force=virtual_object_control_max_force,
                max_torque=virtual_object_control_max_torque,
            )
            setattr(
                env_cfg.actions,
                f"virtual_articulated_object_control_{object_name}",
                voc_cfg,
            )
        else:
            voc_cfg = VirtualRigidObjectControlCfg(
                asset_name=object_name,
                tracking_controller_linear_stiffness=virtual_object_control_linear_stiffness,
                tracking_controller_linear_damping=virtual_object_control_linear_damping,  # critical damping: 2 * sqrt(kp * m)
                tracking_controller_angular_stiffness=virtual_object_control_angular_stiffness,
                tracking_controller_angular_damping=virtual_object_control_angular_damping,  # critical damping: 2 * sqrt(kp * I)
                max_force=virtual_object_control_max_force,
                max_torque=virtual_object_control_max_torque,
            )
            setattr(
                env_cfg.actions, f"virtual_rigid_object_control_{object_name}", voc_cfg
            )


def apply_scene_robot(
    env_cfg: Any, scene_config: SceneConfig, static: bool = False
) -> None:
    """Place robot from the robot registry based on scene_config.robot_name.

    Args:
        env_cfg: The environment configuration to modify.
        scene_config: The scene configuration with robot_name.
        static: If True, disable gravity so the robot holds its initial pose.
    """
    if scene_config.robot_name is None:
        raise ValueError("robot_name not set — cannot configure robot")

    robot_spec = get_robot_spec(scene_config.robot_name)
    if robot_spec is None:
        raise ValueError(f"Unknown robot: {scene_config.robot_name}")

    def _maybe_disable_gravity(cfg: ArticulationCfg) -> ArticulationCfg:
        if not static:
            return cfg
        return cfg.replace(
            spawn=cfg.spawn.replace(
                rigid_props=cfg.spawn.rigid_props.replace(disable_gravity=True),
            ),
        )

    if robot_spec.is_dual_hand:
        env_cfg.scene.right_robot = _maybe_disable_gravity(
            robot_spec.right_cfg
        ).replace(prim_path="{ENV_REGEX_NS}/RightRobot")
        env_cfg.scene.left_robot = _maybe_disable_gravity(robot_spec.left_cfg).replace(
            prim_path="{ENV_REGEX_NS}/LeftRobot"
        )
    elif robot_spec.robot_cfg is not None:
        env_cfg.scene.robot = _maybe_disable_gravity(robot_spec.robot_cfg).replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )


def apply_scene_commands(env_cfg: Any, scene_config: SceneConfig) -> None:
    """Configure the dual-hand tracking command from scene_config fields."""
    if scene_config.robot_name is None:
        raise ValueError("robot_name not set — cannot configure commands")

    robot_spec = get_robot_spec(scene_config.robot_name)
    if robot_spec is None:
        raise ValueError(f"Unknown robot: {scene_config.robot_name}")

    motion_folder = scene_config.motion_folder or os.path.dirname(
        scene_config.motion_file
    )

    # Scene attribute names for the command term to look up objects
    object_body_names = [obj.name for obj in scene_config.scene_objects]

    cmd = env_cfg.commands.dual_hands_object_tracking_command
    cmd.wrist_joint_names = robot_spec.wrist_joint_names
    cmd.finger_joint_names = robot_spec.finger_joint_names
    cmd.wrist_body_name = robot_spec.wrist_body_name
    cmd.fingertip_body_name = robot_spec.fingertip_body_name
    cmd.object_body_names = object_body_names
    cmd.motion_folder = motion_folder
    cmd.motion_filters = scene_config.motion_filters or []


def apply_scene_contact_sensors(env_cfg: Any, scene_config: SceneConfig) -> None:
    """Set up per-side contact sensors between robot hands and the object."""
    if scene_config.robot_name is None:
        raise ValueError("robot_name not set — cannot configure contacts")

    robot_spec = get_robot_spec(scene_config.robot_name)
    if robot_spec is None:
        raise ValueError(f"Unknown robot: {scene_config.robot_name}")

    right_robot_filter_prim_paths = [
        f"{{ENV_REGEX_NS}}/RightRobot/{b.replace('.*', 'right')}"
        for b in robot_spec.hand_contact_bodies
    ]
    left_robot_filter_prim_paths = [
        f"{{ENV_REGEX_NS}}/LeftRobot/{b.replace('.*', 'left')}"
        for b in robot_spec.hand_contact_bodies
    ]

    # Contact sensor on the object body-hand pairs
    env_cfg.object_to_hand_contact_sensor_names = []

    for object in scene_config.scene_objects:
        object_name = object.name
        if isinstance(object, ArticulatedObjectConfig):
            object_body_names = object.body_names
        else:
            object_body_names = ["object"]  # URDF link name for rigid objects

        for body_name in object_body_names:
            for side in ["right", "left"]:
                sensor_name = f"{object_name}_{body_name}_to_{side}_hand_contact_sensor"
                setattr(
                    env_cfg.scene,
                    sensor_name,
                    ContactSensorCfg(
                        prim_path=f"{{ENV_REGEX_NS}}/{object_name}/{body_name}",
                        track_pose=True,
                        debug_vis=False,
                        force_threshold=0.1,
                        history_length=3,
                        filter_prim_paths_expr=(
                            right_robot_filter_prim_paths
                            if side == "right"
                            else left_robot_filter_prim_paths
                        ),
                        track_contact_points=True,
                        track_air_time=True,
                        max_contact_data_count_per_prim=128,
                    ),
                )
                env_cfg.object_to_hand_contact_sensor_names.append(sensor_name)


def apply_scene_config(env_cfg: Any, scene_config: SceneConfig) -> Any:
    """Apply scene config: objects + robot + commands + contacts.

    Skips commands/contacts if the env_cfg doesn't have the required fields
    (e.g. scene viewer with no RL components).
    """
    apply_scene_objects(env_cfg, scene_config)
    apply_scene_virtual_object_controls(env_cfg, scene_config)

    if scene_config.robot_name:
        apply_scene_robot(env_cfg, scene_config)

    if hasattr(env_cfg, "commands") and hasattr(
        env_cfg.commands, "dual_hands_object_tracking_command"
    ):
        apply_scene_commands(env_cfg, scene_config)
        apply_scene_contact_sensors(env_cfg, scene_config)

    env_cfg.episode_length_s = (
        scene_config.episode_length_s
        / env_cfg.commands.dual_hands_object_tracking_command.motion_speed
    )

    return env_cfg
