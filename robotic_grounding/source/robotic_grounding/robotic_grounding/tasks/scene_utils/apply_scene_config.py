from __future__ import annotations

from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

from .scene_config import SceneConfig


def apply_scene_config(env_cfg: Any, scene_config: SceneConfig) -> Any:
    """
    Apply a SceneConfig to an environment configuration in-place.

    Configures the robot initial pose, target object, fixed objects, and tracking command.

    Args:
        env_cfg: Environment configuration to modify.
        scene_config: Scene configuration with objects and poses.

    Returns:
        The modified env_cfg (also modified in-place).
    """
    if scene_config.is_ee_motion:
        # For EE-based motion, use root pose if available
        if scene_config.root_init_translation is not None:
            robot_pos = [
                float(p) + float(o)
                for p, o in zip(
                    scene_config.root_init_translation,
                    scene_config.robot_anchor_offset,
                    strict=True,
                )
            ]
            env_cfg.scene.robot.init_state.pos = tuple(robot_pos)

        if scene_config.root_init_wxyz is not None:
            robot_rot = tuple(float(r) for r in scene_config.root_init_wxyz)
            env_cfg.scene.robot.init_state.rot = robot_rot

        # Store EE-based hand data in env_cfg for use during reset
        env_cfg.scene_config_ee_motion = True
        env_cfg.scene_config_left_hand_qpos = scene_config.left_hand_init_qpos
        env_cfg.scene_config_right_hand_qpos = scene_config.right_hand_init_qpos
        env_cfg.scene_config_head_translation = scene_config.head_init_translation
        env_cfg.scene_config_head_wxyz = scene_config.head_init_wxyz
    else:
        # For full joint motion, use robot_init_qpos
        robot_init = scene_config.robot_init_qpos
        if robot_init is None:
            raise ValueError("robot_init_qpos is required for non-EE motion")
        robot_pos = robot_init[:3]
        robot_rot = robot_init[3:7]
        robot_pos = [
            float(p) + float(o)
            for p, o in zip(robot_pos, scene_config.robot_anchor_offset, strict=True)
        ]
        robot_rot = tuple(float(r) for r in robot_rot)

        env_cfg.scene.robot.init_state.pos = tuple(robot_pos)
        env_cfg.scene.robot.init_state.rot = robot_rot
        env_cfg.scene_config_ee_motion = False

    target = scene_config.target_object
    if target.init_pos is None or target.init_rot is None:
        raise ValueError("target_object must have init_pos and init_rot set")
    # Convert to Python native types for OmegaConf compatibility
    target_pos = tuple(float(p) for p in target.init_pos)
    target_rot = tuple(float(r) for r in target.init_rot)
    target_scale = tuple(float(s) for s in target.scale)

    env_cfg.scene.object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=target.usd_path,
            scale=target_scale,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=1.0,
                disable_gravity=False,
            ),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=target_pos,
            rot=target_rot,
        ),
    )

    for fixed_obj in scene_config.fixed_objects:
        if fixed_obj.init_pos is None or fixed_obj.init_rot is None:
            raise ValueError(
                f"fixed_object {fixed_obj.name} must have init_pos and init_rot set"
            )
        # Convert to Python native types for OmegaConf compatibility
        fixed_pos = tuple(float(p) for p in fixed_obj.init_pos)
        fixed_rot = tuple(float(r) for r in fixed_obj.init_rot)
        fixed_scale = tuple(float(s) for s in fixed_obj.scale)

        fixed_cfg = AssetBaseCfg(
            prim_path=f"/World/envs/env_.*/{fixed_obj.name}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=fixed_obj.usd_path,
                scale=fixed_scale,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    fix_root_link=True,
                ),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=fixed_pos,
                rot=fixed_rot,
            ),
        )
        setattr(env_cfg.scene, fixed_obj.name, fixed_cfg)

    env_cfg.commands.motion.motion_file = scene_config.motion_file
    env_cfg.commands.motion.object_position_key = target.position_key
    env_cfg.commands.motion.object_quaternion_key = target.quaternion_key
    env_cfg.commands.motion.object_pos_offset = target.pos_offset
    env_cfg.commands.motion.robot_anchor_pos_offset = scene_config.robot_anchor_offset

    # Set flag for EE-based motion data
    if hasattr(env_cfg.commands.motion, "is_ee_motion"):
        env_cfg.commands.motion.is_ee_motion = scene_config.is_ee_motion

    if scene_config.file_joint_order is not None:
        if isinstance(scene_config.file_joint_order, list):
            env_cfg.commands.motion.file_joint_names = scene_config.file_joint_order
        elif scene_config.file_joint_order == "isaaclab":
            env_cfg.commands.motion.file_joint_names = None  # No reordering needed
        else:
            from robotic_grounding.assets.joint_order_registry import (  # noqa: PLC0415
                get_joint_order,
            )

            joint_order = get_joint_order(
                scene_config.robot_type, scene_config.file_joint_order
            )
            if joint_order is not None:
                env_cfg.commands.motion.file_joint_names = joint_order
            else:
                raise ValueError(
                    f"Unknown joint order '{scene_config.file_joint_order}' "
                    f"for robot '{scene_config.robot_type}'"
                )

    return env_cfg
