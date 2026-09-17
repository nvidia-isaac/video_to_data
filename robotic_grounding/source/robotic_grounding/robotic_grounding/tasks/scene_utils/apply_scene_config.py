# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
from pathlib import Path
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from pxr import Usd, UsdGeom

from robotic_grounding.assets.articulated_object import ARTICULATED_OBJECT_CFG
from robotic_grounding.assets.rigid_object import RIGID_OBJECT_CFG
from robotic_grounding.assets.robot_registry import get_robot_spec
from robotic_grounding.motion_schema import resolve_playback_timing
from robotic_grounding.tasks.scene_utils.scene_config import (
    ArticulatedObjectConfig,
    ObjectConfig,
    SceneConfig,
)
from robotic_grounding.tasks.v2d.mdp.actions import (
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
    joint_pos = float(obj.init_joint_pos) if obj.init_joint_pos is not None else 0.0
    return ARTICULATED_OBJECT_CFG.replace(
        prim_path=prim_path,
        spawn=ARTICULATED_OBJECT_CFG.spawn.replace(
            asset_path=obj.urdf_path,
            fix_base=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=obj_pos,
            rot=obj_rot,
            joint_pos={".*": joint_pos},
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
                friction={".*": 0.1},
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
                # 8/0 rather than 16/1: measurably faster with no loss of grasp
                # stability. Keep in sync with RIGID_OBJECT_CFG (the URDF path) and
                # with the robot/articulated-object configs, which already use 8/0.
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
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


def _contact_bodies_for_side(
    scene_config: SceneConfig, hand_contact_bodies: list[str], side: str
) -> list[str]:
    """Resolve one side's contact-sensor body names.

    Prefers the robot spec, which may name each side outright when a single ``.*`` ->
    side substitution cannot produce them (Vega's arms are ``L_``/``R_`` prefixed while
    its fingers are ``left_``/``right_``). Falls back to substitution for robots that
    are not in the registry or that use one consistent side token.
    """
    spec = get_robot_spec(scene_config.robot_name)
    if spec is not None and spec.hand_contact_bodies_by_side:
        return spec.contact_bodies_for_side(side)
    return [b.replace(".*", side) for b in hand_contact_bodies]


def apply_scene_objects(env_cfg: Any, scene_config: SceneConfig) -> None:
    """Spawn all scene objects and fixed objects into env_cfg.scene."""
    # Optional object solver-iteration overrides set by the env cfg (-1 = disabled).
    # PhysX steps the whole GPU scene at the max iteration count over all bodies, so one
    # object left at the 16/1 default forces every body (incl. the robot) to 16/1.
    solver_pos_iters = getattr(env_cfg, "object_solver_position_iteration_count", -1)
    solver_vel_iters = getattr(env_cfg, "object_solver_velocity_iteration_count", -1)
    if solver_pos_iters is None or solver_pos_iters < 0:
        solver_pos_iters = None
    if solver_vel_iters is None or solver_vel_iters < 0:
        solver_vel_iters = None
    # Object contact material is deliberately NOT configured here. Two mechanisms were
    # tried and both fail silently: `cfg.spawn.physics_material` is not a field on
    # `UrdfFileCfg`/`UsdFileCfg` (it exists only on `GroundPlaneCfg` and `ShapeCfg`), so
    # assigning it just creates a stray attribute that `dataclasses.asdict` -- and hence
    # the spawner -- drops; and `cfg.spawn.collision_props` offsets never reach the stage
    # because the object's collider is an instance proxy, which
    # `isaaclab.sim.utils.apply_nested` skips without descending.
    #
    # The repo convention is instead to raise friction on the ROBOT with a startup
    # `randomize_rigid_body_material` event (see `v2d_hand_env_cfg.py` and
    # `VegaManipEventsCfg.hand_physics_material`), leaving the object at PhysX defaults.
    # This guard exists so that convention cannot be silently bypassed again.
    for dead_attr in ("object_contact_friction", "object_contact_offset"):
        if getattr(env_cfg, dead_attr, None) is not None:
            raise ValueError(
                f"env_cfg sets `{dead_attr}`, which nothing reads -- object contact "
                "material cannot be delivered through the spawn cfg. Raise friction on "
                "the robot instead with a `randomize_rigid_body_material` startup event "
                "(see VegaManipEventsCfg.hand_physics_material)."
            )

    for obj in scene_config.scene_objects:
        attr_name = obj.name
        prim_path = f"{{ENV_REGEX_NS}}/{attr_name}"

        if isinstance(obj, ArticulatedObjectConfig):
            cfg = _spawn_articulated(obj, prim_path)
        else:
            cfg = _spawn_rigid(obj, prim_path)

        for props_name in ("rigid_props", "articulation_props"):
            props = getattr(cfg.spawn, props_name, None)
            if props is None:
                continue
            if solver_pos_iters is not None:
                props.solver_position_iteration_count = solver_pos_iters
            if solver_vel_iters is not None:
                props.solver_velocity_iteration_count = solver_vel_iters

        setattr(env_cfg.scene, attr_name, cfg)
        if hasattr(env_cfg, "events") and hasattr(
            env_cfg.events, "setup_collision_groups"
        ):
            env_cfg.events.setup_collision_groups.params["object_names"].append(
                attr_name
            )

    # Fixed objects (support surfaces, etc.)
    for fixed_obj in scene_config.fixed_objects:
        if fixed_obj.init_pos is None or fixed_obj.init_rot is None:
            raise ValueError(
                f"fixed_object {fixed_obj.name} must have init_pos and init_rot"
            )
        stage = Usd.Stage.Open(fixed_obj.usd_path)
        for idx, prim in enumerate(stage.Traverse()):
            attr_name = f"{fixed_obj.name}_{idx}"
            xf = UsdGeom.Xformable(prim)
            ops = xf.GetOrderedXformOps()
            translate = ops[0].Get() if ops else (0.0, 0.0, 0.0)

            if prim.IsA(UsdGeom.Cylinder):
                cyl = UsdGeom.Cylinder(prim)
                spawn_cfg = sim_utils.CylinderCfg(
                    radius=cyl.GetRadiusAttr().Get(),
                    height=cyl.GetHeightAttr().Get(),
                )
            elif prim.IsA(UsdGeom.Cube):
                cube = UsdGeom.Cube(prim)
                size = cube.GetSizeAttr().Get()
                # Cube scale encodes per-axis dimensions
                scale_ops = [op for op in ops if "scale" in op.GetName().lower()]
                if scale_ops:
                    sx, sy, sz = scale_ops[0].Get()
                else:
                    sx = sy = sz = 1.0
                spawn_cfg = sim_utils.CuboidCfg(
                    size=(size * sx, size * sy, size * sz),
                )
            else:
                continue

            spawn_cfg.rigid_props = sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True
            )
            spawn_cfg.mass_props = sim_utils.MassPropertiesCfg(mass=100.0)
            spawn_cfg.collision_props = sim_utils.CollisionPropertiesCfg(
                collision_enabled=True
            )
            spawn_cfg.physics_material = sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0
            )
            spawn_cfg.visual_material = sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.14, 0.14, 0.14), metallic=0.7
            )

            fixed_cfg = AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{attr_name}",
                spawn=spawn_cfg,
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=translate,
                    rot=[1.0, 0.0, 0.0, 0.0],
                ),
            )
            setattr(env_cfg.scene, attr_name, fixed_cfg)
            if hasattr(env_cfg, "events") and hasattr(
                env_cfg.events, "setup_collision_groups"
            ):
                env_cfg.events.setup_collision_groups.params[
                    "fixed_object_names"
                ].append(attr_name)


def apply_scene_virtual_object_controls(
    env_cfg: Any, scene_config: SceneConfig
) -> None:
    """Spawn virtual object controls into env_cfg.scene."""
    # Determine command name based on env type
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion"):
        command_name = "motion"
    else:
        command_name = "dual_hands_object_tracking_command"

    for obj in scene_config.scene_objects:
        object_name = obj.name
        if isinstance(obj, ArticulatedObjectConfig):
            voc_cfg = VirtualArticulatedObjectControlCfg(
                asset_name=object_name,
                command_name=command_name,
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
                command_name=command_name,
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
    env_cfg: Any,
    scene_config: SceneConfig,
    static: bool = False,
    use_primitive_urdfs: bool = False,
) -> None:
    """Place robot from the robot registry based on scene_config.robot_name.

    Args:
        env_cfg: The environment configuration to modify.
        scene_config: The scene configuration with robot_name.
        static: If True, disable gravity so the robot holds its initial pose.
        use_primitive_urdfs: If True, use primitive URDFs for the robot.
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
            if not use_primitive_urdfs
            else robot_spec.right_primitive_cfg
        ).replace(prim_path="{ENV_REGEX_NS}/RightRobot")
        env_cfg.scene.left_robot = _maybe_disable_gravity(
            robot_spec.left_cfg
            if not use_primitive_urdfs
            else robot_spec.left_primitive_cfg
        ).replace(prim_path="{ENV_REGEX_NS}/LeftRobot")
        if hasattr(env_cfg, "events") and hasattr(
            env_cfg.events, "setup_collision_groups"
        ):
            env_cfg.events.setup_collision_groups.params["robot_names"].append(
                "RightRobot"
            )
            env_cfg.events.setup_collision_groups.params["robot_names"].append(
                "LeftRobot"
            )
    elif robot_spec.robot_cfg is not None:
        env_cfg.scene.robot = _maybe_disable_gravity(robot_spec.robot_cfg).replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        if hasattr(env_cfg, "events") and hasattr(
            env_cfg.events, "setup_collision_groups"
        ):
            env_cfg.events.setup_collision_groups.params["robot_names"].append("Robot")


def _is_sequence_robot_partition_path(path: str) -> bool:
    parts = Path(path).parts
    return any(p.startswith("sequence_id=") for p in parts) and any(
        p.startswith("robot_name=") for p in parts
    )


def apply_scene_commands(env_cfg: Any, scene_config: SceneConfig) -> None:
    """Configure the dual-hand tracking command from scene_config fields."""
    if scene_config.robot_name is None:
        raise ValueError("robot_name not set — cannot configure commands")

    robot_spec = get_robot_spec(scene_config.robot_name)
    if robot_spec is None:
        raise ValueError(f"Unknown robot: {scene_config.robot_name}")

    motion_path = Path(scene_config.motion_file)
    motion_filters: list[tuple[str, str, str]]
    if _is_sequence_robot_partition_path(scene_config.motion_file):
        motion_folder = str(motion_path if motion_path.is_dir() else motion_path.parent)
        motion_filters = []
    else:
        motion_folder = scene_config.motion_folder or os.path.dirname(
            scene_config.motion_file
        )
        motion_filters = scene_config.motion_filters or []

    # Scene attribute names for the command term to look up objects
    object_body_names = [obj.name for obj in scene_config.scene_objects]

    cmd = env_cfg.commands.dual_hands_object_tracking_command
    cmd.wrist_joint_names = robot_spec.wrist_joint_names
    cmd.finger_joint_names = robot_spec.finger_joint_names
    cmd.wrist_body_name = robot_spec.wrist_body_name
    cmd.fingertip_body_name = robot_spec.fingertip_body_name
    cmd.object_body_names = object_body_names
    cmd.motion_folder = motion_folder
    cmd.motion_filters = motion_filters


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


def apply_scene_config(
    env_cfg: Any,
    scene_config: SceneConfig,
    use_primitive_urdfs: bool = False,
    motion_files: list[str] | None = None,
) -> Any:
    """Apply scene config: objects + robot + commands + contacts.

    Supports dual-hands (V2D) and whole-body environments. Skips
    commands/contacts if the env_cfg doesn't have the required fields
    (e.g. scene viewer).

    Args:
        env_cfg: Environment configuration to update in place.
        scene_config: Parsed scene, robot, and motion configuration.
        use_primitive_urdfs: Whether to select primitive-collision robot assets.
        motion_files: Optional bank of per-sequence motion paths (from
            ``--motion_dir``). Only valid for whole-body motion commands
            that expose a ``motion_files`` field; each env samples one motion
            from the bank per reset. ``scene_config`` is still built from a
            single representative sequence for scene/robot/timing setup.
    """
    if motion_files:
        is_whole_body_motion = hasattr(env_cfg, "commands") and hasattr(
            env_cfg.commands, "motion"
        )
        if not is_whole_body_motion or not hasattr(
            env_cfg.commands.motion, "motion_files"
        ):
            raise ValueError(
                "A motion bank (--motion_dir) was provided, but this env's "
                "command does not support multi-motion sampling. Use a "
                "motion-bank whole-body env (e.g. VegaSharpa-WholeBody-v0) or "
                "pass a single --motion_file."
            )

    is_dual_hands = hasattr(env_cfg, "commands") and hasattr(
        env_cfg.commands, "dual_hands_object_tracking_command"
    )
    is_whole_body = hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion")

    # Reference-only whole-body commands deliberately ignore objects carried
    # solely to satisfy older motion schemas.  Spawning those placeholders
    # would also inject virtual-object actions that the command cannot serve.
    uses_scene_objects = not is_whole_body or getattr(
        env_cfg.commands.motion, "uses_scene_objects", True
    )
    if uses_scene_objects:
        apply_scene_objects(env_cfg, scene_config)
        apply_scene_virtual_object_controls(env_cfg, scene_config)

    # V2D dual-hands: spawn robot + configure commands/contacts
    if is_dual_hands:
        if scene_config.robot_name:
            apply_scene_robot(
                env_cfg, scene_config, use_primitive_urdfs=use_primitive_urdfs
            )
        apply_scene_commands(env_cfg, scene_config)
        apply_scene_contact_sensors(env_cfg, scene_config)
        env_cfg.episode_length_s = (
            scene_config.episode_length_s
            / env_cfg.commands.dual_hands_object_tracking_command.motion_speed
        )

    # Whole-body: robot + actions/obs configured by env cfg; pass through one
    # motion or a bank of motions.
    elif is_whole_body:
        motion_cfg = env_cfg.commands.motion
        uses_motion_bank = bool(motion_files)
        if not hasattr(motion_cfg, "motion_file"):
            raise ValueError(
                "Whole-body motion command config must expose `motion_file`."
            )
        motion_cfg.motion_file = scene_config.motion_file
        if motion_files:
            motion_cfg.motion_files = list(motion_files)

        object_attr_names = (
            [obj.name for obj in scene_config.scene_objects]
            if uses_scene_objects
            else []
        )
        if hasattr(motion_cfg, "object_body_names"):
            motion_cfg.object_body_names = object_attr_names
            if object_attr_names and hasattr(motion_cfg, "object_name"):
                motion_cfg.object_name = object_attr_names[0]
        elif object_attr_names:
            raise ValueError(
                "Scene contains tracked objects, but the selected whole-body "
                "command config does not support object references."
            )

        # Single-motion commands use SceneConfig's duration. A motion bank owns
        # per-environment motion lengths and exact trajectory termination;
        # env_cfg.episode_length_s remains its safety ceiling.
        if not uses_motion_bank:
            whole_body_step_dt = float(env_cfg.sim.dt) * int(env_cfg.decimation)
            reset_freeze_steps = int(getattr(motion_cfg, "reset_freeze_steps", 0))
            motion_speed = float(getattr(motion_cfg, "motion_speed", 1.0))
            resolve_playback_timing(
                float(motion_cfg.dt), whole_body_step_dt, motion_speed
            )
            whole_body_episode_length_s = (
                scene_config.episode_length_s / motion_speed
                + reset_freeze_steps * whole_body_step_dt
            )
            env_cfg.episode_length_s = min(
                env_cfg.episode_length_s, whole_body_episode_length_s
            )

        # Contact sensors for whole-body
        hand_contact_bodies = getattr(motion_cfg, "hand_contact_bodies", [])
        if hand_contact_bodies:
            if not scene_config.scene_objects:
                raise ValueError(
                    "hand_contact_bodies requires at least one tracked scene object."
                )
            contact_sensor_names = []
            for obj in scene_config.scene_objects:
                obj_name = obj.name
                body_names = (
                    obj.body_names
                    if isinstance(obj, ArticulatedObjectConfig)
                    else ["object"]
                )
                for body_name in body_names:
                    for side in ["right", "left"]:
                        filter_prims = [
                            f"{{ENV_REGEX_NS}}/Robot/{b}"
                            for b in _contact_bodies_for_side(
                                scene_config, hand_contact_bodies, side
                            )
                        ]
                        sensor_name = f"{obj_name}_{body_name}_to_{side}_contact_sensor"
                        setattr(
                            env_cfg.scene,
                            sensor_name,
                            ContactSensorCfg(
                                prim_path=f"{{ENV_REGEX_NS}}/{obj_name}/{body_name}",
                                track_pose=True,
                                debug_vis=False,
                                force_threshold=0.1,
                                history_length=3,
                                filter_prim_paths_expr=filter_prims,
                                track_contact_points=True,
                                track_air_time=True,
                                max_contact_data_count_per_prim=128,
                            ),
                        )
                        contact_sensor_names.append(sensor_name)
            motion_cfg.object_contact_sensor_names = contact_sensor_names

        # FrameTransformers for hand-object observations
        hand_targets = getattr(motion_cfg, "hand_frame_target_bodies", [])
        if hand_targets:
            if not scene_config.scene_objects:
                raise ValueError(
                    "hand_frame_target_bodies requires at least one tracked scene object."
                )
            obj = scene_config.scene_objects[0]
            obj_name = obj.name
            body_name = (
                obj.body_names[0]
                if isinstance(obj, ArticulatedObjectConfig)
                else "object"
            )
            obj_prim = f"{{ENV_REGEX_NS}}/{obj_name}/{body_name}"
            for target_body in hand_targets:
                side = "left" if "left" in target_body else "right"
                setattr(
                    env_cfg.scene,
                    f"{side}_hand_object_transform",
                    FrameTransformerCfg(
                        prim_path=obj_prim,
                        target_frames=[
                            FrameTransformerCfg.FrameCfg(
                                prim_path=f"{{ENV_REGEX_NS}}/Robot/{target_body}",
                            )
                        ],
                    ),
                )

    # Visual-DR configs inject texture terms for scene-derived prims here, once object and
    # support-surface names are known -- they come from the motion file and cannot be
    # declared statically. A neutral getattr so every non-DR env cfg is unaffected.
    register_visual_dr = getattr(env_cfg, "register_scene_visual_dr_events", None)
    if register_visual_dr is not None:
        register_visual_dr()

    return env_cfg
