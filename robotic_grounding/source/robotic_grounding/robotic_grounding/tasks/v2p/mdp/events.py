# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv
from pxr import UsdGeom, UsdPhysics


def configure_collision_groups(
    env: ManagerBasedRLEnv,
    env_ids: list[int] | None,
    robot_names: list[str],
    object_names: list[str],
    fixed_object_names: list[str],
    disable_robot_to_object_collisions: bool = False,
    disable_robot_to_fixed_object_collisions: bool = True,
    disable_inter_object_collisions: bool = False,
    disable_object_to_fixed_object_collisions: bool = False,
) -> None:
    """Prestartup event to configure collision groups.

    Robot attributes are added to the RobotGroup.
    Object attributes are added to the ObjectGroup.
    Fixed object attributes are added to the FixedObjectGroup.
    ObjectGroup and FixedObjectGroup are always collide to each other.
    If disable_robot_to_object_collisions is True, the RobotGroup is filtered to not collide with the ObjectGroup.
    If disable_robot_to_fixed_object_collisions is True, the RobotGroup is filtered to not collide with the FixedObjectGroup.

    Args:
        env: The environment instance.
        env_ids: Environment IDs.
        robot_names: List of robot names to add to the RobotGroup.
        object_names: List of object names to add to the ObjectGroup.
        fixed_object_names: List of fixed object names to add to the FixedObjectGroup.
        disable_robot_to_object_collisions: Whether to disable collisions between robots and objects.
        disable_robot_to_fixed_object_collisions: Whether to disable collisions between robots and fixed objects.
        disable_inter_object_collisions: Whether to disable collisions between objects.
        disable_object_to_fixed_object_collisions: Whether to disable collisions between objects and fixed objects.
    """
    del env_ids

    stage = env.sim.stage
    num_envs = env.scene.cfg.num_envs

    # Create collision groups root
    collision_groups_root = "/World/collisionGroups"
    if not stage.GetPrimAtPath(collision_groups_root):
        UsdGeom.Xform.Define(stage, collision_groups_root)

    # Create collision group path and group entities
    robot_group_path = f"{collision_groups_root}/RobotGroup"
    robot_group = UsdPhysics.CollisionGroup.Define(stage, robot_group_path)

    object_group_path = f"{collision_groups_root}/ObjectGroup"
    object_group = UsdPhysics.CollisionGroup.Define(stage, object_group_path)

    fixed_object_group_path = f"{collision_groups_root}/FixedObjectGroup"
    fixed_object_group = UsdPhysics.CollisionGroup.Define(
        stage, fixed_object_group_path
    )

    # Populate RobotGroup — expandPrims so all descendant collision meshes are included
    robot_api = robot_group.GetCollidersCollectionAPI()
    robot_api.CreateExpansionRuleAttr().Set("expandPrims")
    for idx in range(num_envs):
        for robot_name in robot_names:
            robot_api.GetIncludesRel().AddTarget(f"/World/envs/env_{idx}/{robot_name}")

    # Populate ObjectGroup — expandPrims for descendant colliders
    object_api = object_group.GetCollidersCollectionAPI()
    object_api.CreateExpansionRuleAttr().Set("expandPrims")
    for idx in range(num_envs):
        for object_name in object_names:
            object_api.GetIncludesRel().AddTarget(
                f"/World/envs/env_{idx}/{object_name}"
            )

    # Populate FixedObjectGroup — expandPrims for descendant colliders
    fixed_object_api = fixed_object_group.GetCollidersCollectionAPI()
    fixed_object_api.CreateExpansionRuleAttr().Set("expandPrims")
    for idx in range(num_envs):
        for fixed_object_name in fixed_object_names:
            fixed_object_api.GetIncludesRel().AddTarget(
                f"/World/envs/env_{idx}/{fixed_object_name}"
            )

    # Disable collisions between robot and object groups
    if disable_robot_to_object_collisions:
        robot_group.GetFilteredGroupsRel().AddTarget(object_group_path)
        object_group.GetFilteredGroupsRel().AddTarget(robot_group_path)

    # Disable collisions between robot and fixed object groups
    if disable_robot_to_fixed_object_collisions:
        robot_group.GetFilteredGroupsRel().AddTarget(fixed_object_group_path)
        fixed_object_group.GetFilteredGroupsRel().AddTarget(robot_group_path)

    # Disable collisions between objects (intra-ObjectGroup self-filter)
    if disable_inter_object_collisions:
        object_group.GetFilteredGroupsRel().AddTarget(object_group_path)

    # Disable collisions between objects and fixed objects (support surfaces, etc.)
    if disable_object_to_fixed_object_collisions:
        object_group.GetFilteredGroupsRel().AddTarget(fixed_object_group_path)
        fixed_object_group.GetFilteredGroupsRel().AddTarget(object_group_path)
