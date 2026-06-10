# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import schemas
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


def apply_sdf_collision_approximations(
    env: ManagerBasedRLEnv,
    env_ids: list[int] | None,
    sdf_object_names: list[str],
    sdf_resolution: int = 256,
) -> None:
    """Prestartup event: switch named object prims to SDF mesh collision approximation.

    By default, URDF-loaded rigid objects use convex-hull (or convex-decomposition)
    collision approximation, which fills concave cavities with phantom material.
    For objects like the pour_tube vessel (open-top, non-watertight visual mesh),
    SDF preserves the actual hollow geometry — fingertips reaching into the cavity
    correctly experience no contact instead of getting trapped in phantom hull pieces.

    Walks every mesh prim under each named object's path (in all env clones) and
    applies the PhysxSDFMeshCollisionAPI via Isaac Lab's schema helper.

    Args:
        env: The environment instance.
        env_ids: Environment IDs (unused; we walk all env clones).
        sdf_object_names: List of object names (must match scene_objects attr names)
            whose collision meshes should be re-approximated as SDF.
        sdf_resolution: PhysX SDF voxel resolution. Higher = finer cavity detection
            but more GPU memory. 256 is a reasonable default for tabletop objects.
    """
    del env_ids
    if not sdf_object_names:
        return

    stage = env.sim.stage
    num_envs = env.scene.cfg.num_envs
    cfg = schemas.SDFMeshPropertiesCfg(sdf_resolution=sdf_resolution)

    for idx in range(num_envs):
        for obj_name in sdf_object_names:
            root_path = f"/World/envs/env_{idx}/{obj_name}"
            root_prim = stage.GetPrimAtPath(root_path)
            if not root_prim.IsValid():
                continue
            # Walk all descendants; apply SDF schema to every Mesh prim that has the
            # UsdPhysics.CollisionAPI applied. `apply_nested` on the helper does this
            # for us when we call with the root path.
            schemas.define_mesh_collision_properties(
                prim_path=root_path, cfg=cfg, stage=stage
            )


def disable_robot_gravity(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("right_robot"),  # noqa: B008
    disabled: bool = False,
) -> None:
    """Optionally disable per-body gravity on an articulated robot.

    Mirrors ManipTrans's imitator setup (dexhandimitator.py L248 sets
    ``disable_gravity=True`` on the hand asset) — when training the hand
    imitator alone we want to remove the constant downward gravity load so
    the policy doesn't have to learn to fight it, which is what slows
    fingertip convergence in stage1.

    Implementation: at startup, calls the underlying PhysX articulation
    view's ``set_disable_gravities`` to flip gravity off on EVERY body of
    every env instance in the articulation. The same API is used by
    ``IsaacLab/scripts/demos/bin_packing.py`` for the grocery cache.

    Args:
        env: The environment instance.
        env_ids: Environment IDs to apply the change to (None = all).
        asset_cfg: The articulation to modify. Default
            ``SceneEntityCfg("right_robot")``.
        disabled: When True, disable gravity on all bodies; when False this
            event is a no-op (so the term can stay registered in the default
            cfg without changing behaviour).
    """
    del env_ids
    if not disabled:
        return

    asset = env.scene[asset_cfg.name]
    physx_view = asset.root_physx_view

    # IsaacSim's ArticulationView exposes ``set_disable_gravities(data,
    # indices)`` where ``data`` is a (num_envs,) bool/uint8 tensor on CPU.
    # We disable gravity on every instance and let PhysX broadcast across
    # all bodies of the articulation.
    num_envs = physx_view.count
    indices = torch.arange(num_envs, dtype=torch.int32, device="cpu")
    data = torch.ones(num_envs, dtype=torch.bool, device="cpu")
    physx_view.set_disable_gravities(data, indices=indices)
    print(
        f"[disable_robot_gravity] disabled gravity on '{asset_cfg.name}' "
        f"({num_envs} env instances)"
    )
