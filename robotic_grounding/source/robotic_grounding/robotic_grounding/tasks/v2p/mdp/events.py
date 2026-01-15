from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    Note:
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu"
    ).unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)


def reset_joints(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
) -> None:
    """Reset the joints of the object and robot."""
    # Reset the object joints
    object: Articulation = env.scene[object_cfg.name]
    joint_pos = object.data.default_joint_pos[env_ids, object_cfg.joint_ids].clone()
    joint_vel = object.data.default_joint_vel[env_ids, object_cfg.joint_ids].clone()
    # TODO: add randomization
    # clamp to limits
    joint_pos_limits = object.data.soft_joint_pos_limits[env_ids, object_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    joint_vel_limits = object.data.soft_joint_vel_limits[env_ids, object_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)
    # set into the physics simulation
    object.write_joint_state_to_sim(
        joint_pos, joint_vel, joint_ids=object_cfg.joint_ids, env_ids=env_ids
    )

    # Reset the robot joints
    robot: Articulation = env.scene[robot_cfg.name]
    joint_pos = robot.data.default_joint_pos[env_ids, robot_cfg.joint_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids, robot_cfg.joint_ids].clone()
    # TODO: add randomization
    # clamp to limits
    joint_pos_limits = robot.data.soft_joint_pos_limits[env_ids, robot_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    joint_vel_limits = robot.data.soft_joint_vel_limits[env_ids, robot_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)
    # set into the physics simulation
    robot.write_joint_state_to_sim(
        joint_pos, joint_vel, joint_ids=robot_cfg.joint_ids, env_ids=env_ids
    )
