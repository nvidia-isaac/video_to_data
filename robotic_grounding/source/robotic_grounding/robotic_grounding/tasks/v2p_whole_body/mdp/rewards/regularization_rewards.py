# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch

# Import PyTorch Ignite MMD
from ignite.metrics import MaximumMeanDiscrepancy
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers.manager_term_cfg import RewardTermCfg

from robotic_grounding.tasks.v2p_whole_body.utils import MotionDataset


class body_acc_l2(ManagerTermBase):  # noqa: N801
    """
    Directly copied from agile.rl_env.mdp.rewards.aesthetic_rewards.py.

    Penalize body linear and angular accelerations using velocity history tracking (Isaac Gym style).

    This reward term computes world-frame accelerations for a specified body/link.
    If no body_names is specified in asset_cfg, it defaults to using the root link.

    Usage:
        # For root acceleration (default):
        body_acc = RewTerm(func=body_acc_l2, weight=-0.01)

        # For a specific link:
        torso_acc = RewTerm(
            func=body_acc_l2,
            weight=-0.01,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"])},
        )
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv) -> None:
        """Initialize velocity history buffer and resolve body index."""
        super().__init__(cfg, env)

        # Initialize velocity history buffer
        # Shape: [num_envs, 6] where 6 = 3 (lin_vel) + 3 (ang_vel)
        self.prev_body_vel = torch.zeros(
            env.num_envs, 6, device=env.device, dtype=torch.float32
        )

        # Flag to track if this is the first call (skip acceleration computation)
        self.first_call = True

        # Resolve body index if body_names is provided
        self._body_idx: int | None = None
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        if asset_cfg.body_names is not None:
            asset: Articulation = env.scene[asset_cfg.name]
            self._body_idx = asset.find_bodies(asset_cfg.body_names)[0][0]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg | None = None,
    ) -> torch.Tensor:
        """Compute body acceleration penalty by tracking velocity changes in world frame."""
        if asset_cfg is None:
            asset_cfg = SceneEntityCfg("robot")
        # Extract the robot asset
        robot: Articulation = env.scene[asset_cfg.name]

        # Get current velocities (both linear and angular) in world frame
        if self._body_idx is not None:
            # Use specified body velocities
            current_lin_vel = robot.data.body_lin_vel_w[
                :, self._body_idx, :
            ]  # [num_envs, 3]
            current_ang_vel = robot.data.body_ang_vel_w[
                :, self._body_idx, :
            ]  # [num_envs, 3]
        else:
            # Default to root velocities
            current_lin_vel = robot.data.root_lin_vel_w  # [num_envs, 3]
            current_ang_vel = robot.data.root_ang_vel_w  # [num_envs, 3]

        # Concatenate to form 6D velocity vector
        current_body_vel = torch.cat(
            [current_lin_vel, current_ang_vel], dim=-1
        )  # [num_envs, 6]

        if self.first_call:
            # First call: initialize previous velocity and return zeros
            self.prev_body_vel.copy_(current_body_vel)
            self.first_call = False
            return torch.zeros(env.num_envs, device=env.device)

        # Compute acceleration as velocity difference over timestep
        body_acc = (current_body_vel - self.prev_body_vel) / env.step_dt

        # Update velocity history for next call
        self.prev_body_vel.copy_(current_body_vel)

        # Compute L2 penalty on accelerations (sum of squared accelerations)
        return torch.sum(torch.square(body_acc), dim=-1)


def body_ang_vel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """
    Directly copied from agile.rl_env.mdp.rewards.aesthetic_rewards.py.

    Penalize body angular velocity using L2 norm.

    This reward penalizes high angular velocities of a specified body/link,
    useful for reducing shaking/oscillations without affecting linear motion.
    If no body_names is specified in asset_cfg, defaults to using the root link.

    Usage:
        # For root angular velocity:
        root_ang_vel = RewTerm(func=body_ang_vel_l2, weight=-0.01)

        # For a specific link (e.g., torso):
        torso_ang_vel = RewTerm(
            func=body_ang_vel_l2,
            weight=-0.01,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"])},
        )

    Args:
        env: The environment.
        asset_cfg: Asset configuration. Use body_names to specify a link, otherwise uses root.

    Returns:
        L2 norm of the body's angular velocity (sum of squared components).
    """
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    # Extract the robot asset
    robot: Articulation = env.scene[asset_cfg.name]

    # Get angular velocity based on whether body_names is specified
    if asset_cfg.body_ids is not None and len(asset_cfg.body_ids) > 0:
        # Use specified body angular velocity
        body_idx = asset_cfg.body_ids[0]
        ang_vel = robot.data.body_ang_vel_w[:, body_idx, :]  # [num_envs, 3]
    else:
        # Default to root angular velocity
        ang_vel = robot.data.root_ang_vel_w  # [num_envs, 3]

    # Compute L2 penalty (sum of squared angular velocities)
    return torch.sum(torch.square(ang_vel), dim=-1)


class mmd_similarity_reward(ManagerTermBase):  # noqa: N801
    """Reward for encouraging natural robot pose distributions using Maximum Mean Discrepancy."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv) -> None:
        """Initialize the MMD similarity reward."""
        super().__init__(cfg, env)

        # Extract parameters
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.robot: Articulation = env.scene[asset_cfg.name]
        dataset_path = cfg.params.get("dataset_path")
        joint_order_file = cfg.params.get("joint_order_file", None)
        self.variance = cfg.params.get(
            "variance", 10.0
        )  # Higher variance for better discrimination
        self.reward_scale = cfg.params.get("reward_scale", 1.0)
        self.device = env.device

        if dataset_path is None:
            raise ValueError("dataset_path must be provided for mmd_similarity_reward")

        # Load motion dataset
        self.motion_dataset = MotionDataset(
            dataset_path=dataset_path,
            robot=self.robot,
            joint_order_file=joint_order_file,
            device=env.device,
        )
        self.expert_joint_indices = self.motion_dataset.get_joint_indices_for_robot()

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        dataset_path: str,
        joint_order_file: str | None = None,
        variance: float = 1.0,
        asset_cfg: SceneEntityCfg | None = None,
    ) -> torch.Tensor:
        """Compute the MMD-based motion distribution similarity reward."""
        if asset_cfg is None:
            asset_cfg = SceneEntityCfg("robot")
        # Get current robot joint positions (all environments)
        robot_joint_pos = self.robot.data.joint_pos
        agent_batch = robot_joint_pos[:, self.expert_joint_indices]
        num_envs = agent_batch.shape[0]

        # Sample num_envs expert poses directly from dataset
        expert_sample = self.motion_dataset.sample(num_envs)

        # Compute MMD between agent batch and expert sample
        mmd_metric = MaximumMeanDiscrepancy(var=self.variance, device=self.device)
        mmd_metric.update((agent_batch, expert_sample))
        mmd_value = mmd_metric.compute()

        # Convert to tensor if float
        if not isinstance(mmd_value, torch.Tensor):
            mmd_value = torch.tensor(mmd_value, device=self.device)

        # Convert MMD to reward
        reward = torch.exp(-self.reward_scale * mmd_value)

        # Return same reward for all environments
        return reward.expand(num_envs)
