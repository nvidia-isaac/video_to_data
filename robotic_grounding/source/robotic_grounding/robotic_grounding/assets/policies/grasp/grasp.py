from dataclasses import MISSING

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass


@configclass
class GraspPolicyCfg:
    """Configuration for the G1 grasp policy."""

    asset_name: str = "robot"
    """Name of the robot asset in the scene."""

    grasp_start_threshold: float = 0.20
    """Distance at which hand starts closing. (meters)"""

    grasp_end_threshold: float = 0.05
    """Distance at which hand is fully closed. (meters)"""

    grasp_open_threshold: float = 0.10
    """Distance to end of object trajectory at which hands are opened. (meters)"""

    joint_names: list[str] = MISSING  # type: ignore[assignment]
    """List of joint names to control."""


class G1GraspPolicy:
    """Grasp policy that closes hands based on distance to object."""

    right_hand_joints_ids: tuple[int, ...]
    right_hand_joints_names: tuple[str, ...]
    left_hand_joints_ids: tuple[int, ...]
    left_hand_joints_names: tuple[str, ...]

    def __init__(self, cfg: GraspPolicyCfg, env: ManagerBasedRLEnv) -> None:
        """Initialize the grasp policy with config and environment."""
        self.cfg = cfg
        self.device = env.device

        # get joint limits for the joints in joint_names
        self.joint_ids, self.joint_names = env.scene[self.cfg.asset_name].find_joints(
            self.cfg.joint_names
        )

        # get right hand joints
        right_hand_pairs = [
            (joint_id, joint_name)
            for joint_id, joint_name in zip(
                self.joint_ids, self.joint_names, strict=True
            )
            if "right" in joint_name
        ]
        self.right_hand_joints_ids, self.right_hand_joints_names = (
            zip(*right_hand_pairs, strict=True) if right_hand_pairs else ((), ())
        )
        self.right_hand_joint_limits = env.scene[
            self.cfg.asset_name
        ].data.joint_pos_limits[:, self.right_hand_joints_ids, :]

        # get left hand joints
        left_hand_pairs = [
            (joint_id, joint_name)
            for joint_id, joint_name in zip(
                self.joint_ids, self.joint_names, strict=True
            )
            if "left" in joint_name
        ]
        self.left_hand_joints_ids, self.left_hand_joints_names = (
            zip(*left_hand_pairs, strict=True) if left_hand_pairs else ((), ())
        )
        self.left_hand_joint_limits = env.scene[
            self.cfg.asset_name
        ].data.joint_pos_limits[:, self.left_hand_joints_ids, :]

        # Create masks for joints that need inverted close direction (all thumb joints go to lower limit when closing)
        self.left_hand_invert_mask = torch.tensor(
            ["thumb" in name for name in self.left_hand_joints_names],
            dtype=torch.bool,
            device=self.device,
        )
        self.right_hand_invert_mask = torch.tensor(
            ["thumb" in name for name in self.right_hand_joints_names],
            dtype=torch.bool,
            device=self.device,
        )

    def _compute_close_amount(self, distance: torch.Tensor) -> torch.Tensor:
        """Compute close amount [0, 1] based on distance.

        - distance >= grasp_start_threshold: 0.0 (fully open)
        - distance <= grasp_end_threshold: 1.0 (fully closed)
        - in between: linear interpolation
        """
        range_size = self.cfg.grasp_start_threshold - self.cfg.grasp_end_threshold
        close_amount = 1.0 - (distance - self.cfg.grasp_end_threshold) / range_size
        return torch.clamp(close_amount, 0.0, 1.0)

    def __call__(
        self, obs: dict, env: ManagerBasedRLEnv, command_name: str = "motion"
    ) -> dict:
        """Compute hand actions from hand_policy observations."""
        obs = obs["hand_policy"]

        # Compute left hand close amount based on distance (0.0 = open, 1.0 = closed)
        left_hand_object_transform = obs[:, :7]
        left_hand_distance = torch.norm(left_hand_object_transform[:, :3], dim=-1)
        left_hand_close_amount = self._compute_close_amount(left_hand_distance)

        # Compute right hand close amount based on distance
        right_hand_object_transform = obs[:, 7:]
        right_hand_distance = torch.norm(right_hand_object_transform[:, :3], dim=-1)
        right_hand_close_amount = self._compute_close_amount(right_hand_distance)

        # Expand to all joints
        left_hand_actions = (
            left_hand_close_amount.unsqueeze(-1)
            .expand(-1, len(self.left_hand_joints_ids))
            .clone()
        )
        right_hand_actions = (
            right_hand_close_amount.unsqueeze(-1)
            .expand(-1, len(self.right_hand_joints_ids))
            .clone()
        )

        # If command is at end of object trajectory, override to open hands
        command = env.command_manager.get_term(command_name)
        end_object_pos = command._object_pos_w[-1, :3] + env.scene.env_origins
        end_open_hands = (
            torch.norm(command.object_pos_w - end_object_pos, dim=-1)
            < self.cfg.grasp_open_threshold
        )
        end_open_hands = end_open_hands.unsqueeze(-1)
        left_hand_actions = torch.where(
            end_open_hands, torch.zeros_like(left_hand_actions), left_hand_actions
        )
        right_hand_actions = torch.where(
            end_open_hands, torch.zeros_like(right_hand_actions), right_hand_actions
        )

        # Thumbs are inverted: they open at lower limit, close at upper limit
        left_hand_actions[:, self.left_hand_invert_mask] = (
            1.0 - left_hand_actions[:, self.left_hand_invert_mask]
        )
        right_hand_actions[:, self.right_hand_invert_mask] = (
            1.0 - right_hand_actions[:, self.right_hand_invert_mask]
        )

        # Scale to joint limits: 0.0 -> lower limit (open), 1.0 -> upper limit (closed)
        left_hand_actions = (
            left_hand_actions
            * (
                self.left_hand_joint_limits[:, :, 0]
                - self.left_hand_joint_limits[:, :, 1]
            )
            + self.left_hand_joint_limits[:, :, 1]
        )
        right_hand_actions = (
            right_hand_actions
            * (
                self.right_hand_joint_limits[:, :, 1]
                - self.right_hand_joint_limits[:, :, 0]
            )
            + self.right_hand_joint_limits[:, :, 0]
        )

        return {
            "left_hand_actions": left_hand_actions,
            "right_hand_actions": right_hand_actions,
        }
