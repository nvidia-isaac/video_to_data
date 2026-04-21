"""SONIC action with joint-level residuals added AFTER the SONIC encoder/decoder.

RL outputs scaled per-joint residuals that are added to the SONIC decoder output.
Optionally controls finger joints via residuals on top of tracking command reference.
"""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers.action_manager import ActionTermCfg

from .sonic_actions import SONICActionBase


class SONICJointResidualAction(SONICActionBase):
    """Joint residual action: RL residuals added AFTER SONIC output.

    SONIC encoder+decoder produces base joint targets from the tracking trajectory.
    RL adds scaled per-joint residuals on top. Optionally, non-SONIC joints
    (fingers) can also receive RL residuals on top of the tracking command reference.
    """

    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedRLEnv) -> None:
        """Initialize joint residual action with optional finger control."""
        super().__init__(cfg, env)

        if cfg.hand_policy_class is not None:
            self._hand_policy = cfg.hand_policy_class(cfg.hand_policy_cfg, env)
        else:
            self._hand_policy = None

        self._robot = env.scene[cfg.asset_name]
        self._residual_scale = getattr(cfg, "residual_scale", 0.1)
        _frs = getattr(cfg, "finger_residual_scale", -1.0)
        self._finger_residual_scale = _frs if _frs >= 0 else None
        self._finger_residual = getattr(cfg, "finger_residual", False)
        self._use_tanh = getattr(cfg, "use_tanh", True)

        # Optional subset of SONIC joints for residuals
        self._residual_joint_names = getattr(cfg, "residual_joint_names", None)
        if self._residual_joint_names is not None:
            sonic_names = [
                self._robot.joint_names[i] for i in self._sonic_joint_indices
            ]
            self._residual_sonic_indices = [
                sonic_names.index(name) for name in self._residual_joint_names
            ]
            self._num_residual_joints = len(self._residual_sonic_indices)
        else:
            self._residual_sonic_indices = None  # type: ignore[assignment]
            self._num_residual_joints = self._num_sonic_joints

        self._processed_actions = torch.zeros(
            self._num_envs, self._num_joints, device=self._device
        )
        self._raw_actions = torch.zeros(
            self._num_envs, self.action_dim, device=self._device
        )

    @property
    def action_dim(self) -> int:
        """Total action dimensions: body residuals + optional finger residuals."""
        dim = self._num_residual_joints
        if self._finger_residual:
            dim += self._num_direct_joints
        return dim

    def process_actions(self, actions: torch.Tensor) -> None:
        """Apply RL residuals to SONIC output and finger commands."""
        self._raw_actions[:] = actions

        body_residuals = actions[:, : self._num_residual_joints]
        finger_residuals = (
            actions[:, self._num_residual_joints :] if self._finger_residual else None
        )

        # Run SONIC encoder+decoder on tracking data
        sonic_obs = self._build_sonic_observations()
        sonic_actions = self._policy(sonic_obs)

        # Add scaled RL residuals to SONIC output
        if self._residual_sonic_indices is not None:
            full_residuals = torch.zeros_like(sonic_actions)
            full_residuals[:, self._residual_sonic_indices] = body_residuals
        else:
            full_residuals = body_residuals

        squashed = torch.tanh(full_residuals) if self._use_tanh else full_residuals
        sonic_with_residual = sonic_actions + squashed * self._residual_scale

        # Scale and offset
        sonic_scaled = sonic_with_residual * self._scale
        if self._use_default_offset:
            self._processed_actions[:, self._sonic_joint_indices] = (
                sonic_scaled + self._joint_pos_default
            )
        else:
            self._processed_actions[:, self._sonic_joint_indices] = sonic_scaled

        self._last_sonic_actions[:] = sonic_actions

        # Finger joints
        if self._hand_policy is not None and len(self._direct_joint_indices) > 0:
            hand_obs = {"hand_policy": self._env.obs_buf["hand_policy"]}
            hand_actions = self._hand_policy(hand_obs, self._env, self.cfg.command_name)
            self._processed_actions[:, self._hand_policy.left_hand_joints_ids] = (
                hand_actions["left_hand_actions"]
            )
            self._processed_actions[:, self._hand_policy.right_hand_joints_ids] = (
                hand_actions["right_hand_actions"]
            )
        elif len(self._direct_joint_indices) > 0:
            base_joint_pos = self._command.command_joint_pos_multi_future[:, 0, :]
            if finger_residuals is not None:
                finger_scale = (
                    self._finger_residual_scale
                    if self._finger_residual_scale is not None
                    else self._residual_scale
                )
                squashed_fingers = (
                    torch.tanh(finger_residuals) if self._use_tanh else finger_residuals
                )
                self._processed_actions[:, self._direct_joint_indices] = (
                    base_joint_pos[:, self._direct_joint_indices]
                    + squashed_fingers * finger_scale
                )
            else:
                self._processed_actions[:, self._direct_joint_indices] = base_joint_pos[
                    :, self._direct_joint_indices
                ]
