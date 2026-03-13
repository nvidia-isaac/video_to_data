from typing import Any

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers.action_manager import ActionTermCfg

from .sonic_actions import SONICActionBase


class SONICHierachicalAction(SONICActionBase):
    """
    Hierarchical action for SONIC.

    Action structure: [joint_commands, base_ori_6d]
    SONIC-controlled joints use SONIC output, other joints use commands directly.
    """

    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedRLEnv) -> None:
        """Initialize hierarchical SONIC action term."""
        super().__init__(cfg, env)

        # Initialize action buffers with correct dimensions
        self._processed_actions = torch.zeros(
            self._num_envs, self._num_joints, device=self._device
        )
        self._raw_actions = torch.zeros(
            self._num_envs, self.action_dim, device=self._device
        )

    @property
    def action_dim(self) -> int:
        """Return total action dimension (joints + 6D base orientation)."""
        return self._num_joints + 6

    def process_actions(self, actions: torch.Tensor) -> None:
        """Process hierarchical actions through SONIC."""
        self._raw_actions[:] = actions

        joint_commands = actions[:, : self._num_joints]
        base_ori_6d = actions[:, self._num_joints :]

        sonic_obs = self._build_sonic_observations(joint_commands, base_ori_6d)
        sonic_actions = self._policy(sonic_obs)

        sonic_actions_scaled = sonic_actions * self._scale
        if self._use_default_offset:
            sonic_actions_absolute = sonic_actions_scaled + self._joint_pos_default
            self._processed_actions[:, self._sonic_joint_indices] = (
                sonic_actions_absolute
            )
        else:
            sonic_actions_absolute = sonic_actions_scaled
            self._processed_actions[:, self._sonic_joint_indices] = sonic_actions_scaled

        self._last_sonic_actions[:] = sonic_actions

        if len(self._direct_joint_indices) > 0:
            self._processed_actions[:, self._direct_joint_indices] = joint_commands[
                :, self._direct_joint_indices
            ]

    def _build_sonic_observations(
        self, joint_commands: torch.Tensor, base_ori_6d: torch.Tensor
    ) -> dict[str, Any]:
        """Build observation dictionary for SONIC by modifying first future frame."""
        obs_manager = self._env.observation_manager
        tokenizer_term_names = obs_manager._group_obs_term_names["sonic_tokenizer"]
        tokenizer_term_cfgs = obs_manager._group_obs_term_cfgs["sonic_tokenizer"]
        tokenizer_terms = []
        modified_joint_pos = None

        # Extract SONIC joint commands (filter from all joints to SONIC joints only)
        sonic_joint_commands = joint_commands[:, self._sonic_joint_indices]

        for term_name, term_cfg in zip(
            tokenizer_term_names, tokenizer_term_cfgs, strict=True
        ):
            if term_name == "command_joint_pos_multi_future":
                original = term_cfg.func(self._env, **term_cfg.params)
                original_reshaped = original.reshape(
                    self._num_envs, self._num_future_frames, -1
                )
                original_reshaped[:, 0, :] = sonic_joint_commands
                modified_joint_pos = original_reshaped.clone()
                tokenizer_terms.append(original_reshaped.reshape(self._num_envs, -1))

            elif term_name == "command_joint_vel_multi_future":
                original = term_cfg.func(self._env, **term_cfg.params)
                original_reshaped = original.reshape(
                    self._num_envs, self._num_future_frames, -1
                )
                if modified_joint_pos is not None and self._num_future_frames > 1:
                    dt = self._command.cfg.dt_future_frames / self._command.frame_step
                    original_reshaped[:, 0, :] = (
                        modified_joint_pos[:, 1, :] - modified_joint_pos[:, 0, :]
                    ) / dt
                tokenizer_terms.append(original_reshaped.reshape(self._num_envs, -1))

            elif term_name == "motion_anchor_ori_b":
                original = term_cfg.func(self._env, **term_cfg.params)
                original_reshaped = original.reshape(
                    self._num_envs, self._num_future_frames, 6
                )
                original_reshaped[:, 0, :] = base_ori_6d
                tokenizer_terms.append(original_reshaped.reshape(self._num_envs, -1))

            else:
                tokenizer_terms.append(term_cfg.func(self._env, **term_cfg.params))

        tokenizer_obs = torch.cat(tokenizer_terms, dim=-1)
        policy_obs = self._env.obs_buf["sonic_policy"]

        return {"sonic_tokenizer": tokenizer_obs, "sonic_policy": policy_obs}
