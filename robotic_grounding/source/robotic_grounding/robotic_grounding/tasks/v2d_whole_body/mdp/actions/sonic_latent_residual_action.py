# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers.action_manager import ActionTermCfg

from .sonic_actions import SONICActionBase


class SONICLatentResidualAction(SONICActionBase):
    """
    Latent residual action for SONIC.

    Action structure: [latent_residuals]
    Latent residuals are added to encoder token state before decoder inference.
    """

    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedRLEnv) -> None:
        """Initialize latent residual SONIC action term."""
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
        """Action dimension is encoder output dimension (latent space)."""
        return self._policy.encoder_output_dim

    def process_actions(self, actions: torch.Tensor) -> None:
        """Process latent residual actions by adding to token state before decoder."""
        self._raw_actions[:] = actions

        # Build SONIC observations using environment's standard observations
        sonic_obs = self._build_sonic_observations()

        # Run encoder to get base token state
        token_state = self._policy.encode(sonic_obs)

        # Add latent residuals to token state
        modified_token_state = token_state + actions

        # Run decoder with modified token state
        decoder_obs = sonic_obs["sonic_policy"]
        sonic_actions = self._policy.decode(modified_token_state, decoder_obs)

        # Scale and offset SONIC outputs
        sonic_actions_scaled = sonic_actions * self._scale
        if self._use_default_offset:
            sonic_actions_absolute = sonic_actions_scaled + self._joint_pos_default
            self._processed_actions[:, self._sonic_joint_indices] = (
                sonic_actions_absolute
            )
        else:
            self._processed_actions[:, self._sonic_joint_indices] = sonic_actions_scaled

        # Store raw SONIC output for observation functions
        self._last_sonic_actions[:] = sonic_actions

        # For non-SONIC joints, use base positions from command directly
        if len(self._direct_joint_indices) > 0:
            joint_pos_multi_future = self._command.command_joint_pos_multi_future
            base_joint_pos = joint_pos_multi_future[:, 0, :]  # (num_envs, num_joints)
            self._processed_actions[:, self._direct_joint_indices] = base_joint_pos[
                :, self._direct_joint_indices
            ]
