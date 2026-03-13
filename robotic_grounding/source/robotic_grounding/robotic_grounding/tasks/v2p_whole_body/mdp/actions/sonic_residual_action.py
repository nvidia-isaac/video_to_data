import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers.action_manager import ActionTermCfg

from .sonic_actions import SONICActionBase


class SONICResidualAction(SONICActionBase):
    """
    Residual action for SONIC.

    Action structure: [joint_residuals]
    Residuals are added to commanded joint positions for all joints.
    SONIC-controlled joints: (base_pos + residual) processed through SONIC.
    Non-SONIC joints: (base_pos + residual) applied directly.
    """

    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedRLEnv) -> None:
        """Initialize residual SONIC action term."""
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
        """Action dimension is all joints (residuals for all joints)."""
        return self._num_joints

    def process_actions(self, actions: torch.Tensor) -> None:
        """Process residual actions by adding to commanded positions."""
        self._raw_actions[:] = actions

        # Get base joint positions from command (first future frame)
        joint_pos_multi_future = self._command.command_joint_pos_multi_future
        base_joint_pos = joint_pos_multi_future[:, 0, :]  # (num_envs, num_joints)

        # Add residuals to base positions for all joints
        modified_joint_pos = base_joint_pos + actions  # (num_envs, num_joints)

        # Build SONIC observations using environment's standard observations
        sonic_obs = self._build_sonic_observations()

        # Run SONIC policy
        sonic_actions = self._policy(sonic_obs)

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

        # For non-SONIC joints, use (base_pos + residual) directly
        if len(self._direct_joint_indices) > 0:
            self._processed_actions[:, self._direct_joint_indices] = modified_joint_pos[
                :, self._direct_joint_indices
            ]
