import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers.action_manager import ActionTermCfg

from .sonic_actions import SONICActionBase


class SONICLatentAction(SONICActionBase):
    """
    Latent action for SONIC.

    Action structure: [latent_state, direct_joint_pos]
    Latent state is directly used as input to decoder.
    Direct joint positions are used for non-SONIC joints.
    """

    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedRLEnv) -> None:
        """Initialize latent SONIC action term."""
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
        """Action dimension is encoder output dimension (latent space + direct joint positions)."""
        return self._policy.encoder_output_dim + self._num_direct_joints

    def process_actions(self, actions: torch.Tensor) -> None:
        """Process latent residual actions by adding to token state before decoder."""
        self._raw_actions[:] = actions

        # Interpret actions as token state
        token_state = actions[:, : self._policy.encoder_output_dim]
        direct_joint_pos = actions[:, self._policy.encoder_output_dim :]

        # Ensure token state is a code in finite scalar quantizer
        token_state = self._policy.quantize(token_state)

        # Build SONIC observations using environment's standard observations
        sonic_obs = self._build_sonic_observations()

        # Run decoder with modified token state
        decoder_obs = sonic_obs["sonic_policy"]
        sonic_actions = self._policy.decode(token_state, decoder_obs)

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

        # For non-SONIC joints, use direct joint positions from action
        if len(self._direct_joint_indices) > 0:
            self._processed_actions[:, self._direct_joint_indices] = direct_joint_pos
