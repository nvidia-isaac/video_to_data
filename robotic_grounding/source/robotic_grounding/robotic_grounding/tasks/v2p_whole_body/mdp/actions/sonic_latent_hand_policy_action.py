import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers.action_manager import ActionTermCfg

from .sonic_actions import SONICActionBase


class SONICLatentHandPolicyAction(SONICActionBase):
    """
    Latent action for SONIC with a pretrained hand policy for direct joint control.

    Action structure: [latent_state]
    Latent state is directly used as input to decoder for SONIC joints (arm).
    Direct joints (fingers) are controlled by a separate pretrained hand policy
    rather than policy outputs.
    """

    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedRLEnv) -> None:
        """Initialize latent hand policy SONIC action term."""
        super().__init__(cfg, env)

        # Initialize hand policy for direct joint (finger) control
        self._hand_policy = cfg.hand_policy_class(cfg.hand_policy_cfg, env)

        # Initialize action buffers with correct dimensions
        self._processed_actions = torch.zeros(
            self._num_envs, self._num_joints, device=self._device
        )
        self._raw_actions = torch.zeros(
            self._num_envs, self.action_dim, device=self._device
        )

    @property
    def action_dim(self) -> int:
        """Action dimension is just the encoder output dimension (latent space only)."""
        return self._policy.encoder_output_dim

    def process_actions(self, actions: torch.Tensor) -> None:
        """Process latent actions for SONIC arm control and use hand policy for fingers."""
        self._raw_actions[:] = actions

        # Interpret actions as token state (full action is latent space)
        token_state = actions

        # Ensure token state is a code in finite scalar quantizer
        token_state = self._policy.quantize(token_state)

        # Build SONIC observations using environment's standard observations
        sonic_obs = self._build_sonic_observations()

        # Run decoder with token state for arm control
        decoder_obs = sonic_obs["sonic_policy"]
        sonic_actions = self._policy.decode(token_state, decoder_obs)

        if self.cfg.debug:
            # just directly use reference motion
            sonic_obs = self._build_sonic_observations()
            sonic_actions = self._policy(sonic_obs)

        # Scale and offset SONIC outputs for arm joints
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

        # Use hand policy for direct joints (fingers)
        if len(self._direct_joint_indices) > 0:
            hand_policy_obs = self._build_hand_policy_observations()
            hand_policy_actions = self._hand_policy(
                hand_policy_obs, self._env, self.cfg.command_name
            )
            left_hand_actions = hand_policy_actions["left_hand_actions"]
            right_hand_actions = hand_policy_actions["right_hand_actions"]
            self._processed_actions[:, self._hand_policy.left_hand_joints_ids] = (
                left_hand_actions
            )
            self._processed_actions[:, self._hand_policy.right_hand_joints_ids] = (
                right_hand_actions
            )

    def _build_hand_policy_observations(self) -> dict:
        """Build observation dictionary for hand policy."""
        hand_policy_obs = self._env.obs_buf["hand_policy"]
        return {"hand_policy": hand_policy_obs}
