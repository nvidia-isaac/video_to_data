from pathlib import Path
from typing import Any

import isaaclab.utils.string as string_utils
import numpy as np
import onnxruntime as ort
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg


class SonicPolicy:
    """Class to load and inference SONIC ONNX policy checkpoints."""

    def __init__(
        self,
        policy_dir: str,
        num_fsq_levels: int = 64,
        fsq_level_list: int | list[int] = 32,
    ) -> None:
        """Load encoder and decoder ONNX sessions from policy_dir."""
        self.policy_dir = policy_dir
        encoder_path = Path(policy_dir) / "encoder_new.onnx"
        decoder_path = Path(policy_dir) / "decoder_new.onnx"

        if torch.cuda.is_available():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            provider_options = [{"device_id": 0}, {}]
        else:
            providers = ["CPUExecutionProvider"]
            provider_options = [{}]

        self.encoder = ort.InferenceSession(
            str(encoder_path), providers=providers, provider_options=provider_options
        )
        self.decoder = ort.InferenceSession(
            str(decoder_path), providers=providers, provider_options=provider_options
        )

        self.encoder_output_dim = self.encoder.get_outputs()[0].shape[1]
        self.decoder_output_dim = self.decoder.get_outputs()[0].shape[1]

        # fsq
        levels: list[int] = (
            [fsq_level_list] * num_fsq_levels
            if isinstance(fsq_level_list, int)
            else fsq_level_list
        )
        self.fsq_level_list = torch.tensor(
            levels, dtype=torch.int32, device=torch.device("cuda")
        )

    def _onnx_inference_gpu(
        self,
        model: ort.InferenceSession,
        input_tensor: torch.Tensor,
        output_shape: tuple,
        input_name: str = "obs_dict",
    ) -> torch.Tensor:
        """Run ONNX inference on GPU using zero-copy io_binding."""
        input_tensor = input_tensor.contiguous().to(dtype=torch.float32)
        output_tensor = torch.empty(
            output_shape, dtype=torch.float32, device=input_tensor.device
        )

        io_binding = model.io_binding()
        io_binding.bind_input(
            name=input_name,
            device_type="cuda",
            device_id=0,
            element_type=np.float32,
            shape=tuple(input_tensor.shape),
            buffer_ptr=input_tensor.data_ptr(),
        )

        output_name = model.get_outputs()[0].name
        io_binding.bind_output(
            name=output_name,
            device_type="cuda",
            device_id=0,
            element_type=np.float32,
            shape=tuple(output_tensor.shape),
            buffer_ptr=output_tensor.data_ptr(),
        )

        model.run_with_iobinding(io_binding)
        return output_tensor

    def __call__(self, obs: dict) -> torch.Tensor:
        """Run encoder + decoder inference."""
        encoder_obs = obs["sonic_tokenizer"]
        decoder_obs = obs["sonic_policy"]

        batch_size = encoder_obs.shape[0]
        token_state = self._onnx_inference_gpu(
            self.encoder,
            encoder_obs,
            output_shape=(batch_size, self.encoder_output_dim),
        )

        decoder_input = torch.cat([token_state, decoder_obs], dim=1)
        actions = self._onnx_inference_gpu(
            self.decoder,
            decoder_input,
            output_shape=(batch_size, self.decoder_output_dim),
        )

        return actions

    def _round_ste(self, x: torch.Tensor) -> torch.Tensor:
        """Round such that gradient can be backpropagated through the quantization."""
        return x + (torch.round(x) - x).detach()

    def quantize(self, token_state: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        """Quantize token state to finite scalar quantizer range.

        See https://arxiv.org/pdf/2309.15505.
        """
        half_l = (self.fsq_level_list - 1) * (1 + eps) / 2
        offset = torch.where(self.fsq_level_list % 2 == 0, 0.5, 0.0)
        shift = torch.atanh(offset / half_l)
        bounded_z = torch.tanh(token_state + shift) * half_l - offset
        half_width = self.fsq_level_list // 2

        return self._round_ste(bounded_z) / half_width

    def encode(self, obs: dict) -> torch.Tensor:
        """Run encoder inference only to get token state."""
        encoder_obs = obs["sonic_tokenizer"]
        batch_size = encoder_obs.shape[0]
        token_state = self._onnx_inference_gpu(
            self.encoder,
            encoder_obs,
            output_shape=(batch_size, self.encoder_output_dim),
        )
        return token_state

    def decode(
        self, token_state: torch.Tensor, decoder_obs: torch.Tensor
    ) -> torch.Tensor:
        """Run decoder inference with given token state and decoder observations."""
        batch_size = token_state.shape[0]
        decoder_input = torch.cat([token_state, decoder_obs], dim=1)
        actions = self._onnx_inference_gpu(
            self.decoder,
            decoder_input,
            output_shape=(batch_size, self.decoder_output_dim),
        )
        return actions


class SONICActionBase(ActionTerm):
    """
    Base class for all SONIC action terms.

    Provides common initialization and properties for joint mapping, scaling,
    and filtering used by all SONIC action terms.
    """

    cfg: ActionTermCfg
    _asset: Articulation
    _env: ManagerBasedRLEnv
    _policy: SonicPolicy
    _num_envs: int
    _device: torch.device

    # Joint mapping attributes
    _joint_ids: torch.Tensor
    _joint_names: list[str]
    _num_joints: int
    _sonic_joint_ids: torch.Tensor
    _sonic_joint_names: list[str]
    _num_sonic_joints: int
    _sonic_joint_indices: torch.Tensor
    _direct_joint_mask: torch.Tensor
    _direct_joint_indices: torch.Tensor

    # Command term access
    _command: Any  # TrackingCommand instance
    _num_future_frames: int

    # Action buffers
    _processed_actions: torch.Tensor
    _raw_actions: torch.Tensor
    _last_sonic_actions: torch.Tensor

    # Scaling and offset
    _use_default_offset: bool
    _joint_pos_default: torch.Tensor
    _scale: torch.Tensor

    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedRLEnv) -> None:
        """Initialize common SONIC action term components.

        Sets up joint mappings, scaling, default offsets, and action buffers.
        Derived classes should call this first, then add their specific initialization.
        """
        super().__init__(cfg, env)

        self._env = env
        self._policy = SonicPolicy(cfg.policy_dir)
        self._num_envs = env.num_envs
        self._device = env.device

        # Find all controllable joints and SONIC-controlled joints
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names)
        self._num_joints = len(self._joint_names)

        self._sonic_joint_ids, self._sonic_joint_names = self._asset.find_joints(
            cfg.sonic_joint_names
        )
        self._num_sonic_joints = len(self._sonic_joint_names)

        # Map SONIC joints to their indices in the full joint list
        self._sonic_joint_indices = torch.tensor(
            [self._joint_names.index(name) for name in self._sonic_joint_names],
            dtype=torch.long,
            device=self._device,
        )

        # Identify non-SONIC joints for direct control
        self._direct_joint_mask = torch.ones(
            self._num_joints, dtype=torch.bool, device=self._device
        )
        self._direct_joint_mask[self._sonic_joint_indices] = False
        self._direct_joint_indices = torch.where(self._direct_joint_mask)[0]
        self._num_direct_joints = len(self._direct_joint_indices)

        # Get command term for accessing base actions and future frames
        self._command = env.command_manager.get_term(cfg.command_name)
        self._num_future_frames = self._command.num_future_frames

        # Set up scaling for SONIC outputs
        self._use_default_offset = (
            cfg.use_default_offset if hasattr(cfg, "use_default_offset") else True
        )
        if self._use_default_offset:
            self._joint_pos_default = self._asset.data.default_joint_pos[
                :, self._sonic_joint_ids
            ].clone()

        if isinstance(cfg.scale, dict):
            index_list, _, value_list = string_utils.resolve_matching_names_values(
                cfg.scale, self._sonic_joint_names
            )
            self._scale = torch.ones(
                self._num_envs, self._num_sonic_joints, device=self._device
            )
            self._scale[:, index_list] = torch.tensor(value_list, device=self._device)
        else:
            self._scale = (
                torch.ones(self._num_envs, self._num_sonic_joints, device=self._device)
                * cfg.scale
            )

        # Initialize last actions buffer
        if self._use_default_offset:
            self._last_sonic_actions = self._joint_pos_default.clone()
        else:
            self._last_sonic_actions = torch.zeros(
                self._num_envs, self._num_sonic_joints, device=self._device
            )

    @property
    def policy(self) -> SonicPolicy:
        """Get the SONIC policy instance."""
        return self._policy

    @property
    def raw_actions(self) -> torch.Tensor:
        """Get the raw actions received from the agent."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Get the processed actions."""
        return self._processed_actions

    def get_sonic_joint_ids(self) -> torch.Tensor:
        """Get joint IDs for SONIC-controlled joints in the asset."""
        return self._sonic_joint_ids

    def get_sonic_joint_indices(self) -> torch.Tensor:
        """Get indices of SONIC-controlled joints in the full joint list."""
        return self._sonic_joint_indices

    def get_last_sonic_actions(self) -> torch.Tensor:
        """Get last SONIC output actions (before scaling/offset)."""
        return self._last_sonic_actions

    def _build_sonic_observations(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Build observation dictionary for SONIC from environment observations."""
        tokenizer_obs = self._env.obs_buf["sonic_tokenizer"]
        policy_obs = self._env.obs_buf["sonic_policy"]

        return {"sonic_tokenizer": tokenizer_obs, "sonic_policy": policy_obs}

    def apply_actions(self) -> None:
        """Apply processed actions to robot joints."""
        self._asset.set_joint_position_target(
            self._processed_actions, joint_ids=self._joint_ids
        )
