# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

import bisect
import logging
from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

from robotic_grounding.tasks.v2p.mdp.utils import TensorDeque

logger = logging.getLogger(__name__)


class VirtualObjectControlCurriculum(ManagerTermBase):
    """Curriculum for virtual object control.

    Decay the virtual object control scale factor when:
    1. Passed the initial wait period.
    2. Passed the cooldown period.
    3. The episode reward deque is full.
    4. The mean episode length ratio exceeds the threshold.
    5. The mean episode rewards exceed the thresholds.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv) -> None:
        """Initialize the term.

        Args:
            cfg: The configuration of the curriculum.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)

        self._step_dt = self._env.step_dt

        # Command term
        self._command = env.command_manager.get_term(cfg.params["command_name"])

        # Reward manager, reward names, and weights
        self._reward_manager = env.reward_manager
        self._reward_names = list(cfg.params["reward_thresholds"].keys())
        _available_reward_names = self._env.reward_manager._term_names
        for reward_name in self._reward_names:
            assert (
                reward_name in _available_reward_names
            ), f"Reward name {reward_name} not found in available reward names {_available_reward_names}"
        self._reward_weights = torch.tensor(
            [
                self._reward_manager.get_term_cfg(reward_name).weight
                for reward_name in self._reward_names
            ],
            device=self._env.device,
        )

        # Reward thresholds
        self._reward_thresholds = torch.tensor(
            list(cfg.params["reward_thresholds"].values()), device=self._env.device
        )  # (num_reward,)

        # Reward episode previous means
        self._episode_reward_prev_means: torch.Tensor = torch.zeros(
            len(self._reward_names), device=self._env.device
        )  # (num_reward,)

        # Episode reward deque
        self._episode_reward_deque = TensorDeque(
            capacity=cfg.params["deque_maxlen"],
            feature_shape=len(self._reward_names),
            device=self._env.device,
        )

        # Episode length ratio deque
        self._episode_length_ratio_deque = TensorDeque(
            capacity=cfg.params["deque_maxlen"],
            feature_shape=1,
            device=self._env.device,
        )

        # The common step counter when the last decay was applied
        self._last_decay_common_step_counter: int = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        reward_thresholds: dict[str, float],
        episode_length_ratio_threshold: float,
        decay_mode: str,
        deque_maxlen: int,
        command_name: str,
        zero_scale_factor_threshold: float,
        initial_wait_env_steps: int,
        wait_env_steps_since_last_decay: int,
        exponential_decay_factor: float,
        linear_decay_step: float,
    ) -> torch.Tensor:
        """Apply the curriculum."""
        # 1 Add normalized episode reward to the deque
        num_reset_envs = len(env_ids)

        # 1.1 Get episode length
        L = self._env.episode_length_buf[env_ids].unsqueeze(1)  # (num_reset_envs, 1)
        L_max = self._command.tracking_lengths[env_ids].unsqueeze(
            1
        )  # (num_reset_envs, 1)
        self._episode_length_ratio_deque.append_batch(L / L_max)

        # 1.2 Get episode reward
        episode_rewards: dict[str, torch.Tensor] = getattr(
            self._reward_manager, "_episode_sums", {}
        )
        episode_rewards_batch: torch.Tensor = torch.zeros(
            num_reset_envs, len(self._reward_names), device=self._env.device
        )  # (num_reset_envs, num_reward)
        for reward_idx, reward_name in enumerate(self._reward_names):
            episode_rewards_batch[:, reward_idx] = episode_rewards[reward_name][
                env_ids
            ]  # (num_reset_envs,)
        episode_rewards_batch = episode_rewards_batch / (
            self._step_dt * self._command.retargeted_horizon
        )  # (num_reset_envs, num_reward)
        self._episode_reward_deque.append_batch(episode_rewards_batch)

        # 2 Whether the control scale is already zero
        control_scale_is_zero = (
            self._command.virtual_object_controller_scale_factor == 0.0
        )
        if control_scale_is_zero:
            return self._command.virtual_object_controller_scale_factor

        # 3 Wait initial_wait_env_steps steps until the first decay
        pass_wait_iterations = self._env.common_step_counter > initial_wait_env_steps
        if not pass_wait_iterations:
            return self._command.virtual_object_controller_scale_factor

        # 4 Wait for a cooldown period
        pass_cooldown_period = (
            self._env.common_step_counter
            > self._last_decay_common_step_counter + wait_env_steps_since_last_decay
        )
        if not pass_cooldown_period:
            return self._command.virtual_object_controller_scale_factor

        # 5 Wait until the deque is full
        queue_is_full = self._episode_reward_deque.is_full()
        if not queue_is_full:
            return self._command.virtual_object_controller_scale_factor

        # 6 Mean episode lengths exceed the thresholds
        episode_length_ratio_means = torch.mean(
            self._episode_length_ratio_deque.get_all()
        ).item()
        pass_episode_length_ratio_threshold = (
            episode_length_ratio_means > episode_length_ratio_threshold
        )

        if not pass_episode_length_ratio_threshold:
            return self._command.virtual_object_controller_scale_factor

        # 7 Mean episode rewards exceed thresholds
        episode_reward_means = torch.mean(
            self._episode_reward_deque.get_all(), dim=0
        )  # (num_reward,)
        pass_episode_reward_threshold = torch.all(
            episode_reward_means >= self._reward_thresholds
        ).item()
        self._episode_reward_prev_means[:] = episode_reward_means

        if not pass_episode_reward_threshold:
            return self._command.virtual_object_controller_scale_factor

        # 8 Apply decay, set buffer, and clear the deque
        if decay_mode == "exponential":
            self._command.virtual_object_controller_scale_factor *= (
                exponential_decay_factor
            )
        elif decay_mode == "linear":
            self._command.virtual_object_controller_scale_factor -= linear_decay_step
        else:
            raise ValueError(f"Invalid decay mode: {decay_mode}")

        if (
            self._command.virtual_object_controller_scale_factor
            <= zero_scale_factor_threshold
        ):
            self._command.virtual_object_controller_scale_factor *= 0.0

        self._last_decay_common_step_counter = self._env.common_step_counter
        self._episode_reward_deque.clear()
        self._episode_length_ratio_deque.clear()

        return self._command.virtual_object_controller_scale_factor


class FixedTimestepCurriculum(ManagerTermBase):
    """Curriculum for virtual object control with a fixed timestep decay.

    Decay the virtual object control scale factor at a pre-defined timestep schedule
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv) -> None:
        """Initialize the term.

        Args:
            cfg: The configuration of the curriculum.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)

        self._step_dt = self._env.step_dt

        # Number of simulation steps per PPO update
        self._num_steps_per_env = cfg.params["num_steps_per_env"]
        self._last_schedule_index: int = -1

        # Command term
        self._command = env.command_manager.get_term(cfg.params["command_name"])

        # Timestep schedule
        self._timestep_schedule = [
            sim_step * self._num_steps_per_env
            for sim_step in cfg.params["timestep_schedule"]
        ]
        len_decay_schedule = len(self._timestep_schedule)

        # VOC scale factor schedule
        self._voc_scale_factor_schedule = cfg.params[
            "virtual_object_control_scale_factor"
        ]
        assert (
            len(self._voc_scale_factor_schedule) == len_decay_schedule
        ), f"Length of VOC scale factor schedule must be equal to the length of timestep schedule, got {len(self._voc_scale_factor_schedule)} and {len_decay_schedule}"

        # Reward manager, reward names
        self._reward_manager = env.reward_manager
        self._schedule_reward_names = [
            key.replace("rewards_", "")
            for key in cfg.params.keys()
            if key.startswith("rewards_")
        ]
        _available_reward_names = self._env.reward_manager._term_names
        self._schedule_reward_weights = {}
        for reward_name in self._schedule_reward_names:
            assert (
                reward_name in _available_reward_names
            ), f"Reward name {reward_name} not found in available reward names {_available_reward_names}"

            if isinstance(cfg.params[f"rewards_{reward_name}"], float):
                self._schedule_reward_weights[reward_name] = [
                    cfg.params[f"rewards_{reward_name}"]
                ] * len_decay_schedule
            else:
                assert (
                    len(cfg.params[f"rewards_{reward_name}"]) == len_decay_schedule
                ), f"Length of reward {reward_name} schedule must be equal to the length of timestep schedule"
                self._schedule_reward_weights[reward_name] = cfg.params[
                    f"rewards_{reward_name}"
                ]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        command_name: str,
        num_steps_per_env: int,
        timestep_schedule: list[int],
        virtual_object_control_scale_factor: list[float],
        rewards_object_keypoints_tracking_exp: list[float],
        rewards_hand_keypoints_tracking_exp: list[float],
        rewards_hand_joint_pos_tracking_exp: list[float],
        rewards_contact_wrench_support_reward: float | list[float],
        rewards_unintended_contact_penalty: float | list[float],
        rewards_missed_contact_penalty: float | list[float],
    ) -> torch.Tensor:
        """Apply the curriculum."""
        # 1 Check if the current timestep triggers the decay, if not, return the current VOC scale factor
        sim_step_counter = self._env.common_step_counter
        current_schedule_index = min(
            bisect.bisect_right(self._timestep_schedule, sim_step_counter),
            len(self._voc_scale_factor_schedule) - 1,  # clamp to the last index
        )
        if current_schedule_index == self._last_schedule_index:
            return self._command.virtual_object_controller_scale_factor
        self._last_schedule_index = current_schedule_index

        # 2 Set the VOC scale factor for the current timestep
        self._command.virtual_object_controller_scale_factor = (
            0.0 * self._command.virtual_object_controller_scale_factor
            + self._voc_scale_factor_schedule[current_schedule_index]
        )

        # 3 Set the reward weights for the current timestep
        for reward_name in self._schedule_reward_names:
            self._reward_manager.get_term_cfg(reward_name).weight = (
                self._schedule_reward_weights[reward_name][current_schedule_index]
            )

        return self._command.virtual_object_controller_scale_factor
