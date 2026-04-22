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
    6. (Optional) The mean command metric values exceed the metric_thresholds.
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

        # Cached deque statistics — updated each curriculum call and written to
        # the command term's metrics so convergence is visible in W&B even when
        # the deque is cleared after a decay.
        self._deque_reward_means = torch.zeros(
            len(self._reward_names), device=self._env.device
        )
        self._deque_reward_stds = torch.zeros(
            len(self._reward_names), device=self._env.device
        )
        self._deque_ep_len_ratio_mean = torch.zeros(1, device=self._env.device)
        self._deque_ep_len_ratio_std = torch.zeros(1, device=self._env.device)

        # Optional metric thresholds (e.g. contact_wrench_support_ratio, coverage_frac).
        # Sampled at episode end from command metrics; averaged over the deque window.
        # Entries with threshold == 0.0 are treated as disabled and excluded.
        _raw = cfg.params.get("metric_thresholds", {})
        self._metric_names: list[str] = [k for k, v in _raw.items() if float(v) > 0.0]
        self._metric_thresholds = torch.tensor(
            [float(v) for v in _raw.values() if float(v) > 0.0], device=self._env.device
        )
        if self._metric_names:
            self._metric_deque: TensorDeque | None = TensorDeque(
                capacity=cfg.params["deque_maxlen"],
                feature_shape=len(self._metric_names),
                device=self._env.device,
            )
        else:
            self._metric_deque = None
        self._deque_metric_means = torch.zeros(
            len(self._metric_names), device=self._env.device
        )
        self._deque_metric_stds = torch.zeros(
            len(self._metric_names), device=self._env.device
        )

        # Reward baseline retention: gates the NEXT decay on current deque mean
        # being >= (deque mean at last decay) * retention_ratio.
        # Trajectory-relative — handles varying plateau values across sequences.
        # Skipped before the first decay (no baseline yet).
        _raw_retention = cfg.params.get("reward_baseline_retention", {})
        # 0.0 = disabled (same convention as metric_thresholds / reward_thresholds)
        self._baseline_reward_names: list[str] = [
            k for k, v in _raw_retention.items() if float(v) > 0.0
        ]
        for rname in self._baseline_reward_names:
            assert rname in self._reward_names, (
                f"reward_baseline_retention key '{rname}' not in reward_thresholds "
                f"(available: {self._reward_names})"
            )
        self._baseline_reward_indices: list[int] = [
            self._reward_names.index(n) for n in self._baseline_reward_names
        ]
        self._baseline_retention_ratios = torch.tensor(
            [float(v) for v in _raw_retention.values() if float(v) > 0.0],
            device=self._env.device,
        )  # (num_baseline_rewards,)
        # None until the first decay fires; no retention check before first decay.
        self._reward_baselines: torch.Tensor | None = None
        # Cached for W&B logging (0 until first decay).
        self._deque_reward_baselines = torch.zeros(
            len(self._baseline_reward_names), device=self._env.device
        )

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
        fixed_schedule_steps: list | None = None,
        fixed_schedule_values: list | None = None,
        metric_thresholds: dict[str, float] | None = None,
        reward_baseline_retention: dict[str, float] | None = None,
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
            self._step_dt * L.clamp(min=1)
        )  # (num_reset_envs, num_reward) — normalize by actual episode length, not fixed horizon
        self._episode_reward_deque.append_batch(episode_rewards_batch)

        # 1.3 Update and log deque convergence statistics.
        # Written to the command term's metrics so they appear in W&B.
        # Cached so values persist across the deque clear that follows a decay.
        n = env.num_envs
        if len(self._episode_reward_deque) > 0:
            all_rewards = self._episode_reward_deque.get_all()  # (size, num_rewards)
            self._deque_reward_means = all_rewards.mean(dim=0)
            self._deque_reward_stds = (
                all_rewards.std(dim=0)
                if all_rewards.shape[0] > 1
                else torch.zeros_like(self._deque_reward_means)
            )
        if len(self._episode_length_ratio_deque) > 0:
            all_ratios = self._episode_length_ratio_deque.get_all()  # (size, 1)
            self._deque_ep_len_ratio_mean = all_ratios.mean(dim=0)
            self._deque_ep_len_ratio_std = (
                all_ratios.std(dim=0)
                if all_ratios.shape[0] > 1
                else torch.zeros_like(self._deque_ep_len_ratio_mean)
            )
        for i, name in enumerate(self._reward_names):
            self._command.metrics[f"curriculum_reward_mean_{name}"] = (
                self._deque_reward_means[i].expand(n)
            )
            self._command.metrics[f"curriculum_reward_std_{name}"] = (
                self._deque_reward_stds[i].expand(n)
            )
        self._command.metrics["curriculum_ep_len_ratio_mean"] = (
            self._deque_ep_len_ratio_mean.expand(n)
        )
        self._command.metrics["curriculum_ep_len_ratio_std"] = (
            self._deque_ep_len_ratio_std.expand(n)
        )
        for i, name in enumerate(self._baseline_reward_names):
            self._command.metrics[f"curriculum_reward_baseline_{name}"] = (
                self._deque_reward_baselines[i].expand(n)
            )

        # 1.4 Sample command metrics at episode end and update metric deque.
        # Metrics are global averages (same across all envs); we replicate num_reset_envs
        # times so the metric deque fills at the same rate as the reward deque.
        if self._metric_deque is not None:
            metric_vals = torch.stack(
                [self._command.metrics[name][0:1] for name in self._metric_names],
                dim=-1,
            )  # (1, num_metrics)
            self._metric_deque.append_batch(metric_vals.expand(num_reset_envs, -1))
            if len(self._metric_deque) > 0:
                all_metrics = self._metric_deque.get_all()  # (size, num_metrics)
                self._deque_metric_means = all_metrics.mean(dim=0)
                self._deque_metric_stds = (
                    all_metrics.std(dim=0)
                    if all_metrics.shape[0] > 1
                    else torch.zeros_like(self._deque_metric_means)
                )
            for i, name in enumerate(self._metric_names):
                self._command.metrics[f"curriculum_metric_mean_{name}"] = (
                    self._deque_metric_means[i].expand(n)
                )
                self._command.metrics[f"curriculum_metric_std_{name}"] = (
                    self._deque_metric_stds[i].expand(n)
                )

        # 2 Fixed schedule shortcut: set VOC based on common_step_counter thresholds,
        # bypassing all adaptive conditions (episode length, reward thresholds, etc.)
        if decay_mode == "fixed_schedule":
            steps = [int(s) for s in (fixed_schedule_steps or [])]
            values = [float(v) for v in (fixed_schedule_values or [])]
            current_step = self._env.common_step_counter
            current_scale = float(self._command.virtual_object_controller_scale_factor)
            # Walk through schedule: last threshold that has been passed wins
            target_scale = current_scale
            for threshold, value in zip(steps, values):
                if current_step >= threshold:
                    target_scale = value
            if target_scale != current_scale:
                env.video_trigger_pending = True
                self._command.virtual_object_controller_scale_factor = target_scale
                self._last_decay_common_step_counter = current_step
                self._episode_reward_deque.clear()
                self._episode_length_ratio_deque.clear()
                logger.info(
                    "[FixedSchedule] VOC scale %.3f → %.3f at common_step=%d",
                    current_scale,
                    target_scale,
                    current_step,
                )
            return self._command.virtual_object_controller_scale_factor

        # 3 Whether the control scale is already zero
        control_scale_is_zero = (
            self._command.virtual_object_controller_scale_factor == 0.0
        )
        if control_scale_is_zero:
            return self._command.virtual_object_controller_scale_factor

        # 4 Wait initial_wait_env_steps steps until the first decay
        pass_wait_iterations = self._env.common_step_counter > initial_wait_env_steps
        if not pass_wait_iterations:
            return self._command.virtual_object_controller_scale_factor

        # 5 Wait for a cooldown period
        pass_cooldown_period = (
            self._env.common_step_counter
            > self._last_decay_common_step_counter + wait_env_steps_since_last_decay
        )
        if not pass_cooldown_period:
            return self._command.virtual_object_controller_scale_factor

        # 6 Wait until the deque is full
        queue_is_full = self._episode_reward_deque.is_full()
        if not queue_is_full:
            return self._command.virtual_object_controller_scale_factor

        # 7 Mean episode lengths exceed the thresholds
        episode_length_ratio_means = torch.mean(
            self._episode_length_ratio_deque.get_all()
        ).item()
        pass_episode_length_ratio_threshold = (
            episode_length_ratio_means > episode_length_ratio_threshold
        )

        if not pass_episode_length_ratio_threshold:
            return self._command.virtual_object_controller_scale_factor

        # 8 Mean episode rewards exceed thresholds
        episode_reward_means = torch.mean(
            self._episode_reward_deque.get_all(), dim=0
        )  # (num_reward,)
        pass_episode_reward_threshold = torch.all(
            episode_reward_means >= self._reward_thresholds
        ).item()
        self._episode_reward_prev_means[:] = episode_reward_means

        if not pass_episode_reward_threshold:
            return self._command.virtual_object_controller_scale_factor

        # 8.3 Reward baseline retention — require current deque mean >= (baseline at last
        # decay) * retention_ratio. Trajectory-relative: prevents decay when contact quality
        # has regressed since the previous VOC level. Skipped before the first decay.
        if (
            self._reward_baselines is not None
            and len(self._baseline_reward_indices) > 0
        ):
            current_vals = episode_reward_means[self._baseline_reward_indices]
            required = self._reward_baselines * self._baseline_retention_ratios
            if not torch.all(current_vals >= required).item():
                return self._command.virtual_object_controller_scale_factor

        # 8.5 Mean command metric values exceed thresholds (if metric_thresholds configured)
        if self._metric_deque is not None and len(self._metric_deque) > 0:
            metric_means = self._metric_deque.get_all().mean(dim=0)  # (num_metrics,)
            if not torch.all(metric_means >= self._metric_thresholds).item():
                return self._command.virtual_object_controller_scale_factor

        # 9 Signal that a decay is about to happen so the video recorder can capture
        # the current policy before the scale factor changes.
        env.video_trigger_pending = True

        # 10 Apply decay, set buffer, and clear the deque
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
        if self._metric_deque is not None:
            self._metric_deque.clear()

        # Store current deque means as the baseline for the next decay's retention check.
        if len(self._baseline_reward_indices) > 0:
            self._reward_baselines = self._episode_reward_prev_means[
                self._baseline_reward_indices
            ].clone()
            self._deque_reward_baselines[:] = self._reward_baselines

        logger.info(
            "[AdaptiveCurriculum] VOC scale decayed to %.3f at common_step=%d",
            float(self._command.virtual_object_controller_scale_factor),
            self._env.common_step_counter,
        )
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
