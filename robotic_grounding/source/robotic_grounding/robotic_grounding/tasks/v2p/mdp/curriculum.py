# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

import ast as _ast
import bisect
import logging
from collections.abc import Callable, Sequence

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

        # Optional upper-bound metric thresholds: metric must be BELOW the threshold.
        # Read directly from command metrics each step (not averaged via deque).
        # Use for scale-invariant stability signals like contact_wrench_support_reward_cv.
        # Entries with threshold == 0.0 are treated as disabled and excluded.
        # If metric_upper_thresholds_initial_only=True, gate only applies before the first decay.
        _raw_upper = cfg.params.get("metric_upper_thresholds", {})
        self._metric_upper_names: list[str] = [
            k for k, v in _raw_upper.items() if float(v) > 0.0
        ]
        self._metric_upper_thresholds_vals: list[float] = [
            float(v) for v in _raw_upper.values() if float(v) > 0.0
        ]
        self._metric_upper_initial_only: bool = bool(
            cfg.params.get("metric_upper_thresholds_initial_only", False)
        )

        # Custom VOC schedule (decay_mode == "custom_schedule"): explicit list of
        # VOC values to step through one-at-a-time when gates pass, with optional
        # paired reward-weight updates. Empty lists → custom_schedule disabled.
        _raw_voc = cfg.params.get("custom_voc_schedule", [])
        if isinstance(_raw_voc, str):
            _raw_voc = _ast.literal_eval(_raw_voc)
        self._custom_voc_schedule: list[float] = [float(v) for v in (_raw_voc or [])]

        _raw_reward_sched = cfg.params.get("custom_reward_schedules", {})
        self._custom_reward_schedules: dict[str, list[float]] = {}
        for _rname, _weights in (_raw_reward_sched or {}).items():
            if isinstance(_weights, str):
                _weights = _ast.literal_eval(_weights)
            _wlist = [float(w) for w in (_weights or [])]
            if _wlist:
                self._custom_reward_schedules[_rname] = _wlist

        if self._custom_voc_schedule:
            _avail = self._env.reward_manager._term_names
            for _rname, _wlist in self._custom_reward_schedules.items():
                assert (
                    _rname in _avail
                ), f"custom_reward_schedules key '{_rname}' not in rewards: {_avail}"
                assert len(_wlist) == len(self._custom_voc_schedule), (
                    f"custom_reward_schedules['{_rname}'] length {len(_wlist)} != "
                    f"custom_voc_schedule length {len(self._custom_voc_schedule)}"
                )

        # Index into custom_voc_schedule; -1 means no decay has fired yet.
        self._schedule_index: int = -1

        # Force-decay timeout: common_step when this decay opportunity first became
        # eligible (past both initial_wait and cooldown). Reset to None after each decay.
        # Used by max_eligible_wait_env_steps to force a decay if gates never fire.
        self._eligible_since_common_step: int | None = None

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

        # Deferred eval-before-decay: when env.pre_decay_eval_enabled is True,
        # decay application is deferred until the eval callback clears
        # env.pre_decay_eval_pending.  _deferred_decay_fn is a zero-arg callable
        # that applies the pending decay once the eval pass finishes.
        self._decay_deferred: bool = False
        self._deferred_decay_fn: Callable[[], None] | None = None

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
        custom_voc_schedule: list | None = None,
        custom_reward_schedules: dict | None = None,
        max_eligible_wait_env_steps: int = 0,
        metric_upper_thresholds: dict[str, float] | None = None,
        metric_upper_thresholds_initial_only: bool = False,
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

        # 1.5 Deferred eval-before-decay guard.
        # While a decay is pending an eval pass, collect episode data (section 1)
        # normally but skip all gate evaluation.  Once the eval callback clears
        # env.pre_decay_eval_pending, apply the stored decay and return.
        if self._decay_deferred:
            if not getattr(self._env, "pre_decay_eval_pending", False):
                if self._deferred_decay_fn is not None:
                    self._deferred_decay_fn()
                    self._deferred_decay_fn = None
                self._decay_deferred = False
            return self._command.virtual_object_controller_scale_factor

        # 2 Fixed schedule shortcut: set VOC based on common_step_counter thresholds,
        # bypassing all adaptive conditions (episode length, reward thresholds, etc.)
        if decay_mode == "fixed_schedule":
            steps = [int(s) for s in (fixed_schedule_steps or [])]
            values = [float(v) for v in (fixed_schedule_values or [])]
            current_step: int = int(self._env.common_step_counter)
            current_scale = float(self._command.virtual_object_controller_scale_factor)
            # Walk through schedule: last threshold that has been passed wins
            target_scale = current_scale
            for step_threshold, value in zip(steps, values, strict=False):
                if current_step >= step_threshold:
                    target_scale = value
            if target_scale != current_scale:
                if getattr(self._env, "pre_decay_eval_enabled", False):
                    _cs, _ts, _cur_s = current_step, target_scale, current_scale

                    def _apply_fixed(
                        *, _cs: int = _cs, _ts: float = _ts, _cur_s: float = _cur_s
                    ) -> None:
                        self._command.virtual_object_controller_scale_factor = _ts
                        self._last_decay_common_step_counter = _cs
                        self._episode_reward_deque.clear()
                        self._episode_length_ratio_deque.clear()
                        logger.info(
                            "[FixedSchedule] VOC scale %.3f → %.3f at common_step=%d",
                            _cur_s,
                            _ts,
                            _cs,
                        )

                    self._env.pre_decay_eval_pending = True
                    self._deferred_decay_fn = _apply_fixed
                    self._decay_deferred = True
                else:
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

        # 2.5 Custom schedule: exit early if all VOC levels already applied
        if decay_mode == "custom_schedule":
            if self._schedule_index >= len(self._custom_voc_schedule) - 1:
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
            self._eligible_since_common_step = None
            return self._command.virtual_object_controller_scale_factor

        # 5.5 Track when this decay opportunity first became eligible (past both
        # initial_wait and cooldown). If max_eligible_wait_env_steps is set and we
        # have been eligible for longer than that without the gates firing, force decay.
        current_common_step: int = int(self._env.common_step_counter)
        if self._eligible_since_common_step is None:
            self._eligible_since_common_step = current_common_step

        force_decay = (
            decay_mode == "custom_schedule"
            and max_eligible_wait_env_steps > 0
            and current_common_step - self._eligible_since_common_step
            >= max_eligible_wait_env_steps
        )
        if force_decay:
            if getattr(self._env, "pre_decay_eval_enabled", False):
                _idx = self._schedule_index + 1
                _old = float(self._command.virtual_object_controller_scale_factor)
                _new = self._custom_voc_schedule[_idx]
                _ef = current_common_step - self._eligible_since_common_step
                _ccs = current_common_step
                _slen = len(self._custom_voc_schedule)

                def _apply_force(
                    *,
                    _idx: int = _idx,
                    _old: float = _old,
                    _new: float = _new,
                    _ef: int = _ef,
                    _ccs: int = _ccs,
                    _slen: int = _slen,
                ) -> None:
                    self._schedule_index = _idx
                    self._command.virtual_object_controller_scale_factor = _new
                    for rname, weights in self._custom_reward_schedules.items():
                        self._reward_manager.get_term_cfg(rname).weight = weights[_idx]
                    self._last_decay_common_step_counter = _ccs
                    self._eligible_since_common_step = None
                    self._episode_reward_deque.clear()
                    self._episode_length_ratio_deque.clear()
                    if self._metric_deque is not None:
                        self._metric_deque.clear()
                    logger.info(
                        "[CustomSchedule] FORCED VOC %.3f → %.3f at common_step=%d (eligible for %d steps, stage %d/%d)",
                        _old,
                        _new,
                        _ccs,
                        _ef,
                        _idx + 1,
                        _slen,
                    )

                self._env.pre_decay_eval_pending = True
                self._deferred_decay_fn = _apply_force
                self._decay_deferred = True
                return self._command.virtual_object_controller_scale_factor

            env.video_trigger_pending = True
            self._schedule_index += 1
            old_voc = float(self._command.virtual_object_controller_scale_factor)
            new_voc = self._custom_voc_schedule[self._schedule_index]
            self._command.virtual_object_controller_scale_factor = new_voc
            for rname, weights in self._custom_reward_schedules.items():
                self._reward_manager.get_term_cfg(rname).weight = weights[
                    self._schedule_index
                ]
            eligible_for = current_common_step - self._eligible_since_common_step
            self._last_decay_common_step_counter = current_common_step
            self._eligible_since_common_step = None
            self._episode_reward_deque.clear()
            self._episode_length_ratio_deque.clear()
            if self._metric_deque is not None:
                self._metric_deque.clear()
            logger.info(
                "[CustomSchedule] FORCED VOC %.3f → %.3f at common_step=%d (eligible for %d steps, stage %d/%d)",
                old_voc,
                new_voc,
                current_common_step,
                eligible_for,
                self._schedule_index + 1,
                len(self._custom_voc_schedule),
            )
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

        # 8.6 Upper-bound metric thresholds — current metric must be BELOW threshold.
        # Designed for scale-invariant plateau signals like contact_wrench_support_reward_cv
        # where a LOW value indicates the reward has stabilized.
        # When metric_upper_thresholds_initial_only=True, skip this gate after the first decay.
        _upper_gate_active = self._metric_upper_names and (
            not self._metric_upper_initial_only or self._schedule_index == -1
        )
        if _upper_gate_active:
            for name, threshold in zip(  # noqa: B905
                self._metric_upper_names,
                self._metric_upper_thresholds_vals,
            ):
                if name not in self._command.metrics:
                    logger.warning(
                        "[Curriculum] metric_upper_thresholds: '%s' not found in command metrics, skipping.",
                        name,
                    )
                    continue
                current_val = self._command.metrics[name][0].item()
                if current_val > threshold:
                    return self._command.virtual_object_controller_scale_factor

        # 9 Signal that a decay is about to happen.  With pre_decay_eval_enabled,
        # defer the actual decay until the eval callback has run at the pre-decay
        # VOC level and cleared env.pre_decay_eval_pending.
        if getattr(self._env, "pre_decay_eval_enabled", False):
            _cmode = decay_mode
            _exp_f = exponential_decay_factor
            _lin_s = linear_decay_step
            _zero_t = zero_scale_factor_threshold
            _erm = episode_reward_means.clone()
            _ccs = int(self._env.common_step_counter)
            _slen = len(self._custom_voc_schedule)

            def _apply_adaptive(
                *,
                _cmode: str = _cmode,
                _exp_f: float = _exp_f,
                _lin_s: float = _lin_s,
                _zero_t: float = _zero_t,
                _erm: torch.Tensor = _erm,
                _ccs: int = _ccs,
                _slen: int = _slen,
            ) -> None:
                old_voc = float(self._command.virtual_object_controller_scale_factor)
                if _cmode == "exponential":
                    self._command.virtual_object_controller_scale_factor *= _exp_f
                elif _cmode == "linear":
                    self._command.virtual_object_controller_scale_factor -= _lin_s
                elif _cmode == "custom_schedule":
                    self._schedule_index += 1
                    new_voc = self._custom_voc_schedule[self._schedule_index]
                    self._command.virtual_object_controller_scale_factor = new_voc
                    for rname, weights in self._custom_reward_schedules.items():
                        self._reward_manager.get_term_cfg(rname).weight = weights[
                            self._schedule_index
                        ]
                    logger.info(
                        "[CustomSchedule] VOC %.3f → %.3f at common_step=%d (stage %d/%d)",
                        old_voc,
                        new_voc,
                        _ccs,
                        self._schedule_index + 1,
                        _slen,
                    )
                else:
                    raise ValueError(f"Invalid decay mode: {_cmode}")
                if _cmode != "custom_schedule" and (
                    self._command.virtual_object_controller_scale_factor <= _zero_t
                ):
                    self._command.virtual_object_controller_scale_factor *= 0.0
                self._last_decay_common_step_counter = _ccs
                self._eligible_since_common_step = None
                self._episode_reward_deque.clear()
                self._episode_length_ratio_deque.clear()
                if self._metric_deque is not None:
                    self._metric_deque.clear()
                if len(self._baseline_reward_indices) > 0:
                    self._reward_baselines = _erm[self._baseline_reward_indices].clone()
                    self._deque_reward_baselines[:] = self._reward_baselines
                logger.info(
                    "[AdaptiveCurriculum] VOC scale decayed to %.3f at common_step=%d",
                    float(self._command.virtual_object_controller_scale_factor),
                    _ccs,
                )

            self._env.pre_decay_eval_pending = True
            self._deferred_decay_fn = _apply_adaptive
            self._decay_deferred = True
            return self._command.virtual_object_controller_scale_factor

        env.video_trigger_pending = True

        # 10 Apply decay, set buffer, and clear the deque
        if decay_mode == "exponential":
            self._command.virtual_object_controller_scale_factor *= (
                exponential_decay_factor
            )
        elif decay_mode == "linear":
            self._command.virtual_object_controller_scale_factor -= linear_decay_step
        elif decay_mode == "custom_schedule":
            self._schedule_index += 1
            old_voc = float(self._command.virtual_object_controller_scale_factor)
            new_voc = self._custom_voc_schedule[self._schedule_index]
            self._command.virtual_object_controller_scale_factor = new_voc
            for rname, weights in self._custom_reward_schedules.items():
                self._reward_manager.get_term_cfg(rname).weight = weights[
                    self._schedule_index
                ]
            logger.info(
                "[CustomSchedule] VOC %.3f → %.3f at common_step=%d (stage %d/%d)",
                old_voc,
                new_voc,
                self._env.common_step_counter,
                self._schedule_index + 1,
                len(self._custom_voc_schedule),
            )
        else:
            raise ValueError(f"Invalid decay mode: {decay_mode}")

        if decay_mode != "custom_schedule" and (
            self._command.virtual_object_controller_scale_factor
            <= zero_scale_factor_threshold
        ):
            self._command.virtual_object_controller_scale_factor *= 0.0

        self._last_decay_common_step_counter = int(self._env.common_step_counter)
        self._eligible_since_common_step = None
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

        # Termination manager — per-step threshold schedules.
        # Keys prefixed with "termination_<term_name>_<param>" define a list of
        # values (one per schedule step) that override the named termination
        # term's params at each curriculum transition.
        # E.g. "termination_object_away_from_trajectory_position_threshold"
        # maps to env.termination_manager.get_term_cfg(
        #     "object_away_from_trajectory").params["position_threshold"].
        self._termination_manager = env.termination_manager
        self._schedule_termination_params: dict[tuple[str, str], list] = {}
        for key in cfg.params.keys():
            if not key.startswith("termination_"):
                continue
            # Convention: termination_<term_name>_<param_name>
            # Try to match known termination terms greedily.
            suffix = key[len("termination_") :]
            term_name = param_name = None
            for tname in self._termination_manager._term_names:
                if suffix.startswith(tname + "_"):
                    term_name = tname
                    param_name = suffix[len(tname) + 1 :]
                    break
            if term_name is None or param_name is None:
                continue
            vals = cfg.params[key]
            if vals is None:
                continue  # None = no override for this termination param
            if isinstance(vals, (int, float)):
                vals = [float(vals)] * len_decay_schedule
            else:
                assert len(vals) == len_decay_schedule, (
                    f"Termination schedule '{key}' must have {len_decay_schedule} "
                    f"entries (one per timestep_schedule entry), got {len(vals)}"
                )
            self._schedule_termination_params[(term_name, param_name)] = list(vals)

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
        rewards_object_keypoints_tracking_refine: float | list[float] = 0.0,
        rewards_object_meshvert_tracking_fine: float | list[float] = 0.0,
        rewards_object_position_tracking_fine: float | list[float] = 0.0,
        rewards_object_velocity_tracking_exp: float | list[float] = 0.0,
        rewards_hand_skeleton_tracking_exp: float | list[float] = 0.0,
        rewards_dexmachina_contact_tracking_reward: float | list[float] = 0.0,
        rewards_relative_object_pos_reward: float | list[float] = 0.0,
        rewards_relative_object_rot_reward: float | list[float] = 0.0,
        rewards_inter_object_proximity_reward: float | list[float] = 0.0,
        termination_object_away_from_trajectory_position_threshold: (
            list[float] | None
        ) = None,
        termination_object_away_from_trajectory_orientation_threshold: (
            list[float] | None
        ) = None,
        termination_hand_wrist_away_from_trajectory_threshold: (
            list[float] | None
        ) = None,
    ) -> torch.Tensor:
        """Apply the curriculum."""
        # 1 Check if the current timestep triggers the decay, if not, return the current VOC scale factor
        sim_step_counter = self._env.common_step_counter
        current_schedule_index = min(
            bisect.bisect_right(self._timestep_schedule, sim_step_counter),
            len(self._voc_scale_factor_schedule) - 1,  # clamp to the last index
        )
        # DIAG: print common_step + idx every 200 sim-steps so we can confirm
        # whether common_step is growing across resume + curriculum stuck.
        if (
            sim_step_counter % 200
        ) == 0 or current_schedule_index != self._last_schedule_index:
            print(
                f"[curr/voc] common_step={sim_step_counter} idx={current_schedule_index} "
                f"last_idx={self._last_schedule_index} sched_len={len(self._timestep_schedule)} "
                f"voc_target={self._voc_scale_factor_schedule[current_schedule_index]}",
                flush=True,
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

        # 4 Set per-step termination thresholds (if any scheduled)
        for (
            term_name,
            param_name,
        ), schedule in self._schedule_termination_params.items():
            self._termination_manager.get_term_cfg(term_name).params[param_name] = (
                schedule[current_schedule_index]
            )

        return self._command.virtual_object_controller_scale_factor


class TerminationAnnealingCurriculum(ManagerTermBase):
    """Anneal termination thresholds based on episode-length-ratio gating.

    Stage ladder runs from loose (stage 0) to tight (stage N-1). Each stage
    triples (object_position_threshold, object_orientation_threshold,
    wrist_threshold). Advance to next stage when ALL of:

      * `len(window) >= window_iters * num_steps_per_env`     (full window)
      * `mean(ep_len_ratio over window) >= advance_threshold` (policy competent)
      * `iters_in_stage >= min_dwell_iters * num_steps_per_env` (mins dwell)
      * `current_stage < max_stage`                           (have headroom)

    `ep_len_ratio` is the just-completed episode length divided by the
    per-sequence horizon (tracking_lengths from the command term), so the
    metric is sequence-agnostic.

    The curriculum is invoked only on env reset (via curriculum_manager.compute
    in _reset_idx). At call time, env.episode_length_buf[env_ids] holds the
    final length of each just-completed episode.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv) -> None:
        """Initialize ACv2 curriculum from config params."""
        super().__init__(cfg, env)
        self._num_steps_per_env = int(cfg.params["num_steps_per_env"])
        self._stages: list[tuple[float, ...]] = [
            tuple(float(v) for v in s) for s in cfg.params["stages"]
        ]
        self._advance_threshold = float(cfg.params["advance_threshold"])
        self._window_env_steps = (
            int(cfg.params["window_iters"]) * self._num_steps_per_env
        )
        self._min_dwell_env_steps = (
            int(cfg.params["min_dwell_iters"]) * self._num_steps_per_env
        )
        # VOC gate: stages only advance once virtual_object_controller_scale_factor
        # has decayed below this threshold. Prevents the curriculum from rocketing
        # through stages during the VOC-assisted phase (ep_len_ratio is artificially
        # high because the controller drives the object regardless of policy).
        # Default 0.1 — matches the proposed gate in take3_fresh_recipe_proposal.md.
        self._voc_gate_threshold = float(cfg.params.get("voc_gate_threshold", 0.1))
        self._command = env.command_manager.get_term(cfg.params["command_name"])
        self._termination_manager = env.termination_manager

        # Per-env-step state: stage index, when we entered this stage, and a
        # ring buffer of (common_step, ep_len_ratio) tuples for window-mean.
        self._current_stage = 0
        self._stage_entry_common_step = 0
        # list of (common_step, ratio) — pruned by _trim_window().
        self._ep_len_ratio_history: list[tuple[int, float]] = []

        # Apply stage 0 immediately so the policy starts from the loosest
        # thresholds even before the first env reset triggers __call__.
        self._apply_stage(self._current_stage)

    def _apply_stage(self, stage_idx: int) -> None:
        pos, ori, wrist = self._stages[stage_idx]
        try:
            obj_cfg = self._termination_manager.get_term_cfg(
                "object_away_from_trajectory"
            )
            obj_cfg.params["position_threshold"] = pos
            obj_cfg.params["orientation_threshold"] = ori
        except ValueError:
            pass
        try:
            wrist_cfg = self._termination_manager.get_term_cfg(
                "hand_wrist_away_from_trajectory"
            )
            wrist_cfg.params["threshold"] = wrist
        except ValueError:
            pass
        logger.info(
            f"[termination_annealing] stage {stage_idx}: "
            f"pos={pos:.3f} ori={ori:.3f} wrist={wrist:.3f}"
        )

    def _trim_window(self, current_step: int) -> None:
        cutoff = current_step - self._window_env_steps
        if cutoff <= 0:
            return
        self._ep_len_ratio_history = [
            (s, r) for s, r in self._ep_len_ratio_history if s >= cutoff
        ]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        command_name: str,
        num_steps_per_env: int,
        stages: list[list[float]],
        advance_threshold: float,
        window_iters: int,
        min_dwell_iters: int,
        voc_gate_threshold: float = 0.1,
    ) -> torch.Tensor:
        """Apply ACv2 adaptive VOC curriculum step."""
        current_step = self._env.common_step_counter

        # Record just-completed episode lengths. At curriculum-compute time
        # (called from _reset_idx BEFORE scene.reset), episode_length_buf still
        # holds the pre-reset values.
        if env_ids is not None and len(env_ids) > 0:
            L = self._env.episode_length_buf[env_ids].float()
            L_max = self._command.tracking_lengths[env_ids].float().clamp(min=1.0)
            ratios = (L / L_max).cpu().tolist()
            for r in ratios:
                self._ep_len_ratio_history.append((current_step, r))

        self._trim_window(current_step)

        # VOC gate: only advance stages once VOC assistance has nearly fully
        # decayed. While VOC > voc_gate_threshold, the object is being driven
        # by the controller and ep_len_ratio is artificially high (every
        # episode reaches horizon regardless of policy quality). Without this
        # gate, the curriculum rockets through all stages during the VOC=1.0
        # phase and the policy never benefits from loose thresholds when
        # assistance is actually withdrawn.
        try:
            _voc_raw = self._command.virtual_object_controller_scale_factor
            voc_scalar = float(
                _voc_raw.mean().item() if hasattr(_voc_raw, "mean") else _voc_raw
            )
        except Exception:
            voc_scalar = 0.0  # If we can't read VOC, default to "gate open"

        # Eligible to advance?
        in_stage_for = current_step - self._stage_entry_common_step
        if (
            self._current_stage < len(self._stages) - 1
            and in_stage_for >= self._min_dwell_env_steps
            and len(self._ep_len_ratio_history) > 0
            and voc_scalar <= self._voc_gate_threshold
        ):
            # Require at least a partial window of data
            mean_ratio = sum(r for _, r in self._ep_len_ratio_history) / len(
                self._ep_len_ratio_history
            )
            if mean_ratio >= self._advance_threshold:
                self._current_stage += 1
                self._stage_entry_common_step = current_step
                self._ep_len_ratio_history.clear()
                self._apply_stage(self._current_stage)

        return torch.tensor(float(self._current_stage), device=self._env.device)
