# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Curriculum terms for whole-body tracking tasks.

Exposes :class:`FixedTimestepCurriculum`, a fixed-timestep curriculum that
mutates the global VOC target on the whole-body :class:`TrackingCommand` (see
``tracking_command.py``) and optionally schedules reward-term weights. The
per-env applied scale follows the global target through the existing per-step
decay in ``TrackingCommand._update_command``.

VOC-only callers (e.g. the ReconBody env) simply omit ``reward_weight_schedules``;
an empty (timestep, scale) pair disables the term entirely (true no-op).
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase


class FixedTimestepCurriculum(ManagerTermBase):
    """Decay the virtual object control scale on a fixed timestep schedule.

    An empty (timestep_schedule, virtual_object_control_scale_factor) pair disables
    the term (no-op): the VOC scale stays at the command's initial value. Non-empty
    schedules drive both the VOC scale and per-stage reward weights.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv) -> None:
        """Initialize the term.

        Args:
            cfg: The configuration of the curriculum.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)

        self._step_dt = self._env.step_dt

        self._num_steps_per_env = cfg.params["num_steps_per_env"]
        self._last_schedule_index: int = -1

        self._command = env.command_manager.get_term(cfg.params["command_name"])

        self._timestep_schedule = [
            sim_step * self._num_steps_per_env
            for sim_step in cfg.params["timestep_schedule"]
        ]
        len_decay_schedule = len(self._timestep_schedule)

        self._voc_scale_factor_schedule = cfg.params[
            "virtual_object_control_scale_factor"
        ]
        self._disabled = (
            len(self._timestep_schedule) == 0
            and len(self._voc_scale_factor_schedule) == 0
        )
        self._schedule_reward_weights: dict[str, list[float]] = {}
        if self._disabled:
            self._schedule_reward_names = []
            return

        assert (
            len(self._voc_scale_factor_schedule) == len_decay_schedule
            or len(self._voc_scale_factor_schedule) == len_decay_schedule + 1
        ), (
            "Length of VOC scale factor schedule must equal the length of timestep "
            "schedule for legacy configs, or be one longer to provide an explicit "
            f"initial value. Got {len(self._voc_scale_factor_schedule)} and "
            f"{len_decay_schedule}."
        )
        num_schedule_values = len(self._voc_scale_factor_schedule)

        self._reward_manager = env.reward_manager
        reward_weight_schedules = dict(
            cfg.params.get("reward_weight_schedules", {}) or {}
        )
        for key, value in cfg.params.items():
            if key.startswith("rewards_"):
                reward_weight_schedules[key.replace("rewards_", "")] = value
        self._schedule_reward_names = list(reward_weight_schedules.keys())
        _available_reward_names = self._env.reward_manager._term_names
        for reward_name in self._schedule_reward_names:
            assert (
                reward_name in _available_reward_names
            ), f"Reward name {reward_name} not found in available reward names {_available_reward_names}"

            schedule = reward_weight_schedules[reward_name]
            if isinstance(schedule, (int, float)):
                self._schedule_reward_weights[reward_name] = [
                    float(schedule)
                ] * num_schedule_values
            else:
                assert len(schedule) == num_schedule_values, (
                    f"Length of reward {reward_name} schedule must match the length "
                    f"of the VOC scale schedule, got "
                    f"{len(schedule)} and "
                    f"{num_schedule_values}"
                )
                self._schedule_reward_weights[reward_name] = schedule

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        command_name: str,
        num_steps_per_env: int,
        timestep_schedule: list[int],
        virtual_object_control_scale_factor: list[float],
        reward_weight_schedules: dict[str, float | list[float]] | None = None,
    ) -> torch.Tensor:
        """Apply the curriculum."""
        del env_ids, command_name, num_steps_per_env, timestep_schedule
        del virtual_object_control_scale_factor, reward_weight_schedules

        if self._disabled:
            return self._command.virtual_object_controller_scale_factor

        sim_step_counter = self._env.common_step_counter
        current_schedule_index = min(
            bisect.bisect_right(self._timestep_schedule, sim_step_counter),
            len(self._voc_scale_factor_schedule) - 1,  # clamp to the last index
        )
        if current_schedule_index == self._last_schedule_index:
            return self._command.virtual_object_controller_scale_factor
        self._last_schedule_index = current_schedule_index

        self._command.virtual_object_controller_scale_factor = (
            0.0 * self._command.virtual_object_controller_scale_factor
            + self._voc_scale_factor_schedule[current_schedule_index]
        )

        for reward_name in self._schedule_reward_names:
            self._reward_manager.get_term_cfg(reward_name).weight = (
                self._schedule_reward_weights[reward_name][current_schedule_index]
            )

        return self._command.virtual_object_controller_scale_factor
