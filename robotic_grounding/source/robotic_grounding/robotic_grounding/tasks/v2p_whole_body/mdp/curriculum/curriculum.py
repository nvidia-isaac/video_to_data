# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Curriculum terms for whole-body tracking tasks.

Currently exposes a fixed-timestep curriculum that mutates the global VOC
target on the whole-body :class:`TrackingCommand` (see
``tracking_command.py``). The per-env applied scale follows the global target
through the existing per-step decay in ``TrackingCommand._update_command``.

Reward-weight scheduling is intentionally out of scope here to avoid
hard-coding reward parameter names like the v2p hand variant does.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase


class WholeBodyFixedTimestepVOCCurriculum(ManagerTermBase):
    """Fixed-timestep curriculum that schedules the VOC target scale.

    The schedule is expressed in PPO update indices and converted to env
    steps via ``num_steps_per_env``. The active stage is selected with
    ``bisect_right`` against ``env.common_step_counter``; stage transitions
    overwrite the global VOC scale on the configured command term.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv) -> None:
        """Resolve the command term and pre-compute env-step thresholds."""
        super().__init__(cfg, env)

        self._command = env.command_manager.get_term(cfg.params["command_name"])

        num_steps_per_env = int(cfg.params["num_steps_per_env"])
        timestep_schedule: list[int] = list(cfg.params["timestep_schedule"])
        scale_schedule: list[float] = list(
            cfg.params["virtual_object_control_scale_factor"]
        )
        if len(timestep_schedule) != len(scale_schedule):
            raise ValueError(
                "timestep_schedule and virtual_object_control_scale_factor must have the "
                f"same length, got {len(timestep_schedule)} and {len(scale_schedule)}."
            )

        self._env_step_schedule = [
            ppo_step * num_steps_per_env for ppo_step in timestep_schedule
        ]
        self._scale_schedule = scale_schedule
        self._last_schedule_index: int = -1

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        command_name: str,
        num_steps_per_env: int,
        timestep_schedule: list[int],
        virtual_object_control_scale_factor: list[float],
    ) -> torch.Tensor:
        """Set the global VOC target if the active stage changed.

        Args:
            env: The RL environment instance.
            env_ids: Environment ids being reset (unused; the schedule is global).
            command_name: Name of the command term to mutate.
            num_steps_per_env: PPO ``num_steps_per_env`` from the agent config.
            timestep_schedule: PPO update indices at which the VOC scale changes.
            virtual_object_control_scale_factor: VOC target scale per stage.

        Returns:
            The current global VOC scale tensor on the command term.
        """
        del env_ids, command_name  # Bound at __init__; ignored here.
        del num_steps_per_env, timestep_schedule  # Pre-computed in __init__.
        del virtual_object_control_scale_factor  # Pre-computed in __init__.

        sim_step_counter = self._env.common_step_counter
        current_schedule_index = min(
            bisect.bisect_right(self._env_step_schedule, sim_step_counter) - 1,
            len(self._scale_schedule) - 1,
        )
        current_schedule_index = max(current_schedule_index, 0)

        if current_schedule_index == self._last_schedule_index:
            return self._command.virtual_object_controller_scale_factor

        self._last_schedule_index = current_schedule_index
        new_scale = float(self._scale_schedule[current_schedule_index])

        # Preserve dtype/device of the existing (1,) tensor.
        self._command.virtual_object_controller_scale_factor = (
            0.0 * self._command.virtual_object_controller_scale_factor + new_scale
        )

        return self._command.virtual_object_controller_scale_factor
