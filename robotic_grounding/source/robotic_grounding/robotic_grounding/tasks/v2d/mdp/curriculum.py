# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import bisect
import logging
from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

logger = logging.getLogger(__name__)


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
        rewards_object_meshvert_tracking_fine: float | list[float] = 0.0,
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
