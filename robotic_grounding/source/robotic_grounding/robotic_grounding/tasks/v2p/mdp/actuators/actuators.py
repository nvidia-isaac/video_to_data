# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
from isaaclab.actuators import ImplicitActuator
from isaaclab.utils import DelayBuffer
from isaaclab.utils.types import ArticulationActions

if TYPE_CHECKING:
    from robotic_grounding.tasks.v2p.mdp.actuators.actuators_cfg import (
        DelayedImplicitActuatorCfg,
    )


class DelayedImplicitActuator(ImplicitActuator):
    """Implicit actuator with delayed command application.

    This class extends the :class:`ImplicitActuator` class by adding a delay to the actuator commands. The delay
    is implemented using a circular buffer that stores the actuator commands for a certain number of physics steps.
    Code borrowed from https://gitlab-master.nvidia.com/ml_nav/agile/-/tree/main/agile/rl_env/mdp/actuators?ref_type=heads
    """

    cfg: DelayedImplicitActuatorCfg

    def __init__(
        self, cfg: DelayedImplicitActuatorCfg, *args: Any, **kwargs: Any
    ) -> None:
        """Initialize the delayed implicit actuator."""
        super().__init__(cfg, *args, **kwargs)
        # instantiate the delay buffers
        self.positions_delay_buffer = DelayBuffer(
            cfg.max_delay, self._num_envs, device=self._device
        )
        self.velocities_delay_buffer = DelayBuffer(
            cfg.max_delay, self._num_envs, device=self._device
        )
        self.efforts_delay_buffer = DelayBuffer(
            cfg.max_delay, self._num_envs, device=self._device
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the delayed implicit actuator."""
        super().reset(env_ids)
        # number of environments (since env_ids can be a slice)
        if env_ids is None or env_ids == slice(None):
            num_envs = self._num_envs
        else:
            num_envs = len(env_ids)
        # set a new random delay for environments in env_ids
        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self._device,
        )
        # set delays
        self.positions_delay_buffer.set_time_lag(time_lags, env_ids)
        self.velocities_delay_buffer.set_time_lag(time_lags, env_ids)
        self.efforts_delay_buffer.set_time_lag(time_lags, env_ids)
        # reset buffers
        self.positions_delay_buffer.reset(env_ids)
        self.velocities_delay_buffer.reset(env_ids)
        self.efforts_delay_buffer.reset(env_ids)

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        """Compute the delayed implicit actuator."""
        # apply delay based on the delay the model for all the setpoints
        control_action.joint_positions = self.positions_delay_buffer.compute(
            control_action.joint_positions
        )
        control_action.joint_velocities = self.velocities_delay_buffer.compute(
            control_action.joint_velocities
        )
        control_action.joint_efforts = self.efforts_delay_buffer.compute(
            control_action.joint_efforts
        )
        # compte actuator model
        return super().compute(control_action, joint_pos, joint_vel)
