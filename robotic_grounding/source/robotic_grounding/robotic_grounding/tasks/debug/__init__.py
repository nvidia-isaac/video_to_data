# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Debug environments for interactive testing and visualization."""

import gymnasium as gym

from . import agents, sharpa_debug_env_cfg

#################################################
# Register Gym environments.
#################################################

gym.register(
    id="Sharpa-V2P-Debug-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": sharpa_debug_env_cfg.SharpaDebugEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DummyPpoRunnerCfg",
    },
)
