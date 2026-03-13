# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Debug environments for interactive testing and visualization."""

import gymnasium as gym

from . import agents, sharpa_debug_env_cfg, vega_sharpa_debug_env_cfg

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

gym.register(
    id="Vega-Sharpa-Debug-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": vega_sharpa_debug_env_cfg.VegaSharpaDebugEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DummyPpoRunnerCfg",
    },
)
