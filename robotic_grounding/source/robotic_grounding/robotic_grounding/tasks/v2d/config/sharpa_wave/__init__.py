# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import gymnasium as gym

from . import (
    agents,
    sharpa_v2d_env_cfg,
)

#################################################
# Register Gym environments.
#################################################

gym.register(
    id="Sharpa-V2D-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": sharpa_v2d_env_cfg.SharpaV2DEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaV2DPPORunnerCfg",
    },
)

gym.register(
    id="Sharpa-V2D-v0-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": sharpa_v2d_env_cfg.SharpaV2DEnvCfgPlay,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaV2DPPORunnerCfg",
    },
)
