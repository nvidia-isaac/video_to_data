# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import gymnasium as gym

from . import agents  # noqa: F401
from .g1_sonic_env_cfg import (  # noqa: F401
    G1_SONIC_JOINT_NAMES,
    G1SonicEnvCfg,
    G1SonicReconBodyEnvCfg,
    G1SonicReconHandEnvCfg,
    G1SonicReconHandStage1EnvCfg,
    G1SonicReconHandStage2EnvCfg,
    G1SonicReconHandStage3EnvCfg,
)

gym.register(
    id="SonicG1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "robotic_grounding.tasks.v2d_whole_body.config.sonic.g1.g1_sonic_env_cfg:G1SonicEnvCfg",
        "rsl_rl_cfg_entry_point": "robotic_grounding.tasks.v2d_whole_body.config.sonic.g1.agents.rsl_rl_ppo_cfg:G1SonicRslRlPpoCfg",
    },
)

gym.register(
    id="SonicG1-ReconBody-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "robotic_grounding.tasks.v2d_whole_body.config.sonic.g1.g1_sonic_env_cfg:G1SonicReconBodyEnvCfg",
        "rsl_rl_cfg_entry_point": "robotic_grounding.tasks.v2d_whole_body.config.sonic.g1.agents.rsl_rl_ppo_cfg:G1SonicRslRlPpoCfg",
    },
)

gym.register(
    id="SonicG1-ReconHand-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "robotic_grounding.tasks.v2d_whole_body.config.sonic.g1.g1_sonic_env_cfg:G1SonicReconHandEnvCfg",
        "rsl_rl_cfg_entry_point": "robotic_grounding.tasks.v2d_whole_body.config.sonic.g1.agents.rsl_rl_ppo_cfg:G1SonicReconHandRslRlPpoCfg",
    },
)

# ReconHand variant using a fixed episode-length timeout instead of the
# trajectory-end timestep termination.
gym.register(
    id="SonicG1-ReconHand-EpisodeTimeout-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "robotic_grounding.tasks.v2d_whole_body.config.sonic.g1.g1_sonic_env_cfg:G1SonicReconHandEpisodeTimeoutEnvCfg",
        "rsl_rl_cfg_entry_point": "robotic_grounding.tasks.v2d_whole_body.config.sonic.g1.agents.rsl_rl_ppo_cfg:G1SonicReconHandRslRlPpoCfg",
    },
)

# ReconHand training stages — bake the recipe so runs need only per-run flags
# (--motion_file, resume, --run_name). Stage 1/2 use the fixed episode-length
# timeout; stage 3 (full-sequence finetune) uses the base trajectory-end timeout.
gym.register(
    id="SonicG1-ReconHand-Stage1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "robotic_grounding.tasks.v2d_whole_body.config.sonic.g1.g1_sonic_env_cfg:G1SonicReconHandStage1EnvCfg",
        "rsl_rl_cfg_entry_point": "robotic_grounding.tasks.v2d_whole_body.config.sonic.g1.agents.rsl_rl_ppo_cfg:G1SonicReconHandStage1RslRlPpoCfg",
    },
)

gym.register(
    id="SonicG1-ReconHand-Stage2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "robotic_grounding.tasks.v2d_whole_body.config.sonic.g1.g1_sonic_env_cfg:G1SonicReconHandStage2EnvCfg",
        "rsl_rl_cfg_entry_point": "robotic_grounding.tasks.v2d_whole_body.config.sonic.g1.agents.rsl_rl_ppo_cfg:G1SonicReconHandStage2RslRlPpoCfg",
    },
)

gym.register(
    id="SonicG1-ReconHand-Stage3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "robotic_grounding.tasks.v2d_whole_body.config.sonic.g1.g1_sonic_env_cfg:G1SonicReconHandStage3EnvCfg",
        "rsl_rl_cfg_entry_point": "robotic_grounding.tasks.v2d_whole_body.config.sonic.g1.agents.rsl_rl_ppo_cfg:G1SonicReconHandStage3RslRlPpoCfg",
    },
)
