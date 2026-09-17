# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import gymnasium as gym

from . import agents  # noqa: F401
from .vega_sharpa_env_cfg import VegaSharpaWholeBodyEnvCfg  # noqa: F401
from .vega_sharpa_gr00t_env_cfg import (  # noqa: F401
    VegaSharpaGr00tInferenceEnvCfg,
    VegaSharpaGr00tJointInferenceEnvCfg,
    VegaSharpaGr00tRecordEnvCfg,
)
from .vega_sharpa_manip_env_cfg import VegaSharpaManipEnvCfg  # noqa: F401

gym.register(
    id="VegaSharpa-WholeBody-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "robotic_grounding.tasks.v2d_whole_body.config.vega_sharpa.vega_sharpa_env_cfg:VegaSharpaWholeBodyEnvCfg",
        "rsl_rl_cfg_entry_point": "robotic_grounding.tasks.v2d_whole_body.config.vega_sharpa.agents.rsl_rl_ppo_cfg:VegaSharpaWholeBodyRslRlPpoCfg",
    },
)

gym.register(
    id="VegaSharpa-WholeBody-Manip-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "robotic_grounding.tasks.v2d_whole_body.config.vega_sharpa.vega_sharpa_manip_env_cfg:VegaSharpaManipEnvCfg",
        "rsl_rl_cfg_entry_point": "robotic_grounding.tasks.v2d_whole_body.config.vega_sharpa.agents.rsl_rl_ppo_cfg:VegaSharpaWholeBodyRslRlPpoCfg",
    },
)

gym.register(
    id="VegaSharpa-WholeBody-Gr00t-Record-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "robotic_grounding.tasks.v2d_whole_body.config.vega_sharpa."
            "vega_sharpa_gr00t_env_cfg:VegaSharpaGr00tRecordEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "robotic_grounding.tasks.v2d_whole_body.config.vega_sharpa.agents."
            "rsl_rl_ppo_cfg:VegaSharpaWholeBodyRslRlPpoCfg"
        ),
    },
)

gym.register(
    id="VegaSharpa-WholeBody-Gr00t-Inference-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "robotic_grounding.tasks.v2d_whole_body.config.vega_sharpa."
            "vega_sharpa_gr00t_env_cfg:VegaSharpaGr00tInferenceEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "robotic_grounding.tasks.v2d_whole_body.config.vega_sharpa.agents."
            "rsl_rl_ppo_cfg:VegaSharpaWholeBodyRslRlPpoCfg"
        ),
    },
)

gym.register(
    id="VegaSharpa-WholeBody-Gr00t-Joint-Inference-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "robotic_grounding.tasks.v2d_whole_body.config.vega_sharpa."
            "vega_sharpa_gr00t_env_cfg:VegaSharpaGr00tJointInferenceEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "robotic_grounding.tasks.v2d_whole_body.config.vega_sharpa.agents."
            "rsl_rl_ppo_cfg:VegaSharpaWholeBodyRslRlPpoCfg"
        ),
    },
)
