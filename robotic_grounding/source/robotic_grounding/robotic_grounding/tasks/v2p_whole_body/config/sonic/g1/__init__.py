import gymnasium as gym

from . import agents  # noqa: F401
from .g1_sonic_env_cfg import (  # noqa: F401
    G1_SONIC_JOINT_NAMES,
    G1SonicEEEnvCfg,
    G1SonicEnvCfg,
)

gym.register(
    id="SonicG1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "robotic_grounding.tasks.v2p_whole_body.config.sonic.g1.g1_sonic_env_cfg:G1SonicEnvCfg",
        "rsl_rl_cfg_entry_point": "robotic_grounding.tasks.v2p_whole_body.config.sonic.g1.agents.rsl_rl_ppo_cfg:G1SonicRslRlPpoCfg",
    },
)

gym.register(
    id="SonicG1-EE-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "robotic_grounding.tasks.v2p_whole_body.config.sonic.g1.g1_sonic_env_cfg:G1SonicEEEnvCfg",
        "rsl_rl_cfg_entry_point": "robotic_grounding.tasks.v2p_whole_body.config.sonic.g1.agents.rsl_rl_ppo_cfg:G1SonicRslRlPpoCfg",
    },
)
