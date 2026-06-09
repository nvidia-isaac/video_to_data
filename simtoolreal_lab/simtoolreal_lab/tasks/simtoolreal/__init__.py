"""SimToolReal claw_hammer/swing_down task registration."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-SimToolReal-ClawHammer-Direct-v0",
    entry_point=f"{__name__}.simtoolreal_env:SimToolRealEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.simtoolreal_env_cfg:SimToolRealEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
