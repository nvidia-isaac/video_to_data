"""SimToolReal screwdriver task registration (screwdriver tool + passive screw).

Reuses the simtoolreal agent yamls (network arch is object-agnostic: 140 obs, 29 actions).
"""

import gymnasium as gym

from ..simtoolreal import agents  # reuse the shared rl_games configs

gym.register(
    id="Isaac-SimToolReal-Screwdriver-Direct-v0",
    entry_point=f"{__name__}.screwdriver_env:ScrewdriverEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.screwdriver_env_cfg:ScrewdriverEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
