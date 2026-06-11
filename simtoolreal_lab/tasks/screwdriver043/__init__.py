"""SimToolReal screwdriver043 task registration (043 phillips screwdriver + screw.obj screw).

A SEPARATE env from the screwdriver task: it subclasses ScrewdriverEnv (Screwdriver043Env, which
swaps in the cross-slot goal generator) and ScrewdriverEnvCfg, swapping the tool + screw assets.
The original task is untouched.
"""

import gymnasium as gym

from ..simtoolreal import agents  # reuse the shared rl_games configs

gym.register(
    id="Isaac-SimToolReal-Screwdriver043-Direct-v0",
    entry_point=f"{__name__}.screwdriver043_env:Screwdriver043Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.screwdriver043_env_cfg:Screwdriver043EnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
