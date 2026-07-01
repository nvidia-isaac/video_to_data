"""SimToolReal hammer task with the Vega + left-Sharpa robot (robot swap; logic unchanged).

Reuses the existing `HammerEnv` wholesale -- the base env now reads the robot's joint/palm/fingertip
names from cfg, so only `VegaHammerEnvCfg` (swapped robot_cfg + Vega names) differs. The original
`Isaac-SimToolReal-Hammer-Direct-v0` task is untouched.
"""

import gymnasium as gym

from ..simtoolreal import agents  # reuse the shared rl_games configs (140 obs / 29 actions)

gym.register(
    id="Isaac-SimToolReal-Vega-Hammer-Direct-v0",
    entry_point="simtoolreal_lab.tasks.hammer.hammer_env:HammerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vega_hammer_env_cfg:VegaHammerEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
