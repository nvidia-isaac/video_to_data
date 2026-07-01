"""SimToolReal hammer task registration (claw_hammer + prismatic-jointed nail/screw).

A SEPARATE env from the screwdriver tasks: subclasses ScrewdriverEnv (HammerEnv) and
ScrewdriverEnvCfg (HammerEnvCfg), swapping the tool (claw_hammer), the goal generator (nail_traj),
and the physical screw assembly (a PRISMATIC nail-in joint instead of a revolute one). The other
tasks are untouched.
"""

import gymnasium as gym

from ..simtoolreal import agents  # reuse the shared rl_games configs

gym.register(
    id="Isaac-SimToolReal-Hammer-Direct-v0",
    entry_point=f"{__name__}.hammer_env:HammerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hammer_env_cfg:HammerEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
