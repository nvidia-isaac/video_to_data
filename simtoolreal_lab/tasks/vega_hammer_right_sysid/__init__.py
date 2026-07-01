"""Native SimToolReal Vega RIGHT-arm expert with REAL-WORLD SYSID arm PD (train-from-scratch, SAPG).

Same env + recipe as `vega_hammer_right` (reuses `HammerEnv`); only the arm stiffness/damping change
to the real harmonic-drive sysid fit. The original hammer task is untouched.
"""

import gymnasium as gym

from ..simtoolreal import agents  # reuse the shared rl_games configs (140 obs / 29 actions)

gym.register(
    id="Isaac-SimToolReal-Vega-Hammer-Right-Sysid-Direct-v0",
    entry_point="simtoolreal_lab.tasks.hammer.hammer_env:HammerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vega_hammer_right_sysid_env_cfg:VegaHammerRightSysidEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sapg_cfg.yaml",
    },
)
