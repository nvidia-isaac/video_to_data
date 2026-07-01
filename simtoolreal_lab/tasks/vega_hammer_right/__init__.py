"""Native SimToolReal expert on the Vega RIGHT arm + RIGHT Sharpa hand (train-from-scratch, SAPG).

The policy is trained DIRECTLY in the Vega right arm's joint space (no shadow-IIWA / IK / mirror
retarget): it outputs the 29-dim delta-joint action that the base SimToolReal action path applies to
the right arm (0:7) + right hand (7:29); the LEFT arm+hand are parked. Reuses `HammerEnv` wholesale --
only `VegaHammerRightEnvCfg` (right-arm robot swap + training-recipe overrides) differs. The original
hammer task is untouched.
"""

import gymnasium as gym

from ..simtoolreal import agents  # reuse the shared rl_games configs (140 obs / 29 actions)

gym.register(
    id="Isaac-SimToolReal-Vega-Hammer-Right-Direct-v0",
    entry_point="simtoolreal_lab.tasks.hammer.hammer_env:HammerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vega_hammer_right_env_cfg:VegaHammerRightEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sapg_cfg.yaml",
    },
)

# #2: nail-driving STRIKE fine-tune (tighten goals + drivable nail + nail_driven reward). Warm-start
# the grasp+lift expert (00_vega_right_v4) into this so it learns the reorient+swing-down strike.
gym.register(
    id="Isaac-SimToolReal-Vega-Hammer-Right-Strike-Direct-v0",
    entry_point="simtoolreal_lab.tasks.hammer.hammer_env:HammerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vega_hammer_right_strike_env_cfg:VegaHammerRightStrikeEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sapg_cfg.yaml",
    },
)
