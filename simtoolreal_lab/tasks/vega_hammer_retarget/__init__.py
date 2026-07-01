"""Vega hammer task driven by the pretrained left-Sharpa SAPG policy via shadow-IIWA retarget.

The policy controls the Vega RIGHT arm+hand: it drives a virtual ("shadow") IIWA arm in its trained
joint space, the shadow palm EE pose is mirrored across the robot's sagittal plane and IK'd onto the
Vega right arm, and the 22 hand DOFs pass through (mirrored). The env feeds the policy a mirrored
shadow observation. The LEFT arm is parked. The original hammer task is untouched.
"""

import gymnasium as gym

from ..simtoolreal import agents  # reuse the shared rl_games configs (140 obs / 29 actions)

gym.register(
    id="Isaac-SimToolReal-VegaHammerRetarget-Direct-v0",
    entry_point="simtoolreal_lab.tasks.vega_hammer_retarget.vega_hammer_retarget_env:VegaHammerRetargetEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vega_hammer_retarget_env_cfg:VegaHammerRetargetEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
