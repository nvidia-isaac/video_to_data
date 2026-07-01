"""GR1-Screwdriver scene/scaffold task registration.

A standalone DirectRLEnv (NOT a SimToolReal subclass): a fixed-base Fourier GR1T2 humanoid with the
screwdriver-task objects (044 screwdriver + flat screw + thread_test) on a table. Right arm + right
hand are controllable; reward is a placeholder. A starting point to teleop / train / build on -- the
IIWA+Sharpa pretrained policy does not apply to this robot. Other tasks are untouched.
"""

import gymnasium as gym

gym.register(
    id="Isaac-GR1-Screwdriver-Direct-v0",
    entry_point=f"{__name__}.gr1_screwdriver_env:GR1ScrewdriverEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gr1_screwdriver_env_cfg:GR1ScrewdriverEnvCfg",
    },
)
