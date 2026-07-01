"""R1 Pro + Sharpa GR00T-N1.7 task registration.

Fixed-base Galaxea R1 Pro with grafted 22-DOF Sharpa hands, a tabletop scene, and the 3 GR00T
cameras (ego + 2 wrist), set up to run GR00T N1.7 REAL_R1_PRO_SHARPA inference (relative-EEF via
differential IK + absolute hand-joint targets). Standalone DirectRLEnv; other tasks untouched.
"""

import gymnasium as gym

gym.register(
    id="Isaac-R1ProSharpa-GR00T-Direct-v0",
    entry_point=f"{__name__}.r1pro_sharpa_env:R1ProSharpaEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.r1pro_sharpa_env_cfg:R1ProSharpaEnvCfg",
    },
)
