import gymnasium as gym

from . import (
    agents,
    sharpa_v2p_direct_env_cfg,
    sharpa_v2p_env_cfg,
    sharpa_v2p_tracking_env_cfg,
)

#################################################
# Register Gym environments.
#################################################

gym.register(
    id="Sharpa-V2P-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": sharpa_v2p_env_cfg.SharpaV2PEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaV2PPPORunnerCfg",
    },
)

gym.register(
    id="Sharpa-V2P-v0-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": sharpa_v2p_env_cfg.SharpaV2PEnvCfgPlay,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaV2PPPORunnerCfg",
    },
)

gym.register(
    id="Sharpa-V2P-Direct-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": sharpa_v2p_direct_env_cfg.SharpaV2PDirectEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaV2PPPORunnerCfg",
    },
)

gym.register(
    id="Sharpa-V2P-Direct-v0-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": sharpa_v2p_direct_env_cfg.SharpaV2PDirectEnvCfgPlay,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaV2PPPORunnerCfg",
    },
)

gym.register(
    id="Sharpa-V2P-Tracking-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": sharpa_v2p_tracking_env_cfg.SharpaV2PTrackingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaV2PPPORunnerCfg",
    },
)

gym.register(
    id="Sharpa-V2P-Tracking-v0-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": sharpa_v2p_tracking_env_cfg.SharpaV2PTrackingEnvCfgPlay,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaV2PPPORunnerCfg",
    },
)
