# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from isaaclab.utils import configclass

from robotic_grounding.tasks.v2p.v2p_hand_env_cfg import V2PHandEnvCfg

_DEFAULT_MOTION_FILE = "arctic_processed/arctic_s01_box_grab_01/sharpa_wave"


@configclass
class SharpaV2PEnvCfg(V2PHandEnvCfg):
    """Configuration for the Sharpa V2P environment."""

    motion_file: str = _DEFAULT_MOTION_FILE

    def __post_init__(self) -> None:
        """Post-init."""
        super().__post_init__()


@configclass
class SharpaV2PEnvCfgPlay(SharpaV2PEnvCfg):
    """Configuration for the Sharpa V2P environment for playing."""

    def __post_init__(self) -> None:
        """Post-init: reduce num_envs for interactive play."""
        super().__post_init__()
        self.scene.num_envs = 16
