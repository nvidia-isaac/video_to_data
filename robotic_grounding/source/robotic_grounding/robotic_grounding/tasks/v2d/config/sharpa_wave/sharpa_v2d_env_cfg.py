# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from isaaclab.utils import configclass

from robotic_grounding.tasks.v2d.v2d_hand_env_cfg import V2DHandEnvCfg

_DEFAULT_MOTION_FILE = "arctic_processed/arctic_s01_box_grab_01/sharpa_wave"


@configclass
class SharpaV2DEnvCfg(V2DHandEnvCfg):
    """Configuration for the Sharpa V2D environment."""

    motion_file: str = _DEFAULT_MOTION_FILE

    def __post_init__(self) -> None:
        """Post-init."""
        super().__post_init__()


@configclass
class SharpaV2DEnvCfgPlay(SharpaV2DEnvCfg):
    """Configuration for the Sharpa V2D environment for playing."""

    def __post_init__(self) -> None:
        """Post-init: reduce num_envs for interactive play."""
        super().__post_init__()
        self.scene.num_envs = 16
