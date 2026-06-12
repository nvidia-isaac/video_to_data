# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from isaaclab.utils import configclass

from robotic_grounding.tasks.v2p.v2p_hand_env_cfg import CurriculumCfg, V2PHandEnvCfg

_DEFAULT_MOTION_FILE = "arctic_processed/arctic_s01_box_grab_01/sharpa_wave"


@configclass
class SharpaV2PAutoCurrEnvCfg(V2PHandEnvCfg):
    """Sharpa V2P environment using VirtualObjectControlCurriculum (adaptive gates)."""

    motion_file: str = _DEFAULT_MOTION_FILE
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:  # noqa: D105
        super().__post_init__()
