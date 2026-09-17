# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GR00T N1.7 modality config for three-camera dual-hand Sharpa data."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

from groot_finetune.contracts import SHARPA_DUAL_HAND_THREE_CAMERA

_ABS = ActionConfig(
    rep=ActionRepresentation.ABSOLUTE,
    type=ActionType.NON_EEF,
    format=ActionFormat.DEFAULT,
)

sharpa_dual_hand_three_camera_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[camera.key for camera in SHARPA_DUAL_HAND_THREE_CAMERA.cameras],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            field.key for field in SHARPA_DUAL_HAND_THREE_CAMERA.state_fields
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(SHARPA_DUAL_HAND_THREE_CAMERA.action_horizon)),
        modality_keys=[
            field.key for field in SHARPA_DUAL_HAND_THREE_CAMERA.action_fields
        ],
        action_configs=[_ABS] * len(SHARPA_DUAL_HAND_THREE_CAMERA.action_fields),
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(
    sharpa_dual_hand_three_camera_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
