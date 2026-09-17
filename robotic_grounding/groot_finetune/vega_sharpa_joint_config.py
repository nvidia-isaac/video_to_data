# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GR00T modality config for the Vega Sharpa joint-space embodiment.

Joint-space contract (NO wrist-pose/quaternion machinery — the double-cover fragility
of the floating-hand configs does not exist here): state = 58 raw joint radians
[right_arm(7), left_arm(7), right_finger(22), left_finger(22)]; action = 58 ABSOLUTE
joint-position targets in the canonical external-export flat order
[right_arm(7), right_finger(22), left_arm(7), left_finger(22)]. Video carries THREE
views (front + 2 wrist).

Pass this file to ``--modality-config-path`` for ``gr00t/experiment/launch_finetune.py``,
``gr00t/data/stats.py`` and ``gr00t/eval/open_loop_eval.py`` (Isaac-GR00T N1.7). It runs
INSIDE the GR00T env (it imports ``gr00t.*``); it is NOT importable in the
robotic_grounding container.

The ``state`` / ``action`` / ``video`` modality keys below MUST match the
``modality.json`` written by ``convert_to_gr00t.py`` and are the wire contract the
closed-loop client sends/receives. The registered ACTION
key order is the flat-58 reassembly order for closed-loop inference.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

from groot_finetune.contracts import VEGA_SHARPA_JOINT

# Absolute joint-position targets for every action key.
_ABS = ActionConfig(
    rep=ActionRepresentation.ABSOLUTE,
    type=ActionType.NON_EEF,
    format=ActionFormat.DEFAULT,
)

vega_sharpa_joint_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[camera.key for camera in VEGA_SHARPA_JOINT.cameras],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[field.key for field in VEGA_SHARPA_JOINT.state_fields],
    ),
    # Predict a 16-step action chunk; SAME ORDER as the flat-58 canonical layout.
    "action": ModalityConfig(
        delta_indices=list(range(VEGA_SHARPA_JOINT.action_horizon)),
        modality_keys=[field.key for field in VEGA_SHARPA_JOINT.action_fields],
        action_configs=[_ABS] * len(VEGA_SHARPA_JOINT.action_fields),
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

# Custom embodiments always register under NEW_EMBODIMENT.
register_modality_config(
    vega_sharpa_joint_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT
)
