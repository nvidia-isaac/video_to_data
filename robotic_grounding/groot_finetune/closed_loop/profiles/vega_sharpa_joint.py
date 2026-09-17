# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Closed-loop profile for the Vega+Sharpa dual-arm whole-body embodiment.

Maps the ``record`` obs group of the vega whole-body envs (``arm_joint_pos`` 14 =
[R_arm(7), L_arm(7)], ``finger_joint_pos`` 44 = [right(22), left(22)], both RAW radians
in the explicit ``assets.vega_sharpa_wave`` orders) to the GR00T state keys of
``groot_finetune/vega_sharpa_joint_config.py``. Video: front + 2 wrist views. No
per-field transforms — the joint-space contract has no quaternion sign conventions.
"""

from __future__ import annotations

from ...contracts import VEGA_SHARPA_JOINT
from ..embodiment import (
    EmbodimentProfile,
    StateFieldSpec,
    VideoFieldSpec,
    register_profile,
)

_STATE_FIELDS = tuple(
    StateFieldSpec(field.key, field.source_term, field.start, field.end)
    for field in VEGA_SHARPA_JOINT.state_fields
)

_ACTION_FIELDS = tuple(
    StateFieldSpec(field.key, field.source_term, field.start, field.end)
    for field in VEGA_SHARPA_JOINT.action_fields
)

_VIDEO_FIELDS = tuple(
    VideoFieldSpec(camera.key, camera.observation_term)
    for camera in VEGA_SHARPA_JOINT.cameras
)


VEGA_SHARPA_JOINT_PROFILE = EmbodimentProfile(
    name=VEGA_SHARPA_JOINT.contract_id,
    state_fields=_STATE_FIELDS,
    action_fields=_ACTION_FIELDS,
    video_fields=_VIDEO_FIELDS,
    language_key="annotation.human.task_description",
)

register_profile(VEGA_SHARPA_JOINT_PROFILE)
