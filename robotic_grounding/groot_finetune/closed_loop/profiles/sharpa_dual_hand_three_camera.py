# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Closed-loop embodiment profile for the floating dual-hand Sharpa robot."""

from __future__ import annotations

from functools import partial

from ...contracts import SHARPA_DUAL_HAND_THREE_CAMERA as CONTRACT
from ..embodiment import (
    Q_REF_LEFT_WRIST,
    Q_REF_RIGHT_WRIST,
    EmbodimentProfile,
    StateFieldSpec,
    VideoFieldSpec,
    align_quat_to_ref,
    register_profile,
)

_TRANSFORMS = {
    None: None,
    "quaternion_right_ref": partial(align_quat_to_ref, q_ref=Q_REF_RIGHT_WRIST),
    "quaternion_left_ref": partial(align_quat_to_ref, q_ref=Q_REF_LEFT_WRIST),
}

SHARPA_STATE_FIELDS = tuple(
    StateFieldSpec(
        field.key,
        field.source_term,
        field.start,
        field.end,
        _TRANSFORMS[field.transform],
    )
    for field in CONTRACT.state_fields
)
SHARPA_ACTION_FIELDS = tuple(
    StateFieldSpec(
        field.key,
        field.source_term,
        field.start,
        field.end,
        _TRANSFORMS[field.transform],
    )
    for field in CONTRACT.action_fields
)


SHARPA_DUAL_HAND_THREE_CAMERA = EmbodimentProfile(
    name=CONTRACT.contract_id,
    state_fields=SHARPA_STATE_FIELDS,
    action_fields=SHARPA_ACTION_FIELDS,
    video_fields=tuple(
        VideoFieldSpec(camera.key, camera.observation_term)
        for camera in CONTRACT.cameras
    ),
    language_key="annotation.human.task_description",
)

register_profile(SHARPA_DUAL_HAND_THREE_CAMERA)
