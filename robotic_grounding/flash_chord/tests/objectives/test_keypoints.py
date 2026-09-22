# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the object keypoint tracking objective."""

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def test_object_keypoints_translation_offset():
    """Sim object at origin/identity, command translated by t: every axis keypoint has error ‖t‖², so the
    mean-over-keypoints objective = shaped(‖t‖²). Checked in both Gaussian and Laplacian modes."""
    import warp as wp

    from flash_chord.objectives.keypoints import KEYPOINT_VECS_NP, object_keypoints_objective
    from flash_chord.objectives.shaping import ObjectiveShape

    t = np.array([0.1, 0.0, 0.0], dtype=np.float32)  # ‖t‖ = 0.1
    var = 0.1
    with wp.ScopedDevice("cuda:0"):
        body_q = wp.array([wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat(0.0, 0.0, 0.0, 1.0))], dtype=wp.transform)
        object_body_ids = wp.array([0], dtype=wp.int32)
        target_pos = wp.array([wp.vec3(*t.tolist())], dtype=wp.vec3)
        target_quat = wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat)
        kpv = wp.array(KEYPOINT_VECS_NP, dtype=wp.vec3)

        def run(kind):
            rew = wp.zeros(1, dtype=wp.float32)
            wp.launch(object_keypoints_objective, dim=6, inputs=[body_q, object_body_ids, target_pos, target_quat, kpv, 1, 1, var, int(kind)], outputs=[rew])
            return float(rew.numpy()[0]) / 6.0  # mean over 6 keypoints (1 body)

        g = run(ObjectiveShape.GAUSSIAN)
        lap = run(ObjectiveShape.LAPLACIAN)

    d2 = float(t @ t)
    assert abs(g - np.exp(-d2 / var)) < 1e-5  # Gaussian: exp(-‖t‖²/var)
    assert abs(lap - np.exp(-np.sqrt(d2) / var)) < 1e-5  # Laplacian: exp(-‖t‖/var)
