# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Gaussian/Laplacian objective shaping primitive."""

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def _shape(sse, var, kind, threshold=0.0):
    import warp as wp

    from flash_chord.objectives.shaping import shaped_objective

    @wp.kernel
    def _apply(sse: wp.array(dtype=wp.float32), var: float, kind: int, thr: float, out: wp.array(dtype=wp.float32)):
        i = wp.tid()
        out[i] = shaped_objective(sse[i], var, kind, thr)

    with wp.ScopedDevice("cuda:0"):
        s = wp.array(np.asarray(sse, dtype=np.float32), dtype=wp.float32)
        out = wp.zeros(len(sse), dtype=wp.float32)
        wp.launch(_apply, dim=len(sse), inputs=[s, float(var), int(kind), float(threshold)], outputs=[out])
        return out.numpy()


def test_gaussian_form():
    from flash_chord.objectives.shaping import ObjectiveShape

    sse = [0.0, 0.04, 0.09]
    r = _shape(sse, var=0.1, kind=ObjectiveShape.GAUSSIAN)
    assert np.allclose(r, np.exp(-np.asarray(sse) / 0.1), atol=1e-5)  # exp(-sse/var)


def test_laplacian_uses_sqrt_distance():
    from flash_chord.objectives.shaping import ObjectiveShape

    sse = [0.0, 0.04, 0.09]  # d = sqrt(sse) = 0, 0.2, 0.3
    r = _shape(sse, var=0.1, kind=ObjectiveShape.LAPLACIAN)
    assert np.allclose(r, np.exp(-np.sqrt(sse) / 0.1), atol=1e-5)  # exp(-d/var)


def test_threshold_saturates():
    from flash_chord.objectives.shaping import ObjectiveShape

    # Laplacian, d=0.2 < threshold 0.25 -> saturates to 1.0; d=0.3 > 0.25 -> exp(-(0.3-0.25)/0.1)
    r = _shape([0.04, 0.09], var=0.1, kind=ObjectiveShape.LAPLACIAN, threshold=0.25)
    assert abs(r[0] - 1.0) < 1e-6
    assert abs(r[1] - np.exp(-(0.3 - 0.25) / 0.1)) < 1e-5
