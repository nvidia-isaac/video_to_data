# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for quaternion component-order conversions."""

import numpy as np

from flash_chord.utils.quat import wxyz_to_rotvec, wxyz_to_xyzw, xyzw_to_wxyz


def test_known_value():
    assert list(wxyz_to_xyzw([1.0, 0.0, 0.0, 0.0])) == [0.0, 0.0, 0.0, 1.0]
    assert list(xyzw_to_wxyz([0.0, 0.0, 0.0, 1.0])) == [1.0, 0.0, 0.0, 0.0]


def test_batched_roundtrip():
    q = np.arange(20, dtype=float).reshape(5, 4)
    assert np.array_equal(xyzw_to_wxyz(wxyz_to_xyzw(q)), q)
    assert wxyz_to_xyzw(q).shape == (5, 4)


def test_wxyz_to_rotvec_known_values():
    assert np.allclose(wxyz_to_rotvec([1.0, 0.0, 0.0, 0.0]), [0.0, 0.0, 0.0])  # identity
    h = np.sqrt(0.5)  # 90 deg about +z -> rotvec (0, 0, pi/2)
    assert np.allclose(wxyz_to_rotvec([h, 0.0, 0.0, h]), [0.0, 0.0, np.pi / 2], atol=1e-6)
    # batched
    assert wxyz_to_rotvec(np.tile([1.0, 0.0, 0.0, 0.0], (3, 1))).shape == (3, 3)
