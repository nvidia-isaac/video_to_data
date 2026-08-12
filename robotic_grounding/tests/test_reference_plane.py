# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the generalized ``ReferencePlane`` (normal + offset)."""

from __future__ import annotations

import numpy as np
import pytest
from robotic_grounding.retarget.ground_alignment import ReferencePlane


def test_horizontal_factory_matches_legacy_signed_distance() -> None:
    """`horizontal(z=z0)` produces the same signed_distance as the legacy form."""
    plane = ReferencePlane.horizontal(z=0.0)
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.25], [3.0, 4.0, -0.10]])
    np.testing.assert_allclose(plane.signed_distance(pts), [0.0, 0.25, -0.10])

    plane_z1 = ReferencePlane.horizontal(z=1.0)
    np.testing.assert_allclose(plane_z1.signed_distance(pts), [-1.0, -0.75, -1.10])


def test_inclined_signed_distance() -> None:
    """A non-horizontal plane's signed_distance equals ``n . p + offset``."""
    # Plane through the origin tilted ~45 deg around the X axis: normal ~ (0, -1, 1).
    plane = ReferencePlane(normal=(0.0, -1.0, 1.0), offset=0.0)
    pt_on = np.array([0.0, 1.0, 1.0])
    np.testing.assert_allclose(plane.signed_distance(pt_on), 0.0, atol=1e-12)
    pt_above = np.array([0.0, 0.0, 0.5])
    expected = float(np.dot(np.array([0.0, -1.0, 1.0]) / np.sqrt(2.0), pt_above))
    np.testing.assert_allclose(plane.signed_distance(pt_above), expected)


def test_vertical_offset_to_plane_horizontal_equivalence() -> None:
    """For a horizontal plane, ``vertical_offset_to_plane = -signed_distance``."""
    plane = ReferencePlane.horizontal(z=0.5)
    pts = np.array([[0.0, 0.0, 0.4], [0.0, 0.0, 1.5]])
    np.testing.assert_allclose(
        plane.vertical_offset_to_plane(pts), -plane.signed_distance(pts)
    )


def test_vertical_offset_to_plane_inclined_uses_normal_z() -> None:
    """On a tilted plane, dz = -signed_distance / normal_z."""
    plane = ReferencePlane(normal=(0.0, -1.0, 1.0), offset=0.0)
    pts = np.array([[0.0, 0.0, 1.0], [0.0, 0.5, 1.0]])
    sd = plane.signed_distance(pts)
    np.testing.assert_allclose(
        plane.vertical_offset_to_plane(pts), -sd / plane.normal_z
    )
    # Sanity: applying dz to Z should drive signed_distance to zero.
    pts_corrected = pts.copy()
    pts_corrected[:, 2] += plane.vertical_offset_to_plane(pts)
    np.testing.assert_allclose(plane.signed_distance(pts_corrected), 0.0, atol=1e-12)


def test_normal_is_normalized() -> None:
    """Constructor normalizes ``normal`` regardless of input magnitude."""
    plane = ReferencePlane(normal=(0.0, 0.0, 5.0), offset=-2.0)
    np.testing.assert_allclose(plane.normal, (0.0, 0.0, 1.0))
    # Offset is rescaled so the plane equation is preserved: a point at z=2
    # remains on the plane.
    pt = np.array([0.0, 0.0, 2.0])
    np.testing.assert_allclose(plane.signed_distance(pt), 0.0, atol=1e-12)


def test_orient_normal_so_normal_z_positive() -> None:
    """A plane built with normal_z < 0 is flipped so normal_z > 0.

    Input: ``-z - 3 = 0`` -> plane lies at ``z = -3``.
    After flip: ``z + 3 = 0`` -> still ``z = -3``. Both forms represent
    the same plane; the canonical form has normal_z > 0.
    """
    plane = ReferencePlane(normal=(0.0, 0.0, -1.0), offset=-3.0)
    np.testing.assert_allclose(plane.normal, (0.0, 0.0, 1.0))
    np.testing.assert_allclose(plane.offset, 3.0)
    pt = np.array([0.0, 0.0, -3.0])
    np.testing.assert_allclose(plane.signed_distance(pt), 0.0, atol=1e-12)


def test_near_vertical_plane_rejected() -> None:
    """A near-horizontal-Z plane raises (invalid for foot anchoring)."""
    with pytest.raises(ValueError):
        ReferencePlane(normal=(1.0, 0.0, 0.0), offset=0.0)


def test_zero_normal_rejected() -> None:
    """A zero-length normal raises."""
    with pytest.raises(ValueError):
        ReferencePlane(normal=(0.0, 0.0, 0.0), offset=0.0)


def test_horizontal_signed_distance_supports_leading_batch_dims() -> None:
    """Works for ``(T, K, 3)`` arrays without a reshape on the caller."""
    plane = ReferencePlane.horizontal(z=0.0)
    pts = np.zeros((5, 2, 3), dtype=np.float64)
    pts[:, 0, 2] = np.linspace(0.0, 0.4, 5)
    pts[:, 1, 2] = 0.1
    d = plane.signed_distance(pts)
    assert d.shape == (5, 2)
    np.testing.assert_allclose(d[:, 0], np.linspace(0.0, 0.4, 5))
    np.testing.assert_allclose(d[:, 1], 0.1)


def test_signed_distance_rejects_wrong_trailing_dim() -> None:
    """Non-3 last axis raises."""
    plane = ReferencePlane.horizontal()
    with pytest.raises(ValueError):
        plane.signed_distance(np.zeros((3, 2)))
