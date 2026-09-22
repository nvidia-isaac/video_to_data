# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the generalized ``ReferencePlane`` (normal + offset)."""

from __future__ import annotations

import numpy as np
import pytest
from robotic_grounding.retarget.ground_alignment import (
    ReferencePlane,
    compute_object_ground_lift,
    compute_plane_leveling_transform,
)


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


def test_plane_leveling_transform_maps_fitted_plane_to_world_z0() -> None:
    """The leveling rigid transform maps tilted on-plane points onto z=0."""
    plane = ReferencePlane(normal=(0.2, -0.1, 0.97), offset=0.8)
    rotation, translation = compute_plane_leveling_transform(plane)

    normal = np.asarray(plane.normal)
    np.testing.assert_allclose(rotation @ normal, [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-12)

    xy = np.array([[0.0, 0.0], [1.5, -0.7], [-2.0, 0.4]])
    z = -(normal[0] * xy[:, 0] + normal[1] * xy[:, 1] + plane.offset) / normal[2]
    points = np.column_stack([xy, z])
    leveled = (rotation @ points.T).T + translation
    np.testing.assert_allclose(leveled[:, 2], 0.0, atol=1e-12)


def test_plane_leveling_transform_horizontal_plane_is_translation_only() -> None:
    """A horizontal fitted plane at z=-d only needs a +d Z translation."""
    plane = ReferencePlane(normal=(0.0, 0.0, 1.0), offset=0.75)
    rotation, translation = compute_plane_leveling_transform(plane)
    np.testing.assert_allclose(rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(translation, [0.0, 0.0, 0.75], atol=1e-12)


def test_object_ground_lift_clears_frame_zero_penetration() -> None:
    vertices = np.array([[0.0, 0.0, -0.00423], [0.1, 0.0, 0.2]])
    correction = compute_object_ground_lift(vertices, ReferencePlane.horizontal())

    assert correction.minimum_signed_distance == pytest.approx(-0.00423)
    assert correction.penetration_depth == pytest.approx(0.00423)
    assert correction.requested_lift == pytest.approx(0.00473)
    assert correction.applied_lift == pytest.approx(0.00473)
    assert not correction.capped


def test_object_ground_lift_does_not_pull_object_down() -> None:
    vertices = np.array([[0.0, 0.0, 0.2], [0.1, 0.0, 0.3]])
    correction = compute_object_ground_lift(vertices, ReferencePlane.horizontal())

    assert correction.applied_lift == 0.0
    assert not correction.capped


def test_object_ground_lift_ignores_sub_tolerance_penetration() -> None:
    vertices = np.array([[0.0, 0.0, -0.0004], [0.1, 0.0, 0.2]])
    correction = compute_object_ground_lift(vertices, ReferencePlane.horizontal())

    assert correction.penetration_depth == pytest.approx(0.0004)
    assert correction.requested_lift == 0.0
    assert correction.applied_lift == 0.0


def test_object_ground_lift_is_capped_at_one_centimeter() -> None:
    vertices = np.array([[0.0, 0.0, -0.03], [0.1, 0.0, 0.2]])
    correction = compute_object_ground_lift(vertices, ReferencePlane.horizontal())

    assert correction.requested_lift == pytest.approx(0.0305)
    assert correction.applied_lift == pytest.approx(0.01)
    assert correction.capped
