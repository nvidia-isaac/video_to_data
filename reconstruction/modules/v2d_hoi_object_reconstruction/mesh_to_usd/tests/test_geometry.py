import numpy as np
import pytest

from geometry import (
    matrix_to_quaternion_wxyz,
    principal_axis_pose_candidates,
    refine_rotation_to_support_plane,
    triangulate_faces,
)
from physics import quaternion_rotate


def test_triangulate_faces_uses_deterministic_fans():
    triangles = triangulate_faces([4, 3], [0, 1, 2, 3, 4, 5, 6], offset=10)
    assert triangles == [(10, 11, 12), (10, 12, 13), (14, 15, 16)]


def test_triangulate_faces_rejects_mismatched_counts():
    with pytest.raises(ValueError, match="counts"):
        triangulate_faces([3], [0, 1, 2, 3])


def test_matrix_to_quaternion_handles_half_turn():
    rotation = np.diag([1.0, -1.0, -1.0])
    quaternion = matrix_to_quaternion_wxyz(rotation)
    assert quaternion == pytest.approx((0.0, 1.0, 0.0, 0.0))


def test_matrix_to_quaternion_rejects_non_rotation():
    with pytest.raises(ValueError, match="finite"):
        matrix_to_quaternion_wxyz(np.diag([1.0, 1.0, np.nan]))


def test_principal_axis_candidates_cover_six_signed_directions():
    points = np.asarray(
        [
            (x, y, z)
            for x in (-2.0, 2.0)
            for y in (-1.0, 1.0)
            for z in (-0.5, 0.5)
        ]
    )

    candidates = principal_axis_pose_candidates(points)

    assert len(candidates) == 6
    assert {candidate.pose_id for candidate in candidates} == {
        f"principal_axis_{axis}_{direction}"
        for axis in range(3)
        for direction in ("positive", "negative")
    }
    assert sorted(candidate.axis_extent for candidate in candidates) == pytest.approx(
        [1.0, 1.0, 2.0, 2.0, 4.0, 4.0]
    )
    for candidate in candidates:
        assert quaternion_rotate(
            candidate.local_up,
            candidate.rotation_wxyz,
        ) == pytest.approx((0.0, 0.0, 1.0), abs=1e-7)


def test_principal_axis_candidates_reject_zero_extent():
    with pytest.raises(ValueError, match="non-zero extent"):
        principal_axis_pose_candidates([(1.0, 1.0, 1.0)] * 3)


def test_principal_axis_candidates_refine_with_separate_support_geometry():
    visual_points = np.asarray(
        [
            (x, y, z)
            for x in (-2.0, 2.0)
            for y in (-1.0, 1.0)
            for z in (-0.5, 0.5)
        ]
    )
    angle = np.deg2rad(10.0)
    rise = np.tan(angle)
    support_base = np.asarray(
        (
            (-1.0, -1.0, -rise),
            (1.0, -1.0, rise),
            (1.0, 1.0, rise),
            (-1.0, 1.0, -rise),
        )
    )
    support_points = np.vstack((support_base, support_base + (0.0, 0.0, 2.0)))
    support_faces = np.asarray(((0, 2, 1), (0, 3, 2)))

    candidates = principal_axis_pose_candidates(
        visual_points,
        refine_support_plane=True,
        support_points=support_points,
        support_faces=support_faces,
    )

    upright = next(
        candidate
        for candidate in candidates
        if candidate.pose_id == "principal_axis_2_positive"
    )
    assert upright.support_plane.applied
    assert upright.support_plane.correction_degrees == pytest.approx(10.0)
    rotated = support_points @ np.asarray(upright.support_plane.rotation).T
    assert np.ptp(rotated[:4, 2]) == pytest.approx(0.0, abs=1e-10)


def test_support_plane_refinement_levels_a_tilted_base():
    angle = np.deg2rad(10.0)
    rise = np.tan(angle)
    base = np.asarray(
        [
            (-1.0, -1.0, -rise),
            (1.0, -1.0, rise),
            (1.0, 1.0, rise),
            (-1.0, 1.0, -rise),
        ]
    )
    points = np.vstack((base, base + (0.0, 0.0, 2.0)))
    faces = np.asarray(((0, 2, 1), (0, 3, 2)))

    refinement = refine_rotation_to_support_plane(points, faces, np.eye(3))

    assert refinement.applied
    assert refinement.reason == "support_plane_aligned"
    assert refinement.face_count == 2
    assert refinement.correction_degrees == pytest.approx(10.0)
    rotated = points @ np.asarray(refinement.rotation).T
    assert np.ptp(rotated[:4, 2]) == pytest.approx(0.0, abs=1e-10)


def test_support_plane_refinement_falls_back_without_faces():
    points = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 1.0)))

    refinement = refine_rotation_to_support_plane(
        points,
        np.empty((0, 3), dtype=np.int64),
        np.eye(3),
    )

    assert not refinement.applied
    assert refinement.reason == "no_faces"
    assert np.asarray(refinement.rotation) == pytest.approx(np.eye(3))


def test_support_plane_refinement_rejects_a_small_incidental_face():
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.1, 0.0, 0.0),
            (0.0, 0.1, 0.0),
            (10.0, 0.0, 1.0),
            (0.0, 10.0, 1.0),
        )
    )

    refinement = refine_rotation_to_support_plane(
        points,
        np.asarray(((0, 1, 2),)),
        np.eye(3),
    )

    assert not refinement.applied
    assert refinement.reason == "support_area_too_small"
