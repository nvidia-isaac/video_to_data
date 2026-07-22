from __future__ import annotations

import unittest

import numpy as np

from inpainting.phantom_tracker.geometry import (
    cam_crop_to_full,
    cumulative_mano_joint_rotations,
    mirror_rotation_x,
    project_points,
    project_points_intrinsics,
    remap_virtual_camera_points,
    rotation_matrix_to_wxyz,
)


class GeometryTests(unittest.TestCase):
    def test_crop_camera_at_image_center(self) -> None:
        result = cam_crop_to_full(
            np.array([[2.0, 0.1, -0.2]]),
            np.array([[500.0, 250.0]]),
            np.array([100.0]),
            np.array([[1000.0, 500.0]]),
            5000.0,
        )
        np.testing.assert_allclose(result, [[0.1, -0.2, 50.0]])

    def test_project_points(self) -> None:
        result = project_points(np.array([[1.0, 2.0, 10.0]]), 100.0, 200, 100)
        np.testing.assert_allclose(result, [[110.0, 70.0]])

    def test_calibrated_remap_preserves_projection(self) -> None:
        points = np.array([[1.2, -0.7, 20.0], [-0.3, 0.2, 8.0]])
        intrinsics = np.array([[1376.75, 0, 967.69], [0, 1376.24, 526.82], [0, 0, 1]])
        virtual_uv = project_points(points, 37500.0, 1920, 1080)
        calibrated = remap_virtual_camera_points(points, 37500.0, 1920, 1080, intrinsics)
        calibrated_uv = project_points_intrinsics(calibrated, intrinsics)
        np.testing.assert_allclose(calibrated_uv, virtual_uv, atol=1e-9)
        np.testing.assert_allclose(calibrated[:, 2], points[:, 2] * 1376.75 / 37500.0)

    def test_mirrored_rotation_remains_proper(self) -> None:
        angle = np.pi / 3
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        )
        mirrored = mirror_rotation_x(rotation)
        np.testing.assert_allclose(mirrored.T @ mirrored, np.eye(3), atol=1e-8)
        self.assertAlmostEqual(float(np.linalg.det(mirrored)), 1.0)

    def test_mano_rotations_are_accumulated_and_expanded(self) -> None:
        angle = np.pi / 2
        z_rotation = np.array(
            [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        )
        pose = np.broadcast_to(np.eye(3), (15, 3, 3)).copy()
        pose[0] = z_rotation
        parents = np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14])
        mapping = (0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10, 11, 12, 12, 13, 14, 15, 15)
        result = cumulative_mano_joint_rotations(
            np.eye(3), pose, parents, mapping, mirror_left=False
        )
        np.testing.assert_allclose(result[1], z_rotation, atol=1e-8)
        np.testing.assert_allclose(result[2], z_rotation, atol=1e-8)
        np.testing.assert_allclose(result[5], np.eye(3), atol=1e-8)

    def test_mano_anatomical_frames_apply_after_global_accumulation(self) -> None:
        angle = np.pi / 2
        root = np.array(
            [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        )
        x_frame = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
        pose = np.broadcast_to(np.eye(3), (15, 3, 3)).copy()
        parents = np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14])
        mapping = (0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10, 11, 12, 12, 13, 14, 15, 15)
        frames = np.broadcast_to(x_frame, (16, 3, 3)).copy()
        result = cumulative_mano_joint_rotations(
            root,
            pose,
            parents,
            mapping,
            mirror_left=False,
            anatomical_frames=frames,
        )
        np.testing.assert_allclose(result[0], root @ x_frame, atol=1e-8)

    def test_quaternion_identity_is_wxyz(self) -> None:
        np.testing.assert_allclose(rotation_matrix_to_wxyz(np.eye(3)), [1, 0, 0, 0])

    def test_quaternion_direction_matches_rotation_matrix(self) -> None:
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        root_half = np.sqrt(0.5)
        np.testing.assert_allclose(
            rotation_matrix_to_wxyz(rotation), [root_half, 0, 0, root_half]
        )

    def test_quaternion_rejects_reflection(self) -> None:
        with self.assertRaisesRegex(ValueError, "proper orthonormal"):
            rotation_matrix_to_wxyz(np.diag([-1.0, 1.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
