from __future__ import annotations

import unittest

import numpy as np

from inpainting.robot_renderer.transforms import (
    CV_TO_OPENGL,
    TransformError,
    invert_rigid_transform,
    matrix_to_quaternion_wxyz,
    pose_matrix,
    quaternion_wxyz_to_matrix,
    validate_rigid_transform,
    validate_transform_batch,
)


class TransformTests(unittest.TestCase):
    def test_quaternion_matrix_round_trip(self) -> None:
        angle = np.radians(70.0)
        quaternion = np.array((np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)))
        rotation = quaternion_wxyz_to_matrix(quaternion)
        recovered = matrix_to_quaternion_wxyz(rotation)
        np.testing.assert_allclose(recovered, quaternion, atol=1e-10)

    def test_pose_inverse(self) -> None:
        transform = pose_matrix(
            np.array((1.0, -2.0, 3.0)),
            np.array((np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0)),
        )
        inverse = invert_rigid_transform(transform)
        np.testing.assert_allclose(inverse @ transform, np.eye(4), atol=1e-12)

    def test_rejects_reflection_and_bad_bottom_row(self) -> None:
        reflection = np.eye(4)
        reflection[0, 0] = -1.0
        with self.assertRaisesRegex(TransformError, "determinant"):
            validate_rigid_transform(reflection)
        bad = np.tile(np.eye(4), (2, 1, 1))
        bad[1, 3, 3] = 2.0
        with self.assertRaisesRegex(TransformError, r"transforms\[1\]"):
            validate_transform_batch(bad)

    def test_cv_to_opengl_is_proper_rotation(self) -> None:
        validate_rigid_transform(CV_TO_OPENGL)
        point = CV_TO_OPENGL @ np.array((1.0, 2.0, 3.0, 1.0))
        np.testing.assert_array_equal(point, (1.0, -2.0, -3.0, 1.0))


if __name__ == "__main__":
    unittest.main()
