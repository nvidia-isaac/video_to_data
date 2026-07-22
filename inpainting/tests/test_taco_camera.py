from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from inpainting.contracts import ContractError
from inpainting.taco_camera import load_taco_camera, project_world_points


class TacoCameraTest(unittest.TestCase):
    def test_projection(self) -> None:
        intrinsic = np.array([[100, 0, 50], [0, 100, 40], [0, 0, 1]], dtype=float)
        transform = np.eye(4)
        pixels, depth, valid = project_world_points(
            np.array([[0, 0, 2], [1, -1, 2], [0, 0, -1]], dtype=float),
            intrinsic,
            transform,
        )
        np.testing.assert_allclose(pixels[:2], [[50, 40], [100, -10]])
        np.testing.assert_allclose(depth, [2, 2, -1])
        np.testing.assert_array_equal(valid, [True, True, False])

    def test_loader_checks_frame_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            intrinsic_path = root / "k.txt"
            extrinsic_path = root / "e.npy"
            np.savetxt(intrinsic_path, [[100, 0, 50], [0, 100, 40], [0, 0, 1]])
            np.save(extrinsic_path, np.repeat(np.eye(4)[None], 2, axis=0))
            with self.assertRaisesRegex(ContractError, "must be"):
                load_taco_camera(intrinsic_path, extrinsic_path, 3, 100, 80)


if __name__ == "__main__":
    unittest.main()
