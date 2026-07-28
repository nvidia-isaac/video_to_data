from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np

from inpainting.robot_renderer.inputs import RenderInputError, load_render_inputs

from inpainting.robot_renderer.tests.helpers import save_inputs, trajectory_arrays


class RenderInputTests(unittest.TestCase):
    def test_world_input_and_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trajectory, intrinsic, w2c = save_inputs(root)
            inputs = load_render_inputs(
                trajectory_path=trajectory,
                intrinsic_path=intrinsic,
                world_to_camera_path=w2c,
                width=640,
                height=480,
                fps=30.0,
            )
            self.assertEqual(inputs.frame_count, 3)
            report = inputs.projection_report()
            self.assertEqual(report["left"]["inside_image_count"], 3)
            self.assertEqual(report["right"]["positive_depth_count"], 3)

    def test_camera_tracks_are_returned_to_world(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trajectory, intrinsic, w2c = save_inputs(root, coordinate_frame="camera")
            matrices = np.load(w2c)
            matrices[:, 0, 3] = -1.0
            np.save(w2c, matrices)
            inputs = load_render_inputs(
                trajectory_path=trajectory,
                intrinsic_path=intrinsic,
                world_to_camera_path=w2c,
                width=640,
                height=480,
                fps=30.0,
            )
            np.testing.assert_allclose(inputs.left_world_wrist[:, 0, 3], 0.9)
            camera_reprojection = inputs.world_to_camera @ inputs.left_world_wrist
            np.testing.assert_allclose(camera_reprojection[:, 0, 3], -0.1)

    def test_rejects_calibration_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trajectory, intrinsic, w2c = save_inputs(root)
            np.save(w2c, np.tile(np.eye(4), (2, 1, 1)))
            with self.assertRaisesRegex(RenderInputError, "2 frames"):
                load_render_inputs(
                    trajectory_path=trajectory,
                    intrinsic_path=intrinsic,
                    world_to_camera_path=w2c,
                    width=640,
                    height=480,
                    fps=30.0,
                )

    def test_rejects_invalid_frames_instead_of_holding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trajectory, intrinsic, w2c = save_inputs(root)
            arrays = trajectory_arrays()
            arrays["left_valid"][1] = False
            np.savez(trajectory, **arrays)
            with self.assertRaisesRegex(RenderInputError, "no pose hallucination"):
                load_render_inputs(
                    trajectory_path=trajectory,
                    intrinsic_path=intrinsic,
                    world_to_camera_path=w2c,
                    width=640,
                    height=480,
                    fps=30.0,
                )

    def test_rejects_likely_inverted_extrinsic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trajectory, intrinsic, w2c = save_inputs(root, z=-2.0)
            with self.assertRaisesRegex(RenderInputError, "likely inverted"):
                load_render_inputs(
                    trajectory_path=trajectory,
                    intrinsic_path=intrinsic,
                    world_to_camera_path=w2c,
                    width=640,
                    height=480,
                    fps=30.0,
                )

    def test_rejects_calibration_with_all_wrists_off_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trajectory, intrinsic, w2c = save_inputs(root)
            arrays = trajectory_arrays()
            arrays["left_wrist_position"][:, 0] = -10.0
            arrays["right_wrist_position"][:, 0] = 10.0
            np.savez(trajectory, **arrays)
            with self.assertRaisesRegex(RenderInputError, "no wrist center"):
                load_render_inputs(
                    trajectory_path=trajectory,
                    intrinsic_path=intrinsic,
                    world_to_camera_path=w2c,
                    width=640,
                    height=480,
                    fps=30.0,
                )

    def test_rejects_intrinsic_for_different_image_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trajectory, intrinsic, w2c = save_inputs(root)
            np.savetxt(intrinsic, np.array(((500, 0, 1000), (0, 500, 240), (0, 0, 1))))
            with self.assertRaisesRegex(RenderInputError, "outside source geometry"):
                load_render_inputs(
                    trajectory_path=trajectory,
                    intrinsic_path=intrinsic,
                    world_to_camera_path=w2c,
                    width=640,
                    height=480,
                    fps=30.0,
                )


if __name__ == "__main__":
    unittest.main()
