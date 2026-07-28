from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np

from inpainting.parallel_jaw_renderer.inputs import (
    InputError,
    load_parallel_jaw_inputs,
    select_preview_frame,
)
from inpainting.parallel_jaw_renderer.tests.helpers import (
    save_inputs,
    target_arrays,
)


class InputTests(unittest.TestCase):
    def test_loads_exact_world_contract_and_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target, intrinsic, world_to_camera = save_inputs(Path(temporary))
            inputs = load_parallel_jaw_inputs(
                target_path=target,
                intrinsic_path=intrinsic,
                world_to_camera_path=world_to_camera,
                width=640,
                height=480,
                fps=30.0,
            )
            self.assertEqual(inputs.frame_count, 3)
            self.assertEqual(inputs.tracker, "synthetic")
            self.assertEqual(
                inputs.projection_report()["left"]["inside_image_count"], 3
            )

    def test_rejects_extra_keys_and_non_world_coordinates(self) -> None:
        arrays = target_arrays()
        arrays["unexpected"] = np.asarray(1)
        with self.assertRaisesRegex(InputError, "extra="):
            from inpainting.parallel_jaw_renderer.inputs import validate_target_arrays

            validate_target_arrays(arrays)
        arrays.pop("unexpected")
        arrays["coordinate_frame"] = np.asarray("camera")
        with self.assertRaisesRegex(InputError, "exactly 'world'"):
            validate_target_arrays(arrays)

    def test_rejects_invalid_tracks_instead_of_holding(self) -> None:
        arrays = target_arrays()
        arrays["left_valid"][1] = False
        from inpainting.parallel_jaw_renderer.inputs import validate_target_arrays

        with self.assertRaisesRegex(InputError, "holding or hallucinating"):
            validate_target_arrays(arrays)

    def test_one_frame_preview_preserves_source_index_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target, intrinsic, world_to_camera = save_inputs(Path(temporary))
            inputs = load_parallel_jaw_inputs(
                target_path=target,
                intrinsic_path=intrinsic,
                world_to_camera_path=world_to_camera,
                width=640,
                height=480,
                fps=30.0,
            )
            preview = select_preview_frame(inputs, 2)
            self.assertEqual(preview.frame_count, 1)
            self.assertEqual(preview.preview_source_frame_index, 2)
            np.testing.assert_array_equal(preview.target["frame_indices"], (2,))
            np.testing.assert_array_equal(inputs.target["frame_indices"], (0, 1, 2))


if __name__ == "__main__":
    unittest.main()
