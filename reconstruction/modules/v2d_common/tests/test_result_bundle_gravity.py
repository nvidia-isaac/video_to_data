import json

import numpy as np

from v2d.common import result_bundle


def _write_geocalib_calibration(path, vector_camera):
    path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "frame_index": 0,
                        "gravity": {"vector_camera": vector_camera},
                    }
                ]
            }
        )
    )


def test_geocalib_vertical_vector_is_negated_to_bundle_gravity():
    raw = np.array([0.0, -np.sqrt(0.5), -np.sqrt(0.5)])

    converted = result_bundle._geocalib_gravity_to_opencv_camera(raw)

    np.testing.assert_allclose(
        converted,
        [0.0, np.sqrt(0.5), np.sqrt(0.5)],
        atol=1e-8,
    )


def test_z_up_alignment_keeps_downward_pitched_camera_looking_down(tmp_path):
    calibration_path = tmp_path / "geocalib.json"
    _write_geocalib_calibration(
        calibration_path,
        [0.0, -np.sqrt(0.5), -np.sqrt(0.5)],
    )
    arrays = {
        "camera_to_world_transform": np.eye(4, dtype=np.float32)[None, :, :],
        "camera_is_valid": np.array([True]),
    }

    gravity_world, forward_world, *_ = (
        result_bundle._estimate_world_gravity_and_forward_from_result(
            arrays,
            str(calibration_path),
        )
    )
    R_align, _ = result_bundle._z_up_alignment_from_gravity_and_forward(
        gravity_world,
        forward_world,
    )

    aligned_forward = R_align @ forward_world

    np.testing.assert_allclose(gravity_world, [0.0, np.sqrt(0.5), np.sqrt(0.5)])
    np.testing.assert_allclose(R_align @ gravity_world, [0.0, 0.0, -1.0], atol=1e-8)
    assert aligned_forward[2] < -0.7
