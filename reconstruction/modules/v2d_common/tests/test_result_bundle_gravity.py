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


def test_write_result_bundle_bakes_object_pose_scale_into_mesh(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    (frames_dir / "000000.png").write_bytes(b"")

    intrinsics_path = tmp_path / "intrinsics.json"
    intrinsics_path.write_text(
        json.dumps(
            {
                "fx": 100.0,
                "fy": 101.0,
                "cx": 50.0,
                "cy": 51.0,
                "width": 100,
                "height": 100,
            }
        )
    )

    mesh_path = tmp_path / "mesh.obj"
    mesh_path.write_text(
        "v 1 0 0\n"
        "v 0 2 0\n"
        "v 0 0 3\n"
        "f 1 2 3\n"
    )

    poses_dir = tmp_path / "poses"
    poses_dir.mkdir()
    (poses_dir / "000000.json").write_text(
        json.dumps(
            {
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "translation": [1.0, 2.0, 3.0],
                "scale": [2.0, 2.0, 2.0],
            }
        )
    )

    result_dir = tmp_path / "result"
    result_bundle.write_result_bundle(
        result_dir=str(result_dir),
        frames_dir=str(frames_dir),
        intrinsics_path=str(intrinsics_path),
        mesh_path=str(mesh_path),
        object_poses_dir=str(poses_dir),
        object_scale=1.0,
    )

    with np.load(result_dir / "result.npz") as arrays:
        object_pose = arrays["object_to_camera_transform"][0]
        np.testing.assert_allclose(
            np.linalg.svd(object_pose[:3, :3], compute_uv=False),
            [1.0, 1.0, 1.0],
            atol=1e-6,
        )
        np.testing.assert_allclose(object_pose[:3, 3], [1.0, 2.0, 3.0])
        assert float(arrays["object_scale"]) == 1.0

    vertices = []
    for line in (result_dir / "mesh.obj").read_text().splitlines():
        if line.startswith("v "):
            vertices.append([float(v) for v in line.split()[1:4]])
    np.testing.assert_allclose(
        vertices,
        [[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 6.0]],
    )

    manifest = json.loads((result_dir / "manifest.json").read_text())
    assert manifest["sources"]["object_scale"] == 1.0
    assert manifest["sources"]["object_pose_scale_baked_into_mesh"] == [2.0, 2.0, 2.0]
    assert manifest["sources"]["object_transform_convention"] == "rigid_pose_no_scale"
