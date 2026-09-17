from types import SimpleNamespace

import numpy as np
import pytest

from v2d.mv.rig import CameraEntry, CameraParam, RigConfig
from v2d.mv.rig.edex import Camera, Intrinsics
import v2d.mv.rig.rig as rig_module


def _camera(*, focal=(100.0, 110.0), projection=None, transform=None):
    return Camera(
        intrinsics=Intrinsics(
            distortion_model="pinhole",
            distortion_params=[],
            focal=focal,
            principal=[50.0, 60.0],
            size=[1920, 1200],
            projection=projection,
            rectification=np.eye(3),
        ),
        transform=transform,
    )


def _param():
    return CameraParam(
        resolution=np.array([640, 480]),
        D_model="brown5k",
        D=np.arange(5, dtype=np.float32),
        K=np.array([[10.0, 0.0, 20.0], [0.0, 11.0, 21.0], [0.0, 0.0, 1.0]]),
        P=np.full((3, 4), 3.0),
        R=np.full((3, 3), 4.0),
        T=np.full((4, 4), 5.0),
    )


def test_merge_intrinsics_copies_only_k_and_p(monkeypatch):
    projection = np.arange(12, dtype=np.float32).reshape(3, 4)
    calibration_camera = _camera(
        focal=(200.0, 220.0),
        projection=projection,
        transform=np.arange(12, dtype=np.float32).reshape(3, 4),
    )
    monkeypatch.setattr(
        rig_module.EDEXMetadata,
        "read",
        lambda _path: SimpleNamespace(
            header=SimpleNamespace(cameras=[calibration_camera])
        ),
    )

    original = _param()
    rig = RigConfig.__new__(RigConfig)
    rig.cameras = {0: CameraEntry(cam_id=0, name="camera", param=original)}
    unchanged = {
        "resolution": original.resolution.copy(),
        "D": original.D.copy(),
        "R": original.R.copy(),
        "T": original.T.copy(),
    }

    rig.merge_intrinsics("edex")

    np.testing.assert_array_equal(original.K, [[200, 0, 50], [0, 220, 60], [0, 0, 1]])
    np.testing.assert_array_equal(original.P, projection)
    for field, value in unchanged.items():
        np.testing.assert_array_equal(getattr(original, field), value)
    assert original.D_model == "brown5k"

    calibration_camera.intrinsics.projection[0, 0] = -1
    assert original.P[0, 0] == 0


def test_merge_intrinsics_rejects_missing_calibration_camera(monkeypatch):
    monkeypatch.setattr(
        rig_module.EDEXMetadata,
        "read",
        lambda _path: SimpleNamespace(header=SimpleNamespace(cameras=[])),
    )
    rig = RigConfig.__new__(RigConfig)
    rig.cameras = {0: CameraEntry(cam_id=0, name="camera", param=_param())}

    with pytest.raises(ValueError, match="camera IDs missing from calibration EDEX"):
        rig.merge_intrinsics("edex")


def test_merge_intrinsics_rejects_rig_camera_without_loaded_params(monkeypatch):
    monkeypatch.setattr(
        rig_module.EDEXMetadata,
        "read",
        lambda _path: SimpleNamespace(
            header=SimpleNamespace(cameras=[_camera()])
        ),
    )
    rig = RigConfig.__new__(RigConfig)
    rig.cameras = {0: CameraEntry(cam_id=0, name="camera", param=None)}

    with pytest.raises(ValueError, match="rig cameras without loaded params"):
        rig.merge_intrinsics("edex")
