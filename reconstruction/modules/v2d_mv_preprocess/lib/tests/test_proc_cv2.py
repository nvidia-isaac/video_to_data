import cv2
import numpy as np

from v2d.mv.preprocess.lib.image_proc import proc_cv2
from v2d.mv.rig import CameraParam
from v2d.mv.rig.edex import DistortionModel


def _camera_param(projection_x: float) -> CameraParam:
    resolution = np.array([640, 480], dtype=np.int32)
    K = np.array(
        [
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    P = np.column_stack(
        [K, np.array([projection_x, 0.0, 0.0], dtype=np.float32)]
    )
    return CameraParam(
        resolution=resolution,
        D_model=DistortionModel.POLYNOMIAL.value,
        D=np.zeros(8, dtype=np.float32),
        K=K,
        P=P,
        R=np.eye(3, dtype=np.float32),
    )


def test_stereo_rectify_receives_translation_column_vector(monkeypatch):
    original_stereo_rectify = cv2.stereoRectify
    translation_shapes = []

    def checked_stereo_rectify(*args, **kwargs):
        translation_shapes.append(args[6].shape)
        return original_stereo_rectify(*args, **kwargs)

    monkeypatch.setattr(proc_cv2.cv2, "stereoRectify", checked_stereo_rectify)

    processors, params = proc_cv2.image_proc_build_rectify(
        _camera_param(projection_x=0.0),
        _camera_param(projection_x=-50.0),
    )

    assert translation_shapes == [(3, 1)]
    assert len(processors) == 2
    assert len(params) == 2
