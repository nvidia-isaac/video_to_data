import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from v2d.mv.calibration.lib import calibrate_extrinsics as calibration
from v2d.mv.rig import CameraParam, apply_focal_correction


def _camera_param() -> CameraParam:
    return CameraParam(
        resolution=np.array([1920, 1200]),
        D_model="polynomial",
        D=np.array([0.1, 0.2, 0.3]),
        K=np.array([
            [1000.0, 2.0, 960.0],
            [0.0, 900.0, 600.0],
            [0.0, 0.0, 1.0],
        ]),
        P=np.array([
            [1000.0, 2.0, 960.0, -100.0],
            [0.0, 900.0, 600.0, 10.0],
            [0.0, 0.0, 1.0, 0.0],
        ]),
        R=np.eye(3),
        T=np.eye(4),
    )


class _FakeRig:
    def __init__(self, params: list[CameraParam]):
        self.cameras = {
            cam_id: SimpleNamespace(name=f"cam{cam_id}", param=param)
            for cam_id, param in enumerate(params)
        }
        self.saved = None

    def get_camera(self, cam_id):
        return self.cameras[cam_id]

    def get_stereo_pairs(self):
        return []

    def save_camera_params(self, source_path, output_path):
        self.saved = (source_path, output_path)


def test_shared_focal_correction_scales_only_focal_terms():
    param = _camera_param()
    original = copy.deepcopy(param)

    result = apply_focal_correction(param, 0.5)

    assert result is param
    np.testing.assert_allclose(param.K[:2, :2], original.K[:2, :2] * 0.5)
    np.testing.assert_allclose(param.P[:2, :2], original.P[:2, :2] * 0.5)
    np.testing.assert_allclose(param.P[:2, 3], original.P[:2, 3] * 0.5)
    np.testing.assert_allclose(param.K[:2, 2], original.K[:2, 2])
    np.testing.assert_allclose(param.P[:2, 2], original.P[:2, 2])
    np.testing.assert_array_equal(param.resolution, original.resolution)
    np.testing.assert_array_equal(param.D, original.D)
    np.testing.assert_array_equal(param.R, original.R)
    np.testing.assert_array_equal(param.T, original.T)


def test_empty_focal_correction_leaves_intrinsics_unchanged():
    param = _camera_param()
    original = copy.deepcopy(param)
    rig = _FakeRig([param])

    resolved = calibration._apply_focal_corrections(rig, None)

    assert resolved == {}
    np.testing.assert_array_equal(param.K, original.K)
    np.testing.assert_array_equal(param.P, original.P)


@pytest.mark.parametrize(
    "corrections",
    [{1: 0.985}, {0: 0.0}, {0: -1.0}, {0: np.inf}, {0: np.nan}],
)
def test_invalid_focal_correction_rejected_before_mutation(corrections):
    param = _camera_param()
    original = copy.deepcopy(param)
    rig = _FakeRig([param])

    with pytest.raises(ValueError, match="Focal correction"):
        calibration._apply_focal_corrections(rig, corrections)

    np.testing.assert_array_equal(param.K, original.K)
    np.testing.assert_array_equal(param.P, original.P)


def test_calibration_uses_and_persists_corrected_intrinsics(monkeypatch, tmp_path):
    params = [_camera_param(), _camera_param()]
    original_cam0 = copy.deepcopy(params[0])
    original_cam1 = copy.deepcopy(params[1])
    rig = _FakeRig(params)
    captured = {}

    monkeypatch.setattr(
        calibration,
        "chessboard_extract_correspondences",
        lambda **kwargs: (
            [[np.zeros((1, 2)), np.zeros((1, 2))]],
            [0],
        ),
    )

    def fake_pnp(**kwargs):
        captured["pnp_camera_params"] = copy.deepcopy(kwargs["camera_params"])
        return kwargs["camera_params"], np.eye(4)[None]

    def fake_ba(**kwargs):
        captured["ba_camera_params"] = copy.deepcopy(kwargs["camera_params"])
        summary = SimpleNamespace(
            termination_type=calibration.pyceres.TerminationType.CONVERGENCE,
        )
        return summary, kwargs["camera_params"], kwargs["init_target_poses"]

    monkeypatch.setattr(calibration, "extrinsics_estimate_pnp", fake_pnp)
    monkeypatch.setattr(calibration, "extrinsics_solve_ba", fake_ba)
    monkeypatch.setattr(
        calibration,
        "reprojection_error_stats",
        lambda *args, **kwargs: {"rmse_pixels": None},
    )

    output_path = tmp_path / "output" / "edex"
    calibration.calibrate_extrinsics(
        rig=rig,
        rgb_paths=[tmp_path / "cam0", tmp_path / "cam1"],
        calibration_order=[0, 1],
        camera_params_path=tmp_path / "input" / "edex",
        output_camera_params_path=output_path,
        board_size=(1, 1),
        correction_focal={1: 0.5},
    )

    np.testing.assert_array_equal(captured["pnp_camera_params"][0].K, original_cam0.K)
    np.testing.assert_allclose(
        captured["pnp_camera_params"][1].K[:2, :2],
        original_cam1.K[:2, :2] * 0.5,
    )
    np.testing.assert_allclose(
        captured["ba_camera_params"][1].K[:2, :2],
        original_cam1.K[:2, :2] * 0.5,
    )
    np.testing.assert_allclose(
        rig.get_camera(1).param.K[:2, :2],
        original_cam1.K[:2, :2] * 0.5,
    )
    assert rig.saved == (tmp_path / "input" / "edex", output_path)

    report = json.loads((output_path.parent / "calibration_accuracy.json").read_text())
    assert report["correction_focal"] == {"1": 0.5}
