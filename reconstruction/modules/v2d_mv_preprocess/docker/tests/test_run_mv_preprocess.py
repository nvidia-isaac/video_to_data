import importlib

import pytest


runner = importlib.import_module("v2d.mv.preprocess.docker.run_mv_preprocess")


def test_docker_forwards_canonical_calibration_path(monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "run_in_container", lambda **kwargs: calls.append(kwargs))

    runner.run_mv_preprocess(
        rgb_dir="/input/images",
        output_dir="/output",
        calibration_camera_params_path="/calibration/edex",
    )

    assert calls[0]["inputs"]["calibration_camera_params_path"] == "/calibration/edex"
    assert "extrinsics_camera_params_path" not in calls[0]["inputs"]


def test_docker_accepts_deprecated_calibration_path_alias(monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "run_in_container", lambda **kwargs: calls.append(kwargs))

    with pytest.warns(FutureWarning, match="deprecated"):
        runner.run_mv_preprocess(
            rgb_dir="/input/images",
            output_dir="/output",
            extrinsics_camera_params_path="/calibration/edex",
        )

    assert calls[0]["inputs"]["calibration_camera_params_path"] == "/calibration/edex"


def test_docker_rejects_conflicting_calibration_path_names():
    with pytest.warns(FutureWarning, match="deprecated"):
        with pytest.raises(ValueError, match="Conflicting calibration camera params"):
            runner.run_mv_preprocess(
                rgb_dir="/input/images",
                output_dir="/output",
                calibration_camera_params_path="/new/edex",
                extrinsics_camera_params_path="/old/edex",
            )
