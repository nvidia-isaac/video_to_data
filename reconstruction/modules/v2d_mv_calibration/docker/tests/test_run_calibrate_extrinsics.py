import pytest

from v2d.mv.calibration.docker import run_calibrate_extrinsics as runner


@pytest.mark.parametrize("enabled", [False, True])
def test_docker_wrapper_forwards_marker_flag(monkeypatch, enabled):
    captured = {}
    monkeypatch.setattr(runner, "run_in_container", lambda **kwargs: captured.update(kwargs))

    runner.run_calibrate_extrinsics(
        camera_params_path="/tmp/edex",
        rgb_dir="/tmp/images",
        output_dir="/tmp/output",
        use_marker_chessboard=enabled,
    )

    assert captured["extra_args"].get("use_marker_chessboard", False) is enabled


def test_docker_wrapper_forwards_calibration_setup(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        runner,
        "run_in_container",
        lambda **kwargs: captured.update(kwargs),
    )

    runner.run_calibrate_extrinsics(
        camera_params_path="/tmp/edex",
        rgb_dir="/tmp/images",
        output_dir="/tmp/output",
        calibration_setup="stereo4_6x10_100mm_marker",
    )

    assert (
        captured["extra_args"]["calibration_setup"]
        == "stereo4_6x10_100mm_marker"
    )
    assert captured["gpus"] is False
