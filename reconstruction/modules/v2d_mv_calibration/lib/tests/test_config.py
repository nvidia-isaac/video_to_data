from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from v2d.mv.calibration.lib import calibrate_extrinsics as calibration


def test_load_config_without_setup_preserves_legacy_defaults():
    cfg = calibration.load_calibration_config()

    assert cfg.calibration_setup is None
    assert cfg.rig_name == "stereo-4"
    assert list(cfg.calibration_order) == [0, 4, 2, 6]
    assert list(cfg.board_size) == [6, 10]
    assert cfg.square_size == 0.1
    assert cfg.use_marker_chessboard is False
    assert dict(cfg.correction_focal) == {}
    assert cfg.input_suffix == ""


def test_marker_setup_overrides_only_integrated_calibration_fields():
    cfg = calibration.load_calibration_config(
        calibration_setup="stereo4_6x10_100mm_marker",
    )

    assert cfg.calibration_setup == "stereo4_6x10_100mm_marker"
    assert cfg.rig_name == "stereo-4"
    assert list(cfg.calibration_order) == [0, 4, 2, 6]
    assert list(cfg.board_size) == [6, 10]
    assert cfg.square_size == 0.1
    assert cfg.use_marker_chessboard is True
    assert dict(cfg.correction_focal) == {}
    assert cfg.input_suffix == ""


@pytest.mark.parametrize(
    "setup_name",
    ["stereo4_6x10_100mm_marker", "stereo4_6x10_22p58mm_marker"],
)
def test_stereo4_setups_leave_focal_correction_to_preprocessing(setup_name):
    cfg = calibration.load_calibration_config(calibration_setup=setup_name)

    assert dict(cfg.correction_focal) == {}


def test_override_selects_setup_and_independently_sets_input_suffix(tmp_path):
    override_path = tmp_path / "calibration_config.yaml"
    override_path.write_text(
        "calibration_setup: stereo4_6x10_100mm_marker\n"
        'input_suffix: ".h5"\n',
        encoding="utf-8",
    )

    cfg = calibration.load_calibration_config(config_path=override_path)

    assert cfg.calibration_setup == "stereo4_6x10_100mm_marker"
    assert cfg.use_marker_chessboard is True
    assert cfg.input_suffix == ".h5"


def test_explicit_setup_takes_precedence_over_override_selector(tmp_path):
    override_path = tmp_path / "calibration_config.yaml"
    override_path.write_text(
        "calibration_setup: missing_setup\n"
        'input_suffix: ".h5"\n',
        encoding="utf-8",
    )

    cfg = calibration.load_calibration_config(
        config_path=override_path,
        calibration_setup="stereo4_6x10_100mm_marker",
    )

    assert cfg.calibration_setup == "stereo4_6x10_100mm_marker"
    assert cfg.use_marker_chessboard is True
    assert cfg.input_suffix == ".h5"


@pytest.mark.parametrize(
    "setup_name",
    ["../outside", "name.yaml", "missing_setup", 123],
)
def test_invalid_or_unknown_setup_is_rejected(setup_name):
    with pytest.raises(ValueError, match="calibration setup"):
        calibration.load_calibration_config(calibration_setup=setup_name)


@pytest.mark.parametrize("configured_value", [None, False, True])
def test_config_forwards_marker_chessboard_flag(monkeypatch, configured_value):
    class FakeRig:
        def __init__(self, *args, **kwargs):
            pass

        def get_all_cameras(self):
            return [SimpleNamespace(image_path="cam0")]

    captured = {}
    monkeypatch.setattr(calibration, "RigConfig", FakeRig)
    monkeypatch.setattr(
        calibration,
        "calibrate_extrinsics",
        lambda **kwargs: captured.update(kwargs),
    )
    config = {
        "rig_name": "test",
        "camera_params_path": "/tmp/edex",
        "rgb_dir": "/tmp/images",
        "output_camera_params_path": "/tmp/output/edex",
        "calibration_order": [0],
    }
    if configured_value is not None:
        config["use_marker_chessboard"] = configured_value

    calibration.calibrate_extrinsics_from_config(OmegaConf.create(config))

    assert captured["use_marker_chessboard"] is bool(configured_value)


def test_config_normalizes_and_forwards_focal_corrections(monkeypatch):
    class FakeRig:
        def __init__(self, *args, **kwargs):
            pass

        def get_all_cameras(self):
            return [SimpleNamespace(image_path="cam0")]

    captured = {}
    monkeypatch.setattr(calibration, "RigConfig", FakeRig)
    monkeypatch.setattr(
        calibration,
        "calibrate_extrinsics",
        lambda **kwargs: captured.update(kwargs),
    )
    cfg = OmegaConf.create({
        "rig_name": "test",
        "camera_params_path": "/tmp/edex",
        "rgb_dir": "/tmp/images",
        "output_camera_params_path": "/tmp/output/edex",
        "calibration_order": [0],
        "correction_focal": {"6": "0.985", 7: 0.99},
    })

    calibration.calibrate_extrinsics_from_config(cfg)

    assert captured["correction_focal"] == {6: 0.985, 7: 0.99}


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_forwards_marker_chessboard_flag(monkeypatch, enabled):
    captured = {}
    monkeypatch.setattr(
        calibration,
        "calibrate_extrinsics_from_config",
        lambda cfg: captured.update(cfg=cfg),
    )
    argv = [
        "--camera_params_path",
        "/tmp/edex",
        "--rgb_dir",
        "/tmp/images",
        "--output_dir",
        "/tmp/output",
    ]
    if enabled:
        argv.append("--use_marker_chessboard")

    calibration._main(argv)

    assert captured["cfg"].use_marker_chessboard is enabled


def test_cli_forwards_calibration_setup(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        calibration,
        "calibrate_extrinsics_from_config",
        lambda cfg: captured.update(cfg=cfg),
    )

    calibration._main([
        "--camera_params_path",
        "/tmp/edex",
        "--rgb_dir",
        "/tmp/images",
        "--output_dir",
        "/tmp/output",
        "--calibration_setup",
        "stereo4_6x10_100mm_marker",
    ])

    assert captured["cfg"].calibration_setup == "stereo4_6x10_100mm_marker"
    assert captured["cfg"].use_marker_chessboard is True


def test_defaults_remain_legacy_while_osmo_selects_marker_setup():
    reconstruction_dir = Path(__file__).resolve().parents[4]
    default_cfg = OmegaConf.load(
        reconstruction_dir
        / "modules/v2d_mv_calibration/lib/calibrate_extrinsics.yaml"
    )
    workflow = (
        reconstruction_dir / "workflows/mv_hoi/osmo/mv_calibration.yaml"
    ).read_text(encoding="utf-8")

    assert default_cfg.calibration_setup is None
    assert default_cfg.use_marker_chessboard is False
    assert 'calibration_setup: "{{calibration_setup}}"' in workflow
    assert "input_suffix: \".h5\"" in workflow
    assert "use_marker_chessboard: true" not in workflow
    assert "calibration_setup: stereo4_6x10_100mm_marker" in workflow
