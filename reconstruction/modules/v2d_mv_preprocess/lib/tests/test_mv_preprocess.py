import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest
import yaml

LIB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIB_DIR))
_CONFIG_SPEC = importlib.util.spec_from_file_location(
    "mv_preprocess_config_under_test",
    LIB_DIR / "config.py",
)
assert _CONFIG_SPEC is not None and _CONFIG_SPEC.loader is not None
preprocess_config = importlib.util.module_from_spec(_CONFIG_SPEC)
_CONFIG_SPEC.loader.exec_module(preprocess_config)


def _install_v2d_stubs():
    modules = {
        "v2d": types.ModuleType("v2d"),
        "v2d.mv": types.ModuleType("v2d.mv"),
        "v2d.mv.rig": types.ModuleType("v2d.mv.rig"),
        "v2d.mv.preprocess": types.ModuleType("v2d.mv.preprocess"),
        "v2d.mv.preprocess.lib": types.ModuleType("v2d.mv.preprocess.lib"),
        "v2d.mv.preprocess.lib.config": types.ModuleType(
            "v2d.mv.preprocess.lib.config"
        ),
        "v2d.mv.preprocess.lib.image_proc": types.ModuleType(
            "v2d.mv.preprocess.lib.image_proc"
        ),
        "v2d.mv.preprocess.lib.preprocess_stereo": types.ModuleType(
            "v2d.mv.preprocess.lib.preprocess_stereo"
        ),
    }
    modules["v2d.mv.rig"].RigConfig = object
    modules["v2d.mv.preprocess.lib.image_proc"].ImagePipeline = object
    modules["v2d.mv.preprocess.lib.config"].resolve_calibration_camera_params_path = (
        preprocess_config.resolve_calibration_camera_params_path
    )
    modules["v2d.mv.preprocess.lib.preprocess_stereo"].preprocess_stereo = None
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    return previous


def _restore_modules(previous):
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


_PREVIOUS_MODULES = _install_v2d_stubs()
import mv_preprocess as mp
_restore_modules(_PREVIOUS_MODULES)


class IdentityPipeline:
    def map_points(self, pts):
        return pts


class _FakeCamera:
    def __init__(self, cam_id: int, name: str):
        self.cam_id = cam_id
        self.name = name
        self.param = object()


class _FakePair:
    def __init__(self):
        self.name = "front_stereo_camera"
        self.left = _FakeCamera(0, "front_stereo_camera_left")
        self.right = _FakeCamera(1, "front_stereo_camera_right")


class _FakeRig:
    def __init__(self):
        self.pair = _FakePair()
        self.cameras = {
            self.pair.left.cam_id: self.pair.left,
            self.pair.right.cam_id: self.pair.right,
        }
        self.merge_calls = []
        self.events = []
        self.save_calls = []
        self.saved_params = None

    def get_stereo_pairs(self):
        return [self.pair]

    def get_camera(self, cam_id):
        return self.cameras[cam_id]

    def merge_intrinsics(self, path):
        self.merge_calls.append(("intrinsics", path))
        self.events.append("merge_intrinsics")

    def merge_extrinsics(self, path):
        self.merge_calls.append(("extrinsics", path))
        self.events.append("merge_extrinsics")

    def save_camera_params(self, source_path, output_path):
        self.events.append("save")
        self.save_calls.append((source_path, output_path))
        self.saved_params = {
            cam_id: camera.param for cam_id, camera in self.cameras.items()
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("saved")


def _write_metadata(path: Path, object_meta: dict) -> None:
    path.write_text(yaml.dump({"object": object_meta}, sort_keys=False))


def _forward_metadata(tmp_path: Path, object_meta: dict) -> tuple[Path, Path]:
    metadata_path = tmp_path / "hoi_metadata.yaml"
    output_metadata_path = tmp_path / "out" / "hoi_metadata.yaml"
    prompt_path = tmp_path / "out" / "prompt.txt"

    _write_metadata(metadata_path, object_meta)
    mp.forward_hoi_metadata(
        hoi_metadata_path=metadata_path,
        output_hoi_metadata_path=output_metadata_path,
        output_prompt_path=prompt_path,
    )
    return output_metadata_path, prompt_path


def test_forward_metadata_ignores_legacy_bbox_and_never_creates_bbox_directory(
    tmp_path,
):
    output_metadata_path, prompt_path = _forward_metadata(
        tmp_path,
        {
            "id": "blue_trash_can",
            "prompt": "a tall blue rectangular trash can.",
            "bbox": {"front_stereo_camera_left": [10, 20, 30, 40]},
        },
    )

    assert not (tmp_path / "out" / "labeled_bboxes").exists()
    assert prompt_path.read_text() == "a tall blue rectangular trash can."
    assert not (output_metadata_path.parent / "object_bbox_source.txt").exists()

    forwarded = yaml.safe_load(output_metadata_path.read_text())
    assert "bbox" not in forwarded["object"]


def test_forward_metadata_does_not_read_or_modify_existing_labeled_bboxes(
    tmp_path,
):
    bbox_path = tmp_path / "out" / "labeled_bboxes" / "front_stereo_camera_left.json"
    bbox_path.parent.mkdir(parents=True, exist_ok=True)
    original = b'{"manual":"bbox"}\n'
    bbox_path.write_bytes(original)

    output_metadata_path, _prompt_path = _forward_metadata(
        tmp_path,
        {
            "id": "blue_trash_can",
            "prompt": "a tall blue rectangular trash can.",
            "bbox": {"front_stereo_camera_left": [10, 20, 30, 40]},
        },
    )

    assert bbox_path.read_bytes() == original
    assert list(bbox_path.parent.iterdir()) == [bbox_path]
    forwarded = yaml.safe_load(output_metadata_path.read_text())
    assert "bbox" not in forwarded["object"]


def test_forward_metadata_without_object_bbox_or_prompt(tmp_path):
    output_metadata_path, prompt_path = _forward_metadata(
        tmp_path,
        {
            "id": "blue_trash_can",
        },
    )

    assert not (tmp_path / "out" / "labeled_bboxes").exists()
    assert not prompt_path.exists()
    assert not (output_metadata_path.parent / "object_bbox_source.txt").exists()
    forwarded = yaml.safe_load(output_metadata_path.read_text())
    assert forwarded["object"] == {"id": "blue_trash_can"}


def test_forward_metadata_with_no_prompt_does_not_write_empty_prompt_file(tmp_path):
    _output_metadata_path, prompt_path = _forward_metadata(
        tmp_path,
        {
            "id": "blue_trash_can",
            "bbox": {"front_stereo_camera_left": [10, 20, 30, 40]},
        },
    )

    assert not prompt_path.exists()


def test_mv_preprocess_missing_optional_inputs_skips_metadata_extrinsics_and_mesh(
    monkeypatch, caplog, tmp_path,
):
    rig = _FakeRig()

    def fake_preprocess_stereo(**_kwargs):
        return (IdentityPipeline(), IdentityPipeline()), ("left_param", "right_param")

    monkeypatch.setattr(mp, "preprocess_stereo", fake_preprocess_stereo)
    caplog.set_level(logging.INFO, logger=mp.logger.name)

    mp.mv_preprocess(
        rig=rig,
        rgb_paths={
            0: tmp_path / "front_left.h5",
            1: tmp_path / "front_right.h5",
        },
        output_image_dirs={
            0: tmp_path / "images" / "front_stereo_camera_left",
            1: tmp_path / "images" / "front_stereo_camera_right",
        },
        camera_params_path=tmp_path / "raw_edex",
        output_camera_params_path=tmp_path / "out" / "edex",
    )

    assert rig.merge_calls == []
    assert rig.save_calls == [(tmp_path / "raw_edex", tmp_path / "out" / "edex")]
    assert not (tmp_path / "out" / "hoi_metadata.yaml").exists()
    assert not (tmp_path / "out" / "object_mesh").exists()
    assert "No hoi_metadata_path provided" in caplog.text
    assert "No calibration_camera_params_path provided" in caplog.text
    assert "No mesh_path provided" in caplog.text


def test_mv_preprocess_applies_legacy_focal_once_and_persists_processed_params(
    monkeypatch, tmp_path,
):
    rig = _FakeRig()
    calibration_path = tmp_path / "calibration" / "edex"
    correction_focal = {0: 0.99}
    correction_calls = []

    def fake_preprocess_stereo(**kwargs):
        rig.events.append("preprocess_stereo")
        correction_calls.append(kwargs["correction_focal"])
        return (
            (IdentityPipeline(), IdentityPipeline()),
            ("left_param_corrected", "right_param_uncorrected"),
        )

    monkeypatch.setattr(mp, "preprocess_stereo", fake_preprocess_stereo)

    mp.mv_preprocess(
        rig=rig,
        rgb_paths={0: tmp_path / "left.h5", 1: tmp_path / "right.h5"},
        output_image_dirs={
            0: tmp_path / "images" / "left",
            1: tmp_path / "images" / "right",
        },
        camera_params_path=tmp_path / "raw" / "edex",
        output_camera_params_path=tmp_path / "out" / "edex",
        calibration_camera_params_path=calibration_path,
        correction_focal=correction_focal,
    )

    assert rig.merge_calls == [
        ("intrinsics", calibration_path),
        ("extrinsics", calibration_path),
    ]
    assert rig.events == [
        "merge_intrinsics",
        "preprocess_stereo",
        "merge_extrinsics",
        "save",
    ]
    assert correction_calls == [correction_focal]
    assert rig.saved_params == {
        0: "left_param_corrected",
        1: "right_param_uncorrected",
    }


def test_resolve_calibration_path_accepts_canonical_and_deprecated_alias(tmp_path):
    path = tmp_path / "calibration" / "edex"

    assert preprocess_config.resolve_calibration_camera_params_path(path, None) == path
    with pytest.warns(FutureWarning, match="deprecated"):
        assert (
            preprocess_config.resolve_calibration_camera_params_path(None, path)
            == path
        )
    with pytest.warns(FutureWarning, match="deprecated"):
        assert (
            preprocess_config.resolve_calibration_camera_params_path(path, str(path))
            == path
        )


def test_resolve_calibration_path_rejects_conflicting_names(tmp_path):
    with pytest.warns(FutureWarning, match="deprecated"):
        with pytest.raises(ValueError, match="Conflicting calibration camera params"):
            preprocess_config.resolve_calibration_camera_params_path(
                tmp_path / "new" / "edex",
                tmp_path / "old" / "edex",
            )


def test_default_config_uses_canonical_path_and_legacy_focal_correction():
    cfg = yaml.safe_load((LIB_DIR / "mv_preprocess.yaml").read_text())

    assert cfg["calibration_camera_params_path"] is None
    assert cfg["extrinsics_camera_params_path"] is None
    assert cfg["correction_focal"] == {6: 0.985, 7: 0.985}
    assert "labeled_bbox_path_template" not in cfg
