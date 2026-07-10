import json
import logging
import sys
import types
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

LIB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIB_DIR))


def _install_v2d_stubs():
    class BoundingBox:
        def __init__(self, x0, y0, x1, y1):
            self.x0 = x0
            self.y0 = y0
            self.x1 = x1
            self.y1 = y1

        def to_dict(self):
            return {
                "x0": self.x0,
                "y0": self.y0,
                "x1": self.x1,
                "y1": self.y1,
            }

    class FakeFrameSource:
        def __init__(self, path):
            self.path = Path(path)
            self.image_paths = sorted(self.path.glob("*.png"))
            self.stems = [p.stem for p in self.image_paths]
            self.n_frames = len(self.image_paths)

        @classmethod
        def from_path(cls, path):
            return cls(path)

        def __getitem__(self, idx):
            return np.array(Image.open(self.image_paths[idx]).convert("RGB"))

        def close(self):
            pass

    modules = {
        "v2d": types.ModuleType("v2d"),
        "v2d.common": types.ModuleType("v2d.common"),
        "v2d.common.datatypes": types.ModuleType("v2d.common.datatypes"),
        "v2d.common.video": types.ModuleType("v2d.common.video"),
        "v2d.mv": types.ModuleType("v2d.mv"),
        "v2d.mv.rig": types.ModuleType("v2d.mv.rig"),
        "v2d.mv.preprocess": types.ModuleType("v2d.mv.preprocess"),
        "v2d.mv.preprocess.lib": types.ModuleType("v2d.mv.preprocess.lib"),
        "v2d.mv.preprocess.lib.image_proc": types.ModuleType(
            "v2d.mv.preprocess.lib.image_proc"
        ),
        "v2d.mv.preprocess.lib.preprocess_stereo": types.ModuleType(
            "v2d.mv.preprocess.lib.preprocess_stereo"
        ),
    }
    modules["v2d.common.datatypes"].BoundingBox = BoundingBox
    modules["v2d.common.video"].FrameSource = FakeFrameSource
    modules["v2d.mv.rig"].RigConfig = object
    modules["v2d.mv.preprocess.lib.image_proc"].ImagePipeline = object
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
        self.save_calls = []

    def get_stereo_pairs(self):
        return [self.pair]

    def get_camera(self, cam_id):
        return self.cameras[cam_id]

    def merge_extrinsics(self, path):
        self.merge_calls.append(path)

    def save_camera_params(self, source_path, output_path):
        self.save_calls.append((source_path, output_path))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("saved")


def _write_frames(frame_dir: Path, n_frames: int = 3) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(n_frames):
        img = np.zeros((80, 80, 3), dtype=np.uint8)
        Image.fromarray(img, mode="RGB").save(frame_dir / f"{idx:06d}.png")


def _write_metadata(path: Path, object_meta: dict) -> None:
    path.write_text(yaml.dump({"object": object_meta}, sort_keys=False))


def _run_remap(tmp_path: Path, object_meta: dict) -> tuple[Path, Path, Path]:
    metadata_path = tmp_path / "hoi_metadata.yaml"
    frames_dir = tmp_path / "frames" / "front_stereo_camera_left"
    bbox_path = tmp_path / "labeled_bboxes" / "front_stereo_camera_left.json"
    output_metadata_path = tmp_path / "out" / "hoi_metadata.yaml"
    prompt_path = tmp_path / "out" / "prompt.txt"

    _write_metadata(metadata_path, object_meta)
    _write_frames(frames_dir)

    mp.remap_hoi_bboxes(
        hoi_metadata_path=metadata_path,
        pipelines={"front_stereo_camera_left": IdentityPipeline()},
        output_image_dirs={"front_stereo_camera_left": frames_dir},
        labeled_bbox_paths={"front_stereo_camera_left": bbox_path},
        output_hoi_metadata_path=output_metadata_path,
        output_prompt_path=prompt_path,
    )
    return bbox_path, output_metadata_path, prompt_path


def test_remap_with_legacy_bbox_writes_first_frame_label_and_prompt(tmp_path):
    bbox_path, output_metadata_path, prompt_path = _run_remap(
        tmp_path,
        {
            "id": "blue_trash_can",
            "prompt": "a tall blue rectangular trash can.",
            "bbox": {"front_stereo_camera_left": [10, 20, 30, 40]},
        },
    )

    bbox_data = json.loads(bbox_path.read_text())
    assert list(bbox_data.keys()) == ["000000"]
    assert bbox_data["000000"][0]["confidence"] == 1.0
    assert bbox_data["000000"][0]["box"] == {
        "x0": 10.0,
        "y0": 20.0,
        "x1": 40.0,
        "y1": 60.0,
    }
    assert prompt_path.read_text() == "a tall blue rectangular trash can."
    assert not (output_metadata_path.parent / "object_bbox_source.txt").exists()

    forwarded = yaml.safe_load(output_metadata_path.read_text())
    assert "bbox" not in forwarded["object"]


def test_remap_without_object_bbox_skips_bbox_conversion_and_marker(tmp_path):
    bbox_path, output_metadata_path, prompt_path = _run_remap(
        tmp_path,
        {
            "id": "blue_trash_can",
        },
    )

    assert not bbox_path.exists()
    assert not prompt_path.exists()
    assert not (output_metadata_path.parent / "object_bbox_source.txt").exists()
    forwarded = yaml.safe_load(output_metadata_path.read_text())
    assert forwarded["object"] == {"id": "blue_trash_can"}


def test_remap_with_no_prompt_does_not_write_empty_prompt_file(tmp_path):
    _bbox_path, _output_metadata_path, prompt_path = _run_remap(
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
    assert "No extrinsics_camera_params_path provided" in caplog.text
    assert "No mesh_path provided" in caplog.text
