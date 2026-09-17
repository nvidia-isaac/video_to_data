import json
import sys
from pathlib import Path

import numpy as np
import pytest


LIB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIB_DIR))

import verify_bbox_manifest as verifier


class _Source:
    n_frames = 3

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def __getitem__(self, index):
        assert index == 1
        return np.zeros((20, 30, 3), dtype=np.uint8)


def _bbox(path: Path, box: dict) -> None:
    path.write_text(json.dumps({"1": [{
        "confidence": 0.9, "label": "blue_trash_can", "box": box,
    }]}))


def test_render_bbox_preview_validates_frame_and_writes_green_label(tmp_path, monkeypatch):
    bbox_dir, rgb_dir, previews = (
        tmp_path / "bboxes", tmp_path / "images", tmp_path / "previews",
    )
    bbox_dir.mkdir()
    rgb_dir.mkdir()
    (rgb_dir / "back_stereo_camera_left.h5").touch()
    _bbox(
        bbox_dir / "back_stereo_camera_left.json",
        {"x0": 2, "y0": 3, "x1": 15, "y1": 17},
    )
    monkeypatch.setattr(verifier.FrameSource, "from_path", lambda _: _Source())
    records = verifier.render_bbox_previews(bbox_dir, rgb_dir, previews)
    assert records[0]["frame_width"] == 30
    assert (previews / "back_stereo_camera_left_bbox.png").is_file()


def test_render_bbox_preview_rejects_stale_coordinates(tmp_path, monkeypatch):
    bbox_dir, rgb_dir = tmp_path / "bboxes", tmp_path / "images"
    bbox_dir.mkdir()
    rgb_dir.mkdir()
    (rgb_dir / "back_stereo_camera_left.h5").touch()
    _bbox(
        bbox_dir / "back_stereo_camera_left.json",
        {"x0": 2, "y0": 3, "x1": 50, "y1": 17},
    )
    monkeypatch.setattr(verifier.FrameSource, "from_path", lambda _: _Source())
    with pytest.raises(ValueError, match="do not match decoded frame"):
        verifier.render_bbox_previews(bbox_dir, rgb_dir, tmp_path / "previews")
